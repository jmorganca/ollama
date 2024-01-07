#!/bin/bash
# This script is intended to run inside the go generate
# working directory must be llm/generate/

# First we build our default built-in library which will be linked into the CGO
# binary as a normal dependency. This default build is CPU based.
#
# Then we build a CUDA dynamic library (although statically linked with the CUDA
# library dependencies for maximum portability)
#
# Then if we detect ROCm, we build a dynamically loaded ROCm lib.  ROCm is particularly
# important to be a dynamic lib even if it's the only GPU library detected because
# we can't redistribute the objectfiles but must rely on dynamic libraries at
# runtime, which could lead the server not to start if not present.

set -ex
set -o pipefail

# See https://llvm.org/docs/AMDGPUUsage.html#processors for reference
amdGPUs() {
    GPU_LIST=(
        "gfx803"
        "gfx900"
        "gfx906:xnack-"
        "gfx908:xnack-"
        "gfx90a:xnack+"
        "gfx90a:xnack-"
        "gfx1010"
        "gfx1012"
        "gfx1030"
        "gfx1100"
        "gfx1101"
        "gfx1102"
    )
    (
        IFS=$';'
        echo "'${GPU_LIST[*]}'"
    )
}

echo "Starting linux generate script"
if [ -z "${CUDACXX}" -a -x /usr/local/cuda/bin/nvcc ]; then
    export CUDACXX=/usr/local/cuda/bin/nvcc
fi
COMMON_CMAKE_DEFS="-DCMAKE_POSITION_INDEPENDENT_CODE=on -DLLAMA_NATIVE=off -DLLAMA_AVX=on -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off"
source $(dirname $0)/gen_common.sh
init_vars
git_module_setup
apply_patches

if [ -z "${OLLAMA_SKIP_CPU_GENERATE}" ]; then
    # -DLLAMA_AVX -- 2011 Intel Sandy Bridge & AMD Bulldozer
    # -DLLAMA_AVX2 -- 2013 Intel Haswell & 2015 AMD Excavator / 2017 AMD Zen
    # -DLLAMA_AVX512 -- 2017 Intel Skylake and High End DeskTop (HEDT)
    # -DLLAMA_AVX512_VBMI -- 2018 Intel Cannon Lake
    # -DLLAMA_AVX512_VNNI -- 2021 Intel Alder Lake
    # -DLLAMA_FMA (FMA3) -- 2013 Intel Haswell & 2012 AMD Piledriver
    # -DLLAMA_F16C -- 2012 Intel Ivy Bridge & AMD 2011 Bulldozer
    # Rosetta does NOT support AVX, AVX2, AVX512

    COMMON_CPU_DEFS="-DCMAKE_POSITION_INDEPENDENT_CODE=on -DLLAMA_NATIVE=off"
    #
    # CPU first for the default library, set up as lowest common denominator for maximum compatibility
    #
    CMAKE_DEFS="${COMMON_CPU_DEFS} -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off ${CMAKE_DEFS}"
    BUILD_DIR="${LLAMACPP_DIR}/build/linux/cpu"
    echo "Building LCD CPU"
    build
    install

    #
    # ~2011 CPU Dynamic library with more capabilities turned on to optimize performance
    #
    init_vars
    CMAKE_DEFS="${COMMON_CPU_DEFS} -DBUILD_SHARED_LIBS=on -DLLAMA_AVX=on -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off ${CMAKE_DEFS}"
    BUILD_DIR="${LLAMACPP_DIR}/build/linux/cpu_avx"
    CMAKE_TARGETS="${CMAKE_TARGETS} --target ext_server_shared"
    echo "Building AVX CPU"
    build
    install_shared
    #
    # ~2013 CPU Dynamic library
    #
    init_vars
    CMAKE_DEFS="${COMMON_CPU_DEFS} -DBUILD_SHARED_LIBS=on -DLLAMA_AVX=on -DLLAMA_AVX2=on -DLLAMA_AVX512=off -DLLAMA_FMA=on -DLLAMA_F16C=on ${CMAKE_DEFS}"
    BUILD_DIR="${LLAMACPP_DIR}/build/linux/cpu_avx2"
    CMAKE_TARGETS="${CMAKE_TARGETS} --target ext_server_shared"
    echo "Building AVX2 CPU"
    build
    install_shared

    #
    # ~2017 CPU Dynamic library (best perf)
    #
    init_vars
    CMAKE_DEFS="${COMMON_CPU_DEFS} -DBUILD_SHARED_LIBS=on -DLLAMA_AVX=on -DLLAMA_AVX2=on -DLLAMA_AVX512=on -DLLAMA_FMA=on -DLLAMA_F16C=on ${CMAKE_DEFS}"
    BUILD_DIR="${LLAMACPP_DIR}/build/linux/cpu_avx512"
    CMAKE_TARGETS="${CMAKE_TARGETS} --target ext_server_shared"
    echo "Building AVX512 CPU"
    build
    install_shared
else
    echo "Skipping CPU generation step as requested"
fi

if [ -d /usr/local/cuda/lib64/ ]; then
    echo "CUDA libraries detected - building dynamic CUDA library"
    init_vars
    CUDA_MAJOR=$(ls /usr/local/cuda/lib64/libcudart.so.* | head -1 | cut -f3 -d. || true)
    if [ -n "${CUDA_MAJOR}" ]; then
        CUDA_VARIANT=_v${CUDA_MAJOR}
    fi
    CMAKE_DEFS="-DLLAMA_CUBLAS=on ${COMMON_CMAKE_DEFS} ${CMAKE_DEFS}"
    BUILD_DIR="${LLAMACPP_DIR}/build/linux/cuda${CUDA_VARIANT}"
    CUDA_LIB_DIR=/usr/local/cuda/lib64
    build
    install
    gcc -fPIC -g -shared -o ${BUILD_DIR}/lib/libext_server.so \
        -Wl,--whole-archive \
        ${BUILD_DIR}/lib/libext_server.a \
        ${BUILD_DIR}/lib/libcommon.a \
        ${BUILD_DIR}/lib/libllama.a \
        -Wl,--no-whole-archive \
        ${CUDA_LIB_DIR}/libcudart_static.a \
        ${CUDA_LIB_DIR}/libcublas_static.a \
        ${CUDA_LIB_DIR}/libcublasLt_static.a \
        ${CUDA_LIB_DIR}/libcudadevrt.a \
        ${CUDA_LIB_DIR}/libculibos.a \
        -lrt -lpthread -ldl -lstdc++ -lm
fi

if [ -z "${ROCM_PATH}" ]; then
    # Try the default location in case it exists
    ROCM_PATH=/opt/rocm
fi

if [ -z "${CLBlast_DIR}" ]; then
    # Try the default location in case it exists
    if [ -d /usr/lib/cmake/CLBlast ]; then
        export CLBlast_DIR=/usr/lib/cmake/CLBlast
    fi
fi

if [ -d "${ROCM_PATH}" ]; then
    echo "ROCm libraries detected - building dynamic ROCm library"
    if [ -f ${ROCM_PATH}/lib/librocm_smi64.so.? ]; then
        ROCM_VARIANT=_v$(ls ${ROCM_PATH}/lib/librocm_smi64.so.? | cut -f3 -d. || true)
    fi
    init_vars
    CMAKE_DEFS="${COMMON_CMAKE_DEFS} ${CMAKE_DEFS} -DLLAMA_HIPBLAS=on -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ -DAMDGPU_TARGETS=$(amdGPUs) -DGPU_TARGETS=$(amdGPUs)"
    BUILD_DIR="${LLAMACPP_DIR}/build/linux/rocm${ROCM_VARIANT}"
    build
    install
    gcc -fPIC -g -shared -o ${BUILD_DIR}/lib/libext_server.so \
        -Wl,--whole-archive \
        ${BUILD_DIR}/lib/libext_server.a \
        ${BUILD_DIR}/lib/libcommon.a \
        ${BUILD_DIR}/lib/libllama.a \
        -Wl,--no-whole-archive \
        -lrt -lpthread -ldl -lstdc++ -lm \
        -L/opt/rocm/lib -L/opt/amdgpu/lib/x86_64-linux-gnu/ \
        -Wl,-rpath,/opt/rocm/lib,-rpath,/opt/amdgpu/lib/x86_64-linux-gnu/ \
        -lhipblas -lrocblas -lamdhip64 -lrocsolver -lamd_comgr -lhsa-runtime64 -lrocsparse -ldrm -ldrm_amdgpu
fi

cleanup
