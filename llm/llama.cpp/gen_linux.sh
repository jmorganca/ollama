#!/bin/sh
# This script is intended to run inside the go generate
# working directory must be ../llm/llama.cpp

set -ex
set -o pipefail

# TODO - stopped here - map the variables from above over and refine the case statement below

echo "Starting linux generate script"
source $(dirname $0)/gen_common.sh
init_vars
CMAKE_DEFS="-DLLAMA_CUBLAS=on -DLLAMA_NATIVE=off -DLLAMA_AVX=on -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off ${CMAKE_DEFS}"
BUILD_DIR="gguf/build/cuda"
git_module_setup
apply_patches
build
