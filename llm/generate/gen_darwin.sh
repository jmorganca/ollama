#!/bin/bash
# This script is intended to run inside the go generate
# working directory must be ./llm/generate/

# TODO - add hardening to detect missing tools (cmake, etc.)

set -ex
set -o pipefail
echo "Starting darwin generate script"
source $(dirname $0)/gen_common.sh
init_vars
CMAKE_DEFS="-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 -DLLAMA_METAL=on -DLLAMA_ACCELERATE=on ${CMAKE_DEFS}"
BUILD_DIR="${LLAMACPP_DIR}/build/darwin/metal"
case "${GOARCH}" in
"amd64")
    CMAKE_DEFS="-DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_PROCESSOR=x86_64 -DCMAKE_OSX_ARCHITECTURES=x86_64 -DLLAMA_NATIVE=off -DLLAMA_AVX=on -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off ${CMAKE_DEFS}"
    ;;
"arm64")
    CMAKE_DEFS="-DCMAKE_SYSTEM_PROCESSOR=arm64 -DCMAKE_OSX_ARCHITECTURES=arm64 ${CMAKE_DEFS}"
    ;;
*)
    echo "GOARCH must be set"
    echo "this script is meant to be run from within go generate"
    exit 1
    ;;
esac

git_module_setup
apply_patches
build
install
cleanup
