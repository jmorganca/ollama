#!/bin/sh
# This script is intended to run inside the go generate
# working directory must be ../llm/llama.cpp

# TODO - add hardening to detect missing tools (cmake, etc.)

set -ex
set -o pipefail
echo "Starting darwin generate script"
source $(dirname $0)/gen_common.sh
init_vars
CMAKE_DEFS="-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 ${CMAKE_DEFS}"
case "${GOARCH}" in
    "amd64")
        CMAKE_DEFS="-DLLAMA_METAL=off -DCMAKE_SYSTEM_PROCESSOR=x86_64 -DCMAKE_OSX_ARCHITECTURES=x86_64 ${CMAKE_DEFS}"
        BUILD_DIR="gguf/build/cpu"
        ;;
     "arm64")
        CMAKE_DEFS="-DLLAMA_METAL=on -DCMAKE_SYSTEM_PROCESSOR=arm64 -DCMAKE_OSX_ARCHITECTURES=arm64 ${CMAKE_DEFS}"
        BUILD_DIR="gguf/build/metal"
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

# Enable local debug/run usecase
if [ -e "gguf/ggml-metal.metal" ]; then
    cp gguf/ggml-metal.metal ../../
fi
