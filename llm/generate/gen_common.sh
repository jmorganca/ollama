# common logic accross linux and darwin

init_vars() {
    LLAMACPP_DIR=../llama.cpp
    CMAKE_DEFS=""
    CMAKE_TARGETS="--target ggml --target ggml_static --target llama --target build_info --target common --target ext_server --target llava_static"
    if echo "${CGO_CFLAGS}" | grep -- '-g' >/dev/null; then
        CMAKE_DEFS="-DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_VERBOSE_MAKEFILE=on -DLLAMA_GPROF=on -DLLAMA_SERVER_VERBOSE=on"
    else
        # TODO - add additional optimization flags...
        CMAKE_DEFS="-DCMAKE_BUILD_TYPE=Release -DLLAMA_SERVER_VERBOSE=off"
    fi
}

git_module_setup() {
    if [ -n "${OLLAMA_SKIP_PATCHING}" ]; then
        echo "Skipping submodule initialization"
        return
    fi
    # Make sure the tree is clean after the directory moves
    if [ -d "${LLAMACPP_DIR}/gguf" ]; then
        echo "Cleaning up old submodule"
        rm -rf ${LLAMACPP_DIR}
    fi
    git submodule init
    git submodule update --force ${LLAMACPP_DIR}

}

apply_patches() {
    # Wire up our CMakefile
    if ! grep ollama ${LLAMACPP_DIR}/examples/server/CMakeLists.txt; then
        echo 'include (../../../ext_server/CMakeLists.txt) # ollama' >>${LLAMACPP_DIR}/examples/server/CMakeLists.txt
    fi
    # Avoid duplicate main symbols when we link into the cgo binary
    sed -e 's/int main(/int __main(/g' <${LLAMACPP_DIR}/examples/server/server.cpp >${LLAMACPP_DIR}/examples/server/server.cpp.tmp &&
        mv ${LLAMACPP_DIR}/examples/server/server.cpp.tmp ${LLAMACPP_DIR}/examples/server/server.cpp
}

build() {
    cmake -S ${LLAMACPP_DIR} -B ${BUILD_DIR} ${CMAKE_DEFS}
    cmake --build ${BUILD_DIR} ${CMAKE_TARGETS} -j8
}

install() {
    rm -rf ${BUILD_DIR}/lib
    mkdir -p ${BUILD_DIR}/lib
    cp ${BUILD_DIR}/examples/server/libext_server.a ${BUILD_DIR}/lib
    cp ${BUILD_DIR}/common/libcommon.a ${BUILD_DIR}/lib
    cp ${BUILD_DIR}/libllama.a ${BUILD_DIR}/lib
    cp ${BUILD_DIR}/libggml_static.a ${BUILD_DIR}/lib
}

install_shared() {
    rm -rf ${BUILD_DIR}/lib
    mkdir -p ${BUILD_DIR}/lib
    cp ${BUILD_DIR}/examples/server/libext_server_shared.so ${BUILD_DIR}/lib
    cp ${BUILD_DIR}/common/libcommon.a ${BUILD_DIR}/lib
}

# Keep the local tree clean after we're done with the build
cleanup() {
    (cd ${LLAMACPP_DIR}/examples/server/ && git checkout CMakeLists.txt server.cpp)
}
