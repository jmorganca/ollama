# common logic accross linux and darwin

init_vars() {
    PATCHES="0001-Expose-callable-API-for-server.patch"
    CMAKE_DEFS="-DLLAMA_ACCELERATE=on"
    # TODO - LLAMA_K_QUANTS is stale and needs to be mapped to newer cmake settings
    CMAKE_TARGETS="--target ggml --target ggml_static --target llama --target build_info --target common --target ext_server"
    if echo "${CGO_CFLAGS}" | grep -- '-g' > /dev/null ; then
        CMAKE_DEFS="-DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_VERBOSE_MAKEFILE=on -DLLAMA_GPROF=on ${CMAKE_DEFS}"
    else
        # TODO - add additional optimization flags...
        CMAKE_DEFS="-DCMAKE_BUILD_TYPE=Release ${CMAKE_DEFS}"
    fi
}

git_module_setup() {
    # TODO add flags to skip the init/patch logic to make it easier to mod llama.cpp code in-repo
    git submodule init
    git submodule update --force gguf

}

apply_patches() {
    # Workaround git apply not handling creation well for iteration
    rm -f gguf/examples/server/server.h
    for patch in ${PATCHES} ; do
        git -C gguf apply ../patches/${patch}
    done
}

build() {
    cmake -S gguf -B ${BUILD_DIR} ${CMAKE_DEFS}
    cmake --build ${BUILD_DIR} ${CMAKE_TARGETS} -j8 
}