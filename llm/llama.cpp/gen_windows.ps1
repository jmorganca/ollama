#!powershell

$ErrorActionPreference = "Stop"

function init_vars {
    $script:buildDir="gguf/build/wincuda"
    $script:installDir="gguf/build/wincuda/dist"
    $script:patches = @("0001-Expose-callable-API-for-server.patch")
    $script:cmakeDefs = @("-DLLAMA_NATIVE=off", "-DLLAMA_F16C=off", "-DLLAMA_FMA=off", "-DLLAMA_AVX512=off", "-DLLAMA_AVX2=off", "-DLLAMA_AVX=on", "-DLLAMA_K_QUANTS=on", "-DLLAMA_ACCELERATE=on", "-DLLAMA_CUBLAS=ON","-DCMAKE_VERBOSE_MAKEFILE=ON","-DBUILD_SHARED_LIBS=on","-A","x64")

    if ($env:CGO_CFLAGS -contains "-g") {
        $script:cmakeDefs += @("-DCMAKE_VERBOSE_MAKEFILE=on")
        $script:config += "RelWithDebInfo"
    } else {
        $script:config += "Release"
    }
}

function git_module_setup {
    # TODO add flags to skip the init/patch logic to make it easier to mod llama.cpp code in-repo
    & git submodule init
    & git submodule update --force gguf
}

function apply_patches {
    rm -erroraction ignore -path "gguf/examples/server/server.h"
    foreach ($patch in $patches) {
        write-host "Applying patch $patch"
        & git -C gguf apply ../patches/$patch
    }
}

function build {
    write-host "generating config with: cmake -S gguf -B $buildDir $cmakeDefs"
    & cmake --version
    & cmake -S gguf -B $buildDir $cmakeDefs
    write-host "building with: cmake --build $buildDir --config $config"
    & cmake --build $buildDir --config $config
}

function install {
    rm -erroraction ignore -recurse -force -path $installDir
    & cmake --install $buildDir --prefix $installDir --config $config

}

init_vars
git_module_setup
apply_patches
build
install