package llm

//go:generate git submodule init

//go:generate git submodule update --force gguf
//go:generate git -C gguf apply ../patches/0001-update-default-log-target.patch
//go:generate cmake -S gguf -B gguf/build/cpu -DLLAMA_METAL=off -DLLAMA_ACCELERATE=on -DLLAMA_K_QUANTS=on -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_PROCESSOR=x86_64 -DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 -DLLAMA_NATIVE=off -DLLAMA_AVX=on -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=on
//go:generate cmake --build gguf/build/cpu --target server --config Release
//go:generate mv gguf/build/cpu/bin/server gguf/build/cpu/bin/ollama-runner
