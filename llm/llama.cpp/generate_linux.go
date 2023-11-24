package llm

//go:generate git submodule init

//go:generate git submodule update --force gguf
//go:generate git -C gguf apply ../patches/0001-copy-cuda-runtime-libraries.patch
//go:generate git -C gguf apply ../patches/0001-update-default-log-target.patch
//go:generate cmake -S gguf -B gguf/build/cpu -DLLAMA_K_QUANTS=on -DLLAMA_NATIVE=off -DLLAMA_AVX=on -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off
//go:generate cmake --build gguf/build/cpu --target server --config Release
//go:generate mv gguf/build/cpu/bin/server gguf/build/cpu/bin/ollama-runner

//go:generate cmake -S gguf -B gguf/build/cuda -DLLAMA_CUBLAS=on -DLLAMA_ACCELERATE=on -DLLAMA_K_QUANTS=on -DLLAMA_NATIVE=off -DLLAMA_AVX=on -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off -DLLAMA_CUDA_PEER_MAX_BATCH_SIZE=0
//go:generate cmake --build gguf/build/cuda --target server --config Release
//go:generate mv gguf/build/cuda/bin/server gguf/build/cuda/bin/ollama-runner
