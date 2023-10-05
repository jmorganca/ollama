package llm

//go:generate git submodule init

//go:generate git submodule update --force ggml
//go:generate git -C ggml apply ../patches/0001-add-detokenize-endpoint.patch
//go:generate git -C ggml apply ../patches/0002-34B-model-support.patch
//go:generate git -C ggml apply ../patches/0005-ggml-support-CUDA-s-half-type-for-aarch64-1455-2670.patch
//go:generate git -C ggml apply ../patches/0001-copy-cuda-runtime-libraries.patch
//go:generate cmake -S ggml -B ggml/build/cuda -DLLAMA_CUBLAS=on -DLLAMA_ACCELERATE=on -DLLAMA_K_QUANTS=on
//go:generate cmake --build ggml/build/cuda --target server --config Release

//go:generate git submodule update --force gguf
//go:generate git -C gguf apply ../patches/0001-copy-cuda-runtime-libraries.patch
//go:generate git -C gguf apply ../patches/0001-remove-warm-up-logging.patch
//go:generate cmake -S gguf -B gguf/build/cuda -DLLAMA_CUBLAS=on -DLLAMA_ACCELERATE=on -DLLAMA_K_QUANTS=on
//go:generate cmake --build gguf/build/cuda --target server --config Release
