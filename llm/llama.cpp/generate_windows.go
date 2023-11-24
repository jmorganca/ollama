package llm

//go:generate git submodule init

//go:generate git submodule update --force gguf
//go:generate git -C gguf apply ../patches/0001-update-default-log-target.patch
//go:generate cmake -S gguf -B gguf/build/cpu -DLLAMA_K_QUANTS=on -DLLAMA_NATIVE=off -DLLAMA_AVX=on -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off
//go:generate cmake --build gguf/build/cpu --target server --config Release
//go:generate cmd /c move gguf\build\cpu\bin\Release\server.exe gguf\build\cpu\bin\Release\ollama-runner.exe

//go:generate cmake -S ggml -B ggml/build/cuda -DLLAMA_CUBLAS=on -DLLAMA_ACCELERATE=on -DLLAMA_K_QUANTS=on
//go:generate cmake --build ggml/build/cuda --target server --config Release
//go:generate cmd /c move ggml\build\cuda\bin\Release\server.exe ggml\build\cuda\bin\Release\ollama-runner.exe

//go:generate cmake -S gguf -B gguf/build/cuda -DLLAMA_CUBLAS=on -DLLAMA_ACCELERATE=on -DLLAMA_K_QUANTS=on -DLLAMA_NATIVE=off -DLLAMA_AVX=on -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off
//go:generate cmake --build gguf/build/cuda --target server --config Release
//go:generate cmd /c move gguf\build\cuda\bin\Release\server.exe gguf\build\cuda\bin\Release\ollama-runner.exe
