# Ollama Quickstart with Docker

1. Run the Ollama Docker container:

```shell
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

For more detailed information, refer to the [Ollama Docker](https://hub.docker.com/r/ollama/ollama). Please note we are using CPU only, the AI will response slow, if you have GPU, you can follow the instruction to run the docker and using your GPU to improve performance.

2. Open bash in `ollama` container:

```shell
docker exec -it ollama bash
```

3. Chat with Llama 2 inside `ollama` container:

```
root@e4d14c182583:/# ollama run llama2
```
