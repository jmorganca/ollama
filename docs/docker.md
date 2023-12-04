# Docker

It's possible to run Ollama with Docker or Docker Compose.

The official [Ollama Docker image](https://hub.docker.com/r/ollama/ollama)
`ollama/ollama` is available on Docker Hub.

## Docker Hub

To interact with the official Ollama images:

```bash
# CPU only
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
# With GPU
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

## Docker Compose

Or run using Docker Compose and the starter `docker-compose.yaml` in this folder:

```bash
git clone git@github.com:jmorganca/ollama.git
cd ollama/
docker compose up --wait --detach
```

Here's how to execute `ollama run llama2` commands within the container:

```bash
container_id=$(docker ps | grep ollama | awk '{print $1}')
docker exec -it $container_id ollama run llama2
```

Or here is how to open a bash shell in the container:

```bash
docker exec -it $container_id bash
```
