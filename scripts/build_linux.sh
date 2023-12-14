#!/bin/sh

set -eu

export VERSION=${VERSION:-0.0.0}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/jmorganca/ollama/version.Version=$VERSION\" \"-X=github.com/jmorganca/ollama/server.mode=release\"'"

mkdir -p dist

for TARGETARCH in amd64 arm64; do
    docker buildx build --load --progress=plain --platform=linux/$TARGETARCH --build-arg=VERSION --build-arg=GOFLAGS -f Dockerfile.build -t gpubuilder:$TARGETARCH .
    docker create --platform linux/$TARGETARCH --name gpubuilder-$TARGETARCH gpubuilder:$TARGETARCH
    docker cp gpubuilder-$TARGETARCH:/go/src/github.com/jmorganca/ollama/ollama ./dist/ollama-linux-$TARGETARCH
    docker rm gpubuilder-$TARGETARCH

    docker buildx build --load --progress=plain --platform=linux/$TARGETARCH --build-arg=VERSION --build-arg=GOFLAGS -f Dockerfile.cpu -t cpubuilder:$TARGETARCH .
    docker create --platform linux/$TARGETARCH --name cpubuilder-$TARGETARCH cpubuilder:$TARGETARCH
    docker cp cpubuilder-$TARGETARCH:/go/src/github.com/jmorganca/ollama/ollama ./dist/ollama-linux-$TARGETARCH-cpu
    docker rm cpubuilder-$TARGETARCH

done
