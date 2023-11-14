#!/bin/bash

# This script sets up integration tests which run the full stack to verify
# inference locally
set -e
set -o pipefail

REPO=$(dirname $0)/../
export OLLAMA_MODELS=${REPO}/test_data/models
REGISTRY_SCHEME=https
REGISTRY=registry.ollama.ai
TEST_MODEL=library/orca-mini
TEST_MODEL_TAG=latest
ACCEPT_HEADER="Accept: application/vnd.docker.distribution.manifest.v2+json"

mkdir -p ${OLLAMA_MODELS}/manifests/${REGISTRY}/${TEST_MODEL}/
mkdir -p ${OLLAMA_MODELS}/blobs/

echo "Pulling manifest for ${TEST_MODEL}:${TEST_MODEL_TAG}"
curl -s --header "${ACCEPT_HEADER}" \
    -o ${OLLAMA_MODELS}/manifests/${REGISTRY}/${TEST_MODEL}/${TEST_MODEL_TAG} \
    ${REGISTRY_SCHEME}://${REGISTRY}/v2/${TEST_MODEL}/manifests/${TEST_MODEL_TAG} 

CFG_HASH=$(cat ${OLLAMA_MODELS}/manifests/${REGISTRY}/${TEST_MODEL}/${TEST_MODEL_TAG} | jq -r ".config.digest")
echo "Pulling config blob ${CFG_HASH}"
curl -L -C - --header "${ACCEPT_HEADER}" \
        -o ${OLLAMA_MODELS}/blobs/${CFG_HASH} \
        ${REGISTRY_SCHEME}://${REGISTRY}/v2/${TEST_MODEL}/blobs/${CFG_HASH}

for LAYER in $(cat ${OLLAMA_MODELS}/manifests/${REGISTRY}/${TEST_MODEL}/${TEST_MODEL_TAG} | jq -r ".layers[].digest" ) ; do
    echo "Pulling blob ${LAYER}"
    curl -L -C - --header "${ACCEPT_HEADER}" \
        -o ${OLLAMA_MODELS}/blobs/${LAYER} \
        ${REGISTRY_SCHEME}://${REGISTRY}/v2/${TEST_MODEL}/blobs/${LAYER}
done