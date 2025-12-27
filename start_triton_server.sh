#!/bin/bash
# Start Triton Inference Server

set -e

MODEL_REPOSITORY=${1:-"triton_model_repository"}

echo "Starting Triton Inference Server..."
echo "Model repository: $MODEL_REPOSITORY"

# Check if model repository exists
if [ ! -d "$MODEL_REPOSITORY" ]; then
    echo "Error: Model repository not found at $MODEL_REPOSITORY"
    exit 1
fi

# Start Triton server
tritonserver \
    --model-repository="$MODEL_REPOSITORY" \
    --strict-model-config=false \
    --log-verbose=1

echo "Triton server started!"
echo "HTTP endpoint: http://localhost:8000"
echo "gRPC endpoint: localhost:8001"
echo "Metrics endpoint: http://localhost:8002/metrics"



