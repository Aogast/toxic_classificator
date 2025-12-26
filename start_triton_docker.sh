#!/bin/bash
# Start Triton Inference Server via Docker (without GPU)

set -e

MODEL_REPO=$(pwd)/triton_model_repository

echo "Starting Triton Inference Server via Docker (CPU mode)..."
echo "Model repository: $MODEL_REPO"

# Pull Triton image if not exists
if ! docker images | grep -q "nvcr.io/nvidia/tritonserver"; then
    echo "Pulling Triton Server image..."
    docker pull nvcr.io/nvidia/tritonserver:24.01-py3
fi

# Start Triton server (CPU only, no --gpus flag)
echo "Starting Triton Server on ports 8000, 8001, 8002..."
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v "$MODEL_REPO":/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models --log-verbose=1

echo ""
echo "Triton server started!"
echo "HTTP endpoint: http://localhost:8000"
echo "gRPC endpoint: localhost:8001"
echo "Metrics endpoint: http://localhost:8002/metrics"

