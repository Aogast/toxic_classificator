#!/bin/bash
# Convert ONNX model to TensorRT

set -e

ONNX_MODEL=${1:-"models/model.onnx"}
TRT_MODEL=${2:-"models/model.trt"}

echo "Converting ONNX to TensorRT..."
echo "Input:  $ONNX_MODEL"
echo "Output: $TRT_MODEL"

# Using trtexec from TensorRT
trtexec \
    --onnx="$ONNX_MODEL" \
    --saveEngine="$TRT_MODEL" \
    --fp16 \
    --minShapes=input_ids:1x1,attention_mask:1x1 \
    --optShapes=input_ids:4x128,attention_mask:4x128 \
    --maxShapes=input_ids:8x512,attention_mask:8x512 \
    --verbose

echo "TensorRT model saved to: $TRT_MODEL"


