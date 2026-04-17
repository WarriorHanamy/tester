#!/bin/bash
set -e

# Create resnet50 directory
mkdir -p resnet50

# Download ResNet50 ONNX model from HuggingFace if not exists
echo "Checking for ResNet50 ONNX model..."
if [ ! -f "resnet50/model.onnx" ]; then
  echo "Downloading ResNet50 ONNX model..."
  wget -O resnet50/model.onnx https://huggingface.co/onnxmodelzoo/resnet50-v1-12/resolve/main/resnet50-v1-12.onnx
else
  echo "Model already exists, skipping download."
fi

# Convert to TensorRT engine if not exists
echo "Checking for TensorRT engine..."
if [ ! -f "resnet_engine_intro.engine" ]; then
  echo "Converting ONNX to TensorRT engine..."
  docker run --rm \
    --gpus all \
    -v "$(pwd)/resnet50:/workspace/resnet50" \
    -v "$(pwd):/workspace" \
    nvcr.io/nvidia/tensorrt:26.03-py3 \
    trtexec --onnx=/workspace/resnet50/model.onnx --saveEngine=/workspace/resnet_engine_intro.engine --stronglyTyped
else
  echo "Engine already exists, skipping conversion."
fi

echo "Done! Engine saved to resnet_engine_intro.engine"
