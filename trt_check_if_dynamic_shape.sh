#!/bin/bash
set -e

docker run --rm --gpus all \
  -e POLYGRAPHY_AUTOINSTALL_DEPS=1 \
  -v "$PWD:/workspace" -w /workspace \
  nvcr.io/nvidia/tensorrt:26.03-py3 \
  polygraphy inspect model models/vtol_hover/model.onnx
