#!/bin/bash
set -e

docker run --rm --gpus all \
  -v "$PWD:/workspace" -w /workspace \
  nvcr.io/nvidia/tensorrt:26.03-py3 \
  trtexec \
   --onnx=../models/vtol_hover/model.onnx \
   --saveEngine=../engine-output/vtol_hover.fp32.trt.engine \
   --skipInference \
   --memPoolSize=workspace:4096 \
   --timingCacheFile=../cache/vtol_hover.cache \
  --verbose
