# NVIDIA CUDA runtime base image
ARG BASE_IMAGE=nvidia/cuda:12.6.0-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

# Build arguments for flexibility
ARG PYTHON_VERSION=3.10
ARG TENSORRT_VERSION=10.3.0
ARG ONNXRUNTIME_VERSION=1.17.1
ARG CUDA_VERSION=12.6
ARG MODEL_DIR=/opt/models
ARG APP_DIR=/opt/app
ARG MODEL_NAME=vtol_hover

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV CUDA_VERSION=${CUDA_VERSION}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    python${PYTHON_VERSION}-venv \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create Python symlink
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install TensorRT and ONNX Runtime with CUDA support
# Using pip packages for easier management
RUN pip install --no-cache-dir \
    tensorrt==${TENSORRT_VERSION} \
    onnx==1.15.0 \
    onnxruntime-gpu==${ONNXRUNTIME_VERSION} \
    numpy \
    pyyaml \
    pycuda

# Set up model directory
RUN mkdir -p ${MODEL_DIR}
RUN mkdir -p ${APP_DIR}

# Copy the vtol_hover model directory
COPY models/${MODEL_NAME}/ ${MODEL_DIR}/${MODEL_NAME}/

# Copy inference scripts
COPY dockerfiles/inference_smoke_test.py ${APP_DIR}/

# Set working directory
WORKDIR ${APP_DIR}

# Default command to run smoke test
CMD ["python3", "inference_smoke_test.py"]