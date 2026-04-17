# Makefile for TensorRT Inference Docker Build and Test
# No hard-codes - all values configurable via variables or environment

# Docker configuration
DOCKERFILE_DIR := dockerfiles
DOCKERFILE := $(DOCKERFILE_DIR)/direct_trt.Dockerfile
IMAGE_NAME ?= trt-inference
IMAGE_TAG ?= latest
DOCKER_BUILDKIT := 1

# Model configuration
MODEL_SOURCE_DIR ?= ~/server/policies/vtol_hover
MODEL_NAME ?= vtol_hover
ONNX_FILE ?= model.onnx
METADATA_FILE ?= observations_metadata.yaml
LOCAL_MODEL_DIR := ./models

# Container configuration
CONTAINER_NAME ?= trt-smoke-test
GPU_DEVICE ?= all
TAG ?= 12.5.1-cudnn-devel-ubuntu22.04

# Build arguments
BASE_IMAGE ?= nvidia/cuda:$(TAG)
# base image for jetson should cuda 12.6 cuDNN 8.9.6, onnx-runtime 1.16.0, gcc 1.14, python 3.10
# base image for host should cuda 12.5 cuDNN 8.9.7, onnx-runtime 1.16.0, gcc 8.3.1, python 3.10
PYTHON_VERSION ?= 3.10
TENSORRT_VERSION ?= 10.3.0
ONNXRUNTIME_VERSION ?= 1.16.0
CUDA_VERSION ?= 12.5
MODEL_DIR ?= /opt/models
APP_DIR ?= /opt/app

# Test configuration
TEST_TIMEOUT ?= 300
TEST_COMMAND ?= python3 $(APP_DIR)/inference_smoke_test.py

.PHONY: all build test clean help \
        verify-docker verify-host-gpu inspect-base-image verify-image-runtime smoke-test

# Default target
all: build smoke-test

# Help target
help:
	@echo "TensorRT Inference Docker Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  build              - Build the Docker image"
	@echo "  smoke-test         - Run application smoke test (end-to-end usability)"
	@echo "  test               - Alias for smoke-test"
	@echo "  clean              - Remove Docker image and containers"
	@echo ""
	@echo "Verification & diagnostics:"
	@echo "  verify-docker      - Check Docker installation/daemon"
	@echo "  verify-host-gpu    - Check host GPU + Docker GPU compute readiness"
	@echo "  inspect-base-image - Inspect base image toolkit (diagnostic only)"
	@echo "  verify-image-runtime - Verify built image runtime contract (preflight)"
	@echo ""
	@echo "Development:"
	@echo "  shell              - Open interactive shell in container"
	@echo "  debug              - Start debug container"
	@echo "  help               - Show this help message"
	@echo ""
	@echo "Configuration (via environment or make variables):"
	@echo "  IMAGE_NAME          - Docker image name (default: $(IMAGE_NAME))"
	@echo "  IMAGE_TAG           - Docker image tag (default: $(IMAGE_TAG))"
	@echo "  MODEL_SOURCE_DIR    - Source directory for models (default: $(MODEL_SOURCE_DIR))"
	@echo "  BASE_IMAGE          - Base Docker image (default: $(BASE_IMAGE))"
	@echo "  PYTHON_VERSION      - Python version (default: $(PYTHON_VERSION))"
	@echo "  TENSORRT_VERSION    - TensorRT version (default: $(TENSORRT_VERSION))"
	@echo "  TEST_TIMEOUT        - Test timeout in seconds (default: $(TEST_TIMEOUT))"
	@echo "  GPU_DEVICE          - GPU device(s) to expose (default: $(GPU_DEVICE))"

# Prepare model files for Docker build
prepare-models:
	@echo "Preparing model files..."
	@mkdir -p $(LOCAL_MODEL_DIR)
	@echo "Copying latest model from $(MODEL_SOURCE_DIR)..."
	@LATEST_MODEL=$$(ls -td $(MODEL_SOURCE_DIR)/*/ 2>/dev/null | head -1); \
	if [ -z "$$LATEST_MODEL" ]; then \
		echo "No model directories found in $(MODEL_SOURCE_DIR)"; \
		exit 1; \
	fi; \
	echo "Using model: $$LATEST_MODEL"; \
	cp -r "$$LATEST_MODEL" "$(LOCAL_MODEL_DIR)/$(MODEL_NAME)/"
	@echo "Model files prepared in $(LOCAL_MODEL_DIR)/$(MODEL_NAME)/"

# Verify Docker only (no GPU assumptions)
verify-docker:
	@echo "Verifying Docker installation..."
	@docker --version || (echo "Docker not installed" && exit 1)
	@docker info > /dev/null 2>&1 || (echo "Docker daemon not running" && exit 1)
	@echo "Checking Docker compose..."
	@docker compose version >/dev/null 2>&1 && echo "Docker Compose available" || echo "Docker Compose not available (optional)"
	@echo "Docker is ready"

# Verify host GPU compute readiness (independent of built image)
verify-host-gpu:
	@echo "Running host GPU + Docker compute readiness check..."
	@BASE_IMAGE=$(BASE_IMAGE) ./dockerfiles/check-host-gpu.sh

# Optional: inspect base image toolkit versions (diagnostic only)
inspect-base-image:
	@echo "Inspecting base image toolkit (diagnostic)..."
	@BASE_IMAGE=$(BASE_IMAGE) ./dockerfiles/inspect-base-image.sh

# Verify built image runtime contract (requires image to exist)
verify-image-runtime: build verify-host-gpu
	@echo "Verifying built image runtime contract..."
	@TARGET_IMAGE=$(IMAGE_NAME):$(IMAGE_TAG) ./dockerfiles/check-image-runtime.sh

# Build Docker image
build: verify-docker prepare-models
	@echo "Building Docker image $(IMAGE_NAME):$(IMAGE_TAG)..."
	@echo "Building with the following configuration:"
	@echo "  Base image: $(BASE_IMAGE)"
	@echo "  Python version: $(PYTHON_VERSION)"
	@echo "  TensorRT version: $(TENSORRT_VERSION)"
	@echo "  CUDA version: $(CUDA_VERSION)"
	@echo "  Model directory: $(MODEL_SOURCE_DIR)"
	@echo "  Model name: $(MODEL_NAME)"
	@echo ""
	@DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
		--build-arg TENSORRT_VERSION=$(TENSORRT_VERSION) \
		--build-arg ONNXRUNTIME_VERSION=$(ONNXRUNTIME_VERSION) \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		--build-arg MODEL_DIR=$(MODEL_DIR) \
		--build-arg APP_DIR=$(APP_DIR) \
		--build-arg MODEL_NAME=$(MODEL_NAME) \
		--tag $(IMAGE_NAME):$(IMAGE_TAG) \
		--file $(DOCKERFILE) \
		.
	@echo "Build completed: $(IMAGE_NAME):$(IMAGE_TAG)"

# End-to-end application smoke test (authoritative usability check)
smoke-test: build verify-host-gpu
	@echo "Running TensorRT smoke test (end-to-end)..."
	@echo "Test configuration:"
	@echo "  Container name: $(CONTAINER_NAME)"
	@echo "  GPU device: $(GPU_DEVICE)"
	@echo "  Timeout: $(TEST_TIMEOUT)s"
	@echo ""
	@docker rm -f $(CONTAINER_NAME) 2>/dev/null || true
	@docker run \
		--name $(CONTAINER_NAME) \
		--gpus '"device=$(GPU_DEVICE)"' \
		--rm \
		-e MODEL_NAME=$(MODEL_NAME) \
		-e ONNX_FILE=$(ONNX_FILE) \
		-e METADATA_FILE=$(METADATA_FILE) \
		-e MODEL_DIR=$(MODEL_DIR) \
		-e APP_DIR=$(APP_DIR) \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		$(TEST_COMMAND)
	@echo "Smoke test passed"

# Alias for backward compatibility
test: smoke-test

# Clean up Docker resources
clean:
	@echo "Cleaning up Docker resources..."
	@docker rm -f $(CONTAINER_NAME) 2>/dev/null || true
	@docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	@rm -rf $(LOCAL_MODEL_DIR)
	@echo "Cleanup completed"

# Development targets
.PHONY: shell debug

# Open interactive shell in container
shell: build verify-host-gpu
	@echo "Opening interactive shell..."
	@docker run \
		--name $(CONTAINER_NAME)-shell \
		--gpus '"device=$(GPU_DEVICE)"' \
		--rm -it \
		-e MODEL_NAME=$(MODEL_NAME) \
		-e ONNX_FILE=$(ONNX_FILE) \
		-e METADATA_FILE=$(METADATA_FILE) \
		-e MODEL_DIR=$(MODEL_DIR) \
		-e APP_DIR=$(APP_DIR) \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		/bin/bash

# Debug container with additional tools
debug: build verify-host-gpu
	@echo "Starting debug container..."
	@docker run \
		--name $(CONTAINER_NAME)-debug \
		--gpus '"device=$(GPU_DEVICE)"' \
		--rm -it \
		-e MODEL_NAME=$(MODEL_NAME) \
		-e ONNX_FILE=$(ONNX_FILE) \
		-e METADATA_FILE=$(METADATA_FILE) \
		-e MODEL_DIR=$(MODEL_DIR) \
		-e APP_DIR=$(APP_DIR) \
		--entrypoint /bin/bash \
		$(IMAGE_NAME):$(IMAGE_TAG)
