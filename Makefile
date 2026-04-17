# Makefile for TensorRT multi-stage engine build and runtime test

# Docker configuration
DOCKERFILE_DIR := dockerfiles
BUILDER_DOCKERFILE := $(DOCKERFILE_DIR)/builder.Dockerfile
RUNTIME_DOCKERFILE := $(DOCKERFILE_DIR)/runtime.Dockerfile
IMAGE_NAME ?= trt-inference
BUILDER_IMAGE_NAME ?= $(IMAGE_NAME)-builder
IMAGE_TAG ?= latest
DOCKER_BUILDKIT ?= 1

RUNTIME_IMAGE := $(IMAGE_NAME):$(IMAGE_TAG)
BUILDER_IMAGE := $(BUILDER_IMAGE_NAME):$(IMAGE_TAG)

# Model configuration
MODEL_SOURCE_DIR ?= ~/server/policies/vtol_hover
MODEL_NAME ?= vtol_hover
ONNX_FILE ?= model.onnx
METADATA_FILE ?= observations_metadata.yaml
LOCAL_MODEL_DIR := ./models
LOCAL_MODEL_PATH := $(LOCAL_MODEL_DIR)/$(MODEL_NAME)
ENGINE_OUTPUT_DIR ?= ./engine-output
ENGINE_FILE ?= $(MODEL_NAME).engine
ENGINE_HOST_PATH := $(ENGINE_OUTPUT_DIR)/$(ENGINE_FILE)

# Container configuration
CONTAINER_NAME ?= trt-smoke-test
GPU_DEVICE ?= all
BUILDER_TAG ?= 12.5.1-cudnn-devel-ubuntu22.04
RUNTIME_TAG ?= 12.5.1-cudnn-runtime-ubuntu22.04

# Build arguments
BUILDER_BASE_IMAGE ?= nvidia/cuda:$(BUILDER_TAG)
RUNTIME_BASE_IMAGE ?= nvidia/cuda:$(RUNTIME_TAG)
BASE_IMAGE ?= $(RUNTIME_BASE_IMAGE)
PYTHON_VERSION ?= 3.10
MODEL_DIR ?= /opt/models
APP_DIR ?= /opt/app
ENGINE_DIR ?= /opt/engine

# trtexec configuration
TRTEXEC_FLAGS ?= --fp16 --skipInference

# Test configuration
TEST_TIMEOUT ?= 300
TEST_COMMAND ?= python3 $(APP_DIR)/run_inference.py

ifeq ($(GPU_DEVICE),all)
DOCKER_GPUS := all
else
DOCKER_GPUS := '"device=$(GPU_DEVICE)"'
endif

.PHONY: all build build-all build-builder build-engine build-runtime \
	clean clean-all help prepare-models prepare-engine-output \
	verify-docker verify-host-gpu inspect-base-image verify-image-runtime \
	smoke-test test shell debug

all: build smoke-test

help:
	@echo "TensorRT Multi-Stage Build System"
	@echo ""
	@echo "Build pipeline:"
	@echo "  build-builder       - Build the trtexec builder image"
	@echo "  build-engine        - Run trtexec to compile ONNX into a TensorRT engine"
	@echo "  build-runtime       - Build the runtime image with the generated engine"
	@echo "  build-all           - Run builder, engine generation, and runtime build"
	@echo "  build               - Alias for build-all"
	@echo ""
	@echo "Test & diagnostics:"
	@echo "  smoke-test          - Run end-to-end runtime smoke test"
	@echo "  test                - Alias for smoke-test"
	@echo "  verify-docker       - Check Docker installation and daemon"
	@echo "  verify-host-gpu     - Check host GPU and Docker GPU readiness"
	@echo "  inspect-base-image  - Inspect runtime base image toolkit (diagnostic only)"
	@echo "  verify-image-runtime - Verify built runtime image contract"
	@echo ""
	@echo "Development:"
	@echo "  shell               - Open interactive shell in the runtime container"
	@echo "  debug               - Start debug shell in the runtime container"
	@echo "  clean-all           - Remove builder/runtime images and generated artifacts"
	@echo "  clean               - Alias for clean-all"
	@echo ""
	@echo "Configuration:"
	@echo "  IMAGE_NAME          - Runtime image name (default: $(IMAGE_NAME))"
	@echo "  BUILDER_IMAGE_NAME  - Builder image name (default: $(BUILDER_IMAGE_NAME))"
	@echo "  IMAGE_TAG           - Shared image tag (default: $(IMAGE_TAG))"
	@echo "  MODEL_SOURCE_DIR    - Source directory for models (default: $(MODEL_SOURCE_DIR))"
	@echo "  BUILDER_BASE_IMAGE  - Builder CUDA image (default: $(BUILDER_BASE_IMAGE))"
	@echo "  RUNTIME_BASE_IMAGE  - Runtime CUDA image (default: $(RUNTIME_BASE_IMAGE))"
	@echo "  ENGINE_OUTPUT_DIR   - Host directory for generated engines (default: $(ENGINE_OUTPUT_DIR))"
	@echo "  ENGINE_FILE         - Engine filename (default: $(ENGINE_FILE))"
	@echo "  TRTEXEC_FLAGS       - Extra trtexec flags (default: $(TRTEXEC_FLAGS))"
	@echo "  GPU_DEVICE          - GPU device(s) to expose (default: $(GPU_DEVICE))"

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
	rm -rf "$(LOCAL_MODEL_PATH)"; \
	cp -r "$$LATEST_MODEL" "$(LOCAL_MODEL_PATH)"
	@echo "Model files prepared in $(LOCAL_MODEL_PATH)"

prepare-engine-output:
	@mkdir -p $(ENGINE_OUTPUT_DIR)

verify-docker:
	@echo "Verifying Docker installation..."
	@docker --version || (echo "Docker not installed" && exit 1)
	@docker info > /dev/null 2>&1 || (echo "Docker daemon not running" && exit 1)
	@echo "Checking Docker compose..."
	@docker compose version >/dev/null 2>&1 && echo "Docker Compose available" || echo "Docker Compose not available (optional)"
	@echo "Docker is ready"

verify-host-gpu:
	@echo "Running host GPU + Docker compute readiness check..."
	@BASE_IMAGE=$(BASE_IMAGE) bash ./dockerfiles/check-host-gpu.sh

inspect-base-image:
	@echo "Inspecting base image toolkit (diagnostic)..."
	@BASE_IMAGE=$(BASE_IMAGE) bash ./dockerfiles/inspect-base-image.sh

build-builder: verify-docker
	@echo "Building builder image $(BUILDER_IMAGE)..."
	@echo "  Builder base image: $(BUILDER_BASE_IMAGE)"
	@DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build \
		--build-arg BASE_IMAGE=$(BUILDER_BASE_IMAGE) \
		--tag $(BUILDER_IMAGE) \
		--file $(BUILDER_DOCKERFILE) \
		.
	@echo "Builder image ready: $(BUILDER_IMAGE)"

build-engine: build-builder prepare-models prepare-engine-output verify-host-gpu
	@echo "Generating TensorRT engine $(ENGINE_HOST_PATH)..."
	@echo "  Builder image: $(BUILDER_IMAGE)"
	@echo "  trtexec flags: $(TRTEXEC_FLAGS)"
	@docker run --rm \
		--gpus $(DOCKER_GPUS) \
		-v "$(CURDIR)/$(ENGINE_OUTPUT_DIR):$(ENGINE_DIR)" \
		-v "$(CURDIR)/$(LOCAL_MODEL_DIR):$(MODEL_DIR)" \
		$(BUILDER_IMAGE) \
		--onnx=$(MODEL_DIR)/$(MODEL_NAME)/$(ONNX_FILE) \
		--saveEngine=$(ENGINE_DIR)/$(ENGINE_FILE) \
		$(TRTEXEC_FLAGS)
	@test -f $(ENGINE_HOST_PATH)
	@echo "Engine generated: $(ENGINE_HOST_PATH)"

build-runtime: build-engine verify-docker
	@echo "Building runtime image $(RUNTIME_IMAGE)..."
	@echo "  Runtime base image: $(RUNTIME_BASE_IMAGE)"
	@echo "  Engine file: $(ENGINE_HOST_PATH)"
	@DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build \
		--build-arg BASE_IMAGE=$(RUNTIME_BASE_IMAGE) \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
		--build-arg MODEL_DIR=$(MODEL_DIR) \
		--build-arg APP_DIR=$(APP_DIR) \
		--build-arg ENGINE_DIR=$(ENGINE_DIR) \
		--build-arg MODEL_NAME=$(MODEL_NAME) \
		--build-arg METADATA_FILE=$(METADATA_FILE) \
		--tag $(RUNTIME_IMAGE) \
		--file $(RUNTIME_DOCKERFILE) \
		.
	@echo "Runtime image ready: $(RUNTIME_IMAGE)"

build-all: build-runtime

build: build-all

verify-image-runtime: build-runtime verify-host-gpu
	@echo "Verifying built runtime image contract..."
	@TARGET_IMAGE=$(RUNTIME_IMAGE) \
	APP_DIR=$(APP_DIR) \
	ENGINE_DIR=$(ENGINE_DIR) \
	ENGINE_FILE=$(ENGINE_FILE) \
	MODEL_DIR=$(MODEL_DIR) \
	MODEL_NAME=$(MODEL_NAME) \
	METADATA_FILE=$(METADATA_FILE) \
	bash ./dockerfiles/check-image-runtime.sh

smoke-test: build-runtime verify-host-gpu
	@echo "Running TensorRT smoke test (runtime image)..."
	@echo "  Container name: $(CONTAINER_NAME)"
	@echo "  GPU device: $(GPU_DEVICE)"
	@echo "  Engine file: $(ENGINE_FILE)"
	@echo "  Timeout: $(TEST_TIMEOUT)s"
	@docker rm -f $(CONTAINER_NAME) 2>/dev/null || true
	@docker run \
		--name $(CONTAINER_NAME) \
		--gpus $(DOCKER_GPUS) \
		--rm \
		-e MODEL_NAME=$(MODEL_NAME) \
		-e METADATA_FILE=$(METADATA_FILE) \
		-e MODEL_DIR=$(MODEL_DIR) \
		-e APP_DIR=$(APP_DIR) \
		-e ENGINE_DIR=$(ENGINE_DIR) \
		-e ENGINE_FILE=$(ENGINE_FILE) \
		$(RUNTIME_IMAGE) \
		$(TEST_COMMAND)
	@echo "Smoke test passed"

test: smoke-test

shell: build-runtime verify-host-gpu
	@echo "Opening runtime shell..."
	@docker run \
		--name $(CONTAINER_NAME)-shell \
		--gpus $(DOCKER_GPUS) \
		--rm -it \
		-e MODEL_NAME=$(MODEL_NAME) \
		-e METADATA_FILE=$(METADATA_FILE) \
		-e MODEL_DIR=$(MODEL_DIR) \
		-e APP_DIR=$(APP_DIR) \
		-e ENGINE_DIR=$(ENGINE_DIR) \
		-e ENGINE_FILE=$(ENGINE_FILE) \
		$(RUNTIME_IMAGE) \
		/bin/bash

debug: build-runtime verify-host-gpu
	@echo "Starting runtime debug shell..."
	@docker run \
		--name $(CONTAINER_NAME)-debug \
		--gpus $(DOCKER_GPUS) \
		--rm -it \
		-e MODEL_NAME=$(MODEL_NAME) \
		-e METADATA_FILE=$(METADATA_FILE) \
		-e MODEL_DIR=$(MODEL_DIR) \
		-e APP_DIR=$(APP_DIR) \
		-e ENGINE_DIR=$(ENGINE_DIR) \
		-e ENGINE_FILE=$(ENGINE_FILE) \
		--entrypoint /bin/bash \
		$(RUNTIME_IMAGE)

clean-all:
	@echo "Cleaning up Docker resources..."
	@docker rm -f $(CONTAINER_NAME) $(CONTAINER_NAME)-shell $(CONTAINER_NAME)-debug 2>/dev/null || true
	@docker rmi $(RUNTIME_IMAGE) 2>/dev/null || true
	@docker rmi $(BUILDER_IMAGE) 2>/dev/null || true
	@rm -rf $(LOCAL_MODEL_DIR) $(ENGINE_OUTPUT_DIR)
	@echo "Cleanup completed"

clean: clean-all
