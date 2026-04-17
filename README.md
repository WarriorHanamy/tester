# TensorRT Multi-Stage Docker Environment

Docker-based TensorRT workflow that separates engine compilation from runtime inference.

## Architecture

- Builder image: based on `nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04`, installs TensorRT CLI tooling and uses `trtexec` to compile `ONNX -> .engine`
- Runtime image: based on `nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04`, contains only the runtime stack needed to load and execute the generated engine
- Makefile: orchestrates model preparation, engine generation, runtime image build, smoke tests, and diagnostics

## Quick Start

```bash
# Optional: verify host GPU and Docker GPU readiness
make verify-host-gpu

# Build builder image, compile engine, and build runtime image
make build

# Run the runtime smoke test
make smoke-test

# Inspect the built runtime image contract
make verify-image-runtime
```

## Main Targets

### Build pipeline
- `make build-builder` - Build the `trtexec` builder image
- `make build-engine` - Generate `engine-output/<model>.engine` from the ONNX model
- `make build-runtime` - Build the runtime image and bake the generated engine into it
- `make build-all` - Run the full builder -> engine -> runtime pipeline
- `make build` - Alias for `build-all`

### Validation and diagnostics
- `make smoke-test` - Run the runtime container and execute `run_inference.py`
- `make test` - Alias for `smoke-test`
- `make verify-docker` - Verify Docker daemon availability
- `make verify-host-gpu` - Verify host NVIDIA driver and Docker GPU access
- `make inspect-base-image` - Inspect the configured runtime base image
- `make verify-image-runtime` - Verify the built runtime image exposes the expected runtime contract

### Development
- `make shell` - Start an interactive shell in the runtime image
- `make debug` - Start a debug shell in the runtime image
- `make clean-all` - Remove generated models, engine artifacts, and both images
- `make clean` - Alias for `clean-all`

## Makefile Configuration

```bash
# Custom image names
IMAGE_NAME=my-trt-runtime BUILDER_IMAGE_NAME=my-trt-builder make build

# Custom model source
MODEL_SOURCE_DIR=/path/to/policies make build

# Custom engine filename
ENGINE_FILE=my_policy.engine make build-runtime

# Custom trtexec flags
TRTEXEC_FLAGS='--fp16 --verbose' make build-engine
```

Important variables:

- `IMAGE_NAME` - Runtime image name
- `BUILDER_IMAGE_NAME` - Builder image name
- `IMAGE_TAG` - Shared image tag for both images
- `MODEL_SOURCE_DIR` - Host directory containing versioned model directories
- `MODEL_NAME` - Logical model name copied into `./models/<MODEL_NAME>`
- `ONNX_FILE` - ONNX filename inside the prepared model directory
- `METADATA_FILE` - Metadata filename copied into the runtime image
- `ENGINE_OUTPUT_DIR` - Host directory where generated engines are stored
- `ENGINE_FILE` - Generated engine filename
- `TRTEXEC_FLAGS` - Extra `trtexec` flags passed during engine generation (default: `--fp16 --skipInference`)
- `GPU_DEVICE` - GPU selector for `docker run`
- `BUILDER_BASE_IMAGE` - CUDA devel image for the builder
- `RUNTIME_BASE_IMAGE` - CUDA runtime image for the runtime image

## trtexec Guide

`trtexec` is NVIDIA's production CLI for TensorRT engine building, benchmarking, and inspection.

### Project-integrated usage

```bash
# Default compile-only FP16 build
make build-engine

# FP16 build with verbose logs
TRTEXEC_FLAGS='--fp16 --verbose' make build-engine

# Dynamic shape example
TRTEXEC_FLAGS='--fp16 --minShapes=input:1x14 --optShapes=input:8x14 --maxShapes=input:32x14' make build-engine

# Rebuild runtime image after generating a new engine
make build-runtime
```

### Manual builder invocation

```bash
make build-builder prepare-models
mkdir -p engine-output

docker run --rm --gpus all \
  -v "$PWD/models:/opt/models" \
  -v "$PWD/engine-output:/opt/engine" \
  trt-inference-builder:latest \
  --onnx=/opt/models/vtol_hover/model.onnx \
  --saveEngine=/opt/engine/vtol_hover.engine \
  --fp16 --skipInference
```

### Common trtexec flags

- `--fp16` - Enable FP16 tactics
- `--skipInference` - Build the engine without running the default benchmark pass
- `--int8` - Enable INT8 tactics
- `--shapes=name:1x14` - Set a fixed input shape
- `--minShapes`, `--optShapes`, `--maxShapes` - Configure dynamic shape profiles
- `--verbose` - Print detailed builder logs
- `--workspace=N` - Control workspace size in MiB
- `--loadEngine=file.engine` - Load an existing engine for benchmarking

### Benchmark an existing engine

```bash
docker run --rm --gpus all \
  -v "$PWD/engine-output:/opt/engine" \
  trt-inference-builder:latest \
  --loadEngine=/opt/engine/vtol_hover.engine \
  --useCudaGraph \
  --noDataTransfers
```

## Runtime Contract

The runtime image now expects:

- `run_inference.py` under `/opt/app`
- prebuilt engine artifact under `/opt/engine/<ENGINE_FILE>`
- optional metadata under `/opt/models/<MODEL_NAME>/<METADATA_FILE>`

The runtime image does not need:

- `trtexec`
- ONNX parser tooling for engine build
- `onnxruntime`
- `onnx`
- the full CUDA devel toolchain

## Directory Structure

```text
.
├── dockerfiles/
│   ├── builder.Dockerfile
│   ├── runtime.Dockerfile
│   ├── run_inference.py
│   ├── check-host-gpu.sh
│   ├── inspect-base-image.sh
│   └── check-image-runtime.sh
├── engine-output/                  # Generated TensorRT engines
├── models/                         # Prepared model copy for local builds
├── Makefile
└── README.md
```

## Troubleshooting

### Docker cannot see the GPU

```bash
make verify-host-gpu
```

### Runtime image contract check fails

```bash
make verify-image-runtime
```

### trtexec engine build fails

```bash
# Re-run with verbose logs and explicit shapes if needed
TRTEXEC_FLAGS='--fp16 --verbose --shapes=input:1x14' make build-engine
```

### Smoke test fails after engine generation

```bash
make smoke-test
make shell
```
