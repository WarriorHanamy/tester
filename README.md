# TensorRT Inference Docker Environment

Docker-based environment for ONNX model inference using TensorRT with NVIDIA GPU support.

## Quick Start

```bash
# Verify host GPU compute readiness (optional diagnostic)
make verify-host-gpu

# Build the Docker image
make build

# Run end-to-end smoke test (authoritative application usability check)
make smoke-test   # or: make test

# Interactive shell
make shell
```

## Configuration

All configuration is done via Makefile variables or environment variables:

```bash
# Custom image name
IMAGE_NAME=my-trt-model make build

# Custom model directory
MODEL_SOURCE_DIR=/path/to/models make build

# Custom TensorRT version
TENSORRT_VERSION=10.1.0 make build
```

## Available Targets

### Build & Test
- `make build` — Build the Docker image
- `make smoke-test` — Run end-to-end application smoke test (recommended)
- `make test` — Alias for `smoke-test`
- `make clean` — Remove Docker image, containers, and local model copy

### Verification & Diagnostics
- `make verify-docker` — Check Docker installation and daemon
- `make verify-host-gpu` — Check host GPU drivers and Docker GPU compute readiness
- `make inspect-base-image` — Inspect base CUDA image toolkit versions (diagnostic only)
- `make verify-image-runtime` — Verify built image starts and exposes required runtime contract (preflight)

### Development
- `make shell` — Open interactive shell in the built container
- `make debug` — Start debug container with bash entrypoint
- `make help` — Show all available targets and configuration

## Directory Structure

```
.
├── dockerfiles/
│   ├── direct_trt.Dockerfile          # Main Dockerfile
│   ├── inference_smoke_test.py        # Smoke test script (end-to-end app)
│   ├── check-host-gpu.sh              # Host GPU compute readiness check
│   ├── inspect-base-image.sh          # Base image toolkit diagnostics
│   └── check-image-runtime.sh         # Built image runtime contract check
├── models/                             # Local model copy (auto-generated)
├── Makefile                            # Build, verify, and test automation
└── README.md
```

## Environment Variables

### Container runtime
- `MODEL_NAME` — Model directory name (default: vtol_hover)
- `ONNX_FILE` — ONNX model filename (default: model.onnx)
- `METADATA_FILE` — Metadata filename (default: observations_metadata.yaml)
- `MODEL_DIR` — Container model directory (default: /opt/models)
- `APP_DIR` — Container app directory (default: /opt/app)

### Build & runtime control
- `IMAGE_NAME` — Docker image name (default: trt-inference)
- `IMAGE_TAG` — Docker image tag (default: latest)
- `BASE_IMAGE` — Base CUDA image for build (default: nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04)
- `PYTHON_VERSION` — Python version to install (default: 3.10)
- `TENSORRT_VERSION` — TensorRT pip package version (default: 10.3.0)
- `ONNXRUNTIME_VERSION` — ONNX Runtime GPU version (default: 1.16.0)
- `CUDA_VERSION` — CUDA version (default: 12.5)
- `GPU_DEVICE` — GPU device(s) to expose to container (default: all)
- `MODEL_SOURCE_DIR` — Host path containing model directories (default: ~/server/policies/vtol_hover)

## GPU Requirements

- NVIDIA GPU with CUDA support
- NVIDIA driver installed on host
- `nvidia-container-toolkit` installed for Docker GPU support
- `Docker` daemon running with NVIDIA runtime configured

## Troubleshooting

### GPU not available in container

```bash
# 1) Verify host GPU compute readiness
make verify-host-gpu

# 2) Verify the built image can start with GPU
make verify-image-runtime

# Or manually test (replace with your BASE_IMAGE if customized)
docker run --rm --gpus all "${BASE_IMAGE:-nvidia/cuda:12.6.0-runtime-ubuntu22.04}" nvidia-smi
```

### Smoke test fails

```bash
# Check model files exist
ls -la ~/server/policies/vtol_hover/

# Rebuild with verbose output
make clean && IMAGE_NAME=my-trt-model make build
make smoke-test
```

### Model not found

Ensure the model directory exists:
```bash
ls -la ~/server/policies/vtol_hover/
```

The Makefile automatically copies the latest model directory to `./models/` during build.