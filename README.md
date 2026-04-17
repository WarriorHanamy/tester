# TensorRT Inference Docker Environment

Docker-based environment for ONNX model inference using TensorRT with NVIDIA GPU support.

## Quick Start

```bash
# Build the Docker image
make build

# Run smoke test
make test

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

- `make build` - Build the Docker image
- `make test` - Run smoke test in container
- `make clean` - Remove Docker image and containers
- `make verify` - Verify Docker and GPU setup
- `make shell` - Open interactive shell in container
- `make debug` - Start debug container
- `make logs` - Show container logs
- `make info` - Show image information
- `make help` - Show all available targets

## Directory Structure

```
.
├── dockerfiles/
│   ├── direct_trt.Dockerfile    # Main Dockerfile
│   └── inference_smoke_test.py  # Smoke test script
├── models/                      # Local model copy (auto-generated)
├── Makefile                     # Build and test automation
└── README.md
```

## Environment Variables

The following environment variables are available in the container:

- `MODEL_NAME` - Name of the model directory (default: vtol_hover)
- `ONNX_FILE` - ONNX model filename (default: model.onnx)
- `METADATA_FILE` - Metadata filename (default: observations_metadata.yaml)
- `MODEL_DIR` - Container model directory (default: /opt/models)
- `APP_DIR` - Container app directory (default: /opt/app)

## GPU Requirements

- NVIDIA GPU with CUDA support
- NVIDIA drivers installed
- nvidia-container-toolkit installed for Docker GPU support

## Troubleshooting

### GPU not available in container

```bash
# Verify Docker GPU support
make verify-gpu

# Or manually test (replace with your BASE_IMAGE if customized)
docker run --rm --gpus all "${BASE_IMAGE:-nvidia/cuda:12.6.0-runtime-ubuntu22.04}" nvidia-smi
```

### Model not found

Ensure the model directory exists:
```bash
ls -la ~/server/policies/vtol_hover/
```

The Makefile automatically copies the latest model directory to `./models/` during build.