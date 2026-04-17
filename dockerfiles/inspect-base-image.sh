#!/usr/bin/env bash
#
# inspect-base-image.sh — Base image toolkit inventory (diagnostic only)
#
# This script inspects a base CUDA image and prints toolkit versions.
# It is NOT a gate for application usability.
#
# Environment:
#   BASE_IMAGE — base image to inspect (default: nvidia/cuda:12.6.0-runtime-ubuntu22.04)
#
# Exit codes:
#   0  — Success (inspection completed)
#   1  — Failure (image pull/run failed)

set -eo pipefail

print_header() {
  echo "--- $1 ---"
}

print_header_major() {
  echo "=== $1 ==="
}

print_success() {
  echo "[OK] $1"
}

print_warning() {
  echo "[WARN] $1"
}

print_error() {
  echo "[ERROR] $1"
}

print_info() {
  echo "  $1"
}

print_multiline_info() {
  local line
  while IFS= read -r line; do
    if [[ -n "${line//[[:space:]]/}" ]]; then
      print_info "$line"
    fi
  done <<< "$1"
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

docker_image_exists() {
  docker image inspect "$1" >/dev/null 2>&1
}

inspect_base_image() {
  local base_image="$1"
  local inspection_output
  local status=0

  print_header_major "BASE IMAGE TOOLKIT VERIFICATION"
  echo ""

  if [[ -z "$base_image" ]]; then
    print_warning "BASE_IMAGE not set - skipping base image inspection"
    echo ""
    return 0
  fi

  if ! command_exists docker; then
    print_warning "Docker not available - cannot inspect base image"
    echo ""
    return 0
  fi

  print_info "Base image: $base_image"
  echo ""

  # Pull base image if not available
  if ! docker_image_exists "$base_image"; then
    print_info "Pulling base image: $base_image..."
    if ! docker pull "$base_image" >/dev/null 2>&1; then
      print_error "Failed to pull base image: $base_image"
      echo ""
      return 1
    fi
    print_success "Base image pulled successfully"
    echo ""
  else
    print_success "Base image found locally"
    echo ""
  fi

  # Run comprehensive version check inside base image
  print_header "Toolkit Versions Inside Base Image"
  if inspection_output=$(docker run --rm "$base_image" bash -lc '
    echo "=== Python ==="
    python3 --version 2>&1 || python --version 2>&1 || echo "Python not found"

    echo ""
    echo "=== GCC ==="
    gcc --version 2>&1 | head -1 || g++ --version 2>&1 | head -1 || echo "GCC not found"

    echo ""
    echo "=== CUDA Toolkit ==="
    if command -v nvcc >/dev/null 2>&1; then
      nvcc --version 2>&1 | grep "release" || nvcc --version 2>&1
    elif [ -f /usr/local/cuda/version.txt ]; then
      echo "CUDA Version (from version.txt): $(cat /usr/local/cuda/version.txt)"
    elif [ -L /usr/local/cuda ]; then
      echo "CUDA symlink: $(readlink -f /usr/local/cuda)"
      if [ -f "$(readlink -f /usr/local/cuda)/version.txt" ]; then
        echo "CUDA Version: $(cat "$(readlink -f /usr/local/cuda)/version.txt")"
      fi
    else
      echo "CUDA toolkit not detected (runtime image may not include nvcc)"
    fi

    echo ""
    echo "=== CUDA Libraries (cuDNN, etc.) ==="
    ldconfig -p 2>/dev/null | grep -iE "cudnn|cublas|cudart|nvrtc|nvml" | sort -u || echo "No CUDA libraries found via ldconfig"

    echo ""
    echo "=== cuDNN Version (if available) ==="
    if [ -f /usr/include/cudnn_version.h ]; then
      echo "cuDNN header found at /usr/include/cudnn_version.h"
      grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL" /usr/include/cudnn_version.h 2>/dev/null || true
    elif [ -f /usr/include/cudnn.h ]; then
      echo "cuDNN header found at /usr/include/cudnn.h"
      grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL" /usr/include/cudnn.h 2>/dev/null || true
    elif [ -f /usr/local/cuda/include/cudnn_version.h ]; then
      echo "cuDNN header found at /usr/local/cuda/include/cudnn_version.h"
      grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL" /usr/local/cuda/include/cudnn_version.h 2>/dev/null || true
    elif [ -f /usr/local/cuda/include/cudnn.h ]; then
      echo "cuDNN header found at /usr/local/cuda/include/cudnn.h"
      grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL" /usr/local/cuda/include/cudnn.h 2>/dev/null || true
    else
      echo "cuDNN headers not found in standard locations"
    fi

    echo ""
    echo "=== PyTorch ==="
    python3 -c "import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda}\")" 2>&1 || echo "PyTorch not installed or import failed"

    echo ""
    echo "=== NVIDIA Driver (in container) ==="
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1
    else
      echo "nvidia-smi not available in container (expected for runtime-only images)"
    fi

    echo ""
    echo "=== NVIDIA-SMI (full output) ==="
    nvidia-smi 2>&1 || echo "nvidia-smi not available"
  ' 2>/dev/null); then
    print_success "Base image inspection completed"
    print_multiline_info "$inspection_output"
  else
    print_error "Failed to inspect base image: $base_image"
    status=1
  fi

  echo ""
  return "$status"
}

main() {
  local base_image="${BASE_IMAGE:-nvidia/cuda:12.6.0-runtime-ubuntu22.04}"
  inspect_base_image "$base_image"
}

main
