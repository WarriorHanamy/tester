#!/usr/bin/env bash
#
# check-image-runtime.sh — Target image runtime contract presence check
#
# Purpose:
#   Verify that the built application image starts and exposes the expected
#   Python environment, libraries, and entrypoint without executing the full
#   end-to-end inference. This is a lightweight preflight, not the authoritative
#   application usability test.
#
# Environment:
#   TARGET_IMAGE — image to check (default: IMAGE_NAME:IMAGE_TAG from Makefile)
#
# Exit codes:
#   0  — Success (image starts, required Python packages importable)
#   1  — Failure

set -eo pipefail

print_header() {
  echo "--- $1 ---"
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

inspect_target_image() {
  local target_image="$1"
  local inspection_output
  local status=0

  print_header "Target Image Runtime Contract Check"
  echo ""

  if [[ -z "$target_image" ]]; then
    print_warning "TARGET_IMAGE not set - skipping target image inspection"
    echo ""
    return 0
  fi

  if ! command_exists docker; then
    print_warning "Docker not available - cannot inspect target image"
    echo ""
    return 0
  fi

  if ! docker_image_exists "$target_image"; then
    print_warning "Target image not found locally: $target_image"
    print_info "Build it first with: make build"
    echo ""
    return 0
  fi

  print_success "Found target image: $target_image"
  print_info "Starting container with GPU access to verify runtime environment..."
  echo ""

  if inspection_output=$(docker run --rm --gpus all --entrypoint /bin/bash "$target_image" -lc '
    echo "=== Python version ==="
    python3 --version

    echo ""
    echo "=== Required Python packages ==="
    python3 - <<"PY"
import importlib

packages = [
    ("tensorrt", "TensorRT"),
    ("onnxruntime", "ONNX Runtime"),
    ("onnx", "ONNX"),
    ("numpy", "NumPy"),
    ("pycuda", "PyCUDA"),
    ("yaml", "PyYAML"),
]

for module_name, label in packages:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"{label}: {version}")
        if module_name == "onnxruntime":
            providers = ", ".join(module.get_available_providers())
            print(f"  ONNX Runtime providers: {providers}")
    except Exception as exc:
        print(f"{label}: not available ({exc.__class__.__name__}: {exc})")
PY

    echo ""
    echo "=== CUDA libraries (ldconfig) ==="
    ldconfig -p 2>/dev/null | grep -E "libnvinfer|libcudart|libcublas|libcudnn|libnvonnxparser" || true

    echo ""
    echo "=== App entry script presence ==="
    if [ -f "/opt/app/inference_smoke_test.py" ]; then
      echo "Found: /opt/app/inference_smoke_test.py"
    else
      echo "Missing: /opt/app/inference_smoke_test.py"
    fi

    echo ""
    echo "=== Model directory presence ==="
    if [ -d "/opt/models/vtol_hover" ]; then
      echo "Found model dir: /opt/models/vtol_hover"
      ls -1 /opt/models/vtol_hover/ 2>/dev/null || true
    else
      echo "Missing model dir: /opt/models/vtol_hover"
    fi
  ' 2>/dev/null); then
    print_success "Target image started successfully with GPU"
    print_multiline_info "$inspection_output"
  else
    print_error "Target image failed to start with GPU: $target_image"
    echo ""
    return 1
  fi

  echo ""
  return 0
}

main() {
  local target_image="${TARGET_IMAGE:-${IMAGE_NAME:-trt-inference}:${IMAGE_TAG:-latest}}"
  inspect_target_image "$target_image"
}

main
