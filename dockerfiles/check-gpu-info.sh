#!/usr/bin/env bash
#
# check-gpu-info.sh — Query host GPU and Docker GPU support
#
# Exit codes:
#   0  — Success
#   1  — Failure
#

set -eo pipefail

# Output helpers
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

# Check if command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

docker_image_exists() {
  docker image inspect "$1" >/dev/null 2>&1
}

last_nonempty_line() {
  local content="$1"
  local line
  local last_line=""

  while IFS= read -r line; do
    if [[ -n "${line//[[:space:]]/}" ]]; then
      last_line="$line"
    fi
  done <<< "$content"

  printf '%s\n' "$last_line"
}

inspect_target_image() {
  local target_image="$1"
  local inspection_output

  print_header "Target Image Inspection"

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
  if inspection_output=$(docker run --rm --gpus all --entrypoint /bin/bash "$target_image" -lc '
python3 --version
python3 - <<"PY"
import importlib

packages = [
    ("tensorrt", "TensorRT"),
    ("onnxruntime", "ONNX Runtime"),
    ("onnx", "ONNX"),
    ("numpy", "NumPy"),
    ("pycuda", "PyCUDA"),
]

for module_name, label in packages:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"{label}: {version}")
        if module_name == "onnxruntime":
            providers = ", ".join(module.get_available_providers())
            print(f"ONNX Runtime providers: {providers}")
    except Exception as exc:
        print(f"{label}: not available ({exc.__class__.__name__}: {exc})")
PY
printf "CUDA libraries:\n"
ldconfig -p 2>/dev/null | grep -E "libnvinfer|libcudart|libcublas|libcudnn|libnvonnxparser" || true
' 2>/dev/null); then
    print_success "Target image started successfully with GPU"
    print_multiline_info "$inspection_output"
  else
    print_error "Target image failed to start with GPU: $target_image"
    echo ""
    return 1
  fi

  echo ""
}

# Main check function
main() {
  local status=0

  echo "=========================================="
  echo " GPU Device and Docker Support Report"
  echo "=========================================="
  echo ""

  # ============ HOST ============
  print_header_major "HOST"
  echo ""

  # 1. System Information
  print_header "System Information"
  if [[ -f /etc/os-release ]]; then
    print_info "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
  fi
  print_info "Kernel: $(uname -r)"
  print_info "Architecture: $(uname -m)"
  echo ""

  # 2. NVIDIA Driver
  print_header "NVIDIA Driver"
  if command_exists nvidia-smi; then
    local driver_version
    driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    if [[ -n "$driver_version" ]]; then
      print_success "Driver Version: $driver_version"
    else
      print_warning "nvidia-smi found but driver version query failed"
    fi
  else
    print_error "nvidia-smi not found - NVIDIA drivers not installed"
    status=1
  fi
  echo ""

  # 3. GPU Devices
  print_header "GPU Devices"
  if command_exists nvidia-smi; then
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv 2>/dev/null | while IFS= read -r line; do
      print_info "$line"
    done
  else
    print_error "Cannot query GPU devices"
  fi
  echo ""

  # 4. CUDA (host)
  print_header "CUDA (host)"
  if command_exists nvcc; then
    local cuda_version
    cuda_version=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | cut -d',' -f1)
    if [[ -n "$cuda_version" ]]; then
      print_success "CUDA Version: $cuda_version"
    else
      print_warning "nvcc found but version query failed"
    fi
  else
    print_warning "nvcc not found - CUDA toolkit not installed"
  fi

  if [[ -d /usr/local/cuda ]]; then
    print_info "CUDA Path: /usr/local/cuda"
    if [[ -L /usr/local/cuda ]]; then
      print_info "CUDA Link: $(readlink -f /usr/local/cuda)"
    fi
  fi
  echo ""

  # 5. Docker (host)
  print_header "Docker (host)"
  if command_exists docker; then
    local docker_version
    docker_version=$(docker --version 2>/dev/null)
    print_success "Docker: $docker_version"
  else
    print_error "Docker not installed"
    status=1
  fi
  echo ""

  # 6. Docker GPU Support (host)
  print_header "Docker GPU Support (host)"
  if command_exists nvidia-container-cli; then
    print_success "nvidia-container-cli found"
  else
    print_warning "nvidia-container-cli not found"
  fi

  if command_exists nvidia-container-runtime; then
    print_success "nvidia-container-runtime found"
  else
    print_warning "nvidia-container-runtime not found"
  fi

  if [[ -f /etc/docker/daemon.json ]]; then
    if grep -q "nvidia" /etc/docker/daemon.json 2>/dev/null; then
      print_success "Docker configured for NVIDIA runtime"
    else
      print_warning "Docker daemon.json found but no NVIDIA configuration"
    fi
  else
    print_warning "Docker daemon.json not found"
  fi
  echo ""

  # ============ DOCKER ============
  print_header_major "DOCKER"
  echo ""

  # 7. Docker GPU Access Test (base image)
  print_header "Docker GPU Access Test (base image)"
  if command_exists docker && command_exists nvidia-smi; then
    echo "Testing Docker GPU access..."
    local test_image="${BASE_IMAGE:-nvidia/cuda:12.6.0-runtime-ubuntu22.04}"
    if docker run --rm --gpus all "$test_image" nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
      print_success "Docker can access GPUs with image: $test_image"
      local gpu_info_output=""
      local gpu_info=""
      if gpu_info_output=$(docker run --rm --gpus all "$test_image" nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null); then
        gpu_info=$(last_nonempty_line "$gpu_info_output")
      fi
      if [[ -n "$gpu_info" ]]; then
        print_info "Container sees: $gpu_info"
      fi
    else
      print_error "Docker cannot access GPUs with image: $test_image"
      print_info "Try: sudo apt install nvidia-container-toolkit && sudo systemctl restart docker"
      status=1
    fi
  else
    print_warning "Cannot test Docker GPU access (missing docker or nvidia-smi)"
    status=1
  fi
  echo ""

  # 8. Target Image Inspection
  local target_image="${TARGET_IMAGE:-}"
  if ! inspect_target_image "$target_image"; then
    status=1
  fi

  # 9. Summary
  print_header "Summary"
  echo "Required for TensorRT inference:"
  if command_exists nvidia-smi && command_exists docker; then
    local test_image="${BASE_IMAGE:-nvidia/cuda:12.6.0-runtime-ubuntu22.04}"
    if docker run --rm --gpus all "$test_image" nvidia-smi >/dev/null 2>&1; then
      print_success "All requirements met for TensorRT inference (tested with: $test_image)"
    else
      print_error "Docker GPU support missing for image: $test_image"
      status=1
    fi
  else
    print_error "Missing required components"
    status=1
  fi
  echo ""

  echo "=========================================="
  echo " Report complete"
  echo "=========================================="

  return "$status"
}

main
