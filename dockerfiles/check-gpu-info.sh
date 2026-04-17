#!/usr/bin/env bash
#
# check-gpu-info.sh — Query host GPU and Docker GPU support
#                        Plus base image toolkit verification
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

# Inspect base image toolkit versions
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

  # Compatibility check
  print_header "Compatibility Assessment"
  local cuda_version=""
  local cudnn_version=""
  local python_version=""
  local pytorch_version=""
  local gcc_version=""
  local driver_version=""

  # Extract versions from inspection output
  cuda_version=$(echo "$inspection_output" | awk '/=== CUDA Toolkit ===/,/===/ {if ($0 ~ /release/ && !/===/) {print; exit}}')
  python_version=$(echo "$inspection_output" | awk '/=== Python ===/,/===/ {if ($0 ~ /Python/ && !/===/) {print; exit}}')
  gcc_version=$(echo "$inspection_output" | awk '/=== GCC ===/,/===/ {if ($0 ~ /gcc/ && !/===/) {print; exit}}')
  pytorch_version=$(echo "$inspection_output" | awk '/=== PyTorch ===/,/===/ {if ($0 ~ /PyTorch:/ && !/===/) {print; exit}}')
  driver_version=$(echo "$inspection_output" | awk '/=== NVIDIA Driver/,/===/ {if ($0 ~ /[0-9]+\.[0-9]+/ && !/===/) {print; exit}}')

  # Try to extract cuDNN version from headers
  cudnn_version=$(echo "$inspection_output" | awk '
    /=== cuDNN Version ===/,/===/ {
      if (/CUDNN_MAJOR/) { gsub(/.*CUDNN_MAJOR|[[:space:]]+/, ""); major=$0 }
      if (/CUDNN_MINOR/) { gsub(/.*CUDNN_MINOR|[[:space:]]+/, ""); minor=$0 }
      if (/CUDNN_PATCHLEVEL/) { gsub(/.*CUDNN_PATCHLEVEL|[[:space:]]+/, ""); patch=$0 }
    }
    END {
      if (major != "") print "cuDNN: " major "." minor "." patch
      else print "cuDNN version not detectable from headers"
    }
  ')

  # Print extracted info
  echo "Extracted versions:"
  [[ -n "$python_version" ]] && echo "  Python: $python_version" || echo "  Python: not detected"
  [[ -n "$gcc_version" ]] && echo "  GCC: $gcc_version" || echo "  GCC: not detected"
  [[ -n "$cuda_version" ]] && echo "  CUDA: $cuda_version" || echo "  CUDA: not detected"
  [[ -n "$cudnn_version" ]] && echo "  cuDNN: $cudnn_version" || echo "  cuDNN: not detected"
  [[ -n "$pytorch_version" ]] && echo "  PyTorch: $pytorch_version" || echo "  PyTorch: not detected"
  [[ -n "$driver_version" ]] && echo "  Driver: $driver_version" || echo "  Driver: not detected (expected for runtime-only images)"
  echo ""

  # Warnings
  if [[ -z "$cuda_version" ]]; then
    print_warning "CUDA toolkit not found in base image - this may be a CPU-only image"
  fi

  if [[ -z "$cudnn_version" ]] || [[ "$cudnn_version" == *"not detectable"* ]]; then
    print_warning "cuDNN version not detectable - may not be installed or headers missing"
  fi

  if [[ -z "$pytorch_version" ]]; then
    print_info "Note: PyTorch not required for TensorRT-only inference (ONNX → TensorRT)"
  fi

  echo ""
  return "$status"
}

# Main check function
main() {
  local status=0

  echo "=========================================="
  echo " GPU Device and Docker Support Report"
  echo "=========================================="
  echo ""

  # ============ BASE IMAGE CHECK ============
  print_header_major "BASE IMAGE CHECK"
  echo ""
  local base_image="${BASE_IMAGE:-${DEFAULT_BASE_IMAGE:-nvidia/cuda:12.6.0-runtime-ubuntu22.04}}"
  if ! inspect_base_image "$base_image"; then
    status=1
  fi

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
