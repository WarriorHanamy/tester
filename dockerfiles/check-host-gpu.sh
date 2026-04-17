#!/usr/bin/env bash
#
# check-host-gpu.sh — Host + Docker GPU compute readiness
#
# Exit codes:
#   0  — Success (host can provide GPU compute to containers)
#   1  — Failure
#
# GPU selector: this script performs a *readiness* check using --gpus all.
# Use GPU_DEVICE at runtime for selective device exposure.

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

command_exists() {
  command -v "$1" >/dev/null 2>&1
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

main() {
  local status=0

  echo "=========================================="
  echo " Host GPU Compute Readiness Report"
  echo "=========================================="
  echo ""

  # ============ SYSTEM ============
  print_header_major "SYSTEM"
  echo ""
  print_header "System Information"
  if [[ -f /etc/os-release ]]; then
    print_info "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
  fi
  print_info "Kernel: $(uname -r)"
  print_info "Architecture: $(uname -m)"
  echo ""

  # ============ NVIDIA DRIVER ============
  print_header_major "NVIDIA DRIVER"
  echo ""
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

  # ============ GPU DEVICES ============
  print_header_major "GPU DEVICES"
  echo ""
  if command_exists nvidia-smi; then
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv 2>/dev/null | while IFS= read -r line; do
      print_info "$line"
    done
  else
    print_error "Cannot query GPU devices"
  fi
  echo ""

  # ============ HOST CUDA TOOLKIT (optional) ============
  print_header_major "HOST CUDA TOOLKIT (optional for building)"
  echo ""
  print_header "CUDA"
  if command_exists nvcc; then
    local cuda_version
    cuda_version=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | cut -d',' -f1)
    if [[ -n "$cuda_version" ]]; then
      print_success "CUDA Version: $cuda_version"
    else
      print_warning "nvcc found but version query failed"
    fi
  else
    print_warning "nvcc not found - CUDA toolkit not installed (not required for runtime-only hosts)"
  fi
  if [[ -d /usr/local/cuda ]]; then
    print_info "CUDA Path: /usr/local/cuda"
    if [[ -L /usr/local/cuda ]]; then
      print_info "CUDA Link: $(readlink -f /usr/local/cuda)"
    fi
  fi
  echo ""

  # ============ DOCKER ============
  print_header_major "DOCKER"
  echo ""
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

  # ============ DOCKER GPU ACCESS TEST ============
  print_header_major "DOCKER GPU ACCESS TEST"
  echo ""
  print_info "Testing that Docker can run a container with GPU access..."
  local test_image="${BASE_IMAGE:-nvidia/cuda:12.6.0-runtime-ubuntu22.04}"
  if command_exists docker && command_exists nvidia-smi; then
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

  print_header "Summary"
  if command_exists nvidia-smi && command_exists docker; then
    if docker run --rm --gpus all "$test_image" nvidia-smi >/dev/null 2>&1; then
      print_success "Host GPU compute readiness: OK"
    else
      print_error "Host GPU compute readiness: FAILED"
      status=1
    fi
  else
    print_error "Missing required components (nvidia-smi or docker)"
    status=1
  fi
  echo ""

  echo "=========================================="
  echo " Report complete"
  echo "=========================================="

  return "$status"
}

main
