#!/usr/bin/env bash
#
# check-image-runtime.sh — Runtime image contract presence check

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
  local app_dir="${APP_DIR:-/opt/app}"
  local engine_dir="${ENGINE_DIR:-/opt/engine}"
  local engine_file="${ENGINE_FILE:-vtol_hover.engine}"
  local model_dir="${MODEL_DIR:-/opt/models}"
  local model_name="${MODEL_NAME:-vtol_hover}"
  local metadata_file="${METADATA_FILE:-observations_metadata.yaml}"

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
    print_info "Build it first with: make build-runtime"
    echo ""
    return 0
  fi

  print_success "Found target image: $target_image"
  print_info "Starting container with GPU access to verify runtime environment..."
  echo ""

  if inspection_output=$(docker run --rm --gpus all --entrypoint /bin/bash "$target_image" -lc "
    set -e
    echo '=== Python version ==='
    python3 --version

    echo
    echo '=== Required Python packages ==='
    python3 - <<'PY'
import importlib

packages = [
    ('tensorrt', 'TensorRT'),
    ('numpy', 'NumPy'),
    ('yaml', 'PyYAML'),
]

for module_name, label in packages:
    module = importlib.import_module(module_name)
    version = getattr(module, '__version__', 'unknown')
    print(f'{label}: {version}')

try:
    from cuda.bindings import runtime as cudart  # noqa: F401
    print('CUDA Python bindings: cuda.bindings.runtime')
except ImportError:
    from cuda import cudart  # noqa: F401
    print('CUDA Python bindings: cuda.cudart')
PY

    echo
    echo '=== CUDA libraries (ldconfig) ==='
    ldconfig -p 2>/dev/null | grep -E 'libnvinfer|libcudart|libcublas|libcudnn' || true

    echo
    echo '=== App entry script presence ==='
    if [ -f '${app_dir}/run_inference.py' ]; then
      echo 'Found: ${app_dir}/run_inference.py'
    else
      echo 'Missing: ${app_dir}/run_inference.py'
      exit 1
    fi

    echo
    echo '=== Engine artifact presence ==='
    if [ -f '${engine_dir}/${engine_file}' ]; then
      echo 'Found engine: ${engine_dir}/${engine_file}'
      ls -l '${engine_dir}/${engine_file}'
    else
      echo 'Missing engine: ${engine_dir}/${engine_file}'
      exit 1
    fi

    echo
    echo '=== Metadata presence ==='
    if [ -f '${model_dir}/${model_name}/${metadata_file}' ]; then
      echo 'Found metadata: ${model_dir}/${model_name}/${metadata_file}'
    else
      echo 'Metadata not found: ${model_dir}/${model_name}/${metadata_file}'
    fi
  " 2>/dev/null); then
    print_success "Target image started successfully with GPU"
    print_multiline_info "$inspection_output"
  else
    print_error "Target image failed to satisfy runtime contract: $target_image"
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
