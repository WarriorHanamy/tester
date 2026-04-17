#!/usr/bin/env python3
"""TensorRT engine smoke test using TensorRT 10.x APIs."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorrt as trt
import yaml

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart


DEFAULT_ENGINE_DIR = "/opt/engine"
DEFAULT_ENGINE_FILE = "vtol_hover.engine"
DEFAULT_MODEL_DIR = "/opt/models"
DEFAULT_MODEL_NAME = "vtol_hover"
DEFAULT_METADATA_FILE = "observations_metadata.yaml"


def cuda_call(result):
    err, *payload = result
    if err != cudart.cudaError_t.cudaSuccess:
        _, message = cudart.cudaGetErrorName(err)
        raise RuntimeError(f"CUDA error: {message}")
    if not payload:
        return None
    if len(payload) == 1:
        return payload[0]
    return tuple(payload)


@dataclass
class TensorBuffer:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    host: np.ndarray
    device: int
    mode: trt.TensorIOMode

    @property
    def nbytes(self) -> int:
        return self.host.nbytes


def load_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def metadata_dims(metadata: dict) -> dict[str, int]:
    return {
        item["name"]: int(item["dim"])
        for item in metadata.get("low_dim", [])
        if isinstance(item, dict) and "name" in item and "dim" in item
    }


def resolve_input_shape(
    tensor_name: str,
    declared_shape: tuple[int, ...],
    dims_by_name: dict[str, int],
) -> tuple[int, ...]:
    resolved = []
    inferred_dim = dims_by_name.get(tensor_name)

    for index, dim in enumerate(declared_shape):
        if dim >= 0:
            resolved.append(dim)
            continue
        if index == 0:
            resolved.append(1)
            continue
        if inferred_dim is not None and index == len(declared_shape) - 1:
            resolved.append(inferred_dim)
            continue
        resolved.append(1)

    return tuple(resolved)


def make_dummy_input(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        return np.zeros(shape, dtype=dtype)
    if np.issubdtype(dtype, np.bool_):
        return np.zeros(shape, dtype=dtype)
    return np.random.randn(*shape).astype(dtype)


class TensorRTInference:
    def __init__(self, engine_path: Path, metadata_path: Path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.metadata = load_metadata(metadata_path)
        self.dims_by_name = metadata_dims(self.metadata)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda_call(cudart.cudaStreamCreate())
        self.inputs: list[TensorBuffer] = []
        self.outputs: list[TensorBuffer] = []
        self._configure_input_shapes()
        self._allocate_buffers()

    def _load_engine(self, engine_path: Path) -> trt.ICudaEngine:
        with engine_path.open("rb") as handle:
            engine = self.runtime.deserialize_cuda_engine(handle.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        return engine

    def _configure_input_shapes(self) -> None:
        for index in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(tensor_name) != trt.TensorIOMode.INPUT:
                continue

            declared_shape = tuple(self.engine.get_tensor_shape(tensor_name))
            resolved_shape = resolve_input_shape(
                tensor_name,
                declared_shape,
                self.dims_by_name,
            )
            if any(dim < 0 for dim in declared_shape):
                self.context.set_input_shape(tensor_name, resolved_shape)

        missing_shapes = list(self.context.infer_shapes())
        if missing_shapes:
            raise RuntimeError(
                f"TensorRT still requires input shapes for: {missing_shapes}"
            )

    def _allocate_buffers(self) -> None:
        for index in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(index)
            tensor_shape = tuple(self.context.get_tensor_shape(tensor_name))
            if any(dim < 0 for dim in tensor_shape):
                raise RuntimeError(
                    f"Tensor shape not fully specified for {tensor_name}: {tensor_shape}"
                )

            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(tensor_name)))
            size = int(trt.volume(tensor_shape))
            host = np.empty(size, dtype=dtype)
            device = cuda_call(cudart.cudaMalloc(host.nbytes))

            buffer = TensorBuffer(
                name=tensor_name,
                shape=tensor_shape,
                dtype=dtype,
                host=host,
                device=int(device),
                mode=self.engine.get_tensor_mode(tensor_name),
            )
            self.context.set_tensor_address(tensor_name, buffer.device)

            if buffer.mode == trt.TensorIOMode.INPUT:
                self.inputs.append(buffer)
            else:
                self.outputs.append(buffer)

        print(
            f"Prepared {len(self.inputs)} input buffers and {len(self.outputs)} output buffers"
        )

    def create_dummy_inputs(self) -> dict[str, np.ndarray]:
        dummy_inputs = {}
        for buffer in self.inputs:
            dummy = make_dummy_input(buffer.shape, buffer.dtype)
            dummy_inputs[buffer.name] = dummy
            print(f"Created dummy input for {buffer.name}: shape {dummy.shape}")
        return dummy_inputs

    def run_inference(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for buffer in self.inputs:
            array = np.ascontiguousarray(
                inputs[buffer.name], dtype=buffer.dtype
            ).reshape(-1)
            if array.size != buffer.host.size:
                raise RuntimeError(
                    f"Input tensor size mismatch for {buffer.name}: expected {buffer.host.size}, got {array.size}"
                )
            np.copyto(buffer.host, array)
            cuda_call(
                cudart.cudaMemcpyAsync(
                    buffer.device,
                    buffer.host.ctypes.data,
                    buffer.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                )
            )

        if not self.context.execute_async_v3(stream_handle=self.stream):
            raise RuntimeError("TensorRT execute_async_v3() failed")

        outputs = {}
        for buffer in self.outputs:
            cuda_call(
                cudart.cudaMemcpyAsync(
                    buffer.host.ctypes.data,
                    buffer.device,
                    buffer.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                )
            )

        cuda_call(cudart.cudaStreamSynchronize(self.stream))

        for buffer in self.outputs:
            outputs[buffer.name] = buffer.host.reshape(buffer.shape).copy()
        return outputs

    def close(self) -> None:
        for buffer in [*self.inputs, *self.outputs]:
            cuda_call(cudart.cudaFree(buffer.device))
        if self.stream is not None:
            cuda_call(cudart.cudaStreamDestroy(self.stream))
            self.stream = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TensorRT engine smoke test")
    parser.add_argument(
        "--engine-dir",
        default=None,
        help=f"Directory containing engine file (default: $ENGINE_DIR or {DEFAULT_ENGINE_DIR})",
    )
    parser.add_argument(
        "--engine-file",
        default=None,
        help=f"Engine filename (default: $ENGINE_FILE or {DEFAULT_ENGINE_FILE})",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help=f"Model metadata root (default: $MODEL_DIR or {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=f"Model name (default: $MODEL_NAME or {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--metadata-file",
        default=None,
        help=f"Metadata filename (default: $METADATA_FILE or {DEFAULT_METADATA_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine_dir = Path(args.engine_dir or os.getenv("ENGINE_DIR", DEFAULT_ENGINE_DIR))
    model_dir = Path(args.model_dir or os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR))
    model_name = args.model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    engine_file = args.engine_file or os.getenv("ENGINE_FILE", f"{model_name}.engine")
    metadata_file = args.metadata_file or os.getenv(
        "METADATA_FILE", DEFAULT_METADATA_FILE
    )

    engine_path = engine_dir / engine_file
    metadata_path = model_dir / model_name / metadata_file

    print("=" * 60)
    print("TensorRT Engine Smoke Test")
    print("=" * 60)
    print(f"Engine file: {engine_path}")
    print(f"Metadata file: {metadata_path}")

    if not engine_path.exists():
        print(f"ERROR: engine file not found at {engine_path}")
        sys.exit(1)

    inference = None
    try:
        print("\n1. Loading TensorRT engine...")
        inference = TensorRTInference(engine_path, metadata_path)

        print("\n2. Creating dummy input data...")
        dummy_inputs = inference.create_dummy_inputs()

        print("\n3. Running inference...")
        outputs = inference.run_inference(dummy_inputs)

        print("\n4. Processing outputs...")
        for index, (name, output) in enumerate(outputs.items()):
            print(
                f"Output {index} ({name}): shape {output.shape}, values (first 5): {output.flatten()[:5]}"
            )

        print("\n" + "=" * 60)
        print("SMOKE TEST PASSED - TensorRT engine inference successful")
        print("=" * 60)
    except Exception as exc:
        print(f"\nSMOKE TEST FAILED: {exc}")
        raise
    finally:
        if inference is not None:
            inference.close()


if __name__ == "__main__":
    main()
