#!/usr/bin/env python3
"""
TensorRT Inference Smoke Test
Tests ONNX model loading and inference using TensorRT
"""

import os
import sys
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import yaml

# Configuration from environment variables
MODEL_DIR = os.getenv("MODEL_DIR", "/opt/models")
MODEL_NAME = os.getenv("MODEL_NAME", "vtol_hover")
ONNX_FILE = os.getenv("ONNX_FILE", "model.onnx")
METADATA_FILE = os.getenv("METADATA_FILE", "observations_metadata.yaml")


class TensorRTInference:
    def __init__(self, onnx_path, metadata_path=None):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        # Load metadata if available
        self.metadata = None
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = yaml.safe_load(f)

        # Build engine from ONNX
        self.build_engine(onnx_path)

    def build_engine(self, onnx_path):
        """Build TensorRT engine from ONNX file"""
        builder = trt.Builder(self.logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX file
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(f"ONNX Parse Error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX file")

        # Build engine
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)

        # Set workspace size
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        # Build serialized engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Deserialize engine
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()

        print(f"TensorRT engine built successfully from {onnx_path}")

    def prepare_buffers(self):
        """Allocate host and device buffers"""
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            shape = self.engine.get_binding_shape(binding_idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))

            # Calculate size
            size = trt.volume(shape)

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding_idx):
                self.inputs.append(
                    {
                        "host": host_mem,
                        "device": device_mem,
                        "shape": shape,
                        "name": binding,
                    }
                )
            else:
                self.outputs.append(
                    {
                        "host": host_mem,
                        "device": device_mem,
                        "shape": shape,
                        "name": binding,
                    }
                )

        print(
            f"Prepared {len(self.inputs)} input buffers and {len(self.outputs)} output buffers"
        )

    def run_inference(self, input_data):
        """Run inference with given input data"""
        # Copy input data to device
        for i, inp in enumerate(self.inputs):
            np.copyto(inp["host"], input_data[i].ravel())
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)

        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )

        # Copy outputs back to host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

        # Synchronize
        self.stream.synchronize()

        # Return outputs
        return [out["host"].copy().reshape(out["shape"]) for out in self.outputs]


def create_dummy_input(metadata=None):
    """Create dummy input based on metadata or default values"""
    if metadata and "low_dim" in metadata:
        inputs = []
        for item in metadata["low_dim"]:
            name = item["name"]
            dim = item["dim"]
            # Create random input
            dummy = np.random.randn(*([1, dim])).astype(np.float32)
            inputs.append(dummy)
            print(f"Created dummy input for {name}: shape {dummy.shape}")
        return inputs
    else:
        # Default input shape (14 dimensions total based on metadata)
        dummy_input = np.random.randn(1, 14).astype(np.float32)
        print(f"Created default dummy input: shape {dummy_input.shape}")
        return [dummy_input]


def main():
    print("=" * 60)
    print("TensorRT Inference Smoke Test")
    print("=" * 60)

    # Set paths
    model_dir = os.path.join(MODEL_DIR, MODEL_NAME)
    onnx_path = os.path.join(model_dir, ONNX_FILE)
    metadata_path = os.path.join(model_dir, METADATA_FILE)

    print(f"Model directory: {model_dir}")
    print(f"ONNX file: {onnx_path}")
    print(f"Metadata file: {metadata_path}")

    # Check if files exist
    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX file not found at {onnx_path}")
        sys.exit(1)

    try:
        # Initialize TensorRT inference
        print("\n1. Loading ONNX model and building TensorRT engine...")
        trt_inference = TensorRTInference(
            onnx_path, metadata_path if os.path.exists(metadata_path) else None
        )

        print("\n2. Preparing buffers...")
        trt_inference.prepare_buffers()

        print("\n3. Creating dummy input data...")
        dummy_inputs = create_dummy_input(trt_inference.metadata)

        print("\n4. Running inference...")
        outputs = trt_inference.run_inference(dummy_inputs)

        print("\n5. Processing outputs...")
        for i, output in enumerate(outputs):
            print(
                f"Output {i}: shape {output.shape}, values (first 5): {output.flatten()[:5]}"
            )

        print("\n" + "=" * 60)
        print("✓ SMOKE TEST PASSED - TensorRT inference successful!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ SMOKE TEST FAILED: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
