import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import tensorrt_rtx as trt
from cuda.bindings import runtime as cudart


def cuda_assert(call: tuple) -> object:
    res = None
    err = call[0]
    if len(call) > 1:
        res = call[1]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    return res


# Map TensorRT DataTypes to NumPy dtypes
def trt_dtype_to_np(trt_dtype: trt.DataType) -> np.dtype:
    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT32: np.int32,
        trt.DataType.INT8: np.int8,
        trt.DataType.UINT8: np.uint8,
        trt.DataType.BOOL: np.bool_,
    }
    if hasattr(trt.DataType, "INT64"):
        mapping[trt.DataType.INT64] = np.int64
    if hasattr(trt.DataType, "BF16"):
        # Fallback to uint16 container on host if numpy lacks native bfloat16
        mapping[trt.DataType.BF16] = np.uint16
    if trt_dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT DataType: {trt_dtype}")
    return np.dtype(mapping[trt_dtype])


@dataclass
class TensorBinding:
    name: str
    mode: trt.TensorIOMode
    shape: tuple
    dtype: np.dtype
    host_buffer: np.ndarray
    device_ptr: int
    nbytes: int


def create_network_from_onnx(
    builder: trt.Builder, onnx_file_path: str, logger: trt.Logger
) -> trt.INetworkDefinition:
    print(f"[TRT-RTX] Parsing ONNX file: {onnx_file_path}")

    # TensorRT-RTX requires STRONGLY_TYPED networks
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)

    parser = trt.OnnxParser(network, logger)
    if not parser:
        raise RuntimeError("Failed to create ONNX parser!")

    # parse_from_file handles external data files (>2GB models) automatically
    if not parser.parse_from_file(onnx_file_path):
        for i in range(parser.num_errors):
            print(f"Parser Error [{i}]: {parser.get_error(i)}")
        raise RuntimeError("Failed to parse ONNX file!")

    print(f"[TRT-RTX] Network Inputs ({network.num_inputs}):")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  - {inp.name:18s} | shape={inp.shape} | dtype={inp.dtype}")

    print(f"[TRT-RTX] Network Outputs ({network.num_outputs}):")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  - {out.name:18s} | shape={out.shape} | dtype={out.dtype}")

    return network


def build_or_load_engine(
    onnx_path: str,
    engine_path: str,
    force_rebuild: bool,
    logger: trt.Logger,
) -> bytes:
    # 1. Load existing engine if available and not forcing rebuild
    if engine_path and os.path.exists(engine_path) and not force_rebuild and not onnx_path:
        print(f"[TRT-RTX] Loading pre-built serialized engine from: {engine_path}")
        with open(engine_path, "rb") as f:
            return f.read()

    if not onnx_path:
        raise ValueError("Must provide --onnx to build an engine or --load-engine to load an existing one.")

    # 2. Build engine from ONNX
    builder = trt.Builder(logger)
    if not builder:
        raise RuntimeError("Failed to create builder!")

    builder_config = builder.create_builder_config()
    if not builder_config:
        raise RuntimeError("Failed to create builder configuration!")

    network = create_network_from_onnx(builder, onnx_path, logger)

    # If the ONNX model was exported with --dynamic-batch, register an Optimization Profile
    has_dynamic_shapes = any(-1 in network.get_input(i).shape for i in range(network.num_inputs))
    if has_dynamic_shapes:
        print("[TRT-RTX] Dynamic dimensions detected; configuring Optimization Profile (batch=1)...")
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            static_shape = tuple(1 if d == -1 else d for d in inp.shape)
            profile.set_shape(inp.name, min=static_shape, opt=static_shape, max=static_shape)
        builder_config.add_optimization_profile(profile)

    print("[TRT-RTX] Building AOT serialized engine (this may take a moment)...")
    t0 = time.perf_counter()
    serialized_engine = builder.build_serialized_network(network, builder_config)
    if not serialized_engine:
        raise RuntimeError("Failed to build serialized engine!")
    build_time = time.perf_counter() - t0

    # Convert IHostMemory to bytes so we can both save and deserialize cleanly
    engine_bytes = bytes(memoryview(serialized_engine))
    print(f"[TRT-RTX] Built engine in {build_time:.2f}s ({len(engine_bytes) / (1024**2):.2f} MB).")

    # 3. Save to disk if requested
    if engine_path:
        os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        print(f"[TRT-RTX] Saved serialized engine to: {engine_path}")

    return engine_bytes


def populate_dummy_vla_input(name: str, shape: tuple, dtype: np.dtype) -> np.ndarray:
    """Generates realistic dummy inputs matching VLA input conventions."""
    if "image" in name.lower():
        if dtype == np.uint8:
            return np.random.randint(0, 256, size=shape, dtype=np.uint8)
        return np.random.uniform(0.0, 255.0, size=shape).astype(dtype)
    elif "token" in name.lower() or "txt" in name.lower():
        # Valid token IDs (e.g., 1 to 1000)
        return np.random.randint(1, 1000, size=shape, dtype=dtype)
    else:
        # robot_state or initial_noise
        if dtype == np.uint16:  # BF16 host fallback
            return np.zeros(shape, dtype=np.uint16)
        return np.random.randn(*shape).astype(dtype)


def benchmark_inference(
    engine_bytes: bytes,
    warmup_runs: int,
    num_runs: int,
    logger: trt.Logger,
) -> None:
    runtime = trt.Runtime(logger)
    if not runtime:
        raise RuntimeError("Failed to create runtime!")

    inference_engine = runtime.deserialize_cuda_engine(engine_bytes)
    if not inference_engine:
        raise RuntimeError("Failed to deserialize engine!")

    runtime_config = inference_engine.create_runtime_config()
    if not runtime_config:
        raise RuntimeError("Failed to create runtime config!")

    print("[TRT-RTX] Creating execution context (running JIT compilation for target GPU)...")
    t_jit = time.perf_counter()
    context = inference_engine.create_execution_context(runtime_config)
    if not context:
        raise RuntimeError("Failed to create execution context!")
    print(f"[TRT-RTX] JIT compilation finished in {time.perf_counter() - t_jit:.2f}s.")

    stream = cuda_assert(cudart.cudaStreamCreate())
    bindings: Dict[str, TensorBinding] = {}
    inputs: List[TensorBinding] = []
    outputs: List[TensorBinding] = []

    try:
        # Inspect all I/O tensors in the VLA engine and allocate memory
        for i in range(inference_engine.num_io_tensors):
            name = inference_engine.get_tensor_name(i)
            mode = inference_engine.get_tensor_mode(name)
            shape = tuple(context.get_tensor_shape(name))

            # Resolve dynamic dimensions if any remain (-1 -> 1)
            if -1 in shape:
                shape = tuple(1 if d == -1 else d for d in shape)
                if mode == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            dtype = trt_dtype_to_np(inference_engine.get_tensor_dtype(name))

            if mode == trt.TensorIOMode.INPUT:
                host_buf = np.ascontiguousarray(populate_dummy_vla_input(name, shape, dtype))
            else:
                host_buf = np.zeros(shape, dtype=dtype)

            nbytes = host_buf.nbytes
            dev_ptr = cuda_assert(cudart.cudaMallocAsync(nbytes, stream))
            context.set_tensor_address(name, dev_ptr)

            binding = TensorBinding(name, mode, shape, dtype, host_buf, dev_ptr, nbytes)
            bindings[name] = binding
            if mode == trt.TensorIOMode.INPUT:
                inputs.append(binding)
            else:
                outputs.append(binding)

        # Copy initial inputs to device
        for inp in inputs:
            cuda_assert(
                cudart.cudaMemcpyAsync(
                    inp.device_ptr,
                    inp.host_buffer.ctypes.data,
                    inp.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                )
            )
        cuda_assert(cudart.cudaStreamSynchronize(stream))

        # 1. Warmup Phase
        print(f"[TRT-RTX] Running {warmup_runs} warmup iterations...")
        for _ in range(warmup_runs):
            if not context.execute_async_v3(stream_handle=stream):
                raise RuntimeError("Failed to execute inference during warmup!")
        cuda_assert(cudart.cudaStreamSynchronize(stream))

        # 2. Benchmark Phase: End-to-End (H2D + Compute + D2H)
        print(f"[TRT-RTX] Benchmarking {num_runs} iterations (End-to-End H2D + GPU + D2H)...")
        latencies_e2e_ms = []
        for _ in range(num_runs):
            t_start = time.perf_counter()

            for inp in inputs:
                cuda_assert(
                    cudart.cudaMemcpyAsync(
                        inp.device_ptr,
                        inp.host_buffer.ctypes.data,
                        inp.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        stream,
                    )
                )

            if not context.execute_async_v3(stream_handle=stream):
                raise RuntimeError("Failed to execute inference!")

            for out in outputs:
                cuda_assert(
                    cudart.cudaMemcpyAsync(
                        out.host_buffer.ctypes.data,
                        out.device_ptr,
                        out.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        stream,
                    )
                )

            cuda_assert(cudart.cudaStreamSynchronize(stream))
            latencies_e2e_ms.append((time.perf_counter() - t_start) * 1000.0)

        # 3. Benchmark Phase: Pure GPU Compute (Pre-staged inputs)
        t_gpu_start = time.perf_counter()
        for _ in range(num_runs):
            if not context.execute_async_v3(stream_handle=stream):
                raise RuntimeError("Failed to execute inference!")
        cuda_assert(cudart.cudaStreamSynchronize(stream))
        avg_gpu_ms = ((time.perf_counter() - t_gpu_start) * 1000.0) / num_runs

        # Print Summary & Sample Output
        latencies_np = np.array(latencies_e2e_ms)
        print("\n" + "=" * 55)
        print("VLA TensorRT-RTX Inference Benchmark Results")
        print("=" * 55)
        print(f"Iterations             : {num_runs} (after {warmup_runs} warmup)")
        print(f"Pure GPU Compute Mean  : {avg_gpu_ms:.2f} ms ({1000.0 / avg_gpu_ms:.2f} Hz)")
        print(f"End-to-End Mean        : {latencies_np.mean():.2f} ms ({1000.0 / latencies_np.mean():.2f} Hz)")
        print(f"End-to-End Median (p50): {np.percentile(latencies_np, 50):.2f} ms")
        print(f"End-to-End p95 / p99   : {np.percentile(latencies_np, 95):.2f} ms / {np.percentile(latencies_np, 99):.2f} ms")
        print("-" * 55)

        for out in outputs:
            sample_vals = out.host_buffer.flatten()[:6]
            print(f"Output '{out.name}' shape={out.shape}: first 6 values = {sample_vals}")
        print("=" * 55)

    finally:
        for b in bindings.values():
            cuda_assert(cudart.cudaFreeAsync(b.device_ptr, stream))
        cuda_assert(cudart.cudaStreamSynchronize(stream))
        cuda_assert(cudart.cudaStreamDestroy(stream))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build, Save, and Benchmark VLA with TensorRT-RTX")
    parser.add_argument("--onnx", type=str, default=None, help="Path to exported vla_model.onnx")
    parser.add_argument("--engine", type=str, default="vla_model.engine", help="Path to save/load .engine file")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuilding engine even if file exists")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs (default: 10)")
    parser.add_argument("--runs", type=int, default=100, help="Number of timed benchmark runs (default: 100)")
    parser.add_argument("--verbose", action="store_true", help="Enable VERBOSE TensorRT logger")
    args = parser.parse_args()

    log_level = trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO
    trt_logger = trt.Logger(log_level)

    engine_data = build_or_load_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        force_rebuild=args.force_rebuild,
        logger=trt_logger,
    )

    benchmark_inference(
        engine_bytes=engine_data,
        warmup_runs=args.warmup,
        num_runs=args.runs,
        logger=trt_logger,
    )