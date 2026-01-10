"""
Linear Layer: CPU vs GPU Benchmark
====================================

Compares performance of linear layer operations on CPU vs GPU.

Note: GPU will fail due to missing matmul kernel on Apple Silicon.

Run:
  pixi run benchmark-linear
"""

import time
import numpy as np
from max.driver import Accelerator, CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def build_linear_layer_graph(device_type: str, batch_size: int, 
                             input_features: int, output_features: int) -> Graph:
    """Build graph: y = relu(W @ x + b)"""
    device = DeviceRef(device_type)
    input_spec = TensorType(DType.float32, shape=[batch_size, input_features], device=device)
    
    with Graph(f"linear_layer_{device_type}", input_types=[input_spec]) as graph:
        x = graph.inputs[0].tensor
        
        # Random weights
        W = ops.constant(
            np.random.randn(output_features, input_features).astype(np.float32),
            dtype=DType.float32,
            device=x.device
        )
        
        b = ops.constant(
            np.random.randn(output_features).astype(np.float32),
            dtype=DType.float32,
            device=x.device
        )
        
        # Computation
        y = ops.matmul(x, ops.transpose(W, 0, 1))
        y = ops.add(y, b)
        y = ops.relu(y)
        
        graph.output(y)
    
    return graph


def benchmark_device(device_type: str, batch_size: int, input_features: int, 
                    output_features: int, iterations: int = 1000, warmup: int = 100):
    """Benchmark linear layer on specified device."""
    
    print(f"\n{'='*60}")
    print(f"Benchmarking {device_type.upper()}")
    print(f"{'='*60}")
    
    # Initialize device
    try:
        if device_type == "gpu":
            device = Accelerator()
        else:
            device = CPU()
        print(f"✓ Device initialized: {device}")
    except Exception as e:
        print(f"✗ Failed to initialize {device_type}: {e}")
        return None
    
    # Build and compile graph
    try:
        graph = build_linear_layer_graph(device_type, batch_size, input_features, output_features)
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        print(f"✓ Graph compiled")
    except Exception as e:
        print(f"✗ Failed to compile graph: {e}")
        if device_type == "gpu" and "matmul" in str(e).lower():
            print(f"  Note: matmul kernel not available on Apple Silicon GPU")
        return None
    
    # Prepare input
    input_data_np = np.random.randn(batch_size, input_features).astype(np.float32)
    input_data = Tensor.from_numpy(input_data_np).to(device)
    
    # Warmup
    print(f"\nWarming up ({warmup} iterations)...")
    for _ in range(warmup):
        model.execute(input_data)
    
    # Benchmark
    print(f"Running benchmark ({iterations} iterations)...")
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        output = model.execute(input_data)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    # Calculate statistics
    times = np.array(times)
    mean_time = np.mean(times)
    median_time = np.median(times)
    p95_time = np.percentile(times, 95)
    p99_time = np.percentile(times, 99)
    std_time = np.std(times)
    throughput = 1000 / mean_time
    
    results = {
        "device": device_type,
        "mean_ms": mean_time,
        "median_ms": median_time,
        "p95_ms": p95_time,
        "p99_ms": p99_time,
        "std_ms": std_time,
        "throughput": throughput,
        "iterations": iterations
    }
    
    # Verify correctness
    output_np = output[0].to_numpy()
    W = graph  # Would need to extract weights for verification
    
    print(f"\nResults:")
    print(f"  Mean:       {mean_time:.4f} ms")
    print(f"  Median:     {median_time:.4f} ms")
    print(f"  P95:        {p95_time:.4f} ms")
    print(f"  P99:        {p99_time:.4f} ms")
    print(f"  Std Dev:    {std_time:.4f} ms")
    print(f"  Throughput: {throughput:.2f} req/sec")
    
    return results


def main():
    print("="*60)
    print("Linear Layer: CPU vs GPU Benchmark")
    print("="*60)
    
    # Configuration
    batch_size = 32
    input_features = 128
    output_features = 64
    iterations = 1000
    warmup = 100
    
    print(f"\nConfiguration:")
    print(f"  Layer: {input_features} → {output_features}")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {iterations} (warmup: {warmup})")
    
    # Benchmark CPU
    cpu_results = benchmark_device("cpu", batch_size, input_features, output_features, 
                                  iterations, warmup)
    
    # Benchmark GPU
    gpu_results = benchmark_device("gpu", batch_size, input_features, output_features, 
                                  iterations, warmup)
    
    # Compare results
    if cpu_results and gpu_results:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        
        speedup = cpu_results["mean_ms"] / gpu_results["mean_ms"]
        throughput_ratio = gpu_results["throughput"] / cpu_results["throughput"]
        
        print(f"\nCPU:  {cpu_results['mean_ms']:.4f} ms mean")
        print(f"GPU:  {gpu_results['mean_ms']:.4f} ms mean")
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"GPU is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than CPU")
        print(f"Throughput improvement: {throughput_ratio:.2f}x")
        
        # Latency consistency
        cpu_cv = cpu_results["std_ms"] / cpu_results["mean_ms"]
        gpu_cv = gpu_results["std_ms"] / gpu_results["mean_ms"]
        print(f"\nConsistency (lower is better):")
        print(f"  CPU CV: {cpu_cv:.4f}")
        print(f"  GPU CV: {gpu_cv:.4f}")
    elif cpu_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"\nCPU benchmark completed successfully")
        print(f"GPU benchmark failed (expected - matmul kernel not available)")


if __name__ == "__main__":
    main()
