"""
Element-wise Operations Benchmark (CPU vs GPU)
===============================================

Compares performance of element-wise operations on CPU vs GPU.

Run:
  pixi run python examples/python/01_elementwise/benchmark_elementwise.py
"""

import time
import numpy as np
from max.driver import Accelerator, CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def build_elementwise_graph(device_type: str) -> Graph:
    """Build graph: y = relu(x * 2 + 1)"""
    device = DeviceRef(device_type)
    input_spec = TensorType(DType.float32, shape=[4], device=device)
    
    with Graph(f"elementwise_{device_type}", input_types=[input_spec]) as graph:
        x = graph.inputs[0].tensor
        
        multiplier = ops.constant(
            np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
            dtype=DType.float32,
            device=x.device
        )
        
        offset = ops.constant(
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=DType.float32,
            device=x.device
        )
        
        y = ops.mul(x, multiplier)
        y = ops.add(y, offset)
        y = ops.relu(y)
        
        graph.output(y)
    
    return graph


def benchmark_device(device_type: str, iterations: int = 1000, warmup: int = 100):
    """Benchmark element-wise operations on specified device."""
    
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
        graph = build_elementwise_graph(device_type)
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        print(f"✓ Graph compiled")
    except Exception as e:
        print(f"✗ Failed to compile graph: {e}")
        return None
    
    # Prepare input
    input_data_np = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
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
        times.append((end - start) * 1000)  # Convert to milliseconds
    
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
    expected = np.maximum(0, input_data_np * 2.0 + 1.0)
    correct = np.allclose(output_np, expected)
    
    print(f"\nResults:")
    print(f"  Mean:       {mean_time:.4f} ms")
    print(f"  Median:     {median_time:.4f} ms")
    print(f"  P95:        {p95_time:.4f} ms")
    print(f"  P99:        {p99_time:.4f} ms")
    print(f"  Std Dev:    {std_time:.4f} ms")
    print(f"  Throughput: {throughput:.2f} req/sec")
    print(f"  Correctness: {'✓ PASS' if correct else '✗ FAIL'}")
    
    return results


def main():
    print("="*60)
    print("Element-wise Operations: CPU vs GPU Benchmark")
    print("="*60)
    
    iterations = 1000
    warmup = 100
    
    # Benchmark CPU
    cpu_results = benchmark_device("cpu", iterations=iterations, warmup=warmup)
    
    # Benchmark GPU
    gpu_results = benchmark_device("gpu", iterations=iterations, warmup=warmup)
    
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
        
        if cpu_cv < gpu_cv:
            consistency_improvement = (gpu_cv / cpu_cv - 1) * 100
            print(f"  CPU is {consistency_improvement:.1f}% more consistent")
        else:
            consistency_improvement = (cpu_cv / gpu_cv - 1) * 100
            print(f"  GPU is {consistency_improvement:.1f}% more consistent")


if __name__ == "__main__":
    main()
