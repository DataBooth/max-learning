"""
Element-wise Operations: Size Scaling Benchmark
===============================================

Tests CPU vs GPU performance across different tensor sizes to find
where GPU acceleration becomes beneficial.

Run:
  pixi run python examples/python/01_elementwise/benchmark_sizes.py
"""

import time
import numpy as np
from max.driver import Accelerator, CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def build_elementwise_graph(device_type: str, size: int) -> Graph:
    """Build graph: y = relu(x * 2 + 1)"""
    device = DeviceRef(device_type)
    input_spec = TensorType(DType.float32, shape=[size], device=device)
    
    with Graph(f"elementwise_{device_type}_{size}", input_types=[input_spec]) as graph:
        x = graph.inputs[0].tensor
        
        multiplier = ops.constant(
            np.full(size, 2.0, dtype=np.float32),
            dtype=DType.float32,
            device=x.device
        )
        
        offset = ops.constant(
            np.full(size, 1.0, dtype=np.float32),
            dtype=DType.float32,
            device=x.device
        )
        
        y = ops.mul(x, multiplier)
        y = ops.add(y, offset)
        y = ops.relu(y)
        
        graph.output(y)
    
    return graph


def benchmark_size(device, device_type: str, size: int, iterations: int = 100, warmup: int = 10):
    """Benchmark for specific tensor size using provided device."""
    
    # Build and compile graph
    try:
        graph = build_elementwise_graph(device_type, size)
        session = InferenceSession(devices=[device])
        model = session.load(graph)
    except Exception as e:
        return None
    
    # Prepare input
    input_data_np = np.random.randn(size).astype(np.float32)
    input_data = Tensor.from_numpy(input_data_np).to(device)
    
    # Warmup
    for _ in range(warmup):
        model.execute(input_data)
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = model.execute(input_data)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    mean_time = np.mean(times)
    
    return mean_time


def main():
    print("="*70)
    print("Element-wise Operations: Size Scaling Benchmark (CPU vs GPU)")
    print("="*70)
    
    # Test different sizes (up to 8M elements)
    sizes = [4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 8388608]
    iterations = 100
    warmup = 10
    
    print(f"\nConfig: {warmup} warmup iterations, {iterations} benchmark iterations per size")
    print(f"Testing sizes from 4 to 8M elements...\n")
    print(f"{'Size':<12} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<12} {'Winner'}")
    print("-" * 70)
    
    # Initialize devices once
    try:
        cpu_device = CPU()
        gpu_device = Accelerator()
    except Exception as e:
        print(f"Failed to initialize devices: {e}")
        return
    
    results = []
    
    for i, size in enumerate(sizes, 1):
        print(f"[{i}/{len(sizes)}] Testing {size:,} elements...", end="", flush=True)
        
        cpu_time = benchmark_size(cpu_device, "cpu", size, iterations, warmup)
        gpu_time = benchmark_size(gpu_device, "gpu", size, iterations, warmup)
        
        if cpu_time and gpu_time:
            speedup = cpu_time / gpu_time
            winner = "GPU" if speedup > 1.0 else "CPU"
            
            print(f"\r{size:<12,} {cpu_time:<12.4f} {gpu_time:<12.4f} {speedup:<12.2f}x {winner}")
            
            results.append({
                "size": size,
                "cpu_ms": cpu_time,
                "gpu_ms": gpu_time,
                "speedup": speedup
            })
    
    # Find crossover point
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    gpu_wins = [r for r in results if r["speedup"] > 1.0]
    if gpu_wins:
        crossover = gpu_wins[0]
        print(f"\nGPU becomes faster at ~{crossover['size']:,} elements")
        print(f"At {crossover['size']:,}: {crossover['speedup']:.2f}x speedup")
        
        best = max(results, key=lambda x: x["speedup"])
        print(f"\nBest GPU speedup: {best['speedup']:.2f}x at {best['size']:,} elements")
    else:
        print("\nGPU never becomes faster than CPU for these sizes")
        print("This is expected for simple element-wise ops with small tensors")
        print("GPU overhead exceeds compute benefit")


if __name__ == "__main__":
    main()
