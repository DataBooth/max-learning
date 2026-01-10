"""
Linear Layer: CPU vs GPU Benchmark
====================================

Compares performance of linear layer operations on CPU vs GPU.

Note: GPU will fail due to missing matmul kernel on Apple Silicon.

Run:
  pixi run benchmark-linear
"""

import sys
import time
import tomllib
import numpy as np
from pathlib import Path
from max.driver import Accelerator, CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

# Add benchmarks/ to path for benchmark_utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark_utils import generate_markdown_report, save_markdown_report


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


def benchmark_device(device_type: str, config: dict):
    """Benchmark linear layer on specified device."""
    
    print(f"\n{'='*60}")
    print(f"Benchmarking {device_type.upper()}")
    print(f"{'='*60}")
    
    iterations = config['benchmark']['test_iterations']
    warmup = config['benchmark']['warmup_iterations']
    batch_size = config['graph']['batch_size']
    input_features = config['graph']['input_features']
    output_features = config['graph']['output_features']
    
    # Initialize device
    try:
        if device_type == "gpu":
            device = Accelerator()
        else:
            device = CPU()
        print(f"✓ Device initialized: {device}")
    except Exception as e:
        error_msg = f"Failed to initialize {device_type}: {str(e)}"
        print(f"✗ {error_msg}")
        return None, error_msg
    
    # Build and compile graph
    try:
        graph = build_linear_layer_graph(device_type, batch_size, input_features, output_features)
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        print(f"✓ Graph compiled")
    except Exception as e:
        error_msg = f"Failed to compile graph: {str(e)}"
        print(f"✗ {error_msg}")
        if device_type == "gpu" and "matmul" in str(e).lower():
            error_msg += " (matmul kernel not available on Apple Silicon GPU)"
            print(f"  Note: matmul kernel not available on Apple Silicon GPU")
        return None, error_msg
    
    # Prepare input
    input_data_np = np.random.randn(batch_size, input_features).astype(np.float32)
    input_data = Tensor.from_numpy(input_data_np).to(device)
    
    # Warmup
    print(f"\nWarming up ({warmup} iterations)...")
    try:
        for _ in range(warmup):
            model.execute(input_data)
    except Exception as e:
        error_msg = f"Execution failed during warmup: {str(e)}"
        print(f"✗ {error_msg}")
        return None, error_msg
    
    # Benchmark
    print(f"Running benchmark ({iterations} iterations)...")
    times = []
    
    try:
        for _ in range(iterations):
            start = time.perf_counter()
            output = model.execute(input_data)
            end = time.perf_counter()
            times.append((end - start) * 1000)
    except Exception as e:
        error_msg = f"Execution failed during benchmark: {str(e)}"
        print(f"✗ {error_msg}")
        return None, error_msg
    
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
    
    print(f"\nResults:")
    print(f"  Mean:       {mean_time:.4f} ms")
    print(f"  Median:     {median_time:.4f} ms")
    print(f"  P95:        {p95_time:.4f} ms")
    print(f"  P99:        {p99_time:.4f} ms")
    print(f"  Std Dev:    {std_time:.4f} ms")
    print(f"  Throughput: {throughput:.2f} req/sec")
    
    return results, None


def main():
    print("="*60)
    print("Linear Layer: CPU vs GPU Benchmark")
    print("="*60)
    
    # Load configuration
    script_dir = Path(__file__).parent
    config_path = script_dir / "benchmark_config.toml"
    
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    
    print(f"\nLoaded config from: {config_path}")
    print(f"Iterations: {config['benchmark']['test_iterations']}")
    print(f"Warmup: {config['benchmark']['warmup_iterations']}")
    print(f"Layer: {config['graph']['input_features']} → {config['graph']['output_features']}")
    print(f"Batch size: {config['graph']['batch_size']}")
    
    # Benchmark CPU
    cpu_results = None
    cpu_error = None
    if config['devices']['cpu_enabled']:
        result, error = benchmark_device("cpu", config)
        cpu_results = result
        cpu_error = error
    else:
        print("\nSkipping CPU benchmark (disabled in config)")
    
    # Benchmark GPU
    gpu_results = None
    gpu_error = None
    if config['devices']['gpu_enabled']:
        result, error = benchmark_device("gpu", config)
        gpu_results = result
        gpu_error = error
    else:
        print("\nSkipping GPU benchmark (disabled in config)")
    
    # Compare results (console output)
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
    
    # Generate markdown report
    print(f"\n{'='*60}")
    print("GENERATING REPORT")
    print(f"{'='*60}")
    
    report = generate_markdown_report(
        benchmark_name="Linear Layer: CPU vs GPU",
        description="Compares performance of linear layer operations (matmul, add, relu) on CPU vs GPU.",
        config=config,
        cpu_results=cpu_results,
        gpu_results=gpu_results,
        gpu_error=gpu_error
    )
    
    results_dir = script_dir / "results"
    report_path = save_markdown_report(report, results_dir, prefix="cpu_vs_gpu")
    
    print(f"\n✓ Report saved: {report_path}")


if __name__ == "__main__":
    main()
