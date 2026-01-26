"""
Element-wise Operations Benchmark (CPU vs GPU)
===============================================

Compares performance of element-wise operations on CPU vs GPU.

Run:
  pixi run benchmark-elementwise
"""

import time
from pathlib import Path

import numpy as np
import tomllib
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from tqdm import tqdm

# Import from installed package
from utils.benchmark_utils import (
    generate_markdown_report,
    save_csv_report,
    save_json_report,
    save_markdown_report,
)


def build_elementwise_graph(device_type: str, multiplier: float, offset: float, size: int) -> Graph:
    """Build graph: y = relu(x * multiplier + offset)"""
    device = DeviceRef(device_type)
    input_spec = TensorType(DType.float32, shape=[size], device=device)

    with Graph(f"elementwise_{device_type}", input_types=[input_spec]) as graph:
        x = graph.inputs[0].tensor

        multiplier_const = ops.constant(
            np.full(size, multiplier, dtype=np.float32), dtype=DType.float32, device=x.device
        )

        offset_const = ops.constant(
            np.full(size, offset, dtype=np.float32), dtype=DType.float32, device=x.device
        )

        y = ops.mul(x, multiplier_const)
        y = ops.add(y, offset_const)
        y = ops.relu(y)

        graph.output(y)

    return graph


def benchmark_device(device_type: str, config: dict, input_data_np: np.ndarray):
    """Benchmark element-wise operations on specified device."""

    print(f"\n{'=' * 60}")
    print(f"Benchmarking {device_type.upper()}")
    print(f"{'=' * 60}")

    iterations = config["benchmark"]["test_iterations"]
    warmup = config["benchmark"]["warmup_iterations"]
    multiplier = config["graph"]["multiplier"]
    offset = config["graph"]["offset"]
    size = config["graph"]["input_size"]

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
        graph = build_elementwise_graph(device_type, multiplier, offset, size)
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        print("✓ Graph compiled")
    except Exception as e:
        error_msg = f"Failed to compile graph: {str(e)}"
        print(f"✗ {error_msg}")
        return None, error_msg

    # Prepare input
    input_data = Buffer.from_numpy(input_data_np).to(device)

    # Warmup
    print(f"\nWarming up ({warmup} iterations)...")
    try:
        for _ in tqdm(range(warmup), desc="Warmup", unit="iter", ncols=80, mininterval=0.5):
            model.execute(input_data)
    except Exception as e:
        error_msg = f"Execution failed during warmup: {str(e)}"
        print(f"✗ {error_msg}")
        return None, error_msg

    # Benchmark
    print(f"\nRunning benchmark ({iterations} iterations)...")
    times = []

    try:
        for _ in tqdm(range(iterations), desc="Benchmark", unit="iter", ncols=80, mininterval=0.5):
            start = time.perf_counter()
            output = model.execute(input_data)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds
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
        "iterations": iterations,
    }

    # Verify correctness
    output_np = output[0].to_numpy()
    expected = np.maximum(0, input_data_np * multiplier + offset)
    correct = np.allclose(output_np, expected)

    print("\nResults:")
    print(f"  Mean:       {mean_time:.4f} ms")
    print(f"  Median:     {median_time:.4f} ms")
    print(f"  P95:        {p95_time:.4f} ms")
    print(f"  P99:        {p99_time:.4f} ms")
    print(f"  Std Dev:    {std_time:.4f} ms")
    print(f"  Throughput: {throughput:.2f} req/sec")
    print(f"  Correctness: {'✓ PASS' if correct else '✗ FAIL'}")

    return results, None


def main():
    print("=" * 60)
    print("Element-wise Operations: CPU vs GPU Benchmark")
    print("=" * 60)

    # Load configuration
    script_dir = Path(__file__).parent
    config_path = script_dir / "benchmark_config.toml"

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    print(f"\nLoaded config from: {config_path}")
    print(f"Iterations: {config['benchmark']['test_iterations']}")
    print(f"Warmup: {config['benchmark']['warmup_iterations']}")
    print(f"Graph: y = relu(x * {config['graph']['multiplier']} + {config['graph']['offset']})")
    print(f"Input size: {config['graph']['input_size']}")

    # Prepare test data
    test_data = np.array(config["test_data"]["input_values"], dtype=np.float32)

    # Benchmark CPU
    cpu_results = None
    cpu_error = None
    if config["devices"]["cpu_enabled"]:
        result, error = benchmark_device("cpu", config, test_data)
        cpu_results = result
        cpu_error = error
    else:
        print("\nSkipping CPU benchmark (disabled in config)")

    # Benchmark GPU
    gpu_results = None
    gpu_error = None
    if config["devices"]["gpu_enabled"]:
        result, error = benchmark_device("gpu", config, test_data)
        gpu_results = result
        gpu_error = error
    else:
        print("\nSkipping GPU benchmark (disabled in config)")

    # Compare results (console output)
    if cpu_results and gpu_results:
        print(f"\n{'=' * 60}")
        print("COMPARISON")
        print(f"{'=' * 60}")

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
        print("\nConsistency (lower is better):")
        print(f"  CPU CV: {cpu_cv:.4f}")
        print(f"  GPU CV: {gpu_cv:.4f}")

        if cpu_cv < gpu_cv:
            consistency_improvement = (gpu_cv / cpu_cv - 1) * 100
            print(f"  CPU is {consistency_improvement:.1f}% more consistent")
        else:
            consistency_improvement = (cpu_cv / gpu_cv - 1) * 100
            print(f"  GPU is {consistency_improvement:.1f}% more consistent")

    # Generate and save reports
    print(f"\n{'=' * 60}")
    print("GENERATING REPORTS")
    print(f"{'=' * 60}")

    results_dir = script_dir / "results"

    # Markdown report
    report = generate_markdown_report(
        benchmark_name="Element-wise Operations: CPU vs GPU",
        description="Compares performance of element-wise operations (multiply, add, relu) on CPU vs GPU.",
        config=config,
        cpu_results=cpu_results,
        gpu_results=gpu_results,
        gpu_error=gpu_error,
    )
    md_path = save_markdown_report(report, results_dir, prefix="cpu_vs_gpu")
    print(f"\n✓ Markdown report: {md_path}")

    # JSON report
    json_data = {
        "benchmark": config["benchmark"],
        "config": config,
        "results": {"cpu": cpu_results, "gpu": gpu_results},
        "errors": {"cpu": cpu_error, "gpu": gpu_error},
    }
    json_path = save_json_report(json_data, results_dir, prefix="cpu_vs_gpu")
    print(f"✓ JSON report:     {json_path}")

    # CSV report
    csv_data = {}
    if cpu_results:
        csv_data["cpu"] = cpu_results
    if gpu_results:
        csv_data["gpu"] = gpu_results

    if csv_data:
        csv_path = save_csv_report(csv_data, results_dir, prefix="cpu_vs_gpu")
        print(f"✓ CSV report:      {csv_path}")


if __name__ == "__main__":
    main()
