"""
MLP Regression: MAX vs PyTorch Benchmark
=========================================

Compares performance of 3-layer MLP for regression on MAX Graph vs PyTorch.

Run:
  pixi run benchmark-mlp
"""

import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path dynamically
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.python.utils.paths import add_project_root_to_path
add_project_root_to_path()

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark_utils import (
    generate_markdown_report,
    save_markdown_report,
    save_json_report,
    save_csv_report
)

from src.python.max_mlp import MLPRegressionModel


class PyTorchMLP(nn.Module):
    """PyTorch MLP matching our MAX Graph architecture."""
    
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(PyTorchMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def benchmark_max(config: dict, weights: dict, test_data: np.ndarray):
    """Benchmark MAX Graph implementation."""
    print(f"\n{'='*70}")
    print("Benchmarking MAX Graph")
    print(f"{'='*70}")
    
    iterations = config['benchmark']['test_iterations']
    warmup = config['benchmark']['warmup_iterations']
    
    # Build model
    try:
        model = MLPRegressionModel(
            input_size=config['model']['input_size'],
            hidden_size1=config['model']['hidden_size1'],
            hidden_size2=config['model']['hidden_size2'],
            output_size=config['model']['output_size'],
            weights=weights,
            device=config['implementations']['device'],
        )
        print("✓ Model loaded")
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        print(f"✗ {error_msg}")
        return None, error_msg
    
    # Warmup
    print(f"\nWarming up ({warmup} iterations)...")
    try:
        for _ in tqdm(range(warmup), desc="Warmup", unit="iter", ncols=80, mininterval=0.5):
            _ = model.predict(test_data)
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
            _ = model.predict(test_data)
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
        "implementation": "MAX Graph",
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


def benchmark_pytorch(config: dict, weights: dict, test_data: np.ndarray):
    """Benchmark PyTorch implementation."""
    print(f"\n{'='*70}")
    print("Benchmarking PyTorch")
    print(f"{'='*70}")
    
    iterations = config['benchmark']['test_iterations']
    warmup = config['benchmark']['warmup_iterations']
    
    # Build model
    try:
        model = PyTorchMLP(
            input_size=config['model']['input_size'],
            hidden_size1=config['model']['hidden_size1'],
            hidden_size2=config['model']['hidden_size2'],
            output_size=config['model']['output_size'],
        )
        
        # Load weights
        with torch.no_grad():
            model.fc1.weight.data = torch.from_numpy(weights['W1'])
            model.fc1.bias.data = torch.from_numpy(weights['b1'])
            model.fc2.weight.data = torch.from_numpy(weights['W2'])
            model.fc2.bias.data = torch.from_numpy(weights['b2'])
            model.fc3.weight.data = torch.from_numpy(weights['W3'])
            model.fc3.bias.data = torch.from_numpy(weights['b3'])
        
        model.eval()
        print("✓ Model loaded")
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        print(f"✗ {error_msg}")
        return None, error_msg
    
    # Convert test data to PyTorch tensor
    test_tensor = torch.from_numpy(test_data)
    
    # Warmup
    print(f"\nWarming up ({warmup} iterations)...")
    try:
        with torch.no_grad():
            for _ in tqdm(range(warmup), desc="Warmup", unit="iter", ncols=80, mininterval=0.5):
                _ = model(test_tensor)
    except Exception as e:
        error_msg = f"Execution failed during warmup: {str(e)}"
        print(f"✗ {error_msg}")
        return None, error_msg
    
    # Benchmark
    print(f"\nRunning benchmark ({iterations} iterations)...")
    times = []
    
    try:
        with torch.no_grad():
            for _ in tqdm(range(iterations), desc="Benchmark", unit="iter", ncols=80, mininterval=0.5):
                start = time.perf_counter()
                _ = model(test_tensor)
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
        "implementation": "PyTorch",
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
    print("="*70)
    print("MLP Regression: MAX Graph vs PyTorch Benchmark")
    print("="*70)
    
    # Load configuration
    script_dir = Path(__file__).parent
    config_path = script_dir / "benchmark_config.toml"
    
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    
    print(f"\nLoaded config from: {config_path}")
    print(f"Architecture: {config['model']['input_size']} → {config['model']['hidden_size1']} → {config['model']['hidden_size2']} → {config['model']['output_size']}")
    print(f"Batch size: {config['model']['batch_size']}")
    print(f"Iterations: {config['benchmark']['test_iterations']}")
    print(f"Warmup: {config['benchmark']['warmup_iterations']}")
    
    # Load pre-trained weights
    from src.python.utils.paths import get_examples_dir
    weights_path = get_examples_dir() / "03_mlp_regression" / "weights" / "mlp_weights.npz"
    
    if not weights_path.exists():
        print(f"\n✗ Pre-trained weights not found at {weights_path}")
        print("  Please run the MLP training script first.")
        return
    
    weights_data = np.load(weights_path)
    weights = {
        'W1': weights_data['W1'],
        'b1': weights_data['b1'],
        'W2': weights_data['W2'],
        'b2': weights_data['b2'],
        'W3': weights_data['W3'],
        'b3': weights_data['b3'],
    }
    print(f"✓ Loaded pre-trained weights from {weights_path.name}")
    
    # Generate test data (batch of random inputs)
    np.random.seed(42)
    test_data = np.random.randn(
        config['model']['batch_size'],
        config['model']['input_size']
    ).astype(np.float32)
    print(f"✓ Generated test data: {test_data.shape}")
    
    # Benchmark MAX Graph
    max_results = None
    max_error = None
    if config['implementations']['max_enabled']:
        result, error = benchmark_max(config, weights, test_data)
        max_results = result
        max_error = error
    else:
        print("\nSkipping MAX Graph benchmark (disabled in config)")
    
    # Benchmark PyTorch
    pytorch_results = None
    pytorch_error = None
    if config['implementations']['pytorch_enabled']:
        result, error = benchmark_pytorch(config, weights, test_data)
        pytorch_results = result
        pytorch_error = error
    else:
        print("\nSkipping PyTorch benchmark (disabled in config)")
    
    # Verify correctness
    if max_results and pytorch_results:
        print(f"\n{'='*70}")
        print("CORRECTNESS VERIFICATION")
        print(f"{'='*70}")
        
        # Run both implementations on same small test input
        test_input = test_data[:5]  # First 5 samples
        
        # MAX prediction
        max_model = MLPRegressionModel(
            input_size=config['model']['input_size'],
            hidden_size1=config['model']['hidden_size1'],
            hidden_size2=config['model']['hidden_size2'],
            output_size=config['model']['output_size'],
            weights=weights,
            device=config['implementations']['device'],
        )
        max_pred = max_model.predict(test_input)
        
        # PyTorch prediction
        pytorch_model = PyTorchMLP(
            input_size=config['model']['input_size'],
            hidden_size1=config['model']['hidden_size1'],
            hidden_size2=config['model']['hidden_size2'],
            output_size=config['model']['output_size'],
        )
        with torch.no_grad():
            pytorch_model.fc1.weight.data = torch.from_numpy(weights['W1'])
            pytorch_model.fc1.bias.data = torch.from_numpy(weights['b1'])
            pytorch_model.fc2.weight.data = torch.from_numpy(weights['W2'])
            pytorch_model.fc2.bias.data = torch.from_numpy(weights['b2'])
            pytorch_model.fc3.weight.data = torch.from_numpy(weights['W3'])
            pytorch_model.fc3.bias.data = torch.from_numpy(weights['b3'])
        pytorch_model.eval()
        
        with torch.no_grad():
            pytorch_pred = pytorch_model(torch.from_numpy(test_input)).numpy()
        
        # Compare
        max_error = np.abs(max_pred - pytorch_pred).max()
        relative_error = max_error / (np.abs(pytorch_pred).mean() + 1e-8)
        
        print(f"\nMax absolute error: {max_error:.6f}")
        print(f"Relative error: {relative_error:.6f}")
        
        if np.allclose(max_pred, pytorch_pred, rtol=1e-4, atol=1e-5):
            print("✓ Outputs match within tolerance (rtol=1e-4, atol=1e-5)")
        else:
            print("✗ WARNING: Outputs differ significantly!")
            print(f"\nSample predictions (first 3):")
            for i in range(min(3, len(test_input))):
                print(f"  Sample {i+1}: PyTorch={pytorch_pred[i,0]:.6f}, MAX={max_pred[i,0]:.6f}")
    
    # Compare performance
    if max_results and pytorch_results:
        print(f"\n{'='*70}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*70}")
        
        speedup = pytorch_results["mean_ms"] / max_results["mean_ms"]
        throughput_ratio = max_results["throughput"] / pytorch_results["throughput"]
        
        print(f"\nPyTorch: {pytorch_results['mean_ms']:.4f} ms mean")
        print(f"MAX:     {max_results['mean_ms']:.4f} ms mean")
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"MAX is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than PyTorch")
        print(f"Throughput improvement: {throughput_ratio:.2f}x")
        
        # Latency consistency
        pytorch_cv = pytorch_results["std_ms"] / pytorch_results["mean_ms"]
        max_cv = max_results["std_ms"] / max_results["mean_ms"]
        print(f"\nConsistency (lower is better):")
        print(f"  PyTorch CV: {pytorch_cv:.4f}")
        print(f"  MAX CV:     {max_cv:.4f}")
    
    # Generate and save reports
    print(f"\n{'='*70}")
    print("GENERATING REPORTS")
    print(f"{'='*70}")
    
    results_dir = script_dir / "results"
    
    # Markdown report (use generic CPU/GPU labels - PyTorch as CPU, MAX as GPU)
    report = generate_markdown_report(
        benchmark_name="MLP Regression: MAX Graph vs PyTorch",
        description="Compares performance of 3-layer MLP regression on MAX Graph vs PyTorch.",
        config=config,
        cpu_results=pytorch_results,
        gpu_results=max_results,
        gpu_error=max_error
    )
    md_path = save_markdown_report(report, results_dir, prefix="max_vs_pytorch")
    print(f"\n✓ Markdown report: {md_path}")
    
    # JSON report
    json_data = {
        "benchmark": config['benchmark'],
        "config": config,
        "results": {
            "pytorch": pytorch_results,
            "max": max_results
        },
        "errors": {
            "pytorch": pytorch_error,
            "max": max_error
        }
    }
    json_path = save_json_report(json_data, results_dir, prefix="max_vs_pytorch")
    print(f"✓ JSON report:     {json_path}")
    
    # CSV report
    csv_data = {}
    if pytorch_results:
        csv_data['pytorch'] = pytorch_results
    if max_results:
        csv_data['max'] = max_results
    
    if csv_data:
        csv_path = save_csv_report(csv_data, results_dir, prefix="max_vs_pytorch")
        print(f"✓ CSV report:      {csv_path}")


if __name__ == "__main__":
    main()
