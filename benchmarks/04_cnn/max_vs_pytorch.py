"""
CNN MNIST: MAX vs PyTorch Benchmark
====================================

Compares performance of CNN for MNIST classification on MAX Graph vs PyTorch.

Run:
  pixi run benchmark-cnn
"""

import time
from pathlib import Path

import numpy as np
import tomllib
import torch
import torch.nn as nn
import torch.nn.functional as F
from max_cnn import CNNClassificationModel
from tqdm import tqdm

# Import from installed packages
from utils.benchmark_utils import (
    generate_markdown_report,
    save_csv_report,
    save_json_report,
    save_markdown_report,
)
from utils.paths import get_examples_dir


class PyTorchCNN(nn.Module):
    """PyTorch CNN matching our MAX Graph architecture."""

    def __init__(self):
        super(PyTorchCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Conv block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def benchmark_max(config: dict, weights: dict, test_data: np.ndarray):
    """Benchmark MAX Graph implementation."""
    print(f"\n{'=' * 70}")
    print("Benchmarking MAX Graph")
    print(f"{'=' * 70}")

    iterations = config["benchmark"]["test_iterations"]
    warmup = config["benchmark"]["warmup_iterations"]

    # Build model
    try:
        model = CNNClassificationModel(
            input_channels=config["model"]["input_channels"],
            image_height=config["model"]["image_height"],
            image_width=config["model"]["image_width"],
            num_classes=config["model"]["num_classes"],
            weights=weights,
            device=config["implementations"]["device"],
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
        "iterations": iterations,
    }

    print("\nResults:")
    print(f"  Mean:       {mean_time:.4f} ms")
    print(f"  Median:     {median_time:.4f} ms")
    print(f"  P95:        {p95_time:.4f} ms")
    print(f"  P99:        {p99_time:.4f} ms")
    print(f"  Std Dev:    {std_time:.4f} ms")
    print(f"  Throughput: {throughput:.2f} req/sec")

    return results, None


def benchmark_pytorch(config: dict, weights: dict, test_data: np.ndarray):
    """Benchmark PyTorch implementation."""
    print(f"\n{'=' * 70}")
    print("Benchmarking PyTorch")
    print(f"{'=' * 70}")

    iterations = config["benchmark"]["test_iterations"]
    warmup = config["benchmark"]["warmup_iterations"]

    # Build model
    try:
        model = PyTorchCNN()

        # Load weights (convert from MAX RSCF format to PyTorch OIHW)
        with torch.no_grad():
            # Conv weights: RSCF [H,W,I,O] → OIHW [O,I,H,W]
            conv1_w = np.transpose(weights["conv1_W"], (3, 2, 0, 1))  # [32,1,3,3]
            conv2_w = np.transpose(weights["conv2_W"], (3, 2, 0, 1))  # [64,32,3,3]

            model.conv1.weight.data = torch.from_numpy(conv1_w)
            model.conv1.bias.data = torch.from_numpy(weights["conv1_b"])
            model.conv2.weight.data = torch.from_numpy(conv2_w)
            model.conv2.bias.data = torch.from_numpy(weights["conv2_b"])
            model.fc1.weight.data = torch.from_numpy(weights["fc1_W"])
            model.fc1.bias.data = torch.from_numpy(weights["fc1_b"])
            model.fc2.weight.data = torch.from_numpy(weights["fc2_W"])
            model.fc2.bias.data = torch.from_numpy(weights["fc2_b"])

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
            for _ in tqdm(
                range(iterations), desc="Benchmark", unit="iter", ncols=80, mininterval=0.5
            ):
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
        "iterations": iterations,
    }

    print("\nResults:")
    print(f"  Mean:       {mean_time:.4f} ms")
    print(f"  Median:     {median_time:.4f} ms")
    print(f"  P95:        {p95_time:.4f} ms")
    print(f"  P99:        {p99_time:.4f} ms")
    print(f"  Std Dev:    {std_time:.4f} ms")
    print(f"  Throughput: {throughput:.2f} req/sec")

    return results, None


def main():
    print("=" * 70)
    print("CNN MNIST: MAX Graph vs PyTorch Benchmark")
    print("=" * 70)

    # Load configuration
    script_dir = Path(__file__).parent
    config_path = script_dir / "benchmark_config.toml"

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    print(f"\nLoaded config from: {config_path}")
    print("Architecture: Conv(1→32)→Pool→Conv(32→64)→Pool→Flatten→FC(128)→FC(10)")
    print(f"Image size: {config['model']['image_height']}×{config['model']['image_width']}")
    print(f"Batch size: {config['model']['batch_size']}")
    print(f"Iterations: {config['benchmark']['test_iterations']}")
    print(f"Warmup: {config['benchmark']['warmup_iterations']}")

    # Load pre-trained weights
    weights_path = get_examples_dir() / "04_cnn_mnist" / "weights" / "cnn_weights.npz"

    if not weights_path.exists():
        print(f"\n✗ Pre-trained weights not found at {weights_path}")
        print("  Please run the CNN training script first.")
        return

    weights_data = np.load(weights_path)
    weights = {
        "conv1_W": weights_data["conv1_W"],
        "conv1_b": weights_data["conv1_b"],
        "conv2_W": weights_data["conv2_W"],
        "conv2_b": weights_data["conv2_b"],
        "fc1_W": weights_data["fc1_W"],
        "fc1_b": weights_data["fc1_b"],
        "fc2_W": weights_data["fc2_W"],
        "fc2_b": weights_data["fc2_b"],
    }
    print(f"✓ Loaded pre-trained weights from {weights_path.name}")

    # Generate test data (batch of random images in NCHW format)
    np.random.seed(42)
    test_data = np.random.randn(
        config["model"]["batch_size"],
        config["model"]["input_channels"],
        config["model"]["image_height"],
        config["model"]["image_width"],
    ).astype(np.float32)
    print(f"✓ Generated test data: {test_data.shape}")

    # Benchmark MAX Graph
    max_results = None
    max_error = None
    if config["implementations"]["max_enabled"]:
        result, error = benchmark_max(config, weights, test_data)
        max_results = result
        max_error = error
    else:
        print("\nSkipping MAX Graph benchmark (disabled in config)")

    # Benchmark PyTorch
    pytorch_results = None
    pytorch_error = None
    if config["implementations"]["pytorch_enabled"]:
        result, error = benchmark_pytorch(config, weights, test_data)
        pytorch_results = result
        pytorch_error = error
    else:
        print("\nSkipping PyTorch benchmark (disabled in config)")

    # Verify correctness
    if max_results and pytorch_results:
        print(f"\n{'=' * 70}")
        print("CORRECTNESS VERIFICATION")
        print(f"{'=' * 70}")

        # Run both implementations on same small test input
        test_input = test_data[:5]  # First 5 images

        # MAX prediction
        max_model = CNNClassificationModel(
            input_channels=config["model"]["input_channels"],
            image_height=config["model"]["image_height"],
            image_width=config["model"]["image_width"],
            num_classes=config["model"]["num_classes"],
            weights=weights,
            device=config["implementations"]["device"],
        )
        max_pred, max_probs = max_model.predict(test_input)

        # PyTorch prediction
        pytorch_model = PyTorchCNN()
        with torch.no_grad():
            conv1_w = np.transpose(weights["conv1_W"], (3, 2, 0, 1))
            conv2_w = np.transpose(weights["conv2_W"], (3, 2, 0, 1))
            pytorch_model.conv1.weight.data = torch.from_numpy(conv1_w)
            pytorch_model.conv1.bias.data = torch.from_numpy(weights["conv1_b"])
            pytorch_model.conv2.weight.data = torch.from_numpy(conv2_w)
            pytorch_model.conv2.bias.data = torch.from_numpy(weights["conv2_b"])
            pytorch_model.fc1.weight.data = torch.from_numpy(weights["fc1_W"])
            pytorch_model.fc1.bias.data = torch.from_numpy(weights["fc1_b"])
            pytorch_model.fc2.weight.data = torch.from_numpy(weights["fc2_W"])
            pytorch_model.fc2.bias.data = torch.from_numpy(weights["fc2_b"])
        pytorch_model.eval()

        with torch.no_grad():
            pytorch_logits = pytorch_model(torch.from_numpy(test_input)).numpy()
            # Apply softmax
            exp_logits = np.exp(pytorch_logits - np.max(pytorch_logits, axis=-1, keepdims=True))
            pytorch_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            pytorch_pred = np.argmax(pytorch_probs, axis=1)

        # Compare predictions
        pred_match = np.array_equal(max_pred, pytorch_pred)

        # Compare probabilities
        max_error_probs = np.abs(max_probs - pytorch_probs).max()
        relative_error = max_error_probs / (np.abs(pytorch_probs).mean() + 1e-8)

        print(f"\nPredictions match: {'✓ Yes' if pred_match else '✗ No'}")
        print(f"Max probability error: {max_error_probs:.6f}")
        print(f"Relative error: {relative_error:.6f}")

        if np.allclose(max_probs, pytorch_probs, rtol=1e-4, atol=1e-5):
            print("✓ Probabilities match within tolerance (rtol=1e-4, atol=1e-5)")
        else:
            print("✗ WARNING: Probabilities differ significantly!")
            print("\nSample predictions (first 3):")
            for i in range(min(3, len(test_input))):
                print(f"  Sample {i + 1}: PyTorch={pytorch_pred[i]}, MAX={max_pred[i]}")

    # Compare performance
    if max_results and pytorch_results:
        print(f"\n{'=' * 70}")
        print("PERFORMANCE COMPARISON")
        print(f"{'=' * 70}")

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
        print("\nConsistency (lower is better):")
        print(f"  PyTorch CV: {pytorch_cv:.4f}")
        print(f"  MAX CV:     {max_cv:.4f}")

    # Generate and save reports
    print(f"\n{'=' * 70}")
    print("GENERATING REPORTS")
    print(f"{'=' * 70}")

    results_dir = script_dir / "results"

    # Markdown report (use generic CPU/GPU labels - PyTorch as CPU, MAX as GPU)
    report = generate_markdown_report(
        benchmark_name="CNN MNIST: MAX Graph vs PyTorch",
        description="Compares performance of CNN for MNIST classification on MAX Graph vs PyTorch.",
        config=config,
        cpu_results=pytorch_results,
        gpu_results=max_results,
        gpu_error=max_error,
    )
    md_path = save_markdown_report(report, results_dir, prefix="max_vs_pytorch")
    print(f"\n✓ Markdown report: {md_path}")

    # JSON report
    json_data = {
        "benchmark": config["benchmark"],
        "config": config,
        "results": {"pytorch": pytorch_results, "max": max_results},
        "errors": {"pytorch": pytorch_error, "max": max_error},
    }
    json_path = save_json_report(json_data, results_dir, prefix="max_vs_pytorch")
    print(f"✓ JSON report:     {json_path}")

    # CSV report
    csv_data = {}
    if pytorch_results:
        csv_data["pytorch"] = pytorch_results
    if max_results:
        csv_data["max"] = max_results

    if csv_data:
        csv_path = save_csv_report(csv_data, results_dir, prefix="max_vs_pytorch")
        print(f"✓ CSV report:      {csv_path}")


if __name__ == "__main__":
    main()
