#!/usr/bin/env python
"""Config-driven benchmark harness for ML inference implementations.

Usage:
    python benchmark.py                    # Use default config
    python benchmark.py --config my.toml   # Use custom config
"""

import argparse
import importlib
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil
import tomli
from transformers import pipeline

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add benchmarks directory to path for shared utilities
BENCHMARKS_DIR = Path(__file__).parent.parent
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from benchmark_utils import get_machine_id

@dataclass
class BenchmarkResult:
    """Results from benchmarking a single implementation."""
    name: str
    description: str
    load_time_s: float
    mean_ms: float
    median_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    throughput_rps: float
    iterations: int
    raw_latencies_ms: list[float] | None = None


@dataclass
class ValidationResult:
    """Results from correctness validation."""
    text: str
    expected: str
    predicted: str
    confidence: float
    correct: bool


class ImplementationLoader:
    """Loads and wraps different implementation types."""
    
    @staticmethod
    def load(impl_config: dict, model_path: str) -> tuple[Any, Callable]:
        """Load implementation and return (model, predict_fn)."""
        impl_type = impl_config.get('type')
        
        if impl_type == 'custom':
            return ImplementationLoader._load_custom(impl_config, model_path)
        elif impl_type == 'huggingface_pipeline':
            return ImplementationLoader._load_huggingface(impl_config, model_path)
        elif impl_type == 'onnx':
            return ImplementationLoader._load_onnx(impl_config)
        else:
            raise ValueError(f"Unknown implementation type: {impl_type}")
    
    @staticmethod
    def _load_custom(config: dict, model_path: str):
        """Load custom Python implementation."""
        module_path = config['module']
        class_name = config['class_name']
        
        # Import module
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        
        # Instantiate with model_path
        model = cls(Path(model_path))
        
        # Predict function
        def predict_fn(text: str) -> dict:
            result = model.predict(text)
            # Normalize output format
            return {
                'label': result['label'],
                'score': result['confidence']
            }
        
        return model, predict_fn
    
    @staticmethod
    def _load_huggingface(config: dict, model_path: str):
        """Load HuggingFace pipeline."""
        device = config.get('device', 'cpu')
        model = pipeline('sentiment-analysis', model=model_path, device=device)
        
        def predict_fn(text: str) -> dict:
            result = model(text)[0]
            return result
        
        return model, predict_fn
    
    @staticmethod
    def _load_onnx(config: dict):
        """Load ONNX Runtime implementation."""
        raise NotImplementedError("ONNX support not yet implemented")


def load_test_data(file_path: Path, repeat: int = 1, categories: list[str] | None = None) -> list[dict]:
    """Load test data from JSONL file."""
    data = []
    with open(file_path) as f:
        for line in f:
            item = json.loads(line)
            # Filter by category if specified
            if categories and item.get('category') not in categories:
                continue
            data.append(item)
    
    # Repeat dataset
    return data * repeat


def benchmark_implementation(
    name: str,
    description: str,
    model: Any,
    predict_fn: Callable,
    test_data: list[dict],
    warmup: int,
    iterations: int,
    include_raw: bool = False
) -> BenchmarkResult:
    """Benchmark a single implementation."""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {name}")
    print(f"{'='*70}")
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for item in test_data[:warmup]:
        predict_fn(item['text'])
    
    # Benchmark
    print(f"Running benchmark ({iterations} iterations)...")
    latencies = []
    
    for i, item in enumerate(test_data[:iterations]):
        start = time.perf_counter()
        _ = predict_fn(item['text'])
        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{iterations}...")
    
    # Calculate statistics
    return BenchmarkResult(
        name=name,
        description=description,
        load_time_s=0,  # Set by caller
        mean_ms=statistics.mean(latencies),
        median_ms=statistics.median(latencies),
        p50_ms=float(np.percentile(latencies, 50)),
        p95_ms=float(np.percentile(latencies, 95)),
        p99_ms=float(np.percentile(latencies, 99)),
        min_ms=min(latencies),
        max_ms=max(latencies),
        std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        throughput_rps=1000 / statistics.mean(latencies),
        iterations=iterations,
        raw_latencies_ms=latencies if include_raw else None
    )


def validate_implementation(
    predict_fn: Callable,
    validation_data: list[dict]
) -> list[ValidationResult]:
    """Validate implementation correctness."""
    results = []
    for item in validation_data:
        pred = predict_fn(item['text'])
        results.append(ValidationResult(
            text=item['text'],
            expected=item['expected_label'],
            predicted=pred['label'],
            confidence=pred['score'],
            correct=pred['label'] == item['expected_label']
        ))
    return results


def print_results(results: list[BenchmarkResult], baseline_name: str | None = None):
    """Print benchmark results to console."""
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    
    # Find baseline for comparison
    baseline = None
    if baseline_name:
        baseline = next((r for r in results if r.name == baseline_name), None)
    
    for result in results:
        print(f"\n{result.name}")
        print(f"  Description: {result.description}")
        print(f"  Mean latency:    {result.mean_ms:8.2f} ms")
        print(f"  Median latency:  {result.median_ms:8.2f} ms")
        print(f"  P95 latency:     {result.p95_ms:8.2f} ms")
        print(f"  P99 latency:     {result.p99_ms:8.2f} ms")
        print(f"  Min latency:     {result.min_ms:8.2f} ms")
        print(f"  Max latency:     {result.max_ms:8.2f} ms")
        print(f"  Std deviation:   {result.std_ms:8.2f} ms")
        print(f"  Throughput:      {result.throughput_rps:8.2f} req/sec")
        
        if baseline and baseline != result:
            speedup = baseline.mean_ms / result.mean_ms
            print(f"  Speedup vs {baseline_name}: {speedup:.2f}x")
    
    # Summary comparison
    if len(results) > 1:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        
        fastest = min(results, key=lambda r: r.mean_ms)
        print(f"\nFastest: {fastest.name}")
        print(f"  Mean latency: {fastest.mean_ms:.2f} ms")
        print(f"  Throughput: {fastest.throughput_rps:.2f} req/sec")
        
        if baseline and fastest != baseline:
            speedup = baseline.mean_ms / fastest.mean_ms
            print(f"  Speedup: {speedup:.2f}x faster than {baseline_name}")


def generate_markdown_report(results: list[BenchmarkResult], config: dict, validation_results: dict) -> str:
    """Generate markdown benchmark report."""
    baseline_name = config['output'].get('baseline')
    baseline = next((r for r in results if r.name == baseline_name), None) if baseline_name else None
    fastest = min(results, key=lambda r: r.mean_ms)
    
    md = []
    md.append(f"# {config['benchmark']['name']}")
    md.append("")
    md.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"**Description**: {config['benchmark']['description']}")
    md.append("")
    
    # Executive Summary
    md.append("## Executive Summary")
    md.append("")
    md.append(f"**Winner**: {fastest.name} 🏆")
    md.append("")
    
    # Compare fastest to all others
    comparisons = [r for r in results if r != fastest]
    if comparisons:
        slowest = comparisons[0]  # Since we sort by mean
        speedup = slowest.mean_ms / fastest.mean_ms
        md.append(f"- **{speedup:.2f}x faster** than {slowest.name}")
        throughput_increase = ((fastest.throughput_rps / slowest.throughput_rps) - 1) * 100
        md.append(f"- **{throughput_increase:.1f}% more throughput** than {slowest.name}")
        p95_improvement = ((slowest.p95_ms - fastest.p95_ms) / slowest.p95_ms) * 100
        md.append(f"- **{p95_improvement:.1f}% better P95 latency** than {slowest.name}")
    md.append("")
    
    # Configuration
    md.append("## Configuration")
    md.append("")
    md.append(f"- **Model**: `{config['model']['path']}`")
    md.append(f"- **Warmup iterations**: {config['benchmark']['warmup_iterations']}")
    md.append(f"- **Test iterations**: {config['benchmark']['test_iterations']}")
    md.append(f"- **Implementations tested**: {len(results)}")
    md.append("")
    
    # Results Table
    md.append("## Performance Results")
    md.append("")
    md.append("| Implementation | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Throughput (req/s) | Load Time (s) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    
    for result in results:
        winner_marker = " 🏆" if result == fastest else ""
        md.append(f"| {result.name}{winner_marker} | {result.mean_ms:.2f} | {result.median_ms:.2f} | "
                 f"{result.p95_ms:.2f} | {result.p99_ms:.2f} | {result.throughput_rps:.2f} | {result.load_time_s:.2f} |")
    md.append("")
    
    # Detailed Metrics
    md.append("## Detailed Metrics")
    md.append("")
    
    for result in results:
        md.append(f"### {result.name}")
        md.append("")
        md.append(f"{result.description}")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|---|---:|")
        md.append(f"| Mean latency | {result.mean_ms:.2f} ms |")
        md.append(f"| Median latency | {result.median_ms:.2f} ms |")
        md.append(f"| P50 latency | {result.p50_ms:.2f} ms |")
        md.append(f"| P95 latency | {result.p95_ms:.2f} ms |")
        md.append(f"| P99 latency | {result.p99_ms:.2f} ms |")
        md.append(f"| Min latency | {result.min_ms:.2f} ms |")
        md.append(f"| Max latency | {result.max_ms:.2f} ms |")
        md.append(f"| Std deviation | {result.std_ms:.2f} ms |")
        md.append(f"| Throughput | {result.throughput_rps:.2f} req/sec |")
        md.append(f"| Load time | {result.load_time_s:.2f} seconds |")
        
        if baseline and baseline != result:
            speedup = baseline.mean_ms / result.mean_ms
            md.append(f"| Speedup vs {baseline_name} | {speedup:.2f}x |")
        md.append("")
    
    # Validation Results
    md.append("## Correctness Validation")
    md.append("")
    
    for impl_name, val_results in validation_results.items():
        correct = sum(1 for v in val_results if v.correct)
        accuracy = 100 * correct / len(val_results)
        md.append(f"### {impl_name}")
        md.append("")
        md.append(f"**Accuracy**: {correct}/{len(val_results)} ({accuracy:.1f}%)")
        md.append("")
        md.append("| Text | Expected | Predicted | Confidence | Result |")
        md.append("|---|---|---|---:|---|")
        
        for v in val_results:
            status = "✓" if v.correct else "✗"
            text_short = v.text[:40] + "..." if len(v.text) > 40 else v.text
            md.append(f"| {text_short} | {v.expected} | {v.predicted} | {v.confidence:.4f} | {status} |")
        md.append("")
    
    # System Info
    md.append("## System Information")
    md.append("")
    import platform
    import psutil
    
    # Hardware
    md.append("### Hardware")
    md.append("")
    md.append(f"- **OS**: {platform.system()} {platform.release()} ({platform.version()})")
    md.append(f"- **Machine**: {platform.machine()}")
    md.append(f"- **Processor**: {platform.processor()}")
    md.append(f"- **CPU Cores**: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    
    # Memory
    mem = psutil.virtual_memory()
    md.append(f"- **RAM**: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
    md.append("")
    
    # Software
    md.append("### Software")
    md.append("")
    md.append(f"- **Python**: {platform.python_version()}")
    
    # Key library versions
    try:
        import transformers
        md.append(f"- **transformers**: {transformers.__version__}")
    except: pass
    
    try:
        import torch
        md.append(f"- **torch**: {torch.__version__}")
    except: pass
    
    try:
        import max
        # MAX doesn't have __version__, check for version info
        md.append(f"- **max**: (Modular MAX Engine)")
    except: pass
    
    try:
        import numpy
        md.append(f"- **numpy**: {numpy.__version__}")
    except: pass
    
    md.append("")
    
    return "\n".join(md)


def save_results(results: list[BenchmarkResult], config: dict, output_dir: Path, validation_results: dict = None):
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    machine_id = get_machine_id()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"benchmark_{machine_id}_{timestamp}" if config['benchmark'].get('timestamp_results') else "benchmark"
    
    # JSON output
    if 'json' in config['output']['formats']:
        json_file = output_dir / f"{base_name}.json"
        with open(json_file, 'w') as f:
            data = {
                'benchmark': config['benchmark'],
                'results': [asdict(r) for r in results],
                'timestamp': timestamp
            }
            json.dump(data, f, indent=config['output']['json'].get('indent', 2))
        print(f"\nResults saved to: {json_file}")
    
    # CSV output
    if 'csv' in config['output']['formats']:
        csv_file = output_dir / f"{base_name}.csv"
        import csv
        with open(csv_file, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
                if config['output']['csv'].get('include_header', True):
                    writer.writeheader()
                for result in results:
                    row = asdict(result)
                    row.pop('raw_latencies_ms', None)  # Don't include raw data in CSV
                    writer.writerow(row)
        print(f"Results saved to: {csv_file}")
    
    # Markdown output
    if 'markdown' in config['output']['formats']:
        md_file = output_dir / f"{base_name}.md"
        markdown = generate_markdown_report(results, config, validation_results or {})
        with open(md_file, 'w') as f:
            f.write(markdown)
        print(f"Markdown report saved to: {md_file}")


def main():
    parser = argparse.ArgumentParser(description="Config-driven ML inference benchmark")
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'benchmark_config.toml',
                       help='Path to TOML config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'rb') as f:
        config = tomli.load(f)
    
    # Resolve paths relative to config file location
    config_dir = args.config.parent
    
    print("="*70)
    print(config['benchmark']['name'])
    print(config['benchmark']['description'])
    print("="*70)
    
    # Load test data (resolve paths relative to config)
    test_data = load_test_data(
        config_dir / config['test_data']['benchmark_file'],
        repeat=config['test_data']['repeat'],
        categories=config['test_data'].get('categories')
    )
    
    validation_data = load_test_data(
        config_dir / config['test_data']['validation_file']
    )
    
    print(f"\nTest data loaded:")
    print(f"  Benchmark: {len(test_data)} samples")
    print(f"  Validation: {len(validation_data)} samples")
    
    # Benchmark each enabled implementation
    results = []
    all_validation_results = {}
    
    for impl_key, impl_config in config.get('implementations', {}).items():
        if not impl_config.get('enabled', False):
            print(f"\nSkipping {impl_key} (disabled)")
            continue
        
        print(f"\n{'='*70}")
        print(f"Loading: {impl_config['name']}")
        print(f"{'='*70}")
        
        # Load implementation (resolve model path relative to project root)
        load_start = time.perf_counter()
        model_path = config_dir / config['model']['path']
        model, predict_fn = ImplementationLoader.load(impl_config, str(model_path))
        load_time = time.perf_counter() - load_start
        print(f"Loaded in {load_time:.2f} seconds")
        
        # Benchmark
        result = benchmark_implementation(
            name=impl_config['name'],
            description=impl_config['description'],
            model=model,
            predict_fn=predict_fn,
            test_data=test_data,
            warmup=config['benchmark']['warmup_iterations'],
            iterations=config['benchmark']['test_iterations'],
            include_raw=config['output'].get('include_raw_data', False)
        )
        result.load_time_s = load_time
        results.append(result)
        
        # Validate
        print(f"\nValidating correctness...")
        validation_results = validate_implementation(predict_fn, validation_data)
        all_validation_results[impl_config['name']] = validation_results
        correct = sum(1 for v in validation_results if v.correct)
        print(f"  Accuracy: {correct}/{len(validation_results)} ({100*correct/len(validation_results):.1f}%)")
        
        for v in validation_results:
            status = "✓" if v.correct else "✗"
            print(f"  {status} '{v.text[:40]}...' → {v.predicted} (expected: {v.expected})")
    
    # Print results
    if 'console' in config['output']['formats']:
        print_results(results, baseline_name=config['output'].get('baseline'))
    
    # Save results (resolve output dir relative to config)
    if config['benchmark'].get('save_results'):
        output_dir = config_dir / config['benchmark']['results_dir']
        save_results(results, config, output_dir, all_validation_results)
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
