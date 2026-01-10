"""
Shared utilities for benchmarks
================================

Common functions for benchmark configuration, reporting, and error handling.
"""

import csv
import json
import platform
import psutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def get_gpu_info() -> str:
    """Get GPU information (macOS specific for now)."""
    try:
        if platform.system() == "Darwin":  # macOS
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Extract chipset info
            for line in result.stdout.split('\n'):
                if 'Chipset Model:' in line or 'Chip Model:' in line:
                    return line.split(':', 1)[1].strip()
        return "Unknown"
    except Exception:
        return "Unknown"


def get_machine_id() -> str:
    """Get a short machine identifier for filenames.
    
    Returns a sanitised string like 'm1-pro', 'm2-max', etc.
    """
    gpu = get_gpu_info().lower()
    
    # Extract chip identifier from GPU info
    if 'apple' in gpu:
        # e.g. "Apple M1 Pro" -> "m1-pro"
        parts = gpu.replace('apple', '').strip().split()
        return '-'.join(parts).replace(' ', '-')
    elif 'unknown' in gpu or not gpu:
        # Fallback to machine architecture
        machine = platform.machine().lower()
        return machine
    else:
        # Generic GPU identifier
        return gpu.split()[0].lower()


def get_system_info() -> Dict[str, str]:
    """Collect system information for benchmark reports."""
    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor(),
        "gpu": get_gpu_info(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": f"{psutil.virtual_memory().total / (1024**3):.1f}",
        "python_version": platform.python_version(),
    }


REPORT_TEMPLATE = """# {benchmark_name}

**Date**: {timestamp}  
**Description**: {description}

## System Information

{system_info}

## Configuration

```toml
{config_toml}
```

## Results

### CPU

{cpu_results}

### GPU

{gpu_results}

{comparison}
"""


def generate_markdown_report(
    benchmark_name: str,
    description: str,
    config: dict,
    cpu_results: Optional[dict],
    gpu_results: Optional[dict],
    gpu_error: Optional[str] = None
) -> str:
    """
    Generate a markdown report for benchmark results using template.
    
    Args:
        benchmark_name: Name of the benchmark
        description: Brief description
        config: Configuration dictionary
        cpu_results: CPU benchmark results (or None if failed)
        gpu_results: GPU benchmark results (or None if failed)
        gpu_error: GPU error message if applicable
    
    Returns:
        Markdown formatted report string
    """
    # System info
    sys_info = get_system_info()
    system_info_lines = [f"- **{k.replace('_', ' ').title()}**: {v}" for k, v in sys_info.items()]
    system_info_str = "\n".join(system_info_lines)
    
    # Config as TOML
    config_lines = []
    for section, values in config.items():
        if isinstance(values, dict):
            config_lines.append(f"[{section}]")
            for k, v in values.items():
                if isinstance(v, str):
                    config_lines.append(f'{k} = "{v}"')
                elif isinstance(v, bool):
                    config_lines.append(f'{k} = {str(v).lower()}')
                else:
                    config_lines.append(f'{k} = {v}')
            config_lines.append("")
    config_toml_str = "\n".join(config_lines)
    
    # CPU results
    if cpu_results:
        cpu_lines = [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean Latency | {cpu_results['mean_ms']:.4f} ms |",
            f"| Median Latency | {cpu_results['median_ms']:.4f} ms |",
            f"| P95 Latency | {cpu_results['p95_ms']:.4f} ms |",
            f"| P99 Latency | {cpu_results['p99_ms']:.4f} ms |",
            f"| Std Dev | {cpu_results['std_ms']:.4f} ms |",
            f"| Throughput | {cpu_results['throughput']:.2f} req/sec |",
            f"| Iterations | {cpu_results['iterations']} |",
        ]
        cpu_results_str = "\n".join(cpu_lines)
    else:
        cpu_results_str = "❌ CPU benchmark failed"
    
    # GPU results
    if gpu_results:
        gpu_lines = [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean Latency | {gpu_results['mean_ms']:.4f} ms |",
            f"| Median Latency | {gpu_results['median_ms']:.4f} ms |",
            f"| P95 Latency | {gpu_results['p95_ms']:.4f} ms |",
            f"| P99 Latency | {gpu_results['p99_ms']:.4f} ms |",
            f"| Std Dev | {gpu_results['std_ms']:.4f} ms |",
            f"| Throughput | {gpu_results['throughput']:.2f} req/sec |",
            f"| Iterations | {gpu_results['iterations']} |",
        ]
        gpu_results_str = "\n".join(gpu_lines)
    else:
        gpu_lines = ["❌ GPU benchmark failed"]
        if gpu_error:
            gpu_lines.append("")
            gpu_lines.append(f"**Error**: {gpu_error}")
        gpu_results_str = "\n".join(gpu_lines)
    
    # Comparison
    comparison_str = ""
    if cpu_results and gpu_results:
        speedup = cpu_results["mean_ms"] / gpu_results["mean_ms"]
        cpu_cv = cpu_results["std_ms"] / cpu_results["mean_ms"]
        gpu_cv = gpu_results["std_ms"] / gpu_results["mean_ms"]
        
        comp_lines = [
            "## Comparison",
            "",
            f"- **Speedup**: {speedup:.2f}x",
            f"- **GPU is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than CPU**",
            "",
            f"- **CPU Consistency (CV)**: {cpu_cv:.4f}",
            f"- **GPU Consistency (CV)**: {gpu_cv:.4f}",
        ]
        
        if cpu_cv < gpu_cv:
            improvement = (gpu_cv / cpu_cv - 1) * 100
            comp_lines.append(f"- CPU is {improvement:.1f}% more consistent")
        else:
            improvement = (cpu_cv / gpu_cv - 1) * 100
            comp_lines.append(f"- GPU is {improvement:.1f}% more consistent")
        
        comparison_str = "\n".join(comp_lines)
    
    # Fill template
    return REPORT_TEMPLATE.format(
        benchmark_name=benchmark_name,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        description=description,
        system_info=system_info_str,
        config_toml=config_toml_str,
        cpu_results=cpu_results_str,
        gpu_results=gpu_results_str,
        comparison=comparison_str
    )


def save_markdown_report(report: str, output_dir: Path, prefix: str = "benchmark") -> Path:
    """
    Save markdown report with timestamp and machine identifier.
    
    Args:
        report: Markdown content
        output_dir: Directory to save in
        prefix: Filename prefix
    
    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    machine_id = get_machine_id()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{machine_id}_{timestamp}.md"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        f.write(report)
    
    return filepath


def save_json_report(
    data: dict,
    output_dir: Path,
    prefix: str = "benchmark",
    indent: int = 2
) -> Path:
    """
    Save benchmark data as JSON with timestamp and machine identifier.
    
    Args:
        data: Dictionary containing benchmark data
        output_dir: Directory to save in
        prefix: Filename prefix
        indent: JSON indentation level
    
    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    machine_id = get_machine_id()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{machine_id}_{timestamp}.json"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)
    
    return filepath


def save_csv_report(
    data: dict,
    output_dir: Path,
    prefix: str = "benchmark",
    include_header: bool = True
) -> Path:
    """
    Save benchmark data as CSV with timestamp and machine identifier.
    
    Args:
        data: Dictionary containing benchmark data with 'cpu' and/or 'gpu' keys
        output_dir: Directory to save in
        prefix: Filename prefix
        include_header: Whether to include CSV header
    
    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    machine_id = get_machine_id()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{machine_id}_{timestamp}.csv"
    filepath = output_dir / filename
    
    # Flatten data for CSV
    rows = []
    for device, results in data.items():
        if results:  # Skip if None/failed
            row = {'device': device}
            row.update(results)
            rows.append(row)
    
    if rows:
        with open(filepath, 'w', newline='') as f:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if include_header:
                writer.writeheader()
            writer.writerows(rows)
    
    return filepath
