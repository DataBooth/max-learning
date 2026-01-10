# Benchmarking Guide

This document explains the benchmarking methodology, metrics, and how to interpret results across the MAX Learning repository.

## Quick Start

```bash
# Run benchmarks
pixi run benchmark-elementwise      # Element-wise ops: CPU vs GPU
pixi run benchmark-linear           # Linear layer: CPU vs GPU
pixi run benchmark-distilbert       # DistilBERT: MAX vs PyTorch

# View results
ls benchmarks/01_elementwise/results/
ls benchmarks/02_linear_layer/results/
ls benchmarks/03_distilbert/results/
```

## Benchmark Structure

Each benchmark follows this pattern:

```
benchmarks/XX_name/
├── cpu_vs_gpu.py              # Main benchmark script
├── benchmark_config.toml      # Configuration (iterations, sizes, etc.)
├── results/                   # Timestamped markdown reports
│   └── benchmark_YYYYMMDD_HHMMSS.md
└── README.md                  # Benchmark-specific documentation
```

## Understanding Metrics

### Latency Metrics

**Mean Latency** - Average execution time across all iterations
- Most commonly used metric
- Good for typical performance
- Can be skewed by outliers

**Median Latency** - Middle value when sorted
- More robust to outliers than mean
- Better represents "typical" performance
- Use when you see high variance

**P95 Latency** - 95th percentile
- 95% of requests complete faster than this
- Important for user experience
- Useful for capacity planning

**P99 Latency** - 99th percentile
- 99% of requests complete faster than this
- Critical for tail latency analysis
- Important for SLAs in production

### Throughput

**Requests/Second** - How many operations can complete per second
- Calculated as: `1000 / mean_latency_ms`
- Higher is better
- Key metric for batch processing

### Consistency Metrics

**Standard Deviation (Std Dev)** - Absolute measure of variability
- How much latencies vary from the mean
- Same units as latency (milliseconds)
- Lower is better (more predictable)

**Coefficient of Variation (CV)** - Relative measure of consistency
- Calculated as: `std_dev / mean`
- Dimensionless (expressed as ratio or percentage)
- Allows comparison across different scales
- **Lower is better** - means more consistent performance

#### Why CV Matters

CV normalises variability, making it useful for comparing consistency across different workloads:

```
Example 1: Fast but inconsistent
Mean: 1ms, Std: 0.5ms → CV = 0.5 (50% variation)

Example 2: Slow but consistent  
Mean: 10ms, Std: 0.5ms → CV = 0.05 (5% variation)
```

Example 2 is more consistent despite having the same absolute variation. This is important because:
- **Predictability**: Lower CV means more predictable performance
- **Planning**: Easier to set timeouts and SLAs
- **User Experience**: Fewer frustrating "slow" requests

### Speedup

**Speedup Factor** - Relative performance improvement
- Calculated as: `baseline_time / optimised_time`
- > 1.0 means optimised is faster
- < 1.0 means baseline is faster
- Example: 2.5x means optimised is 2.5 times faster

## Benchmark Methodology

### 1. Warmup Phase

```python
for _ in range(warmup_iterations):
    model.execute(input_data)
```

**Why warmup?**
- JIT compilation happens on first runs
- Caches need to populate
- GPU kernels need to load
- First runs are not representative

**Typical values**: 50-100 iterations

### 2. Measurement Phase

```python
for _ in range(test_iterations):
    start = time.perf_counter()
    output = model.execute(input_data)
    end = time.perf_counter()
    times.append((end - start) * 1000)
```

**Key points:**
- Use `time.perf_counter()` for high-resolution timing
- Measure end-to-end including data transfer
- Store all measurements for statistical analysis

**Typical values**: 100-1000 iterations

### 3. Statistical Analysis

```python
mean = np.mean(times)
median = np.median(times)
p95 = np.percentile(times, 95)
p99 = np.percentile(times, 99)
std = np.std(times)
cv = std / mean
```

## Configuration

All benchmarks use TOML configuration files:

```toml
[benchmark]
warmup_iterations = 100
test_iterations = 1000

[graph]
# Graph-specific parameters
size = 4
multiplier = 2.0

[devices]
cpu_enabled = true
gpu_enabled = true
```

**Benefits:**
- No hardcoded values in code
- Easy to adjust without code changes
- Consistent format across benchmarks
- Version controlled

## Interpreting Results

### CPU vs GPU Comparison

**When GPU is slower:**
- Small workloads → dispatch overhead dominates
- Memory transfer costs exceed compute savings
- GPU kernels not optimised for operation

**When GPU is faster:**
- Large workloads → parallel processing benefits
- Compute-bound operations
- Well-optimised kernels available

**Our findings (Apple Silicon):**
- Element-wise ops: GPU works but CPU faster (small tensors)
- Linear layers: GPU blocked (no matmul kernel)
- Future: Expect GPU advantage with larger models

### MAX vs PyTorch Comparison

**MAX advantages:**
- Ahead-of-time compilation
- Graph optimisation
- Reduced framework overhead
- Better memory layout

**PyTorch advantages:**
- Mature ecosystem
- More kernel coverage
- Dynamic execution flexibility
- Extensive optimisations

**Our findings:**
- DistilBERT: 5.58x speedup with MAX on M1 CPU
- Trade-off: Compilation time vs inference speed
- Sweet spot: Production inference workloads

## Common Pitfalls

### 1. Not Warming Up
```python
# ❌ Bad: First run includes compilation
times = []
for _ in range(100):
    start = time.perf_counter()
    output = model.execute(input)
    times.append(time.perf_counter() - start)
```

```python
# ✅ Good: Separate warmup phase
for _ in range(50):
    model.execute(input)  # Warmup

times = []
for _ in range(100):
    start = time.perf_counter()
    output = model.execute(input)
    times.append(time.perf_counter() - start)
```

### 2. Too Few Iterations
- **Problem**: High variance, unreliable statistics
- **Solution**: Use 100+ iterations for stable measurements

### 3. Including Setup Time
```python
# ❌ Bad: Includes array creation
start = time.perf_counter()
input_data = np.array([1.0, 2.0, 3.0, 4.0])
output = model.execute(input_data)
elapsed = time.perf_counter() - start
```

```python
# ✅ Good: Only measure execution
input_data = np.array([1.0, 2.0, 3.0, 4.0])  # Setup
start = time.perf_counter()
output = model.execute(input_data)
elapsed = time.perf_counter() - start
```

### 4. Ignoring System Load
- Close other applications
- Disable background processes
- Run multiple times and take best
- Use CPU affinity if available

## Best Practices

### 1. Report System Information
Always include:
- Hardware (CPU, GPU, RAM)
- OS and version
- Framework versions
- Compiler flags

Our benchmarks auto-capture this in `benchmark_utils.py`.

### 2. Use Configuration Files
- Makes benchmarks reproducible
- Easy to share exact parameters
- Version control friendly
- No magic numbers in code

### 3. Generate Reports
- Timestamped for tracking
- Include configuration
- Show all key metrics
- Add context/notes

### 4. Test Error Handling
```python
try:
    result = benchmark_device("gpu", config)
except Exception as e:
    print(f"GPU failed: {e}")
    # Continue with CPU results
```

## Automated Reporting

Our benchmarks use `benchmark_utils.py` for consistent reporting:

```python
from benchmark_utils import (
    get_system_info,
    generate_markdown_report, 
    save_markdown_report
)

# Run benchmarks...
report = generate_markdown_report(
    benchmark_name="My Benchmark",
    description="What this tests",
    config=config,
    cpu_results=cpu_results,
    gpu_results=gpu_results,
    gpu_error=gpu_error
)

save_markdown_report(report, results_dir, prefix="my_benchmark")
```

**Features:**
- GPU detection (shows "Apple M1 Pro")
- Templated format
- Automatic timestamping
- System info capture
- Graceful error handling

## References

### Timing
- [Python time.perf_counter()](https://docs.python.org/3/library/time.html#time.perf_counter)
- Highest resolution timer available
- Monotonic (doesn't go backwards)
- Includes time during sleep

### Statistics
- **Mean**: Good for typical performance
- **Median**: Robust to outliers
- **Percentiles**: Understand distribution tails
- **CV**: Compare consistency across scales

### Performance Analysis
- [Systems Performance by Brendan Gregg](http://www.brendangregg.com/systems-performance-2nd-edition-book.html)
- [The Tail at Scale](https://www2.cs.duke.edu/courses/cps296.4/fall13/838-CloudPapers/dean_longtail.pdf)

## Contributing

When adding new benchmarks:
1. Follow the numbered directory structure
2. Create `benchmark_config.toml` for configuration
3. Use `benchmark_utils.py` for reporting
4. Add README explaining what's being tested
5. Document expected results and known limitations
6. Include error handling for GPU/device failures

## Questions?

- Check example benchmarks in `01_elementwise/` and `02_linear_layer/`
- See `benchmark_utils.py` for implementation details
- Review existing reports in `results/` directories
