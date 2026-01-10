# Linear Layer Benchmarks

Benchmarks comparing CPU vs GPU performance for linear layer operations (matrix multiplication + bias + ReLU) in MAX Graph.

## Benchmarks

### cpu_vs_gpu.py

**Purpose**: Compare CPU and GPU performance for a typical neural network linear layer.

**Operations tested**:
- `ops.matmul` - Matrix multiplication
- `ops.add` - Bias addition
- `ops.relu` - ReLU activation

**What it measures**:
- Mean latency
- Median latency
- P95/P99 latency (tail latency)
- Throughput (requests/sec)
- Consistency (standard deviation)

**Configuration**:
- Input: Batch size 32, 128 input features → 64 output features
- Warmup: 100 iterations
- Benchmark: 1000 iterations
- Configurable via `benchmark_config.toml`

**Run**:
```bash
pixi run benchmark-linear
# or
python benchmarks/02_linear_layer/cpu_vs_gpu.py
```

**Output formats**:
- Console (summary)
- JSON (structured data)
- CSV (tabular data)
- Markdown (report)

Results saved to `results/cpu_vs_gpu_MACHINE_YYYYMMDD_HHMMSS.*`

**Expected outcome**: GPU benchmark currently fails on Apple Silicon due to missing matmul kernel. CPU benchmarks complete successfully.

---

## Key Findings

From M1 Pro testing:

**CPU Performance**:
- Successfully benchmarks linear layer operations
- Typical latency for batch_size=32, 128→64 layer
- Demonstrates CPU baseline for comparison

**GPU Status**:
- ⚠️ Matmul kernel not yet available for Apple Silicon GPU
- Benchmark fails gracefully with informative error message
- This is a known limitation documented in the codebase

**Why GPU fails**:
- MAX Graph's GPU kernel support for Apple Silicon is still evolving
- Matrix multiplication (`ops.matmul`) kernel not yet implemented for Metal
- Works fine on CPU; GPU support coming in future MAX releases

---

## Configuration

The benchmark is fully configurable via `benchmark_config.toml`:

**Adjustable parameters**:
- `warmup_iterations` - Number of warmup iterations (default: 100)
- `test_iterations` - Number of benchmark iterations (default: 1000)
- `batch_size` - Input batch size (default: 32)
- `input_features` - Number of input features (default: 128)
- `output_features` - Number of output features (default: 64)
- `cpu_enabled` / `gpu_enabled` - Toggle CPU/GPU benchmarks

**Precision settings**:
- `latency_sigfigs` - Significant figures for latency (default: 3)
- `throughput_sigfigs` - Significant figures for throughput (default: 3)
- `ratio_sigfigs` - Significant figures for ratios/CV (default: 3)

---

## Features

- ✅ **Progress bars** - Visual feedback with tqdm showing warmup/benchmark progress
- ✅ **Configurable precision** - Results formatted with appropriate significant figures
- ✅ **Multiple output formats** - Markdown, JSON, CSV for different use cases
- ✅ **Graceful failure handling** - GPU failures don't crash the benchmark
- ✅ **Machine identification** - Results include machine ID in filename

---

## See Also

- [Apple Silicon GPU Findings](../../docs/APPLE_SILICON_GPU_FINDINGS.md) - GPU limitations and findings
- [Element-wise benchmarks](../01_elementwise/) - Simpler operations showing GPU overhead
- [DistilBERT benchmarks](../03_distilbert/) - Real-world model comparison
