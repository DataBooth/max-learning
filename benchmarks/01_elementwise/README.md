# Element-wise Operations Benchmarks

Benchmarks comparing CPU vs GPU performance for simple element-wise operations in MAX Graph.

## Benchmarks

### cpu_vs_gpu.py

**Purpose**: Compare CPU and GPU performance for element-wise operations with fixed tensor size.

**Operations tested**:
- `ops.mul` - Element-wise multiplication
- `ops.add` - Element-wise addition
- `ops.relu` - ReLU activation

**What it measures**:
- Mean latency
- Median latency
- P95/P99 latency
- Throughput (requests/sec)
- Consistency (standard deviation)

**Configuration**:
- Tensor size: 4 elements
- Warmup: 100 iterations
- Benchmark: 1000 iterations

**Run**:
```bash
pixi run benchmark-elementwise
# or
python benchmarks/01_elementwise/cpu_vs_gpu.py
```

**Expected outcome**: CPU significantly faster due to GPU dispatch overhead for tiny tensors.

---

### cpu_vs_gpu_scaling.py

**Purpose**: Test how CPU vs GPU performance scales across different tensor sizes.

**Operations tested**:
- Same as cpu_vs_gpu.py

**What it measures**:
- Performance across sizes: 4 to 8M elements
- Identifies crossover point where GPU becomes faster (if any)
- Shows scaling characteristics

**Configuration**:
- Tensor sizes: [4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 8388608]
- Warmup: 10 iterations per size
- Benchmark: 100 iterations per size

**Run**:
```bash
pixi run benchmark-elementwise-sizes
# or
python benchmarks/01_elementwise/cpu_vs_gpu_scaling.py
```

**Expected outcome**: CPU remains faster across all sizes for simple element-wise ops. GPU overhead dominates.

---

## Key Findings

From M1 Pro testing:

| Size | CPU (ms) | GPU (ms) | Speedup | Winner |
|------|----------|----------|---------|--------|
| 4 | 0.03 | 0.6 | 0.05x | CPU (19x faster) |
| 1M | 0.33 | 3.0 | 0.11x | CPU (9x faster) |
| 8M | 2.6 | 27 | 0.10x | CPU (10x faster) |

**Why GPU is slower**:
- GPU dispatch overhead (~1ms) dominates for simple operations
- Small tensors fit entirely in CPU cache
- Element-wise ops don't benefit from GPU parallelism at these sizes
- Would need much larger tensors or more complex compute patterns

## See Also

- [Apple Silicon GPU Findings](../../docs/APPLE_SILICON_GPU_FINDINGS.md) - Detailed analysis
- [Example code](../../examples/python/01_elementwise/) - Element-wise operations example
