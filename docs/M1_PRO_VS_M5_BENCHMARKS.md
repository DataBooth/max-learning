# Apple Silicon Benchmark Comparison: M1 Pro vs M5

**Date**: 2026-01-10  
**Benchmarks**: Element-wise Operations, Linear Layer, DistilBERT Sentiment Analysis  
**Test Date**: M1 Pro & M5 both run 10 January 2026

## Executive Summary

The M5 chip shows **dramatic performance improvements** across all benchmarks compared to M1 Pro:

- **2.4-4.6× faster** for transformer inference (DistilBERT)
- **3× faster** for simple element-wise operations
- **2.2× faster** GPU performance (MPS)
- **More consistent** latency (lower coefficient of variation)

**🎯 Surprise Finding**: On M5, PyTorch CPU performance (9.84ms) nearly matches MAX Engine (9.70ms), whereas on M1 Pro, MAX Engine was 1.9× faster than PyTorch CPU. This suggests significant PyTorch optimisations for newer Apple Silicon.

---

## System Specifications

| Specification | M1 Pro | M5 |
|---------------|--------|-----|
| **Chip** | Apple M1 Pro | Apple M5 |
| **CPU Cores** | 8 physical, 8 logical | 10 physical, 10 logical |
| **RAM** | 16 GB | 32 GB |
| **OS** | Darwin 24.6.0 | Darwin 25.2.0 |
| **Python** | 3.13.11 | 3.13.11 |

---

## Benchmark 1: Element-wise Operations

**Workload**: Simple element-wise operations (multiply, add, ReLU) on 4-element tensor  
**Why This Matters**: Tests overhead and basic operation performance

### CPU Performance

| Metric | M1 Pro | M5 | M5 Improvement |
|--------|--------|-----|----------------|
| **Mean Latency** | 0.0432 ms | 0.0142 ms | **3.0× faster** ⚡ |
| **Median Latency** | 0.00988 ms | 0.0113 ms | Similar |
| **P95 Latency** | 0.212 ms | 0.0278 ms | **7.6× better** |
| **Throughput** | 23,100 req/s | 70,500 req/s | **3.1× higher** 🚀 |
| **Consistency (CV)** | 1.74 | 0.486 | **3.6× more consistent** |

### GPU Performance

| Metric | M1 Pro | M5 | M5 Improvement |
|--------|--------|-----|----------------|
| **Mean Latency** | 1.03 ms | 0.644 ms | **1.6× faster** |
| **Median Latency** | 1.02 ms | 0.632 ms | **1.6× faster** |
| **P95 Latency** | 1.54 ms | 0.919 ms | **1.7× better** |
| **Throughput** | 967 req/s | 1,550 req/s | **1.6× higher** |
| **Consistency (CV)** | 0.368 | 0.265 | **1.4× more consistent** |

### Analysis

**CPU Dominance**: Both chips show CPU significantly faster than GPU for tiny tensors (4 elements):
- M1 Pro: CPU 24× faster than GPU
- M5: CPU 45× faster than GPU

**Why?** GPU dispatch overhead (~0.6-1ms) dominates for tiny operations. The M5's improved CPU makes this gap even more pronounced.

**Key Insight**: M5's 3× CPU speedup is impressive, but both chips confirm that GPU acceleration requires larger workloads to amortise overhead.

---

## Benchmark 2: DistilBERT Sentiment Analysis

**Workload**: Real-world transformer model (66M parameters, 6 layers)  
**Why This Matters**: Production inference workload representative of NLP applications  
**Test**: 1000 iterations, 30-sample validation, 50-sample benchmark dataset

### MAX Engine (CPU)

| Metric | M1 Pro | M5 | M5 Improvement |
|--------|--------|-----|----------------|
| **Mean Latency** | 23.6 ms | 9.70 ms | **2.4× faster** ⚡ |
| **Median Latency** | 23.1 ms | 9.56 ms | **2.4× faster** |
| **P95 Latency** | 29.9 ms | 10.9 ms | **2.7× better** |
| **P99 Latency** | 36.0 ms | 11.4 ms | **3.2× better** |
| **Throughput** | 42.3 req/s | 103 req/s | **2.4× higher** 🚀 |
| **Std Deviation** | 3.36 ms | 0.758 ms | **4.4× more consistent** |
| **Load Time** | 1.43 s | 1.07 s | 25% faster |
| **Accuracy** | 96.7% (29/30) | 96.7% (29/30) | Identical ✓ |

### PyTorch CPU

| Metric | M1 Pro | M5 | M5 Improvement |
|--------|--------|-----|----------------|
| **Mean Latency** | 44.8 ms | 9.84 ms | **4.6× faster** ⚡⚡ |
| **Median Latency** | 42.4 ms | 10.3 ms | **4.1× faster** |
| **P95 Latency** | 69.8 ms | 11.1 ms | **6.3× better** |
| **P99 Latency** | 94.6 ms | 11.6 ms | **8.2× better** |
| **Throughput** | 22.3 req/s | 102 req/s | **4.6× higher** 🚀🚀 |
| **Std Deviation** | 15.1 ms | 1.38 ms | **10.9× more consistent** |
| **Accuracy** | 96.7% (29/30) | 96.7% (29/30) | Identical ✓ |

### PyTorch MPS (GPU)

| Metric | M1 Pro | M5 | M5 Improvement |
|--------|--------|-----|----------------|
| **Mean Latency** | 7.97 ms | 3.59 ms | **2.2× faster** ⚡ |
| **Median Latency** | 7.57 ms | 3.56 ms | **2.1× faster** |
| **P95 Latency** | 10.8 ms | 3.83 ms | **2.8× better** |
| **P99 Latency** | 13.0 ms | 4.91 ms | **2.6× better** |
| **Throughput** | 125 req/s | 279 req/s | **2.2× higher** 🚀 |
| **Std Deviation** | 1.34 ms | 0.408 ms | **3.3× more consistent** |
| **Load Time** | 0.352 s | 0.189 s | 46% faster |
| **Accuracy** | 96.7% (29/30) | 96.7% (29/30) | Identical ✓ |

---

## Performance Rankings

### M1 Pro: Speed Comparison

| Rank | Implementation | Mean Latency | Throughput | Relative Speed |
|------|----------------|--------------|------------|----------------|
| 🥇 | PyTorch MPS (GPU) | 7.97 ms | 125 req/s | Baseline (fastest) |
| 🥈 | MAX Engine (CPU) | 23.6 ms | 42.3 req/s | 3.0× slower |
| 🥉 | PyTorch CPU | 44.8 ms | 22.3 req/s | 5.6× slower |

**M1 Pro Key Insight**: MAX Engine provides significant CPU advantage (1.9× faster than PyTorch CPU)

### M5: Speed Comparison

| Rank | Implementation | Mean Latency | Throughput | Relative Speed |
|------|----------------|--------------|------------|----------------|
| 🥇 | PyTorch MPS (GPU) | 3.59 ms | 279 req/s | Baseline (fastest) |
| 🥈 | MAX Engine (CPU) | 9.70 ms | 103 req/s | 2.7× slower |
| 🥉 | PyTorch CPU | 9.84 ms | 102 req/s | 2.7× slower |

**M5 Key Insight**: MAX Engine and PyTorch CPU are now **virtually identical** (9.70ms vs 9.84ms)!

---

## Cross-Generation Analysis

### PyTorch CPU Performance Evolution

The most surprising finding is PyTorch CPU's massive improvement on M5:

| Implementation | M1 Pro | M5 | Improvement |
|----------------|--------|-----|-------------|
| PyTorch CPU | 44.8 ms | 9.84 ms | **4.6× faster** ⚡⚡ |
| MAX Engine | 23.6 ms | 9.70 ms | **2.4× faster** ⚡ |

**Analysis**: PyTorch CPU improved **nearly 2× more** than MAX Engine between M1 Pro and M5. This suggests:

1. **PyTorch optimisations**: Recent PyTorch releases may include M-series specific optimisations
2. **M5 architecture**: The M5's CPU architecture may better suit PyTorch's execution patterns
3. **MAX Engine already optimised**: MAX Engine was already highly optimised on M1 Pro, leaving less room for improvement

### GPU (MPS) Consistency

GPU performance scaled consistently across generations:

| Metric | M1 Pro MPS | M5 MPS | Improvement |
|--------|------------|--------|-------------|
| Latency | 7.97 ms | 3.59 ms | **2.2× faster** |
| Throughput | 125 req/s | 279 req/s | **2.2× higher** |

The 2.2× improvement is more predictable than CPU's variable gains (2.4-4.6×).

---

## Hardware Architecture Impact

### M5 Advantages

**CPU Improvements**:
- 10 cores vs 8 cores (25% more)
- But delivering 2.4-4.6× speedup (far beyond core count)
- Suggests: architectural improvements, better cache, improved instruction pipeline

**GPU Improvements**:
- 2.2× faster for transformer workload
- 1.6× faster for simple operations
- More consistent performance (lower CV)

**Memory**:
- 32 GB vs 16 GB (2× more)
- Larger models and batch sizes possible
- Better for concurrent workloads

### Consistency Analysis

M5 shows dramatically lower variance across all metrics:

| Workload | Metric | M1 Pro CV | M5 CV | Improvement |
|----------|--------|-----------|-------|-------------|
| Element-wise | CPU | 1.74 | 0.486 | 3.6× more consistent |
| DistilBERT | MAX CPU | 0.142 | 0.078 | 1.8× more consistent |
| DistilBERT | PyTorch CPU | 0.337 | 0.140 | 2.4× more consistent |
| DistilBERT | PyTorch MPS | 0.168 | 0.114 | 1.5× more consistent |

**Why This Matters**: Lower variance = more predictable performance = better for production systems with SLA requirements.

---

## Practical Implications

### For Production Inference

**M5 Changes the Game**:
- **PyTorch CPU is now viable**: At 102 req/s (vs 22.3 on M1 Pro), CPU-only deployment is practical
- **MAX Engine advantage narrowed**: Was 1.9× faster, now only 1% faster than PyTorch CPU
- **GPU still fastest**: 279 req/s makes MPS the clear winner for latency-critical applications

### When to Use Each Implementation (M5)

**PyTorch MPS (GPU)** - Best for:
- Single inference latency: **3.59 ms** 🏆
- Maximum throughput: **279 req/s**
- When model fits in GPU memory
- Latency-critical applications

**MAX Engine (CPU)** - Best for:
- CPU-only deployments
- Framework portability needs
- Slightly better than PyTorch CPU (1% faster)
- More predictable compilation/optimisation

**PyTorch CPU** - Best for:
- Rapid prototyping (no compilation)
- Dynamic models
- Nearly identical performance to MAX (9.84ms vs 9.70ms)
- Largest ecosystem support

### ROI Analysis

**Upgrading M1 Pro → M5**:
- **2.4-4.6× faster inference**: Directly reduces compute costs
- **2.2× GPU throughput**: Handles 2.2× more concurrent requests
- **4× better tail latency**: P99 improved from 94.6ms → 11.6ms (PyTorch CPU)
- **More consistent**: Lower CV means fewer outliers, better user experience

For production ML inference workloads, the M5 delivers **exceptional ROI**.

---

## Technical Observations

### 1. GPU Overhead Break-Even

Element-wise operations (4 elements) still show CPU faster than GPU on both chips:
- M1 Pro: GPU 24× slower than CPU
- M5: GPU 45× slower than CPU

**Conclusion**: GPU dispatch overhead (~0.6-1ms) requires larger tensors to amortise. For DistilBERT (large model), GPU becomes advantageous.

### 2. Model Compilation Time

MAX Engine load times improved modestly:
- M1 Pro: 1.43s
- M5: 1.07s (25% faster)

PyTorch MPS load times improved significantly:
- M1 Pro: 0.352s
- M5: 0.189s (46% faster)

**Implication**: M5's faster compilation benefits quick iteration cycles.

### 3. Accuracy Consistency

All implementations maintain **identical 96.7% accuracy** across both chips. The same misclassification occurs: "It was okay." → POSITIVE (expected NEGATIVE).

**Conclusion**: Performance improvements don't compromise correctness.

---

## Methodology

### Benchmark Configuration

**Element-wise Operations**:
- Tensor size: 4 elements
- Warmup: 100 iterations
- Test: 1000 iterations
- Operations: multiply, add, ReLU

**DistilBERT Sentiment Analysis**:
- Model: 66M parameters, 6 transformer layers
- Warmup: 100 iterations
- Test: 1000 iterations
- Benchmark data: 50 samples × 20 repeats = 1000 samples
- Validation: 30 samples for correctness testing

**Precision**:
- Latency: 3 significant figures
- Throughput: 3 significant figures
- Consistency (CV): 3 significant figures

### Environment

Both benchmarks run with:
- Same Python version (3.13.11)
- Same MAX Engine version
- Same PyTorch/Transformers versions
- Same benchmark code (config-driven)

---

## Recommendations

### For M1 Pro Users

1. **MAX Engine** provides the best CPU performance (1.9× faster than PyTorch CPU)
2. **PyTorch MPS** for absolute best performance (3× faster than MAX Engine)
3. **Consider upgrading** to M5 for production workloads (4.6× improvement)

### For M5 Users

1. **PyTorch MPS** is the clear winner (2.7× faster than CPU options)
2. **MAX Engine vs PyTorch CPU**: Nearly identical now, choose based on:
   - MAX: Framework portability, slightly lower variance
   - PyTorch: Ecosystem, dynamic models, no compilation
3. **CPU options now viable**: 100+ req/s makes CPU-only deployment practical

### For Future Development

The convergence of MAX Engine and PyTorch CPU performance on M5 suggests:
- PyTorch has caught up with framework-specific optimisations
- M5 architecture benefits both frameworks
- GPU (MPS) remains the performance frontier
- Focus optimisations on GPU kernels for maximum impact

---

## Conclusions

1. **M5 is a game-changer**: 2.4-4.6× faster for transformer inference
2. **PyTorch CPU caught up**: Now matches MAX Engine on M5 (was 1.9× slower on M1 Pro)
3. **GPU still king**: 2.2× faster than CPU options, with excellent consistency
4. **Production ready**: M5 makes Apple Silicon viable for serious ML inference workloads

The M5's dramatic improvements across CPU and GPU make it an excellent choice for ML development and production inference, with PyTorch MPS GPU delivering **279 requests/second** for DistilBERT sentiment analysis.

---

## Appendix: Raw Data

### Element-wise Operations - M1 Pro vs M5

**CPU Results**:
```
                M1 Pro          M5          Improvement
Mean:           0.0432 ms       0.0142 ms   3.0× faster
Median:         0.00988 ms      0.0113 ms   Similar
P95:            0.212 ms        0.0278 ms   7.6× better
P99:            0.372 ms        0.0379 ms   9.8× better
Throughput:     23,100 req/s    70,500 req/s 3.1× higher
CV:             1.74            0.486       3.6× more consistent
```

**GPU Results**:
```
                M1 Pro          M5          Improvement
Mean:           1.03 ms         0.644 ms    1.6× faster
Median:         1.02 ms         0.632 ms    1.6× faster
P95:            1.54 ms         0.919 ms    1.7× better
P99:            1.66 ms         0.998 ms    1.7× better
Throughput:     967 req/s       1,550 req/s 1.6× higher
CV:             0.368           0.265       1.4× more consistent
```

### DistilBERT - M1 Pro vs M5

**MAX Engine**:
```
                M1 Pro          M5          Improvement
Mean:           23.6 ms         9.70 ms     2.4× faster
P95:            29.9 ms         10.9 ms     2.7× better
P99:            36.0 ms         11.4 ms     3.2× better
Throughput:     42.3 req/s      103 req/s   2.4× higher
```

**PyTorch CPU**:
```
                M1 Pro          M5          Improvement
Mean:           44.8 ms         9.84 ms     4.6× faster
P95:            69.8 ms         11.1 ms     6.3× better
P99:            94.6 ms         11.6 ms     8.2× better
Throughput:     22.3 req/s      102 req/s   4.6× higher
```

**PyTorch MPS (GPU)**:
```
                M1 Pro          M5          Improvement
Mean:           7.97 ms         3.59 ms     2.2× faster
P95:            10.8 ms         3.83 ms     2.8× better
P99:            13.0 ms         4.91 ms     2.6× better
Throughput:     125 req/s       279 req/s   2.2× higher
```

---

**Generated**: 2026-01-10  
**Repository**: https://github.com/DataBooth/max-learning  
**Benchmarks**: `benchmarks/01_elementwise/`, `benchmarks/03_distilbert/`
