# DistilBERT Sentiment Analysis Benchmarks

Benchmarks comparing MAX Graph vs PyTorch (HuggingFace) performance for DistilBERT sentiment classification.

Compares three implementations:
- **MAX Engine** - Custom MAX Graph implementation (CPU)
- **PyTorch CPU** - HuggingFace Transformers on CPU
- **PyTorch MPS** - HuggingFace Transformers on Apple Silicon GPU

## Benchmarks

### max_vs_pytorch.py

**Purpose**: Compare MAX Graph custom implementation against HuggingFace PyTorch (CPU and Apple Silicon GPU) for production inference workloads.

**Model**: DistilBERT fine-tuned for binary sentiment classification (positive/negative)

**Implementations tested**:
- MAX Engine (custom MAX Graph, CPU)
- HuggingFace PyTorch CPU
- HuggingFace PyTorch MPS (Apple Silicon GPU)

**What it measures**:
- Mean inference latency
- Median latency
- P95/P99 latency (tail latency)
- Throughput (requests/sec)
- Consistency (coefficient of variation)
- Prediction accuracy on validation set

**Configuration**:
- Warmup: 100 iterations
- Benchmark: 1000 iterations
- Benchmark dataset: 50 samples × 20 repeats = 1000 samples
- Validation dataset: 30 samples for correctness testing
- Configurable via `benchmark_config.toml` (enable/disable implementations)

**Run**:
```bash
pixi run benchmark-distilbert
# or
python benchmarks/03_distilbert/max_vs_pytorch.py
```

**Output formats**:
- Console (summary)
- JSON (structured data)
- CSV (tabular data)
- Markdown (report)

Results saved to `results/benchmark_YYYYMMDD_HHMMSS.*`

---

## Key Findings

From M1 Pro testing (1000 iterations):

| Implementation | Mean Latency | Throughput | P95 Latency | Accuracy |
|----------------|--------------|------------|-------------|----------|
| **PyTorch MPS (GPU)** 🏆 | 12.3 ms | 81.4 req/s | 17.7 ms | 96.7% |
| **MAX Engine (CPU)** | 26.1 ms | 38.3 req/s | 35.6 ms | 96.7% |
| **PyTorch CPU** | 50.1 ms | 20.0 req/s | 74.5 ms | 96.7% |

### Performance Analysis

**PyTorch MPS (GPU) is fastest**:
- 2.1× faster than MAX Engine
- 4.1× faster than PyTorch CPU
- Leverages Apple Silicon GPU (Metal Performance Shaders)
- Best for single-inference latency

**MAX Engine vs PyTorch CPU**:
- MAX is 1.9× faster than PyTorch CPU
- Better consistency (lower tail latency)
- Ahead-of-time compilation with hardware optimisations
- No Python interpreter overhead during inference

### Trade-offs

**PyTorch MPS (GPU)**:
- ✅ Fastest single-inference latency
- ✅ Highest throughput
- ✅ Easy to enable (just set `device="mps"`)
- ⚠️ Apple Silicon only
- ⚠️ Limited to models that fit in GPU memory

**MAX Engine**:
- ✅ 2× faster than PyTorch CPU
- ✅ Consistent performance (low tail latency)
- ✅ Framework portability (not locked to PyTorch)
- ✅ CPU-only, works anywhere
- ⚠️ Requires graph compilation step

**PyTorch CPU**:
- ✅ Fastest prototyping (no compilation)
- ✅ Largest ecosystem and model availability
- ✅ Dynamic computation graphs
- ⚠️ Slowest inference performance

### When to Use Each

**Use PyTorch MPS (GPU) when**:
- Running on Apple Silicon (M1/M2/M3)
- Need absolute lowest latency
- Model fits in GPU memory
- Single or small batch inference

**Use MAX Engine when**:
- Need CPU-only deployment
- Require consistent/predictable latency
- Want framework portability
- Production inference services on any hardware
- Long-running services (compilation overhead amortised)

**Use PyTorch CPU when**:
- Rapid prototyping and experimentation
- Frequently changing models
- Need dynamic computation graphs
- Research and development

---

## Benchmark Details

### System Information

The benchmark automatically captures:
- Hardware: CPU model, cores, RAM
- Software: Python, MAX, PyTorch, transformers versions
- OS: Operating system and architecture
- Timestamp: When benchmark was run

### Benchmark Data

Uses benchmark and validation datasets with known ground truth labels:
- Location: `benchmark_data/`
- Format: JSONL (one JSON object per line)
- Files: `sentiment_benchmark.jsonl`, `sentiment_validation.jsonl`
- Fields: `text`, `expected_label`, `category` (optional)

### Metrics Explained

- **Mean latency**: Average time per inference
- **Median latency**: Middle value (less affected by outliers)
- **P95/P99**: 95th/99th percentile (tail latency)
- **Throughput**: Inferences per second
- **CV (Coefficient of Variation)**: std_dev / mean (consistency metric, lower is better)

---

## See Also

- [MAX Framework Guide](../../docs/MAX_FRAMEWORK_GUIDE.md) - MAX concepts and patterns
- [Project Status](../../docs/PROJECT_STATUS.md) - Implementation details and learnings
- [DistilBERT implementation](../../src/python/max_distilbert/) - Source code
- [Example code](../../examples/python/03_distilbert_sentiment/) - Usage example
