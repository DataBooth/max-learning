# DistilBERT Sentiment Analysis Benchmarks

Benchmarks comparing MAX Graph vs PyTorch (HuggingFace) performance for DistilBERT sentiment classification.

## Benchmarks

### max_vs_pytorch.py

**Purpose**: Compare MAX Graph custom implementation against HuggingFace PyTorch for production inference workloads.

**Model**: DistilBERT fine-tuned for binary sentiment classification (positive/negative)

**What it measures**:
- Mean inference latency
- Median latency
- P95/P99 latency (tail latency)
- Throughput (requests/sec)
- Consistency (coefficient of variation)
- Prediction accuracy on validation set

**Configuration**:
- Warmup: 10 iterations
- Benchmark: 100 iterations
- Test dataset: `test_data/sentiment_validation.jsonl`

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

From M1 Pro testing (100 iterations):

| Metric | MAX | PyTorch | Improvement |
|--------|-----|---------|-------------|
| **Mean latency** | 45.88 ms | 255.85 ms | **5.58x faster** |
| **Throughput** | 21.80 req/sec | 3.91 req/sec | **5.58x higher** |
| **P95 latency** | 67.61 ms | 451.75 ms | **85% better** |
| **Consistency (CV)** | 0.34 | 2.71 | **8x more consistent** |
| **Accuracy** | 80% | 80% | **Identical** |

### Why MAX is Faster

1. **Ahead-of-time compilation**: Graph is compiled once, optimised for target hardware
2. **Minimal overhead**: No Python interpreter overhead during inference
3. **Memory efficiency**: Optimised tensor memory layout
4. **Hardware-specific optimisations**: Tailored to M1 architecture

### Trade-offs

**MAX advantages**:
- ✅ Significantly faster inference
- ✅ Lower tail latency (more predictable)
- ✅ Better for production serving
- ✅ Framework portability (not locked to PyTorch)

**PyTorch advantages**:
- ✅ Faster prototyping (no compilation step)
- ✅ Larger ecosystem
- ✅ More model availability
- ✅ Dynamic computation graphs

### When to Use MAX

**Good fit**:
- Production inference services
- High-throughput batch processing
- Latency-sensitive applications
- Need for hardware portability
- Long-running services (compilation overhead amortised)

**Less ideal**:
- One-off predictions
- Rapid prototyping
- Frequently changing models
- Research experimentation

---

## Benchmark Details

### System Information

The benchmark automatically captures:
- Hardware: CPU model, cores, RAM
- Software: Python, MAX, PyTorch, transformers versions
- OS: Operating system and architecture
- Timestamp: When benchmark was run

### Test Data

Uses validation dataset with known ground truth labels:
- Location: `test_data/sentiment_validation.jsonl`
- Format: JSONL (one JSON object per line)
- Fields: `text`, `label` (0=negative, 1=positive)

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
