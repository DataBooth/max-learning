# DistilBERT Sentiment Analysis Example

## Overview

Demonstrates how to use a pre-trained DistilBERT model for sentiment analysis using MAX Graph. This is a real-world example showing how to deploy transformer models for production inference.

## The Problem

Given a text review, classify the sentiment as POSITIVE or NEGATIVE:

```
Input:  "This movie was fantastic!"
Output: POSITIVE (confidence: 0.9999)

Input:  "Terrible product, waste of money."
Output: NEGATIVE (confidence: 0.9998)
```

## Model Architecture

**DistilBERT** is a distilled (smaller, faster) version of BERT:
- **6 layers** (vs 12 in BERT-base)
- **66M parameters** (vs 110M in BERT-base)
- **40% smaller, 60% faster** than BERT
- **Retains 97% of BERT's performance**

Fine-tuned on sentiment analysis datasets for binary classification.

## Files

- **`distilbert_sentiment.py`** - Main example script
- **`distilbert_config.toml`** - Configuration file
- **Model**: `models/distilbert-sentiment/` (downloaded automatically)

## Running the Example

```bash
# Run with default configuration
pixi run example-distilbert

# Or directly
python examples/python/03_distilbert_sentiment/distilbert_sentiment.py

# With custom config
python examples/python/03_distilbert_sentiment/distilbert_sentiment.py --config my_config.toml

# Custom text input
python examples/python/03_distilbert_sentiment/distilbert_sentiment.py --text "Best purchase ever!"
```

## Configuration

Edit `distilbert_config.toml` to change settings:

```toml
[model]
path = "../../models/distilbert-sentiment"  # Model directory

[test_data]
texts = [
    "This movie was fantastic!",
    "Terrible product, waste of money.",
    "It was okay.",
    "Best thing I've ever bought!",
    "Complete disaster."
]
```

## How It Works

### 1. Model Loading

```python
from src.python.max_distilbert.inference import DistilBertSentimentClassifier

classifier = DistilBertSentimentClassifier(model_path)
```

The classifier:
- Loads pre-trained weights from HuggingFace format (SafeTensors)
- Builds the MAX Graph with all transformer layers
- Compiles the graph for optimised inference

### 2. Text Processing

```python
result = classifier.predict("This movie was fantastic!")
```

Behind the scenes:
1. **Tokenisation** - Convert text to token IDs
2. **Embedding** - Look up token embeddings
3. **Transformer layers** - 6 layers of self-attention and feed-forward
4. **Classification** - Project to 2 classes (POSITIVE/NEGATIVE)
5. **Softmax** - Convert to probabilities

### 3. Output

```python
{
    'label': 'POSITIVE',
    'confidence': 0.9999,
    'logits': [5.2, -5.1]
}
```

## Implementation Details

The MAX Graph implementation (`src/python/max_distilbert/`) includes:

- **Embeddings**: Token + position embeddings
- **Attention**: Multi-head self-attention mechanism
- **Feed-forward**: Two-layer MLP in each transformer block
- **Layer normalisation**: Pre-normalisation pattern
- **Classification head**: Linear projection to 2 classes

All implemented using MAX Graph operations for optimal performance.

## Performance

Compared to PyTorch (see `benchmarks/03_distilbert/`):

| Implementation | Latency | Throughput |
|----------------|---------|------------|
| **PyTorch MPS (GPU)** 🏆 | 12.3 ms | 81.4 req/s |
| **MAX Engine (CPU)** | 26.1 ms | 38.3 req/s |
| **PyTorch CPU** | 50.1 ms | 20.0 req/s |

**MAX Engine is 2× faster than PyTorch CPU** with identical accuracy (96.7%).

## Output Formats

The example demonstrates:
- **Console output** - Human-readable predictions
- **Structured data** - Label, confidence, logits
- **Batch processing** - Multiple texts at once

## Key Concepts Demonstrated

### 1. Real Model Deployment

Unlike simple examples, this shows:
- Loading pre-trained weights from HuggingFace
- Handling transformer architectures
- Production-ready inference code

### 2. MAX Graph for Transformers

Demonstrates how to implement:
- Self-attention mechanisms
- Layer normalisation
- Residual connections
- Large tensor operations

### 3. Performance Optimisation

Shows MAX advantages:
- Ahead-of-time compilation
- Hardware-optimised execution
- Minimal overhead compared to PyTorch

## Benchmarking

For detailed performance analysis:

```bash
# Full benchmark with all implementations
pixi run benchmark-distilbert

# This compares:
# - MAX Engine (CPU)
# - PyTorch CPU
# - PyTorch MPS (Apple Silicon GPU)
```

See [benchmarks/03_distilbert/README.md](../../../benchmarks/03_distilbert/README.md) for results.

## Model Download

The first time you run the example, it will download the model:

```bash
pixi run download-models
```

This downloads from HuggingFace and converts to SafeTensors format (what MAX uses).

Model size: ~268MB (distilbert-base-uncased fine-tuned)

## Troubleshooting

### Model not found

**Problem**: `models/distilbert-sentiment/` doesn't exist

**Solution**: Run model download
```bash
pixi run download-models
```

### Import errors

**Problem**: Can't import `src.python.max_distilbert`

**Solution**: Run from repository root, ensure `src/` is in Python path

### GPU not working

**Note**: MAX Engine runs on CPU. For GPU inference, use PyTorch MPS:
```python
from transformers import pipeline
model = pipeline('sentiment-analysis', model=model_path, device='mps')
```

## Comparison to Alternatives

### vs PyTorch

**MAX advantages**:
- 2× faster inference on CPU
- No training overhead
- Simpler deployment

**PyTorch advantages**:
- GPU support (MPS is 2× faster than MAX CPU)
- Larger ecosystem
- Better for prototyping

### vs ONNX Runtime

**MAX advantages**:
- Programmatic graphs (no file conversion)
- More flexible

**ONNX advantages**:
- More mature
- Broader hardware support

## Next Steps

After understanding this example:

1. **Modify the model** - Try different transformer architectures
2. **Benchmark performance** - Compare against PyTorch on your hardware
3. **Production deployment** - Use MAX for serving inference at scale
4. **Explore MAX Graph** - Build custom architectures from scratch

## Resources

- **MAX Documentation**: https://docs.modular.com/max/
- **MAX Graph Tutorial**: https://llm.modular.com
- **DistilBERT Paper**: https://arxiv.org/abs/1910.01108
- **HuggingFace Model**: https://huggingface.co/distilbert-base-uncased

## Implementation Code

The full DistilBERT implementation is in:
- `src/python/max_distilbert/inference.py` - Main classifier
- `src/python/max_distilbert/model.py` - Model architecture
- `src/python/max_distilbert/modules/` - Transformer components

See [src/python/max_distilbert/README.md](../../../src/python/max_distilbert/README.md) for architecture details.

## Related Examples

- **01_elementwise** - Basic MAX Graph operations
- **02_linear_layer** - Matrix operations and linear layers
- **benchmarks/03_distilbert** - Detailed performance comparison
