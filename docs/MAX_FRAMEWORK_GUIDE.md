# MAX Framework Guide

## What is MAX?

**MAX (Modular Accelerated Xecution)** is a graph compiler and runtime system for ML inference developed by Modular. It's designed to solve the ML infrastructure fragmentation problem by providing hardware-portable, high-performance inference across different accelerators.

### The Problem MAX Solves

Traditional ML deployment requires:
- **NVIDIA GPUs**: Write CUDA code, use PyTorch + CUDA runtime
- **AMD GPUs**: Port to ROCm or accept slower performance  
- **Apple Silicon**: Rewrite in Metal or fall back to CPU
- Each handoff introduces complexity and vendor lock-in

### MAX's Solution

**Hardware-Portable Graph Compiler**:
1. You define **WHAT** to compute (computation graph in Python/Mojo)
2. MAX compiler optimizes **HOW** to compute (fuses ops, selects kernels)
3. Runs optimized on **WHERE** (CPU, NVIDIA GPU, AMD GPU, Apple GPU†)

† Apple GPU support coming to Python/Graph APIs

Think of MAX as **LLVM for ML workloads**.

## Key Concepts

### 1. Graph-Based Computation

MAX uses a declarative graph API:

```python
from max.graph import Graph, TensorType, ops
from max.dtype import DType

# Define computation graph
with Graph("my_model", input_types=[TensorType(DType.float32, shape=[1, 768])]) as graph:
    x = graph.inputs[0].tensor
    
    # Operations
    y = ops.matmul(x, weight)
    y = ops.add(y, bias)
    y = ops.relu(y)
    
    graph.output(y)
```

Key characteristics:
- **Declarative**: Describe computation, not execution
- **Device-agnostic**: No CUDA/Metal/ROCm code
- **Optimizable**: Compiler fuses operations, optimizes memory layout

### 2. Weight Management

MAX separates weights from graph definition:

```python
from max.graph.weights import load_weights

# Load weights (auto-detects format: safetensors, pytorch, gguf)
weights = load_weights([model_path])

# Allocate in graph with proper dtype
weight_tensor = weights.layer.weight.allocate(DType.float32).cast(target_dtype)
```

Benefits:
- Hot-reload weights without recompiling graph
- Multiple weight formats supported
- Memory-efficient loading

### 3. Compilation and Execution

```python
from max.engine import InferenceSession

# Create session
session = InferenceSession(devices=[device])

# Compile and load graph (happens once)
model = session.load(graph, weights_registry=weights.allocated_weights)

# Execute (fast path)
outputs = model.execute(input_tensor)
```

**Compilation cost**: Paid once upfront, amortized over many inferences.

## MAX APIs

### Python Graph API (Production-Ready)
- Build graphs in Python
- Access to all MAX ops
- This is what we use in this project
- Best for: Custom models, production deployments

### Python Pipeline API  
- Pre-built pipelines for common models (Llama, Gemma, Mistral)
- OpenAI-compatible serving with MAX Serve
- Best for: GenAI/LLM serving, quick deployment

### Mojo API (Experimental)
- Write graphs in Mojo (MAX's systems programming language)
- Lower-level control
- GPU kernel development
- Best for: Custom kernels, maximum performance

## Our DistilBERT Implementation

### Architecture

```
┌─────────────────────────────────────────┐
│  Input: Token IDs + Attention Mask      │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  DistilBertEmbeddings                   │
│  - Word embeddings (vocab → 768)        │
│  - Position embeddings (0-511 → 768)    │
│  - LayerNorm + Dropout                  │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  DistilBertTransformerEncoder (6 layers)│
│                                          │
│  Each layer:                             │
│    - Multi-head attention (12 heads)    │
│    - Layer norm                          │
│    - Feed-forward network (768→3072→768)│
│    - Layer norm                          │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  Classification Head                     │
│  - Extract [CLS] token                   │
│  - Pre-classifier (768 → 768) + ReLU    │
│  - Classifier (768 → 2)                  │
└──────────────┬──────────────────────────┘
               ↓
        Logits (NEGATIVE/POSITIVE)
```

### Key Implementation Files

```
src/max_distilbert/
├── embeddings.py       - Custom embeddings (no token types)
├── transformer.py      - DistilBERT-specific attention & FFN
├── graph.py           - Graph building + classification head
├── inference.py       - High-level inference wrapper
└── model_config.py    - Configuration helpers
```

### Why Custom Implementation?

1. **DistilBERT not in MAX Pipeline Registry**: Registry focuses on GenAI/LLMs
2. **Learning opportunity**: Understand MAX Graph API deeply
3. **Customization**: Can modify architecture for specific needs
4. **Performance**: Direct control over graph construction

### Implementation Highlights

#### Custom Embeddings (No Token Types)
```python
class DistilBertEmbeddings(Module):
    def __call__(self, input_ids):
        # Word embeddings
        embeddings = ops.gather(self.word_embeddings, input_ids, axis=0)
        
        # Position embeddings (no token type!)
        position_ids = ops.range(0, seq_length, 1, dtype=DType.int64, device=device)
        position_embeddings = ops.gather(self.position_embeddings, position_ids, axis=0)
        
        # Combine and normalize
        embeddings = embeddings + position_embeddings
        return ops.layer_norm(embeddings, self.LayerNorm_weight, self.LayerNorm_bias, epsilon)
```

#### DistilBERT-Specific Attention
```python
# DistilBERT uses q_lin, k_lin, v_lin (not query, key, value like BERT)
query = ops.matmul(hidden_states, ops.transpose(self.q_lin_weight, 1, 0)) + self.q_lin_bias
key = ops.matmul(hidden_states, ops.transpose(self.k_lin_weight, 1, 0)) + self.k_lin_bias
value = ops.matmul(hidden_states, ops.transpose(self.v_lin_weight, 1, 0)) + self.v_lin_bias
```

#### Pre-Classifier Layer (Critical!)
```python
# DistilBERT sequence classification has this extra layer
pooled_output = ops.matmul(cls_output, ops.transpose(self.pre_classifier_weight, 1, 0))
pooled_output = ops.relu(pooled_output)
logits = ops.matmul(pooled_output, ops.transpose(self.classifier_weight, 1, 0))
```

## Performance Characteristics

### Compilation Overhead
- **One-time cost**: ~2-3 seconds for DistilBERT
- **Amortized quickly**: After ~50 inferences on our benchmark
- **Worth it for**: Production services, batch processing

### Runtime Performance
On Apple M3 CPU (100 iterations):
- **MAX**: 45.88ms mean latency (21.80 req/sec)
- **PyTorch**: 255.85ms mean latency (3.91 req/sec)
- **Speedup**: 5.58x faster!

Key advantages:
- **Lower latency**: 5-6x faster mean
- **More consistent**: 8x lower variance
- **Better tail latency**: 85% better P95

### When to Use MAX

**Good fit**:
- Production inference services
- High-throughput batch processing
- Need hardware portability
- Want to avoid framework lock-in

**Less ideal**:
- One-off predictions
- Rapid prototyping (compilation overhead)
- Model still changing frequently

## Common Patterns

### Linear Layer Pattern
```python
# MAX doesn't have ops.linear(), use matmul + bias
output = ops.matmul(input, ops.transpose(weight, 1, 0)) + bias
```

### Attention Mask Pattern
```python
# Convert 1/0 mask to additive mask for attention scores
ones = ops.constant(1.0, dtype=mask.dtype, device=mask.device)
inverted_mask = ones - mask
additive_mask = inverted_mask * ops.constant(-10000.0, dtype=mask.dtype, device=mask.device)
attention_scores = attention_scores + additive_mask
```

### Multi-Head Attention Reshape
```python
# Use permute for 4D tensors, transpose for 2D
query = ops.reshape(query, [batch, seq_len, num_heads, head_size])
query = ops.permute(query, [0, 2, 1, 3])  # → [batch, num_heads, seq_len, head_size]
```

## Debugging Tips

1. **Weight loading errors**: Check weight names with `safetensors`
   ```python
   from safetensors import safe_open
   tensors = safe_open(path, framework='pt')
   print(list(tensors.keys()))
   ```

2. **Shape mismatches**: Print shapes during graph building
   ```python
   print(f"Tensor shape: {tensor.shape}")
   ```

3. **Op signature errors**: Check MAX docs for correct parameter order
   - Most ops use positional args, not kwargs
   - Example: `ops.layer_norm(x, weight, bias, epsilon)` not `eps=`

4. **Device issues**: Ensure all ops use same device
   ```python
   device = input_ids.device
   const = ops.constant(value, dtype=dtype, device=device)
   ```

## Resources

- **MAX Documentation**: https://docs.modular.com/max/
- **MAX Graph API**: https://docs.modular.com/max/graph/
- **Build LLM from Scratch**: https://llm.modular.com
- **Modular Forums**: https://forum.modular.com
- **GitHub Issues**: https://github.com/modular/modular/issues

## Next Steps

After mastering this example:
1. Try other architectures (BERT, RoBERTa, etc.)
2. Explore MAX Pipeline API for LLMs
3. Experiment with custom ops via `ops.custom()`
4. Learn Mojo for custom kernel development
5. Deploy with MAX Serve for production inference
