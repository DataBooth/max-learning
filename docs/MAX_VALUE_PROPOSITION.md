# What Problem Does MAX Solve?

*Insights from building a custom DistilBERT sentiment classifier*

## The Traditional ML Inference Problem

**Pain Points:**
1. **Hardware Lock-in** - Different code for each platform (CUDA/ROCm/Metal)
2. **Framework Overhead** - PyTorch/TensorFlow designed for training, not optimized inference  
3. **Performance Ceiling** - Python/framework abstractions limit optimisation
4. **Manual Work** - Hand-tuning kernels expensive and platform-specific

## MAX's Solution: "Write Once, Run Optimized Everywhere"

```
Your Model (MAX Graph API - Python or Mojo)
              ↓
     MAX Graph Compiler ← The magic happens here
              ↓
   Optimized Mojo Kernels (auto-generated)
              ↓
CPU / NVIDIA GPU / AMD GPU / Apple GPU (coming)
```

### The Graph Compiler Innovation

**You define WHAT to compute (the graph):**
```python
with Graph("my_model", input_types=[...]) as graph:
    embeddings = ops.embedding(input_ids, ...)
    hidden = ops.layer_norm(embeddings, ...)
    output = ops.linear(hidden, ...)
    graph.output(output)
```

**MAX decides HOW to compute it optimally:**
1. Analyzes entire computation graph
2. Fuses operations (LayerNorm + GELU + Linear → single kernel)
3. Selects optimal Mojo kernels for target hardware
4. Plans memory layout and data movement  
5. Generates hardware-specific code

**The Result:**
- ✅ Same code runs optimized on any supported hardware
- ✅ Performance rivals hand-tuned CUDA/Metal
- ✅ No PyTorch, CUDA, or ROCm dependencies
- ✅ Extend with custom Mojo ops (not CUDA)

## What MAX Is NOT (Misconceptions We Had)

❌ **Not an ONNX drop-in**: MAX Graph is programmatic (Python/Mojo), not file-based  
❌ **Not for all models yet**: Pipeline registry focuses on GenAI/LLMs  
❌ **Not Python-free yet**: Mojo API experimental; Python API is production-ready  
❌ **Not magic**: Still requires understanding your model architecture

## Why We Built a Custom Graph

**The Situation:**
- DistilBERT classification not in MAX's pre-built pipeline registry
- Registry optimized for generative models (Llama, Gemma, Mistral)
- Classification models require custom graph approach

**What This Teaches:**
- MAX's **extensibility** - can support any architecture
- **Graph API** is the foundation (pipelines built on top)
- Custom graphs enable models not yet in registry

## Apple Silicon GPU Support

**Current Status:** Not available yet in Python/Graph APIs

**What We Saw:**
```
WARNING: accelerator_count() returns 0 on Apple devices.  
While Mojo now supports Apple GPUs, that support has not been  
enabled in MAX and Python APIs yet
```

**When It Lands:**
```python
# Today
device = CPU()

# Future (no other changes needed!)
device = Accelerator()  # Automatically uses M1 GPU
```

No Metal code, no platform-specific logic. Pure hardware abstraction.

## Our Implementation Journey

### ✅ What Worked
1. Unified `modular` package includes all dependencies
2. `load_weights()` auto-detects format (safetensors/pytorch/gguf)
3. MAX Graph API is intuitive for defining models
4. Weight loading from HuggingFace seamless

### ❌ Current Blocker
BERT's reusable components expect `type_vocab_size` (token type embeddings).  
DistilBERT doesn't have this (removed for efficiency).

**Solution:** Create DistilBERT-specific embedding layer or make BERT components more flexible.

### 🎓 Key Insights

1. **MAX is a compiler platform**, not just an inference runtime
2. **Graph building ≠ model loading** - you programmatically construct computation
3. **PipelineConfig is for pre-built pipelines** - custom graphs need minimal config
4. **Mojo kernels are the performance layer** - Graph API orchestrates them

## The Vision: ML Without Platform Lock-in

**Today's Reality:**
```
If NVIDIA GPU: Write CUDA, use PyTorch + CUDA runtime
If AMD GPU: Port to ROCm or use slower fallback
If Apple GPU: Rewrite in Metal or use CPU
```

**MAX's Future:**
```python
# Write once
graph = build_my_model(...)
session = InferenceSession(devices=[Accelerator()])
model = session.load(graph)

# Runs optimized on whatever hardware you have
# NVIDIA, AMD, Apple, future accelerators
```

## For Your Use Case

**DistilBERT Sentiment Classification:**
- ✅ MAX Graph API can express the architecture
- ✅ Weights load from HuggingFace
- 🚧 Need custom embedding layer (90% there!)
- 🎯 Once complete: Hardware-portable, optimized inference

**Next Steps:**
1. Complete DistilBERT-specific embedding layer
2. Finish graph compilation
3. Benchmark vs ONNX Runtime
4. Wait for Apple GPU support in MAX
5. Enjoy portable, fast inference

---

**Bottom Line:** MAX solves ML infrastructure fragmentation by providing a compiler that targets multiple hardware backends from a single graph definition, with performance that matches platform-specific hand-tuned code.
