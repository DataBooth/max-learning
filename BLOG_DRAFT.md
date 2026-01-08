# Building a Sentiment Classifier with Mojo and MAX Engine: A Journey

*A technical exploration of migrating from Python to Mojo for ML inference*

## Project Goal

Build a high-performance sentiment analysis service using Mojo and Modular's MAX Engine, starting from a simple lexicon-based classifier and evolving to a transformer-based model.

---

## Chapter 1: The Initial Success - Lexicon-Based MVP (v0.1.0)

### What We Built
- Simple sentiment classifier using word-based lexicons
- MVP lexicon: 29 curated words with sentiment scores
- AFINN-111 lexicon: 2,476 words for production use
- Pure Mojo implementation with zero Python dependencies

### Key Technical Decisions
- **External lexicon files** over hardcoded dictionaries for maintainability
- **Dict[String, Float32]** for O(1) word lookup performance
- **Averaged sentiment scores** across all words in text

### Lessons Learnt
1. **Mojo's String handling** - StringSlice from file reading requires conversion to String for Dict keys
2. **Logger API differences** - `log.warn()` doesn't exist; use `log.info()` instead
3. **Git workflow** - Established clean feature branch → PR → merge to main workflow

### Performance
- Fast startup, minimal memory footprint
- Perfect for rule-based sentiment where lexicons are sufficient

---

## Chapter 2: The Pivot - Discovering MAX's Actual Purpose (v0.2.0)

### The Original Plan
Integrate MAX Engine to use pre-trained DistilBERT (distilbert-base-uncased-finetuned-sst-2-english) for advanced sentiment analysis.

### What We Discovered

#### Phase 1: Infrastructure Setup ✅
- Downloaded DistilBERT from HuggingFace
- Converted to ONNX format (255MB model.onnx.data + 804KB model.onnx)
- Added MAX Engine to dependencies via pixi

#### Phase 2: The ONNX Assumption ❌
**Initial approach**: Load ONNX file directly with MAX Engine Python API
```python
# What we tried:
session = InferenceSession.from_onnx("model.onnx")  # ❌ This doesn't exist
```

**Reality**: MAX Engine doesn't load ONNX files directly. Instead:
- `InferenceSession` loads **MAX Graph** objects (built programmatically)
- ONNX support exists, but requires custom op extensions for unsupported operations

#### Phase 3: The Mojo API Wall ❌
**Attempt**: Use Mojo MAX Engine API
```mojo
from max import engine  // ❌ unable to locate module 'max'
from tensor import Tensor  // ❌ package 'tensor' does not contain 'Tensor'
```

**Discovery**: Mojo MAX Engine API is "still in development and subject to change" per documentation. Not production-ready yet.

### The Fundamental Misunderstanding

**What we thought MAX was**: A drop-in ONNX inference runtime like ONNX Runtime but faster.

**What MAX actually is**: 
1. A **serving platform** for GenAI/LLM models with OpenAI-compatible endpoints
2. A **graph compiler** and **runtime** for models built with MAX Graph API
3. Optimised for **generative models** (Llama, Gemma, Mistral), not discriminative classifiers

### Key Insight: MAX Engine Has Three API Layers
1. **Python API** (`from max.engine import InferenceSession`) - ✅ Works, verified
2. **Mojo API** (`from max import engine`) - ❌ Not accessible/experimental
3. **C API** - For C/C++ integration

### The LayoutTensor Question
**Question**: Should we use LayoutTensor for tensor operations?

**Answer**: No. LayoutTensor is for writing custom GPU kernels with optimised memory layouts. For loading pre-trained models, MAX Engine handles tensor operations internally.

---

## Chapter 3: The Discovery - MAX Has BERT Examples!

### The Breakthrough
Found `examples/embedding-architecture/bert/` in the Modular repo showing:
- Complete BERT implementation using MAX Graph API
- Reusable components: `EmbeddingLayer`, `TransformerEncoder`, `PoolingLayer`
- Weight adapter system for loading HuggingFace checkpoints
- Pattern we can adapt for DistilBERT classification

### Why Build a Custom MAX Graph?

#### Educational Value
1. **Understand MAX's architecture** - Graph building → compilation → optimised execution
2. **Learn the Graph API** - Python API first, Mojo later
3. **See the optimisation pipeline** - How MAX fuses ops and selects kernels

#### Technical Benefits
1. **Hardware portability** - Same code runs optimised on CPU/GPU (NVIDIA/AMD)
2. **Performance** - Graph compiler automatically fuses operations
3. **No framework lock-in** - No PyTorch, TensorFlow, or ONNX Runtime dependency
4. **Path to Mojo** - Python Graph → profile → optimise critical paths in Mojo
5. **Custom kernels** - Can extend with Mojo ops via `ops.custom()`

#### The Adaptation Strategy
```
BERT (embeddings) → DistilBERT (classification)
- Use existing: EmbeddingLayer, TransformerEncoder (6 layers instead of 12)
- Add new: Classification head (linear layer for 2-class sentiment)
```

### Estimated Effort
2-4 hours to adapt the BERT example, reusing existing MAX components.

---

## Chapter 4: Building the Custom MAX Graph [CURRENT PROGRESS]

### Architecture Overview
```
Input (token IDs, attention mask)
  ↓
Embedding Layer (word + position embeddings)
  ↓
Transformer Encoder (6 layers, DistilBERT configuration)
  ↓
[CLS] Token Pooling
  ↓
Classification Head (Linear: 768 → 2 classes)
  ↓
Output (logits for POSITIVE/NEGATIVE)
```

### What We Built

Created three files in `src/max_distilbert/`:

1. **`model_config.py`** - DistilBERT configuration helpers
2. **`graph.py`** - Core MAX Graph implementation:
   - `DistilBertClassifier` module (6-layer transformer + classification head)
   - `build_graph()` function to construct the computation graph
   - Reuses Modular's BERT example components (`EmbeddingLayer`, `TransformerEncoder`)
3. **`inference.py`** - Inference wrapper and demo script

### Key Implementation Details

**Graph Construction** (`graph.py`):
```python
class DistilBertClassifier(Module):
    def __init__(...):
        # Embeddings (no token_type_ids for DistilBERT)
        self.embeddings = EmbeddingLayer(...)
        
        # 6-layer transformer encoder
        self.encoder = TransformerEncoder(...)
        
        # Classification head (768 → 2 classes)
        self.classifier_weight = weights.classifier.weight
        self.classifier_bias = weights.classifier.bias

    def __call__(self, input_ids, attention_mask):
        # Embeddings → Transformer → [CLS] token → Linear projection
        embeddings = self.embeddings(input_ids)
        encoder_output = self.encoder(embeddings, attention_mask)
        cls_token = encoder_output[:, 0, :]  # Extract [CLS]
        logits = ops.linear(cls_token, self.classifier_weight, ...)
        return logits
```

### Challenges Encountered

#### 1. Dependency Management
**Problem**: MAX pipelines module has deep dependency chain (pydantic, psutil, msgspec, pyzmq, grpcio, protobuf)

**Solution**: Use unified `modular` package instead of individual `max` + `mojo` packages:
```toml
# Before
max = ">=26.1.0..."
mojo = "*"
# Manually adding dependencies...

# After
modular = "*"  # Includes all MAX dependencies!
```

**Learning**: Modular provides a complete package that includes all necessary dependencies. Don't add MAX/Mojo separately.

#### 2. PipelineConfig Requirements
**Problem**: The BERT example components expect `PipelineConfig`, but creating one requires a HuggingFace model path and triggers full pipeline resolution.

**Current Status**: Investigating how to either:
- Create a minimal config that satisfies the embedding/encoder requirements
- Bypass PipelineConfig entirely and create standalone versions of the components
- Use a different approach that doesn't depend on the pipeline infrastructure

**Blocker**: `PipelineConfig(max_length=512)` fails with:
```
ValueError: model must be provided and must be a valid Hugging Face repository
```

### Next Steps

1. **Resolve PipelineConfig dependency** - Either:
   - Provide dummy model path to satisfy requirements
   - Create simplified config class that components accept
   - Extract and adapt embedding/encoder components to work standalone

2. **Complete inference implementation** once config issue resolved

3. **Test with sample inputs** and verify correctness

4. **Benchmark** against ONNX Runtime baseline

5. **Document performance characteristics** and learnings

---

## Key Learnings So Far

### About MAX Engine
1. **Primary use case**: Serving GenAI/LLM models, not classification models
2. **Architecture**: Graph building (Python/Mojo) → compiler optimisation → hardware-agnostic execution
3. **Mojo API maturity**: Python API is stable; Mojo API is experimental
4. **ONNX support**: Custom extensions required for unsupported ops, not plug-and-play

### About Project Evolution
1. **Start simple**: Lexicon-based MVP validated the concept quickly
2. **Research first**: Understanding MAX's actual purpose saved us from the wrong path
3. **Use examples**: Modular's BERT example provides the blueprint
4. **Iterate based on discoveries**: Pivot from "load ONNX" to "build custom graph"

### Technical Debt & Decisions
1. **Downloaded ONNX model**: Not currently used, but contains the weights we need
2. **Python first, Mojo later**: Pragmatic given Mojo API limitations
3. **Custom graph approach**: More work upfront, but better long-term path

---

## What's Next

1. Implement DistilBERT MAX Graph
2. Load pre-trained weights from HuggingFace
3. Create inference API
4. Benchmark performance vs alternatives
5. (Future) Optimise critical paths in Mojo
6. (Future) Deploy with MAX Serve

---

## Resources & References

- [MAX Documentation](https://docs.modular.com/max/)
- [Modular BERT Example](https://github.com/modular/modular/tree/main/examples/embedding-architecture/bert)
- [MAX Graph API](https://docs.modular.com/max/graph/)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)

---

## Appendix: MAX in the ML Ecosystem - What Does It Actually Compete With?

### Inference Runtimes

**ONNX Runtime**
- **Shared**: Both optimise ML inference
- **ONNX Runtime**: File-based (.onnx models), framework-agnostic deployment, primarily CPU/NVIDIA GPU via CUDA
- **MAX**: Programmatic graphs (Python/Mojo API), compiler-first approach, hardware-portable (CPU/NVIDIA/AMD/Apple)
- **Verdict**: Competing approaches - ONNX is "export and run", MAX is "compile and optimise"

**TensorRT (NVIDIA)**
- **Shared**: High-performance inference optimisation
- **TensorRT**: NVIDIA GPU only (CUDA-specific optimisations), deep kernel integration
- **MAX**: Multi-hardware via Mojo compilation layer, abstracts hardware via graph compiler
- **Verdict**: Direct competition for NVIDIA inference workloads, but MAX offers portability

**CoreML (Apple)**
- **Shared**: Hardware-optimised inference
- **CoreML**: Apple ecosystem only (Metal/ANE), tight iOS/macOS integration
- **MAX**: Cross-platform (including Apple when GPU support ships), server/edge/cloud focus
- **Verdict**: Complementary - CoreML for mobile apps, MAX for services

### Training Frameworks

**PyTorch / TensorFlow**
- **Shared**: Define computation graphs
- **PyTorch/TF**: Training + inference, research-friendly, framework lock-in (different APIs)
- **MAX**: Inference only (currently), production-optimised, framework-agnostic (consumes their outputs)
- **Verdict**: Not competitors - MAX *consumes* models trained in PyTorch/TF

### Compiler Projects

**TVM (Apache)**
- **Shared**: ML compilation, hardware abstraction
- **TVM**: Open source, community-driven, wide hardware support but complex to tune
- **MAX**: Commercial (Modular Inc.), integrated with Mojo, opinionated compilation with Mojo kernel library
- **Verdict**: Closest competitor - both are ML compilers, MAX has Mojo advantage

**XLA (Google)**
- **Shared**: Compiler-based optimisation
- **XLA**: TensorFlow/JAX specific, primarily Google hardware (TPU) + GPU/CPU
- **MAX**: Framework-agnostic, CPU/NVIDIA/AMD/Apple
- **Verdict**: Similar approach, different ecosystems

### MAX's Unique Position

**What MAX actually competes with**: The fragmentation itself

Instead of choosing between:
- ONNX Runtime (file-based, limited hardware)
- TensorRT (NVIDIA only)
- CoreML (Apple only)
- PyTorch (training overhead)

MAX provides: "Write your inference graph once in Python/Mojo, we'll compile it optimally for any hardware"

### The LLVM Analogy

MAX is closest to **LLVM for ML workloads**:
- **LLVM**: C/C++ → IR → optimised native code (x86/ARM/etc.)
- **MAX**: Python/Mojo graph → MAX IR → optimised execution (CPU/GPU/etc.)

**What makes MAX different**:
1. **Mojo integration**: Can drop to low-level when needed
2. **Graph compiler**: Not just JIT, full graph optimisation
3. **Hardware portability**: Same code, different hardware
4. **GenAI focus**: Pipeline registry optimised for LLMs

### For This Project

Why MAX fits our DistilBERT sentiment classifier:
- Custom architecture not in pre-built pipelines ✅
- Want Apple Silicon GPU eventually (hardware portability) ✅
- Need production performance (graph compilation) ✅
- Don't want CUDA/Metal/ROCm code (abstraction) ✅

**Alternative approaches**:
- ONNX Runtime: Export to ONNX, run, but limited Apple GPU support
- TensorRT: NVIDIA only, can't use on M3 MacBook
- PyTorch: Keep full framework overhead for inference
- CoreML: Lock into Apple ecosystem

**With MAX**: Write graph once, runs on Mac today (CPU), Apple GPU tomorrow, cloud NVIDIA/AMD later - all with optimal performance.

**The trade-off**: Less mature than ONNX Runtime, smaller community than PyTorch, requires building custom graphs. But we gain true hardware portability and excellent performance.

---

*This document is a living record of our exploration. Updates as we progress...*
