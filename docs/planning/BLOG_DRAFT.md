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

## Chapter 4: Building the Custom MAX Graph [COMPLETED ✅]

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

### Challenges Encountered & Solutions

#### 1. Dependency Management ✅
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

#### 2. PipelineConfig Requirements ✅
**Problem**: The BERT example components expect `PipelineConfig`, but creating one requires a HuggingFace model path and triggers full pipeline resolution.

**Solution**: Built custom DistilBERT components from scratch instead of reusing BERT example:
- `embeddings.py` - Custom embeddings without token type support
- `transformer.py` - DistilBERT-specific attention layers
- `graph.py` - Graph builder with classification head
- Created minimal `SimpleConfig` to satisfy remaining interface requirements

**Learning**: When example code has heavy dependencies, it's often cleaner to build custom components tailored to your use case.

#### 3. Missing Pre-Classifier Layer 🐛 → ✅
**Problem**: Initial implementation had tiny logits (~0.05) instead of expected large values (>15).

**Root cause**: DistilBERT sequence classification has an extra layer we missed:
```python
# Missing layer:
[CLS] token (768) → pre_classifier (768→768) → ReLU → classifier (768→2)

# What we had:
[CLS] token (768) → classifier (768→2)  # ❌ Wrong!
```

**Solution**: Added the pre-classifier layer:
```python
self.pre_classifier_weight = weights.pre_classifier.weight.allocate(...)
self.pre_classifier_bias = weights.pre_classifier.bias.allocate(...)

# In forward pass:
pooled_output = ops.matmul(cls_output, ops.transpose(self.pre_classifier_weight, 1, 0))
pooled_output = ops.relu(pooled_output)
logits = ops.matmul(pooled_output, ops.transpose(self.classifier_weight, 1, 0))
```

**Result**: After fix, achieved **100% accuracy parity** with HuggingFace (99.99% confidence on positive examples).

#### 4. Package Import Structure ✅
**Problem**: Benchmarking harness couldn't import `max_distilbert` modules.

**Solution**: Made it a proper Python package:
```python
# src/max_distilbert/__init__.py
from .graph import build_graph
from .inference import DistilBertSentimentClassifier

# Now can import:
from max_distilbert import DistilBertSentimentClassifier
```

### Performance Results

Built comprehensive benchmarking framework with 100 iterations:

**Benchmark Results (Apple M1 CPU)**:
- **MAX Engine**: 45.88ms mean latency (21.80 req/sec)
- **HuggingFace PyTorch**: 255.85ms mean latency (3.91 req/sec)
- **Speedup**: **5.58x faster** 🚀

**Additional metrics**:
- P95 latency: 67.61ms (MAX) vs 451.75ms (PyTorch) = **85% better**
- Standard deviation: 14.95ms (MAX) vs 113.22ms (PyTorch) = **8x more consistent**
- Validation accuracy: **80%** (both implementations, identical predictions)

**Trade-offs**:
- MAX compilation overhead: ~2.3s (one-time cost, amortized after ~50 inferences)
- HuggingFace load time: 0.15s (faster cold start)

---

## Key Learnings

### Understanding "Graphs" in MAX Graph

**Why is it called a "graph"?** The term is overloaded in computing, but here it refers to a **computational graph** (or **dataflow graph**)—literally a graph data structure:

- **Nodes** = Operations (add, multiply, matmul, relu, etc.)
- **Edges** = Data flow between operations (tensors)
- **Directed Acyclic Graph (DAG)** = Operations execute in dependency order

**Simple Example**: For `y = relu(x * 2 + 1)`:
```
Input (x)
   │
   ├──> [multiply by 2] ──> intermediate
   │
   └──> [add 1] ──> intermediate2
        │
        └──> [relu] ──> Output (y)
```

This isn't:
- ❌ A chart/plot (matplotlib graph)
- ❌ Graph theory problems (shortest path, networks)
- ❌ Knowledge graphs or social graphs

It's literally a **data structure** representing computation flow.

**Why use computational graphs?**

1. **Enables optimisation**: Before execution, the compiler can:
   - Fuse operations (mul+add+relu → single kernel)
   - Eliminate redundant computations
   - Reorder operations for better memory access
   - Choose optimal hardware kernels

2. **Separates definition from execution**:
   ```python
   # Phase 1: Define the graph (what to compute)
   with Graph("model", input_types=[...]) as graph:
       y = ops.relu(ops.add(ops.mul(x, 2.0), 1.0))
       graph.output(y)
   
   # Phase 2: Compile (optimise for hardware)
   model = session.load(graph)
   
   # Phase 3: Execute (run many times, fast!)
   output = model.execute(input_data)
   ```

**Historical context**:
- **1960s-1970s**: Dataflow graphs for parallel computation
- **2015**: TensorFlow popularised "computation graphs" for deep learning
- **2016**: PyTorch introduced dynamic computational graphs
- **Now**: Most ML frameworks use graphs internally (even if hidden)

**MAX Graph vs other frameworks**:
- **MAX Graph**: Explicit, static, ahead-of-time compiled
- **TensorFlow 2.x**: Hidden graphs (via `@tf.function`)
- **PyTorch**: Dynamic graphs (built during execution)
- **ONNX**: File-based graph interchange format

When you write `with Graph(...) as graph:`, you're literally constructing a directed acyclic graph of operations that MAX then compiles and optimises.

### About MAX Engine
1. **Primary use case**: Serving GenAI/LLM models, but works great for custom architectures
2. **Architecture**: Graph building (Python/Mojo) → compiler optimisation → hardware-agnostic execution
3. **Computational graphs**: Explicit DAG representation enables powerful compiler optimisations
4. **Mojo API maturity**: Python API is production-ready; Mojo API is experimental
5. **ONNX support**: Custom extensions required for unsupported ops, not plug-and-play
6. **Performance**: Significant speedups (1.9-5.6x) with better consistency than PyTorch on CPU

### About MAX Graph API Patterns
1. **Linear layers**: No `ops.linear()`, use `ops.matmul(x, ops.transpose(W, 1, 0)) + bias`
2. **Attention masks**: Convert 1/0 masks to additive masks (`-10000.0` for masked positions)
3. **Multi-head attention**: Use `ops.permute()` for 4D tensor transposes
4. **Layer normalisation**: `ops.layer_norm(x, weight, bias, epsilon)` - epsilon is positional, not kwarg
5. **Weight loading**: `weight.allocate(DType.float32).cast(target_dtype)`
6. **Device handling**: Use `DeviceRef` for graph building, `CPU()` for sessions
7. **Module from max.nn**: Import `Module` from `max.nn`, not `max.graph`

### Critical Implementation Details
1. **Model-specific layers**: Always check the exact architecture (e.g., DistilBERT's pre-classifier layer)
2. **Weight names vary**: DistilBERT uses `q_lin`, `k_lin`, `v_lin` not `query`, `key`, `value`
3. **Token types**: DistilBERT has no token type embeddings (only word + position)
4. **Debugging**: Use `safetensors` library to inspect weight names when troubleshooting

### About Project Evolution
1. **Start simple**: Lexicon-based MVP (v0.1.0) validated the concept quickly
2. **Research first**: Understanding MAX's actual purpose saved us from wrong paths
3. **Don't reuse blindly**: Custom components can be cleaner than adapting heavy examples
4. **Iterate based on discoveries**: Pivot from "load ONNX" to "build custom graph"
5. **Build from scratch**: Writing custom embeddings/attention gave deeper understanding

### About Benchmarking
1. **Config-driven**: TOML configuration makes benchmarks reproducible and shareable
2. **Separate test data**: Keep test cases in JSON/JSONL for version control
3. **Multiple formats**: Generate console, JSON, CSV, and markdown reports
4. **System info matters**: Record hardware, OS, and library versions for context
5. **Warmup iterations**: Account for compilation/JIT overhead (we used 10)

### Technical Decisions & Trade-offs
1. **Python first, Mojo later**: Pragmatic given Mojo API maturity
2. **Custom graph approach**: More work upfront, but better control and performance
3. **Compilation overhead**: 2-3s one-time cost is acceptable for production inference
4. **No ONNX**: Weights from HuggingFace safetensors, not ONNX export

---

## Chapter 5: Documentation & Knowledge Sharing

### What We Created

1. **MAX Framework Guide** (`docs/MAX_FRAMEWORK_GUIDE.md`)
   - What MAX is and the problem it solves (ML infrastructure fragmentation)
   - Key concepts: graphs, weights, compilation
   - MAX APIs: Python Graph (production), Pipeline (LLMs), Mojo (experimental)
   - Our DistilBERT implementation explained
   - Common patterns and debugging tips
   - Performance characteristics and when to use MAX

2. **Minimal MAX Example** (`examples/minimal_max_example.py`)
   - Simplest possible working example (matrix multiply + bias + ReLU)
   - ~100 lines, demonstrates complete workflow
   - Build graph → compile → execute → verify
   - Perfect for learning the basics

3. **Project Status Document** (`docs/PROJECT_STATUS.md`)
   - Current implementation status (v0.2.0 complete)
   - Key files and their purposes
   - What we learned (API patterns, implementation details)
   - Performance characteristics
   - Future directions

4. **Comprehensive Benchmarking Framework**
   - Config-driven harness supporting multiple implementations
   - Test data in JSONL format (version controlled)
   - Multiple output formats (console, JSON, CSV, markdown)
   - System information reporting for reproducibility

### Why Documentation Matters

**For future us**: MAX is new, APIs are evolving, patterns aren't widely known yet. Writing this down captures what we learned while it's fresh.

**For others**: Few examples exist of custom MAX Graph implementations outside GenAI. This could help others building classification/regression models.

**For learning**: Writing the minimal example forced us to distill the essential patterns. Teaching is the best way to verify understanding.

---

## Chapter 6: Apple Silicon GPU Experiments

### The Question

After achieving 5.58x CPU speedup, we wondered: **Can we run MAX on Apple Silicon GPU?**

From Modular Discord feedback:
> "That's mostly a matter of adding kernels for it. There are still a few kernels without generic fallbacks, but some models in MAX might work."

### The Experiments

We created two test cases:

#### Experiment 1: Element-wise Operations
**Operations**: `ops.mul`, `ops.add`, `ops.relu`  
**Expected**: Should have GPU kernels (simple operations)

#### Experiment 2: Matrix Multiplication  
**Operations**: `ops.matmul`, `ops.transpose`  
**Expected**: Might not have GPU kernels yet

### Initial Roadblock: Xcode 26 Toolchain Issue

**Problem**: GPU compilation failed with cryptic error
```
xcrun: error: unable to find utility "metallib", not a developer tool or in PATH
```

**Investigation**:
- Xcode 26.2 is installed and configured
- `metal` compiler exists
- But `metallib` tool is missing

**Discovery**: Xcode 26 changed to on-demand component downloads (like iOS SDKs). The Metal Toolchain (~750MB) must be explicitly downloaded.

**Solution**:
```bash
xcodebuild -downloadComponent MetalToolchain
```

### Results

#### Element-wise Operations: ✅ **SUCCESS!**

```
✓ Accelerator device found: Device(type=gpu,id=0)
✓ Graph compiled and loaded on GPU  
✓ Inference executed on GPU
✓ Results match NumPy validation

Operations tested:
  - Element-wise multiplication (ops.mul)
  - Element-wise addition (ops.add)
  - Element-wise ReLU (ops.relu)
```

**First successful GPU inference on Apple Silicon with MAX!** 🎉

#### Matrix Multiplication: ❌ **Missing Kernel**

```
✓ Accelerator device found
✓ Graph built
✗ Compilation failed: matmul kernel not available

Error: Current compilation target does not support operation: mma
(mma = matrix multiply-accumulate)
```

**Conclusion**: `matmul` doesn't have Apple Silicon GPU kernel yet.

### Implications

**What works on GPU**:
- ✅ Element-wise operations  
- ✅ Graph compilation
- ✅ Data transfer (CPU ↔ GPU)
- ✅ Correct results

**What doesn't work yet**:
- ❌ Matrix multiplication
- ❌ Therefore: No transformer models (heavy matmul usage)
- ❌ Therefore: DistilBERT stays on CPU for now

**Hardware portability status**:
- CPU: ✅ Excellent performance (5.58x speedup)
- Apple GPU: ⏳ Partial support (element-wise ops work, matmul coming)
- NVIDIA/AMD GPU: ✅ Should work (MAX primary GPU target)

### Key Learnings from GPU Work

1. **Xcode 26 requires manual Metal Toolchain download** - new behaviour from Apple
2. **MAX GPU kernel availability varies by operation** - exactly as Modular team said
3. **Element-wise ops work great on Apple GPU** - promising foundation
4. **Matmul is the blocker for transformers** - most important operation for neural networks
5. **CPU performance is already excellent** - 5.58x speedup means GPU is "nice to have" not "must have"

### Documentation Created

- `examples/python/elementwise_gpu.py` - Working GPU example
- `examples/python/linear_layer.py` - matmul test (fails)
- `examples/python/README_gpu_experiments.md` - Complete findings
- `docs/XCODE_COMPATIBILITY.md` - Xcode 26 toolchain issue details

---

## What's Next

### Completed ✅
1. ✅ Lexicon-based MVP (v0.1.0)
2. ✅ Custom DistilBERT MAX Graph (v0.2.0)
3. ✅ Comprehensive benchmarking framework
4. ✅ Documentation and examples

### Potential Future Work

**Short-term**:
- Add more test cases (edge cases, longer texts)
- Experiment with quantization (int8, int4)
- Multi-batch inference optimisation
- Add FastAPI wrapper for REST API

**Medium-term**:
- Try other architectures (BERT, RoBERTa, ALBERT)
- Explore MAX Pipeline API for LLMs
- Deploy with MAX Serve (production inference server)
- GPU support when Apple GPU APIs are available in Python

**Long-term**:
- Mojo implementation when APIs stabilize
- Custom kernel development for specific ops
- Multi-model serving pipeline
- Distributed inference across devices

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
- TensorRT: NVIDIA only, can't use on M1 MacBook
- PyTorch: Keep full framework overhead for inference
- CoreML: Lock into Apple ecosystem

**With MAX**: Write graph once, runs on Mac today (CPU), Apple GPU tomorrow, cloud NVIDIA/AMD later - all with optimal performance.

**The trade-off**: Less mature than ONNX Runtime, smaller community than PyTorch, requires building custom graphs. But we gain true hardware portability and excellent performance.

---

*This document is a living record of our exploration. Updates as we progress...*
