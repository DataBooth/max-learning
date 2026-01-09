# MAX Graph Python Examples

Progressive examples demonstrating MAX Graph API from simple to complex.

## Examples Overview

### 01 - Element-wise Operations
**Complexity**: ⭐  
**Operations**: `ops.mul`, `ops.add`, `ops.relu`  
**CPU/GPU**: Both supported

The simplest possible MAX Graph - element-wise operations only.

```bash
pixi run example-elementwise-cpu    # Run on CPU
pixi run example-elementwise-gpu    # Run on Apple Silicon GPU
```

**What you'll learn**:
- Basic graph construction
- Device handling (CPU vs GPU)
- Data transfer between devices
- Verifying results

**GPU Support**: ✅ Works on Apple Silicon (M1/M2/M3)

---

### 02 - Linear Layer
**Complexity**: ⭐⭐  
**Operations**: `ops.matmul`, `ops.transpose`, `ops.add`, `ops.relu`  
**CPU/GPU**: CPU only (matmul GPU kernel not available)

A single linear layer with bias and activation - the building block of neural networks.

```bash
pixi run example-linear
```

**What you'll learn**:
- Matrix operations in MAX
- Linear layer pattern: `matmul(x, transpose(W)) + bias`
- Why transpose is needed
- Weight management with constants

**GPU Support**: ❌ `matmul` operation doesn't have Apple Silicon GPU kernel yet

---

### 03 - DistilBERT Sentiment Classification
**Complexity**: ⭐⭐⭐⭐⭐  
**Operations**: Full transformer (embeddings, multi-head attention, feed-forward)  
**CPU/GPU**: CPU only (uses matmul extensively)

Production-quality sentiment classifier using custom DistilBERT implementation.

```bash
pixi run example-distilbert
```

**Performance**:
- 5.58x faster than HuggingFace PyTorch (on M1 CPU)
- 85% better P95 latency
- 100% accuracy parity

**What you'll learn**:
- Complete transformer architecture
- Custom embeddings (no token types)
- DistilBERT-specific attention patterns
- Classification head design
- Production model structure

**Implementation**: See `src/python/max_distilbert/` for model code

**GPU Support**: ❌ Transformer models use matmul heavily, waiting for GPU kernels

---

## Learning Path

**Start here**: Element-wise example (01)
- Understand MAX Graph basics
- Test GPU support on your hardware

**Then**: Linear layer example (02)  
- Learn matrix operations
- Understand why matmul doesn't work on GPU yet

**Finally**: DistilBERT example (03)
- See a real production model
- Understand performance benefits

## Device Support Summary

| Example | CPU | Apple GPU | NVIDIA/AMD GPU |
|---------|-----|-----------|----------------|
| Element-wise | ✅ | ✅ | ✅ (should work) |
| Linear layer | ✅ | ❌ (missing matmul) | ✅ (should work) |
| DistilBERT | ✅ | ❌ (missing matmul) | ✅ (should work) |

## GPU Notes

### Apple Silicon GPU
- **Element-wise ops work!** (tested on M1)
- **Matrix multiply (matmul) not yet available**
- Requires Xcode 26 Metal Toolchain: `xcodebuild -downloadComponent MetalToolchain`

See `01_elementwise/README.md` for detailed GPU findings.

### NVIDIA/AMD GPU
- Not tested in this project
- MAX's primary GPU targets
- Should support all operations including matmul

## Performance

**DistilBERT Benchmarks** (M1 CPU, 100 iterations):
- MAX: 45.88ms mean latency
- PyTorch: 255.85ms mean latency  
- **Speedup: 5.58x** 🚀

See `benchmarks/results/` for detailed benchmark data.

## Directory Structure

```
examples/python/
├── README.md (this file)
├── 01_elementwise/
│   ├── elementwise.py
│   └── README.md
├── 02_linear_layer/
│   ├── linear_layer.py
│   └── README.md
└── 03_distilbert_sentiment/
    ├── distilbert_sentiment.py
    └── README.md (todo)
```

## Related Documentation

- **MAX Framework Guide**: `docs/MAX_FRAMEWORK_GUIDE.md`
- **Project Status**: `docs/PROJECT_STATUS.md`
- **Benchmarking**: `benchmarks/`
- **Model Implementation**: `src/python/max_distilbert/`

## MAX Resources

- [MAX Documentation](https://docs.modular.com/max/)
- [MAX Graph API](https://docs.modular.com/max/graph/)
- [Graph Operations Reference](https://docs.modular.com/max/graph/ops)
- [Modular Forums](https://forum.modular.com)
