# Release Notes - v0.3.0 Community Release

**Release Date:** 11 January 2026  
**Status:** Ready for community feedback

## Overview

v0.3.0 represents a major expansion of the MAX Learning repository, growing from 3 examples to 6 progressive examples with comprehensive testing, benchmarking, and documentation. This release transforms the repository from a focused DistilBERT demonstration into a complete learning path for MAX Graph API.

## What's New

### 🎯 Six Progressive Examples

1. **Element-wise Operations** - Basic ops (mul, add, relu)
   - ✅ Works on CPU and Apple Silicon GPU
   - Includes minimal and full versions

2. **Linear Layer** - Matrix operations (matmul, transpose, bias)
   - ✅ Works on CPU
   - ❌ Apple Silicon GPU (matmul kernel missing)
   - Includes minimal and full versions

3. **DistilBERT Sentiment** - Full transformer (66M parameters)
   - 🚀 5.58x speedup vs PyTorch on M1 CPU
   - Production-quality inference

4. **MLP Regression** - Multi-layer perceptron
   - 3 hidden layers
   - Housing price prediction
   - NEW in v0.3.0

5. **CNN MNIST** - Convolutional neural network
   - 2 conv + 2 dense layers
   - Digit classification
   - NEW in v0.3.0

6. **RNN Forecast** - Recurrent network
   - 🚧 WIP - Parked due to MAX Graph API limitations
   - NEW in v0.3.0

### ⭐ Minimal Examples

New self-contained minimal examples for element-wise and linear layer operations:
- No configuration files
- No helper functions
- Pure MAX Graph API
- Clear 4-step structure: Build → Compile → Run → Results
- ~120-140 lines each
- Perfect starting point for learning

### 🧪 Comprehensive Testing

- **49 tests** (up from 30)
- Correctness validation for all examples
- Tests mirror example structure:
  - `tests/python/01_elementwise/`
  - `tests/python/02_linear_layer/`
  - `tests/python/03_distilbert/`
  - `tests/python/03_mlp/`
  - `tests/python/04_cnn_mnist/`

### 📊 Enhanced Benchmarking

- **MAX vs PyTorch comparisons** for all neural network models
- Correctness verification (max absolute error checks)
- Reports in MD/JSON/CSV formats
- Machine ID in filenames for multi-system tracking
- New benchmarks:
  - `benchmarks/03_mlp/` - MLP vs PyTorch
  - `benchmarks/04_cnn/` - CNN vs PyTorch

### 📦 Package Restructuring

- All Python modules now installable via `src/python/pyproject.toml`
- Eliminated ALL `sys.path` manipulation from codebase
- Created `utils` package with:
  - `paths.py` - Project root detection via pixi.toml/.git search
  - `benchmark_utils.py` - Shared benchmarking utilities
- Editable installs via pixi: `[pypi-dependencies]`
- Version synchronised: pyproject.toml and pixi.toml both at 0.3.0

### 📚 Documentation Overhaul

- **README.md** completely rewritten:
  - Clear "Why This Repository?" section
  - All 6 examples documented in order (1️⃣-6️⃣)
  - Performance highlights (successes AND limitations)
  - Progressive learning path articulated
  - Repository structure expanded

- **Example READMEs updated**:
  - Document minimal vs full versions
  - Explain when to use each
  - Include expected output

- **Community announcements**:
  - Long version: comprehensive feature documentation
  - Short version: 120-word summary
  - Moved to `docs/planning/`

## Performance Results

### ✅ Where MAX Excels

**DistilBERT (M1 CPU, batch=1)**
- MAX: 45.88ms mean
- PyTorch: 255.85ms mean
- **Result: 5.58x speedup** 🚀

### ⚠️ Where PyTorch is Faster

**MLP Regression (M1 CPU, batch=2048)**
- MAX: 142ms
- PyTorch: 0.56ms
- Result: PyTorch ~253x faster

**CNN MNIST (M1 CPU, batch=256)**
- PyTorch: ~5x faster than MAX
- Note: Both produce identical predictions

**Takeaway:** MAX shines on transformers but PyTorch is faster on smaller workloads (MLP, CNN) on Apple Silicon.

### 🍎 Apple Silicon GPU Status

- ✅ **Element-wise operations working** (first reported MAX Graph GPU inference)
- ❌ **Matrix operations blocked** - matmul kernel not available
- Blocks: transformers, MLP, CNN on GPU

## Breaking Changes

None - this is a purely additive release.

## Migration Guide

If upgrading from v0.2.0:
1. Run `pixi install` to install new package structure
2. No code changes required - all examples remain compatible
3. New examples are opt-in

## What's Next (v0.4.0 candidates)

- Larger models: LLaMA, Mistral via MAX Pipeline API
- Batch inference optimisation
- Quantisation experiments (INT8/INT4)
- More GPU work when matmul kernels available

## Installation

```bash
git clone https://github.com/DataBooth/max-learning
cd max-learning
pixi install

# Start with minimal examples
pixi run python examples/python/01_elementwise/elementwise_minimal.py
pixi run python examples/python/02_linear_layer/linear_layer_minimal.py
```

## Requirements

- MAX 25.1.0 or later
- Pixi package manager
- Python 3.11+

## Contributors

This release was made possible by thorough testing, benchmarking, and documentation work.

## Acknowledgements

- [Modular](https://modular.com) for MAX Engine
- [DataBooth](https://www.databooth.com.au) for sponsoring this learning exploration
- Community for early feedback

## Licence

MIT Licence

---

**Full Changelog:** https://github.com/DataBooth/max-learning/compare/v0.2.0...v0.3.0
