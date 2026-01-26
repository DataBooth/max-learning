# Release Notes

## v0.3.0 - Public Release (January 2026)

**MAX/Mojo Version**: `26.1.0.dev2026010718` (locked)

### What's New

#### Six Progressive Examples
- **1️⃣ Element-wise Operations** - Basic graph construction with `mul`, `add`, `relu`
- **2️⃣ Linear Layer** - Matrix operations with `matmul`, `transpose`
- **3️⃣ DistilBERT Sentiment** - Production transformer (66M parameters, 5.58x speedup vs PyTorch)
- **4️⃣ MLP Regression** - Multi-layer perceptron for housing prices
- **5️⃣ CNN MNIST** - Convolutional network for digit classification
- **6️⃣ RNN Forecast** - (WIP) Parked due to API limitations

#### Minimal Examples
Each example includes a minimal version showing pure MAX Graph API without abstractions:
- `elementwise_minimal.py` - 116 lines of clear, educational code
- `linear_layer_minimal.py` - 140 lines showing matrix operations

#### Infrastructure Improvements
- **Comprehensive testing**: 49 pytest tests with correctness validation
- **Performance benchmarks**: MAX vs PyTorch comparisons for all models
- **Package restructuring**: All Python modules installable, no sys.path manipulation
- **Systematic benchmarking**: TOML configs, MD/JSON/CSV reports with machine IDs
- **Version locking**: MAX version locked to prevent breaking API changes
- **Code quality**: Pre-commit hooks, ruff formatting

#### Documentation
- **Learning journey framing**: Repository positioned as one developer's exploration
- **Mojo context**: Documentation explaining Mojo Graph API deprecation (May 2025)
- **GPU findings**: Apple Silicon GPU experiments (element-wise ops working, matmul blocked)
- **Terminology feedback**: Submitted InferenceSession clarification to Modular docs team

### Performance Highlights

- **DistilBERT (M1 CPU)**: 5.58x faster than PyTorch (45.88ms vs 255.85ms)
- **MLP Regression**: PyTorch 253x faster (honest reporting of both wins and losses)
- **CNN MNIST**: PyTorch ~5x faster (both produce identical predictions)
- **Apple Silicon GPU**: Element-wise ops working ✅, matmul kernel missing ❌

### Requirements

- MAX/Mojo: `26.1.0.dev2026010718` (locked in `pixi.toml`)
- Pixi package manager
- Python 3.11+

### Breaking Changes

None - this is the initial public release.

### Known Limitations

- RNN example parked due to MAX Graph API sequence processing limitations
- Apple Silicon GPU matmul kernels not yet available
- MAX slower than PyTorch on some workloads (MLP, CNN)

---

## v0.2.0 - MAX Graph DistilBERT (December 2025)

**Internal version** - not publicly released

### Features
- Full MAX Graph implementation of DistilBERT
- 5.58x speedup over PyTorch on M1
- Comprehensive documentation and guides
- Numbered examples for learning
- Apple Silicon GPU experiments (element-wise ops working)

---

## v0.1.0 - Lexicon-based Baseline (November 2025)

**Internal version** - not publicly released

### Features
- Pure Mojo sentiment classifier
- Simple lexicon-based approach
- Benchmarking foundation

---

## Version Support Policy

**MAX API Version Locking**: This repository locks the MAX version to prevent breaking changes. Users should:
1. Use the locked version for guaranteed compatibility
2. Update `modular = "==X.X.X"` in `pixi.toml` when ready to modernise
3. Test all examples after updating MAX version
4. Report any breaking changes as GitHub issues

**Learning Resource**: This is a learning repository documenting one developer's exploration of MAX. Not an authoritative guide.

---

## Changelog Format

We follow these principles:
- **Honest reporting**: Document both performance wins and losses
- **Version transparency**: Lock and document MAX versions used
- **Learning focus**: Emphasize educational value over production readiness
- **Community engagement**: Welcome corrections and improvements

---

**Maintained by**: [DataBooth](https://www.databooth.com.au)  
**Licence**: Apache 2.0  
**Repository**: https://github.com/DataBooth/max-learning
