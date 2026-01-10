# MAX Learning Repository - Community Announcement

## 🚀 What Is This?

An **introductory learning repository** for Modular's MAX Engine, documenting a systematic journey from simple operations through to production transformer models. This captures real learning experiences, working code, and actual performance results to help others get started with MAX Graph API.

**Repository:** https://github.com/DataBooth/max-learning

## 🎯 Why Does This Exist?

### The Learning Journey
MAX is powerful, but getting started can be challenging. This repository shares:
- **Real learning progression** - from confusion to clarity
- **Working examples** - not theoretical, actually runs
- **Actual benchmarks** - real hardware, real numbers
- **Mistakes and solutions** - what didn't work and why

### The Gap It Fills
Documentation is excellent, but sometimes you need:
- Copy-paste examples that work immediately
- Progressive complexity (start simple, build up)
- Real performance data from actual hardware
- Practical patterns for common operations
- Troubleshooting guidance (especially GPU on Apple Silicon)

## 🔥 Key Highlights

### 1. Apple Silicon GPU Breakthrough ✅
**First reported MAX Graph inference on Apple Silicon GPU**
- Element-wise operations working (`ops.mul`, `ops.add`, `ops.relu`)
- Documented what works, what doesn't, and why
- Fixed Xcode 26 Metal Toolchain issue
- Clear limitations: no matmul kernel yet (blocks transformers)
- See: [`docs/APPLE_SILICON_GPU_FINDINGS.md`](docs/APPLE_SILICON_GPU_FINDINGS.md)

### 2. Impressive CPU Performance 📈
**DistilBERT: 5.58x faster than PyTorch on M1 CPU**
- Mean latency: 45.88ms (MAX) vs 255.85ms (PyTorch)
- 85% better P95 latency
- 8x more consistent performance
- 100% accuracy parity

### 3. Progressive Learning Path 🎓
**Numbered examples build understanding step-by-step:**

**01: Element-wise Operations** - The basics
```bash
pixi run example-elementwise-cpu   # mul, add, relu
pixi run example-elementwise-gpu   # Works on Apple Silicon!
```

**02: Linear Layer** - Matrix operations
```bash
pixi run example-linear            # matmul, bias, activation
```

**03: DistilBERT Sentiment** - Full transformer
```bash
pixi run example-distilbert        # Production-ready model
```

### 4. Systematic Benchmarking 📊
- Configuration-driven (TOML files, no magic numbers)
- Statistical rigour (mean, median, P95, P99, coefficient of variation)
- GPU detection and system info capture
- Timestamped markdown reports
- See: [`benchmarks/BENCHMARK_README.md`](benchmarks/BENCHMARK_README.md)

## 🎁 What You Get

### Immediate Value
- **Working examples** - copy, paste, run
- **Benchmarking framework** - adapt for your experiments
- **GPU workarounds** - specifically for Apple Silicon
- **Test suite** - 30 tests showing validation approaches
- **Configuration patterns** - production-ready structure

### Learning Resources
- **MAX Framework Guide** - Patterns and best practices learned
- **Benchmark Guide** - Understanding metrics and methodology
- **Mermaid Diagrams** - Visualising computation flows
- **Real troubleshooting** - Actual issues and solutions

## 🚦 Getting Started

```bash
# Clone
git clone https://github.com/DataBooth/max-learning
cd max-learning

# Install (requires Pixi package manager)
pixi install

# Run progressive examples
pixi run example-elementwise-cpu   # Start here
pixi run example-linear            # Then this  
pixi run example-distilbert        # Finally this

# Run tests (30 tests)
pixi run test-python

# Run benchmarks
pixi run benchmark-elementwise     # CPU vs GPU
pixi run benchmark-linear          # Linear layer
pixi run benchmark-distilbert      # MAX vs PyTorch
```

Complete documentation in [`README.md`](README.md)

## 📁 Repository Structure

```
max-learning/
├── examples/python/
│   ├── 01_elementwise/      # Simple ops, CPU/GPU support, config-driven
│   ├── 02_linear_layer/     # Matrix ops, config-driven
│   └── 03_distilbert_sentiment/  # Full transformer wrapper
├── benchmarks/
│   ├── benchmark_utils.py   # Shared utilities (GPU detection, reporting)
│   ├── 01_elementwise/      # CPU vs GPU benchmarks + config
│   ├── 02_linear_layer/     # Linear benchmarks + config
│   └── 03_distilbert/       # MAX vs PyTorch + config
├── tests/python/            # 30 tests mirroring examples structure
├── docs/                    # Comprehensive guides
└── src/python/max_distilbert/   # Custom DistilBERT implementation
```

## 🎯 Who Is This For?

### You Should Use This If:
- ✅ Getting started with MAX Graph API
- ✅ Want to see working code, not just documentation
- ✅ Benchmarking MAX on your hardware
- ✅ Building ML inference services
- ✅ Exploring Apple Silicon GPU for MAX
- ✅ Need practical configuration patterns

### This Might Not Be For You If:
- ❌ Looking for MAX Pipeline API examples (we focus on Graph API)
- ❌ Training models (this is inference-only)
- ❌ Need many model architectures (currently: 1 transformer, 2 simple ops)
- ❌ Want production-scale deployment patterns (this is learning-focused)

## 🔬 Technical Details

### Test Environment
- **Hardware:** MacBook Pro M1 Pro, 16GB RAM
- **OS:** macOS 15.7.3
- **MAX:** 25.1.0
- **Python:** 3.11+
- **Future:** Planning M4 Pro comparison tests

### What Currently Works
- ✅ Element-wise operations on Apple Silicon GPU
- ✅ All operations on CPU (both simple and transformer)
- ✅ DistilBERT transformer on CPU (5.58x faster than PyTorch)
- ✅ Configuration-driven benchmarks
- ✅ Comprehensive test coverage

### Known Limitations
- ⚠️ **CPU-constrained**: No matmul kernel for Apple Silicon GPU (blocks transformers)
- ⚠️ **No ETA**: Waiting on Modular team for advanced GPU kernels
- ⚠️ **macOS focused**: Tested on Apple Silicon (patterns are portable)
- ⚠️ **Limited scope**: Single transformer architecture

## 🔮 Future Plans

### Near-term
- M4 Pro benchmark comparisons (intra and inter-chip)
- Configurable precision in benchmark reporting
- More simple operation examples
- Additional documentation based on community feedback

### Dependent on MAX Development
- More GPU examples (when kernels available)
- Additional transformer architectures
- Larger model examples
- Quantisation experiments

**Note:** Progress on GPU-heavy examples is blocked until Modular releases more Apple Silicon kernels. This is a learning repository that will grow as MAX capabilities expand.

## 🔍 How This Compares to Official MAX Examples

The official [Modular MAX repository](https://github.com/modular/modular) contains over 450,000 lines of production-grade code, including full LLM implementations (Llama 3.1) and the world's largest repository of open source CPU and GPU kernels.

**This repository is different:**

### What Official Examples Provide
- Production-ready implementations at scale
- Llama3 compatible with 20,000+ model variants on Hugging Face
- Advanced features: serving, deployment, custom ops, multi-GPU
- Both Python and Mojo implementations
- Reference implementations for extending MAX

### What This Repository Provides
- **Guided learning companion** to official documentation
- Progressive complexity: start simple, build understanding
- **First working examples** of MAX Graph on Apple Silicon GPU
- Detailed troubleshooting (including failures and solutions)
- Systematic benchmarking methodology explained
- Copy-paste code that works immediately

### Positioning

> **Think of this as the "learn by doing" companion to the official MAX documentation.**

It bridges the gap between reading API docs and building production models, with special focus on getting started on Apple Silicon. Not a replacement for official examples, but a complement that helps you understand them better.

**Recommended path:**
1. Start here to understand MAX Graph fundamentals
2. Move to official examples for production patterns
3. Contribute back your learnings to both!

## 🤝 Contributing & Feedback

**This is a work in progress!** While tests and benchmarks are available for each example, the repository is actively evolving.

**We welcome:**
- ✅ **Bug reports** - What broke? How can we fix it?
- ✅ **Corrections** - Spotted an error? Please let us know!
- ✅ **Performance data** - Share your hardware results
- ✅ **New examples** - Especially simple operations that work on current MAX
- ✅ **Documentation improvements** - Help make things clearer
- ✅ **GPU findings** - Any accelerator, any platform
- ✅ **Feedback** - What worked? What was confusing? What's missing?

**How to contribute:**
- Open an issue on GitHub for bugs or suggestions
- Start a discussion for questions or ideas
- Submit PRs for improvements (code, docs, configs)
- Share your benchmark results from different hardware

See issues on GitHub or start a discussion!

## 📚 Key Resources

### In This Repository
- [`README.md`](README.md) - Complete guide and quick start
- [`docs/MAX_FRAMEWORK_GUIDE.md`](docs/MAX_FRAMEWORK_GUIDE.md) - MAX Graph patterns learned
- [`docs/APPLE_SILICON_GPU_FINDINGS.md`](docs/APPLE_SILICON_GPU_FINDINGS.md) - GPU exploration
- [`benchmarks/BENCHMARK_README.md`](benchmarks/BENCHMARK_README.md) - Understanding metrics
- [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) - Current state and learnings

### Official Resources
- [Modular MAX Documentation](https://docs.modular.com/max/)
- [MAX Graph API](https://docs.modular.com/max/graph/)
- [Build LLM from Scratch](https://llm.modular.com)
- [Modular Forums](https://forum.modular.com)

## 🙏 Acknowledgements

- **[Modular](https://modular.com)** for creating MAX and encouraging community GPU experiments
- **[DataBooth](https://www.databooth.com.au)** for sponsoring this learning exploration
- **Community** for early feedback and shared discoveries

## 📄 Licence

MIT Licence - See LICENCE file

---

## Questions? Found This Helpful? Want to Contribute?

- **GitHub:** https://github.com/DataBooth/max-learning
- **Issues:** Bug reports and feature requests
- **Discussions:** Share your results, ask questions

**This is a learning journey - let's explore MAX together! 🚀**

---

*Note: This is an introductory/learning repository, not production infrastructure. It captures real experiences getting started with MAX, including what worked, what didn't, and why. Contributions and feedback welcome as we all learn together.*
