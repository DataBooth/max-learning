# MAX Learning Repository - Community Announcement

## What Is This?

An **introductory learning repository** for Modular's MAX Engine, documenting a systematic journey from simple operations through to transformer models. This captures real learning experiences, working code, and actual performance results to help others get started with MAX Graph API.

**Repository:** https://github.com/DataBooth/max-learning  
**Version:** 0.3.0

## Why Does This Exist?

### The Learning Journey
MAX is powerful, but getting started can be challenging. This repository shares:
- **Real learning progression** - from simple operations to transformers
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

## Key Highlights

### 1. Apple Silicon GPU Breakthrough
**First reported MAX Graph inference on Apple Silicon GPU**
- Element-wise operations working (`ops.mul`, `ops.add`, `ops.relu`)
- Documented what works, what doesn't, and why
- Fixed Xcode 26 Metal Toolchain issue
- Clear limitations: no matmul kernel yet (blocks transformers)
- See: [`docs/APPLE_SILICON_GPU_FINDINGS.md`](APPLE_SILICON_GPU_FINDINGS.md)

### 2. Impressive CPU Performance
**DistilBERT: 5.58x faster than PyTorch on M1 CPU**
- Mean latency: 45.88ms (MAX) vs 255.85ms (PyTorch)
- 85% better P95 latency
- 8x more consistent performance
- 100% accuracy parity

### 3. Progressive Learning Path
**Six numbered examples build understanding step-by-step:**

Each example includes a **minimal version** (pure MAX Graph API, no abstractions) and a **full version** (with configuration).

**1️⃣ Element-wise Operations** - The basics
```bash
pixi run python examples/python/01_elementwise/elementwise_minimal.py  # Start here!
pixi run example-elementwise-cpu   # Full version: mul, add, relu
pixi run example-elementwise-gpu   # Works on Apple Silicon GPU!
```

**2️⃣ Linear Layer** - Matrix operations
```bash
pixi run python examples/python/02_linear_layer/linear_layer_minimal.py  # Learn matmul
pixi run example-linear            # Full version: matmul, bias, activation
```

**3️⃣ DistilBERT Sentiment** - Full transformer
```bash
pixi run example-distilbert        # 66M parameters, 5.58x faster than PyTorch
```

**4️⃣ MLP Regression** - Multi-layer perceptron
```bash
pixi run example-mlp               # Housing price prediction, 3 hidden layers
```

**5️⃣ CNN MNIST** - Convolutional network
```bash
pixi run example-cnn               # Digit classification, 2 conv + 2 dense layers
```

**6️⃣ RNN Forecast** - Recurrent network (WIP)
```bash
# Parked due to MAX Graph API limitations with sequence processing
```

### 4. Systematic Benchmarking
- **Multiple formats**: Markdown + JSON + CSV reports
- **Machine identifiers**: Filenames include chip ID (e.g., `cpu_vs_gpu_m1-pro_20260110_190904.md`)
- **Configuration-driven**: TOML files, 1000 iterations per benchmark
- **Statistical rigour**: mean, median, P95, P99, coefficient of variation
- **GPU detection**: Automatic system info capture
- **Cleanup tools**: Easy report management
- See: [`benchmarks/BENCHMARK_README.md`](../benchmarks/BENCHMARK_README.md)

## Getting Started

### Quick Start
```bash
# Clone
git clone https://github.com/DataBooth/max-learning
cd max-learning

# Install (requires Pixi package manager)
pixi install

# Start with minimal examples to learn MAX Graph API
pixi run python examples/python/01_elementwise/elementwise_minimal.py
pixi run python examples/python/02_linear_layer/linear_layer_minimal.py

# Progress through full examples (models download automatically)
pixi run example-elementwise-cpu   # 1️⃣
pixi run example-linear            # 2️⃣  
pixi run example-distilbert        # 3️⃣
pixi run example-mlp               # 4️⃣
pixi run example-cnn               # 5️⃣
```

### Available Pixi Tasks

**Examples** (progressive learning):
```bash
pixi run example-elementwise-cpu   # 1️⃣ Element-wise ops on CPU
pixi run example-elementwise-gpu   # 1️⃣ Element-wise ops on GPU
pixi run example-linear            # 2️⃣ Linear layer
pixi run example-distilbert        # 3️⃣ DistilBERT transformer
pixi run example-mlp               # 4️⃣ MLP regression
pixi run example-cnn               # 5️⃣ CNN MNIST classifier
```

**Testing** (49 tests, all passing):
```bash
pixi run test-python               # Run pytest suite
pixi run test-all                  # Run all tests
```

**Benchmarking** (generates MD + JSON + CSV):
```bash
pixi run benchmark-elementwise     # 1️⃣ CPU vs GPU
pixi run benchmark-linear          # 2️⃣ Linear layer
pixi run benchmark-distilbert      # 3️⃣ MAX vs PyTorch
pixi run benchmark-mlp             # 4️⃣ MAX vs PyTorch
pixi run benchmark-cnn             # 5️⃣ MAX vs PyTorch
pixi run benchmark-all             # Run all benchmarks
```

**Maintenance**:
```bash
pixi run download-models           # Manually download models
pixi run clean-reports-all         # Clean all benchmark reports
pixi run clean-reports-md          # Clean just Markdown reports
pixi run clean-reports-json        # Clean just JSON reports
pixi run clean-reports-csv         # Clean just CSV reports
```

Complete documentation in [`README.md`](../README.md)

## What You Get

### Immediate Value
- **Minimal examples** - pure MAX Graph API without abstractions
- **Working implementations** - 6 progressive examples, copy, paste, run
- **Benchmarking framework** - adapt for your experiments
- **GPU workarounds** - specifically for Apple Silicon
- **Test suite** - 49 tests showing validation approaches with correctness checks
- **Configuration patterns** - production-ready structure

### Learning Resources
- **MAX Framework Guide** - Patterns and best practices learned
- **Benchmark Guide** - Understanding metrics and methodology
- **Mermaid Diagrams** - Visualising computation flows
- **Real troubleshooting** - Actual issues and solutions

## Repository Structure

```
max-learning/
├── examples/python/
│   ├── 01_elementwise/          # Element-wise ops (minimal + full)
│   ├── 02_linear_layer/         # Linear layer (minimal + full)
│   ├── 03_distilbert_sentiment/ # DistilBERT transformer
│   ├── 03_mlp_regression/       # MLP for housing prices
│   ├── 04_cnn_mnist/            # CNN digit classifier
│   └── 05_rnn_forecast/         # RNN (WIP)
├── benchmarks/
│   ├── 01_elementwise/          # CPU vs GPU
│   ├── 02_linear_layer/         # CPU vs GPU
│   ├── 03_distilbert/           # MAX vs PyTorch
│   ├── 03_mlp/                  # MAX vs PyTorch
│   └── 04_cnn/                  # MAX vs PyTorch
├── tests/python/                # 49 tests mirroring examples
├── docs/                        # Comprehensive guides
└── src/python/
    ├── max_*/                   # MAX implementations
    └── utils/                   # Shared utilities (paths, benchmarks)
```

## Who Is This For?

### You Should Use This If:
- Getting started with MAX Graph API
- Want to see working code, not just documentation
- Benchmarking MAX on your hardware
- Building ML inference services
- Exploring Apple Silicon GPU for MAX
- Need practical configuration patterns

### This Might Not Be For You If:
- Looking for MAX Pipeline API examples (we focus on Graph API)
- Training models (this is inference-only)
- Need many model architectures (currently: 1 transformer, 1 MLP, 1 CNN, 2 basic ops)
- Want production-scale deployment patterns (this is learning-focused)

## Technical Details

### Test Environment
- **Hardware:** MacBook Pro M1 Pro, 16GB RAM
- **OS:** macOS 15.7.3
- **MAX:** 25.1.0
- **Python:** 3.11+
- **Benchmarks:** 1000 iterations with 100 warmup iterations

### What Currently Works
- Element-wise operations on Apple Silicon GPU
- All operations on CPU (element-wise, linear, transformers, MLP, CNN)
- DistilBERT transformer on CPU (5.58x faster than PyTorch)
- Configuration-driven benchmarks with multiple output formats
- Comprehensive test coverage (49 tests with correctness validation)
- Package-based structure (no sys.path manipulation)

### Known Limitations
- **GPU-constrained**: No matmul kernel for Apple Silicon GPU (blocks transformers, MLP, CNN)
- **No ETA**: Waiting on Modular team for advanced GPU kernels
- **macOS focused**: Tested on Apple Silicon (patterns are portable)
- **Performance**: PyTorch faster on some workloads (MLP, CNN) - MAX shines on transformers
- **RNN**: Parked due to MAX Graph API limitations with sequence processing

## How This Compares to Official MAX Examples

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

## Contributing & Feedback

**This is v0.3.0 - ready for community feedback!** All 49 tests passing, benchmarks working across all examples, minimal examples highlighting MAX Graph API.

**We welcome:**
- Bug reports - What broke? How can we fix it?
- Corrections - Spotted an error? Please let us know!
- Performance data - Share your hardware results
- New examples - Especially simple operations that work on current MAX
- Documentation improvements - Help make things clearer
- GPU findings - Any accelerator, any platform
- Feedback - What worked? What was confusing? What's missing?

**How to contribute:**
- Open an issue on GitHub for bugs or suggestions
- Start a discussion for questions or ideas
- Submit PRs for improvements (code, docs, configs)
- Share your benchmark results from different hardware

## Key Resources

### In This Repository
- [`README.md`](../README.md) - Complete guide and quick start
- [`docs/MAX_FRAMEWORK_GUIDE.md`](MAX_FRAMEWORK_GUIDE.md) - MAX Graph patterns learned
- [`docs/APPLE_SILICON_GPU_FINDINGS.md`](APPLE_SILICON_GPU_FINDINGS.md) - GPU exploration
- [`benchmarks/BENCHMARK_README.md`](../benchmarks/BENCHMARK_README.md) - Understanding metrics
- [`docs/PROJECT_STATUS.md`](PROJECT_STATUS.md) - Current state and learnings

### Official Resources
- [Modular MAX Documentation](https://docs.modular.com/max/)
- [MAX Graph API](https://docs.modular.com/max/graph/)
- [Build LLM from Scratch](https://llm.modular.com)
- [Modular Forums](https://forum.modular.com)

## Acknowledgements

- **[Modular](https://modular.com)** for creating MAX and encouraging community GPU experiments
- **[DataBooth](https://www.databooth.com.au)** for sponsoring this learning exploration
- **Community** for early feedback and shared discoveries

## Licence

MIT Licence - See LICENCE file

---

## Questions? Found This Helpful? Want to Contribute?

- **GitHub:** https://github.com/DataBooth/max-learning
- **Issues:** Bug reports and feature requests
- **Discussions:** Share your results, ask questions

**This is a learning journey - let's explore MAX together!**

---

*Note: This is v0.3.0 - an introductory/learning repository ready for community feedback. It captures real experiences getting started with MAX, including what worked, what didn't, and why. All contributions and feedback welcome as we all learn together.*
