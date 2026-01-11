# MAX Inference Experiments 🔥

## Why This Repository?

MAX (Modular Accelerated Xecution) is Modular's framework for high-performance ML inference, promising significant speedups over traditional frameworks. However, as a relatively new framework, there's limited community knowledge about how to actually build with it.

This repository fills that gap by providing:

1. **Progressive learning path** - Six examples building from basics to production models
2. **Minimal examples** - Self-contained code highlighting graph construction without abstractions
3. **Working implementations** - Complete, tested code you can run immediately
4. **Real benchmarks** - Actual performance measurements with correctness validation
5. **Production insights** - What works, what doesn't, and why (including Apple Silicon GPU findings)
6. **Testing patterns** - Comprehensive pytest suite showing how to validate MAX implementations

**Who is this for?** Engineers and researchers wanting to understand MAX Graph API through hands-on examples.

## Status

**Version**: 0.3.0  
**Stage**: Community Release  
**Last Updated**: January 2026

## Learning Path: Examples in Order

Each example includes both a **minimal version** (no abstractions, pure MAX Graph API) and a **full version** (with configuration and helpers).

### 1️⃣ Element-wise Operations
**Path**: `examples/python/01_elementwise/`  
**Operation**: `y = relu(x * 2.0 + 1.0)`  
**Learn**: Basic graph construction, operations (mul, add, relu)  
**Status**: ✅ Works on CPU and Apple Silicon GPU

### 2️⃣ Linear Layer
**Path**: `examples/python/02_linear_layer/`  
**Operation**: `y = relu(x @ W^T + b)`  
**Learn**: Matrix operations (matmul, transpose), parameter handling  
**Status**: ✅ Works on CPU, ❌ GPU (matmul kernel missing)

### 3️⃣ DistilBERT Sentiment
**Path**: `examples/python/03_distilbert_sentiment/`  
**Model**: Full transformer (6 layers, 66M parameters)  
**Learn**: Production model loading, tokenisation, multi-layer architecture  
**Performance**: **5.58x speedup** vs PyTorch on M1 CPU

### 4️⃣ MLP Regression
**Path**: `examples/python/03_mlp_regression/`  
**Model**: Multi-layer perceptron (3 hidden layers)  
**Learn**: Sequential layers, housing price prediction  
**Benchmarks**: MAX vs PyTorch comparison included

### 5️⃣ CNN MNIST
**Path**: `examples/python/04_cnn_mnist/`  
**Model**: Convolutional neural network (2 conv + 2 dense layers)  
**Learn**: Convolutions, pooling, flattening, digit classification  
**Benchmarks**: MAX vs PyTorch comparison included

### 6️⃣ RNN Forecast (WIP)
**Path**: `examples/python/05_rnn_forecast/`  
**Status**: 🚧 Parked due to MAX Graph API limitations with sequence processing

## Quick Start

```bash
# Install dependencies
pixi install

# Run examples - start with minimal versions to learn MAX Graph API
pixi run python examples/python/01_elementwise/elementwise_minimal.py
pixi run python examples/python/02_linear_layer/linear_layer_minimal.py

# Or use pixi tasks for full versions with configuration
pixi run example-elementwise-cpu   # 1️⃣ Element-wise: mul, add, relu
pixi run example-elementwise-gpu   # 1️⃣ Same ops on Apple Silicon GPU
pixi run example-linear            # 2️⃣ Linear layer: matmul + bias + relu
pixi run example-distilbert        # 3️⃣ DistilBERT transformer
pixi run example-mlp               # 4️⃣ MLP regression
pixi run example-cnn               # 5️⃣ CNN MNIST classifier

# Run tests (49 tests total)
pixi run test-python               # Full pytest suite
pixi run test-mojo                 # Mojo tests

# Run benchmarks (generates MD + JSON + CSV reports)
pixi run benchmark-elementwise     # 1️⃣ Element-wise: CPU vs GPU
pixi run benchmark-linear          # 2️⃣ Linear layer: CPU vs GPU  
pixi run benchmark-distilbert      # 3️⃣ DistilBERT: MAX vs PyTorch
pixi run benchmark-mlp             # 4️⃣ MLP: MAX vs PyTorch
pixi run benchmark-cnn             # 5️⃣ CNN: MAX vs PyTorch
pixi run benchmark-all             # Run all benchmarks

# Cleanup benchmark reports
pixi run clean-reports-all         # Remove all benchmark reports
```

## Performance Highlights

### DistilBERT (M1 CPU)
- **MAX**: 45.88ms mean, 21.80 req/sec
- **PyTorch**: 255.85ms mean, 3.91 req/sec
- **Speedup**: **5.58x faster** with 85% better P95 latency

### MLP Regression (M1 CPU, batch=2048)
- **MAX**: 142ms per batch
- **PyTorch**: 0.56ms per batch
- **Note**: PyTorch significantly faster on this workload (~253x)

### CNN MNIST (M1 CPU, batch=256)  
- **PyTorch**: ~5x faster than MAX
- **Note**: Both produce identical predictions (correctness validated)

### Apple Silicon GPU
- ✅ **Element-wise operations working** (first reported MAX Graph GPU inference)
- ❌ **Matrix operations blocked** - `matmul` kernel not yet available
- See [Apple Silicon GPU Findings](docs/APPLE_SILICON_GPU_FINDINGS.md) for details

## Repository Structure

```
├── src/python/
│   ├── max_*/                     # MAX implementations (distilbert, mlp, cnn, rnn)
│   ├── utils/                     # Shared utilities (paths, benchmarks)
│   └── pyproject.toml             # Package configuration
├── src/mojo/
│   └── lexicon_classifier/        # Pure Mojo baseline
├── examples/python/
│   ├── 01_elementwise/            # Element-wise ops (minimal + full)
│   ├── 02_linear_layer/           # Linear layer (minimal + full)
│   ├── 03_distilbert_sentiment/   # DistilBERT transformer
│   ├── 03_mlp_regression/         # MLP for housing prices
│   ├── 04_cnn_mnist/              # CNN digit classifier
│   └── 05_rnn_forecast/           # RNN (WIP)
├── tests/python/                  # pytest suite (49 tests)
│   ├── 01_elementwise/
│   ├── 02_linear_layer/
│   ├── 03_distilbert/
│   ├── 03_mlp/
│   └── 04_cnn_mnist/
├── benchmarks/
│   ├── 01_elementwise/            # CPU vs GPU
│   ├── 02_linear_layer/           # CPU vs GPU
│   ├── 03_distilbert/             # MAX vs PyTorch
│   ├── 03_mlp/                    # MAX vs PyTorch
│   └── 04_cnn/                    # MAX vs PyTorch
├── docs/
│   ├── MAX_FRAMEWORK_GUIDE.md     # Comprehensive MAX guide
│   ├── PROJECT_STATUS.md          # Current status & learnings
│   └── APPLE_SILICON_GPU_FINDINGS.md  # GPU experiments
└── models/                        # Downloaded models (gitignored)
```

## Completed Milestones

### ✅ v0.1.0 - Lexicon-based Baseline
- Pure Mojo sentiment classifier
- Simple lexicon-based approach
- Benchmarking foundation

### ✅ v0.2.0 - MAX Graph DistilBERT
- Full MAX Graph implementation of DistilBERT
- 5.58x speedup over PyTorch on M1
- Comprehensive documentation & guides
- Numbered examples for learning
- Apple Silicon GPU experiments (element-wise ops working)

### ✅ v0.3.0 - Community Release
- **Six progressive examples**: element-wise → linear → DistilBERT → MLP → CNN → RNN (WIP)
- **Minimal examples**: Self-contained code highlighting MAX Graph API without abstractions
- **Comprehensive testing**: 49 pytest tests with correctness validation
- **Performance benchmarks**: MAX vs PyTorch comparisons for all models
- **Package restructuring**: All Python modules installable, no sys.path manipulation
- **Systematic benchmarking**: TOML configs, MD/JSON/CSV reports with machine IDs
- Australian spelling throughout documentation
- Ready for community feedback

## Future Directions

- **Larger models**: LLaMA, Mistral via MAX Pipeline API
- **Batch inference**: Throughput optimisation
- **Quantisation**: INT8/INT4 experiments
- **More GPU work**: When matmul kernels available for Apple Silicon

## Requirements

- MAX 25.1.0 or later
- Pixi package manager
- Python 3.11+ (for MAX Python API)

## Key Dependencies

- **MAX Engine**: Graph compilation and inference
- **Transformers**: Model and tokenizer loading
- **PyTorch**: For benchmarking comparisons
- **pytest**: Testing framework

## Recommended Learning Path

1. **Start with minimal examples**: Run `pixi run python examples/python/01_elementwise/elementwise_minimal.py` to see pure MAX Graph API
2. **Progress through numbered examples**: Work through 1️⃣ → 6️⃣ in order, each building on previous concepts
3. **Read the guides**: `docs/MAX_FRAMEWORK_GUIDE.md` explains MAX concepts in depth
4. **Run benchmarks**: See real performance comparisons and correctness validation
5. **Review tests**: Study `tests/python/` to see validation patterns
6. **Explore GPU findings**: Understand current Apple Silicon GPU capabilities and limitations

## Sponsorship

This project is sponsored by [DataBooth](https://www.databooth.com.au/posts/mojo) as part of our exploration of high-performance AI infrastructure with Mojo.

## Acknowledgements

- Modular team for creating Mojo
- Community contributions to mojo-toml and mojo-dotenv

## Licence

MIT Licence - see LICENCE file for details
