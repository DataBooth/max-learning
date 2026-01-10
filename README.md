# MAX Inference Experiments 🔥

Learning and experimenting with Modular's MAX framework for high-performance ML inference. Starting from simple examples through to production transformer models.

## Status

**Version**: 0.3.0  
**Stage**: Community Release

## What's Here

### Implementations
- **MAX Graph DistilBERT**: Full transformer sentiment classifier achieving **5.58x speedup** over PyTorch on M1
- **Mojo Lexicon Classifier**: Simple sentiment classifier built in pure Mojo (v0.1.0 baseline)
- **Apple Silicon GPU**: First reported successful MAX Graph inference on Apple GPU

### Learning Resources
- **Numbered Examples**: Progressive learning path from element-wise ops → linear layers → full transformers
- **Comprehensive Docs**: MAX Framework Guide, implementation details, GPU experiments
- **Benchmarking**: CPU vs PyTorch, CPU vs GPU comparisons
- **Testing**: Full pytest suite validating implementations

## Quick Start

```bash
# Install dependencies
pixi install

# Run examples (progressive learning)
pixi run example-elementwise-cpu   # Simple ops: mul, add, relu
pixi run example-elementwise-gpu   # Same ops on Apple Silicon GPU
pixi run example-linear            # Linear layer (matmul + bias + relu)
pixi run example-distilbert        # Full transformer (auto-downloads models first)

# Run tests
pixi run test-python               # Full pytest suite (30 tests)
pixi run test-mojo                 # Mojo tests

# Run benchmarks
pixi run benchmark-elementwise     # Element-wise: CPU vs GPU
pixi run benchmark-distilbert      # DistilBERT: MAX vs PyTorch
```

## Performance Results

### DistilBERT Sentiment (M1 CPU)
- **MAX**: 45.88ms mean, 21.80 req/sec
- **PyTorch**: 255.85ms mean, 3.91 req/sec
- **Speedup**: **5.58x faster**, 85% better P95 latency

### Apple Silicon GPU (Element-wise)
- ✅ First reported MAX Graph inference on Apple GPU
- 🚧 `matmul` kernel not yet available (blocks transformers)
- See [Apple Silicon GPU Findings](docs/APPLE_SILICON_GPU_FINDINGS.md)

## Repository Structure

```
├── src/
│   ├── python/max_distilbert/     # MAX DistilBERT implementation
│   └── mojo/lexicon_classifier/   # Pure Mojo baseline
├── examples/python/
│   ├── 01_elementwise/            # Element-wise ops (CPU/GPU)
│   ├── 02_linear_layer/           # Linear layer example
│   └── 03_distilbert_sentiment/   # Full transformer
├── tests/python/                  # pytest suite (21 tests)
├── benchmarks/
│   ├── 01_elementwise/            # CPU vs GPU benchmarks
│   │   └── results/               # Benchmark outputs
│   └── 03_distilbert/             # MAX vs PyTorch benchmarks
│       ├── benchmark.toml         # Configuration
│       ├── results/               # Benchmark outputs
│       └── test_data/             # Test datasets
├── docs/
│   ├── MAX_FRAMEWORK_GUIDE.md     # Comprehensive MAX guide
│   ├── PROJECT_STATUS.md          # Current status & learnings
│   ├── BLOG_DRAFT.md              # Implementation journey
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
- Complete test suite (30 tests)
- Numbered examples for learning
- Apple Silicon GPU experiments

### ✅ v0.3.0 - Community Release
- Systematic benchmarking with TOML configs
- Comprehensive benchmarking guide
- All examples use configuration files
- Test organisation mirrors examples structure
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

## Learning Path

1. **Start with examples**: `examples/python/README.md` has progressive learning path
2. **Read the guides**: `docs/MAX_FRAMEWORK_GUIDE.md` explains MAX concepts
3. **Run benchmarks**: See real performance comparisons
4. **Explore GPU**: Apple Silicon GPU findings and limitations
5. **Review tests**: See how to validate MAX implementations

## Sponsorship

This project is sponsored by [DataBooth](https://www.databooth.com.au/posts/mojo) as part of our exploration of high-performance AI infrastructure with Mojo.

## Acknowledgements

- Modular team for creating Mojo
- Community contributions to mojo-toml and mojo-dotenv

## Licence

MIT Licence - see LICENCE file for details
