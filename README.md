# mojo-inference-service 🔥

A high-performance AI model inference service built in Mojo, demonstrating real-world usage of configuration management with [mojo-toml](https://github.com/databooth/mojo-toml) and [mojo-dotenv](https://github.com/databooth/mojo-dotenv).

## Status

**Version**: 0.1.0-dev  
**Stage**: Early Development

## Features

- **Pure Mojo inference**: Simple text classification built from scratch in Mojo
- **Configuration management**: Using TOML for application settings via [mojo-toml](https://github.com/databooth/mojo-toml)
- **Secrets management**: Using .env for API keys via [mojo-dotenv](https://github.com/databooth/mojo-dotenv)
- **Structured logging**: Built on Mojo's standard library Logger
- **Benchmarking**: Compare performance against Python equivalents
- **Future-ready**: Architecture designed for Modular MAX integration (v0.2.0)

## Quick Start

```bash
# Install dependencies
pixi install

# Run inference
pixi run inference --text "This product is amazing!"
# Output: POSITIVE (confidence: 0.87)

# Run with configuration
pixi run inference --text "Terrible experience" --config config.toml

# Run tests
pixi run test
```

## Configuration

### config.toml

```toml
[model]
type = "sentiment-analysis"
algorithm = "logistic-regression"  # Simple classifier for MVP
vocab_size = 10000
embedding_dim = 100

[inference]
confidence_threshold = 0.5
max_length = 512

[logging]
level = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

[server]
# Future: HTTP API configuration
host = "0.0.0.0"
port = 8080
```

### .env

```bash
# API keys and secrets (never commit this file!)
HUGGINGFACE_API_KEY=your_key_here
AUTH_TOKEN=your_token_here
```

## Architecture

```
mojo-inference-service/
├── src/
│   ├── main.mojo              # Entry point & CLI
│   ├── config.mojo            # Configuration loading (TOML + .env)
│   ├── classifier.mojo        # Simple sentiment classifier (pure Mojo)
│   ├── embeddings.mojo        # Word embeddings & tokenization
│   └── utils.mojo             # Helper functions
├── data/                      # Training data & embeddings (gitignored)
├── tests/
│   ├── test_config.mojo
│   ├── test_classifier.mojo
│   └── fixtures/
├── examples/
│   └── example_inference.mojo
├── benchmarks/
│   └── compare_python.mojo    # Performance comparison
├── config.toml                # Application configuration
├── .env.example               # Template for secrets
├── pixi.toml                  # Project dependencies
└── README.md
```

## Development Roadmap

### MVP (v0.1.0) - Pure Mojo Classifier
- [x] Project structure
- [ ] Configuration loading (mojo-toml + mojo-dotenv)
- [ ] Logging setup with config-driven levels
- [ ] Simple sentiment classifier (logistic regression)
- [ ] Word tokenization & embeddings
- [ ] CLI interface with argument parsing
- [ ] Unit tests for classifier & config
- [ ] Benchmark vs Python equivalent

### v0.2.0 - Modular MAX Integration
- [ ] Integrate Modular MAX engine
- [ ] Load PyTorch/TensorFlow models
- [ ] HTTP API endpoint (using Lightbug framework)
- [ ] Model caching & warming
- [ ] Advanced benchmarks (MAX vs pure Mojo vs Python)

### v0.3.0 - Production Ready
- [ ] Batch inference support
- [ ] Multiple model support
- [ ] Metrics & monitoring
- [ ] Health check endpoints
- [ ] Docker containerization
- [ ] Production deployment guide

## Requirements

- Mojo 25.1 or later
- Pixi package manager

## Dependencies

### MVP (v0.1.0)
- [mojo-toml](https://github.com/databooth/mojo-toml) - TOML configuration parsing
- [mojo-dotenv](https://github.com/databooth/mojo-dotenv) - Environment variable loading
- Mojo standard library (Logger, collections, etc.)

### Future (v0.2.0+)
- [Modular MAX](https://www.modular.com/max) - High-performance model serving
- [Lightbug](https://github.com/saviorand/lightbug_http) - HTTP framework for Mojo

## Use Cases

- **Learn Mojo ML**: Understand ML inference from scratch in Mojo
- **Configuration management**: Demonstrate best practices with mojo-toml and mojo-dotenv
- **Performance benchmarking**: Compare pure Mojo vs Python for text classification
- **Foundation for MAX**: Clean architecture ready for Modular MAX integration
- **Production inference**: Path from simple classifier to high-performance serving

## Sponsorship

This project is sponsored by [DataBooth](https://www.databooth.com.au/posts/mojo) as part of our exploration of high-performance AI infrastructure with Mojo.

## Acknowledgements

- Modular team for creating Mojo
- Community contributions to mojo-toml and mojo-dotenv

## Licence

MIT Licence - see LICENCE file for details
