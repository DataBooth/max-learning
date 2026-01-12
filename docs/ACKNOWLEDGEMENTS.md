# Acknowledgements and Attributions

This document acknowledges external resources, tutorials, and documentation that inspired or informed the examples in this repository.

## Official MAX Documentation

### MLP Example (03_mlp_regression)
The MLP implementation using `max.nn.Module` was inspired by the official MAX documentation:
- **Source**: [Build an MLP block as a module](https://docs.modular.com/max/develop/build-an-mlp-block)
- **Concepts adapted**: 
  - Using `Module` class for layer composition
  - `__call__` method for forward pass
  - Layer stacking with ReLU activations
- **Our additions**:
  - Applied to regression task (California housing dataset)
  - Pre-trained weights included
  - Complete end-to-end inference example
  - Benchmarking against PyTorch

### MAX Graph API Fundamentals
General MAX Graph concepts learned from:
- [Get started with MAX graphs (Python tutorial)](https://docs.modular.com/max/develop/get-started-with-max-graph-in-python)
- [Introduction to MAX Graph](https://docs.modular.com/max/graph/)
- [MAX Graph Operations Reference](https://docs.modular.com/max/graph/ops)
- [Device Management](https://docs.modular.com/max/graph/devices)

## Datasets

### California Housing
- **Source**: scikit-learn's `fetch_california_housing()`
- **Usage**: MLP regression example
- **License**: BSD-3-Clause (scikit-learn)

### MNIST
- **Source**: torchvision's MNIST dataset
- **Usage**: CNN classifier example
- **License**: Creative Commons Attribution-Share Alike 3.0

## Models

### DistilBERT
- **Source**: Hugging Face model hub
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Original paper**: "DistilBERT, a distilled version of BERT" (Sanh et al., 2019)
- **License**: Apache 2.0

## Community Contributions

This repository has been improved through:
- Feedback from the Modular Discord community
- Early testing and bug reports from users
- Suggestions for additional examples and clarifications

## Tools and Frameworks

- **MAX Engine**: Modular's high-performance inference framework
- **Pixi**: Package manager for managing dependencies
- **PyTorch**: Used for benchmarking comparisons and dataset loading
- **Hugging Face Transformers**: Model and tokeniser loading

## Learning Resources

Key resources that informed our understanding:
- Modular's official documentation and tutorials
- MAX GitHub repository examples
- Community discussions on Modular forums and Discord

---

## How to Contribute Attributions

If you notice missing attributions or have suggestions for acknowledgements:
1. Open an issue on GitHub
2. Provide the source/resource details
3. Describe how it relates to our examples

We strive to properly acknowledge all inspirations and sources.
