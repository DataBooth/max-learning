# Models Directory

This directory contains **large pre-trained models** downloaded from HuggingFace or other external sources.

## Design Philosophy

**models/** vs **examples/.../weights/**:
- `models/` = Large, complex, pre-trained models (transformers, foundation models)
- `examples/.../weights/` = Small, simple, locally-trained weights (MLP, CNN, RNN)

This separation keeps:
- Simple examples self-contained and reproducible
- Large models centralized and cached
- Clear distinction between "training from scratch" examples and "pre-trained model inference" examples

For example:
- MLP (example 03): Trains small network on California housing → stores weights locally
- CNN (example 04): Trains small CNN on MNIST → stores weights locally  
- DistilBERT (example 06): Uses pre-trained 66M parameter model → downloads to models/

## Current Models

### DistilBERT Sentiment (SST-2)

**Model**: `distilbert-base-uncased-finetuned-sst-2-english`  
**Source**: HuggingFace  
**Format**: SafeTensors (for MAX Graph)  
**Size**: ~268MB  
**Parameters**: 66M  
**Classes**: 2 (POSITIVE, NEGATIVE)  
**Accuracy**: ~91% on SST-2 test set

**Licence**: Apache 2.0  
**Citation**: Sanh et al., 2019 - "DistilBERT, a distilled version of BERT"

## Download Instructions

### Automatic Download (Recommended)

```bash
pixi run download-models
```

This will:
1. Download the model from HuggingFace
2. Save in SafeTensors format (for MAX Graph)
3. Place files in `models/distilbert-sentiment/`

**Note**: Models are automatically downloaded when running `pixi run example-distilbert` or `pixi run benchmark-distilbert`.

## Directory Structure

```
models/
├── README.md                    # This file
├── download_models.sh           # Download wrapper script
├── download_models.py           # Python download script
└── distilbert-sentiment/       # Downloaded model (gitignored)
    ├── model.safetensors        # Model weights (SafeTensors format)
    ├── vocab.txt                # Vocabulary (30,522 tokens)
    ├── config.json              # Model configuration
    └── tokenizer_config.json    # Tokenizer settings
```

## Using Models with MAX Graph

Models are loaded in `src/python/max_distilbert/inference.py`:

```python
from max.graph.weights import load_weights

# Load SafeTensors weights
weights = load_weights(["models/distilbert-sentiment/model.safetensors"])
```

## Model Performance

| Model | Params | Size | Inference (CPU) | Accuracy |
|-------|--------|------|-----------------|----------|
| Lexicon (MVP) | 29 words | <5KB | <1ms | ~70-80% |
| DistilBERT | 66M | 260MB | ~50-100ms | ~91% |

*Note: Inference times are approximate and depend on hardware.*

## Adding New Models

To add a new sentiment model:

1. Download from HuggingFace with SafeTensors format
2. Update `src/python/max_distilbert/` to support the new model
3. Add configuration in example/benchmark config files
4. Update this README

## Troubleshooting

### Model Not Found

If you see "Model not found" errors:
- Run `pixi run download-models` to download
- Check that `models/distilbert-sentiment/model.safetensors` exists
- Verify file permissions

### Download Fails

If model download fails:
- Ensure you have internet connection
- Check HuggingFace is accessible
- Verify `transformers` package is installed in pixi environment

### MAX Graph Load Error

If MAX Graph can't load the model:
- Verify SafeTensors file exists and is valid
- Check MAX version compatibility (requires MAX 25.1.0+)
- Review `src/python/max_distilbert/inference.py` for correct paths

## References

- HuggingFace Model: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
- DistilBERT Paper: https://arxiv.org/abs/1910.01108
- MAX Engine Docs: https://docs.modular.com/engine/
