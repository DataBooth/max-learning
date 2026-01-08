# Models Directory

This directory contains pre-trained transformer models for sentiment analysis, loaded via MAX Engine.

## Current Models

### DistilBERT Sentiment (SST-2)

**Model**: `distilbert-base-uncased-finetuned-sst-2-english`  
**Source**: HuggingFace  
**Size**: ~260MB  
**Parameters**: 66M  
**Classes**: 2 (POSITIVE, NEGATIVE)  
**Accuracy**: ~91% on SST-2 test set

**Licence**: Apache 2.0  
**Citation**: Sanh et al., 2019 - "DistilBERT, a distilled version of BERT"

## Download Instructions

### Automatic Download (Recommended)

```bash
./models/download_models.sh
```

This script will:
1. Check if `modular` Python package is available
2. Download the model from HuggingFace
3. Convert to ONNX format for MAX Engine
4. Place files in `models/distilbert-sentiment/`

### Manual Download

If you prefer to download manually:

```bash
# Install dependencies
pip install transformers torch onnx

# Download and convert
python models/convert_to_onnx.py
```

## Directory Structure

```
models/
├── README.md                    # This file
├── download_models.sh           # Download script
├── convert_to_onnx.py          # Conversion utility
└── distilbert-sentiment/       # Downloaded model (gitignored)
    ├── model.onnx               # ONNX format for MAX Engine
    ├── vocab.txt                # Vocabulary (30,522 tokens)
    ├── config.json              # Model configuration
    └── tokenizer_config.json    # Tokenizer settings
```

## Using Models with MAX Engine

Models are loaded in `src/max_classifier.mojo`:

```mojo
from max import engine

var session = engine.InferenceSession()
var model = session.load("models/distilbert-sentiment/model.onnx")
```

## Model Performance

| Model | Params | Size | Inference (CPU) | Accuracy |
|-------|--------|------|-----------------|----------|
| Lexicon (MVP) | 29 words | <5KB | <1ms | ~70-80% |
| DistilBERT | 66M | 260MB | ~50-100ms | ~91% |

*Note: Inference times are approximate and depend on hardware.*

## Adding New Models

To add a new sentiment model:

1. Download from HuggingFace
2. Convert to ONNX: `python -m transformers.onnx --model=<model-name> models/<model-name>/`
3. Update `src/max_classifier.mojo` to support the new model
4. Add configuration in `config.toml`
5. Update this README

## Troubleshooting

### Model Not Found

If you see "Model not found" errors:
- Run `./models/download_models.sh` to download
- Check that `models/distilbert-sentiment/model.onnx` exists
- Verify file permissions

### ONNX Conversion Fails

If ONNX conversion fails:
- Install required packages: `pip install transformers torch onnx optimum`
- Use the optimum library: `optimum-cli export onnx --model distilbert-base-uncased-finetuned-sst-2-english models/distilbert-sentiment/`

### MAX Engine Load Error

If MAX Engine can't load the model:
- Verify ONNX file is valid: `python -m onnx models/distilbert-sentiment/model.onnx`
- Check MAX Engine version compatibility
- Try PyTorch format instead: keep `pytorch_model.bin`

## References

- HuggingFace Model: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
- DistilBERT Paper: https://arxiv.org/abs/1910.01108
- MAX Engine Docs: https://docs.modular.com/engine/
