# Lexicon-Based Sentiment Classifier (v0.1.0 Baseline)

## Overview

This is a **simple baseline sentiment classifier** written in pure Mojo. It does **not use MAX Graph or MAX Engine** - it's a straightforward lexicon-based approach using word counting.

**Purpose**: This served as the v0.1.0 baseline before implementing MAX Graph examples. It's preserved here as:
- A working example of pure Mojo (no MAX dependencies)
- A baseline for comparing with MAX Graph implementations
- Reference for simple sentiment analysis approach

## What It Does

Classifies text sentiment as POSITIVE or NEGATIVE by:
1. Loading a simple word lexicon (positive/negative words)
2. Counting positive and negative words in input text
3. Returning sentiment based on word count difference

**Algorithm**: Simple word matching, no ML model involved.

## Not Part of MAX Graph Learning

⚠️ **Important**: This example does **not** teach MAX Graph concepts. For MAX Graph examples, see:
- `examples/python/01_elementwise/` - Start here for MAX Graph
- `examples/python/02_linear_layer/` - Continue with this
- And so on through the numbered progression

## Files

- `main.mojo` - Entry point
- `classifier.mojo` - Sentiment classification logic
- `cli.mojo` - Command-line argument parsing
- `config.mojo` - Configuration loading (TOML)
- `config.toml` - Configuration file
- `utils.mojo` - Utility functions

## Running

```bash
# Build
pixi run mojo-build

# Run
pixi run mojo-run "This is great!"
```

Or directly with mojo:
```bash
mojo run examples/mojo/lexicon_baseline/main.mojo "This is great!"
```

## Dependencies

- `mojo-toml` - For TOML configuration parsing
- `mojo-dotenv` - For environment variable loading

Note: These are specified in the MOJO_PATH in pixi.toml.

## Architecture

```
Input Text
    ↓
Tokenize (split on whitespace)
    ↓
Count Positive Words
Count Negative Words
    ↓
Compare Counts
    ↓
Return: POSITIVE | NEGATIVE | NEUTRAL
```

## Example Output

```
═════════════════════════════════════════════════
SENTIMENT ANALYSIS RESULT
═════════════════════════════════════════════════
Input: This is great!

Label: POSITIVE
Confidence: 0.85
Score: 1.0
═════════════════════════════════════════════════
```

## Limitations

1. **No ML model**: Just word counting
2. **Simple lexicon**: Limited vocabulary
3. **No context**: Doesn't understand negation, sarcasm, etc.
4. **No preprocessing**: No stemming, lemmatisation, etc.
5. **English only**: Lexicon is English words

## Why This Exists

This was created as a **v0.1.0 baseline** to:
- Establish basic infrastructure (config, CLI, etc.)
- Have something working before implementing MAX Graph
- Provide a comparison point for MAX Graph implementations

For actual sentiment analysis with ML models, see `examples/python/03_distilbert_sentiment/` which uses MAX Graph with a DistilBERT transformer.

## Comparison with MAX Graph Examples

| Feature | Lexicon Baseline | DistilBERT (MAX Graph) |
|---------|-----------------|------------------------|
| Language | Pure Mojo | Python (MAX Graph) |
| Approach | Word counting | Transformer neural network |
| Accuracy | Low (basic) | High (66M parameters) |
| Speed | Very fast | 45ms per inference |
| Dependencies | None (pure Mojo) | MAX Engine, transformers |
| Use case | Educational baseline | Production inference |

## Future

This baseline will remain as a reference, but the focus of this repository is on MAX Graph examples. For Mojo + MAX Graph examples, see the planned:
- `examples/mojo/01_elementwise/` (future)
- `examples/mojo/02_linear_layer/` (future)

See `docs/planning/MOJO_EXAMPLES_PLAN.md` for details.

## Version

This code is from **v0.1.0** of the max-learning repository. It has not been updated since then as the focus shifted to MAX Graph.

## Licence

MIT Licence - see repository root LICENCE file
