# Data Directory

This directory contains word embeddings and training data for the sentiment classifier.

## MVP (v0.1.0) - Simple Embeddings

For the initial version, we use a small set of pre-computed word embeddings for common sentiment words.

### Structure

```
data/
├── embeddings.txt          # Simple word vectors (text format)
├── vocab.txt               # Vocabulary list
└── sentiment_lexicon.txt   # Positive/negative words
```

### Generating Embeddings

The embeddings will be generated programmatically in Mojo or loaded from a simple text file:

```
# Format: word score
happy 0.9
sad -0.8
amazing 0.95
terrible -0.9
...
```

## Future (v0.2.0+) - Advanced Embeddings

When we integrate Modular MAX, we can use:
- Pre-trained embeddings (GloVe, Word2Vec, FastText)
- Contextual embeddings (BERT, etc.)
- Model-specific tokenizers

## Note

Data files are gitignored to keep the repository lightweight. They'll be generated during setup or downloaded separately.
