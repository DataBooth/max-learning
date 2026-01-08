# MVP Sentiment Lexicon

## Overview

This directory contains the manually curated sentiment lexicon used in mojo-inference-service v0.1.0.

**File**: `sentiment_lexicon.txt`  
**Size**: 29 words (15 positive, 14 negative)  
**Format**: `word score` (tab or space separated)  
**Score Range**: -1.0 (most negative) to 1.0 (most positive)

## Purpose

- **Simplicity**: Small, understandable vocabulary for MVP
- **Transparency**: Every word manually selected and scored
- **Speed**: Minimal memory footprint, instant loading
- **Control**: Full control over sentiment interpretations

## Lexicon Contents

### Positive Words (15)

- amazing, excellent, perfect: 0.95
- fantastic, wonderful, awesome, love, great: 0.9
- brilliant, best: 0.85
- good, beautiful, happy: 0.8
- enjoy: 0.7
- nice: 0.6

### Negative Words (14)

- worst: -0.95
- terrible, awful, horrible, hate: -0.9
- useless: -0.85
- bad, waste: -0.8
- disappointing, disappointed: -0.75
- poor, annoying, sad: -0.7
- boring: -0.6

## Usage

Load in Mojo via `src/embeddings.mojo`:

```mojo
var lexicon = load_sentiment_lexicon("data/mvp/sentiment_lexicon.txt")
```

## Updating

Edit `sentiment_lexicon.txt` directly. Format: one word per line with score separated by space or tab. Comments start with `#`.

## Limitations

- Only 29 words vs 2,476 in AFINN
- No negation or intensifier handling
- General sentiment, not domain-specific

For broader coverage, see `../afinn/`.
