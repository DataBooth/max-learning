# AFINN Sentiment Lexicon

## Overview

AFINN is a widely-used sentiment lexicon developed by Finn Årup Nielsen. This directory contains the **AFINN-111** version downloaded from the official repository.

**File**: `AFINN-111.txt`  
**Size**: 2,476 words  
**Format**: `word\tscore` (tab-separated)  
**Score Range**: -5 (most negative) to +5 (most positive)  
**Licence**: Open Data Commons Open Database Licence (ODbL) v1.0

## Source

- **Repository**: https://github.com/fnielsen/afinn
- **Download**: https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-111.txt
- **Citation**: Finn Årup Nielsen, "A new ANEW: evaluation of a word list for sentiment analysis in microblogs", 2011

## Differences from MVP Lexicon

| Feature | MVP | AFINN-111 |
|---------|-----|-----------|
| Size | 29 words | 2,476 words |
| Coverage | Basic | Comprehensive |
| Score range | -1.0 to 1.0 | -5 to +5 |
| Curation | Manual (general) | Research-based (social media) |
| Format | Space/tab | Tab only |

## Using AFINN with mojo-inference-service

### Option 1: Convert to MVP Format

```bash
# Normalise scores from [-5, 5] to [-1.0, 1.0]
awk '{print $1, $2/5.0}' AFINN-111.txt > ../mvp/sentiment_lexicon_afinn.txt
```

### Option 2: Modify Code to Load AFINN

Update `src/embeddings.mojo::load_sentiment_lexicon()`:

```mojo
# Read AFINN format (tab-separated)
# Normalise scores: divide by 5.0 to get [-1.0, 1.0] range
```

## Sample Entries

```
abandon	-2
awesome	4
terrible	-3
love	3
hate	-3
```

## Advantages

- **Comprehensive**: 85× more words than MVP
- **Research-backed**: Validated on social media text
- **Battle-tested**: Used in thousands of sentiment analysis projects
- **Maintained**: Active repository with updates

## Disadvantages

- **Integer scores**: Less granular than MVP's float scores
- **Social media bias**: Optimised for Twitter/microblogs
- **Larger memory**: ~100KB vs ~1KB for MVP

## References

- Nielsen, F. Å. (2011). "A new ANEW: Evaluation of a word list for sentiment analysis in microblogs". Proceedings of the ESWC2011 Workshop on 'Making Sense of Microposts'.
- AFINN GitHub: https://github.com/fnielsen/afinn
- Licence: https://opendatacommons.org/licenses/odbl/1.0/
