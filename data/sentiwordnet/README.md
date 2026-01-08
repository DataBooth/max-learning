# SentiWordNet Lexicon

## Overview

SentiWordNet is a lexical resource for opinion mining based on WordNet. Each synset (set of synonyms) in WordNet is assigned positivity, negativity, and objectivity scores.

**Status**: Not yet downloaded  
**Size**: ~117,000 entries (synsets)  
**Score Range**: Positivity, Negativity, Objectivity scores (each 0.0 to 1.0)  
**Licence**: Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)

## Source

- **Website**: https://github.com/aesuli/SentiWordNet
- **Download**: http://sentiwordnet.isti.cnr.it/  
- **Citation**: Baccianella, S., Esuli, A., & Sebastiani, F. (2010). "SentiWordNet 3.0: An Enhanced Lexical Resource for Sentiment Analysis and Opinion Mining"

## Format

```
# SentiWordNet format: POS ID PosScore NegScore SynsetTerms Gloss
a 00001740 0.125 0 able#1 (usually followed by `to') having the necessary means...
a 00002098 0 0.75 unable#1 (usually followed by `to') not having the necessary means...
```

**Fields**:
- POS: Part of speech (a=adjective, n=noun, v=verb, r=adverb)
- ID: WordNet synset offset
- PosScore: Positivity score [0, 1]
- NegScore: Negativity score [0, 1]
- SynsetTerms: Words in synset with sense numbers
- Gloss: WordNet definition

## Why Not Included by Default

1. **Complexity**: Requires parsing WordNet format and handling synsets
2. **Size**: 117K+ entries vs 2,476 for AFINN
3. **Licence**: Requires attribution
4. **Parsing overhead**: More complex format than simple word-score pairs

## Using SentiWordNet

### Download

```bash
cd data/sentiwordnet
wget https://github.com/aesuli/sentiwordnet/raw/master/data/SentiWordNet_3.0.0.txt
```

### Convert to Simple Format

Extract first sense of each word:

```python
# Python script to convert SentiWordNet to simple format
import re

with open('SentiWordNet_3.0.0.txt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) >= 5:
            pos_score = float(parts[2])
            neg_score = float(parts[3])
            synset = parts[4].split()[0]  # First term
            word = synset.split('#')[0]
            
            # Calculate net sentiment
            net_score = pos_score - neg_score
            
            if abs(net_score) > 0.1:  # Filter weak sentiment
                print(f"{word}\t{net_score:.3f}")
```

## Advantages Over AFINN

- **Part-of-speech aware**: Different scores for "good" (adj) vs "good" (noun)
- **Sense-specific**: "bank" (financial) vs "bank" (river)
- **Comprehensive**: Covers most English words
- **Graded sentiment**: More nuanced than integer scores

## Disadvantages

- **Complexity**: Requires WordNet understanding
- **Weak signals**: Many words have near-zero scores
- **Not colloquial**: Less effective for social media/informal text
- **Larger footprint**: Significant memory usage

## References

- Baccianella, S., Esuli, A., & Sebastiani, F. (2010). "SentiWordNet 3.0: An Enhanced Lexical Resource for Sentiment Analysis and Opinion Mining". LREC 2010.
- WordNet: https://wordnet.princeton.edu/
- Licence: https://creativecommons.org/licenses/by-sa/3.0/
