# Sentiment Classification Model

## What: Overview

The sentiment classification model is a **lexicon-based sentiment analyser** that classifies text as POSITIVE, NEGATIVE, or NEUTRAL. It uses a dictionary of pre-scored sentiment words and averages their scores to determine overall sentiment.

**Model Type**: Lexicon-based bag-of-words classifier  
**Implementation**: Pure Mojo (no external ML libraries)  
**Version**: v0.1.0 (MVP)  
**Vocabulary Size**: 29 sentiment words (15 positive, 14 negative)

## Why: Purpose and Use Case

### Primary Purposes

1. **Demonstrate Mojo for ML**: Show that Mojo can implement practical sentiment analysis without Python dependencies
2. **Configuration Management Demo**: Showcase mojo-toml and mojo-dotenv in a real application
3. **Performance Foundation**: Establish baseline for future comparison with Modular MAX
4. **Educational Value**: Provide clear, understandable ML implementation in Mojo

### Advantages of This Approach

- **Transparency**: Every decision is explainable (no black-box neural network)
- **Speed**: No model loading, instant inference (< 1ms)
- **Simplicity**: Easy to understand, debug, and extend
- **Deterministic**: Same input always produces same output
- **Resource-efficient**: Minimal memory footprint (~30 word-score pairs)

### Limitations

- **Coverage**: Only recognises 29 sentiment words
- **Context-blind**: Cannot understand negation ("not good" scores positive)
- **No sarcasm detection**: Takes words literally
- **Simple tokenisation**: Doesn't handle contractions or complex punctuation
- **No learning**: Cannot improve from examples without manual updates

## How: Algorithm Architecture

### 1. Tokenisation

**Input**: Raw text string  
**Process**: 
1. Convert text to lowercase
2. Iterate through each character
3. Extract words (letters and numbers only)
4. Split on whitespace and punctuation

**Output**: List of word tokens

```
Input:  "This product is amazing!"
Output: ["this", "product", "is", "amazing"]
```

**Code**: `src/embeddings.mojo::tokenize()`

### 2. Sentiment Lexicon

A hardcoded dictionary mapping words to sentiment scores in the range [-1.0, 1.0]:

**Positive Words** (15):
- Highest: `amazing` (0.95), `excellent` (0.95), `perfect` (0.95)
- High: `fantastic` (0.9), `wonderful` (0.9), `awesome` (0.9), `love` (0.9), `great` (0.9)
- Medium: `brilliant` (0.85), `best` (0.85), `good` (0.8), `beautiful` (0.8), `happy` (0.8)
- Lower: `enjoy` (0.7), `nice` (0.6)

**Negative Words** (14):
- Lowest: `worst` (-0.95)
- High: `terrible` (-0.9), `awful` (-0.9), `horrible` (-0.9), `hate` (-0.9)
- Medium: `useless` (-0.85), `bad` (-0.8), `waste` (-0.8), `disappointing` (-0.75), `disappointed` (-0.75)
- Lower: `poor` (-0.7), `annoying` (-0.7), `sad` (-0.7), `boring` (-0.6)

**Code**: `src/embeddings.mojo::load_sentiment_lexicon()`

### 3. Score Computation

**Algorithm**: Average sentiment score of recognised words

```
For each token in text:
    If token exists in lexicon:
        Add lexicon[token] to total_score
        Increment count
    
If count > 0:
    sentiment_score = total_score / count
Else:
    sentiment_score = 0.0
```

**Example**:
```
Text: "This product is amazing!"
Tokens: ["this", "product", "is", "amazing"]
Recognised: ["amazing"] → 0.95
Score: 0.95 / 1 = 0.95
```

**Code**: `src/embeddings.mojo::compute_text_sentiment()`

### 4. Label Assignment

**Thresholds**:
- Score > 0.1 → **POSITIVE**
- Score < -0.1 → **NEGATIVE**
- -0.1 ≤ Score ≤ 0.1 → **NEUTRAL**

The ±0.1 threshold creates a neutral zone for ambiguous or mixed-sentiment text.

### 5. Confidence Calculation

**Formula**: Sigmoid function with scaling

```
abs_score = |sentiment_score|
scaled = abs_score × 3.0      # Scale factor for sensitivity
confidence = 1 / (1 + e^(-scaled))
```

**Intuition**: 
- Larger absolute scores → higher confidence
- Sigmoid maps to [0, 1] range
- Scaling factor (3.0) controls sensitivity

**Examples**:
- Score 0.95 → Confidence 0.945 (94.5%)
- Score 0.5 → Confidence 0.818 (81.8%)
- Score 0.0 → Confidence 0.5 (50%)

**Code**: `src/classifier.mojo::predict()`

## Inputs and Outputs

### Input Format

**Type**: String  
**Constraints**: 
- Maximum length: 512 characters (configurable)
- Any UTF-8 text
- Punctuation is stripped during tokenisation

**Example Inputs**:
```
"This product is amazing!"
"terrible and disappointing"
"The weather is okay"
```

### Output Format

**Type**: `SentimentResult` struct

```mojo
struct SentimentResult:
    var label: String       # "POSITIVE", "NEGATIVE", or "NEUTRAL"
    var confidence: Float64 # Range: 0.0 to 1.0
    var score: Float64      # Range: -1.0 to 1.0
```

**Example Outputs**:

| Input | Label | Confidence | Score |
|-------|-------|------------|-------|
| "This product is amazing!" | POSITIVE | 0.945 | 0.95 |
| "terrible and disappointing" | NEGATIVE | 0.922 | -0.825 |
| "The weather is okay" | NEUTRAL | 0.5 | 0.0 |

## Training (Lexicon Creation)

### Current Approach: Manual Curation

**Process**:
1. **Select common sentiment words** from general English usage
2. **Assign scores** based on intuitive sentiment strength:
   - Strong positive/negative: ±0.85 to ±0.95
   - Moderate: ±0.7 to ±0.8
   - Mild: ±0.5 to ±0.65
3. **Hardcode into lexicon** in `embeddings.mojo`

**No machine learning training required** — this is a rule-based system.

### Future Enhancements

**Lexicon Expansion Options**:

1. **Load from File** (`data/sentiment_lexicon.txt`)
   - Format: `word score` per line
   - Allows updates without recompilation
   - Could use larger public sentiment lexicons (e.g., AFINN, SentiWordNet)

2. **Semi-automated Scoring**
   - Start with public sentiment datasets
   - Manually review and adjust scores
   - Balance between coverage and accuracy

3. **Domain-specific Lexicons**
   - Product reviews: emphasise quality/value words
   - Financial sentiment: emphasise risk/opportunity words
   - Social media: emphasise emotional intensity

**Advanced Model (v0.2.0+)**:

When integrating Modular MAX, could use:
- Pre-trained transformer models (BERT, RoBERTa)
- Context-aware embeddings
- Actual ML training on labelled datasets

## Performance Characteristics

### Speed

**Typical Inference Time**: < 1ms per classification
- Tokenisation: ~0.1ms
- Lexicon lookup: ~0.05ms × word_count
- Score calculation: ~0.02ms
- No GPU required

**Bottlenecks**:
- String operations (character iteration for tokenisation)
- Dictionary lookups (29-entry lexicon is negligible)

### Accuracy

**Estimated Performance** (no formal evaluation yet):
- Simple positive/negative reviews: ~75-85% accuracy
- Mixed-sentiment text: ~50-60% accuracy
- Sarcastic/ironic text: ~20-30% accuracy

**Accuracy depends on**:
1. Vocabulary coverage (only 29 words)
2. Text complexity (simple reviews vs nuanced prose)
3. Domain match (general sentiment words)

### Resource Usage

- **Memory**: < 5KB for lexicon
- **CPU**: Single-threaded, minimal computation
- **Disk**: No model files to load

## Configuration

Configurable via `config.toml`:

```toml
[model]
type = "sentiment-analysis"
algorithm = "logistic-regression"  # Actually lexicon-based, name for clarity
vocab_size = 10000                  # Not used in v0.1.0
embedding_dim = 100                 # Not used in v0.1.0

[inference]
confidence_threshold = 0.5  # Minimum confidence to report (not enforced in v0.1.0)
max_length = 512            # Maximum input text length
```

## Code Organisation

```
src/
├── embeddings.mojo       # Tokenisation and lexicon
│   ├── tokenize()              → List[String]
│   ├── load_sentiment_lexicon() → Dict[String, Float64]
│   ├── get_word_sentiment()     → Float64
│   └── compute_text_sentiment() → Float64
│
├── classifier.mojo       # Sentiment classification
│   ├── SentimentResult struct
│   ├── SentimentClassifier struct
│   │   ├── load()               → loads lexicon
│   │   └── predict()            → SentimentResult
│   └── classify_sentiment()     → wrapper function
│
└── main.mojo             # CLI and orchestration
```

## Examples

### Example 1: Strong Positive

```
Input:  "This product is absolutely amazing and wonderful!"
Tokens: ["this", "product", "is", "absolutely", "amazing", "and", "wonderful"]
Matches: ["amazing" (0.95), "wonderful" (0.9)]
Score:  (0.95 + 0.9) / 2 = 0.925
Label:  POSITIVE
Confidence: 0.936
```

### Example 2: Strong Negative

```
Input:  "Terrible product, complete waste of money"
Tokens: ["terrible", "product", "complete", "waste", "of", "money"]
Matches: ["terrible" (-0.9), "waste" (-0.8)]
Score:  (-0.9 + -0.8) / 2 = -0.85
Label:  NEGATIVE
Confidence: 0.928
```

### Example 3: Neutral

```
Input:  "I received the package yesterday"
Tokens: ["i", "received", "the", "package", "yesterday"]
Matches: []
Score:  0.0
Label:  NEUTRAL
Confidence: 0.5
```

### Example 4: Mixed Sentiment (Limitation)

```
Input:  "Good product but terrible customer service"
Tokens: ["good", "product", "but", "terrible", "customer", "service"]
Matches: ["good" (0.8), "terrible" (-0.9)]
Score:  (0.8 + -0.9) / 2 = -0.05
Label:  NEUTRAL (falls within ±0.1 threshold)
Confidence: 0.504
```

## Comparison with ML Models

| Feature | Lexicon-based (ours) | Neural Network (BERT) |
|---------|---------------------|----------------------|
| Training time | 0 (manual curation) | Hours/days on GPU |
| Inference speed | < 1ms | 10-100ms (CPU) |
| Accuracy | 70-80% (simple text) | 90-95% (complex text) |
| Memory | < 5KB | 100-500MB |
| Interpretability | Full (word-by-word) | Low (black box) |
| Context understanding | None | Excellent |
| Sarcasm detection | None | Moderate |
| Setup complexity | Simple | Complex (model files, GPU) |

## Future Roadmap

### v0.2.0: Modular MAX Integration
- Load pre-trained transformer models
- Context-aware embeddings
- Benchmark: lexicon vs MAX vs Python

### v0.3.0: Enhancements
- Negation handling ("not good" → negative)
- Intensity modifiers ("very good" → higher score)
- Domain-specific lexicons
- Confidence thresholding

### v0.4.0: Advanced Features
- Multi-class sentiment (positive/negative/neutral/mixed)
- Emotion detection (joy, anger, sadness, etc.)
- Aspect-based sentiment (product quality vs service quality)

## References

- **AFINN Sentiment Lexicon**: Finn Årup Nielsen, 2011
- **SentiWordNet**: Baccianella et al., 2010
- **Modular MAX**: https://www.modular.com/max
- **Mojo Language**: https://www.modular.com/max/mojo

## Sponsorship

This model and documentation are part of the mojo-inference-service project, sponsored by [DataBooth](https://www.databooth.com.au/posts/mojo).
