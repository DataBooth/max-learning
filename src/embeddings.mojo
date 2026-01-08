"""
Word embeddings and tokenization module.

Simple tokenizer and sentiment lexicon for MVP.
Future versions will support advanced embeddings (GloVe, Word2Vec, etc.)
"""

from logger import Logger, Level


fn tokenize(text: String) -> List[String]:
    """
    Simple word tokenizer.
    
    Args:
        text: Input text to tokenize.
    
    Returns:
        List of tokens (words).
    
    Note:
        MVP implementation: Basic whitespace splitting and lowercasing.
        Future: Handle punctuation, contractions, special characters.
    """
    var tokens = List[String]()
    
    # Convert to lowercase
    var lower_text = text.lower()
    
    # Simple split on whitespace and common punctuation
    var current_word = String("")
    
    for i in range(len(lower_text)):
        var c = lower_text[i]
        
        # Check if it's a letter or number
        if (c >= 'a' and c <= 'z') or (c >= '0' and c <= '9'):
            current_word += c
        else:
            # End of word
            if len(current_word) > 0:
                tokens.append(current_word)
                current_word = String("")
    
    # Add last word if exists
    if len(current_word) > 0:
        tokens.append(current_word)
    
    return tokens^


fn load_sentiment_lexicon(path: String = "data/sentiment_lexicon.txt") raises -> Dict[String, Float64]:
    """
    Load sentiment lexicon from file.
    
    Args:
        path: Path to sentiment lexicon file.
    
    Returns:
        Dictionary mapping words to sentiment scores (-1.0 to 1.0).
    
    Raises:
        Error if file cannot be read.
    
    Format:
        word score
        happy 0.9
        sad -0.8
    """
    var log = Logger[Level.INFO]()
    log.info("Loading sentiment lexicon...")
    
    # For MVP, create hardcoded lexicon with common sentiment words
    var lexicon = Dict[String, Float64]()
    
    # Positive words
    lexicon["good"] = 0.8
    lexicon["great"] = 0.9
    lexicon["excellent"] = 0.95
    lexicon["amazing"] = 0.95
    lexicon["wonderful"] = 0.9
    lexicon["fantastic"] = 0.9
    lexicon["love"] = 0.9
    lexicon["best"] = 0.85
    lexicon["awesome"] = 0.9
    lexicon["perfect"] = 0.95
    lexicon["brilliant"] = 0.85
    lexicon["beautiful"] = 0.8
    lexicon["happy"] = 0.8
    lexicon["enjoy"] = 0.7
    lexicon["nice"] = 0.6
    
    # Negative words
    lexicon["bad"] = -0.8
    lexicon["terrible"] = -0.9
    lexicon["awful"] = -0.9
    lexicon["horrible"] = -0.9
    lexicon["worst"] = -0.95
    lexicon["hate"] = -0.9
    lexicon["poor"] = -0.7
    lexicon["disappointing"] = -0.75
    lexicon["disappointed"] = -0.75
    lexicon["useless"] = -0.85
    lexicon["waste"] = -0.8
    lexicon["sad"] = -0.7
    lexicon["boring"] = -0.6
    lexicon["annoying"] = -0.7
    
    log.info("Loaded", len(lexicon), "sentiment words")
    return lexicon^


fn get_word_sentiment(word: String, lexicon: Dict[String, Float64]) raises -> Float64:
    """
    Get sentiment score for a word.
    
    Args:
        word: Word to look up.
        lexicon: Sentiment lexicon dictionary.
    
    Returns:
        Sentiment score (0.0 if word not in lexicon).
    """
    if word in lexicon:
        try:
            return lexicon[word]
        except:
            return 0.0
    return 0.0


fn compute_text_sentiment(tokens: List[String], lexicon: Dict[String, Float64]) raises -> Float64:
    """
    Compute overall sentiment score for text.
    
    Args:
        tokens: List of word tokens.
        lexicon: Sentiment lexicon dictionary.
    
    Returns:
        Aggregated sentiment score.
    
    Note:
        MVP: Simple average of word sentiments.
        Future: Weighted averaging, negation handling, etc.
    """
    var total_score = 0.0
    var count = 0
    
    for i in range(len(tokens)):
        var score = get_word_sentiment(tokens[i], lexicon)
        if score != 0.0:
            total_score += score
            count += 1
    
    if count > 0:
        return total_score / Float64(count)
    
    return 0.0
