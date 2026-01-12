"""
Sentiment classifier module.

Simple sentiment analysis using logistic regression in pure Mojo.
MVP v0.1.0 - No external model files required.
"""

from logger import Logger, Level
from math import exp
from embeddings import tokenize, load_sentiment_lexicon, compute_text_sentiment


struct SentimentResult(Copyable, Movable):
    """Result from sentiment classification."""
    var label: String  # "POSITIVE" or "NEGATIVE"
    var confidence: Float64  # 0.0 to 1.0
    var score: Float64  # Raw sentiment score

    fn __init__(out self, label: String, confidence: Float64, score: Float64):
        self.label = label
        self.confidence = confidence
        self.score = score


struct SentimentClassifier(Copyable, Movable):
    """
    Simple sentiment classifier using word-based scoring.

    MVP implementation: Uses a sentiment lexicon (positive/negative word scores)
    to classify text. This is a lightweight approach that demonstrates:
    - Pure Mojo implementation
    - Configuration via mojo-toml
    - Performance benchmarking

    Future (v0.2.0): Replace with Modular MAX for advanced models.
    """
    var confidence_threshold: Float64
    var max_length: Int
    var loaded: Bool
    var lexicon: Dict[String, Float64]

    fn __init__(out self, confidence_threshold: Float64, max_length: Int):
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length
        self.loaded = False
        self.lexicon = Dict[String, Float64]()

    fn load(mut self) raises:
        """
        Load the classifier (initialize sentiment lexicon).

        For MVP, this creates a simple in-memory sentiment dictionary.
        Future versions will load from data/ directory.
        """
        var log = Logger[Level.INFO]()
        log.info("Loading sentiment classifier...")

        # Load sentiment lexicon
        self.lexicon = load_sentiment_lexicon()

        self.loaded = True
        log.info("Classifier loaded successfully")

    fn predict(self, input_text: String) raises -> SentimentResult:
        """
        Classify text sentiment.

        Args:
            input_text: Text to classify.

        Returns:
            SentimentResult with prediction.

        Raises:
            Error if classifier not loaded.
        """
        if not self.loaded:
            raise Error("Classifier not loaded. Call load() first.")

        var log = Logger[Level.DEBUG]()
        log.debug("Classifying text:", input_text)

        # 1. Tokenize input
        var tokens = tokenize(input_text)
        log.debug("Tokens:", len(tokens))

        # 2. Compute sentiment score
        var score = compute_text_sentiment(tokens, self.lexicon)
        log.debug("Raw score:", score)

        # 3. Determine label based on score
        var label: String
        if score > 0.1:
            label = "POSITIVE"
        elif score < -0.1:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        # 4. Calculate confidence using sigmoid
        # Map score to confidence: larger absolute scores = higher confidence
        var abs_score = score if score > 0.0 else -score
        var scaled_score = abs_score * 3.0  # Scale factor for sensitivity
        var confidence = 1.0 / (1.0 + exp(-scaled_score))  # Sigmoid

        return SentimentResult(label, confidence, score)


fn classify_sentiment(
    classifier: SentimentClassifier,
    text: String,
) raises -> SentimentResult:
    """
    High-level sentiment classification function.

    Args:
        classifier: Loaded classifier instance.
        text: Input text for classification.

    Returns:
        SentimentResult with prediction.
    """
    return classifier.predict(text)
