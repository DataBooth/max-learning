"""Tests for DistilBERT sentiment classifier."""
import sys
from pathlib import Path

import pytest

# Import from installed packages
from utils.paths import get_models_dir
from max_distilbert import DistilBertSentimentClassifier


@pytest.fixture(scope="module")
def model_path():
    """Get path to the DistilBERT model."""
    path = get_models_dir() / "distilbert-sentiment"
    if not path.exists():
        pytest.skip(f"Model not found at {path}. Run: models/download_models.sh")
    return path


@pytest.fixture(scope="module")
def classifier(model_path):
    """Load the classifier once for all tests."""
    return DistilBertSentimentClassifier(model_path)


class TestDistilBertSentimentClassifier:
    """Tests for DistilBERT sentiment classifier."""

    def test_classifier_initializes(self, classifier):
        """Test that classifier initializes successfully."""
        assert classifier is not None
        assert classifier.model is not None
        assert classifier.tokenizer is not None
        assert classifier.config is not None

    def test_positive_sentiment(self, classifier):
        """Test prediction on clearly positive text."""
        result = classifier.predict("This movie was absolutely fantastic! I loved it.")
        
        assert result["label"] == "POSITIVE"
        assert result["confidence"] > 0.9
        assert result["positive_score"] > result["negative_score"]

    def test_negative_sentiment(self, classifier):
        """Test prediction on clearly negative text."""
        result = classifier.predict("Terrible experience. Would not recommend at all.")
        
        assert result["label"] == "NEGATIVE"  
        assert result["confidence"] > 0.9
        assert result["negative_score"] > result["positive_score"]

    def test_result_structure(self, classifier):
        """Test that result dict has correct structure."""
        result = classifier.predict("Test text")
        
        assert isinstance(result, dict)
        assert "label" in result
        assert "confidence" in result
        assert "positive_score" in result
        assert "negative_score" in result
        
        assert result["label"] in ["POSITIVE", "NEGATIVE"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["positive_score"] <= 1.0
        assert 0.0 <= result["negative_score"] <= 1.0

    def test_scores_sum_to_one(self, classifier):
        """Test that probability scores sum to approximately 1.0."""
        result = classifier.predict("Test text")
        
        total = result["positive_score"] + result["negative_score"]
        assert abs(total - 1.0) < 0.001  # Should sum to 1 (softmax output)

    def test_consistency(self, classifier):
        """Test that same input produces same output."""
        text = "This is a consistent test sentence."
        
        result1 = classifier.predict(text)
        result2 = classifier.predict(text)
        
        assert result1["label"] == result2["label"]
        assert abs(result1["confidence"] - result2["confidence"]) < 1e-6
        assert abs(result1["positive_score"] - result2["positive_score"]) < 1e-6

    def test_empty_string(self, classifier):
        """Test handling of empty string."""
        result = classifier.predict("")
        
        # Should not crash and return valid structure
        assert "label" in result
        assert result["label"] in ["POSITIVE", "NEGATIVE"]

    def test_long_text_truncation(self, classifier):
        """Test that long text gets truncated properly."""
        # Create text much longer than 512 tokens (max sequence length)
        long_text = "This is fantastic! " * 200
        
        result = classifier.predict(long_text)
        
        # Should still work (gets truncated to max_length)
        assert result["label"] == "POSITIVE"
        assert result["confidence"] > 0.0

    def test_special_characters(self, classifier):
        """Test handling of special characters and emojis."""
        result = classifier.predict("This is amazing!!! 😊 👍 #great @awesome")
        
        # Should handle gracefully
        assert result["label"] in ["POSITIVE", "NEGATIVE"]
        assert 0.0 <= result["confidence"] <= 1.0

    def test_mixed_sentiment(self, classifier):
        """Test on text with mixed sentiment."""
        result = classifier.predict("The movie had great acting but terrible plot.")
        
        # Should return one of the labels (model's best guess)
        assert result["label"] in ["POSITIVE", "NEGATIVE"]
        # Confidence might be lower for mixed sentiment
        assert 0.0 <= result["confidence"] <= 1.0


class TestModelConfiguration:
    """Tests for model configuration and attributes."""

    def test_config_attributes(self, classifier):
        """Test that config has expected DistilBERT attributes."""
        assert classifier.config.hidden_size == 768
        assert classifier.config.num_hidden_layers == 6
        assert classifier.config.num_attention_heads == 12
        assert classifier.config.max_position_embeddings == 512

    def test_label_mapping(self, classifier):
        """Test that label mapping is correct."""
        assert classifier.id2label[0] == "NEGATIVE"
        assert classifier.id2label[1] == "POSITIVE"

    def test_device_setup(self, classifier):
        """Test that device is configured correctly."""
        assert classifier.device is not None
        assert classifier.device_ref is not None


class TestTokenization:
    """Tests for tokenization behaviour."""

    def test_tokenizer_loaded(self, classifier):
        """Test that tokenizer is loaded."""
        assert classifier.tokenizer is not None

    def test_tokenization_works(self, classifier):
        """Test that tokenizer can process text."""
        # This tests the tokenizer is functional
        tokens = classifier.tokenizer("Test text", return_tensors="np")
        
        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        assert tokens["input_ids"].shape[1] <= 512  # Max length
