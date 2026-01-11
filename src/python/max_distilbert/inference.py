"""Inference script for DistilBERT sentiment classification using MAX Graph."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.graph.weights import load_weights
from transformers import AutoConfig, AutoTokenizer

from .graph import build_graph


@dataclass
class SimpleModelConfig:
    """Minimal model config."""
    quantization_encoding: None = None


@dataclass
class SimpleConfig:
    """Minimal config that satisfies embedding/encoder layer requirements."""
    max_length: int = 512
    pool_embeddings: bool = False
    model_config: SimpleModelConfig = None
    
    def __post_init__(self):
        if self.model_config is None:
            self.model_config = SimpleModelConfig()


class DistilBertSentimentClassifier:
    """Sentiment classifier using DistilBERT with MAX Graph."""

    def __init__(self, model_path: str | Path):
        """Initialise the classifier.
        
        Args:
            model_path: Path to the HuggingFace model directory
        """
        self.model_path = Path(model_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        
        # Load model configuration
        self.config = AutoConfig.from_pretrained(str(self.model_path))
        
        # DistilBERT doesn't use token type embeddings, but BERT's embedding layer
        # expects this attribute. Add a dummy value for compatibility.
        if not hasattr(self.config, 'type_vocab_size'):
            self.config.type_vocab_size = 2  # Dummy value, won't be used
        
        # Create simple config (no full pipeline infrastructure needed)
        self.pipeline_config = SimpleConfig(max_length=512)
        
        # Set device (CPU for now, Apple GPU support coming)
        self.device = CPU()
        self.device_ref = DeviceRef.from_device(self.device)
        
        # Load weights
        weights_path = self.model_path / "model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights file not found: {weights_path}. "
                "Please ensure the model is downloaded with safetensors format."
            )
        
        self.weights = load_weights([weights_path])
        
        # Build and compile graph
        print("Building MAX Graph...")
        graph = build_graph(
            pipeline_config=self.pipeline_config,
            weights=self.weights,
            huggingface_config=self.config,
            dtype=DType.float32,
            input_device=self.device_ref,
        )
        
        # Create inference session and load model
        print("Compiling graph with MAX Engine...")
        session = InferenceSession(devices=[self.device])
        self.model = session.load(graph, weights_registry=self.weights.allocated_weights)
        
        print("Model loaded successfully!")
        
        # Label mapping
        self.id2label = self.config.id2label

    def predict(self, text: str) -> dict[str, float]:
        """Predict sentiment for the given text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with 'label', 'confidence', 'positive_score', 'negative_score'
        """
        # Tokenise input
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        # Convert to MAX Tensors
        input_ids = Tensor.from_numpy(inputs["input_ids"].astype(np.int64)).to(
            self.device
        )
        attention_mask = Tensor.from_numpy(
            inputs["attention_mask"].astype(np.float32)
        ).to(self.device)
        
        # Run inference
        outputs = self.model.execute(input_ids, attention_mask)
        logits = outputs[0].to_numpy()
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Get prediction
        predicted_class = int(np.argmax(probs, axis=-1)[0])
        confidence = float(probs[0, predicted_class])
        
        label = self.id2label[predicted_class]
        negative_score = float(probs[0, 0])
        positive_score = float(probs[0, 1])
        
        return {
            "label": label,
            "confidence": confidence,
            "positive_score": positive_score,
            "negative_score": negative_score,
        }


def main():
    """Demo the classifier."""
    # Add project root to path
    import sys
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    from src.python.utils.paths import get_models_dir
    
    # Model path
    model_path = get_models_dir() / "distilbert-sentiment"
    
    if not model_path.exists():
        print(f"Error: Model directory not found: {model_path}")
        print("Please run: ./models/download_models.sh")
        sys.exit(1)
    
    # Initialise classifier
    print("Loading DistilBERT sentiment classifier with MAX Graph...\n")
    classifier = DistilBertSentimentClassifier(model_path)
    
    # Test examples
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible experience. Would not recommend.",
        "It was okay, nothing special.",
        "Best product I've ever bought!",
        "Complete waste of money and time.",
    ]
    
    print("\nRunning sentiment analysis...\n")
    for text in test_texts:
        result = classifier.predict(text)
        print(f"Text: {text}")
        print(f"  → {result['label']} (confidence: {result['confidence']:.2%})")
        print(
            f"     Positive: {result['positive_score']:.2%}, "
            f"Negative: {result['negative_score']:.2%}\n"
        )


if __name__ == "__main__":
    main()
