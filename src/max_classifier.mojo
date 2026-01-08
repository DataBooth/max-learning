"""
MAX Engine sentiment classifier using pre-trained DistilBERT model.

This module provides high-performance sentiment inference using Modular's MAX Engine
with a transformer model (DistilBERT) converted to ONNX format.
"""

from max import engine
from pathlib import Path
from tensor import Tensor, TensorShape
from algorithm import vectorize
from python import Python
from logger import Logger, Level


struct SentimentLabel:
    """Sentiment classification labels."""
    var NEGATIVE: String
    var POSITIVE: String
    
    fn __init__(inout self):
        self.NEGATIVE = "NEGATIVE"
        self.POSITIVE = "POSITIVE"


struct MAXSentimentResult:
    """Result from MAX Engine sentiment classification."""
    var label: String
    var confidence: Float64
    var score: Float64  # Raw score before softmax
    var positive_prob: Float64
    var negative_prob: Float64
    
    fn __init__(
        inout self,
        label: String,
        confidence: Float64,
        score: Float64,
        positive_prob: Float64,
        negative_prob: Float64
    ):
        self.label = label
        self.confidence = confidence
        self.score = score
        self.positive_prob = positive_prob
        self.negative_prob = negative_prob
    
    fn copy(self) -> Self:
        """Create a copy of this result."""
        return MAXSentimentResult(
            self.label,
            self.confidence,
            self.score,
            self.positive_prob,
            self.negative_prob
        )


struct MAXSentimentClassifier:
    """
    Sentiment classifier using MAX Engine with DistilBERT.
    
    This classifier loads a pre-trained DistilBERT model in ONNX format
    and provides high-performance inference for sentiment analysis.
    
    Example:
        var classifier = MAXSentimentClassifier("models/distilbert-sentiment/model.onnx")
        classifier.load()
        var result = classifier.predict(input_ids, attention_mask)
        print(result.label, result.confidence)
    """
    var model_path: String
    var session: engine.InferenceSession
    var model: engine.Model
    var is_loaded: Bool
    var log: Logger[Level.INFO]
    
    fn __init__(inout self, model_path: String = "models/distilbert-sentiment/model.onnx") raises:
        """
        Initialize the MAX Engine classifier.
        
        Args:
            model_path: Path to the ONNX model file.
        """
        self.model_path = model_path
        self.is_loaded = False
        self.log = Logger[Level.INFO]()
        
        # Initialize session (will be properly set in load())
        # Note: We need to handle this carefully as InferenceSession may need devices
        self.session = engine.InferenceSession()
        
        # Model will be loaded later
        # We can't initialize it here because Model doesn't have a default constructor
        # This is a limitation we'll work around in load()
    
    fn load(inout self) raises:
        """
        Load the ONNX model into MAX Engine.
        
        This compiles the model and prepares it for inference.
        """
        if self.is_loaded:
            self.log.info("Model already loaded")
            return
        
        self.log.info("Loading MAX Engine model from", self.model_path)
        
        # Create path object
        var path = Path(self.model_path)
        
        # Load model with MAX Engine
        # The model will be compiled and optimized for the target hardware
        self.model = self.session.load(path)
        
        self.is_loaded = True
        self.log.info("✅ MAX Engine model loaded successfully")
        
        # Print model metadata for debugging
        self.log.info("Model input metadata:")
        var inputs = self.model.input_metadata
        for i in range(len(inputs)):
            var inp = inputs[i]
            self.log.info("  Input", i, ":", inp.name, "shape:", str(inp.shape))
        
        self.log.info("Model output metadata:")
        var outputs = self.model.output_metadata
        for i in range(len(outputs)):
            var out = outputs[i]
            self.log.info("  Output", i, ":", out.name, "shape:", str(out.shape))
    
    fn predict(
        inout self,
        input_ids: Tensor,
        attention_mask: Tensor
    ) raises -> MAXSentimentResult:
        """
        Run sentiment inference on tokenized input.
        
        Args:
            input_ids: Tensor of token IDs (shape: [batch_size, seq_len])
            attention_mask: Tensor of attention mask (shape: [batch_size, seq_len])
        
        Returns:
            MAXSentimentResult with label, confidence, and probabilities.
        
        Note:
            Model expects DistilBERT format inputs with max_length=512
        """
        if not self.is_loaded:
            raise Error("Model not loaded. Call load() first.")
        
        self.log.info("Running MAX Engine inference...")
        
        # Execute model
        # DistilBERT outputs logits of shape [batch_size, num_classes]
        # For sentiment: num_classes = 2 (NEGATIVE, POSITIVE)
        var outputs = self.model.execute(input_ids, attention_mask)
        
        # Get logits (first output)
        var logits = outputs[0]
        
        self.log.info("Model output shape:", str(logits.shape))
        
        # Apply softmax to get probabilities
        # logits shape: [1, 2] for single input
        var probs = self._softmax(logits)
        
        # Extract probabilities for each class
        # Index 0: NEGATIVE, Index 1: POSITIVE
        var neg_prob = self._get_tensor_value(probs, 0)
        var pos_prob = self._get_tensor_value(probs, 1)
        
        # Determine label and confidence
        var label: String
        var confidence: Float64
        var raw_score: Float64
        
        if pos_prob > neg_prob:
            label = SentimentLabel().POSITIVE
            confidence = pos_prob
            raw_score = self._get_tensor_value(logits, 1)
        else:
            label = SentimentLabel().NEGATIVE
            confidence = neg_prob
            raw_score = self._get_tensor_value(logits, 0)
        
        self.log.info("Prediction:", label, "confidence:", confidence)
        
        return MAXSentimentResult(
            label,
            confidence,
            raw_score,
            pos_prob,
            neg_prob
        )
    
    fn _softmax(self, logits: Tensor) raises -> Tensor:
        """
        Apply softmax to logits to get probabilities.
        
        Args:
            logits: Raw model outputs (shape: [batch_size, num_classes])
        
        Returns:
            Probabilities (shape: [batch_size, num_classes])
        
        Note:
            This is a simple implementation for 2-class classification.
            For production, use optimized MAX ops.
        """
        # For now, we'll use Python numpy for softmax
        # TODO: Replace with pure Mojo implementation or MAX ops
        var np = Python.import_module("numpy")
        
        # Convert tensor to numpy
        var logits_np = logits.to_numpy()
        
        # Apply softmax: exp(x) / sum(exp(x))
        var exp_logits = np.exp(logits_np - np.max(logits_np))
        var probs_np = exp_logits / np.sum(exp_logits)
        
        # Convert back to tensor
        # Note: This is a temporary solution until we have pure Mojo softmax
        return Tensor.from_numpy(probs_np)
    
    fn _get_tensor_value(self, tensor: Tensor, index: Int) raises -> Float64:
        """
        Extract a single float value from tensor at given index.
        
        Args:
            tensor: Input tensor
            index: Index to extract
        
        Returns:
            Float value at index
        """
        # Convert to numpy and extract value
        var np = Python.import_module("numpy")
        var np_array = tensor.to_numpy()
        var value = np_array.flat[index]
        return Float64(value)
