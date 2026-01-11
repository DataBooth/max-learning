"""Tests for CNN MNIST example."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Import from installed packages
from utils.paths import get_examples_dir
from max_cnn import CNNClassificationModel


class TestCNNMNIST:
    """Tests for the CNN MNIST example."""

    @pytest.fixture
    def model_weights(self):
        """Create simple test weights for a small CNN."""
        np.random.seed(42)
        return {
            # Conv1: 1->8 channels, 3x3 kernel (RSCF format)
            'conv1_W': np.random.randn(3, 3, 1, 8).astype(np.float32) * 0.01,
            'conv1_b': np.zeros(8, dtype=np.float32),
            # Conv2: 8->16 channels, 3x3 kernel (RSCF format)
            'conv2_W': np.random.randn(3, 3, 8, 16).astype(np.float32) * 0.01,
            'conv2_b': np.zeros(16, dtype=np.float32),
            # FC1: 16*7*7 -> 32
            'fc1_W': np.random.randn(32, 16*7*7).astype(np.float32) * 0.01,
            'fc1_b': np.zeros(32, dtype=np.float32),
            # FC2: 32 -> 10
            'fc2_W': np.random.randn(10, 32).astype(np.float32) * 0.01,
            'fc2_b': np.zeros(10, dtype=np.float32),
        }

    @pytest.fixture
    def model(self, model_weights):
        """Create a test CNN model."""
        return CNNClassificationModel(
            input_channels=1,
            image_height=28,
            image_width=28,
            num_classes=10,
            weights=model_weights,
            device="cpu",
        )

    def test_model_initialization(self, model_weights):
        """Test that model initializes without errors."""
        model = CNNClassificationModel(
            input_channels=1,
            image_height=28,
            image_width=28,
            num_classes=10,
            weights=model_weights,
            device="cpu",
        )
        assert model is not None
        assert model.input_channels == 1
        assert model.image_height == 28
        assert model.image_width == 28
        assert model.num_classes == 10

    def test_single_prediction(self, model):
        """Test prediction on a single image."""
        # Random 28x28 grayscale image (NCHW format)
        X = np.random.randn(1, 1, 28, 28).astype(np.float32)
        predictions, probabilities = model.predict(X)
        
        assert predictions is not None
        assert probabilities is not None
        assert predictions.shape == (1,)
        assert probabilities.shape == (1, 10)
        assert 0 <= predictions[0] <= 9
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)

    def test_batch_prediction(self, model):
        """Test prediction on multiple images."""
        X = np.random.randn(5, 1, 28, 28).astype(np.float32)
        predictions, probabilities = model.predict(X)
        
        assert predictions.shape == (5,)
        assert probabilities.shape == (5, 10)
        assert np.all((predictions >= 0) & (predictions <= 9))
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)

    def test_probability_distribution(self, model):
        """Test that probabilities sum to 1 and are in valid range."""
        X = np.random.randn(3, 1, 28, 28).astype(np.float32)
        _, probabilities = model.predict(X)
        
        # All probabilities should be between 0 and 1
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        
        # Sum should be 1 for each sample
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)

    def test_deterministic_output(self, model):
        """Test that same input produces same output."""
        X = np.random.randn(2, 1, 28, 28).astype(np.float32)
        
        pred1, prob1 = model.predict(X)
        pred2, prob2 = model.predict(X)
        
        assert np.array_equal(pred1, pred2)
        assert np.allclose(prob1, prob2, rtol=1e-5, atol=1e-5)

    def test_different_inputs_produce_output(self, model):
        """Test that model produces valid output for different inputs."""
        X1 = np.ones((1, 1, 28, 28), dtype=np.float32)
        X2 = np.zeros((1, 1, 28, 28), dtype=np.float32)
        
        pred1, prob1 = model.predict(X1)
        pred2, prob2 = model.predict(X2)
        
        # Both should produce valid outputs
        assert pred1.shape == (1,)
        assert pred2.shape == (1,)
        assert prob1.shape == (1, 10)
        assert prob2.shape == (1, 10)
        assert not np.isnan(prob1).any()
        assert not np.isnan(prob2).any()

    def test_with_loaded_weights(self):
        """Test with actual pre-trained weights from the example."""
        weights_path = get_examples_dir() / "04_cnn_mnist" / "weights" / "cnn_weights.npz"
        data_path = get_examples_dir() / "04_cnn_mnist" / "data" / "mnist_samples.npz"
        
        if not weights_path.exists() or not data_path.exists():
            pytest.skip("Pre-trained weights or data not found")
        
        weights_data = np.load(weights_path)
        weights = {
            'conv1_W': weights_data['conv1_W'],
            'conv1_b': weights_data['conv1_b'],
            'conv2_W': weights_data['conv2_W'],
            'conv2_b': weights_data['conv2_b'],
            'fc1_W': weights_data['fc1_W'],
            'fc1_b': weights_data['fc1_b'],
            'fc2_W': weights_data['fc2_W'],
            'fc2_b': weights_data['fc2_b'],
        }
        
        model = CNNClassificationModel(
            input_channels=1,
            image_height=28,
            image_width=28,
            num_classes=10,
            weights=weights,
            device="cpu",
        )
        
        # Load test samples
        data = np.load(data_path)
        test_images = data['images'][:5]  # First 5 samples
        test_labels = data['labels'][:5]
        
        predictions, probabilities = model.predict(test_images)
        
        assert predictions.shape == (5,)
        assert probabilities.shape == (5, 10)
        
        # With trained model, should get decent accuracy on these samples
        # (At least better than random 10% on MNIST)
        accuracy = (predictions == test_labels).mean()
        assert accuracy > 0.2  # Very conservative threshold

    def test_zero_input(self, model):
        """Test with zero input (blank image)."""
        X = np.zeros((1, 1, 28, 28), dtype=np.float32)
        predictions, probabilities = model.predict(X)
        
        assert predictions is not None
        assert probabilities is not None
        assert predictions.shape == (1,)
        assert probabilities.shape == (1, 10)

    def test_confidence_values(self, model):
        """Test that confidence values are reasonable."""
        X = np.random.randn(5, 1, 28, 28).astype(np.float32)
        predictions, probabilities = model.predict(X)
        
        # Get confidence for each prediction
        confidences = probabilities[np.arange(len(predictions)), predictions]
        
        # With random weights, confidences should be relatively low but valid
        assert np.all(confidences > 0)
        assert np.all(confidences <= 1.0)

    def test_large_batch(self, model):
        """Test with larger batch to ensure no memory issues."""
        X = np.random.randn(50, 1, 28, 28).astype(np.float32)
        predictions, probabilities = model.predict(X)
        
        assert predictions.shape == (50,)
        assert probabilities.shape == (50, 10)
