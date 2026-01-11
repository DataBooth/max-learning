"""Tests for MLP regression example."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path dynamically
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.python.utils.paths import get_project_root, get_examples_dir, add_project_root_to_path
add_project_root_to_path()

from src.python.max_mlp import MLPRegressionModel


class TestMLPRegression:
    """Tests for the MLP regression example."""

    @pytest.fixture
    def model_weights(self):
        """Create simple test weights for a small MLP."""
        np.random.seed(42)
        return {
            'W1': np.random.randn(32, 8) * 0.01,
            'b1': np.zeros(32),
            'W2': np.random.randn(16, 32) * 0.01,
            'b2': np.zeros(16),
            'W3': np.random.randn(1, 16) * 0.01,
            'b3': np.zeros(1),
        }

    @pytest.fixture
    def model(self, model_weights):
        """Create a test MLP model."""
        return MLPRegressionModel(
            input_size=8,
            hidden_size1=32,
            hidden_size2=16,
            output_size=1,
            weights=model_weights,
            device="cpu",
        )

    def test_model_initialization(self, model_weights):
        """Test that model initializes without errors."""
        model = MLPRegressionModel(
            input_size=8,
            hidden_size1=32,
            hidden_size2=16,
            output_size=1,
            weights=model_weights,
            device="cpu",
        )
        assert model is not None
        assert model.input_size == 8
        assert model.hidden_size1 == 32
        assert model.hidden_size2 == 16
        assert model.output_size == 1

    def test_single_prediction(self, model):
        """Test prediction on a single sample."""
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        predictions = model.predict(X)
        
        assert predictions is not None
        assert predictions.shape == (1, 1)
        assert not np.isnan(predictions).any()

    def test_batch_prediction(self, model):
        """Test prediction on multiple samples."""
        X = np.random.randn(5, 8).astype(np.float32)
        predictions = model.predict(X)
        
        assert predictions is not None
        assert predictions.shape == (5, 1)
        assert not np.isnan(predictions).any()

    def test_output_range(self, model):
        """Test that outputs are reasonable (not exploding)."""
        X = np.random.randn(10, 8).astype(np.float32)
        predictions = model.predict(X)
        
        # With small random weights, outputs should be small
        assert np.abs(predictions).max() < 10.0

    def test_deterministic_output(self, model):
        """Test that same input produces same output."""
        X = np.random.randn(3, 8).astype(np.float32)
        
        pred1 = model.predict(X)
        pred2 = model.predict(X)
        
        assert np.allclose(pred1, pred2, rtol=1e-5, atol=1e-5)

    def test_different_inputs_produce_output(self, model):
        """Test that model produces valid output for different inputs."""
        X1 = np.ones((1, 8), dtype=np.float32)
        X2 = np.zeros((1, 8), dtype=np.float32)
        
        pred1 = model.predict(X1)
        pred2 = model.predict(X2)
        
        # Both should produce valid outputs
        assert pred1.shape == (1, 1)
        assert pred2.shape == (1, 1)
        assert not np.isnan(pred1).any()
        assert not np.isnan(pred2).any()

    def test_with_loaded_weights(self):
        """Test with actual pre-trained weights from the example."""
        weights_path = get_examples_dir() / "03_mlp_regression" / "weights" / "mlp_weights.npz"
        
        if not weights_path.exists():
            pytest.skip("Pre-trained weights not found")
        
        weights_data = np.load(weights_path)
        weights = {
            'W1': weights_data['W1'],
            'b1': weights_data['b1'],
            'W2': weights_data['W2'],
            'b2': weights_data['b2'],
            'W3': weights_data['W3'],
            'b3': weights_data['b3'],
        }
        
        model = MLPRegressionModel(
            input_size=8,
            hidden_size1=128,
            hidden_size2=64,
            output_size=1,
            weights=weights,
            device="cpu",
        )
        
        # Test with a sample input (scaled)
        scaler_mean = weights_data['scaler_mean']
        scaler_scale = weights_data['scaler_scale']
        
        X_raw = np.array([[3.88, 41.0, 6.98, 1.02, 322.0, 2.56, 37.88, -122.23]], dtype=np.float32)
        X_scaled = (X_raw - scaler_mean) / scaler_scale
        
        predictions = model.predict(X_scaled)
        
        assert predictions.shape == (1, 1)
        # Prediction should be in reasonable range for California housing (0-5 in units of $100k)
        assert 0 < predictions[0, 0] < 10

    def test_zero_input(self, model):
        """Test with zero input."""
        X = np.zeros((1, 8), dtype=np.float32)
        predictions = model.predict(X)
        
        assert predictions is not None
        assert predictions.shape == (1, 1)
        assert not np.isnan(predictions).any()

    def test_large_batch(self, model):
        """Test with larger batch to ensure no memory issues."""
        X = np.random.randn(100, 8).astype(np.float32)
        predictions = model.predict(X)
        
        assert predictions.shape == (100, 1)
        assert not np.isnan(predictions).any()
