"""Tests for minimal MAX graph example."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Import the example module
example_path = Path(__file__).parent.parent.parent / "examples" / "python"
sys.path.insert(0, str(example_path))

# Import directly to test the functions
from minimal_max_graph import build_minimal_graph

from max.driver import CPU
from max.engine import InferenceSession


class TestMinimalMaxGraph:
    """Tests for the minimal MAX graph example."""

    def test_graph_builds(self):
        """Test that the graph builds without errors."""
        graph = build_minimal_graph()
        assert graph is not None
        assert graph.name == "minimal_model"

    def test_graph_compiles(self):
        """Test that the graph compiles successfully."""
        graph = build_minimal_graph()
        device = CPU()
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        assert model is not None

    def test_inference_runs(self):
        """Test that inference executes successfully."""
        graph = build_minimal_graph()
        device = CPU()
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        
        # Create test input
        input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        
        # Run inference
        output = model.execute(input_data)[0]
        output_np = output.to_numpy()
        
        assert output_np is not None
        assert output_np.shape == (1, 2)

    def test_output_correctness(self):
        """Test that output matches expected values."""
        graph = build_minimal_graph()
        device = CPU()
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        
        # Create test input
        input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        
        # Run inference
        output = model.execute(input_data)[0]
        output_np = output.to_numpy()
        
        # Verify with NumPy calculation
        # W shape: [2, 4], input shape: [1, 4]
        W = np.array([[1.0, 0.5, -0.5, 2.0], [-1.0, 0.5, 1.5, -2.0]], dtype=np.float32)
        b = np.array([0.1, -0.1], dtype=np.float32)
        expected = np.maximum(0, input_data @ W.T + b)  # ReLU(x @ W^T + b)
        
        assert np.allclose(output_np, expected, rtol=1e-5, atol=1e-5)

    def test_different_inputs(self):
        """Test with different input values."""
        graph = build_minimal_graph()
        device = CPU()
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        
        # Test various inputs
        test_inputs = [
            np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),  # All zeros
            np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32),  # All ones
            np.array([[-1.0, -2.0, -3.0, -4.0]], dtype=np.float32),  # Negative
        ]
        
        W = np.array([[1.0, 0.5, -0.5, 2.0], [-1.0, 0.5, 1.5, -2.0]], dtype=np.float32)
        b = np.array([0.1, -0.1], dtype=np.float32)
        
        for input_data in test_inputs:
            output = model.execute(input_data)[0]
            output_np = output.to_numpy()
            expected = np.maximum(0, input_data @ W.T + b)
            assert np.allclose(output_np, expected, rtol=1e-5, atol=1e-5)

    def test_relu_activation(self):
        """Test that ReLU activation works correctly."""
        graph = build_minimal_graph()
        device = CPU()
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        
        # Input that should produce negative pre-activation values
        input_data = np.array([[-5.0, -5.0, -5.0, -5.0]], dtype=np.float32)
        
        output = model.execute(input_data)[0]
        output_np = output.to_numpy()
        
        # All outputs should be >= 0 due to ReLU
        assert np.all(output_np >= 0)
