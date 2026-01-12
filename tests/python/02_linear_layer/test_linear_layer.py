"""Tests for linear layer example."""

import sys

import numpy as np

# Import utils from installed package
from utils.paths import get_examples_dir

# Import the example module
example_path = get_examples_dir() / "02_linear_layer"
sys.path.insert(0, str(example_path))

# Import directly to test the functions
from linear_layer import build_linear_layer_graph
from max.driver import CPU, Tensor
from max.engine import InferenceSession


class TestLinearLayer:
    """Tests for the linear layer example."""

    # Test constants matching the config
    W = np.array([[1.0, 0.5, -0.5, 2.0], [-1.0, 0.5, 1.5, -2.0]], dtype=np.float32)
    b = np.array([0.1, -0.1], dtype=np.float32)

    def test_graph_builds(self):
        """Test that the graph builds without errors."""
        graph = build_linear_layer_graph("cpu", 1, 4, 2, self.W, self.b)
        assert graph is not None
        assert graph.name == "linear_layer_cpu"

    def test_graph_compiles(self):
        """Test that the graph compiles successfully."""
        graph = build_linear_layer_graph("cpu", 1, 4, 2, self.W, self.b)
        device = CPU()
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        assert model is not None

    def test_inference_runs(self):
        """Test that inference executes successfully."""
        graph = build_linear_layer_graph("cpu", 1, 4, 2, self.W, self.b)
        device = CPU()
        session = InferenceSession(devices=[device])
        model = session.load(graph)

        # Create test input
        input_data_np = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        input_data = Tensor.from_numpy(input_data_np).to(device)

        # Run inference
        output = model.execute(input_data)[0]
        output_np = output.to_numpy()

        assert output_np is not None
        assert output_np.shape == (1, 2)

    def test_output_correctness(self):
        """Test that output matches expected values."""
        graph = build_linear_layer_graph("cpu", 1, 4, 2, self.W, self.b)
        device = CPU()
        session = InferenceSession(devices=[device])
        model = session.load(graph)

        # Create test input
        input_data_np = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        input_data = Tensor.from_numpy(input_data_np).to(device)

        # Run inference
        output = model.execute(input_data)[0]
        output_np = output.to_numpy()

        # Verify with NumPy calculation
        expected = np.maximum(0, input_data_np @ self.W.T + self.b)  # ReLU(x @ W^T + b)

        assert np.allclose(output_np, expected, rtol=1e-5, atol=1e-5)

    def test_different_inputs(self):
        """Test with different input values."""
        graph = build_linear_layer_graph("cpu", 1, 4, 2, self.W, self.b)
        device = CPU()
        session = InferenceSession(devices=[device])
        model = session.load(graph)

        # Test various inputs
        test_inputs = [
            np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),  # All zeros
            np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32),  # All ones
            np.array([[-1.0, -2.0, -3.0, -4.0]], dtype=np.float32),  # Negative
        ]

        for input_data_np in test_inputs:
            input_data = Tensor.from_numpy(input_data_np).to(device)
            output = model.execute(input_data)[0]
            output_np = output.to_numpy()
            expected = np.maximum(0, input_data_np @ self.W.T + self.b)
            assert np.allclose(output_np, expected, rtol=1e-5, atol=1e-5)

    def test_relu_activation(self):
        """Test that ReLU activation works correctly."""
        graph = build_linear_layer_graph("cpu", 1, 4, 2, self.W, self.b)
        device = CPU()
        session = InferenceSession(devices=[device])
        model = session.load(graph)

        # Input that should produce negative pre-activation values
        input_data_np = np.array([[-5.0, -5.0, -5.0, -5.0]], dtype=np.float32)
        input_data = Tensor.from_numpy(input_data_np).to(device)

        output = model.execute(input_data)[0]
        output_np = output.to_numpy()

        # All outputs should be >= 0 due to ReLU
        assert np.all(output_np >= 0)
