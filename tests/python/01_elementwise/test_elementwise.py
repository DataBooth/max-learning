"""Tests for elementwise operations example."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Import the example module
example_path = Path(__file__).parent.parent.parent.parent / "examples" / "python" / "01_elementwise"
sys.path.insert(0, str(example_path))

from elementwise import build_elementwise_graph

from max.driver import CPU, Tensor
from max.engine import InferenceSession


class TestElementwiseOperations:
    """Tests for element-wise operations example."""

    # Test constants
    multiplier = 2.0
    offset = 1.0
    size = 4

    def test_graph_builds_cpu(self):
        """Test that CPU graph builds without errors."""
        graph = build_elementwise_graph("cpu", self.multiplier, self.offset, self.size)
        assert graph is not None
        assert graph.name == "elementwise_cpu"

    def test_graph_compiles_cpu(self):
        """Test that CPU graph compiles successfully."""
        graph = build_elementwise_graph("cpu", self.multiplier, self.offset, self.size)
        device = CPU()
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        assert model is not None

    def test_inference_runs_cpu(self):
        """Test that CPU inference executes successfully."""
        graph = build_elementwise_graph("cpu", self.multiplier, self.offset, self.size)
        device = CPU()
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        
        # Create test input
        input_data_np = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
        input_data = Tensor.from_numpy(input_data_np).to(device)
        
        # Run inference
        output = model.execute(input_data)[0]
        output_np = output.to_numpy()
        
        assert output_np is not None
        assert output_np.shape == (4,)

    def test_output_correctness_cpu(self):
        """Test that CPU output matches expected values."""
        graph = build_elementwise_graph("cpu", self.multiplier, self.offset, self.size)
        device = CPU()
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        
        # Create test input
        input_data_np = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
        input_data = Tensor.from_numpy(input_data_np).to(device)
        
        # Run inference
        output = model.execute(input_data)[0]
        output_np = output.to_numpy()
        
        # Verify with NumPy calculation: relu(x * multiplier + offset)
        expected = np.maximum(0, input_data_np * self.multiplier + self.offset)
        
        assert np.allclose(output_np, expected, rtol=1e-5, atol=1e-5)

    def test_relu_activation(self):
        """Test that ReLU activation works correctly (negative values become 0)."""
        graph = build_elementwise_graph("cpu", self.multiplier, self.offset, self.size)
        device = CPU()
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        
        # Input that should produce negative pre-activation values
        input_data_np = np.array([-5.0, -5.0, -5.0, -5.0], dtype=np.float32)
        input_data = Tensor.from_numpy(input_data_np).to(device)
        
        output = model.execute(input_data)[0]
        output_np = output.to_numpy()
        
        # All outputs should be >= 0 due to ReLU
        assert np.all(output_np >= 0)

    def test_different_sizes(self):
        """Test with different tensor sizes."""
        test_sizes = [1, 4, 16, 64]
        device = CPU()
        
        for size in test_sizes:
            graph = build_elementwise_graph("cpu", self.multiplier, self.offset, size)
            session = InferenceSession(devices=[device])
            model = session.load(graph)
            
            input_data_np = np.random.randn(size).astype(np.float32)
            input_data = Tensor.from_numpy(input_data_np).to(device)
            
            output = model.execute(input_data)[0]
            output_np = output.to_numpy()
            
            expected = np.maximum(0, input_data_np * self.multiplier + self.offset)
            assert np.allclose(output_np, expected, rtol=1e-5, atol=1e-5)

    def test_different_parameters(self):
        """Test with different multiplier and offset values."""
        device = CPU()
        test_params = [
            (1.0, 0.0),   # Identity-like
            (0.5, 2.0),   # Different scale
            (3.0, -1.0),  # Negative offset
        ]
        
        for mult, off in test_params:
            graph = build_elementwise_graph("cpu", mult, off, self.size)
            session = InferenceSession(devices=[device])
            model = session.load(graph)
            
            input_data_np = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
            input_data = Tensor.from_numpy(input_data_np).to(device)
            
            output = model.execute(input_data)[0]
            output_np = output.to_numpy()
            
            expected = np.maximum(0, input_data_np * mult + off)
            assert np.allclose(output_np, expected, rtol=1e-5, atol=1e-5)

    def test_all_positive_inputs(self):
        """Test with all positive inputs (ReLU should not affect)."""
        graph = build_elementwise_graph("cpu", self.multiplier, self.offset, self.size)
        device = CPU()
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        
        input_data_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        input_data = Tensor.from_numpy(input_data_np).to(device)
        
        output = model.execute(input_data)[0]
        output_np = output.to_numpy()
        
        # With positive inputs, ReLU shouldn't change the values
        expected = input_data_np * self.multiplier + self.offset
        assert np.allclose(output_np, expected, rtol=1e-5, atol=1e-5)

    def test_zero_input(self):
        """Test with zero input."""
        graph = build_elementwise_graph("cpu", self.multiplier, self.offset, self.size)
        device = CPU()
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        
        input_data_np = np.zeros(self.size, dtype=np.float32)
        input_data = Tensor.from_numpy(input_data_np).to(device)
        
        output = model.execute(input_data)[0]
        output_np = output.to_numpy()
        
        # 0 * mult + offset, then relu
        expected = np.maximum(0, self.offset)
        assert np.allclose(output_np, np.full(self.size, expected), rtol=1e-5, atol=1e-5)
