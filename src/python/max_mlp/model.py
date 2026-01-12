"""MLP (Multi-Layer Perceptron) model for regression using MAX Graph.

Demonstrates:
- Layer stacking (3 linear layers)
- ReLU activations
- Regression output (continuous prediction)
"""

from __future__ import annotations

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn import Module


class MLPRegressor(Module):
    """3-layer MLP for regression tasks."""

    def __init__(
        self,
        input_size: int,
        hidden_size1: int,
        hidden_size2: int,
        output_size: int,
        weights: dict[str, np.ndarray],
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        """Initialise MLP layers.

        Args:
            input_size: Number of input features
            hidden_size1: Size of first hidden layer
            hidden_size2: Size of second hidden layer
            output_size: Number of outputs (1 for regression)
            weights: Dictionary with keys 'W1', 'b1', 'W2', 'b2', 'W3', 'b3'
            dtype: Data type for computations
            device: Device reference
        """
        super().__init__()

        # Layer 1: input_size → hidden_size1
        self.W1 = ops.constant(weights["W1"].astype(np.float32), dtype=dtype, device=device)
        self.b1 = ops.constant(weights["b1"].astype(np.float32), dtype=dtype, device=device)

        # Layer 2: hidden_size1 → hidden_size2
        self.W2 = ops.constant(weights["W2"].astype(np.float32), dtype=dtype, device=device)
        self.b2 = ops.constant(weights["b2"].astype(np.float32), dtype=dtype, device=device)

        # Layer 3: hidden_size2 → output_size
        self.W3 = ops.constant(weights["W3"].astype(np.float32), dtype=dtype, device=device)
        self.b3 = ops.constant(weights["b3"].astype(np.float32), dtype=dtype, device=device)

    def __call__(self, x: TensorValue) -> TensorValue:
        """Forward pass through MLP.

        Args:
            x: Input tensor, shape [batch_size, input_size]

        Returns:
            predictions: Output tensor, shape [batch_size, output_size]
        """
        # Layer 1: x @ W1^T + b1 → ReLU
        h1 = ops.matmul(x, ops.transpose(self.W1, 0, 1))
        h1 = ops.add(h1, self.b1)
        h1 = ops.relu(h1)

        # Layer 2: h1 @ W2^T + b2 → ReLU
        h2 = ops.matmul(h1, ops.transpose(self.W2, 0, 1))
        h2 = ops.add(h2, self.b2)
        h2 = ops.relu(h2)

        # Layer 3: h2 @ W3^T + b3 (no activation for regression)
        output = ops.matmul(h2, ops.transpose(self.W3, 0, 1))
        output = ops.add(output, self.b3)

        return output


def build_mlp_graph(
    input_size: int,
    hidden_size1: int,
    hidden_size2: int,
    output_size: int,
    weights: dict[str, np.ndarray],
    dtype: DType,
    device: DeviceRef,
    batch_size: int = 1,
) -> Graph:
    """Build the MLP regression graph.

    Args:
        input_size: Number of input features
        hidden_size1: Size of first hidden layer
        hidden_size2: Size of second hidden layer
        output_size: Number of outputs
        weights: Dictionary containing W1, b1, W2, b2, W3, b3
        dtype: Data type for computations
        device: Device to run on
        batch_size: Batch size for inference

    Returns:
        Compiled MAX Graph
    """
    # Define input tensor type
    input_type = TensorType(dtype, shape=[batch_size, input_size], device=device)

    # Build graph
    with Graph("mlp_regressor", input_types=[input_type]) as graph:
        x = graph.inputs[0].tensor

        # Instantiate model
        mlp = MLPRegressor(
            input_size=input_size,
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            output_size=output_size,
            weights=weights,
            dtype=dtype,
            device=device,
        )

        # Forward pass
        predictions = mlp(x)
        graph.output(predictions)

    return graph
