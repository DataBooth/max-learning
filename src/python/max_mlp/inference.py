"""High-level inference API for MLP regression."""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef

from .model import build_mlp_graph


class MLPRegressionModel:
    """MLP regression model for inference."""

    def __init__(
        self,
        input_size: int,
        hidden_size1: int,
        hidden_size2: int,
        output_size: int,
        weights: dict[str, np.ndarray],
        device: str = "cpu",
    ):
        """Initialise the MLP regression model.

        Args:
            input_size: Number of input features
            hidden_size1: Size of first hidden layer
            hidden_size2: Size of second hidden layer
            output_size: Number of outputs (1 for regression)
            weights: Dictionary containing model weights
            device: Device to run on ("cpu" or "gpu")
        """
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        # Set device
        if device == "cpu":
            self.device = CPU()
        else:
            raise NotImplementedError("GPU support coming soon")

        self.device_ref = DeviceRef.from_device(self.device)

        # Build and compile graph
        graph = build_mlp_graph(
            input_size=input_size,
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            output_size=output_size,
            weights=weights,
            dtype=DType.float32,
            device=self.device_ref,
        )

        # Create inference session and load model
        session = InferenceSession(devices=[self.device])
        self.model = session.load(graph)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data.

        Args:
            X: Input features, shape [batch_size, input_size] or [input_size]

        Returns:
            predictions: Output predictions, shape [batch_size, output_size]
        """
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Process each sample individually (since graph has batch_size=1)
        predictions_list = []
        for i in range(len(X)):
            sample = X[i : i + 1]  # Keep 2D shape [1, input_size]

            # Convert to MAX Tensor
            X_tensor = Tensor.from_numpy(sample.astype(np.float32)).to(self.device)

            # Run inference
            output = self.model.execute(X_tensor)[0]
            pred = output.to_numpy()
            predictions_list.append(pred)

        # Combine all predictions
        predictions = np.vstack(predictions_list)

        return predictions
