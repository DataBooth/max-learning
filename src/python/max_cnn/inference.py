"""High-level inference API for CNN image classification."""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef

from .model import build_cnn_graph


class CNNClassificationModel:
    """CNN classification model for inference."""

    def __init__(
        self,
        input_channels: int,
        image_height: int,
        image_width: int,
        num_classes: int,
        weights: dict[str, np.ndarray],
        device: str = "cpu",
    ):
        """Initialise the CNN classification model.

        Args:
            input_channels: Number of input channels (1 for grayscale)
            image_height: Image height (e.g., 28 for MNIST)
            image_width: Image width (e.g., 28 for MNIST)
            num_classes: Number of output classes (e.g., 10 for digits)
            weights: Dictionary containing model weights
            device: Device to run on ("cpu" or "gpu")
        """
        self.input_channels = input_channels
        self.image_height = image_height
        self.image_width = image_width
        self.num_classes = num_classes

        # Set device
        if device == "cpu":
            self.device = CPU()
        elif device == "gpu":
            from max.driver import Accelerator

            self.device = Accelerator()
        else:
            raise ValueError(f"Unknown device: {device}")

        self.device_ref = DeviceRef.from_device(self.device)

        # Build and compile graph
        graph = build_cnn_graph(
            input_channels=input_channels,
            image_height=image_height,
            image_width=image_width,
            num_classes=num_classes,
            weights=weights,
            dtype=DType.float32,
            device=self.device_ref,
        )

        # Create inference session and load model
        session = InferenceSession(devices=[self.device])
        self.model = session.load(graph)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Make predictions on input images.

        Args:
            X: Input images, shape [batch_size, channels, height, width] or [channels, height, width]

        Returns:
            predictions: Predicted class labels, shape [batch_size]
            probabilities: Class probabilities, shape [batch_size, num_classes]
        """
        # Handle single image
        if X.ndim == 3:
            X = X.reshape(1, *X.shape)

        # Process each image individually (since graph has batch_size=1)
        predictions_list = []
        probabilities_list = []

        for i in range(len(X)):
            sample = X[i : i + 1]  # Keep 4D shape [1, channels, height, width]

            # Convert to MAX Tensor
            X_tensor = Buffer.from_numpy(sample.astype(np.float32)).to(self.device)

            # Run inference
            output = self.model.execute(X_tensor)[0]
            logits = output.to_numpy()

            # Apply softmax to get probabilities
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            # Get prediction
            pred_class = int(np.argmax(probs, axis=-1)[0])

            predictions_list.append(pred_class)
            probabilities_list.append(probs[0])

        predictions = np.array(predictions_list)
        probabilities = np.vstack(probabilities_list)

        return predictions, probabilities
