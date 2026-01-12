"""CNN (Convolutional Neural Network) model for image classification using MAX Graph.

Demonstrates:
- 2D convolutions for spatial feature extraction
- Max pooling for downsampling
- Flatten operation to transition from spatial to fully connected
- Conv → Pool → Conv → Pool → FC pattern
"""

from __future__ import annotations

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn import Module


class CNNClassifier(Module):
    """CNN for image classification (e.g., MNIST digits)."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        weights: dict[str, np.ndarray],
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        """Initialise CNN layers.

        Architecture: Conv(1→32) → Pool → Conv(32→64) → Pool → Flatten → FC(128) → FC(10)

        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            num_classes: Number of output classes
            weights: Dictionary with conv1_W, conv1_b, conv2_W, conv2_b, fc1_W, fc1_b, fc2_W, fc2_b
            dtype: Data type for computations
            device: Device reference
        """
        super().__init__()

        # Conv layer 1: 1 → 32 channels, 3x3 kernel
        self.conv1_W = ops.constant(
            weights["conv1_W"].astype(np.float32), dtype=dtype, device=device
        )
        self.conv1_b = ops.constant(
            weights["conv1_b"].astype(np.float32), dtype=dtype, device=device
        )

        # Conv layer 2: 32 → 64 channels, 3x3 kernel
        self.conv2_W = ops.constant(
            weights["conv2_W"].astype(np.float32), dtype=dtype, device=device
        )
        self.conv2_b = ops.constant(
            weights["conv2_b"].astype(np.float32), dtype=dtype, device=device
        )

        # FC layer 1: flattened → 128
        self.fc1_W = ops.constant(weights["fc1_W"].astype(np.float32), dtype=dtype, device=device)
        self.fc1_b = ops.constant(weights["fc1_b"].astype(np.float32), dtype=dtype, device=device)

        # FC layer 2 (output): 128 → num_classes
        self.fc2_W = ops.constant(weights["fc2_W"].astype(np.float32), dtype=dtype, device=device)
        self.fc2_b = ops.constant(weights["fc2_b"].astype(np.float32), dtype=dtype, device=device)

    def __call__(self, x: TensorValue) -> TensorValue:
        """Forward pass through CNN.

        Args:
            x: Input tensor, shape [batch_size, channels, height, width] (NCHW)

        Returns:
            logits: Output tensor, shape [batch_size, num_classes]
        """
        # Convert from NCHW to NHWC (MAX Graph default)
        # Input: [batch, 1, 28, 28] → [batch, 28, 28, 1]
        # NCHW [0, 1, 2, 3] → NHWC [0, 2, 3, 1]
        x = ops.transpose(x, 1, 2)  # [B, C, H, W] → [B, H, C, W]
        x = ops.transpose(x, 2, 3)  # [B, H, C, W] → [B, H, W, C]

        # Conv block 1: Conv → ReLU → MaxPool
        # Input: [batch, 28, 28, 1]
        x = ops.conv2d(x, self.conv1_W, bias=self.conv1_b, stride=(1, 1), padding=(1, 1, 1, 1))
        # After conv: [batch, 28, 28, 32]
        x = ops.relu(x)
        x = ops.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        # After pool: [batch, 14, 14, 32]

        # Conv block 2: Conv → ReLU → MaxPool
        x = ops.conv2d(x, self.conv2_W, bias=self.conv2_b, stride=(1, 1), padding=(1, 1, 1, 1))
        # After conv: [batch, 14, 14, 64]
        x = ops.relu(x)
        x = ops.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        # After pool: [batch, 7, 7, 64]

        # Transpose NHWC to NCHW before flatten (to match PyTorch flatten order)
        # [batch, 7, 7, 64] → [batch, 64, 7, 7]
        x = ops.transpose(x, 3, 1)  # [B, H, W, C] → [B, C, W, H]
        x = ops.transpose(x, 2, 3)  # [B, C, W, H] → [B, C, H, W]

        # Flatten: [batch, 64, 7, 7] → [batch, 3136] (channel-first order)
        x = ops.flatten(x, start_dim=1)

        # FC block 1: Linear → ReLU
        x = ops.matmul(x, ops.transpose(self.fc1_W, 0, 1))
        x = ops.add(x, self.fc1_b)
        x = ops.relu(x)
        # After FC1: [batch, 128]

        # FC block 2 (output): Linear (no activation, logits for classification)
        x = ops.matmul(x, ops.transpose(self.fc2_W, 0, 1))
        logits = ops.add(x, self.fc2_b)
        # Output: [batch, 10]

        return logits


def build_cnn_graph(
    input_channels: int,
    image_height: int,
    image_width: int,
    num_classes: int,
    weights: dict[str, np.ndarray],
    dtype: DType,
    device: DeviceRef,
    batch_size: int = 1,
) -> Graph:
    """Build the CNN classification graph.

    Args:
        input_channels: Number of input channels (1 for grayscale)
        image_height: Image height (28 for MNIST)
        image_width: Image width (28 for MNIST)
        num_classes: Number of output classes (10 for digits)
        weights: Dictionary containing all layer weights
        dtype: Data type for computations
        device: Device to run on
        batch_size: Batch size for inference

    Returns:
        Compiled MAX Graph
    """
    # Define input tensor type: [batch, channels, height, width]
    input_type = TensorType(
        dtype, shape=[batch_size, input_channels, image_height, image_width], device=device
    )

    # Build graph
    with Graph("cnn_classifier", input_types=[input_type]) as graph:
        x = graph.inputs[0].tensor

        # Instantiate model
        cnn = CNNClassifier(
            input_channels=input_channels,
            num_classes=num_classes,
            weights=weights,
            dtype=dtype,
            device=device,
        )

        # Forward pass
        logits = cnn(x)
        graph.output(logits)

    return graph
