"""
Minimal Linear Layer Example
==============================

Demonstrates a simple linear layer (fully connected) with MAX Graph.
No helper functions - just the core flow:
1. Build a graph
2. Compile it
3. Run inference

Operation: y = relu(x @ W^T + b)

Run:
  python examples/python/02_linear_layer/linear_layer_minimal.py
"""

import numpy as np
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

# ============================================================================
# Configuration
# ============================================================================
BATCH_SIZE = 1
INPUT_FEATURES = 4
OUTPUT_FEATURES = 2

# Define weights
# W shape: [output_features, input_features] = [2, 4]
W = np.array(
    [
        [1.0, 0.5, -0.5, 2.0],  # First output neuron
        [-1.0, 0.5, 1.5, -2.0],  # Second output neuron
    ],
    dtype=np.float32,
)

# b shape: [output_features] = [2]
b = np.array([0.1, -0.1], dtype=np.float32)

# ============================================================================
# 1. BUILD GRAPH
# ============================================================================
print("Building MAX Graph...")
print(f"Linear layer: {INPUT_FEATURES} → {OUTPUT_FEATURES}")
print()

# Define device
device = DeviceRef("cpu")

# Define input: [batch_size, input_features] = [1, 4]
input_spec = TensorType(DType.float32, shape=[BATCH_SIZE, INPUT_FEATURES], device=device)

# Build the computation graph
with Graph("linear_layer_minimal", input_types=[input_spec]) as graph:
    # Get input tensor
    x = graph.inputs[0].tensor  # [1, 4]

    # Create weight constants
    W_const = ops.constant(W, dtype=DType.float32, device=device)
    b_const = ops.constant(b, dtype=DType.float32, device=device)

    # Linear transformation: x @ W^T
    # We need to transpose W from [2, 4] to [4, 2]
    W_transposed = ops.transpose(W_const, 0, 1)  # [4, 2]

    # Matrix multiplication: [1, 4] @ [4, 2] = [1, 2]
    y = ops.matmul(x, W_transposed)

    # Add bias: [1, 2] + [2] = [1, 2] (broadcasting)
    y = ops.add(y, b_const)

    # Apply ReLU activation
    y = ops.relu(y)

    # Mark output
    graph.output(y)

print("✓ Graph built\n")

# ============================================================================
# 2. COMPILE GRAPH
# ============================================================================
print("Compiling graph...")

# Initialize CPU device
device_obj = CPU()

# Create inference session
session = InferenceSession(devices=[device_obj])

# Load (compile) the graph
model = session.load(graph)

print("✓ Graph compiled\n")

# ============================================================================
# 3. RUN INFERENCE
# ============================================================================
print("Running inference...")

# Create input data: [1, 2, 3, 4]
input_data_np = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

# Convert to MAX Tensor
input_tensor = Tensor.from_numpy(input_data_np).to(device_obj)

# Execute model
output = model.execute(input_tensor)

# Convert output back to numpy
output_np = output[0].to_numpy()

print("✓ Inference complete\n")

# ============================================================================
# 4. DISPLAY RESULTS
# ============================================================================
print("=" * 60)
print("RESULTS")
print("=" * 60)
print("Operation: y = relu(x @ W^T + b)")
print()
print(f"Input shape:  {input_data_np.shape}")
print(f"Weight shape: {W.shape}")
print(f"Bias shape:   {b.shape}")
print()
print(f"Input:  {input_data_np}")
print(f"Output: {output_np}")
print()

# Verify with NumPy
# x @ W^T + b, then ReLU
expected = np.maximum(0, input_data_np @ W.T + b)
print(f"Expected (NumPy): {expected}")
print(f"Match: {np.allclose(output_np, expected)}")
print("=" * 60)
