"""
Minimal Element-wise Operations Example
========================================

Demonstrates MAX Graph basics with element-wise operations.
No helper functions - just the core flow:
1. Build a graph
2. Compile it
3. Run inference

Operation: y = relu(x * 2.0 + 1.0)

Run:
  python examples/python/01_elementwise/elementwise_minimal.py
"""

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

# ============================================================================
# Configuration
# ============================================================================
INPUT_SIZE = 4
MULTIPLIER = 2.0
OFFSET = 1.0

# ============================================================================
# 1. BUILD GRAPH
# ============================================================================
print("Building MAX Graph...")

# Define device (CPU in this case)
device = DeviceRef("cpu")

# Define input specification: float32 tensor of size 4
input_spec = TensorType(DType.float32, shape=[INPUT_SIZE], device=device)

# Build the computation graph
with Graph("elementwise_minimal", input_types=[input_spec]) as graph:
    # Get input tensor from graph
    x = graph.inputs[0].tensor

    # Create constants for multiply and add
    multiplier_const = ops.constant(
        np.full(INPUT_SIZE, MULTIPLIER, dtype=np.float32), dtype=DType.float32, device=device
    )

    offset_const = ops.constant(
        np.full(INPUT_SIZE, OFFSET, dtype=np.float32), dtype=DType.float32, device=device
    )

    # Build computation: y = relu(x * multiplier + offset)
    y = ops.mul(x, multiplier_const)  # x * 2.0
    y = ops.add(y, offset_const)  # + 1.0
    y = ops.relu(y)  # relu(...)

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

# Create input data: [1.0, -2.0, 3.0, -4.0]
input_data_np = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)

# Convert to MAX Tensor
input_tensor = Buffer.from_numpy(input_data_np).to(device_obj)

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
print(f"Operation: y = relu(x * {MULTIPLIER} + {OFFSET})")
print()
print(f"Input:  {input_data_np}")
print(f"Output: {output_np}")
print()

# Verify with NumPy
expected = np.maximum(0, input_data_np * MULTIPLIER + OFFSET)
print(f"Expected (NumPy): {expected}")
print(f"Match: {np.allclose(output_np, expected)}")
print("=" * 60)
