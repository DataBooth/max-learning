"""Test individual conv2d operation to isolate the bug."""

import numpy as np
import torch
import torch.nn as nn
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

# Simple test: 1x1x3x3 input, 1->1 conv, 3x3 kernel
print("Testing conv2d operation...")
print("=" * 60)

# Create simple input: 1 batch, 1 channel, 3x3 image
input_nchw = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)

print(f"Input (NCHW): shape={input_nchw.shape}")
print(input_nchw[0, 0])

# Create 3x3 kernel (identity-ish)
kernel_oihw = np.array([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]], dtype=np.float32)

bias = np.array([0.0], dtype=np.float32)

print(f"\nKernel (OIHW): shape={kernel_oihw.shape}")
print(kernel_oihw[0, 0])

# PyTorch version
print("\n--- PyTorch ---")
conv_torch = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
conv_torch.weight.data = torch.from_numpy(kernel_oihw)
conv_torch.bias.data = torch.from_numpy(bias)

with torch.no_grad():
    output_torch = conv_torch(torch.from_numpy(input_nchw))
    print(f"Output shape: {output_torch.shape}")
    print(f"Output:\n{output_torch[0, 0].numpy()}")

# MAX Graph version
print("\n--- MAX Graph ---")

# Convert kernel to RSCF format: OIHW [1,1,3,3] -> RSCF [3,3,1,1]
kernel_rscf = np.transpose(kernel_oihw, (2, 3, 1, 0))
print(f"Kernel (RSCF): shape={kernel_rscf.shape}")

# Convert input to NHWC: NCHW [1,1,3,3] -> NHWC [1,3,3,1]
input_nhwc = np.transpose(input_nchw, (0, 2, 3, 1))
print(f"Input (NHWC): shape={input_nhwc.shape}")
print(input_nhwc[0, :, :, 0])

device = CPU()
device_ref = DeviceRef.from_device(device)

# Build graph
input_type = TensorType(DType.float32, shape=[1, 3, 3, 1], device=device_ref)

with Graph("test_conv", input_types=[input_type]) as graph:
    x = graph.inputs[0].tensor

    kernel_const = ops.constant(kernel_rscf, dtype=DType.float32, device=device_ref)
    bias_const = ops.constant(bias, dtype=DType.float32, device=device_ref)

    y = ops.conv2d(x, kernel_const, bias=bias_const, stride=(1, 1), padding=(1, 1, 1, 1))
    graph.output(y)

session = InferenceSession(devices=[device])
model = session.load(graph)

# Run
input_tensor = Tensor.from_numpy(input_nhwc).to(device)
output_max = model.execute(input_tensor)[0].to_numpy()

print(f"Output shape: {output_max.shape} (NHWC)")

# Convert back to NCHW for comparison
output_max_nchw = np.transpose(output_max, (0, 3, 1, 2))
print(f"Output (converted to NCHW): shape={output_max_nchw.shape}")
print(f"Output:\n{output_max_nchw[0, 0]}")

# Compare
print("\n--- Comparison ---")
diff = np.abs(output_torch[0, 0].numpy() - output_max_nchw[0, 0])
print(f"Max difference: {diff.max():.6f}")
print(
    f"Match: {'YES ✓' if np.allclose(output_torch[0, 0].numpy(), output_max_nchw[0, 0], atol=1e-5) else 'NO ✗'}"
)
