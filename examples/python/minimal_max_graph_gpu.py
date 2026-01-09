"""
Minimal MAX Graph Example - GPU Version
========================================

Same as minimal_max_graph.py but attempts to run on Apple Silicon GPU.

This tests whether the basic operations (matmul, transpose, add, relu)
have GPU kernels available on Apple Silicon.

Run: pixi run python examples/python/minimal_max_graph_gpu.py
"""

import numpy as np
from max.driver import Accelerator, CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def build_minimal_graph_gpu() -> Graph:
    """Build a minimal graph: y = relu(W @ x + b) for GPU"""
    
    # Define input types - use GPU device
    device = DeviceRef("gpu")
    input_spec = TensorType(DType.float32, shape=[1, 4], device=device)
    
    # Create graph
    with Graph("minimal_model_gpu", input_types=[input_spec]) as graph:
        # Get input tensor
        x = graph.inputs[0].tensor
        
        # Define weights inline
        # Weight matrix: [2, 4] - maps 4 features to 2 outputs
        W = ops.constant(
            np.array([[1.0, 0.5, -0.5, 2.0], [-1.0, 0.5, 1.5, -2.0]], dtype=np.float32),
            dtype=DType.float32,
            device=x.device
        )
        
        # Bias: [2]
        b = ops.constant(
            np.array([0.1, -0.1], dtype=np.float32),
            dtype=DType.float32,
            device=x.device
        )
        
        # Computation: y = relu(x @ W^T + b)
        y = ops.matmul(x, ops.transpose(W, 0, 1))  # [1, 4] @ [4, 2] → [1, 2]
        y = ops.add(y, b)  # [1, 2] + [2] → [1, 2]
        y = ops.relu(y)  # [1, 2] → [1, 2]
        
        # Output result
        graph.output(y)
    
    return graph


def main():
    print("=== Minimal MAX Graph Example - GPU ===\n")
    
    # Check if GPU/Accelerator is available
    try:
        gpu_device = Accelerator()
        print(f"✓ Accelerator device found: {gpu_device}\n")
    except Exception as e:
        print(f"✗ Accelerator device not available: {e}\n")
        print("Falling back to CPU...\n")
        gpu_device = CPU()
    
    # 1. Build graph
    print("1. Building computation graph for GPU...")
    try:
        graph = build_minimal_graph_gpu()
        print(f"   Graph '{graph.name}' created\n")
    except Exception as e:
        print(f"   ✗ Graph building failed: {e}\n")
        return
    
    # 2. Create inference session and compile
    print("2. Creating inference session and compiling graph...")
    try:
        session = InferenceSession(devices=[gpu_device])
        model = session.load(graph)
        print("   ✓ Graph compiled and loaded on GPU\n")
    except Exception as e:
        print(f"   ✗ Compilation failed: {e}\n")
        print("   This might mean some operations don't have GPU kernels.\n")
        return
    
    # 3. Prepare input
    print("3. Preparing input...")
    input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    print(f"   Input shape: {input_data.shape}")
    print(f"   Input data: {input_data}\n")
    
    # 4. Execute
    print("4. Executing inference on GPU...")
    try:
        output = model.execute(input_data)[0]
        output_np = output.to_numpy()
        print(f"   Output shape: {output_np.shape}")
        print(f"   Output data: {output_np}\n")
    except Exception as e:
        print(f"   ✗ Execution failed: {e}\n")
        return
    
    # 5. Verify with NumPy
    print("5. Verifying with NumPy...")
    W = np.array([[1.0, 0.5, -0.5, 2.0], [-1.0, 0.5, 1.5, -2.0]], dtype=np.float32)
    b = np.array([0.1, -0.1], dtype=np.float32)
    expected = np.maximum(0, input_data @ W.T + b)
    print(f"   Expected: {expected}")
    match = np.allclose(output_np, expected)
    print(f"   Match: {match}\n")
    
    if match:
        print("✓ Success! All operations worked on GPU!")
    else:
        print("✗ Results don't match - possible GPU kernel issue")


if __name__ == "__main__":
    main()
