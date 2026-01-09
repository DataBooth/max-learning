"""
Minimal MAX Graph Example
==========================

The simplest possible example of using the MAX Graph API.

Demonstrates:
- Building a computation graph (simple matrix multiply + bias + activation)
- Loading weights
- Compiling and executing

Run: uv run python examples/minimal_max_example.py
"""

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def build_minimal_graph() -> Graph:
    """Build a minimal graph: y = relu(W @ x + b)"""
    
    # Define input types
    device = DeviceRef("cpu")
    input_spec = TensorType(DType.float32, shape=[1, 4], device=device)  # Batch size 1, 4 features
    
    # Create graph
    with Graph("minimal_model", input_types=[input_spec]) as graph:
        # Get input tensor
        x = graph.inputs[0].tensor
        
        # Define weights inline (normally loaded from file)
        # Weight matrix: [2, 4] - maps 4 features to 2 outputs (transposed storage)
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
        # Note: MAX uses matmul(x, transpose(W)) pattern for linear layers
        y = ops.matmul(x, ops.transpose(W, 0, 1))  # [1, 4] @ [4, 2] → [1, 2]
        y = ops.add(y, b)  # [1, 2] + [2] → [1, 2]
        y = ops.relu(y)  # [1, 2] → [1, 2]
        
        # Output result
        graph.output(y)
    
    return graph


def main():
    print("=== Minimal MAX Graph Example ===\n")
    
    # 1. Build graph
    print("1. Building computation graph...")
    graph = build_minimal_graph()
    print(f"   Graph '{graph.name}' created\n")
    
    # 2. Create inference session and compile
    print("2. Creating inference session and compiling graph...")
    device = CPU()
    session = InferenceSession(devices=[device])
    model = session.load(graph)
    print("   Graph compiled and loaded\n")
    
    # 3. Prepare input
    print("3. Preparing input...")
    input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    print(f"   Input shape: {input_data.shape}")
    print(f"   Input data: {input_data}\n")
    
    # 4. Execute
    print("4. Executing inference...")
    output = model.execute(input_data)[0]  # Returns tuple, get first output
    output_np = output.to_numpy()  # Convert to numpy for comparison
    print(f"   Output shape: {output_np.shape}")
    print(f"   Output data: {output_np}\n")
    
    # 5. Verify with NumPy
    print("5. Verifying with NumPy...")
    W = np.array([[1.0, 0.5, -0.5, 2.0], [-1.0, 0.5, 1.5, -2.0]], dtype=np.float32)
    b = np.array([0.1, -0.1], dtype=np.float32)
    expected = np.maximum(0, input_data @ W.T + b)  # relu(x @ W^T + b)
    print(f"   Expected: {expected}")
    print(f"   Match: {np.allclose(output_np, expected)}\n")
    
    print("✓ Complete!")


if __name__ == "__main__":
    main()
