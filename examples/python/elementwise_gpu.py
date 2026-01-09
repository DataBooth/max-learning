"""
Ultra-Minimal GPU Example - Element-wise Operations Only
=========================================================

The simplest possible GPU test using only element-wise operations:
- Addition
- Multiplication  
- ReLU activation

No matrix multiplication, so more likely to work on Apple Silicon GPU.

Run: pixi run python examples/python/elementwise_gpu.py
"""

import numpy as np
from max.driver import Accelerator, CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def build_elementwise_graph() -> Graph:
    """Build ultra-minimal graph: y = relu(x * 2 + 1)"""
    
    # Define input types - use GPU device
    device = DeviceRef("gpu")
    input_spec = TensorType(DType.float32, shape=[4], device=device)
    
    # Create graph
    with Graph("elementwise_gpu", input_types=[input_spec]) as graph:
        # Get input tensor
        x = graph.inputs[0].tensor
        
        # Define constants
        multiplier = ops.constant(
            np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
            dtype=DType.float32,
            device=x.device
        )
        
        offset = ops.constant(
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=DType.float32,
            device=x.device
        )
        
        # Element-wise computation: y = relu(x * 2 + 1)
        y = ops.mul(x, multiplier)  # Element-wise multiply
        y = ops.add(y, offset)      # Element-wise add
        y = ops.relu(y)             # Element-wise ReLU
        
        # Output result
        graph.output(y)
    
    return graph


def main():
    print("=== Ultra-Minimal GPU Example (Element-wise Only) ===\n")
    
    # Check if Accelerator is available
    try:
        gpu_device = Accelerator()
        print(f"✓ Accelerator device found: {gpu_device}\n")
        device_name = "GPU"
    except Exception as e:
        print(f"✗ Accelerator not available: {e}\n")
        print("Falling back to CPU...\n")
        gpu_device = CPU()
        device_name = "CPU"
    
    # 1. Build graph
    print(f"1. Building computation graph for {device_name}...")
    try:
        graph = build_elementwise_graph()
        print(f"   Graph '{graph.name}' created\n")
    except Exception as e:
        print(f"   ✗ Graph building failed: {e}\n")
        return
    
    # 2. Create inference session and compile
    print("2. Compiling graph...")
    try:
        session = InferenceSession(devices=[gpu_device])
        model = session.load(graph)
        print(f"   ✓ Graph compiled and loaded on {device_name}\n")
    except Exception as e:
        print(f"   ✗ Compilation failed: {e}\n")
        print("   Some operations might not have GPU kernels available.\n")
        return
    
    # 3. Prepare input
    print("3. Preparing input...")
    input_data = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    print(f"   Input: {input_data}\n")
    
    # 4. Execute
    print(f"4. Executing inference on {device_name}...")
    try:
        output = model.execute(input_data)[0]
        output_np = output.to_numpy()
        print(f"   Output: {output_np}\n")
    except Exception as e:
        print(f"   ✗ Execution failed: {e}\n")
        return
    
    # 5. Verify with NumPy
    print("5. Verifying with NumPy...")
    expected = np.maximum(0, input_data * 2.0 + 1.0)  # relu(x * 2 + 1)
    print(f"   Expected: {expected}")
    match = np.allclose(output_np, expected)
    print(f"   Match: {match}\n")
    
    if match:
        print(f"✓ Success! Element-wise operations worked on {device_name}!")
        print(f"\nOperations tested:")
        print(f"  - Element-wise multiplication (ops.mul)")
        print(f"  - Element-wise addition (ops.add)")
        print(f"  - Element-wise ReLU (ops.relu)")
    else:
        print("✗ Results don't match - possible issue")


if __name__ == "__main__":
    main()
