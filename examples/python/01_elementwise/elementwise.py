"""
Element-wise Operations Example
================================

Demonstrates the simplest MAX Graph operations:
- Element-wise multiplication
- Element-wise addition
- ReLU activation

Supports both CPU and GPU execution.

Run:
  pixi run python examples/python/01_elementwise/elementwise.py --device cpu
  pixi run python examples/python/01_elementwise/elementwise.py --device gpu
"""

import argparse
import numpy as np
from max.driver import Accelerator, CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def build_elementwise_graph(device_type: str) -> Graph:
    """Build graph: y = relu(x * 2 + 1)
    
    Args:
        device_type: "cpu" or "gpu"
    """
    device = DeviceRef(device_type)
    input_spec = TensorType(DType.float32, shape=[4], device=device)
    
    with Graph(f"elementwise_{device_type}", input_types=[input_spec]) as graph:
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
        y = ops.mul(x, multiplier)
        y = ops.add(y, offset)
        y = ops.relu(y)
        
        graph.output(y)
    
    return graph


def main():
    parser = argparse.ArgumentParser(description="Element-wise operations example")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to run on (cpu or gpu)"
    )
    args = parser.parse_args()
    
    print(f"=== Element-wise Operations Example ({args.device.upper()}) ===\n")
    
    # 1. Get device
    print(f"1. Initializing {args.device.upper()} device...")
    try:
        if args.device == "gpu":
            device = Accelerator()
            print(f"   ✓ GPU device: {device}\n")
        else:
            device = CPU()
            print(f"   ✓ CPU device: {device}\n")
    except Exception as e:
        print(f"   ✗ Device initialization failed: {e}\n")
        if args.device == "gpu":
            print("   Tip: Ensure Metal Toolchain is installed:")
            print("   xcodebuild -downloadComponent MetalToolchain\n")
        return
    
    # 2. Build graph
    print("2. Building computation graph...")
    try:
        graph = build_elementwise_graph(args.device)
        print(f"   Graph '{graph.name}' created\n")
    except Exception as e:
        print(f"   ✗ Graph building failed: {e}\n")
        return
    
    # 3. Compile
    print("3. Compiling graph...")
    try:
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        print(f"   ✓ Graph compiled and loaded\n")
    except Exception as e:
        print(f"   ✗ Compilation failed: {e}\n")
        return
    
    # 4. Prepare input
    print("4. Preparing input...")
    input_data_np = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    input_data = Tensor.from_numpy(input_data_np).to(device)
    print(f"   Input: {input_data_np}\n")
    
    # 5. Execute
    print(f"5. Executing inference on {args.device.upper()}...")
    try:
        output = model.execute(input_data)[0]
        output_np = output.to_numpy()
        print(f"   Output: {output_np}\n")
    except Exception as e:
        print(f"   ✗ Execution failed: {e}\n")
        return
    
    # 6. Verify
    print("6. Verifying with NumPy...")
    expected = np.maximum(0, input_data_np * 2.0 + 1.0)
    print(f"   Expected: {expected}")
    match = np.allclose(output_np, expected)
    print(f"   Match: {match}\n")
    
    if match:
        print(f"✓ Success! Element-wise operations worked on {args.device.upper()}!")
        print(f"\nOperations tested:")
        print(f"  - ops.mul (element-wise multiplication)")
        print(f"  - ops.add (element-wise addition)")
        print(f"  - ops.relu (element-wise activation)")
    else:
        print(f"✗ Results don't match - possible issue")


if __name__ == "__main__":
    main()
