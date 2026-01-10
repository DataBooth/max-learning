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
import tomllib
from pathlib import Path
import numpy as np
from max.driver import Accelerator, CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def build_elementwise_graph(device_type: str, multiplier_val: float, offset_val: float, input_size: int) -> Graph:
    """Build graph: y = relu(x * multiplier + offset)
    
    Args:
        device_type: "cpu" or "gpu"
        multiplier_val: Multiplier constant
        offset_val: Offset constant
        input_size: Input tensor size
    """
    device = DeviceRef(device_type)
    input_spec = TensorType(DType.float32, shape=[input_size], device=device)
    
    with Graph(f"elementwise_{device_type}", input_types=[input_spec]) as graph:
        x = graph.inputs[0].tensor
        
        # Define constants from config
        multiplier = ops.constant(
            np.full(input_size, multiplier_val, dtype=np.float32),
            dtype=DType.float32,
            device=x.device
        )
        
        offset = ops.constant(
            np.full(input_size, offset_val, dtype=np.float32),
            dtype=DType.float32,
            device=x.device
        )
        
        # Element-wise computation: y = relu(x * multiplier + offset)
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
        default=None,
        help="Device to run on (cpu or gpu). Overrides config default."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="elementwise_config.toml",
        help="Path to config file (default: elementwise_config.toml)"
    )
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent / args.config
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    
    # Get device (CLI arg overrides config)
    device_type = args.device if args.device else config["device"]["default"]
    
    # Extract config values
    multiplier = config["graph"]["multiplier"]
    offset = config["graph"]["offset"]
    input_size = config["graph"]["input_size"]
    input_values = config["test_data"]["input_values"]
    
    print(f"=== Element-wise Operations Example ({device_type.upper()}) ===\n")
    print(f"Configuration: multiplier={multiplier}, offset={offset}, size={input_size}\n")
    
    # 1. Get device
    print(f"1. Initializing {device_type.upper()} device...")
    try:
        if device_type == "gpu":
            device = Accelerator()
            print(f"   ✓ GPU device: {device}\n")
        else:
            device = CPU()
            print(f"   ✓ CPU device: {device}\n")
    except Exception as e:
        print(f"   ✗ Device initialization failed: {e}\n")
        if device_type == "gpu":
            print("   Tip: Ensure Metal Toolchain is installed:")
            print("   xcodebuild -downloadComponent MetalToolchain\n")
        return
    
    # 2. Build graph
    print("2. Building computation graph...")
    try:
        graph = build_elementwise_graph(device_type, multiplier, offset, input_size)
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
    input_data_np = np.array(input_values, dtype=np.float32)
    if len(input_data_np) != input_size:
        print(f"   ✗ Input size mismatch: got {len(input_data_np)}, expected {input_size}\n")
        return
    input_data = Tensor.from_numpy(input_data_np).to(device)
    print(f"   Input: {input_data_np}\n")
    
    # 5. Execute
    print(f"5. Executing inference on {device_type.upper()}...")
    try:
        output = model.execute(input_data)[0]
        output_np = output.to_numpy()
        print(f"   Output: {output_np}\n")
    except Exception as e:
        print(f"   ✗ Execution failed: {e}\n")
        return
    
    # 6. Verify
    print("6. Verifying with NumPy...")
    expected = np.maximum(0, input_data_np * multiplier + offset)
    print(f"   Expected: {expected}")
    match = np.allclose(output_np, expected)
    print(f"   Match: {match}\n")
    
    if match:
        print(f"✓ Success! Element-wise operations worked on {device_type.upper()}!")
        print(f"\nOperations tested:")
        print(f"  - ops.mul (element-wise multiplication)")
        print(f"  - ops.add (element-wise addition)")
        print(f"  - ops.relu (element-wise activation)")
    else:
        print(f"✗ Results don't match - possible issue")


if __name__ == "__main__":
    main()
