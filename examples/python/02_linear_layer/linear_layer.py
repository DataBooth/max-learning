"""
Linear Layer Example
====================

Demonstrates a simple linear layer with MAX Graph.

Computes: y = relu(W @ x + b)

Demonstrates:
- Matrix multiplication (linear transformation)
- Bias addition
- ReLU activation
- Weight loading from config

Run:
  pixi run example-linear
  python examples/python/02_linear_layer/linear_layer.py --device cpu
"""

import argparse
from pathlib import Path

import numpy as np
import tomllib
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def build_linear_layer_graph(
    device_type: str,
    batch_size: int,
    input_features: int,
    output_features: int,
    weights_W: np.ndarray,
    bias_b: np.ndarray,
) -> Graph:
    """Build graph: y = relu(W @ x + b)

    Args:
        device_type: "cpu" or "gpu"
        batch_size: Batch size
        input_features: Number of input features
        output_features: Number of output features
        weights_W: Weight matrix [output_features, input_features]
        bias_b: Bias vector [output_features]
    """
    device = DeviceRef(device_type)
    input_spec = TensorType(DType.float32, shape=[batch_size, input_features], device=device)

    with Graph(f"linear_layer_{device_type}", input_types=[input_spec]) as graph:
        x = graph.inputs[0].tensor

        # Load weights from config
        W = ops.constant(weights_W.astype(np.float32), dtype=DType.float32, device=x.device)

        b = ops.constant(bias_b.astype(np.float32), dtype=DType.float32, device=x.device)

        # Computation: y = relu(x @ W^T + b)
        # Note: MAX uses matmul(x, transpose(W)) pattern for linear layers
        y = ops.matmul(x, ops.transpose(W, 0, 1))  # [batch, in] @ [in, out] → [batch, out]
        y = ops.add(y, b)  # [batch, out] + [out] → [batch, out]
        y = ops.relu(y)  # [batch, out] → [batch, out]

        graph.output(y)

    return graph


def main():
    parser = argparse.ArgumentParser(description="Linear layer example")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default=None,
        help="Device to run on (cpu or gpu). Overrides config default.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="linear_layer_config.toml",
        help="Path to config file (default: linear_layer_config.toml)",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(__file__).parent / args.config
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # Get device (CLI arg overrides config)
    device_type = args.device if args.device else config["device"]["default"]

    # Extract config values
    batch_size = config["graph"]["batch_size"]
    input_features = config["graph"]["input_features"]
    output_features = config["graph"]["output_features"]
    weights_W = np.array(config["weights"]["W"])
    bias_b = np.array(config["weights"]["b"])
    input_values = np.array(config["test_data"]["input_values"])

    print(f"=== Linear Layer Example ({device_type.upper()}) ===\n")
    print(f"Configuration: {input_features} features → {output_features} outputs\n")

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
            print("   Note: GPU requires matmul kernel (not yet available for Apple Silicon)\n")
        return

    # 2. Build graph
    print("2. Building computation graph...")
    try:
        graph = build_linear_layer_graph(
            device_type, batch_size, input_features, output_features, weights_W, bias_b
        )
        print(f"   Graph '{graph.name}' created\n")
    except Exception as e:
        print(f"   ✗ Graph building failed: {e}\n")
        return

    # 3. Compile
    print("3. Compiling graph...")
    try:
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        print("   ✓ Graph compiled and loaded\n")
    except Exception as e:
        print(f"   ✗ Compilation failed: {e}\n")
        if device_type == "gpu" and "matmul" in str(e).lower():
            print("   Note: matmul operation not available on Apple Silicon GPU yet\n")
        return

    # 4. Prepare input
    print("4. Preparing input...")
    input_data_np = input_values.astype(np.float32)
    input_data = Buffer.from_numpy(input_data_np).to(device)
    print(f"   Input shape: {input_data_np.shape}")
    print(f"   Input data: {input_data_np}\n")

    # 5. Execute
    print(f"5. Executing inference on {device_type.upper()}...")
    try:
        output = model.execute(input_data)[0]
        output_np = output.to_numpy()
        print(f"   Output shape: {output_np.shape}")
        print(f"   Output data: {output_np}\n")
    except Exception as e:
        print(f"   ✗ Execution failed: {e}\n")
        return

    # 6. Verify
    print("6. Verifying with NumPy...")
    expected = np.maximum(0, input_data_np @ weights_W.T + bias_b)
    print(f"   Expected: {expected}")
    match = np.allclose(output_np, expected)
    print(f"   Match: {match}\n")

    if match:
        print(f"✓ Success! Linear layer worked on {device_type.upper()}!")
        print("\nOperations tested:")
        print("  - ops.matmul (matrix multiplication)")
        print("  - ops.transpose (weight matrix transposition)")
        print("  - ops.add (bias addition)")
        print("  - ops.relu (activation)")
    else:
        print("✗ Results don't match - possible issue")


if __name__ == "__main__":
    main()
