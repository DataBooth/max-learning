"""
CNN MNIST Example
==================

Demonstrates a CNN for image classification using MAX Graph.

Architecture: Conv(1→32)→Pool→Conv(32→64)→Pool→Flatten→FC(128)→FC(10)

Demonstrates:
- 2D convolutions for spatial feature extraction
- Max pooling for downsampling
- Flatten operation (Conv → FC transition)
- Image classification on MNIST digits
- Optional GPU acceleration

Run:
  pixi run example-cnn
  python examples/python/04_cnn_mnist/cnn_mnist.py
  python examples/python/04_cnn_mnist/cnn_mnist.py --device gpu
"""

import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.python.max_cnn import CNNClassificationModel


def load_mnist_samples():
    """Load cached MNIST test samples.
    
    Returns:
        images: Test images [10, 1, 28, 28]
        labels: True labels [10]
    """
    # Load cached samples
    data_path = Path(__file__).parent / "data" / "mnist_samples.npz"
    data = np.load(data_path)
    
    images = data['images']
    labels = data['labels']
    
    return images, labels


def visualize_digit(image: np.ndarray):
    """Create simple ASCII visualization of MNIST digit.
    
    Args:
        image: Image array [1, 28, 28]
    """
    # Remove channel dimension and normalize to 0-9 range for ASCII
    img = image[0]  # [28, 28]
    
    # Denormalize (images are normalized with mean=0.1307, std=0.3081)
    img = img * 0.3081 + 0.1307
    
    # Scale to 0-9 for ASCII chars
    img = np.clip(img * 10, 0, 9).astype(int)
    
    # ASCII chars from dark to light
    chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    
    # Simple 2x downsampling for display
    display = []
    for i in range(0, 28, 2):
        row = ""
        for j in range(0, 28, 2):
            val = int(img[i:i+2, j:j+2].mean())
            row += chars[min(val * 6, len(chars)-1)]
        display.append(row)
    
    return "\\n  ".join(display)


def main():
    parser = argparse.ArgumentParser(description="CNN MNIST classification example")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to run on (cpu or gpu)"
    )
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent / "cnn_config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    
    print("=== CNN MNIST Classification Example ===\\n")
    
    # Extract config
    input_channels = config["model"]["input_channels"]
    image_height = config["model"]["image_height"]
    image_width = config["model"]["image_width"]
    num_classes = config["model"]["num_classes"]
    
    print(f"Architecture: Conv(1→32)→Pool→Conv(32→64)→Pool→Flatten→FC(128)→FC({num_classes})\\n")
    
    # Load pre-trained weights
    print("1. Loading pre-trained weights...")
    weights_path = Path(__file__).parent / "weights" / "cnn_weights.npz"
    weights_data = np.load(weights_path)
    weights = {
        'conv1_W': weights_data['conv1_W'],
        'conv1_b': weights_data['conv1_b'],
        'conv2_W': weights_data['conv2_W'],
        'conv2_b': weights_data['conv2_b'],
        'fc1_W': weights_data['fc1_W'],
        'fc1_b': weights_data['fc1_b'],
        'fc2_W': weights_data['fc2_W'],
        'fc2_b': weights_data['fc2_b'],
    }
    print(f"   ✓ Weights loaded (trained to 99.10% test accuracy)\\n")
    
    # Load dataset
    print("2. Loading cached MNIST test samples...")
    images, labels = load_mnist_samples()
    print(f"   ✓ Loaded {len(images)} samples ({image_height}×{image_width} grayscale images)\\n")
    
    # Build and compile model
    print(f"3. Building MAX Graph on {args.device.upper()}...")
    try:
        model = CNNClassificationModel(
            input_channels=input_channels,
            image_height=image_height,
            image_width=image_width,
            num_classes=num_classes,
            weights=weights,
            device=args.device,
        )
        print(f"   ✓ Graph compiled and loaded\\n")
    except Exception as e:
        if args.device == "gpu":
            print(f"   ✗ GPU compilation failed: {e}")
            print(f"   Note: Convolution ops may not be available on Apple Silicon GPU yet\\n")
            return
        else:
            raise
    
    # Make predictions
    print("4. Running inference on test images...\\n")
    predictions, probabilities = model.predict(images)
    
    # Display results
    print("=" * 80)
    print(f"{'Image':<8} {'True':<6} {'Predicted':<12} {'Confidence':<12} {'Status':<8}")
    print("=" * 80)
    
    correct = 0
    for i in range(len(images)):
        true_label = labels[i]
        pred_label = predictions[i]
        confidence = probabilities[i, pred_label]
        is_correct = pred_label == true_label
        status = "✓" if is_correct else "✗"
        correct += is_correct
        
        print(f"#{i+1:<7} {true_label:<6} {pred_label:<12} {confidence:.1%}{'':<7} {status}")
    
    accuracy = 100. * correct / len(images)
    print("=" * 80)
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{len(images)})\\n")
    
    # Show a sample digit visualization
    print("Sample digit visualization (Image #1):")
    print(f"  {visualize_digit(images[0])}")
    print(f"  → Predicted: {predictions[0]}, True: {labels[0]}\\n")
    
    print(f"✓ CNN classification example completed on {args.device.upper()}!")
    print("\\nOperations demonstrated:")
    print("  - ops.conv2d (2D convolutions for feature extraction)")
    print("  - ops.max_pool2d (spatial downsampling)")
    print("  - ops.flatten (transition from spatial to fully connected)")
    print("  - ops.matmul, ops.add, ops.relu (fully connected layers)")
    print(f"\\nModel implementation: src/python/max_cnn/")


if __name__ == "__main__":
    main()
