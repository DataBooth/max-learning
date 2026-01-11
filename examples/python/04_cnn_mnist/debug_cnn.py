"""
Debug script to compare PyTorch vs MAX Graph CNN outputs.

This script runs the same input through both implementations and compares:
1. Input data format
2. Weight shapes and values
3. Intermediate activations
4. Final predictions
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.python.max_cnn import CNNClassificationModel


class PyTorchCNN(nn.Module):
    """PyTorch CNN matching our architecture."""
    
    def __init__(self):
        super(PyTorchCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Conv block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def main():
    print("=" * 80)
    print("CNN DEBUG: PyTorch vs MAX Graph Comparison")
    print("=" * 80)
    
    # Load weights
    print("\n1. Loading weights...")
    weights_path = Path(__file__).parent / "weights" / "cnn_weights.npz"
    weights_data = np.load(weights_path)
    
    print(f"   Conv1 weight shape (RSCF): {weights_data['conv1_W'].shape}")
    print(f"   Conv2 weight shape (RSCF): {weights_data['conv2_W'].shape}")
    print(f"   FC1 weight shape: {weights_data['fc1_W'].shape}")
    print(f"   FC2 weight shape: {weights_data['fc2_W'].shape}")
    
    # Load test sample
    print("\n2. Loading test sample...")
    data_path = Path(__file__).parent / "data" / "mnist_samples.npz"
    data = np.load(data_path)
    test_image = data['images'][0:1]  # First image, keep batch dim
    test_label = data['labels'][0]
    
    print(f"   Test image shape: {test_image.shape} (NCHW)")
    print(f"   Test label: {test_label}")
    print(f"   Image range: [{test_image.min():.3f}, {test_image.max():.3f}]")
    
    # Create PyTorch model and load weights
    print("\n3. Creating PyTorch model...")
    pytorch_model = PyTorchCNN()
    pytorch_model.eval()
    
    # Load weights back to PyTorch format
    # MAX weights are RSCF [H, W, I, O], convert to PyTorch [O, I, H, W]
    conv1_w_max = weights_data['conv1_W']  # [3, 3, 1, 32]
    conv1_w_torch = np.transpose(conv1_w_max, (3, 2, 0, 1))  # [32, 1, 3, 3]
    
    conv2_w_max = weights_data['conv2_W']  # [3, 3, 32, 64]
    conv2_w_torch = np.transpose(conv2_w_max, (3, 2, 0, 1))  # [64, 32, 3, 3]
    
    pytorch_model.conv1.weight.data = torch.from_numpy(conv1_w_torch)
    pytorch_model.conv1.bias.data = torch.from_numpy(weights_data['conv1_b'])
    pytorch_model.conv2.weight.data = torch.from_numpy(conv2_w_torch)
    pytorch_model.conv2.bias.data = torch.from_numpy(weights_data['conv2_b'])
    pytorch_model.fc1.weight.data = torch.from_numpy(weights_data['fc1_W'])
    pytorch_model.fc1.bias.data = torch.from_numpy(weights_data['fc1_b'])
    pytorch_model.fc2.weight.data = torch.from_numpy(weights_data['fc2_W'])
    pytorch_model.fc2.bias.data = torch.from_numpy(weights_data['fc2_b'])
    
    print("   ✓ PyTorch weights loaded")
    
    # Run PyTorch inference
    print("\n4. Running PyTorch inference...")
    with torch.no_grad():
        test_tensor = torch.from_numpy(test_image)
        pytorch_output = pytorch_model(test_tensor)
        pytorch_probs = F.softmax(pytorch_output, dim=1)
        pytorch_pred = pytorch_output.argmax(dim=1).item()
        pytorch_conf = pytorch_probs[0, pytorch_pred].item()
    
    print(f"   PyTorch prediction: {pytorch_pred} (confidence: {pytorch_conf:.1%})")
    print(f"   PyTorch logits: {pytorch_output[0].numpy()}")
    
    # Create MAX Graph model
    print("\n5. Creating MAX Graph model...")
    weights_dict = {
        'conv1_W': weights_data['conv1_W'],
        'conv1_b': weights_data['conv1_b'],
        'conv2_W': weights_data['conv2_W'],
        'conv2_b': weights_data['conv2_b'],
        'fc1_W': weights_data['fc1_W'],
        'fc1_b': weights_data['fc1_b'],
        'fc2_W': weights_data['fc2_W'],
        'fc2_b': weights_data['fc2_b'],
    }
    
    max_model = CNNClassificationModel(
        input_channels=1,
        image_height=28,
        image_width=28,
        num_classes=10,
        weights=weights_dict,
        device="cpu",
    )
    print("   ✓ MAX Graph model loaded")
    
    # Run MAX Graph inference
    print("\n6. Running MAX Graph inference...")
    max_preds, max_probs = max_model.predict(test_image)
    max_pred = max_preds[0]
    max_conf = max_probs[0, max_pred]
    
    print(f"   MAX prediction: {max_pred} (confidence: {max_conf:.1%})")
    print(f"   MAX probabilities: {max_probs[0]}")
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"True label:        {test_label}")
    print(f"PyTorch prediction: {pytorch_pred} {'✓' if pytorch_pred == test_label else '✗'}")
    print(f"MAX prediction:     {max_pred} {'✓' if max_pred == test_label else '✗'}")
    print(f"Match: {'YES ✓' if pytorch_pred == max_pred else 'NO ✗ - BUG FOUND!'}")
    
    if pytorch_pred != max_pred:
        print("\n🔍 DIVERGENCE DETECTED!")
        print("   Predictions differ between PyTorch and MAX Graph.")
        print("   This indicates a weight loading or layout issue.")
        
        # Show probability distribution comparison
        print("\n   Class probabilities:")
        print("   " + "-" * 60)
        print(f"   {'Class':<8} {'PyTorch':<15} {'MAX':<15} {'Diff':<15}")
        print("   " + "-" * 60)
        for i in range(10):
            pt_prob = pytorch_probs[0, i].item()
            mx_prob = max_probs[0, i]
            diff = abs(pt_prob - mx_prob)
            marker = "  ⚠️" if diff > 0.1 else ""
            print(f"   {i:<8} {pt_prob:>6.1%}{'':<8} {mx_prob:>6.1%}{'':<8} {diff:>6.1%}{marker}")


if __name__ == "__main__":
    main()
