"""
Train CNN on MNIST dataset and save weights + data.

This is a helper script to pre-train the CNN model using PyTorch.
The trained weights and dataset are cached for reproducibility.

Purpose:
- Download and cache MNIST dataset
- Train CNN with PyTorch
- Save weights and data to the repo
- Enable fully reproducible MAX Graph inference demo

Run once to generate weights:
    python examples/python/04_cnn_mnist/train_cnn.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from torchvision import datasets, transforms


class PyTorchCNN(nn.Module):
    """CNN architecture matching our MAX Graph implementation."""
    
    def __init__(self):
        super(PyTorchCNN, self).__init__()
        # Conv layer 1: 1 → 32 channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Conv layer 2: 32 → 64 channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # FC layer 1: 64*7*7 → 128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # FC layer 2: 128 → 10
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
        
        # FC block 1
        x = self.fc1(x)
        x = F.relu(x)
        
        # FC block 2 (output)
        x = self.fc2(x)
        return x


def train_and_save_weights():
    """Train CNN with PyTorch and cache data + weights for reproducibility."""
    
    print("=" * 80)
    print("Training CNN on MNIST Dataset")
    print("=" * 80)
    
    # Load MNIST dataset
    print("\n1. Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    train_dataset = datasets.MNIST(
        root='./mnist_data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./mnist_data', train=False, transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print(f"   ✓ Loaded {len(train_dataset)} train samples, {len(test_dataset)} test samples")
    
    # Cache dataset
    print("\n2. Caching MNIST dataset...")
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Get first 10 test samples for demo
    test_images = []
    test_labels = []
    for img, label in test_dataset:
        test_images.append(img.numpy())
        test_labels.append(label)
        if len(test_images) >= 10:
            break
    
    test_images = np.array(test_images, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.int64)
    
    np.savez(
        data_dir / "mnist_samples.npz",
        images=test_images,
        labels=test_labels,
    )
    print(f"   ✓ Cached 10 test samples to: {data_dir / 'mnist_samples.npz'}")
    
    # Train CNN
    print("\n3. Training CNN (Conv→Pool→Conv→Pool→FC→FC)...")
    print("   Architecture: Conv(1→32)→Pool→Conv(32→64)→Pool→Flatten→FC(128)→FC(10)")
    
    device = torch.device("cpu")  # Use CPU for consistency
    model = PyTorchCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train for 2 epochs (enough for good accuracy)
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 200 == 0:
                print(f"   Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    print(f"   ✓ Training completed")
    
    # Evaluate
    print("\n4. Evaluating model...")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_dataset)
    
    print(f"   ✓ Test Loss: {test_loss:.4f}")
    print(f"   ✓ Test Accuracy: {accuracy:.2f}% ({correct}/{len(test_dataset)})")
    
    # Extract weights (convert to MAX Graph format)
    print("\n5. Extracting weights...")
    # Conv weights: PyTorch is [O, I, H, W], MAX expects RSCF [H, W, I, O]
    conv1_w_torch = model.conv1.weight.detach().cpu().numpy()  # [32, 1, 3, 3]
    conv1_w_max = np.transpose(conv1_w_torch, (2, 3, 1, 0))    # [3, 3, 1, 32] RSCF
    
    conv2_w_torch = model.conv2.weight.detach().cpu().numpy()  # [64, 32, 3, 3]
    conv2_w_max = np.transpose(conv2_w_torch, (2, 3, 1, 0))    # [3, 3, 32, 64] RSCF
    
    weights = {
        'conv1_W': conv1_w_max,
        'conv1_b': model.conv1.bias.detach().cpu().numpy(),
        'conv2_W': conv2_w_max,
        'conv2_b': model.conv2.bias.detach().cpu().numpy(),
        'fc1_W': model.fc1.weight.detach().cpu().numpy(),
        'fc1_b': model.fc1.bias.detach().cpu().numpy(),
        'fc2_W': model.fc2.weight.detach().cpu().numpy(),
        'fc2_b': model.fc2.bias.detach().cpu().numpy(),
    }
    
    print(f"   ✓ Extracted weights: conv1{weights['conv1_W'].shape}, conv2{weights['conv2_W'].shape}")
    print(f"                        fc1{weights['fc1_W'].shape}, fc2{weights['fc2_W'].shape}")
    
    # Save weights
    print("\n6. Saving weights...")
    weights_dir = Path(__file__).parent / "weights"
    weights_dir.mkdir(exist_ok=True)
    
    np.savez(
        weights_dir / "cnn_weights.npz",
        conv1_W=weights['conv1_W'].astype(np.float32),
        conv1_b=weights['conv1_b'].astype(np.float32),
        conv2_W=weights['conv2_W'].astype(np.float32),
        conv2_b=weights['conv2_b'].astype(np.float32),
        fc1_W=weights['fc1_W'].astype(np.float32),
        fc1_b=weights['fc1_b'].astype(np.float32),
        fc2_W=weights['fc2_W'].astype(np.float32),
        fc2_b=weights['fc2_b'].astype(np.float32),
    )
    
    print(f"   ✓ Weights saved to: {weights_dir / 'cnn_weights.npz'}")
    print("\n" + "=" * 80)
    print("✓ Training complete! Data + weights ready for MAX Graph inference.")
    print("=" * 80)
    
    # Show sample predictions
    print("\nSample predictions on test set:")
    print("-" * 60)
    model.eval()
    with torch.no_grad():
        for i in range(10):
            img, label = test_dataset[i]
            img_batch = img.unsqueeze(0).to(device)
            output = model(img_batch)
            pred = output.argmax(dim=1).item()
            correct_mark = "✓" if pred == label else "✗"
            print(f"  Image {i}: True={label}, Predicted={pred} {correct_mark}")
    print("-" * 60)


if __name__ == "__main__":
    train_and_save_weights()
