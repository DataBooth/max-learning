"""
MLP Regression Example
=======================

Demonstrates a 3-layer MLP for regression using MAX Graph.

Architecture: Input(8) → FC(128)+ReLU → FC(64)+ReLU → FC(1)

Demonstrates:
- Multi-layer feedforward network
- Layer stacking pattern
- Regression output (continuous values)
- ReLU activations between layers
- No activation on output layer (regression)

Run:
  pixi run example-mlp
  python examples/python/03_mlp_regression/mlp_regression.py
"""

import sys
import tomllib
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.python.max_mlp import MLPRegressionModel


def load_california_housing_sample(n_samples=10):
    """Load a sample from California housing dataset.
    
    Args:
        n_samples: Number of samples to load
    
    Returns:
        X_sample: Features [n_samples, 8 features]
        y_true: Target values [n_samples, 1]
        feature_names: Names of the 8 features
    """
    # Load California housing dataset from sklearn (auto-cached)
    housing = fetch_california_housing()
    
    # Take first n_samples
    X_sample = housing.data[:n_samples].astype(np.float32)
    y_true = housing.target[:n_samples].reshape(-1, 1).astype(np.float32)
    feature_names = housing.feature_names
    
    return X_sample, y_true, feature_names


def main():
    # Load configuration
    config_path = Path(__file__).parent / "mlp_config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    
    print("=== MLP Regression Example ===\n")
    
    # Extract config
    input_size = config["model"]["input_size"]
    hidden_size1 = config["model"]["hidden_size1"]
    hidden_size2 = config["model"]["hidden_size2"]
    output_size = config["model"]["output_size"]
    
    print(f"Architecture: Input({input_size}) → FC({hidden_size1})+ReLU → FC({hidden_size2})+ReLU → FC({output_size})\n")
    
    # Initialize random weights (in practice, these would be trained)
    print("1. Initialising model with random weights...")
    np.random.seed(42)
    weights = {
        'W1': np.random.randn(hidden_size1, input_size) * 0.01,
        'b1': np.zeros(hidden_size1),
        'W2': np.random.randn(hidden_size2, hidden_size1) * 0.01,
        'b2': np.zeros(hidden_size2),
        'W3': np.random.randn(output_size, hidden_size2) * 0.01,
        'b3': np.zeros(output_size),
    }
    print(f"   ✓ Weights initialised\n")
    
    # Load dataset
    print("2. Loading California housing dataset from sklearn...")
    X_sample, y_true, feature_names = load_california_housing_sample(n_samples=10)
    print(f"   ✓ Loaded {len(X_sample)} samples with {X_sample.shape[1]} features")
    print(f"   Features: {', '.join(feature_names)}\n")
    
    # Build and compile model
    print("3. Building MAX Graph...")
    model = MLPRegressionModel(
        input_size=input_size,
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2,
        output_size=output_size,
        weights=weights,
        device="cpu",
    )
    print("   ✓ Graph compiled and loaded\n")
    
    # Make predictions
    print("4. Running inference on sample data...\n")
    predictions = model.predict(X_sample)
    
    # Display results
    print("=" * 80)
    print(f"{'Sample':<8} {'True Value':<15} {'Predicted':<15} {'Error':<15}")
    print("=" * 80)
    
    mse = 0.0
    for i in range(len(X_sample)):
        true_val = y_true[i, 0]
        pred_val = predictions[i, 0]
        error = abs(true_val - pred_val)
        mse += (true_val - pred_val) ** 2
        
        print(f"#{i+1:<7} ${true_val*100:.1f}k{'':<8} ${pred_val*100:.1f}k{'':<8} ${error*100:.1f}k")
    
    mse /= len(X_sample)
    print("=" * 80)
    print(f"Mean Squared Error: ${np.sqrt(mse)*100:.1f}k (RMSE)\n")
    
    print("Note: This model uses random weights for demonstration.")
    print("In practice, weights would be trained on the full California housing dataset.\n")
    
    print("✓ MLP regression example completed!")
    print("\nOperations demonstrated:")
    print("  - ops.matmul (3 linear transformations)")
    print("  - ops.add (bias additions)")
    print("  - ops.relu (2 activations)")
    print("  - Multi-layer feedforward architecture")
    print(f"\nModel implementation: src/python/max_mlp/")


if __name__ == "__main__":
    main()
