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

# Import from installed package
from max_mlp import MLPRegressionModel


def load_california_housing_sample(n_samples=10):
    """Load a sample from cached California housing dataset.
    
    Args:
        n_samples: Number of samples to load
    
    Returns:
        X_sample: Features [n_samples, 8 features] (standardised)
        y_true: Target values [n_samples, 1]
        feature_names: Names of the 8 features
        scaler_params: Dictionary with 'mean' and 'scale' for standardisation
    """
    # Load cached dataset
    data_path = Path(__file__).parent / "data" / "california_housing.npz"
    data = np.load(data_path)
    
    X = data['data']
    y = data['target']
    feature_names = data['feature_names'].tolist()
    
    # Load scaler parameters (saved during training)
    weights_path = Path(__file__).parent / "weights" / "mlp_weights.npz"
    weights_data = np.load(weights_path)
    scaler_mean = weights_data['scaler_mean']
    scaler_scale = weights_data['scaler_scale']
    
    # Take first n_samples
    X_sample = X[:n_samples]
    y_true = y[:n_samples].reshape(-1, 1)
    
    # Standardise using training scaler parameters
    X_sample_scaled = (X_sample - scaler_mean) / scaler_scale
    
    scaler_params = {'mean': scaler_mean, 'scale': scaler_scale}
    
    return X_sample_scaled, y_true, feature_names, scaler_params


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
    
    # Load pre-trained weights
    print("1. Loading pre-trained weights...")
    weights_path = Path(__file__).parent / "weights" / "mlp_weights.npz"
    weights_data = np.load(weights_path)
    weights = {
        'W1': weights_data['W1'],
        'b1': weights_data['b1'],
        'W2': weights_data['W2'],
        'b2': weights_data['b2'],
        'W3': weights_data['W3'],
        'b3': weights_data['b3'],
    }
    print(f"   ✓ Weights loaded (trained with R²=0.79, RMSE=$52.9k)\n")
    
    # Load dataset
    print("2. Loading cached California housing dataset...")
    X_sample, y_true, feature_names, scaler_params = load_california_housing_sample(n_samples=10)
    print(f"   ✓ Loaded {len(X_sample)} samples with {X_sample.shape[1]} features")
    print(f"   Features: {', '.join(feature_names)}")
    print(f"   ✓ Features standardised (mean={scaler_params['mean'][0]:.2f}, scale={scaler_params['scale'][0]:.2f})\n")
    
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
    
    print("Note: This model was trained on 16,512 California housing samples.")
    print("Test set performance: R²=0.79, RMSE=$52.9k\n")
    
    print("✓ MLP regression example completed!")
    print("\nOperations demonstrated:")
    print("  - ops.matmul (3 linear transformations)")
    print("  - ops.add (bias additions)")
    print("  - ops.relu (2 activations)")
    print("  - Multi-layer feedforward architecture")
    print(f"\nModel implementation: src/python/max_mlp/")


if __name__ == "__main__":
    main()
