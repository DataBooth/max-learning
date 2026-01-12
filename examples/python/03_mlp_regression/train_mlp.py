"""
Train MLP on California housing dataset and save weights + data.

This is a helper script to pre-train the MLP model using sklearn.
The trained weights and dataset are cached for reproducibility.

Purpose:
- Download and cache California housing data
- Train MLP with sklearn
- Save weights and data to the repo
- Enable fully reproducible MAX Graph inference demo

Run once to generate weights:
    python examples/python/03_mlp_regression/train_mlp.py
"""

from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def train_and_save_weights():
    """Train MLP with sklearn and cache data + weights for reproducibility."""

    print("=" * 80)
    print("Training MLP on California Housing Dataset")
    print("=" * 80)

    # Load dataset
    print("\n1. Loading California housing dataset from sklearn...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names
    print(f"   ✓ Loaded {len(X)} samples with {X.shape[1]} features")

    # Cache dataset for reproducibility
    print("\n2. Caching dataset...")
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    np.savez(
        data_dir / "california_housing.npz",
        data=X.astype(np.float32),
        target=y.astype(np.float32),
        feature_names=feature_names,
    )
    print(f"   ✓ Dataset cached to: {data_dir / 'california_housing.npz'}")
    print(f"   ✓ Size: {X.nbytes + y.nbytes} bytes (~{(X.nbytes + y.nbytes) / 1024:.1f} KB)")

    # Split data
    print("\n3. Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   ✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # Standardise features (important for neural networks)
    print("\n4. Standardising features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   ✓ Features scaled (mean=0, std=1)")

    # Train MLP
    print("\n5. Training 3-layer MLP (128 → 64 → 1)...")
    print("   Architecture: Input(8) → FC(128)+ReLU → FC(64)+ReLU → FC(1)")

    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        verbose=False,
        early_stopping=True,
        validation_fraction=0.1,
    )

    mlp.fit(X_train_scaled, y_train)
    print(f"   ✓ Training completed in {mlp.n_iter_} iterations")

    # Evaluate
    print("\n6. Evaluating model...")
    train_score = mlp.score(X_train_scaled, y_train)
    test_score = mlp.score(X_test_scaled, y_test)

    # Calculate RMSE
    y_pred_test = mlp.predict(X_test_scaled)
    rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))

    print(f"   ✓ Train R² score: {train_score:.4f}")
    print(f"   ✓ Test R² score: {test_score:.4f}")
    print(f"   ✓ Test RMSE: ${rmse * 100:.1f}k")

    # Extract weights
    print("\n7. Extracting weights...")
    weights = {
        "W1": mlp.coefs_[0].T.astype(np.float32),  # Transpose for MAX Graph format
        "b1": mlp.intercepts_[0].astype(np.float32),
        "W2": mlp.coefs_[1].T.astype(np.float32),
        "b2": mlp.intercepts_[1].astype(np.float32),
        "W3": mlp.coefs_[2].T.astype(np.float32),
        "b3": mlp.intercepts_[2].astype(np.float32),
    }

    # Also save scaler parameters
    scaler_params = {
        "mean": scaler.mean_.astype(np.float32),
        "scale": scaler.scale_.astype(np.float32),
    }

    print(
        f"   ✓ Extracted weights: W1{weights['W1'].shape}, W2{weights['W2'].shape}, W3{weights['W3'].shape}"
    )

    # Save weights
    print("\n8. Saving weights...")
    weights_dir = Path(__file__).parent / "weights"
    weights_dir.mkdir(exist_ok=True)

    np.savez(
        weights_dir / "mlp_weights.npz",
        W1=weights["W1"],
        b1=weights["b1"],
        W2=weights["W2"],
        b2=weights["b2"],
        W3=weights["W3"],
        b3=weights["b3"],
        scaler_mean=scaler_params["mean"],
        scaler_scale=scaler_params["scale"],
    )

    print(f"   ✓ Weights saved to: {weights_dir / 'mlp_weights.npz'}")
    print("\n" + "=" * 80)
    print("✓ Training complete! Data + weights ready for MAX Graph inference.")
    print("=" * 80)

    # Show sample predictions
    print("\nSample predictions on test set:")
    print("-" * 60)
    sample_indices = [0, 100, 500, 1000, 2000]
    for i in sample_indices:
        true_val = y_test[i]
        pred_val = y_pred_test[i]
        error = abs(true_val - pred_val)
        print(
            f"  True: ${true_val * 100:6.1f}k  Predicted: ${pred_val * 100:6.1f}k  Error: ${error * 100:5.1f}k"
        )
    print("-" * 60)


if __name__ == "__main__":
    train_and_save_weights()
