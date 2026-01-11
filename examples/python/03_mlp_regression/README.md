# MLP Regression Example

Demonstrates a 3-layer Multi-Layer Perceptron (MLP) for regression using MAX Graph.

## Architecture

```
Input(8) → FC(128) + ReLU → FC(64) + ReLU → FC(1)
```

- **Input**: 8 features (California housing dataset)
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 64 neurons with ReLU activation
- **Output**: 1 continuous value (house price prediction)

## What This Demonstrates

1. **Multi-layer feedforward network** - Stacking multiple linear transformations
2. **Layer composition pattern** - How to build deeper networks in MAX Graph
3. **Regression task** - Predicting continuous values (vs classification)
4. **ReLU activations** - Non-linearity between layers
5. **No output activation** - Common pattern for regression

## MAX Graph Operations Used

- `ops.matmul` - Matrix multiplication for linear transformations (3×)
- `ops.transpose` - Weight matrix transposition
- `ops.add` - Bias additions (3×)
- `ops.relu` - ReLU activations (2×, not on output layer)

## Running the Example

```bash
# Via pixi task
pixi run example-mlp

# Or directly
python examples/python/03_mlp_regression/mlp_regression.py
```

## Dataset

Uses the **California Housing dataset** from sklearn (20,640 samples, 8 features):

- **Features**: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- **Target**: Median house value (in $100,000s)
- **Example runs on**: 10 samples for demonstration

The dataset is automatically downloaded and cached by sklearn on first use (~500 KB).

## Output Example

```
=== MLP Regression Example ===

Architecture: Input(8) → FC(128)+ReLU → FC(64)+ReLU → FC(1)

1. Initialising model with random weights...
   ✓ Weights initialised

2. Loading California housing dataset from sklearn...
   ✓ Loaded 10 samples with 8 features
   Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

3. Building MAX Graph...
   ✓ Graph compiled and loaded

4. Running inference on sample data...

================================================================================
Sample   True Value      Predicted       Error          
================================================================================
#1       $452.6k         $-0.5k         $453.1k
...
================================================================================

✓ MLP regression example completed!
```

## Implementation Details

The example uses **random weights** for demonstration purposes. In a real scenario:
1. Weights would be trained on the full dataset using gradient descent
2. The model would learn to minimize mean squared error (MSE)
3. A trained model would achieve much lower prediction error

## Configuration

See `mlp_config.toml` for model hyperparameters:
- Input size: 8 features
- Hidden layer sizes: 128, 64
- Output size: 1 (single continuous value)

## Implementation Structure

```
src/python/max_mlp/
├── __init__.py           # Package exports
├── model.py              # MLPRegressor class and graph builder
└── inference.py          # High-level MLPRegressionModel API

examples/python/03_mlp_regression/
├── README.md             # This file
├── mlp_regression.py     # Demo script
└── mlp_config.toml       # Model configuration
```

## Learning Progression

This example builds on:
- **01_elementwise** - Basic operations (mul, add, relu)
- **02_linear_layer** - Single linear transformation

And leads to:
- **04_cnn_mnist** (next) - Spatial convolutions for image classification
- **05_rnn_forecast** - Sequential processing for time series
- **06_distilbert_sentiment** - Transformer architecture with attention

## Next Steps

To train this model properly, you would:
1. Load the full California housing dataset (20K samples)
2. Implement a training loop with MSE loss
3. Use gradient descent to update weights
4. Evaluate on a held-out test set

This example focuses on MAX Graph inference patterns rather than training.
