# CNN MNIST Example

**⚠️ TODO: Fix accuracy issue (currently 30%, should be ~99%)**  
There's a layout conversion bug between PyTorch (NCHW) and MAX Graph (NHWC). The code runs but predictions are incorrect.

Demonstrates a CNN for image classification using MAX Graph.

## Architecture

```
Conv(1→32)→Pool→Conv(32→64)→Pool→Flatten→FC(128)→FC(10)
```

- **Conv Block 1**: 3×3 conv (1→32 channels), ReLU, 2×2 max pooling
- **Conv Block 2**: 3×3 conv (32→64 channels), ReLU, 2×2 max pooling  
- **Flatten**: Spatial (7×7×64) to vector (3136)
- **FC Block**: Linear (3136→128), ReLU, Linear (128→10)

## What This Demonstrates

1. **2D Convolutions** - Spatial feature extraction with ops.conv2d
2. **Max Pooling** - Downsampling with ops.max_pool2d
3. **Flatten operation** - Transition from convolutional to fully connected layers
4. **Image classification** - MNIST digit recognition (0-9)
5. **Layout conversion** - NCHW (PyTorch) to NHWC (MAX Graph)

## MAX Graph Operations Used

- `ops.conv2d` - 2D convolution for spatial feature extraction
- `ops.max_pool2d` - Max pooling for downsampling
- `ops.flatten` - Flatten spatial dimensions
- `ops.transpose` - Convert between NCHW and NHWC layouts
- `ops.matmul`, `ops.add`, `ops.relu` - Fully connected layers

## Running the Example

```bash
# Via pixi task
pixi run example-cnn

# Or directly
python examples/python/04_cnn_mnist/cnn_mnist.py

# GPU not yet supported (matmul kernels unavailable)
# python examples/python/04_cnn_mnist/cnn_mnist.py --device gpu
```

## Dataset

Uses **MNIST** handwritten digits (28×28 grayscale images):
- **Training**: 60,000 samples
- **Test**: 10,000 samples
- **Classes**: 10 digits (0-9)
- **Demo**: 10 cached test samples

The dataset is automatically downloaded by PyTorch on first training run.

## Training

The model is trained using PyTorch and weights are converted to MAX Graph format:

```bash
cd examples/python/04_cnn_mnist
python train_cnn.py
```

This will:
1. Download MNIST dataset (~10 MB)
2. Train CNN for 2 epochs (~2 minutes on M1 Pro)
3. Achieve ~99% test accuracy
4. Save weights to `weights/cnn_weights.npz`
5. Cache 10 test samples to `data/mnist_samples.npz`

## Known Issues

### Accuracy Problem (30% instead of 99%)
**Status**: Active bug  
**Cause**: Layout conversion issue between PyTorch (NCHW) and MAX Graph (NHWC)  
**Impact**: Model runs but produces incorrect predictions  
**Workaround**: None yet

The weight conversion from PyTorch format `[O, I, H, W]` to MAX Graph RSCF format `[H, W, I, O]` may have an error, or the input data layout conversion needs adjustment.

### GPU Not Supported
**Status**: Expected limitation  
**Cause**: MAX Graph matmul kernels not available for Apple Silicon GPU yet  
**Error**: `Current compilation target does not support operation: mma`  
**Workaround**: Use CPU (default)

## Output Example

```
=== CNN MNIST Classification Example ===

Architecture: Conv(1→32)→Pool→Conv(32→64)→Pool→Flatten→FC(128)→FC(10)

1. Loading pre-trained weights...
   ✓ Weights loaded (trained to 99.10% test accuracy)

2. Loading cached MNIST test samples...
   ✓ Loaded 10 samples (28×28 grayscale images)

3. Building MAX Graph on CPU...
   ✓ Graph compiled and loaded

4. Running inference on test images...

================================================================================
Image    True   Predicted    Confidence   Status  
================================================================================
#1       7      6            42.3%        ✗
...
================================================================================
Accuracy: 30.0% (3/10)  # TODO: Should be ~100%
```

## Implementation Structure

```
src/python/max_cnn/
├── __init__.py           # Package exports
├── model.py              # CNNClassifier class and graph builder
└── inference.py          # High-level CNNClassificationModel API

examples/python/04_cnn_mnist/
├── README.md             # This file
├── cnn_mnist.py          # Demo script
├── cnn_config.toml       # Model configuration
├── train_cnn.py          # PyTorch training script
├── data/                 # Cached test samples
│   └── mnist_samples.npz
└── weights/              # Trained weights (RSCF format)
    └── cnn_weights.npz
```

## Learning Progression

This example builds on:
- **01_elementwise** - Basic operations (mul, add, relu)
- **02_linear_layer** - Single linear transformation
- **03_mlp_regression** - Multi-layer feedforward networks

And demonstrates new concepts:
- **Spatial operations** - Convolution and pooling for 2D data
- **Conv→FC pattern** - Common architecture for vision tasks
- **Layout conversions** - Handling different tensor formats

Next examples:
- **05_rnn_forecast** - Sequential processing for time series
- **06_distilbert_sentiment** - Transformer architecture with attention

## Debug Notes

To investigate the accuracy issue:
1. Verify weight shapes after conversion (RSCF format)
2. Check input data preprocessing (normalization, layout)
3. Compare intermediate activations between PyTorch and MAX Graph
4. Test with identity weights to isolate the bug

The model compiles and runs successfully, so the issue is in data/weight handling, not the graph construction.
