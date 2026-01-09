# Apple Silicon GPU Experiments

## Overview

This directory contains experiments testing MAX Graph operations on Apple Silicon GPU (M3).

## Files

- **`elementwise_gpu.py`** - Element-wise operations (add, multiply, relu)
- **`minimal_max_graph_gpu.py`** - Linear layer with matmul

## Findings

### Current Status (January 2026)

Apple Silicon GPU support in MAX is **partial** - depends on kernel availability.

**Source**: Modular Discord feedback:
> "That's mostly a matter of adding kernels for it. There are still a few kernels without generic fallbacks, but some models in MAX might work."

### Experiment Results

#### Element-wise Operations (`elementwise_gpu.py`)

**Operations tested**:
- `ops.mul` - Element-wise multiplication
- `ops.add` - Element-wise addition  
- `ops.relu` - Element-wise ReLU

**Result**: ✅ GPU kernels exist, ❌ Metal toolchain issue

```
✓ Accelerator device found: Device(type=gpu,id=0)
✓ Graph 'elementwise_gpu' created
✗ Compilation failed: Xcode/Metal toolchain not properly configured
```

**Error**:
```
max/kernels/src/Mogg/MOGGKernelAPI:1:1: error: Please make sure Xcode is installed and setup correctly
xcrun: error: unable to find utility "metallib", not a developer tool or in PATH
```

**What this means**:
- GPU kernels ARE available for these operations
- Compilation attempts to generate Metal shaders
- Xcode command-line tools need proper setup

**Potential fix**:
```bash
# Install Xcode command-line tools
xcode-select --install

# Accept Xcode license
sudo xcodebuild -license accept

# Set Xcode path
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
```

#### Matrix Operations (`minimal_max_graph_gpu.py`)

**Operations tested**:
- `ops.matmul` - Matrix multiplication
- `ops.transpose` - Matrix transpose
- `ops.add` - Addition
- `ops.relu` - ReLU

**Result**: ❌ Missing GPU kernel for `matmul`

```
✓ Accelerator device found: Device(type=gpu,id=0)
✓ Graph 'minimal_model_gpu' created
✗ Compilation failed: matmul kernel not available
```

**Error**:
```
max/kernels/src/linalg/matmul/gpu/_multistage_gemm_gpu.mojo:714:4: error: function instantiation failed
...
note: constraint failed: Current compilation target does not support operation: mma
```

**What this means**:
- `matmul` operation doesn't have GPU kernel for Apple Silicon yet
- Specifically missing: `mma` (matrix multiply-accumulate) operation
- Would need CPU fallback for models using matmul

## Key Takeaways

### What Works (with proper setup)
✅ Element-wise operations (add, mul, relu)  
✅ GPU device detection
✅ Graph building for GPU

### What Doesn't Work Yet
❌ Matrix multiplication (`matmul`) on GPU  
❌ Metal toolchain auto-configuration

### Implications for Our DistilBERT Model

Our DistilBERT implementation uses:
- Heavy use of `matmul` (every transformer layer)
- Attention operations (which use matmul)
- Feed-forward networks (which use matmul)

**Conclusion**: DistilBERT won't run on Apple Silicon GPU yet due to missing `matmul` kernels.

## Comparison to Other Frameworks

### MAX (Current)
- GPU kernels: Partial (element-wise ✅, matmul ❌)
- Requires: Xcode/Metal setup
- Status: Work in progress

### PyTorch with MPS
- GPU support: Complete for most operations
- Matmul: ✅ Works on Apple Silicon
- Status: Production-ready

### ONNX Runtime with CoreML
- GPU support: Via CoreML backend
- Status: Production-ready for inference

## Running the Examples

### Prerequisites
1. Apple Silicon Mac (M1/M2/M3)
2. Xcode command-line tools installed
3. MAX installed via pixi

### Commands

```bash
# Element-wise operations (will fail on Metal toolchain)
pixi run python examples/python/elementwise_gpu.py

# Matrix operations (will fail on missing kernel)
pixi run python examples/python/minimal_max_graph_gpu.py
```

## Future Work

Once Metal toolchain is properly configured:
1. ✅ Element-wise operations should work on GPU
2. ❌ Matmul-based models still need CPU fallback
3. ⏳ Wait for Modular to add more GPU kernels

## References

- **MAX Documentation**: https://docs.modular.com/max/
- **MAX Graph Devices**: https://docs.modular.com/max/graph/devices
- **Modular Forums**: https://forum.modular.com
- **Modular Discord**: https://discord.gg/modular

## Contributing

If you get GPU execution working:
1. Document your setup steps
2. Note which operations work
3. Share findings in Modular Discord/Forums
