# Minimal MAX Graph Example

## Overview

This is the **simplest possible working example** of the MAX Graph API. It demonstrates the complete workflow from graph construction to inference in under 100 lines of code.

## What It Does

Implements a single-layer neural network: **y = ReLU(W^T @ x + b)**

- **Input**: 4 features
- **Output**: 2 values
- **Operation**: Matrix multiply → add bias → ReLU activation

## How It Fits Into MAX Framework

### MAX Architecture Layers

```
┌─────────────────────────────────────────────┐
│  1. Define Graph (Python)                   │  ← This example focuses here
│     - Specify computation                   │
│     - Define input/output types             │
│     - Build operation sequence              │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│  2. Compilation (MAX Compiler)              │  ← Happens automatically
│     - Optimise operations                   │
│     - Fuse ops where possible               │
│     - Select hardware-specific kernels      │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│  3. Execution (MAX Runtime)                 │  ← model.execute()
│     - Run optimised code                    │
│     - Hardware-agnostic                     │
│     - High performance                      │
└─────────────────────────────────────────────┘
```

This example demonstrates **Step 1** (graph definition) and shows how Steps 2-3 happen seamlessly.

## Key Concepts Demonstrated

### 1. Graph Construction

```python
with Graph("minimal_model", input_types=[input_spec]) as graph:
    x = graph.inputs[0].tensor
    # ... operations ...
    graph.output(y)
```

**MAX docs**: [Graph Construction](https://docs.modular.com/max/graph/)

### 2. Operations (ops)

```python
y = ops.matmul(x, ops.transpose(W, 0, 1))  # Matrix multiply
y = ops.add(y, b)                           # Add bias
y = ops.relu(y)                             # Activation
```

All operations are **declarative** - you describe *what* to compute, not *how*.

**MAX docs**: [Operations Reference](https://docs.modular.com/max/graph/ops)

### 3. Device Handling

```python
device = CPU()                              # Create device
session = InferenceSession(devices=[device])  # Session for compilation
```

MAX abstracts hardware - same code works on CPU, NVIDIA GPU, AMD GPU, etc.

**MAX docs**: [Device Management](https://docs.modular.com/max/graph/devices)

### 4. Compilation & Execution

```python
model = session.load(graph)     # Compile (one-time cost)
output = model.execute(input)   # Run inference (fast!)
```

**Compilation** happens once, **execution** is the fast path.

## Running the Example

```bash
# From repository root
pixi run python examples/python/minimal_max_graph.py
```

**Expected output**:
```
=== Minimal MAX Graph Example ===

1. Building computation graph...
   Graph 'minimal_model' created

2. Creating inference session and compiling graph...
   Graph compiled and loaded

3. Preparing input...
   Input shape: (1, 4)
   Input data: [[1. 2. 3. 4.]]

4. Executing inference...
   Output shape: (1, 2)
   Output data: [[8.6 0. ]]

5. Verifying with NumPy...
   Expected: [[8.6 0. ]]
   Match: True

✓ Complete!
```

## What Makes This "Minimal"?

1. **Inline weights**: Uses `ops.constant()` instead of loading from files
2. **Single layer**: Just one matmul + bias + activation
3. **Fixed shapes**: No dynamic batch sizes or variable length
4. **No model file**: Weights defined in code, not loaded

This keeps the example focused on **MAX Graph API fundamentals**.

## Comparison to Real Models

### This Example

```python
# Weights defined inline
W = ops.constant(np.array([[...]], dtype=np.float32), ...)

# Simple operation
y = ops.matmul(x, ops.transpose(W, 0, 1)) + b
```

### Real Model (e.g., DistilBERT)

```python
# Weights loaded from file
weights = load_weights([model_path])
W = weights.layer.weight.allocate(DType.float32).cast(dtype)

# Complex operations
embeddings = self.embeddings(input_ids)
hidden = self.encoder(embeddings, attention_mask)
logits = self.classifier(hidden[:, 0, :])
```

**Key difference**: Real models load weights from files and have many layers. The **graph construction pattern** is the same.

## Next Steps

After understanding this example:

1. **distilbert_sentiment.py** - See how a real model uses MAX Graph
2. **src/python/max_distilbert/** - Study a complete transformer implementation
3. **MAX Graph Tutorial** - [Build LLM from Scratch](https://llm.modular.com)

## MAX Framework Resources

- **MAX Documentation**: https://docs.modular.com/max/
- **MAX Graph API**: https://docs.modular.com/max/graph/
- **Graph Operations**: https://docs.modular.com/max/graph/ops
- **Device Management**: https://docs.modular.com/max/graph/devices
- **Modular Forums**: https://forum.modular.com

## Common Patterns

### Linear Layer

MAX doesn't have `ops.linear()`, use this pattern:

```python
output = ops.matmul(x, ops.transpose(W, 1, 0)) + bias
```

### Device Consistency

All ops must use the same device:

```python
device = x.device  # Get device from input
const = ops.constant(value, dtype=dtype, device=device)
```

### Shape Inspection

During development, print shapes:

```python
print(f"Tensor shape: {tensor.shape}")
```

## Troubleshooting

### "Device mismatch" errors

**Problem**: Operations on different devices

**Solution**: Pass `device=x.device` to all `ops.constant()` calls

### "Incompatible function arguments" in InferenceSession

**Problem**: Using `DeviceRef` instead of `CPU()`

**Solution**: Use `CPU()` for session:
```python
device = CPU()  # Not DeviceRef
session = InferenceSession(devices=[device])
```

### Import errors

**Problem**: MAX not installed

**Solution**: Run `pixi install` to set up environment

## Why MAX Over Alternatives?

**vs PyTorch**: No training overhead, better inference performance, hardware-portable

**vs ONNX Runtime**: Programmatic graphs (not file-based), more flexible

**vs TensorRT**: Not locked to NVIDIA, works on Apple Silicon + AMD

**Trade-off**: Compilation overhead (~seconds) vs inference speed (5-6x faster)

See also: `docs/MAX_FRAMEWORK_GUIDE.md` for detailed comparison.
