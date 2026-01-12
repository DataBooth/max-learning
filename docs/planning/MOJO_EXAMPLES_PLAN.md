# Plan: Mojo MAX Graph Examples

## Objective

Create Mojo equivalents of Python examples 01 and 02 to demonstrate:
1. MAX Graph API is available in both Python and Mojo
2. Side-by-side comparison showing similarities and differences
3. Performance comparison between Python and Mojo implementations

## Why This Matters

- **Language comparison**: Show developers familiar with Python how Mojo differs
- **Performance insights**: Measure if Mojo provides benefits for graph construction
- **Learning path**: Help users understand when to use Python vs Mojo
- **Completeness**: Parallel the Python learning progression in Mojo

## Proposed Examples

### 1. Element-wise Operations (Mojo)
**Location**: `examples/mojo/01_elementwise/`

**Files**:
- `elementwise.mojo` - Equivalent to Python minimal example
- `elementwise_config.toml` - Same config as Python version
- `README.md` - Documentation comparing Python vs Mojo

**Operations**: `y = relu(x * 2.0 + 1.0)`

**Key differences to highlight**:
- Import statements (`from max.graph import ...`)
- Type annotations (Mojo is statically typed)
- Memory management
- Compilation/execution model

### 2. Linear Layer (Mojo)
**Location**: `examples/mojo/02_linear_layer/`

**Files**:
- `linear_layer.mojo` - Equivalent to Python minimal example
- `linear_layer_config.toml` - Same config as Python version
- `README.md` - Documentation with Python comparison

**Operation**: `y = relu(x @ W^T + b)`

**Key differences to highlight**:
- Matrix operations in Mojo
- Parameter handling
- Type system benefits

## Structure

### Directory Layout
```
examples/mojo/
├── 01_elementwise/
│   ├── elementwise.mojo
│   ├── elementwise_config.toml
│   └── README.md
├── 02_linear_layer/
│   ├── linear_layer.mojo
│   ├── linear_layer_config.toml
│   └── README.md
└── README.md (overview of Mojo examples)
```

## Documentation Requirements

### Main Mojo README (`examples/mojo/README.md`)
Should include:
- Overview of Mojo MAX Graph API
- Why use Mojo vs Python for MAX Graph
- Links to both examples
- Comparison table: Python vs Mojo features
- When to choose each language

### Individual Example READMEs
Each should include:
- Side-by-side code comparison with Python equivalent
- Performance comparison (if measurable)
- Key differences explained
- Migration notes (Python → Mojo)

## Testing

Create Mojo tests in `tests/mojo/`:
- `tests/mojo/01_elementwise/`
- `tests/mojo/02_linear_layer/`

Tests should verify:
- Correctness (same results as Python)
- Successful compilation
- Successful execution

## Pixi Tasks

Add to `pixi.toml`:
```toml
[tasks]
# Mojo examples
example-elementwise-mojo = "mojo run examples/mojo/01_elementwise/elementwise.mojo"
example-linear-mojo = "mojo run examples/mojo/02_linear_layer/linear_layer.mojo"

# Mojo tests
test-mojo-all = "mojo test tests/mojo/"
test-elementwise-mojo = "mojo test tests/mojo/01_elementwise/"
test-linear-mojo = "mojo test tests/mojo/02_linear_layer/"
```

## Benchmarking

Consider adding:
- `benchmarks/mojo_vs_python/` - Compare graph construction and execution times
- Metrics to capture:
  - Graph build time
  - Compilation time
  - Execution time
  - Memory usage

## Open Questions

1. **MAX Graph API completeness in Mojo**: Are all ops available? Any limitations vs Python?
2. **Mojo Module API**: Is `max.nn.Module` available in Mojo like in Python?
3. **Packaging**: How to structure Mojo code for reusability (equivalent to Python packages)?
4. **Dependencies**: Does Mojo version need different setup than Python examples?

## Implementation Steps

1. ✅ Create this plan document
2. ☐ Research Mojo MAX Graph API documentation
3. ☐ Create `examples/mojo/README.md` (overview)
4. ☐ Implement `01_elementwise/elementwise.mojo`
5. ☐ Create `01_elementwise/README.md` with Python comparison
6. ☐ Add tests for elementwise example
7. ☐ Implement `02_linear_layer/linear_layer.mojo`
8. ☐ Create `02_linear_layer/README.md` with Python comparison
9. ☐ Add tests for linear layer example
10. ☐ Add pixi tasks for Mojo examples
11. ☐ Create benchmarking comparison (optional but valuable)
12. ☐ Update main README to mention Mojo examples
13. ☐ Update release notes for next version

## Success Criteria

- [ ] Two working Mojo examples that produce same results as Python equivalents
- [ ] Clear documentation explaining Python vs Mojo differences
- [ ] Tests passing for both examples
- [ ] Pixi tasks working for easy execution
- [ ] README updated to include Mojo learning path

## Timeline

Proposed for **v0.4.0** release.

## Resources

- [MAX Graph API (Mojo)](https://docs.modular.com/max/graph/) - Check for Mojo-specific docs
- [Mojo Programming Manual](https://docs.modular.com/mojo/manual/)
- Existing `src/mojo/lexicon_classifier/` - Reference for Mojo project structure

---

**Status**: Planning  
**Priority**: Medium (enhances learning value, shows language comparison)  
**Complexity**: Medium (requires Mojo knowledge, API research)
