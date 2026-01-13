# Mojo MAX Graph Examples - Status

## Current State (2026-01-13)

### Understanding the MAX Architecture

From the Modular team:
> "All kernels used by MAX are written in Mojo. MAX as a framework builds on Mojo, which is the fundamental layer for interacting with hardware. MAX itself is an orchestration framework for building graphs of these Mojo kernels which then are optimized via a graph compiler and run."

**Key Points**:
- **All of MAX is built on Mojo** - every kernel is Mojo code
- **MAX is the orchestration layer** - builds and optimizes graphs of Mojo kernels
- **Python API is for graph construction** - currently the primary way to build MAX graphs

### History of Mojo Graph API

Modular **did** have a Mojo API for graph construction, but:
- It was [open-sourced and deprecated](https://forum.modular.com/t/mojo-max-bindings/1499/3) in May 2025
- Built when Mojo was in a very different state
- Required complete reworking to modernize
- Maintenance issues were blocking GPU programming improvements

The Python Graph API represents a "v2 rewrite" incorporating lessons learned from the initial Mojo API.

### Current Reality

Attempting to import MAX Graph modules in Mojo:
```mojo
from max.graph import Graph, TensorType, ops
from max.tensor import Tensor, TensorShape
```

Results in: `error: unable to locate module 'max'`

**Your Options Today**:
1. **Python MAX Graph API** ✅ (recommended)
   - [Tutorial](https://docs.modular.com/max/develop/get-started-with-max-graph-in-python)
   - Stable, production-ready
   - Achieves state-of-the-art performance
   - Easy integration with NumPy, PyTorch, Hugging Face
   
2. **Write Mojo Kernels** ✅ 
   - All MAX kernels are open-source Mojo code
   - Use [MAX AI Kernels library](https://docs.modular.com/mojo/lib#max-ai-kernels-library)
   - Integrate custom ops into Python MAX graphs
   
3. **Pure Mojo** ✅
   - No MAX Graph dependency (like `lexicon_baseline`)
   - Direct hardware interaction
   
4. **Mojo MAX Graph API** ⏸️
   - Previously existed, now deprecated/archived
   - May be revisited in the future
   - No current timeline

### Repository Focus

Given the current state, this repository focuses on:

1. **Python MAX Graph examples** (fully functional) - progressive learning from simple ops to transformers
2. **Pure Mojo examples** (like `examples/mojo/lexicon_baseline/`) - showcase Mojo without MAX dependencies
3. **Future preparation** - if/when Mojo MAX Graph becomes available, this structure is ready

### Why This Directory Exists

This directory documents:
- **Investigation findings** about MAX architecture and Mojo's role
- **Historical context** - Mojo Graph API existed but was deprecated (May 2025)
- **Current best practices** - Python API is the production path for graph construction
- **Future possibilities** - Mojo Graph API may be revisited with better design
- **What works today** - Writing custom Mojo kernels and using Python for orchestration

This information helps developers understand:
- Why examples are in Python, not Mojo
- How MAX actually works under the hood (all Mojo kernels)
- The intentional design decision to use Python for graph construction
- That this isn't a "missing feature" but a deliberate architectural choice

## Current Structure
```
examples/mojo/01_elementwise/
└── README.md (this file - documents investigation and current state)
```

## For Working Examples

**Python MAX Graph** (fully functional):
- `examples/python/01_elementwise/` - working elementwise operations
- All 6 progressive Python examples are functional

**Pure Mojo** (no MAX Graph dependency):
- `examples/mojo/lexicon_baseline/` - sentiment analysis in pure Mojo

## References

- [Modular Forum: Mojo max bindings](https://forum.modular.com/t/mojo-max-bindings/1499/3) - Brad Larson's explanation of why Mojo Graph API was deprecated
- [Get started with MAX graphs in Python](https://docs.modular.com/max/develop/get-started-with-max-graph-in-python) - Official tutorial
- [MAX AI Kernels library](https://docs.modular.com/mojo/lib#max-ai-kernels-library) - Mojo kernel documentation
- [GitHub issue #5546](https://github.com/modular/max/issues/5546) - Community discussion

---

**Last Updated**: 2026-01-13  
**Mojo Version**: 0.26.1.0.dev2026010718  
**MAX Version**: 26.1.0.dev2026010718
