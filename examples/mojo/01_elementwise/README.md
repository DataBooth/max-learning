# Mojo MAX Graph Examples - Status

## Current State (2026-01-12)

**Important**: Modular has indicated they may revisit Mojo interfaces for MAX in the future, but there are no concrete timelines or public APIs available currently ([GitHub issue #5546](https://github.com/modular/max/issues/5546)).

### What We Learned

Attempting to import MAX Graph modules in Mojo:
```mojo
from max.graph import Graph, TensorType, ops
from max.tensor import Tensor, TensorShape
```

Results in: `error: unable to locate module 'max'`

**Current Reality**:
- The Python MAX Graph API is fully functional and production-ready (see `examples/python/`)
  - Tutorial: [Get started with MAX graphs in Python](https://docs.modular.com/max/develop/get-started-with-max-graph-in-python)
- Mojo provides the [MAX AI Kernels library](https://docs.modular.com/mojo/lib#max-ai-kernels-library) for low-level computational primitives
  - These are building blocks, not graph construction APIs
  - Used to implement custom operations and kernels
- MAX Graph for Mojo is not yet available - no public timeline exists
- Current options:
  1. **Python MAX Graph API** ✅ (build graphs in Python) 
  2. **Native Mojo** ✅ (pure Mojo without MAX Graph, like `lexicon_baseline`)
  3. **Mojo with MAX AI Kernels** ✅ (custom ops/kernels)
  4. **Mojo MAX Graph API** ❌ (would be graph building in Mojo - doesn't exist)

### Repository Focus

Given the current state, this repository focuses on:

1. **Python MAX Graph examples** (fully functional) - progressive learning from simple ops to transformers
2. **Pure Mojo examples** (like `examples/mojo/lexicon_baseline/`) - showcase Mojo without MAX dependencies
3. **Future preparation** - if/when Mojo MAX Graph becomes available, this structure is ready

### Why This Directory Exists

This directory structure is preserved to:
- Document our investigation and findings
- Maintain parallel structure with functional Python examples  
- Be ready to add Mojo implementations if the API becomes available
- Demonstrate the current state of the MAX/Mojo toolchain (January 2026)

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

---

**Last Updated**: 2026-01-12  
**Mojo Version**: 0.26.1.0.dev2026010718  
**MAX Version**: 26.1.0.dev2026010718
