# Mojo Source Modules

This directory contains reusable Mojo modules/libraries for MAX Graph examples.

## Structure

Similar to `src/python/`, this directory contains:
- Reusable model implementations
- Shared utilities
- Modules that can be imported by examples

## Planned Modules

When we implement MAX Graph examples in Mojo (v0.4.0), this will include:

```
src/mojo/
├── max_elementwise/   # Elementwise operations module
├── max_linear/        # Linear layer module
├── utils/             # Shared utilities
└── README.md          # This file
```

## Current Status

**Empty** - No Mojo MAX Graph modules yet.

The lexicon classifier was previously here but has been moved to `examples/mojo/lexicon_baseline/` as it's not a reusable MAX Graph module (it's a v0.1.0 baseline with no MAX dependencies).

## Future

When implementing Mojo MAX Graph examples:
1. Create reusable modules here (e.g., `max_elementwise/`)
2. Use them in `examples/mojo/01_elementwise/`, etc.
3. Mirror the Python structure: `src/python/max_*` → `src/mojo/max_*`

See `docs/planning/MOJO_EXAMPLES_PLAN.md` for details.

## Why Keep This Directory?

- **Consistency**: Mirrors `src/python/` structure
- **Reusability**: Place for shared Mojo code
- **Organization**: Separates examples from libraries
- **Future-ready**: Ready for MAX Graph Mojo implementations

---

For current Mojo code, see `examples/mojo/lexicon_baseline/` (v0.1.0 baseline, non-MAX Graph).
