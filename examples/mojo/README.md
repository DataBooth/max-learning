# Mojo Examples

This directory contains Mojo-based examples for the max-learning repository.

## Current Examples

### Lexicon Baseline (Non-MAX Graph)

**Path**: `lexicon_baseline/`

A simple sentiment classifier written in pure Mojo that **does not use MAX Graph or MAX Engine**. This was the v0.1.0 baseline before implementing MAX Graph examples.

**Use case**: Educational baseline and reference for pure Mojo (no MAX dependencies).

**See**: `lexicon_baseline/README.md` for details.

## Future: MAX Graph in Mojo

The MAX Graph API is available in both Python and Mojo. Future examples will demonstrate MAX Graph using Mojo:

### Planned Examples

**01_elementwise/** (planned)
- Element-wise operations in Mojo
- Equivalent to `examples/python/01_elementwise/`
- Shows MAX Graph API in Mojo

**02_linear_layer/** (planned)
- Linear layer in Mojo
- Equivalent to `examples/python/02_linear_layer/`
- Matrix operations with MAX Graph in Mojo

**See**: `docs/planning/MOJO_EXAMPLES_PLAN.md` for full plan.

## Why Mojo + MAX Graph?

MAX Graph API works in both Python and Mojo, allowing you to:
- Use Python for rapid prototyping and flexibility
- Use Mojo for maximum performance and control
- Share the same MAX Engine backend

The planned Mojo examples will show:
- How MAX Graph code translates from Python to Mojo
- Performance comparisons
- When to use which language

## Current Focus

The repository currently focuses on Python examples for MAX Graph. The Mojo MAX Graph examples are planned for v0.4.0.

For now, start with the Python examples:
1. `examples/python/01_elementwise/elementwise_minimal.py` - Start here
2. `examples/python/02_linear_layer/linear_layer_minimal.py` - Continue here
3. Progress through the numbered examples

## Running Mojo Examples

### Lexicon Baseline

```bash
# Using pixi tasks
pixi run mojo-build   # Build the lexicon classifier
pixi run mojo-run     # Run with example text

# Or directly
mojo run examples/mojo/lexicon_baseline/main.mojo "Your text here"
```

### Future MAX Graph Examples

Will follow similar pattern:
```bash
mojo run examples/mojo/01_elementwise/elementwise.mojo
mojo run examples/mojo/02_linear_layer/linear_layer.mojo
```

## Dependencies

- **Mojo**: Modular's Mojo language
- **mojo-toml**: TOML parsing (for lexicon baseline)
- **mojo-dotenv**: Environment variables (for lexicon baseline)
- **MAX Engine**: (for future MAX Graph examples)

## Learning Path

1. **Python MAX Graph examples** (1️⃣-6️⃣) - Start here, current focus
2. **Mojo MAX Graph examples** (planned) - Compare with Python
3. **Lexicon baseline** (optional) - Pure Mojo reference, no MAX

---

For questions about Mojo + MAX Graph, see:
- [MAX Graph Documentation](https://docs.modular.com/max/graph/)
- [Mojo Programming Manual](https://docs.modular.com/mojo/manual/)
- `docs/planning/MOJO_EXAMPLES_PLAN.md` (this repository)
