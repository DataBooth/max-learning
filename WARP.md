# WARP
This file provides project-specific guidance for Warp/Oz when working in this repository.

## Project overview
- This repo is a progressive learning path for Modular MAX’s Python Graph API (with benchmarks and tests).
- Dependency and environment management is done with Pixi (`pixi.toml`, `pixi.lock`).

## Quick start
```bash
pixi install
```

## Common tasks
Run these from the repository root.

### Run examples
```bash
# Minimal examples
pixi run python examples/python/01_elementwise/elementwise_minimal.py
pixi run python examples/python/02_linear_layer/linear_layer_minimal.py

# Pixi tasks (full examples)
pixi run example-elementwise-cpu
pixi run example-elementwise-gpu
pixi run example-linear
pixi run example-mlp
pixi run example-cnn
pixi run example-distilbert
```

### Tests
```bash
pixi run test-python
```

### Lint/format
```bash
pixi run ruff-check
pixi run ruff-format
pixi run pre-commit
```

### Benchmarks
```bash
pixi run benchmark-elementwise
pixi run benchmark-linear
pixi run benchmark-distilbert
pixi run benchmark-mlp
pixi run benchmark-cnn
pixi run benchmark-all
```

### Update MAX version safely
MAX nightlies can introduce breaking changes. Prefer the scripted workflow:

```bash
python scripts/update_max_version.py
```

## Repo structure notes
- Python package code lives in `src/python/` and is installed (editable) via Pixi.
- Tests live in `tests/python/`.
- Benchmarks live in `benchmarks/`.

## Conventions
- Use Ruff for Python formatting/linting (see `ruff.toml`).
- Prefer Australian English for documentation and comments.
- Keep `modular` locked in `pixi.toml` unless you are intentionally updating it (use the script above).
- Avoid committing large artefacts (models, benchmark outputs). The repo gitignores `models/`, `data/`, and `benchmarks/*/results/`.
