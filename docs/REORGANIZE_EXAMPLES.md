# Examples Reorganization Plan

## Current Status

Started reorganization but need to complete. Created directory structure:
- `examples/python/01_elementwise/` - Created, has new unified example
- `examples/python/02_linear_layer/` - Created, needs files
- `examples/python/03_distilbert_sentiment/` - Created, needs files

## Remaining TODOs

### 1. Complete Element-wise Example (01_elementwise/)
- [x] Create `elementwise.py` with `--device cpu|gpu` support
- [ ] Move/create README from old GPU experiments doc
- [ ] Remove old `elementwise_gpu.py` from root

### 2. Linear Layer Example (02_linear_layer/)
- [ ] Rename/move `minimal_max_graph.py` → `linear_layer.py`
- [ ] Move `README_minimal_max_graph.md` → `README.md`
- [ ] Add `--device cpu|gpu` support (will fail on GPU but documented)
- [ ] Remove old files from root

### 3. DistilBERT Sentiment Example (03_distilbert_sentiment/)
- [ ] Create `distilbert_sentiment.py` wrapper
  ```python
  # Simple wrapper around src/python/max_distilbert/inference.py
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))
  from max_distilbert import DistilBertSentimentClassifier
  
  def main():
      model_path = Path(__file__).parent.parent.parent / "models" / "distilbert-sentiment"
      classifier = DistilBertSentimentClassifier(model_path)
      # Demo examples...
  ```
- [ ] Create README explaining the example

### 4. Update Documentation
- [ ] Update main examples README
- [ ] Move GPU experiments README to appropriate location
- [ ] Update all cross-references

### 5. Update pixi.toml tasks
```toml
[tasks]
# Examples
example-elementwise-cpu = "python examples/python/01_elementwise/elementwise.py --device cpu"
example-elementwise-gpu = "python examples/python/01_elementwise/elementwise.py --device gpu"
example-linear = "python examples/python/02_linear_layer/linear_layer.py"
example-distilbert = "python examples/python/03_distilbert_sentiment/distilbert_sentiment.py"
```

### 6. Clean up old files
- [ ] Remove `examples/python/minimal_max_graph.py`
- [ ] Remove `examples/python/minimal_max_graph_gpu.py`
- [ ] Remove `examples/python/elementwise_gpu.py`
- [ ] Remove or consolidate READMEs in root

## Benefits of New Structure

1. **Numbered progression**: 01, 02, 03 shows learning path
2. **Self-contained**: Each example in its own directory with README
3. **Device flexibility**: CPU/GPU support where applicable
4. **Better names**: "linear_layer" clearer than "minimal_max_graph"
5. **Complete coverage**: Element-wise → Linear → Full model

## Quick Commands

```bash
# After reorganization complete:
pixi run example-elementwise-cpu
pixi run example-elementwise-gpu
pixi run example-linear
pixi run example-distilbert
```

## Notes

- Keep GPU experiments documentation but integrate into example READMEs
- Each README should explain:
  - What operations are used
  - How it fits in MAX framework
  - CPU/GPU support status
  - Links to MAX docs
