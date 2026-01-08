# Debugging Plan: MAX DistilBERT Accuracy Issue

## Current Status

✅ **Working**: 
- Graph builds and compiles successfully
- Weights load correctly
- Model runs inference without errors
- Produces predictions

❌ **Problem**:
- Confidence scores much lower than HuggingFace baseline
  - MAX: ~50-60% confidence (logits ~0.1-0.3)
  - HF: ~99% confidence (logits ~10)
- One prediction incorrect ("waste of money" → POSITIVE)

## Hypothesis List

### 1. Attention Mask Format/Application ⭐ HIGH PRIORITY
**Issue**: Attention mask might not be applied correctly
**Evidence**: 
- We convert 0/1 mask to additive mask with -10000
- PyTorch SDPA uses different mask format
**Test**: Verify mask is applied correctly by checking attention scores with/without mask

### 2. Weight Transposition
**Issue**: Linear layer weights might need transposition
**Evidence**: We do `matmul(x, transpose(W, 1, 0))` but maybe should be different
**Test**: Check weight shapes and multiplication order against HF

### 3. Layer Normalization Order or Epsilon
**Issue**: Layer norm might have wrong epsilon or be applied at wrong point
**Evidence**: We use 1e-12 from config
**Test**: Verify epsilon value and application order matches HF exactly

### 4. Embedding Layer
**Issue**: Position embeddings or word embeddings might be wrong
**Evidence**: Simple inputs should have strong embeddings from pre-training
**Test**: Extract embeddings from both models and compare

### 5. Dropout/Training Mode
**Issue**: Model might think it's in training mode
**Evidence**: We don't implement dropout, assuming inference mode
**Test**: Verify no dropout is being applied anywhere

## Systematic Debugging Approach

### Phase 1: Isolate the Layer ⭐ START HERE
1. **Test embeddings only**
   - Run both HF and MAX embeddings on same input
   - Compare outputs numerically
   - If different → bug in embeddings
   - If same → bug is downstream

2. **Test first transformer layer**
   - Get embedding output
   - Run through first transformer block
   - Compare MAX vs HF outputs
   - Narrow down to attention vs FFN

3. **Test classifier head**
   - Get transformer output (use HF encoder)
   - Run through MAX classifier head
   - Compare with HF classifier head
   - Check if weights are actually loaded

### Phase 2: Detailed Component Analysis
Once we know which layer has the bug:

**If Embeddings**:
- Check word_embeddings gather operation
- Check position_embeddings indexing
- Check LayerNorm parameters (weight, bias, epsilon)

**If Attention**:
- Check Q, K, V projections individually
- Check attention score calculation and scaling  
- Check attention mask application
- Check output projection

**If FFN**:
- Check lin1 projection
- Check GELU activation
- Check lin2 projection

**If Classifier**:
- Check weight loading
- Check matrix multiplication order
- Check bias addition

### Phase 3: Numerical Comparison
For each suspected component:
1. Extract weights from safetensors
2. Run same input through HF and MAX
3. Compare intermediate outputs
4. Find first point where outputs diverge
5. Fix the bug at that point

## Quick Win Tests

### Test 1: Verify Weight Loading
```python
# Check if classifier weights match
import torch
from safetensors import safe_open

tensors = safe_open('models/distilbert-sentiment/model.safetensors', framework='pt')
classifier_weight_expected = tensors.get_tensor('classifier.weight').numpy()

# Compare with what MAX loaded
# (need to extract from compiled model)
```

### Test 2: Check Attention Mask Effect
Run inference with:
- All 1s mask (no masking)
- Correct mask
- All 0s mask (mask everything)

If results are similar, mask isn't being used correctly.

### Test 3: Simple Input Test
Test with single token inputs:
- "good" → should be POSITIVE with high confidence
- "bad" → should be NEGATIVE with high confidence

If these fail, embeddings or classifier head is broken.

## Tools Needed

1. **Extraction script** to get intermediate outputs from HF model
2. **Comparison script** to numerically compare tensors
3. **Visualization** of attention weights to verify they make sense

## Expected Timeline

- **Phase 1**: 1-2 hours
- **Phase 2**: 1-2 hours  
- **Phase 3**: 30 min - 1 hour

**Total**: 3-5 hours to full fix

## Notes

- The fact that model runs and produces *some* correct predictions suggests:
  - Architecture is correct
  - Weights are mostly loaded
  - Bug is likely subtle (wrong order, sign, scale, etc.)
  
- Small logit magnitudes suggest:
  - Output isn't being amplified properly
  - Could be missing activation somewhere
  - Could be attention not focusing properly

## Next Actions

1. ✅ Create this debugging plan
2. ⏳ Implement embeddings comparison test
3. ⏳ Run systematic layer-by-layer comparison
4. ⏳ Fix identified bugs
5. ⏳ Verify full model accuracy matches HF
