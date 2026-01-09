#!/usr/bin/env python
"""Test embeddings layer in isolation."""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

# Load HuggingFace model
model_path = Path("models/distilbert-sentiment")
tokenizer = AutoTokenizer.from_pretrained(model_path)
hf_model = AutoModel.from_pretrained(model_path)
hf_model.eval()

# Simple test input
text = "good"
inputs = tokenizer(text, return_tensors="pt")
print(f"Text: '{text}'")
print(f"Input IDs: {inputs['input_ids']}")
print(f"Input IDs shape: {inputs['input_ids'].shape}")

# Get HuggingFace embeddings
with torch.no_grad():
    # Get just the embeddings (before transformer)
    word_embeddings = hf_model.embeddings.word_embeddings(inputs['input_ids'])
    position_ids = torch.arange(inputs['input_ids'].shape[1]).unsqueeze(0)
    position_embeddings = hf_model.embeddings.position_embeddings(position_ids)
    embeddings = word_embeddings + position_embeddings
    embeddings = hf_model.embeddings.LayerNorm(embeddings)
    
    print(f"\nHuggingFace Embeddings:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean().item():.6f}")
    print(f"  Std: {embeddings.std().item():.6f}")
    print(f"  Min: {embeddings.min().item():.6f}")
    print(f"  Max: {embeddings.max().item():.6f}")
    print(f"  First 5 values: {embeddings[0, 0, :5].numpy()}")

# Now let's check what happens with the full model
with torch.no_grad():
    full_output = hf_model(**inputs)
    hidden_states = full_output.last_hidden_state
    cls_token = hidden_states[:, 0, :]
    
    print(f"\nHuggingFace Transformer Output ([CLS] token):")
    print(f"  Shape: {cls_token.shape}")
    print(f"  Mean: {cls_token.mean().item():.6f}")
    print(f"  Std: {cls_token.std().item():.6f}")
    print(f"  Min: {cls_token.min().item():.6f}")
    print(f"  Max: {cls_token.max().item():.6f}")
    print(f"  First 5 values: {cls_token[0, :5].numpy()}")

# Now test with MAX (we'll need to add this part)
print("\n" + "="*70)
print("TODO: Compare with MAX embeddings output")
print("="*70)
