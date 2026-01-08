#!/usr/bin/env python
"""Simple comparison test between MAX and HuggingFace."""

from pathlib import Path
from src.max_distilbert.inference import DistilBertSentimentClassifier
from transformers import pipeline

model_path = Path("models/distilbert-sentiment")

# Load both models
print("Loading MAX model...")
max_classifier = DistilBertSentimentClassifier(model_path)

print("Loading HuggingFace model...")
hf_classifier = pipeline('sentiment-analysis', model=str(model_path), device='cpu')

# Test cases
test_texts = [
    "good",
    "bad",
    "great movie",
    "terrible"
]

print("\n" + "="*70)
print("Comparison Test:")
print("="*70)

for text in test_texts:
    # MAX prediction
    max_result = max_classifier.predict(text)
    
    # HF prediction  
    hf_result = hf_classifier(text)[0]
    
    print(f"\nText: '{text}'")
    print(f"  MAX: {max_result['label']:8} (confidence: {max_result['confidence']:.4f})")
    print(f"       Logits: [{max_result.get('debug_logits', 'N/A')}]")
    print(f"  HF:  {hf_result['label']:8} (confidence: {hf_result['score']:.4f})")
    
    match = "✓" if max_result['label'] == hf_result['label'] else "✗"
    print(f"  Match: {match}")
