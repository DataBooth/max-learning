#!/usr/bin/env python3
"""
Download HuggingFace transformer models for MAX Graph.

This script downloads a sentiment analysis model from HuggingFace with
safetensors weights for use with MAX Graph API.
"""

import sys
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    print("❌ Error: Required packages not installed")
    print("Please ensure you're running via: pixi run download-models")
    sys.exit(1)


def download_model(
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    output_dir: str = "distilbert-sentiment",
    force: bool = False,
):
    """Download a HuggingFace model with safetensors weights.
    
    Args:
        model_name: HuggingFace model identifier
        output_dir: Directory name to save model
        force: If True, re-download even if model exists
    """
    
    print(f"🔥 Downloading {model_name} for MAX Graph")
    print("=" * 70)
    
    # Create output directory
    output_path = Path(__file__).parent / output_dir
    
    # Check if model already exists
    if not force and output_path.exists():
        safetensors_path = output_path / "model.safetensors"
        if safetensors_path.exists():
            print(f"\n✅ Model already exists at {output_path}")
            print("   Use --force to re-download")
            return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📦 Downloading model from HuggingFace...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Save tokenizer and model (includes safetensors)
        tokenizer.save_pretrained(output_path)
        model.save_pretrained(output_path)
        
        print(f"✅ Model and tokenizer saved to {output_path}")
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        sys.exit(1)
    
    # Verify required files for MAX Graph
    print(f"\n📋 Verifying output files...")
    required_files = ["model.safetensors", "vocab.txt", "config.json"]
    
    all_present = True
    for file in required_files:
        file_path = output_path / file
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✅ {file}: {size:.2f} MB")
        else:
            print(f"  ❌ {file}: NOT FOUND")
            all_present = False
    
    if all_present:
        print(f"\n🎉 Model download complete!")
        print(f"📁 Files saved to: {output_path.absolute()}")
        print(f"\n💡 MAX Graph will load weights from model.safetensors")
    else:
        print(f"\n⚠️  Some files are missing. Please check the output.")
        sys.exit(1)


if __name__ == "__main__":
    # Parse command line arguments
    force = "--force" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--force"]
    
    model_name = args[0] if len(args) > 0 else "distilbert-base-uncased-finetuned-sst-2-english"
    output_dir = args[1] if len(args) > 1 else "distilbert-sentiment"
    
    download_model(model_name, output_dir, force=force)
