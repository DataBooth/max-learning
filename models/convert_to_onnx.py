#!/usr/bin/env python3
"""
Convert HuggingFace transformer models to ONNX format for MAX Engine.

This script downloads a sentiment analysis model from HuggingFace and converts
it to ONNX format for use with Modular MAX Engine.
"""

import os
import sys
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError:
    print("Error: Required packages not installed")
    print("Please run: pip install transformers torch")
    sys.exit(1)


def convert_model_to_onnx(
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    output_dir: str = "distilbert-sentiment",
):
    """Download and convert a HuggingFace model to ONNX format."""
    
    print(f"🔥 Converting {model_name} to ONNX format for MAX Engine")
    print("=" * 70)
    
    # Create output directory
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📦 Downloading model from HuggingFace...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Save tokenizer files
        tokenizer.save_pretrained(output_path)
        print(f"✅ Tokenizer saved to {output_path}")
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        sys.exit(1)
    
    print(f"\n🔄 Converting to ONNX format...")
    
    # Prepare dummy input for export
    dummy_input = "This is a test sentence for ONNX export."
    inputs = tokenizer(
        dummy_input,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )
    
    # Export to ONNX
    onnx_path = output_path / "model.onnx"
    
    try:
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
            do_constant_folding=True,
            opset_version=14,
        )
        print(f"✅ ONNX model saved to {onnx_path}")
        
    except Exception as e:
        print(f"❌ Error converting to ONNX: {e}")
        print("\nℹ️  Trying alternative method with optimum library...")
        
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            
            ort_model = ORTModelForSequenceClassification.from_pretrained(
                model_name, export=True
            )
            ort_model.save_pretrained(output_path)
            print(f"✅ ONNX model saved using optimum")
            
        except ImportError:
            print("❌ optimum library not available")
            print("Install with: pip install optimum[onnxruntime]")
            sys.exit(1)
        except Exception as e2:
            print(f"❌ Error with optimum: {e2}")
            sys.exit(1)
    
    # Verify files
    print(f"\n📋 Verifying output files...")
    required_files = ["model.onnx", "vocab.txt", "config.json"]
    
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
        print(f"\n🎉 Model conversion complete!")
        print(f"📁 Files saved to: {output_path.absolute()}")
        print(f"\n💡 You can now use this model with MAX Engine in Mojo:")
        print(f'   from max import engine')
        print(f'   var session = engine.InferenceSession()')
        print(f'   var model = session.load("{output_path}/model.onnx")')
    else:
        print(f"\n⚠️  Some files are missing. Please check the output.")
        sys.exit(1)


if __name__ == "__main__":
    # Allow custom model name from command line
    model_name = sys.argv[1] if len(sys.argv) > 1 else "distilbert-base-uncased-finetuned-sst-2-english"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "distilbert-sentiment"
    
    convert_model_to_onnx(model_name, output_dir)
