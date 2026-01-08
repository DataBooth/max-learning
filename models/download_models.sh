#!/usr/bin/env bash
#
# Download and convert sentiment analysis models for MAX Engine
#
# Usage: ./models/download_models.sh

set -e  # Exit on error

echo "🔥 mojo-inference-service Model Download"
echo "=========================================="
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    echo "Please install Python 3.8 or later"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Check/install required packages
echo "📦 Checking Python dependencies..."

REQUIRED_PACKAGES=("transformers" "torch")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $package" &> /dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo "⚠️  Missing packages: ${MISSING_PACKAGES[*]}"
    echo ""
    read -p "Install missing packages? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📥 Installing packages..."
        python3 -m pip install "${MISSING_PACKAGES[@]}"
    else
        echo "❌ Cannot proceed without required packages"
        echo "Please run: python3 -m pip install transformers torch"
        exit 1
    fi
fi

echo "✅ All dependencies available"
echo ""

# Run conversion script
echo "🔄 Downloading and converting DistilBERT model..."
echo ""

if [ -f "convert_to_onnx.py" ]; then
    python3 convert_to_onnx.py
else
    echo "❌ Error: convert_to_onnx.py not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Model download complete!"
echo ""
echo "Next steps:"
echo "  1. Build the Mojo project: pixi run build"
echo "  2. Run inference: pixi run inference --text 'Amazing product!' --model transformer"
echo ""
