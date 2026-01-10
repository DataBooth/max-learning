#!/usr/bin/env bash
#
# Download HuggingFace models for MAX Graph
#
# Usage: pixi run download-models
# Note: This is a wrapper script. Use `pixi run download-models` to ensure
#       the correct Python environment with transformers is used.

set -e  # Exit on error

# Change to script directory
cd "$(dirname "$0")"

# Run download script with pixi's Python
python download_models.py
