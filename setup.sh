#!/usr/bin/env bash
# Setup script for LlamaBlend Vision LoRA integration
# Usage: chmod +x setup.sh && ./setup.sh

set -e

# Navigate to script directory
cd "$(dirname "$0")"

# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Clone adapters if not present
mkdir -p adapters
cd adapters
if [ ! -d "CLIP-LoRA" ]; then
  git clone https://huggingface.co/MaxZanella/CLIP-LoRA
else
  echo "Adapter 'CLIP-LoRA' already exists, skipping clone."
fi
cd ..

echo "Setup complete!"
echo "To run inference:"
echo "  source .venv/bin/activate"
echo "  python3 Scripts/VisionLoraIntegration.py /path/to/your/image.jpg"
