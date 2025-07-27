# LlamaBlend Vision LoRA Integration Package

This package provides a turnkey setup to integrate the CLIP-LoRA adapter with Meta’s Llama-3.2 Vision model in your LlamaBlend macOS app.

## Contents

- `setup.sh` — bootstrap your environment  
- `requirements.txt` — Python dependencies  
- `Scripts/VisionLoraIntegration.py` — integration script for vision-LoRA + Llama  
- `adapters/CLIP-LoRA` — will be cloned by `setup.sh`  
- `README.md` — this file  

## Quick Start

1. **Download**  
   - Download and unzip this package on your Mac.

2. **Click**  
   - In Terminal, navigate into the package folder:
     ```bash
     cd path/to/LlamaBlend_Vision_LoRA_Package
     ```
   - Make the setup script executable and run it:
     ```bash
     chmod +x setup.sh
     ./setup.sh
     ```

3. **Tada!**  
   - Activate the virtual environment and run inference:
     ```bash
     source .venv/bin/activate
     python3 Scripts/VisionLoraIntegration.py /path/to/your/image.jpg
     ```

## Integrate with Xcode

1. **Add Script**  
   - In Xcode, add `Scripts/VisionLoraIntegration.py` to your project.

2. **Run Script Phase**  
   - Add a Run Script build phase with:
     ```bash
     cd "${SRCROOT}"
     source .venv/bin/activate
     python3 Scripts/VisionLoraIntegration.py
     ```

3. **Build & Run**  
   - Save and build: adapter will download and stitch into LlamaBlend automatically.

Enjoy your vision-to-text pipeline in LlamaBlend!
