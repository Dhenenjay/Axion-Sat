#!/bin/bash
# Lambda Labs Bootstrap - Install Warp CLI and setup environment
# Run this on your Lambda Labs instance

set -e

echo "========================================"
echo "Lambda Labs + Warp CLI Setup"
echo "========================================"
echo ""

# 1. Install Warp CLI
echo "1/4 Installing Warp CLI..."
curl -fsSL https://warp.dev/install.sh | bash

# Add to PATH
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# 2. Clone repository
echo "2/4 Cloning Axion-Sat repository..."
cd ~
if [ ! -d "Axion-Sat" ]; then
    git clone https://github.com/YOUR_USERNAME/Axion-Sat.git
else
    cd Axion-Sat
    git pull origin main
    cd ~
fi

# 3. Install dependencies
echo "3/4 Installing Python dependencies..."
cd ~/Axion-Sat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e pangaea-bench/
pip install hydra-core omegaconf tqdm numpy scikit-learn einops albumentations timm

# 4. Download TerraMind weights
echo "4/4 Downloading TerraMind weights..."
mkdir -p weights/hf/TerraMind-1.0-large
if [ ! -f "weights/hf/TerraMind-1.0-large/TerraMind_v1_large.pt" ]; then
    wget -O weights/hf/TerraMind-1.0-large/TerraMind_v1_large.pt \
        https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large/resolve/main/TerraMind_v1_large.pt
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Authenticate Warp CLI:"
echo "   warp auth login"
echo ""
echo "2. Start new Warp session with this conversation:"
echo "   warp init"
echo ""
echo "3. In the Warp chat, say:"
echo "   'Continue from where we left off. Run the pyramid decoder training.'"
echo ""
echo "Warp will have full context of our conversation and can:"
echo "  - Run the training"
echo "  - Monitor progress"
echo "  - Debug issues"
echo "  - Report results"
echo ""
