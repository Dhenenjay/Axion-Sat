#!/bin/bash
set -e

echo "=========================================="
echo "Axion-Sat Lambda Labs Setup"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get install -y wget aria2 zstd

# Clone repository
echo "Cloning Axion-Sat repository..."
cd ~
git clone https://github.com/Dhenenjay/Axion-Sat.git
cd Axion-Sat

# Setup Python environment
echo "Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Create data directories
echo "Creating data directories..."
mkdir -p data/raw
mkdir -p data/processed/tiles_paired
mkdir -p precomputed_stage1

# Download BigEarthNet data
echo "=========================================="
echo "Downloading BigEarthNet S1 data (~51GB)..."
echo "=========================================="
aria2c -x 16 -s 16 \
  "https://zenodo.org/records/10891137/files/BigEarthNet-S1.tar.zst?download=1" \
  -d data/raw -o BigEarthNet-S1.tar.zst

echo "=========================================="
echo "Downloading BigEarthNet S2 data (~68GB)..."
echo "=========================================="
aria2c -x 16 -s 16 \
  "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1" \
  -d data/raw -o BigEarthNet-S2.tar.zst

# Extract data
echo "=========================================="
echo "Extracting S1 data..."
echo "=========================================="
tar -I zstd -xf data/raw/BigEarthNet-S1.tar.zst -C data/raw/
rm data/raw/BigEarthNet-S1.tar.zst

echo "=========================================="
echo "Extracting S2 data..."
echo "=========================================="
tar -I zstd -xf data/raw/BigEarthNet-S2.tar.zst -C data/raw/
rm data/raw/BigEarthNet-S2.tar.zst

# Convert tiles
echo "=========================================="
echo "Converting to paired tiles..."
echo "=========================================="
python scripts/build_tiles.py \
  --s1_dir data/raw/BigEarthNet-S1 \
  --s2_dir data/raw/BigEarthNet-S2 \
  --output_dir data/processed/tiles_paired

# Run Stage 1 precompute
echo "=========================================="
echo "Running Stage 1 precompute (timesteps=3)..."
echo "=========================================="
python scripts/00_precompute_stage1_fast.py \
  --tiles_dir data/processed/tiles_paired \
  --output_dir precomputed_stage1 \
  --timesteps 3

echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Precomputed outputs saved to: precomputed_stage1/"
echo ""
echo "Next steps:"
echo "1. Download precomputed outputs to your local machine"
echo "2. Run Stage 2 training locally or continue on Lambda"
echo "=========================================="
