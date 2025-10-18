#!/bin/bash
# Lambda Labs Setup and Training Script
# Run this on your Lambda Labs instance

set -e  # Exit on error

echo "=========================================="
echo "Lambda Labs - Pyramid Decoder Training"
echo "=========================================="
echo ""

# 1. Update system and install dependencies
echo "1/6 Installing dependencies..."
sudo apt-get update -qq
sudo apt-get install -y git wget tmux htop nvtop

# 2. Setup Python environment
echo "2/6 Setting up Python environment..."
python3 -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install hydra-core omegaconf tqdm numpy scikit-learn einops albumentations timm

# 3. Clone/setup codebase
echo "3/6 Setting up codebase..."
if [ ! -d "Axion-Sat" ]; then
    echo "Transferring code from your machine..."
    echo "Run this on your Windows machine:"
    echo "  scp -i C:\Users\Dhenenjay\Downloads\Dhenenjay.pem -r C:\Users\Dhenenjay\Axion-Sat ubuntu@<LAMBDA_IP>:~/"
    echo ""
    echo "Press Enter after transfer is complete..."
    read
fi

cd ~/Axion-Sat/pangaea-bench

# 4. Download TerraMind weights if needed
echo "4/6 Checking TerraMind weights..."
WEIGHTS_PATH="../weights/hf/TerraMind-1.0-large/TerraMind_v1_large.pt"
if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "Downloading TerraMind weights..."
    mkdir -p ../weights/hf/TerraMind-1.0-large
    wget -O $WEIGHTS_PATH https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large/resolve/main/TerraMind_v1_large.pt
fi

# 5. Download HLSBurnScars dataset if needed
echo "5/6 Checking dataset..."
if [ ! -d "./data/hlsburnscars" ]; then
    echo "Dataset will be auto-downloaded during training..."
fi

# 6. Start training in tmux session with live monitoring
echo "6/6 Starting training..."
echo ""
echo "Training will run in tmux session 'pyramid_training'"
echo "To monitor progress:"
echo "  tmux attach -t pyramid_training"
echo ""
echo "To detach from tmux: Ctrl+B, then D"
echo "To see GPU usage: watch -n1 nvidia-smi"
echo ""

# Create tmux session with split panes
tmux new-session -d -s pyramid_training

# Top pane: Training output
tmux send-keys -t pyramid_training "cd ~/Axion-Sat/pangaea-bench" C-m
tmux send-keys -t pyramid_training "python run_windows.py encoder=terramind_optical decoder=seg_pyramid_upernet dataset=hlsburnscars task=segmentation criterion=cross_entropy preprocessing=seg_default lr_scheduler=multi_step_lr use_wandb=False finetune=False train=True work_dir=../results/pangaea_pyramid num_workers=4 test_num_workers=1 | tee training.log" C-m

# Split window horizontally
tmux split-window -h -t pyramid_training

# Bottom pane: Live monitoring
tmux send-keys -t pyramid_training "watch -n2 'nvidia-smi && echo && echo \"=== Latest mIoU ===\" && tail -20 ~/Axion-Sat/pangaea-bench/training.log | grep mIoU'" C-m

# Attach to session
tmux attach -t pyramid_training

echo ""
echo "Training complete! Check results in:"
echo "  ~/Axion-Sat/results/pangaea_pyramid/"
