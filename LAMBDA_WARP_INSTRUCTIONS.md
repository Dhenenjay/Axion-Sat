# Lambda Labs + Warp CLI Setup Instructions

## Why This Approach is Better

✅ **Warp CLI maintains conversation context** - I can continue helping you on Lambda Labs
✅ **No manual SSH juggling** - Warp handles the connection
✅ **Live debugging** - I can see errors and fix them in real-time
✅ **Progress monitoring** - I can check training status and report back
✅ **Code execution** - I can run commands directly on Lambda Labs

---

## Step-by-Step Instructions

### 1. Push Code to GitHub (on Windows)

```powershell
cd C:\Users\Dhenenjay\Axion-Sat

# Add all changes
git add .
git commit -m "Add pyramid decoder for beating benchmark without fine-tuning"

# Push to your repo (replace with your GitHub username)
git push origin main
```

**Note:** Make sure you've set your GitHub remote URL first:
```powershell
git remote -v
# If not set: git remote add origin https://github.com/YOUR_USERNAME/Axion-Sat.git
```

---

### 2. Create Lambda Labs Instance

1. Go to https://cloud.lambdalabs.com/instances
2. Launch instance:
   - **GPU**: 1x A100 (40GB) or RTX 6000 Ada
   - **Region**: Any available
   - **SSH Key**: Upload `Dhenenjay.pem` or use existing
3. **Copy the instance IP address** (e.g., `155.248.XXX.XXX`)

---

### 3. SSH into Lambda Labs and Run Bootstrap

```powershell
# From Windows PowerShell
ssh -i "C:\Users\Dhenenjay\Downloads\Dhenenjay.pem" ubuntu@<LAMBDA_IP>
```

Once connected to Lambda Labs:

```bash
# Download and run bootstrap script
wget https://raw.githubusercontent.com/YOUR_USERNAME/Axion-Sat/main/lambda_bootstrap.sh
chmod +x lambda_bootstrap.sh
./lambda_bootstrap.sh
```

**Or manually paste the bootstrap script if you haven't pushed to GitHub yet.**

---

### 4. Authenticate Warp CLI

After bootstrap completes:

```bash
# Login to Warp
warp auth login
```

This will:
1. Open a browser window
2. Ask you to authenticate with your Warp account
3. Link the Lambda Labs instance to your Warp account

---

### 5. Start Warp Session with This Conversation

```bash
# Initialize Warp in the current directory
cd ~/Axion-Sat
warp init
```

This starts a new Warp Agent Mode session on Lambda Labs.

---

### 6. Continue the Conversation in Warp

In the Warp CLI chat, type:

```
Continue from where we left off. We're now on Lambda Labs.
Run the pyramid decoder training to beat the 83.446% baseline.

The training script is: run_pangaea_pyramid.bat (but adapt it for Linux)
Monitor progress and report metrics as they come in.
```

**I (the AI) will then:**
- ✅ Adapt the Windows batch script to Linux
- ✅ Run the training command
- ✅ Monitor GPU usage and training progress
- ✅ Report epoch-by-epoch results
- ✅ Debug any errors that occur
- ✅ Extract final test mIoU when done

---

## Quick Command Reference

### On Lambda Labs:

```bash
# Check GPU
nvidia-smi

# Monitor training
tail -f ~/Axion-Sat/pangaea-bench/training.log

# Check Python packages
pip list | grep torch

# See running processes
htop

# Disk space
df -h

# Check Warp status
warp status
```

### Reconnecting Later:

```bash
# SSH back in
ssh -i C:\Users\Dhenenjay\Downloads\Dhenenjay.pem ubuntu@<LAMBDA_IP>

# Reattach to Warp session
cd ~/Axion-Sat
warp attach
```

---

## Expected Timeline

- **Setup**: ~10 minutes (dependencies, weights download)
- **Training**: ~2-3 hours on A100
- **Evaluation**: ~10 minutes

**Total**: ~3 hours

---

## What Warp CLI Provides

🔥 **Full conversation context** - I remember everything we discussed
🔥 **Real-time code execution** - I can run and debug commands
🔥 **Live monitoring** - I can check training progress and report back
🔥 **Intelligent debugging** - If errors occur, I can investigate and fix
🔥 **Results reporting** - I'll extract and format final metrics

---

## Fallback: If Warp Doesn't Work

If Warp CLI has issues, you can still use tmux:

```bash
cd ~/Axion-Sat/pangaea-bench

# Start training in tmux
tmux new -s training
python run_windows.py encoder=terramind_optical decoder=seg_pyramid_upernet \
  dataset=hlsburnscars task=segmentation criterion=cross_entropy \
  preprocessing=seg_default lr_scheduler=multi_step_lr use_wandb=False \
  finetune=False train=True work_dir=../results/pangaea_pyramid \
  num_workers=4 test_num_workers=1

# Detach: Ctrl+B then D
# Reattach: tmux attach -s training
```

---

## After Training Completes

Results will be in:
```
~/Axion-Sat/results/pangaea_pyramid/<timestamp>_terramind_optical_seg_pyramid_upernet_hlsburnscars/
```

Check test results:
```bash
cat ~/Axion-Sat/results/pangaea_pyramid/*/test.log-0 | grep "Mean.*mIoU"
```

Compare to baseline:
- **Baseline (UPerNet)**: 83.446% mIoU
- **Target (Pyramid)**: 86-89% mIoU
- **Stretch goal**: >90% mIoU

---

## Benefits of This Approach

| Method | Context | Debug | Monitor | Convenience |
|--------|---------|-------|---------|-------------|
| SSH + Manual | ❌ | ⚠️ | ⚠️ | ⚠️ |
| Warp CLI | ✅ | ✅ | ✅ | ✅ |

**Warp CLI is the best option** - it's like having me sitting next to you on the Lambda Labs instance, watching the training happen and ready to help!
