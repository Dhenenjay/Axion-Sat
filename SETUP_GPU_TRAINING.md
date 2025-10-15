# Setup Guide: GPU Training for Axion-Sat

**Status**: Training is ready, but GPU and some dependencies need to be configured.

---

## 🚨 CURRENT ISSUES

### 1. **PyTorch CPU-Only Version Detected**
- Current: PyTorch `2.8.0+cpu` (CPU only)
- Required: PyTorch with CUDA support for GPU training

### 2. **Missing GPU/CUDA**
- `nvidia-smi` command not found
- CUDA not available in PyTorch

### 3. **Missing TerraTorch**
- Required for TerraMind model loading
- Installation needed: `pip install terratorch`

---

## 🔧 SETUP STEPS

### Step 1: Check if You Have an NVIDIA GPU

**Option A: Check in Windows Device Manager**
1. Press `Win + X`
2. Select "Device Manager"
3. Expand "Display adapters"
4. Look for NVIDIA GPU (e.g., RTX 3060, GTX 1080, etc.)

**Option B: Try running:**
```powershell
Get-WmiObject Win32_VideoController | Select-Object Name
```

---

### Step 2: Install NVIDIA Drivers and CUDA Toolkit (if GPU exists)

If you have an NVIDIA GPU but `nvidia-smi` doesn't work:

1. **Download NVIDIA Drivers**:
   - Visit: https://www.nvidia.com/Download/index.aspx
   - Select your GPU model
   - Download and install latest driver

2. **Install CUDA Toolkit 12.1** (recommended):
   - Visit: https://developer.nvidia.com/cuda-downloads
   - Download CUDA Toolkit 12.1 for Windows
   - Run installer (this will also install `nvidia-smi`)

3. **Verify Installation**:
   ```powershell
   nvidia-smi
   ```
   Should show GPU information

---

### Step 3: Install PyTorch with CUDA Support

**Uninstall CPU-only PyTorch:**
```powershell
pip uninstall torch torchvision torchaudio
```

**Install CUDA-enabled PyTorch:**

**For CUDA 12.1 (recommended):**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8 (if 12.1 doesn't work):**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify CUDA is available:**
```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

Should print:
```
CUDA available: True
CUDA version: 12.1
```

---

### Step 4: Install TerraTorch

```powershell
pip install terratorch
```

**Alternative (if that fails):**
```powershell
pip install git+https://github.com/IBM/terratorch.git
```

---

### Step 5: Install Optional Dependencies for Better Performance

```powershell
# For better metrics
pip install torchmetrics

# For perceptual loss (LPIPS)
pip install lpips

# For better image processing
pip install timm
```

---

## ✅ VERIFICATION

After completing setup, verify everything works:

```powershell
# Test GPU availability
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Test TerraTorch
python -c "import terratorch; print('TerraTorch imported successfully')"

# Test metrics
python -c "import torchmetrics; print('torchmetrics available')"
python -c "import lpips; print('lpips available')"
```

**Expected output:**
```
CUDA: True
Device: NVIDIA GeForce RTX 3060 (or your GPU model)
TerraTorch imported successfully
torchmetrics available
lpips available
```

---

## 🚀 READY TO TRAIN

Once setup is complete, run training:

### Full Training (199,762 tiles):
```powershell
python .\scripts\train_stage1.py `
  --config .\configs\train\stage1.lowvr.yaml `
  --data-dir .\data\tiles\benv2_catalog `
  --output-dir .\outputs\stage1_benv2 `
  --tile-size 256 `
  --timesteps 12 `
  --batch-size 1 `
  --grad-accum-steps 8
```

### Quick Test (100 tiles, 500 steps):
```powershell
python .\scripts\train_stage1.py `
  --config .\configs\train\stage1.lowvr.yaml `
  --data-dir .\data\tiles\benv2_catalog `
  --output-dir .\outputs\stage1_benv2_test `
  --tile-size 256 `
  --timesteps 12 `
  --batch-size 1 `
  --grad-accum-steps 8 `
  --max-tiles 100 `
  --steps 500
```

### If OOM (Out of Memory):
```powershell
# Reduce tile size and timesteps
python .\scripts\train_stage1.py `
  --config .\configs\train\stage1.lowvr.yaml `
  --data-dir .\data\tiles\benv2_catalog `
  --output-dir .\outputs\stage1_benv2 `
  --tile-size 224 `
  --timesteps 10 `
  --batch-size 1 `
  --grad-accum-steps 8
```

---

## 📊 EXPECTED TRAINING TIMES

**With GPU (RTX 3060 12GB):**
- 500 steps (test): ~15-30 minutes
- 10,000 steps (full): ~5-10 hours

**With CPU (not recommended):**
- 500 steps: ~8-16 hours
- 10,000 steps: ~7-14 days ⚠️

---

## 🔥 IF YOU DON'T HAVE A GPU

### Option 1: Cloud GPU (Recommended)

**Google Colab (Free/Pro):**
- Free: Tesla T4 (limited hours)
- Pro: A100/V100 ($10-12/month)
- Upload your data to Google Drive
- Run training in notebook

**Kaggle (Free):**
- Free: Tesla P100 (30 hours/week)
- Direct notebook access

**Lambda Labs / RunPod / Vast.ai:**
- Rent high-end GPUs by the hour
- $0.20-$2.00/hour depending on GPU

### Option 2: CPU Training (Very Slow)

If you must use CPU:
1. Reduce dataset: `--max-tiles 50`
2. Reduce steps: `--steps 100`
3. Reduce tile size: `--tile-size 128`
4. Reduce timesteps: `--timesteps 6`
5. Disable validation: `--val-every 999999`

```powershell
python .\scripts\train_stage1.py `
  --config .\configs\train\stage1.lowvr.yaml `
  --data-dir .\data\tiles\benv2_catalog `
  --output-dir .\outputs\stage1_benv2_cpu_test `
  --tile-size 128 `
  --timesteps 6 `
  --batch-size 1 `
  --grad-accum-steps 4 `
  --max-tiles 50 `
  --steps 100 `
  --val-every 999999
```

---

## 📝 TROUBLESHOOTING

### "CUDA out of memory" Error:
1. Reduce `--tile-size` (256 → 224 → 192 → 128)
2. Reduce `--timesteps` (12 → 10 → 8 → 6)
3. Close other GPU applications
4. Restart your machine

### "RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED":
1. Reinstall CUDA toolkit
2. Reinstall PyTorch with CUDA
3. Update GPU drivers

### Training is very slow:
1. Verify GPU is being used (check console output)
2. Verify `nvidia-smi` shows GPU usage during training
3. Check if another process is using the GPU

---

## 🎯 SUMMARY

**Before training, you need:**
1. ✅ NVIDIA GPU with drivers (or use cloud GPU)
2. ✅ CUDA Toolkit 12.1
3. ✅ PyTorch with CUDA support
4. ✅ TerraTorch library
5. ✅ Optional: torchmetrics, lpips

**Current status:**
- ❌ GPU/CUDA not available
- ❌ PyTorch is CPU-only version
- ❌ TerraTorch not installed
- ✅ Training code is ready (bug fixed)
- ✅ Dataset is available (199,762 tiles)

**Next action:**
Follow the setup steps above based on your hardware availability.
