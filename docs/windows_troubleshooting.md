# Windows Troubleshooting Guide

This document covers common Windows-specific issues encountered when setting up and running the Axion-Sat pipeline, with detailed solutions and workarounds.

---

## Table of Contents

- [GDAL DLL Issues](#gdal-dll-issues)
- [bitsandbytes Compatibility](#bitsandbytes-compatibility)
- [xformers Installation](#xformers-installation)
- [Out of Memory (OOM) Errors](#out-of-memory-oom-errors)
- [Path Length Issues](#path-length-issues)
- [CUDA/PyTorch Issues](#cudapytorch-issues)
- [General Python Environment Issues](#general-python-environment-issues)

---

## GDAL DLL Issues

**Problem**: GDAL is notoriously difficult to install on Windows due to DLL dependencies. Common errors:

```
ImportError: DLL load failed while importing _gdal: The specified module could not be found.
OSError: [WinError 126] The specified module could not be found
ImportError: No module named 'osgeo'
```

### Solution 1: Install via Conda (Recommended)

Conda handles all DLL dependencies automatically.

```powershell
# Create environment with GDAL
conda create -n axion-sat python=3.11 gdal rasterio geopandas -c conda-forge

# Activate environment
conda activate axion-sat

# Install remaining dependencies
pip install -r requirements.txt
```

**Why this works**: Conda packages include pre-compiled binaries and all required DLLs.

### Solution 2: Use OSGeo4W

OSGeo4W provides a complete geospatial stack for Windows.

**Steps**:

1. **Download OSGeo4W installer**:
   - Visit: https://trac.osgeo.org/osgeo4w/
   - Download: `osgeo4w-setup-x86_64.exe`

2. **Install GDAL**:
   ```powershell
   # Run installer, select:
   # - Express Install
   # - GDAL, Python bindings
   ```

3. **Set environment variables**:
   ```powershell
   # Add to PATH (adjust version as needed)
   $env:PATH = "C:\OSGeo4W64\bin;$env:PATH"
   $env:GDAL_DATA = "C:\OSGeo4W64\share\gdal"
   $env:PROJ_LIB = "C:\OSGeo4W64\share\proj"
   
   # Make permanent (PowerShell admin)
   [System.Environment]::SetEnvironmentVariable("PATH", "C:\OSGeo4W64\bin;$env:PATH", "Machine")
   [System.Environment]::SetEnvironmentVariable("GDAL_DATA", "C:\OSGeo4W64\share\gdal", "Machine")
   [System.Environment]::SetEnvironmentVariable("PROJ_LIB", "C:\OSGeo4W64\share\proj", "Machine")
   ```

4. **Install Python bindings**:
   ```powershell
   pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-IC:\OSGeo4W64\include" --global-option="-LC:\OSGeo4W64\lib"
   ```

### Solution 3: Pre-built Wheels from Christoph Gohlke

Unofficial Windows binaries often work when official packages fail.

**Steps**:

1. **Download wheels**:
   - Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/
   - Download matching your Python version (e.g., `cp311` for Python 3.11, `win_amd64` for 64-bit)
   - Files needed: `GDAL`, `rasterio`, `Fiona`

2. **Install wheels**:
   ```powershell
   pip install GDAL-3.8.1-cp311-cp311-win_amd64.whl
   pip install rasterio-1.3.9-cp311-cp311-win_amd64.whl
   pip install Fiona-1.9.5-cp311-cp311-win_amd64.whl
   ```

### Verification

Test if GDAL is working:

```powershell
python -c "from osgeo import gdal; print(f'GDAL version: {gdal.__version__}')"
python -c "import rasterio; print(f'rasterio version: {rasterio.__version__}')"
```

### Common GDAL Environment Variables

```powershell
# Essential
$env:GDAL_DATA = "C:\path\to\gdal\data"
$env:PROJ_LIB = "C:\path\to\proj\share"

# Optional (for debugging)
$env:CPL_DEBUG = "ON"
$env:GDAL_CACHEMAX = "512"  # Cache size in MB
```

---

## bitsandbytes Compatibility

**Problem**: `bitsandbytes` (for 8-bit/4-bit quantization) has limited Windows support. Errors:

```
ImportError: bitsandbytes is not supported on Windows
RuntimeError: No CUDA-compatible bitsandbytes binary found
```

### Background

- `bitsandbytes` is primarily developed for Linux
- Windows support is community-maintained and often lags behind
- 8-bit/4-bit quantization can significantly reduce memory usage

### Solution 1: Use bitsandbytes-windows Fork (Recommended)

Community fork with Windows support:

```powershell
# Install Windows-compatible fork
pip install bitsandbytes-windows
```

**Note**: This fork may not have the latest features from mainline bitsandbytes.

### Solution 2: Compile from Source

For advanced users who need latest features:

```powershell
# Prerequisites
# - Visual Studio 2019+ with C++ tools
# - CUDA Toolkit 11.8+

# Clone and build
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes
python setup.py install
```

### Solution 3: Fallback to FP16/FP32 (Safest)

If `bitsandbytes` installation fails, disable quantization:

**In training config** (`configs/hardware.lowvr.yaml`):

```yaml
training:
  # Use FP16 instead of 8-bit
  precision: fp16
  
  # Disable quantization
  use_8bit_optimizer: false
  use_4bit_quantization: false

model:
  # Load model in FP16
  load_in_8bit: false
  load_in_4bit: false
```

**In code**:

```python
# Check if bitsandbytes is available
try:
    import bitsandbytes as bnb
    USE_8BIT = True
except ImportError:
    print("⚠️  bitsandbytes not available, falling back to FP16")
    USE_8BIT = False

# Conditional optimizer
if USE_8BIT:
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

### Performance Comparison

| Configuration | VRAM Usage | Speed | Compatibility |
|---------------|------------|-------|---------------|
| **8-bit (bitsandbytes)** | ~40% reduction | 90% of FP16 | Linux > Windows |
| **FP16 (mixed precision)** | Baseline | Fast | Excellent |
| **FP32 (full precision)** | 2× FP16 | Slower | Universal |

**Recommendation**: Use FP16 on Windows unless you absolutely need 8-bit quantization.

---

## xformers Installation

**Problem**: `xformers` (memory-efficient attention) lacks official Windows wheels. Errors:

```
ERROR: Could not find a version that satisfies the requirement xformers
No matching distribution found for xformers
```

### Background

- `xformers` provides memory-efficient attention for Transformers
- Can reduce VRAM usage by 15-30%
- Critical for low-VRAM systems

### Solution 1: Pre-built Wheels (Recommended)

Unofficial Windows wheels are available:

```powershell
# For PyTorch 2.1.0 + CUDA 12.1 (adjust as needed)
pip install xformers==0.0.22 --extra-index-url https://download.pytorch.org/whl/cu121

# Alternative: use pre-release wheels
pip install --pre xformers
```

**Find compatible wheel**:
- Visit: https://github.com/facebookresearch/xformers/releases
- Download wheel matching your Python + CUDA version
- Install: `pip install xformers-0.0.22-cp311-cp311-win_amd64.whl`

### Solution 2: Build from Source

**Prerequisites**:
- Visual Studio 2019+ with C++ build tools
- CUDA Toolkit 11.8+
- CMake 3.18+

```powershell
# Install build dependencies
pip install ninja

# Build xformers (takes 30-60 minutes)
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

### Solution 3: Disable xformers (Fallback)

If installation fails, disable memory-efficient attention:

**In config** (`configs/hardware.lowvr.yaml`):

```yaml
memory:
  # Disable xformers
  xformers: false
  
  # Alternative: use PyTorch's native attention
  use_sdpa: true  # Scaled Dot-Product Attention (PyTorch 2.0+)
```

**In code**:

```python
# Check xformers availability
try:
    import xformers
    import xformers.ops
    USE_XFORMERS = True
except ImportError:
    print("⚠️  xformers not available, using default attention")
    USE_XFORMERS = False

# Conditional usage
if USE_XFORMERS:
    model.enable_xformers_memory_efficient_attention()
else:
    # Fall back to PyTorch 2.0 SDPA
    model.enable_sdpa_attention()
```

### Performance Impact

| Attention Type | VRAM Usage | Speed | Windows Support |
|----------------|------------|-------|-----------------|
| **xformers** | Lowest | Fast | Limited (wheels) |
| **SDPA (PyTorch 2.0+)** | Medium | Fast | Excellent |
| **Default (vanilla)** | Highest | Baseline | Universal |

**Recommendation**: Try `xformers` wheels first, fall back to SDPA if unavailable.

---

## Out of Memory (OOM) Errors

**Problem**: CUDA out of memory errors during training or inference:

```
RuntimeError: CUDA out of memory. Tried to allocate X GB (GPU 0; Y GB total capacity)
torch.cuda.OutOfMemoryError: CUDA out of memory
```

### Quick Diagnosis

Check current GPU usage:

```powershell
# PowerShell
nvidia-smi

# Or use watch for live updates (Windows Terminal with WSL)
watch -n 1 nvidia-smi
```

### Recipe: Progressive Memory Reduction

Try these steps in order until OOM is resolved:

#### **Level 1: Quick Wins (Minimal Impact on Quality)**

```yaml
# configs/hardware.lowvr.yaml
training:
  batch_size: 1
  grad_accum_steps: 8  # Effective batch = 8
  precision: fp16      # Use FP16 instead of FP32

memory:
  xformers: true
  channels_last: true
```

**Expected savings**: ~20-30% VRAM

#### **Level 2: Tile Size Reduction**

```yaml
data:
  tile_size: 384  # Reduce from 512
  # Alternative: 320, 256

stage1:
  model:
    tm_s1:
      timesteps: 12  # Reduce from 50
    tm_s2:
      timesteps: 12  # Reduce from 50

stage3:
  model:
    conditional:
      timesteps: 10  # Reduce from 50
```

**Expected savings**: ~40-50% VRAM

**Impact**: Smaller tiles = more processing overhead, slightly degraded quality at tile boundaries

#### **Level 3: Model Optimization**

```yaml
training:
  gradient_checkpointing: true  # Trade compute for memory

stage2:
  model:
    use_lora: true
    lora_r: 4  # Reduce from 8 or 16
    freeze_base: true

memory:
  cpu_offload: true  # Offload optimizer to CPU
```

**Expected savings**: ~30-40% VRAM

**Impact**: Slower training, slight quality reduction with lower LoRA rank

#### **Level 4: Aggressive Reduction (Last Resort)**

```yaml
data:
  tile_size: 256      # Very small tiles
  num_workers: 1      # Reduce data loader memory

training:
  batch_size: 1
  grad_accum_steps: 16  # Effective batch still reasonable

stage1:
  model:
    tm_s1:
      timesteps: 8    # Minimal timesteps
      embed_dim: 128  # Reduce from 256
    tm_s2:
      timesteps: 8
      embed_dim: 128

stage3:
  model:
    conditional:
      timesteps: 6
      embed_dim: 128

inference:
  batch_size: 1
```

**Expected savings**: ~60-70% VRAM

**Impact**: Noticeable quality degradation, much slower training

### Memory Optimization Checklist

```markdown
- [ ] Use FP16 mixed precision
- [ ] Enable gradient checkpointing
- [ ] Reduce batch size to 1
- [ ] Increase gradient accumulation steps
- [ ] Enable xformers (if available)
- [ ] Reduce tile size (512 → 384 → 256)
- [ ] Reduce timesteps (50 → 12 → 8)
- [ ] Use LoRA instead of full fine-tuning
- [ ] Reduce LoRA rank (16 → 8 → 4)
- [ ] Enable CPU offloading
- [ ] Reduce embedding dimensions (256 → 128)
- [ ] Clear CUDA cache between operations: `torch.cuda.empty_cache()`
- [ ] Close unnecessary applications
- [ ] Restart Python kernel/session
```

### Monitoring Memory Usage

**In Python**:

```python
import torch

def print_gpu_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {allocated:.2f} GB allocated / {reserved:.2f} GB reserved / {total:.2f} GB total")

# Call before/after operations
print_gpu_usage()
model.train()
print_gpu_usage()
```

**Use CUDA profiler**:

```python
with torch.cuda.profiler.profile():
    output = model(input)
```

---

## Path Length Issues

**Problem**: Windows has a 260-character path limit (MAX_PATH). Long paths cause:

```
FileNotFoundError: [Errno 2] No such file or directory
OSError: [WinError 206] The filename or extension is too long
```

### Solution 1: Enable Long Path Support (Windows 10+)

**Via Registry** (Requires Admin):

```powershell
# Run PowerShell as Administrator
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1

# Restart computer
Restart-Computer
```

**Via Group Policy**:
1. Open `gpedit.msc`
2. Navigate to: `Computer Configuration > Administrative Templates > System > Filesystem`
3. Enable: "Enable Win32 long paths"
4. Restart

### Solution 2: Use Shorter Paths

```powershell
# Move project closer to drive root
# Bad:  C:\Users\YourName\Documents\Projects\ML\Satellite\Axion-Sat
# Good: C:\Projects\Axion-Sat

# Use symbolic links
mklink /J C:\AS C:\Users\YourName\Long\Path\To\Axion-Sat
cd C:\AS
```

### Solution 3: Check Paths Proactively

```powershell
# Use our path checker tool
python tools/win_path_check.py --fix
```

See [Path Length Checker documentation](../tools/win_path_check.py) for details.

---

## CUDA/PyTorch Issues

### Problem: PyTorch Not Using GPU

**Symptoms**:
```python
>>> torch.cuda.is_available()
False
```

**Solution**:

```powershell
# Verify CUDA is installed
nvidia-smi

# Reinstall PyTorch with correct CUDA version
# For CUDA 12.1
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Problem: CUDA Version Mismatch

**Error**:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution**: Ensure PyTorch CUDA version matches installed CUDA Toolkit.

```powershell
# Check installed CUDA
nvidia-smi  # Look for "CUDA Version: X.Y"

# Install matching PyTorch
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Problem: Multiple CUDA Versions

**Solution**: Use the newest compatible version.

```powershell
# List CUDA installations
ls "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"

# Set PATH to use specific version (PowerShell)
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin;$env:PATH"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
```

---

## General Python Environment Issues

### Problem: Package Conflicts

**Symptoms**: Dependency resolver errors, import failures after updates.

**Solution**: Use a clean virtual environment.

```powershell
# Create fresh environment
python -m venv .venv --clear

# Activate
.\.venv\Scripts\Activate.ps1

# Install from pinned versions
pip install -r pip-freeze.txt
```

### Problem: pip SSL Errors

**Error**:
```
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solution**:

```powershell
# Update certificates
pip install --upgrade certifi

# Or use trusted host (temporary workaround)
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org <package>
```

### Problem: Script Execution Policy

**Error**:
```
cannot be loaded because running scripts is disabled on this system
```

**Solution**:

```powershell
# Check current policy
Get-ExecutionPolicy

# Set policy (run as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Quick Reference: Memory Reduction Hierarchy

From most to least effective for VRAM savings:

1. **Tile Size**: `512 → 384` (40% savings)
2. **Batch Size**: `4 → 1` (75% savings)
3. **Gradient Checkpointing**: Enabled (30% savings)
4. **Timesteps**: `50 → 12` (25% savings)
5. **LoRA**: Full fine-tune → LoRA (20% savings)
6. **CPU Offload**: Enabled (20% savings)
7. **xformers**: Enabled (15% savings)
8. **Precision**: FP32 → FP16 (50% savings)
9. **Embedding Dim**: `256 → 128` (30% savings)

**Optimal Low-VRAM Recipe** (for 8 GB VRAM):

```yaml
data:
  tile_size: 384

training:
  batch_size: 1
  grad_accum_steps: 8
  precision: fp16
  gradient_checkpointing: true

stage1:
  model:
    tm_s1:
      timesteps: 12
    tm_s2:
      timesteps: 12

stage2:
  model:
    use_lora: true
    lora_r: 8

stage3:
  model:
    conditional:
      timesteps: 10

memory:
  cpu_offload: true
  xformers: true
```

---

## Getting Help

If you encounter issues not covered here:

1. **Run diagnostics**:
   ```powershell
   python tests/test_imports.py
   python tools/win_path_check.py
   .\scripts\04_disk_budget.ps1 -Verbose
   ```

2. **Check logs**:
   ```powershell
   # Training logs
   cat logs/training.log
   
   # System info
   systeminfo
   nvidia-smi
   ```

3. **Create issue**:
   - Include output from diagnostic scripts
   - Attach error logs
   - Specify Windows version, Python version, GPU model

4. **Community resources**:
   - Project GitHub Issues
   - PyTorch Forums (for CUDA issues)
   - Stack Overflow (tag: `windows`, `pytorch`, `gdal`)

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-13  
**Tested On**: Windows 10/11, Python 3.11, CUDA 12.1  
**Maintainer**: Axion-Sat Development Team