# Downloading Models from HuggingFace

This guide covers how to download the required foundation models (TerraMind and Prithvi) from HuggingFace for the Axion-Sat pipeline.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [TerraMind 1.0 Large](#terramind-10-large)
- [Prithvi EO 2.0 600M](#prithvi-eo-20-600m)
- [License Information](#license-information)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Install Git LFS

Git Large File Storage (LFS) is required for downloading model weights.

```bash
git lfs install
```

**Verification**:
```bash
git lfs version
# Output: git-lfs/3.x.x (GitHub; windows amd64; go 1.x.x)
```

### Install HuggingFace CLI (Optional)

For the `huggingface-cli` download method:

```bash
pip install huggingface-hub
```

---

## TerraMind 1.0 Large

**Model**: `ibm-esa-geospatial/TerraMind-1.0-large`  
**Repository**: https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large  
**License**: Apache-2.0

### Method 1: Full Clone (Recommended)

Downloads all model files including weights:

```bash
git clone https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large
```

**Notes**:
- Downloads ~10+ GB of model weights
- Requires Git LFS to be installed
- Model files stored in `./TerraMind-1.0-large/`

### Method 2: Pointer-Only Clone

Downloads repository structure without large files (pointers only):

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large
```

**Notes**:
- Fast download (metadata only)
- Model weights are **not** downloaded
- Useful for inspecting model card, configs, and file structure
- Pull specific files later with: `git lfs pull --include="*.safetensors"`

**Windows PowerShell equivalent**:
```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
```

### Method 3: HuggingFace CLI

Downloads model files without git repository:

```bash
huggingface-cli download ibm-esa-geospatial/TerraMind-1.0-large
```

**Notes**:
- Downloads to HuggingFace cache directory
  - Linux/Mac: `~/.cache/huggingface/hub/`
  - Windows: `%USERPROFILE%\.cache\huggingface\hub\`
- No `.git` directory (saves space)
- Automatically resumes interrupted downloads

**Download to specific directory**:
```bash
huggingface-cli download ibm-esa-geospatial/TerraMind-1.0-large --local-dir ./weights/terramind
```

**Download specific files only**:
```bash
huggingface-cli download ibm-esa-geospatial/TerraMind-1.0-large --include "*.safetensors" --local-dir ./weights/terramind
```

---

## Prithvi EO 2.0 600M

**Model**: `ibm-nasa-geospatial/Prithvi-EO-2.0-600M`  
**Repository**: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M  
**License**: Apache-2.0

### Method 1: Full Clone (Recommended)

Downloads all model files including weights:

```bash
git clone https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M
```

**Notes**:
- Downloads ~2-3 GB of model weights
- Requires Git LFS to be installed
- Model files stored in `./Prithvi-EO-2.0-600M/`

### Method 2: Pointer-Only Clone

Downloads repository structure without large files (pointers only):

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M
```

**Notes**:
- Fast download (metadata only)
- Model weights are **not** downloaded
- Useful for inspecting model card, configs, and file structure
- Pull specific files later with: `git lfs pull --include="*.safetensors"`

**Windows PowerShell equivalent**:
```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
```

### Method 3: HuggingFace CLI

Downloads model files without git repository:

```bash
huggingface-cli download ibm-nasa-geospatial/Prithvi-EO-2.0-600M
```

**Notes**:
- Downloads to HuggingFace cache directory
- No `.git` directory (saves space)
- Automatically resumes interrupted downloads

**Download to specific directory**:
```bash
huggingface-cli download ibm-nasa-geospatial/Prithvi-EO-2.0-600M --local-dir ./weights/prithvi
```

**Download specific files only**:
```bash
huggingface-cli download ibm-nasa-geospatial/Prithvi-EO-2.0-600M --include "*.safetensors" --local-dir ./weights/prithvi
```

---

## License Information

Both models are released under the **Apache License 2.0**:

### TerraMind 1.0 Large
- **License**: Apache-2.0
- **HuggingFace**: https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large
- **Citation**: arXiv:2504.11171
- **Developers**: IBM, ESA, Forschungszentrum Jülich

### Prithvi EO 2.0 600M
- **License**: Apache-2.0
- **HuggingFace**: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M
- **Developers**: IBM, NASA

### Apache License 2.0 Summary

✅ **Permissions**:
- Commercial use
- Modification
- Distribution
- Patent use
- Private use

❌ **Limitations**:
- Trademark use
- Liability
- Warranty

📋 **Conditions**:
- License and copyright notice
- State changes
- Disclosure of modifications

For full license text, see: https://www.apache.org/licenses/LICENSE-2.0

---

## Troubleshooting

### Git LFS Not Installed

**Error**:
```
Error downloading object: [...]: Smudge error: Error downloading [...]
```

**Solution**:
```bash
git lfs install
# Re-clone or pull files
git lfs pull
```

### Slow Download Speed

**Problem**: Git clone is slow for large models.

**Solution 1**: Use HuggingFace CLI with multi-threading:
```bash
huggingface-cli download ibm-esa-geospatial/TerraMind-1.0-large --resume-download
```

**Solution 2**: Use pointer-only clone, then pull files:
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large
cd TerraMind-1.0-large
git lfs pull --include="*.safetensors"
```

### Disk Space Issues

**Problem**: Not enough space for full model.

**Solution**: Download specific files only:
```bash
# Clone without weights
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large
cd TerraMind-1.0-large

# Pull only specific checkpoint
git lfs pull --include="model.safetensors"
```

### Authentication Required

Some models may require HuggingFace authentication.

**Login**:
```bash
huggingface-cli login
# Enter your HuggingFace token
```

**Get token**: https://huggingface.co/settings/tokens

### Windows Path Length Issues

**Problem**: Paths exceed 260 characters.

**Solution**: Enable long path support or use shorter directory names.

```powershell
# Enable long paths (requires admin)
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
Restart-Computer

# Or clone to shorter path
cd C:\
git clone https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large TM
```

---

## Recommended Workflow

For Axion-Sat pipeline, we recommend:

### 1. Create weights directory
```bash
mkdir weights
cd weights
```

### 2. Download TerraMind
```bash
# Using git clone
git clone https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large terramind

# OR using HuggingFace CLI
huggingface-cli download ibm-esa-geospatial/TerraMind-1.0-large --local-dir terramind
```

### 3. Download Prithvi
```bash
# Using git clone
git clone https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M prithvi

# OR using HuggingFace CLI
huggingface-cli download ibm-nasa-geospatial/Prithvi-EO-2.0-600M --local-dir prithvi
```

### 4. Verify downloads
```bash
# Check directory structure
ls -R weights/

# Expected structure:
# weights/
# ├── terramind/
# │   ├── model.safetensors
# │   ├── config.json
# │   └── ...
# └── prithvi/
#     ├── model.safetensors
#     ├── config.json
#     └── ...
```

### 5. Update config
Edit `configs/hardware.lowvr.yaml`:

```yaml
stage1:
  model:
    name: terramind_s1_s2
    checkpoint: ./weights/terramind/model.safetensors

stage2:
  model:
    name: prithvi_refiner
    checkpoint: ./weights/prithvi/model.safetensors
```

---

## Additional Resources

- **TerraMind Paper**: https://arxiv.org/abs/2504.11171
- **Prithvi Model Card**: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M
- **HuggingFace Documentation**: https://huggingface.co/docs/huggingface_hub
- **Git LFS Documentation**: https://git-lfs.com/

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-13  
**Models Covered**: TerraMind 1.0 Large, Prithvi EO 2.0 600M

<citations>
  <document>
      <document_type>WEB_PAGE</document_type>
      <document_id>https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large</document_id>
  </document>
</citations>