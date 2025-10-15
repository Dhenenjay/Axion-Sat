# BigEarthNet v2 Streaming Converter Guide

## ✅ Script Created
**Location:** `scripts/ingest_benv2_streaming.py`

## 🚀 Quick Start

### Basic Usage
```powershell
python scripts/ingest_benv2_streaming.py ^
  --s2_root data/raw/BENv2/S2 ^
  --s1_root data/raw/BENv2/S1 ^
  --out_dir data/tiles/benv2_catalog ^
  --target_gb 50 ^
  --min_free_gb 30 ^
  --float16 true
```

## 📋 What It Does

1. **Finds and pairs** S1 and S2 patches by ID
2. **Reads bands**:
   - S2: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
   - S1: VV and VH polarizations
3. **Aligns** S1 to S2 grid (resamples if needed)
4. **Normalizes** data to [0, 1] range
5. **Saves** `.npz` tiles + `.json` metadata
6. **Deletes** original S1/S2 folders after verification
7. **Monitors** disk space and pauses if low
8. **Splits** data into train/val/test (80/10/10)

## 🎯 Command Line Arguments

### Required
- `--s2_root`: Path to S2 data (e.g., `data/raw/BENv2/S2`)
- `--s1_root`: Path to S1 data (e.g., `data/raw/BENv2/S1`)
- `--out_dir`: Output directory (e.g., `data/tiles/benv2_catalog`)

### Optional
- `--target_gb`: Stop after creating this many GB (default: 50)
- `--min_free_gb`: Pause if free space below this (default: 30)
- `--float16`: Use float16 to save space (default: true)
- `--train_ratio`: Train split ratio (default: 0.8)
- `--val_ratio`: Val split ratio (default: 0.1)
- `--countries`: Filter by country (e.g., `keep=DE,FR,IT` or `drop=ES,PT`)
- `--max_per_country`: Max patches per country
- `--seed`: Random seed for splits (default: 42)

## 📊 Output Format

### NPZ Files
Each `.npz` file contains:
```python
{
    's2_b2': (H, W) float16,  # Blue band
    's2_b3': (H, W) float16,  # Green band
    's2_b4': (H, W) float16,  # Red band
    's2_b8': (H, W) float16,  # NIR band
    's1_vv': (H, W) float16,  # VV polarization
    's1_vh': (H, W) float16,  # VH polarization
}
```

### JSON Metadata
Each `.json` file contains:
```json
{
  "id": "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP",
  "country": "DE",
  "split": "train",
  "height": 120,
  "width": 120,
  "date_s2": "20170613",
  "date_s1": "20170610",
  "s2_normalization": {
    "min": 0.0,
    "max": 8000.0,
    "mean": 1500.0,
    "std": 800.0
  },
  "s1_normalization": {
    "min": -25.0,
    "max": 5.0,
    "mean": -10.0,
    "std": 6.0
  }
}
```

## 📝 Logging

Progress is logged to `logs/benv2_ingest.jsonl`:
```json
{"id": "...", "split": "train", "country": "DE", "ok": true, "bytes_npz": 500000, "deleted_s1_mb": 2.5, "deleted_s2_mb": 8.0, "free_gb": 45.2, "status": "ok", "timestamp": "2024-10-14 06:15:30"}
```

## 🔍 Example Use Cases

### 1. Convert 50GB for training
```powershell
python scripts/ingest_benv2_streaming.py ^
  --s2_root data/raw/BENv2/S2 ^
  --s1_root data/raw/BENv2/S1 ^
  --out_dir data/tiles/benv2_catalog ^
  --target_gb 50 ^
  --min_free_gb 30
```

### 2. Only German and French patches
```powershell
python scripts/ingest_benv2_streaming.py ^
  --s2_root data/raw/BENv2/S2 ^
  --s1_root data/raw/BENv2/S1 ^
  --out_dir data/tiles/benv2_catalog ^
  --target_gb 50 ^
  --countries keep=DE,FR
```

### 3. Limited patches per country for balanced dataset
```powershell
python scripts/ingest_benv2_streaming.py ^
  --s2_root data/raw/BENv2/S2 ^
  --s1_root data/raw/BENv2/S1 ^
  --out_dir data/tiles/benv2_catalog ^
  --target_gb 50 ^
  --max_per_country 1000
```

### 4. Custom split ratios (70/15/15)
```powershell
python scripts/ingest_benv2_streaming.py ^
  --s2_root data/raw/BENv2/S2 ^
  --s1_root data/raw/BENv2/S1 ^
  --out_dir data/tiles/benv2_catalog ^
  --target_gb 50 ^
  --train_ratio 0.7 ^
  --val_ratio 0.15
```

## ⚡ Features

### ✅ Streaming Deletion
- Deletes S1/S2 folders immediately after successful tile creation
- Frees up disk space as it goes
- Original data ~110 GB → Tiles ~50 GB

### ✅ Space Monitoring
- Checks free disk space before each patch
- Pauses and waits if space is low
- Clear console warnings

### ✅ Robust Error Handling
- Atomic file writes (`.tmp` → rename)
- Verification after writing
- Skips failed patches and continues
- Detailed error logging

### ✅ Country Stratification
- Automatically extracts country from tile ID
- Splits train/val/test within each country
- Prevents geographic leakage

### ✅ Progress Tracking
- Console progress every 10 patches
- JSON log for every patch
- Final summary with statistics

## 🔧 Dependencies

The script requires:
- `numpy`
- `rasterio`
- `scipy` (for image resampling)

Install with:
```powershell
pip install numpy rasterio scipy
```

## 📈 Expected Performance

- **Processing speed**: ~2-5 seconds per patch
- **Compression ratio**: ~10:1 (110GB → ~11GB for full dataset)
- **Tile size**: ~0.5-2 MB per patch (varies by patch size)
- **Estimated time for 50GB**: ~30-60 minutes

## ⚠️ Important Notes

1. **Irreversible deletion**: Original S1/S2 folders are permanently deleted after successful conversion
2. **Verification**: Each tile is verified before deletion (checks for NaNs, correct shapes)
3. **Windows paths**: Handles long Windows paths safely
4. **Disk space**: Monitor `--min_free_gb` carefully to avoid crashes

## 🐛 Troubleshooting

### Issue: "No matching S1 patches found"
- Check that S1 and S2 folder structures match
- Verify patch ID naming convention

### Issue: "Processing failed" errors
- Check that .tif files exist in patch subdirectories
- Verify band naming (B02, B03, B04, B08 for S2; VV, VH for S1)

### Issue: Script pauses frequently
- Increase `--min_free_gb` if you have more space available
- Or decrease `--target_gb` to create smaller dataset

### Issue: Out of memory
- Use `--float16 true` (default)
- Process fewer patches at once

## 📊 Monitoring Progress

### Check log file
```powershell
Get-Content logs/benv2_ingest.jsonl | Select-Object -Last 10
```

### Count successful conversions
```powershell
(Get-Content logs/benv2_ingest.jsonl | ConvertFrom-Json | Where-Object { $_.ok -eq $true }).Count
```

### Check current output size
```powershell
$size = (Get-ChildItem data/tiles/benv2_catalog -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host "$([math]::Round($size, 2)) GB"
```

## ✅ Ready to Run!

Your folder structure is already set up. Just activate your virtual environment and run:

```powershell
.venv\Scripts\activate
python scripts/ingest_benv2_streaming.py --s2_root data/raw/BENv2/S2 --s1_root data/raw/BENv2/S1 --out_dir data/tiles/benv2_catalog --target_gb 50 --min_free_gb 30
```

Good luck! 🚀
