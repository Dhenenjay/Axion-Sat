# Stage 3 Inference Guide

Quick reference for running Stage 3 inference to produce final SAR-grounded optical outputs.

## Overview

The `infer_stage3.py` script processes folders of Stage 2 outputs (opt_v2) and SAR tiles (S1) to produce final grounded optical outputs (opt_final/opt_v3).

### Pipeline Position

```
Stage 1: S1 → opt_v1 (TerraMind SAR-to-optical)
Stage 2: opt_v1 → opt_v2 (Prithvi refinement)
Stage 3: opt_v2 + S1 → opt_final ✅ (SAR grounding)
```

---

## Quick Start

### Basic Usage

```bash
python scripts/infer_stage3.py \
  --checkpoint checkpoints/stage3/best_model.pt \
  --opt-v2-dir stage2_outputs/ \
  --s1-dir tiles/s1/ \
  --output-dir outputs/stage3_final/
```

### With Custom Settings

```bash
python scripts/infer_stage3.py \
  --checkpoint checkpoints/stage3/best_model.pt \
  --opt-v2-dir stage2_outputs/ \
  --s1-dir tiles/s1/ \
  --output-dir outputs/stage3_final/ \
  --batch-size 4 \
  --timesteps 10 \
  --save-intermediate
```

---

## Input Requirements

### 1. Trained Stage 3 Model

A checkpoint file from `train_stage3.py`:

```
checkpoints/stage3/best_model.pt
```

### 2. Stage 2 Outputs (opt_v2)

NPZ files containing refined optical tiles (4 channels):

```
stage2_outputs/
├── tile_001.npz  # Contains 'opt_v2' or 's2_b2', 's2_b3', 's2_b4', 's2_b8'
├── tile_002.npz
└── ...
```

**Supported formats:**
- `opt_v2`: Direct 4-channel array (4, H, W)
- `s2_b2, s2_b3, s2_b4, s2_b8`: Individual bands
- `opt_v2_b2, opt_v2_b3, opt_v2_b4, opt_v2_b8`: Alternative naming

### 3. SAR Tiles (S1)

NPZ files containing Sentinel-1 SAR data (2 channels):

```
tiles/s1/
├── tile_001.npz  # Contains 's1_vv', 's1_vh' or 's1'
├── tile_002.npz
└── ...
```

**Supported formats:**
- `s1`: Direct 2-channel array (2, H, W)
- `s1_vv, s1_vh`: VV and VH polarizations
- `vv, vh`: Alternative naming

### Tile Matching

The script automatically matches tiles by **filename**. For example:
- `stage2_outputs/tile_001.npz` ↔ `tiles/s1/tile_001.npz`

---

## Output Structure

After inference, the output directory will contain:

```
outputs/stage3_final/
├── tile_001.npz              # Final grounded optical (opt_final)
├── tile_001.json             # Tile statistics (optional)
├── tile_002.npz
├── tile_002.json
├── ...
├── inference_summary.json    # Summary statistics
└── tile_statistics.json      # All tile statistics
```

### Output NPZ Format

Each output file contains:

```python
tile.npz:
  - opt_final: (4, H, W) - Final grounded optical [B, G, R, NIR]
  - opt_v3: (4, H, W) - Alias for opt_final
  
  # If --save-intermediate:
  - opt_v2: (4, H, W) - Stage 2 input
  - s1: (2, H, W) - SAR input
```

### Statistics Files

**`inference_summary.json`**: Overall summary
```json
{
  "total_tiles": 1000,
  "processed": 998,
  "failed": 2,
  "success_rate": 0.998,
  "processing_time_seconds": 245.67,
  "tiles_per_second": 4.06,
  "aggregate_stats": {
    "opt_final_mean": 0.3456,
    "opt_final_std": 0.1234,
    "change_mean": 0.0234,
    "change_max": 0.1567
  }
}
```

**`tile_statistics.json`**: Per-tile statistics
```json
[
  {
    "tile_name": "tile_001.npz",
    "opt_final_mean": 0.3421,
    "opt_final_std": 0.1198,
    "change_mean": 0.0221,
    "change_max": 0.1432,
    "blue_mean": 0.2987,
    "green_mean": 0.3512,
    "red_mean": 0.3698,
    "nir_mean": 0.3587
  }
]
```

---

## Command-Line Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--checkpoint` | Path to trained Stage 3 checkpoint |
| `--opt-v2-dir` | Directory containing opt_v2 tiles |
| `--s1-dir` | Directory containing S1 SAR tiles |
| `--output-dir` | Output directory for opt_final tiles |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | `1` | Batch size for processing |
| `--timesteps` | From checkpoint | Number of diffusion timesteps |
| `--device` | Auto-detect | Device: `cuda` or `cpu` |

### Output Options

| Flag | Description |
|------|-------------|
| `--save-intermediate` | Save opt_v2 and s1 in output files |
| `--no-stats` | Do not save statistics files |
| `--quiet` | Suppress progress output |

---

## Examples

### 1. Basic Inference

Process all matching tiles with default settings:

```bash
python scripts/infer_stage3.py \
  --checkpoint checkpoints/stage3/best_model.pt \
  --opt-v2-dir stage2_outputs/ \
  --s1-dir tiles/s1/ \
  --output-dir outputs/stage3_final/
```

### 2. Batch Processing (Faster)

Process multiple tiles at once:

```bash
python scripts/infer_stage3.py \
  --checkpoint checkpoints/stage3/best_model.pt \
  --opt-v2-dir stage2_outputs/ \
  --s1-dir tiles/s1/ \
  --output-dir outputs/stage3_final/ \
  --batch-size 8
```

### 3. Save Intermediate Inputs

Keep opt_v2 and S1 in output for debugging:

```bash
python scripts/infer_stage3.py \
  --checkpoint checkpoints/stage3/best_model.pt \
  --opt-v2-dir stage2_outputs/ \
  --s1-dir tiles/s1/ \
  --output-dir outputs/stage3_final/ \
  --save-intermediate
```

### 4. Faster Inference (Fewer Timesteps)

Reduce diffusion timesteps for faster processing:

```bash
python scripts/infer_stage3.py \
  --checkpoint checkpoints/stage3/best_model.pt \
  --opt-v2-dir stage2_outputs/ \
  --s1-dir tiles/s1/ \
  --output-dir outputs/stage3_final/ \
  --timesteps 5 \
  --batch-size 4
```

### 5. CPU Inference

Run on CPU (no GPU):

```bash
python scripts/infer_stage3.py \
  --checkpoint checkpoints/stage3/best_model.pt \
  --opt-v2-dir stage2_outputs/ \
  --s1-dir tiles/s1/ \
  --output-dir outputs/stage3_final/ \
  --device cpu
```

### 6. Quiet Mode

Suppress progress bars and verbose output:

```bash
python scripts/infer_stage3.py \
  --checkpoint checkpoints/stage3/best_model.pt \
  --opt-v2-dir stage2_outputs/ \
  --s1-dir tiles/s1/ \
  --output-dir outputs/stage3_final/ \
  --quiet
```

---

## Progress Monitoring

### Console Output

During inference:

```
================================================================================
Stage 3 Inference: SAR Grounding
================================================================================

Configuration:
  Checkpoint: checkpoints/stage3/best_model.pt
  opt_v2 directory: stage2_outputs/
  S1 directory: tiles/s1/
  Output directory: outputs/stage3_final/
  Batch size: 1
  Timesteps: 10
  Device: cuda

Loading Stage 3 model...
✓ Model loaded from checkpoints/stage3/best_model.pt
  Epoch: 28
  Timesteps: 10

Finding tile pairs...
✓ Found 1000 matching tile pairs

Processing tiles: 100%|████████████████| 1000/1000 [04:05<00:00, 4.06it/s]

================================================================================
Inference Complete!
================================================================================

Results:
  Total tiles: 1000
  Processed: 998
  Failed: 2
  Success rate: 99.8%
  Processing time: 245.67s
  Throughput: 4.06 tiles/s

Aggregate Statistics:
  Mean output value: 0.3456
  Mean output std: 0.1234
  Mean change from Stage 2: 0.0234
  Max change from Stage 2: 0.1567

Output saved to: outputs/stage3_final/
  Summary: outputs/stage3_final/inference_summary.json
  Statistics: outputs/stage3_final/tile_statistics.json

⚠ Warning: 2 tiles failed:
  - tile_987.npz
  - tile_988.npz

================================================================================
```

---

## Performance

### Processing Speed

| Configuration | GPU | Throughput | Batch Time |
|---------------|-----|------------|------------|
| Batch=1, T=10 | RTX 3060 | ~4 tiles/s | ~0.25s/tile |
| Batch=4, T=10 | RTX 3060 | ~6 tiles/s | ~0.67s/batch |
| Batch=8, T=10 | RTX 3060 | ~7 tiles/s | ~1.14s/batch |
| Batch=1, T=5 | RTX 3060 | ~7 tiles/s | ~0.14s/tile |

**T**: Number of diffusion timesteps

### Memory Usage

| Configuration | VRAM | RAM |
|---------------|------|-----|
| Batch=1 | ~4 GB | ~2 GB |
| Batch=4 | ~8 GB | ~4 GB |
| Batch=8 | ~12 GB | ~6 GB |

---

## Troubleshooting

### Issue: No matching tiles found

**Error:**
```
ValueError: No matching tile pairs found between stage2_outputs/ and tiles/s1/
```

**Solution:** Ensure tiles have matching filenames:
```bash
# Check file names
ls stage2_outputs/
ls tiles/s1/

# Should have same names: tile_001.npz, tile_002.npz, etc.
```

### Issue: Out of memory (OOM)

**Solution:** Reduce batch size
```bash
python scripts/infer_stage3.py ... --batch-size 1
```

Or reduce timesteps:
```bash
python scripts/infer_stage3.py ... --timesteps 5
```

### Issue: Invalid output values

**Warning:**
```
Invalid output for tile_001.npz: Values out of expected range
```

**Possible causes:**
1. Corrupted checkpoint
2. Mismatched input data (wrong normalization)
3. Model not properly trained

**Solution:** Check input data statistics:
```python
import numpy as np

# Check opt_v2
data = np.load('stage2_outputs/tile_001.npz')
opt_v2 = data['opt_v2']
print(f"opt_v2 range: [{opt_v2.min():.3f}, {opt_v2.max():.3f}]")

# Check S1
data = np.load('tiles/s1/tile_001.npz')
s1 = np.stack([data['s1_vv'], data['s1_vh']])
print(f"S1 range: [{s1.min():.3f}, {s1.max():.3f}]")
```

### Issue: Slow processing

**Solution:** Increase batch size or reduce timesteps
```bash
# Faster processing
python scripts/infer_stage3.py \
  --batch-size 4 \
  --timesteps 5 \
  ...
```

### Issue: Failed tiles

**Warning:**
```
⚠ Warning: 10 tiles failed:
  - tile_123.npz
  - tile_456.npz
```

**Solution:** Check failed tiles manually:
```python
# Inspect failed tile
import numpy as np

data = np.load('stage2_outputs/tile_123.npz')
print("Keys:", list(data.keys()))
print("Shapes:", {k: v.shape for k, v in data.items()})

# Check for NaN or invalid values
for k, v in data.items():
    print(f"{k}: NaN={np.isnan(v).any()}, Inf={np.isinf(v).any()}")
```

---

## Integration with Pipeline

### Full 3-Stage Pipeline

```bash
# Stage 1: SAR → opt_v1
python scripts/infer_stage1.py \
  --checkpoint checkpoints/stage1/best_model.pt \
  --s1-dir tiles/s1/ \
  --output-dir outputs/stage1/

# Stage 2: opt_v1 → opt_v2
python scripts/infer_stage2.py \
  --checkpoint checkpoints/stage2/best_model.pt \
  --opt-v1-dir outputs/stage1/ \
  --output-dir outputs/stage2/

# Stage 3: opt_v2 + S1 → opt_final
python scripts/infer_stage3.py \
  --checkpoint checkpoints/stage3/best_model.pt \
  --opt-v2-dir outputs/stage2/ \
  --s1-dir tiles/s1/ \
  --output-dir outputs/stage3_final/
```

### Using opt_final

The final output `opt_final` can be:
1. **Visualized** with RGB composites
2. **Analyzed** for spectral indices (NDVI, etc.)
3. **Exported** to GeoTIFF for GIS use
4. **Compared** with ground truth for validation

---

## Output Validation

### Automatic Validation

The script automatically validates outputs:
- ✓ No NaN values
- ✓ No Inf values
- ✓ Values within expected range [-10, 10]
- ✓ Correct shape (4, H, W)

### Manual Validation

Check output quality:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load output
data = np.load('outputs/stage3_final/tile_001.npz')
opt_final = data['opt_final']

# RGB visualization
rgb = opt_final[[2, 1, 0], :, :]  # Red, Green, Blue
rgb = np.clip(rgb, 0, 1)
rgb = np.transpose(rgb, (1, 2, 0))

plt.figure(figsize=(10, 10))
plt.imshow(rgb)
plt.title('opt_final RGB Composite')
plt.axis('off')
plt.savefig('opt_final_rgb.png', dpi=150, bbox_inches='tight')
```

---

## Best Practices

1. **Batch Size**: Start with batch=1, increase if GPU memory allows
2. **Timesteps**: Use checkpoint default (usually 10) for best quality
3. **Validation**: Always check `inference_summary.json` for success rate
4. **Storage**: Output files are compressed NPZ (saves ~50% space)
5. **Backup**: Keep original opt_v2 and S1 tiles separate from outputs

---

## Related Scripts

- `train_stage3.py`: Train Stage 3 model
- `evaluate_stage3.py`: Evaluate Stage 3 outputs (coming soon)
- `visualize_stage3.py`: Visualize Stage 3 results (coming soon)

---

## Contact

For issues or questions, please open an issue on the Axion-Sat repository.
