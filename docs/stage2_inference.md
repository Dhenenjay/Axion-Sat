# Stage 2 Inference Guide

## Overview

The `infer_stage2.py` script processes opt_v1 tiles (from Stage 1 TerraMind) through the trained Prithvi refinement model to produce opt_v2 tiles (refined optical features for Stage 3).

## Quick Start

```bash
python scripts/infer_stage2.py \
    --checkpoint outputs/stage2/best.pt \
    --input_dir data/stage1_outputs \
    --output_dir data/stage2_outputs
```

## Input Requirements

### Directory Structure

```
data/stage1_outputs/
├── tile_0001.npz          # Contains 'opt_v1' key (4, H, W)
├── tile_0001.json         # Metadata with 'month' and 'biome_code'
├── tile_0002.npz
├── tile_0002.json
└── ...
```

### opt_v1 Tile Format (NPZ)

```python
{
    'opt_v1': np.ndarray,  # Shape: (4, H, W)
                           # Channels: [B02, B03, B04, B08]
                           # Values: [0, 1] normalized
}
```

### Metadata Format (JSON)

```json
{
    "month": 6,              // 1-12 or "jun", "june", etc.
    "biome_code": 3,         // 0-15 integer
    "tile_id": "tile_0001",
    // ... other metadata preserved
}
```

## Output Format

### opt_v2 Tiles (NPZ)

```python
{
    'opt_v2': np.ndarray,  # Shape: (4, H, W)
                           # Channels: [B02, B03, B04, B08]
                           # Values: [0, 1] refined optical
}
```

### Metadata (JSON)

```json
{
    "month": 6,
    "biome_code": 3,
    "stage": "stage2_refined",
    "timestamp": "2025-10-14T16:20:00",
    "processing": {
        "input_file": "data/stage1_outputs/tile_0001.npz",
        "checkpoint": "outputs/stage2/best.pt",
        "device": "cuda:0",
        "use_amp": true,
        "processing_time_ms": 12.5
    }
}
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | *required* | Path to trained Stage 2 checkpoint (.pt) |
| `--input_dir` | *required* | Directory with opt_v1 tiles |
| `--output_dir` | *required* | Directory to save opt_v2 tiles |
| `--config` | `configs/hardware.lowvr.yaml` | Config file (for model architecture) |
| `--batch_size` | `4` | Batch size for inference |
| `--device` | `cuda` | Device: `cuda` or `cpu` |
| `--no_amp` | `False` | Disable FP16 mixed precision |
| `--no_skip_existing` | `False` | Reprocess existing tiles |

## Usage Examples

### Basic Inference

```bash
python scripts/infer_stage2.py \
    --checkpoint outputs/stage2/best.pt \
    --input_dir data/stage1_outputs \
    --output_dir data/stage2_outputs
```

### Large Batch (High VRAM)

```bash
python scripts/infer_stage2.py \
    --checkpoint outputs/stage2/best.pt \
    --input_dir data/stage1_outputs \
    --output_dir data/stage2_outputs \
    --batch_size 16
```

### CPU Inference

```bash
python scripts/infer_stage2.py \
    --checkpoint outputs/stage2/best.pt \
    --input_dir data/stage1_outputs \
    --output_dir data/stage2_outputs \
    --device cpu \
    --batch_size 1
```

### Disable Mixed Precision (More Stable)

```bash
python scripts/infer_stage2.py \
    --checkpoint outputs/stage2/best.pt \
    --input_dir data/stage1_outputs \
    --output_dir data/stage2_outputs \
    --no_amp
```

### Reprocess All Tiles

```bash
python scripts/infer_stage2.py \
    --checkpoint outputs/stage2/best.pt \
    --input_dir data/stage1_outputs \
    --output_dir data/stage2_outputs \
    --no_skip_existing
```

## Performance

### Expected Throughput

| Hardware | Batch Size | Throughput | Time/1000 tiles |
|----------|------------|------------|-----------------|
| RTX 3060 (12GB) | 4 | ~80 tiles/sec | 12.5 seconds |
| RTX 4060 Ti (16GB) | 8 | ~120 tiles/sec | 8.3 seconds |
| CPU (8 cores) | 1 | ~5 tiles/sec | 3.3 minutes |

### Memory Usage

| Batch Size | GPU Memory | Recommendation |
|------------|------------|----------------|
| 1 | ~2-3 GB | Minimal VRAM systems |
| 4 | ~4-6 GB | Standard (default) |
| 8 | ~8-10 GB | High VRAM |
| 16 | ~12-16 GB | Maximum performance |

## Output Verification

### Check Processing Status

```bash
# Count input tiles
ls data/stage1_outputs/*.npz | wc -l

# Count output tiles
ls data/stage2_outputs/*.npz | wc -l

# Should match if all processed successfully
```

### Inspect Output Tile

```python
import numpy as np
import json

# Load opt_v2 tile
data = np.load('data/stage2_outputs/tile_0001.npz')
opt_v2 = data['opt_v2']

print(f"Shape: {opt_v2.shape}")  # Should be (4, H, W)
print(f"Range: [{opt_v2.min():.3f}, {opt_v2.max():.3f}]")  # Should be [0, 1]
print(f"Mean: {opt_v2.mean():.3f}")
print(f"Channels: [B02, B03, B04, B08]")

# Load metadata
with open('data/stage2_outputs/tile_0001.json') as f:
    meta = json.load(f)
    
print(f"\nMetadata:")
print(f"  Stage: {meta['stage']}")
print(f"  Processing time: {meta['processing']['processing_time_ms']:.1f}ms")
```

### Visual Quality Check

```python
import matplotlib.pyplot as plt
import numpy as np

# Load opt_v1 and opt_v2
opt_v1 = np.load('data/stage1_outputs/tile_0001.npz')['opt_v1']
opt_v2 = np.load('data/stage2_outputs/tile_0001.npz')['opt_v2']

# Create RGB composites (R=B04, G=B03, B=B02)
rgb_v1 = np.stack([opt_v1[2], opt_v1[1], opt_v1[0]], axis=-1)
rgb_v2 = np.stack([opt_v2[2], opt_v2[1], opt_v2[0]], axis=-1)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(np.clip(rgb_v1 * 2, 0, 1))
axes[0].set_title('opt_v1 (Stage 1: TerraMind)')
axes[0].axis('off')

axes[1].imshow(np.clip(rgb_v2 * 2, 0, 1))
axes[1].set_title('opt_v2 (Stage 2: Prithvi Refined)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('stage2_comparison.png', dpi=150)
print("Saved comparison to stage2_comparison.png")
```

## Troubleshooting

### No Tiles Found

**Problem**: `No NPZ tiles found in {input_dir}`

**Solution**:
- Check input directory path
- Ensure tiles have `.npz` extension
- Verify opt_v1 tiles exist from Stage 1

### Missing Metadata

**Problem**: Tiles missing `month` or `biome_code`

**Solution**:
- Script uses defaults: month=6 (June), biome=0 (unknown)
- Add metadata JSON sidecars for better results
- Metadata improves conditioning but is not required

### Out of Memory

**Problem**: `CUDA out of memory`

**Solution**:
1. Reduce batch size: `--batch_size 1` or `--batch_size 2`
2. Use CPU fallback (automatic)
3. Switch to CPU: `--device cpu`

### Slow Processing

**Problem**: Processing very slow (< 5 tiles/sec on GPU)

**Diagnosis**:
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check CPU fallback
grep "CPU fallback" inference.log
```

**Solutions**:
- Check GPU utilization (should be > 80%)
- If CPU fallback occurring, reduce batch size
- Ensure CUDA drivers up to date
- Check no other processes using GPU

### Checkpoint Loading Error

**Problem**: `Error loading checkpoint` or `State dict mismatch`

**Solution**:
- Ensure checkpoint from Stage 2 training (not Stage 1 or 3)
- Check config matches training config
- Verify checkpoint file not corrupted

### Different Tile Sizes

**Problem**: Input tiles have varying sizes

**Solution**:
- Script handles different sizes automatically
- Prithvi interpolates to match input size
- Performance optimal with consistent sizes (120x120 or 256x256)

## Pipeline Integration

### Full Pipeline: Stage 1 → 2 → 3

```bash
# Stage 1: SAR → opt_v1
python scripts/infer_stage1.py \
    --input_dir data/sar_tiles \
    --output_dir data/stage1_outputs

# Stage 2: opt_v1 → opt_v2
python scripts/infer_stage2.py \
    --checkpoint outputs/stage2/best.pt \
    --input_dir data/stage1_outputs \
    --output_dir data/stage2_outputs

# Stage 3: opt_v2 → segmentation
python scripts/infer_stage3.py \
    --checkpoint outputs/stage3/best.pt \
    --input_dir data/stage2_outputs \
    --output_dir data/final_segmentations
```

### Batch Processing Script

```bash
#!/bin/bash
# process_all.sh

# Process all AOIs through Stage 2
for aoi in iowa kenya amazon; do
    echo "Processing $aoi..."
    
    python scripts/infer_stage2.py \
        --checkpoint outputs/stage2/best.pt \
        --input_dir data/stage1_outputs/$aoi \
        --output_dir data/stage2_outputs/$aoi \
        --batch_size 8
    
    echo "Completed $aoi"
done

echo "All AOIs processed!"
```

## Quality Metrics (Optional)

### Compute NDVI Preservation

```python
import numpy as np

def compute_ndvi(optical):
    """Compute NDVI from 4-band optical."""
    red = optical[2]  # B04
    nir = optical[3]  # B08
    return (nir - red) / (nir + red + 1e-8)

# Load tiles
opt_v1 = np.load('data/stage1_outputs/tile_0001.npz')['opt_v1']
opt_v2 = np.load('data/stage2_outputs/tile_0001.npz')['opt_v2']

# Compute NDVI
ndvi_v1 = compute_ndvi(opt_v1)
ndvi_v2 = compute_ndvi(opt_v2)

# NDVI preservation RMSE
ndvi_rmse = np.sqrt(np.mean((ndvi_v1 - ndvi_v2)**2))
print(f"NDVI RMSE: {ndvi_rmse:.4f}")
print(f"Target: < 0.05 (excellent preservation)")
```

### Edge Preservation Check

```python
from scipy import ndimage

def compute_edge_strength(image):
    """Compute edge strength using Sobel."""
    edges_x = ndimage.sobel(image, axis=0)
    edges_y = ndimage.sobel(image, axis=1)
    return np.sqrt(edges_x**2 + edges_y**2)

# Load tiles
opt_v1 = np.load('data/stage1_outputs/tile_0001.npz')['opt_v1']
opt_v2 = np.load('data/stage2_outputs/tile_0001.npz')['opt_v2']

# Compute edge preservation (on NIR band)
edges_v1 = compute_edge_strength(opt_v1[3])
edges_v2 = compute_edge_strength(opt_v2[3])

edge_preservation = np.corrcoef(edges_v1.flatten(), edges_v2.flatten())[0, 1]
print(f"Edge Preservation (correlation): {edge_preservation:.4f}")
print(f"Target: > 0.90 (excellent preservation)")
```

## Best Practices

✓ **DO**:
- Use batch size 4-8 for optimal GPU utilization
- Skip existing tiles with default `--skip_existing`
- Monitor CPU fallback count (should be 0 or low)
- Verify output tile count matches input count
- Check sample outputs visually

✗ **DON'T**:
- Use batch size > 16 (diminishing returns, higher OOM risk)
- Disable AMP unless debugging (slower without benefit)
- Process on CPU unless necessary (10x slower)
- Ignore metadata warnings (impacts quality)
- Skip output verification

## Summary

Stage 2 inference:
- ✅ Processes opt_v1 tiles to opt_v2
- ✅ Maintains spatial structure and spectral relationships
- ✅ ~80 tiles/sec on mid-range GPU
- ✅ Automatic CPU fallback for robustness
- ✅ Resumable (skips existing by default)
- ✅ Progress tracking with ETA

**Next step**: Feed opt_v2 tiles to Stage 3 for final segmentation!

---

*Version: 1.0*  
*Last Updated: 2025-10-14*
