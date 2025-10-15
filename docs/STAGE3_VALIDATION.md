# Stage 3 Validation Guide

Quick reference for validating Stage 3 outputs with SAR agreement and LPIPS metrics.

## Overview

The `val_stage3.py` script validates Stage 3 grounding by computing:
1. **SAR Edge Agreement** ↑ (higher is better)
2. **LPIPS Change** ≤ threshold (lower is better)

## Quick Start

```bash
python scripts/val_stage3.py \
  --opt-v2-dir stage2_outputs/ \
  --opt-final-dir stage3_outputs/ \
  --s1-dir tiles/s1/ \
  --output-dir validation/stage3/
```

## Metrics

### 1. SAR Edge Agreement

**Purpose**: Measures how well optical edges align with SAR edges.

- **Range**: [0, 1]
- **Target**: > 0.7
- **Interpretation**:
  - < 0.5: Poor alignment
  - 0.5 - 0.7: Moderate alignment
  - > 0.7: Good alignment
  - > 0.8: Excellent alignment

**Formula**: Cosine similarity between normalized edge maps

### 2. LPIPS Change

**Purpose**: Measures perceptual distance from Stage 2 to Stage 3.

- **Range**: [0, ∞]
- **Target**: < 0.15
- **Interpretation**:
  - < 0.10: Minimal perceptual change
  - 0.10 - 0.15: Acceptable change
  - > 0.15: Noticeable perceptual change

**Formula**: Learned perceptual similarity using AlexNet features

## Outputs

### 1. CSV File (`validation_metrics.csv`)

Per-tile metrics:

```csv
tile_name,sar_agreement_v2,sar_agreement_final,sar_agreement_improvement,lpips_change,...
tile_001.npz,0.6543,0.7234,0.0691,0.0987,...
tile_002.npz,0.6821,0.7156,0.0335,0.1123,...
```

### 2. Summary JSON (`validation_summary.json`)

Aggregate statistics:

```json
{
  "total_tiles": 1000,
  "validated": 998,
  "sar_agreement": {
    "v2_mean": 0.6543,
    "final_mean": 0.7234,
    "improvement_mean": 0.0691,
    "tiles_improved": 876,
    "tiles_degraded": 122
  },
  "lpips_change": {
    "mean": 0.1123,
    "median": 0.1087,
    "tiles_below_threshold": 834,
    "percent_below_threshold": 83.5
  }
}
```

### 3. Example Visualizations

Generated in `examples/` directory:

- **Best cases**: Largest SAR agreement improvements
- **Worst cases**: Smallest/negative improvements
- **Median case**: Middle of distribution

Each visualization includes:
- Stage 2 RGB composite
- Stage 3 RGB composite
- SAR grayscale
- Difference map
- Edge visualization
- Metrics summary

## Command-Line Arguments

### Required

| Argument | Description |
|----------|-------------|
| `--opt-v2-dir` | Directory with Stage 2 outputs |
| `--opt-final-dir` | Directory with Stage 3 outputs |
| `--s1-dir` | Directory with S1 SAR tiles |
| `--output-dir` | Output directory for results |

### Optional

| Argument | Default | Description |
|----------|---------|-------------|
| `--s2-truth-dir` | None | Ground truth directory (enables PSNR/SSIM) |
| `--num-examples` | 5 | Number of visualization examples |
| `--lpips-threshold` | 0.15 | LPIPS change threshold |
| `--device` | Auto | Device: cuda or cpu |
| `--quiet` | False | Suppress progress output |

## Examples

### Basic Validation

```bash
python scripts/val_stage3.py \
  --opt-v2-dir stage2_outputs/ \
  --opt-final-dir stage3_outputs/ \
  --s1-dir tiles/s1/ \
  --output-dir validation/stage3/
```

### With Ground Truth

```bash
python scripts/val_stage3.py \
  --opt-v2-dir stage2_outputs/ \
  --opt-final-dir stage3_outputs/ \
  --s1-dir tiles/s1/ \
  --s2-truth-dir tiles/ground_truth/ \
  --output-dir validation/stage3/ \
  --num-examples 10
```

### Custom Threshold

```bash
python scripts/val_stage3.py \
  --opt-v2-dir stage2_outputs/ \
  --opt-final-dir stage3_outputs/ \
  --s1-dir tiles/s1/ \
  --output-dir validation/stage3/ \
  --lpips-threshold 0.12
```

## Interpreting Results

### Good Stage 3 Model

✅ **SAR Agreement**:
- Mean improvement > 0.05
- >80% tiles improved
- Final mean > 0.70

✅ **LPIPS Change**:
- Mean < 0.12
- >80% below threshold
- Median < 0.10

### Problematic Model

⚠️ **SAR Agreement**:
- Mean improvement < 0.01
- <50% tiles improved
- Final mean < 0.60

⚠️ **LPIPS Change**:
- Mean > 0.20
- <60% below threshold
- Large variance (std > 0.10)

## Output Structure

```
validation/stage3/
├── validation_metrics.csv       # Per-tile metrics
├── validation_summary.json      # Aggregate statistics
└── examples/                    # Visualizations
    ├── best_1_tile_001.png
    ├── best_2_tile_045.png
    ├── best_3_tile_123.png
    ├── worst_1_tile_678.png
    ├── worst_2_tile_789.png
    └── median_tile_456.png
```

## CSV Columns

| Column | Description |
|--------|-------------|
| `tile_name` | Tile filename |
| `sar_agreement_v2` | Stage 2 SAR agreement |
| `sar_agreement_final` | Stage 3 SAR agreement |
| `sar_agreement_improvement` | Change (final - v2) |
| `lpips_change` | Perceptual distance v2→final |
| `mean_change` | Mean pixel change |
| `max_change` | Max pixel change |
| `blue_mean`, `green_mean`, `red_mean`, `nir_mean` | Channel statistics |

If ground truth provided:
| Column | Description |
|--------|-------------|
| `psnr_v2`, `psnr_final` | PSNR to ground truth |
| `ssim_v2`, `ssim_final` | SSIM to ground truth |
| `lpips_v2_truth`, `lpips_final_truth` | LPIPS to ground truth |

## Visualization Layout

```
┌─────────────────┬─────────────────┬─────────────────┐
│  Stage 2 RGB    │  Stage 3 RGB    │     SAR         │
│  SAR Agr: 0.654 │  SAR Agr: 0.723 │  (Grayscale)    │
├─────────────────┼─────────────────┼─────────────────┤
│ Difference Map  │  Edge Map       │  Metrics Text   │
│  (Hot colormap) │  (Viridis)      │  (Summary)      │
└─────────────────┴─────────────────┴─────────────────┘
```

## Performance

### Validation Speed

- **LPIPS**: ~0.5-1s per tile (GPU)
- **SAR Agreement**: ~0.01s per tile
- **Total**: ~1-2s per tile with visualization

### Memory Usage

- **GPU**: ~2 GB (LPIPS model)
- **RAM**: ~4 GB (tile loading)

## Troubleshooting

### Issue: LPIPS not available

**Warning:**
```
lpips package not available. Install with: pip install lpips
Falling back to VGG feature distance.
```

**Solution:**
```bash
pip install lpips
```

### Issue: No matching tiles

**Error:**
```
ValueError: No matching tiles found
```

**Solution:** Ensure matching filenames across directories:
```bash
ls stage2_outputs/
ls stage3_outputs/
ls tiles/s1/
# Should all have same tile names
```

### Issue: Out of memory

**Solution:** Run on CPU or smaller batch:
```bash
python scripts/val_stage3.py --device cpu ...
```

## Integration with Pipeline

### Full Validation Workflow

```bash
# 1. Run Stage 3 inference
python scripts/infer_stage3.py \
  --checkpoint checkpoints/stage3/best_model.pt \
  --opt-v2-dir stage2_outputs/ \
  --s1-dir tiles/s1/ \
  --output-dir stage3_outputs/

# 2. Validate Stage 3 outputs
python scripts/val_stage3.py \
  --opt-v2-dir stage2_outputs/ \
  --opt-final-dir stage3_outputs/ \
  --s1-dir tiles/s1/ \
  --output-dir validation/stage3/

# 3. Analyze results
cat validation/stage3/validation_summary.json
```

## Best Practices

1. **Always validate on held-out test set**
2. **Check SAR agreement improvement distribution**
3. **Review example visualizations (best/worst)**
4. **Compare with/without ground truth**
5. **Monitor LPIPS threshold pass rate**

## References

- **SAR Edge Agreement**: Custom metric based on Sobel edges + cosine similarity
- **LPIPS**: [Zhang et al., 2018 - The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924)

---

For issues or questions, please open an issue on the Axion-Sat repository.
