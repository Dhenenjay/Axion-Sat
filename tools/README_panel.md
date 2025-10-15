# Panel Visualization Tool - Quick Reference

## Purpose

Generate side-by-side comparison panels showing:
1. **Synthetic optical (v1)** - Generated from SAR
2. **Ground truth S2** - Real Sentinel-2 imagery
3. **v1 + SAR edges** - Synthetic with edge overlay

## Quick Start

```bash
# 1. Run inference first
python scripts/infer_stage1.py \
  --input-dir tiles/ \
  --checkpoint checkpoints/stage1/best_model.pt

# 2. Generate comparison panels
python tools/panel_v1_vs_truth.py \
  --tile-dir tiles/ \
  --output-dir panels/
```

## Common Commands

### Process all tiles
```bash
python tools/panel_v1_vs_truth.py --tile-dir tiles/ --output-dir panels/
```

### Process specific tiles
```bash
python tools/panel_v1_vs_truth.py \
  --tile-dir tiles/ \
  --output-dir panels/ \
  --tile-ids tile001 tile002
```

### High-quality output for papers
```bash
python tools/panel_v1_vs_truth.py \
  --tile-dir tiles/ \
  --output-dir figures/ \
  --dpi 300 \
  --format pdf
```

### Adjust edge sensitivity
```bash
# More edges (lower threshold = more sensitive)
python tools/panel_v1_vs_truth.py --tile-dir tiles/ --output-dir panels/ --edge-threshold 0.05

# Fewer edges (higher threshold = less sensitive)
python tools/panel_v1_vs_truth.py --tile-dir tiles/ --output-dir panels/ --edge-threshold 0.15
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tile-dir` | *(required)* | Directory with tile NPZ files |
| `--output-dir` | *(required)* | Where to save panel images |
| `--tile-ids` | *all tiles* | Specific tiles to process |
| `--opt-v1-suffix` | `_opt_v1` | Suffix for synthetic optical files |
| `--edge-threshold` | `0.1` | SAR edge detection threshold (0.0-1.0) |
| `--figsize` | `18 6` | Figure size in inches (width height) |
| `--dpi` | `100` | Output resolution |
| `--format` | `png` | Output format (png, jpg, pdf, svg) |

## Edge Threshold Guide

| Scene Type | Recommended Threshold |
|------------|----------------------|
| Urban (many buildings) | 0.10 - 0.15 |
| Rural/Agricultural | 0.05 - 0.10 |
| Forest | 0.08 - 0.12 |
| Coastal | 0.10 - 0.15 |
| Water-dominated | 0.15 - 0.20 |

## Output

- **Filename pattern**: `{tile_id}_panel.{format}`
- **Example**: `tile001_panel.png`
- **Layout**: 3 panels side-by-side (synthetic | truth | synthetic+edges)

## Prerequisites

Before running, ensure:
1. ✅ Tile NPZ files exist in `--tile-dir`
2. ✅ Synthetic optical (`*_opt_v1.npz`) files exist
3. ✅ Output directory is writable

## Troubleshooting

### "opt_v1 not found"
→ Run inference first: `python scripts/infer_stage1.py --input-dir tiles/ --checkpoint ...`

### No edges visible
→ Lower threshold: `--edge-threshold 0.05`

### Blurry panels
→ Increase DPI: `--dpi 200` or `--dpi 300`

### Out of memory
→ Process fewer tiles at once: `--tile-ids tile001 tile002`

## Full Documentation

See `docs/panel_visualization.md` for:
- Detailed parameter descriptions
- Technical implementation details
- Advanced usage examples
- Integration with training workflow
