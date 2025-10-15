# SAR-Optical Alignment Correction Guide

Complete guide for detecting, diagnosing, and correcting spatial misalignment between SAR and optical imagery in satellite tiles.

## Table of Contents
- [Understanding Misalignment](#understanding-misalignment)
- [Detection Methods](#detection-methods)
- [Correction Approaches](#correction-approaches)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Understanding Misalignment

### What is Misalignment?

Misalignment occurs when SAR and optical imagery from the same geographic location don't align pixel-for-pixel. This manifests as spatial offsets (typically 0.5-5 pixels) between the two data types.

### Common Causes

1. **Geolocation Errors**
   - Satellite ephemeris uncertainties
   - Imprecise metadata timestamps
   - Orbital position errors

2. **Terrain Effects**
   - SAR layover and foreshortening
   - Incomplete terrain correction
   - DEM inaccuracies

3. **Resampling Artifacts**
   - Reprojection to different CRS
   - Interpolation during resampling
   - Grid alignment issues

4. **Acquisition Geometry**
   - Different sensor viewing angles
   - Side-looking SAR vs nadir optical
   - Parallax effects

### Why It Matters

**For ML models**:
- ❌ Models learn spurious correlations
- ❌ Reduced prediction accuracy
- ❌ Poor generalization to new areas

**For analysis**:
- ❌ Incorrect feature extraction
- ❌ Errors in change detection
- ❌ Unreliable multi-sensor fusion

---

## Detection Methods

### 1. Visual Inspection (`show_alignment`)

**When to use**: Quick quality control, visual confirmation

```python
from axs_lib.viz import show_alignment

# Check alignment visually
show_alignment('data/tiles/tile_r00000_c00000.npz')
```

**What to look for**:
- ✅ **Good**: SAR edges align with optical features (roads, buildings, water)
- ❌ **Bad**: SAR edges offset from optical features
- ⚠️ **Uncertain**: Noisy edges, ambiguous features

![Alignment visualization example](../examples/alignment_check_example.png)

### 2. Quantitative Metrics

**When to use**: Batch processing, automated QC

```python
from axs_lib.viz import check_alignment_quality

metrics = check_alignment_quality('tile.npz')
print(f"Edge coverage: {metrics['edge_percent']:.2f}%")
print(f"SAR valid data: {metrics['sar_valid_percent']:.1f}%")
```

**Interpretation**:
- `edge_percent < 1%`: Very few edges (increase sensitivity)
- `edge_percent > 20%`: Too many edges (reduce sensitivity)
- `sar_valid_percent < 80%`: Low SAR coverage (check data quality)

### 3. Cross-Correlation Analysis

**When to use**: Precise offset measurement

```python
from axs_lib.alignment import estimate_offset_correlation

tile = np.load('tile.npz')
offset_y, offset_x, confidence = estimate_offset_correlation(
    tile['s1_vv'],
    tile['s2_b4']
)

print(f"Offset: ({offset_y:.2f}, {offset_x:.2f}) pixels")
print(f"Confidence: {confidence:.3f}")
```

**Confidence levels**:
- `> 0.7`: High confidence, reliable offset
- `0.3 - 0.7`: Medium confidence, visually verify
- `< 0.3`: Low confidence, manual inspection needed

---

## Correction Approaches

### Method 1: Automatic Correction (Recommended)

**Best for**: Most cases, batch processing

```python
from axs_lib.alignment import correct_tile_alignment

# Single tile
result = correct_tile_alignment(
    'tile_r00000_c00000.npz',
    output_path='tile_r00000_c00000_aligned.npz',
    method='correlation',  # or 'edges'
    max_offset=20
)

print(f"Applied offset: {result['offset']}")
print(f"Confidence: {result['confidence']:.3f}")
```

**Parameters**:
- `method='correlation'`: Standard, works for most cases
- `method='edges'`: Use when intensity distributions differ significantly
- `max_offset=20`: Maximum search range (pixels)

### Method 2: Batch Correction

**Best for**: Processing entire datasets

```python
from axs_lib.alignment import correct_tiles_batch

results = correct_tiles_batch(
    tile_dir='data/tiles/nairobi_2024-01-15',
    output_dir='data/tiles/nairobi_2024-01-15_aligned',
    method='correlation',
    min_confidence=0.3  # Skip tiles below this
)

print(f"Corrected {len(results)} tiles")
```

**Workflow**:
1. Processes all `.npz` files in directory
2. Estimates offset for each tile
3. Applies correction to all SAR bands
4. Saves to output directory
5. Reports statistics

### Method 3: Manual Offset

**Best for**: Known offsets, low-confidence cases

```python
from axs_lib.alignment import apply_offset
import numpy as np

tile = np.load('tile.npz')

# Apply known offset
sar_corrected = apply_offset(
    tile['s1_vv'],
    offset=(1.5, -0.8),  # (y, x) in pixels
    order=3  # Cubic interpolation
)

# Save corrected tile
np.savez_compressed(
    'tile_corrected.npz',
    s1_vv=sar_corrected,
    s1_vh=apply_offset(tile['s1_vh'], (1.5, -0.8)),
    **{k: tile[k] for k in tile.files if k.startswith('s2_')}
)
```

### Method 4: Preventive Correction (In Pipeline)

**Best for**: Avoiding issues from the start

Update `scripts/build_tiles.py` to include alignment correction:

```python
# In align_bands() function, after alignment
if 's1_vv' in aligned_arrays and 's2_b4' in aligned_arrays:
    from axs_lib.alignment import correct_alignment
    
    for s1_band in ['s1_vv', 's1_vh']:
        if s1_band in aligned_arrays:
            aligned_arrays[s1_band], _, _ = correct_alignment(
                aligned_arrays[s1_band],
                aligned_arrays['s2_b4'],
                return_offset=True
            )
```

---

## Usage Examples

### Example 1: Complete Workflow

```python
#!/usr/bin/env python3
"""Complete alignment correction workflow"""

from pathlib import Path
from axs_lib.viz import show_alignment
from axs_lib.alignment import correct_tile_alignment, validate_correction

# 1. Visual inspection
print("Step 1: Visual inspection")
show_alignment(
    'tile_original.npz',
    save_path='check_before.png',
    show_plot=False
)

# 2. Automatic correction
print("\nStep 2: Applying correction")
result = correct_tile_alignment(
    'tile_original.npz',
    output_path='tile_corrected.npz'
)

# 3. Validation
print("\nStep 3: Validation")
show_alignment(
    'tile_corrected.npz',
    save_path='check_after.png',
    show_plot=False
)

metrics = validate_correction(
    'tile_original.npz',
    'tile_corrected.npz'
)

print(f"\nImprovement: {metrics['improvement_percent']:.1f}%")
```

### Example 2: Batch with QC

```python
from axs_lib.alignment import correct_tiles_batch
import json

# Correct all tiles
results = correct_tiles_batch(
    'data/tiles/dataset_v1',
    output_dir='data/tiles/dataset_v1_aligned',
    min_confidence=0.4
)

# Analyze results
successful = [r for r in results if 'error' not in r]
low_conf = [r for r in successful if r['confidence'] < 0.5]

print(f"\nSummary:")
print(f"  Total: {len(results)}")
print(f"  Successful: {len(successful)}")
print(f"  Low confidence: {len(low_conf)}")

# Save report
with open('alignment_report.json', 'w') as f:
    json.dump(results, f, indent=2)

# Flag tiles for manual review
if low_conf:
    print(f"\nTiles requiring manual review:")
    for r in low_conf:
        print(f"  - {Path(r['input_path']).name}")
        print(f"    Offset: {r['offset']}, Confidence: {r['confidence']:.3f}")
```

### Example 3: Integration with Training

```python
import pandas as pd
from pathlib import Path

# Load tile index
index = pd.read_csv('data/index/dataset_tiles.csv')

# Update paths to use aligned tiles
index['tile_path_aligned'] = index['tile_path'].apply(
    lambda p: str(Path(p).parent.parent / (Path(p).parent.name + '_aligned') / Path(p).name)
)

# Filter by confidence
alignment_results = pd.read_json('alignment_report.json')
high_conf = alignment_results[alignment_results['confidence'] > 0.5]

# Update index
index_filtered = index[index['tile_id'].isin(
    [Path(r['input_path']).stem for _, r in high_conf.iterrows()]
)]

# Save for training
index_filtered.to_csv('data/index/dataset_tiles_aligned.csv', index=False)
print(f"Training set: {len(index_filtered)} high-quality aligned tiles")
```

---

## Best Practices

### 1. When to Correct

✅ **Do correct**:
- Offsets > 1 pixel
- Visual misalignment obvious
- Training multi-sensor fusion models
- Batch processing large datasets

❌ **Don't correct**:
- Offsets < 0.5 pixels (sub-pixel, negligible)
- Very low confidence (< 0.2)
- Single-sensor models (SAR-only or optical-only)

### 2. Choosing Methods

| Scenario | Method | Confidence Threshold |
|----------|--------|---------------------|
| Urban areas | `correlation` | 0.5+ |
| Vegetated areas | `edges` | 0.3+ |
| Water bodies | `edges` | 0.3+ |
| Mixed terrain | `correlation` | 0.4+ |
| Mountainous | Manual inspection | N/A |

### 3. Quality Control

**Always**:
1. Visual inspection of sample tiles before/after
2. Check confidence scores
3. Validate on holdout tiles
4. Monitor edge alignment visually

**Never**:
- Blindly correct all tiles without inspection
- Use single confidence threshold for all terrain types
- Assume correction always improves alignment

### 4. Interpolation Settings

```python
# High-quality (slower)
apply_offset(data, offset, order=3)  # Cubic

# Standard (balanced)
apply_offset(data, offset, order=1)  # Linear

# Fast (less accurate)
apply_offset(data, offset, order=0)  # Nearest neighbor
```

**Recommendations**:
- Training data: `order=3` (cubic)
- Inference: `order=1` (linear)
- Quick preview: `order=0` (nearest)

---

## Troubleshooting

### Problem 1: Low Confidence Scores

**Symptoms**: All tiles have confidence < 0.3

**Possible causes**:
- Insufficient overlap between SAR and optical
- Very different acquisition times
- Poor data quality (clouds, noise)

**Solutions**:
1. Check data quality with `show_tile_overview()`
2. Try `method='edges'` instead of `'correlation'`
3. Increase `max_offset` search range
4. Filter tiles with low valid data percentage

```python
# Check data coverage
from axs_lib.viz import check_alignment_quality
metrics = check_alignment_quality('tile.npz')

if metrics['sar_valid_percent'] < 50:
    print("Low SAR coverage - skip this tile")
```

### Problem 2: Inconsistent Offsets

**Symptoms**: Large variation in offsets across tiles (> 5 pixels)

**Possible causes**:
- Terrain effects (mountainous regions)
- Poor initial geolocation
- Mixed datasets from different sources

**Solutions**:
1. Process tiles by geographic region
2. Use spatially-varying correction (future feature)
3. Check for consistent offset patterns

```python
# Analyze offset patterns
offsets = [(r['offset'], r['confidence']) for r in results]
import numpy as np

mean_offset = np.mean([o[0] for o in offsets], axis=0)
std_offset = np.std([o[0] for o in offsets], axis=0)

print(f"Mean offset: {mean_offset}")
print(f"Std offset: {std_offset}")

# Flag outliers
for r in results:
    offset_magnitude = np.linalg.norm(r['offset'])
    if offset_magnitude > np.linalg.norm(mean_offset) + 2 * np.linalg.norm(std_offset):
        print(f"Outlier: {r['input_path']}")
```

### Problem 3: Correction Makes It Worse

**Symptoms**: Alignment worse after correction

**Possible causes**:
- False correlation peak
- Ambiguous features
- Insufficient overlap

**Solutions**:
1. Revert to original
2. Try different method (`edges` vs `correlation`)
3. Use manual offset with visual guidance

```python
# Compare before/after
from axs_lib.alignment import validate_correction

metrics = validate_correction('original.npz', 'corrected.npz')

if metrics['improvement_percent'] < 0:
    print("Correction degraded alignment - revert!")
    # Use original tile
```

### Problem 4: Interpolation Artifacts

**Symptoms**: Blocky or blurry SAR after correction

**Solutions**:
- Use higher-order interpolation (`order=3`)
- Check for NaN handling
- Verify data types preserved

```python
# Preserve data quality
corrected = apply_offset(
    data,
    offset,
    order=3,  # Cubic for smoothness
    fill_value=np.nan  # Don't extrapolate
)

# Check statistics
print(f"Original range: [{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")
print(f"Corrected range: [{np.nanmin(corrected):.2f}, {np.nanmax(corrected):.2f}]")
```

---

## Advanced Topics

### Subpixel Accuracy

The correction achieves subpixel accuracy (< 0.1 pixels) using:
1. **Phase correlation**: FFT-based, very precise
2. **Upsampling**: Interpolates correlation peak
3. **Cubic spline**: Smooth subpixel shifts

```python
# Maximum precision
offset_y, offset_x, conf = estimate_offset_correlation(
    sar, optical,
    upsample_factor=100  # Very precise but slow
)
print(f"Subpixel offset: ({offset_y:.3f}, {offset_x:.3f})")
```

### Terrain-Specific Correction

Different terrain types may require different approaches:

```python
def correct_by_terrain(tile_path, terrain_type):
    if terrain_type == 'urban':
        return correct_tile_alignment(tile_path, method='correlation')
    elif terrain_type == 'water':
        return correct_tile_alignment(tile_path, method='edges', max_offset=10)
    elif terrain_type == 'vegetation':
        return correct_tile_alignment(tile_path, method='edges')
    else:
        return correct_tile_alignment(tile_path, method='correlation')
```

### Validation Metrics

Comprehensive validation beyond visual inspection:

```python
from axs_lib.alignment import estimate_offset_correlation
from axs_lib.viz import compute_sar_edges

def comprehensive_validation(original, corrected):
    # Load tiles
    orig = np.load(original)
    corr = np.load(corrected)
    
    # Compute offsets
    orig_offset = estimate_offset_correlation(orig['s1_vv'], orig['s2_b4'])
    corr_offset = estimate_offset_correlation(corr['s1_vv'], corr['s2_b4'])
    
    # Edge overlap metric
    sar_edges = compute_sar_edges(corr['s1_vv'])
    opt_edges = compute_sar_edges(corr['s2_b4'])
    edge_overlap = np.sum(sar_edges & opt_edges) / np.sum(sar_edges | opt_edges)
    
    return {
        'offset_improvement': orig_offset[2] - corr_offset[2],
        'edge_overlap': edge_overlap,
        'final_confidence': corr_offset[2]
    }
```

---

## References

### Scientific Literature

1. **Image Registration Methods**
   - Zitová & Flusser (2003): "Image registration methods: a survey"
   - Ye & Shan (2014): "A survey on L1 regression"

2. **SAR-Optical Fusion**
   - Schmitt & Zhu (2016): "Data fusion and remote sensing"
   - Zhang (2010): "Multi-source remote sensing data fusion: status and trends"

3. **Cross-Correlation Techniques**
   - Lewis (1995): "Fast normalized cross-correlation"
   - Guizar-Sicairos et al. (2008): "Efficient subpixel image registration algorithms"

### Tools and Libraries

- [scikit-image registration](https://scikit-image.org/docs/stable/api/skimage.registration.html)
- [OpenCV phase correlation](https://docs.opencv.org/master/d7/df3/group__imgproc__motion.html)
- [GDAL geolocation](https://gdal.org/programs/gdal_translate.html#cmdoption-gdal_translate-gcp)

---

## Quick Reference

### Command Cheatsheet

```bash
# Visualize alignment
python -c "from axs_lib.viz import show_alignment; show_alignment('tile.npz')"

# Correct single tile
python -c "from axs_lib.alignment import correct_tile_alignment; correct_tile_alignment('tile.npz')"

# Batch correct
python -c "from axs_lib.alignment import correct_tiles_batch; correct_tiles_batch('data/tiles/dataset')"
```

### Python API Quick Reference

```python
# Detection
from axs_lib.viz import show_alignment, check_alignment_quality
show_alignment('tile.npz')
metrics = check_alignment_quality('tile.npz')

# Correction
from axs_lib.alignment import correct_tile_alignment, correct_tiles_batch
result = correct_tile_alignment('tile.npz', output_path='tile_aligned.npz')
results = correct_tiles_batch('data/tiles/dir')

# Validation
from axs_lib.alignment import validate_correction
metrics = validate_correction('original.npz', 'corrected.npz')
```

---

*Last Updated: 2025-10-13*  
*Version: 1.0*  
*Questions? Open an issue on GitHub*
