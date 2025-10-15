# Stage 3: Adaptive Weighting Features

**Version:** 1.0.0  
**Date:** 2025-10-14

---

## Overview

Stage 3 now supports **adaptive spatial weighting** for SAR edge alignment loss, allowing the model to prioritize physically important regions during training. Two new weighting strategies have been implemented:

1. **Urban/High-Backscatter Weighting** (`--urban-weight`)
2. **DEM-Aware Slope Weighting** (`--dem-weight`)

These features enable domain-specific optimization for applications requiring high fidelity in urban areas or mountainous terrain.

---

## 1. Urban/High-Backscatter Weighting

### Motivation

**Urban areas exhibit distinct SAR characteristics:**
- **Double-bounce scattering** from building facades creates high backscatter
- **Corner reflectors** (building-ground junctions) produce strong SAR returns
- **Geometric structures** (roads, buildings) have clear edges in SAR

These properties make SAR particularly reliable for urban mapping. By increasing the loss weight in high-backscatter regions, the model learns to better align optical edges with urban structures visible in SAR.

### Implementation

The urban mask is computed based on SAR backscatter intensity:

```python
def compute_urban_mask(backscatter, threshold=0.7):
    """
    Identify urban/high-backscatter regions.
    
    - Normalizes backscatter to [0, 1]
    - Computes 70th percentile threshold
    - Creates soft mask using sigmoid
    
    Returns:
        Mask in [0, 1] where 1 = high confidence urban
    """
    bs_norm = normalize(backscatter)
    threshold_val = quantile(bs_norm, 0.7)
    urban_mask = sigmoid((bs_norm - threshold_val) * 10.0)
    return urban_mask
```

### Usage

```bash
# Basic usage (no urban weighting)
python scripts/train_stage3.py \
    --data-dir tiles/ \
    --output-dir checkpoints/stage3/ \
    --urban-weight 1.0  # Default: disabled

# Enable urban weighting (2x boost in urban areas)
python scripts/train_stage3.py \
    --data-dir tiles/ \
    --output-dir checkpoints/stage3/ \
    --urban-weight 2.0  # 2x weight for high-backscatter regions

# Strong urban weighting (3x boost)
python scripts/train_stage3.py \
    --data-dir tiles/ \
    --output-dir checkpoints/stage3/ \
    --urban-weight 3.0  # 3x weight - for primarily urban datasets
```

### Weight Application

The spatial weight is computed as:

```python
# Rural areas: weight = 1.0
# Urban areas: weight = urban_weight (e.g., 2.0)
# Transition: smooth interpolation via urban_mask

urban_boost = 1.0 + (urban_weight - 1.0) * urban_mask
spatial_weight = base_weight * urban_boost
```

**Effect on Loss:**

```
urban_weight = 1.0: No change (uniform weighting)
urban_weight = 2.0: Urban regions get 2x weight
urban_weight = 3.0: Urban regions get 3x weight
```

### Recommended Values

| Dataset Type | `--urban-weight` | Use Case |
|--------------|------------------|----------|
| Mixed rural/urban | `1.0` | Default, balanced |
| Suburban | `1.5` | Moderate urban focus |
| Urban-focused | `2.0` - `2.5` | Cities, infrastructure |
| Dense urban | `3.0` | City centers, high-rises |

### Output

Training log will show:

```
Building Loss Function
================================================================================
✓ Stage 3 loss initialized
  SAR weight: 1.0
  Cycle weight: 0.5
  Identity weight: 0.3
  LPIPS weight: 0.1
  Spectral weight: 1.0
  Urban weight: 2.0 ✓ ENABLED     ← Indicates urban weighting active
  DEM weight: 0.0 (disabled)
```

---

## 2. DEM-Aware Slope Weighting

### Motivation

**Terrain slopes are critical for physical consistency:**
- **Steep slopes** represent significant terrain features (mountains, valleys)
- SAR geometry is particularly sensitive to topography
- **Layover and foreshortening** effects in SAR are slope-dependent
- Alignment on steep slopes is crucial for geospatial accuracy

By increasing loss weight on steep slopes, the model prioritizes alignment of prominent terrain features.

### Implementation

Slope is computed from Digital Elevation Model (DEM):

```python
def compute_dem_slope_weight(dem):
    """
    Compute slope-based weighting from DEM.
    
    - Computes spatial gradients (∂z/∂x, ∂z/∂y)
    - Calculates slope magnitude
    - Normalizes and applies sigmoid
    
    Returns:
        Slope weight in [0, 1] where 1 = steep slope
    """
    grad_x = abs(dem[:, :, :, 1:] - dem[:, :, :, :-1])
    grad_y = abs(dem[:, :, 1:, :] - dem[:, :, :-1, :])
    
    slope = sqrt(grad_x^2 + grad_y^2)
    slope_norm = normalize(slope)
    
    # Sigmoid: gentle slopes → 0.3-0.5, steep slopes → 0.8-1.0
    slope_weight = sigmoid((slope_norm - 0.5) * 6.0)
    
    return slope_weight
```

### Usage

**Requirements:**
- DEM data must be provided as NPZ tiles with key `'dem'`
- DEM should be in meters (elevation)
- DEM spatial resolution should match optical/SAR tiles

```bash
# Basic usage (no DEM weighting)
python scripts/train_stage3.py \
    --data-dir tiles/ \
    --output-dir checkpoints/stage3/ \
    --dem-weight 0.0  # Default: disabled

# Enable moderate DEM weighting (30% slope-aware)
python scripts/train_stage3.py \
    --data-dir tiles/ \
    --output-dir checkpoints/stage3/ \
    --dem-weight 0.3  # 30% slope weighting

# Strong DEM weighting (70% slope-aware)
python scripts/train_stage3.py \
    --data-dir tiles/ \
    --output-dir checkpoints/stage3/ \
    --dem-weight 0.7  # Mountainous terrain

# Full DEM weighting (100% slope-based)
python scripts/train_stage3.py \
    --data-dir tiles/ \
    --output-dir checkpoints/stage3/ \
    --dem-weight 1.0  # Extreme terrain
```

### Weight Blending

The DEM weight parameter controls blending between base weight and slope weight:

```python
# dem_weight = 0.0: Use base weight only (no slope influence)
# dem_weight = 1.0: Use slope weight only (fully slope-dependent)
# dem_weight = 0.5: 50/50 blend

spatial_weight = (1 - dem_weight) * base_weight + dem_weight * slope_weight
```

### Recommended Values

| Terrain Type | `--dem-weight` | Use Case |
|--------------|----------------|----------|
| Flat/Plains | `0.0` | Default, no slopes |
| Rolling hills | `0.2` - `0.3` | Gentle terrain |
| Mountainous | `0.5` - `0.7` | Significant slopes |
| Extreme terrain | `0.8` - `1.0` | Alps, Himalayas |

### Data Format

DEM tiles should be structured as:

```python
# tiles/dem/tile_001.npz
{
    'dem': np.ndarray(shape=(1, H, W), dtype=float32)  # Elevation in meters
}
```

**Spatial Alignment:**
- DEM spatial dimensions (H, W) must match optical/SAR tiles
- DEM should be co-registered with optical/SAR data
- Elevation values in meters (typical range: -100 to 8000m)

### Output

Training log will show:

```
Building Loss Function
================================================================================
✓ Stage 3 loss initialized
  SAR weight: 1.0
  Cycle weight: 0.5
  Identity weight: 0.3
  LPIPS weight: 0.1
  Spectral weight: 1.0
  Urban weight: 1.0 (disabled)
  DEM weight: 0.5 ✓ ENABLED     ← Indicates DEM weighting active
```

---

## 3. Combined Weighting

Both urban and DEM weighting can be used simultaneously for datasets with both urban areas and complex terrain:

```bash
python scripts/train_stage3.py \
    --data-dir tiles/ \
    --output-dir checkpoints/stage3/ \
    --urban-weight 2.0 \
    --dem-weight 0.5
```

**Weight Computation Flow:**

```python
# 1. Initialize as ones
spatial_weight = ones_like(edge_diff)

# 2. Apply base backscatter weighting
if adaptive_weighting:
    spatial_weight *= sigmoid(backscatter)

# 3. Apply urban boost
if urban_weight > 1.0:
    urban_mask = compute_urban_mask(backscatter)
    urban_boost = 1.0 + (urban_weight - 1.0) * urban_mask
    spatial_weight *= urban_boost

# 4. Blend with DEM slope weight
if dem_weight > 0.0 and dem is not None:
    slope_weight = compute_dem_slope_weight(dem)
    spatial_weight = (1 - dem_weight) * spatial_weight + dem_weight * slope_weight

# 5. Apply to loss
edge_loss = (edge_diff * spatial_weight).sum() / spatial_weight.sum()
```

---

## 4. Implementation Details

### Location in Codebase

**Modified Files:**

1. **`axs_lib/stage3_losses.py`**
   - Added `compute_urban_mask()` method
   - Added `compute_dem_slope_weight()` method
   - Updated `edge_alignment_loss()` to apply spatial weighting
   - Added `urban_weight` and `dem_weight` parameters to `SARConsistencyLoss`
   - Updated `Stage3Loss.__init__()` to accept new parameters

2. **`scripts/train_stage3.py`**
   - Added `--urban-weight` CLI argument
   - Added `--dem-weight` CLI argument
   - Updated loss instantiation to pass new parameters
   - Added logging for urban/DEM weight status

### Key Methods

#### Urban Mask Computation

```python
def compute_urban_mask(self, backscatter: torch.Tensor, threshold: float = 0.7) -> torch.Tensor:
    """
    Args:
        backscatter: SAR backscatter intensity (B, 1, H, W)
        threshold: Percentile threshold (default: 0.7 = 70th percentile)
    
    Returns:
        Urban mask (B, 1, H, W) in [0, 1]
    """
```

#### DEM Slope Weight Computation

```python
def compute_dem_slope_weight(self, dem: torch.Tensor) -> torch.Tensor:
    """
    Args:
        dem: Digital Elevation Model (B, 1, H, W) in meters
    
    Returns:
        Slope weight (B, 1, H, W) in [0, 1]
    """
```

#### Edge Alignment with Weighting

```python
def edge_alignment_loss(
    self,
    opt: torch.Tensor,
    s1: torch.Tensor,
    sar_features: Dict[str, torch.Tensor],
    dem: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes edge loss with adaptive spatial weighting.
    
    Applies:
    1. Backscatter weighting (base adaptive)
    2. Urban boost (if urban_weight > 1.0)
    3. DEM slope blending (if dem_weight > 0.0 and DEM provided)
    """
```

---

## 5. Validation & Metrics

### Expected Behavior

**With Urban Weighting:**
- SAR agreement should improve more in urban regions
- Buildings and roads should align better with SAR edges
- Rural areas maintain baseline alignment

**With DEM Weighting:**
- SAR agreement on steep slopes should improve
- Mountain ridges and valleys align better
- Flat regions maintain baseline alignment

### Ablation Study

Compare performance with and without adaptive weighting:

```bash
# Baseline (no adaptive weighting)
python scripts/train_stage3.py \
    --data-dir tiles/ \
    --output-dir checkpoints/stage3_baseline/ \
    --urban-weight 1.0 \
    --dem-weight 0.0

# With urban weighting
python scripts/train_stage3.py \
    --data-dir tiles/ \
    --output-dir checkpoints/stage3_urban/ \
    --urban-weight 2.0 \
    --dem-weight 0.0

# With DEM weighting
python scripts/train_stage3.py \
    --data-dir tiles/ \
    --output-dir checkpoints/stage3_dem/ \
    --urban-weight 1.0 \
    --dem-weight 0.5

# With both
python scripts/train_stage3.py \
    --data-dir tiles/ \
    --output-dir checkpoints/stage3_combined/ \
    --urban-weight 2.0 \
    --dem-weight 0.5
```

Then compare using ablation script:

```bash
python scripts/ablate_grounding.py \
    --opt-v2-dir stage2_outputs/ \
    --opt-final-dir checkpoints/stage3_urban/outputs/ \
    --s1-dir tiles/s1/ \
    --n-tiles 50
```

---

## 6. Best Practices

### Urban Weighting

**✓ DO:**
- Use for datasets with significant urban coverage (>20%)
- Start with `--urban-weight 2.0` for mixed datasets
- Increase to 3.0 for dense urban areas
- Monitor urban vs rural SAR agreement separately

**✗ DON'T:**
- Use urban weighting on purely rural datasets
- Set urban_weight < 1.0 (would reduce urban importance)
- Combine with very high SAR weights (can cause instability)

### DEM Weighting

**✓ DO:**
- Ensure DEM is properly co-registered with optical/SAR
- Use for mountainous terrain (elevation range > 500m)
- Start with `--dem-weight 0.3` and increase gradually
- Validate DEM quality before training

**✗ DON'T:**
- Use on flat terrain (slopes < 5 degrees)
- Apply without DEM data available
- Set dem_weight > 0.8 without careful validation
- Use low-resolution DEMs (should match optical/SAR resolution)

### Combined Weighting

**✓ DO:**
- Use for datasets with both urban areas and terrain
- Balance urban_weight and dem_weight (avoid both at maximum)
- Monitor convergence carefully (adaptive weighting can affect training dynamics)
- Validate on held-out test set

**Example Configuration (Mediterranean coast with mountains):**
```bash
--urban-weight 2.0  # Cities along coast
--dem-weight 0.5    # Mountain ranges inland
```

---

## 7. Troubleshooting

### Issue: Training Instability with High Weights

**Symptoms:**
- Loss fluctuates wildly
- NaN losses appear
- SAR agreement decreases

**Solutions:**
```bash
# Reduce weights
--urban-weight 1.5  # Instead of 3.0
--dem-weight 0.3    # Instead of 0.8

# Reduce learning rate
--lr 5e-5  # Instead of 1e-4

# Increase gradient accumulation
--grad-accum 16  # Instead of 8
```

### Issue: No Improvement in Urban Areas

**Symptoms:**
- Urban SAR agreement not improving
- Buildings still misaligned

**Solutions:**
```bash
# Increase urban weight
--urban-weight 3.0  # From 2.0

# Increase overall SAR weight
--sar-weight 1.5  # From 1.0

# Check backscatter normalization
# Verify high-backscatter regions align with urban areas
```

### Issue: DEM Weighting Not Working

**Symptoms:**
- Slope regions not improving
- No difference with/without DEM weight

**Possible Causes:**
- DEM not loaded (check NPZ files have 'dem' key)
- DEM spatial mismatch (H, W dimensions don't match)
- DEM resolution too coarse
- Terrain too flat (slopes < 5 degrees)

**Solutions:**
```bash
# Verify DEM loading
python -c "import numpy as np; d = np.load('tiles/dem/tile_001.npz'); print(d.keys(), d['dem'].shape)"

# Check DEM statistics
python -c "import numpy as np; d = np.load('tiles/dem/tile_001.npz'); dem = d['dem']; print(f'Range: {dem.min():.1f} to {dem.max():.1f}m, Std: {dem.std():.1f}m')"

# Ensure DEM is passed to loss
# (Implementation handles this automatically if 'dem' key exists in NPZ)
```

---

## 8. Summary

### Quick Reference

```bash
# Default (no adaptive weighting)
python scripts/train_stage3.py --data-dir tiles/ --output-dir out/

# Urban-focused
python scripts/train_stage3.py --data-dir tiles/ --output-dir out/ --urban-weight 2.0

# Mountainous terrain
python scripts/train_stage3.py --data-dir tiles/ --output-dir out/ --dem-weight 0.5

# Combined (urban + terrain)
python scripts/train_stage3.py --data-dir tiles/ --output-dir out/ --urban-weight 2.0 --dem-weight 0.5
```

### Parameter Summary

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `--urban-weight` | `1.0` | `1.0` - `3.0` | Boost SAR loss in high-backscatter regions |
| `--dem-weight` | `0.0` | `0.0` - `1.0` | Blend slope-based weighting |

### Key Takeaways

1. **Urban weighting** prioritizes alignment in built-up areas where SAR is most reliable
2. **DEM weighting** prioritizes alignment on terrain features critical for geospatial accuracy
3. Both can be combined for datasets with mixed urban/terrain characteristics
4. Start conservative (urban=2.0, dem=0.3) and adjust based on validation metrics
5. Monitor SAR agreement separately for weighted vs non-weighted regions

---

**Document Version**: 1.0.0  
**Last Updated**: 2025-10-14  
**Maintainer**: Axion-Sat Project Team
