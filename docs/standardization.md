# Data Standardization in TerraMind

## Overview

TerraMind was pretrained on large-scale satellite imagery using specific normalization statistics. The `--standardize` flag controls whether these pretraining statistics are applied to your input/output data during training and inference.

## Quick Answer

**Should I use `--standardize`?**

✅ **YES** (default) - Use standardization for best results  
❌ **NO** - Only if you have custom-normalized data or specific requirements

## What is Standardization?

Standardization transforms data to have zero mean and unit variance using precomputed statistics:

```python
standardized_data = (raw_data - mean) / std
```

### TerraMind Pretraining Statistics

**Sentinel-1 (SAR)**:
- VV mean: `-12.619`, std: `5.115`
- VH mean: `-20.229`, std: `5.642`

**Sentinel-2 (Optical)**:
- B02 mean: `0.074`, std: `0.068`
- B03 mean: `0.090`, std: `0.076`
- B04 mean: `0.096`, std: `0.089`
- B08 mean: `0.255`, std: `0.143`

These statistics were computed from **500B tokens** of satellite imagery during TerraMind's pretraining.

## Usage

### Enable Standardization (Default)

```bash
# Explicit flag (same as default)
python scripts/train_stage1.py \
  --data-dir tiles/ \
  --output-dir runs/exp1/ \
  --standardize

# Implicit (default behavior)
python scripts/train_stage1.py \
  --data-dir tiles/ \
  --output-dir runs/exp1/
```

### Disable Standardization

```bash
python scripts/train_stage1.py \
  --data-dir tiles/ \
  --output-dir runs/exp1/ \
  --no-standardize
```

⚠️ **Warning**: Disabling standardization may significantly reduce model quality!

## When to Use Each Mode

### ✅ Use Standardization (--standardize) When:

1. **Using pretrained TerraMind weights** (recommended)
   - Model expects standardized inputs
   - Best transfer learning performance

2. **Training from scratch on new data**
   - Provides stable gradients
   - Prevents numerical issues
   - Faster convergence

3. **Standard workflow** (99% of use cases)
   - Default for good reason
   - Works well across diverse datasets

### ❌ Disable Standardization (--no-standardize) When:

1. **Data already normalized** to TerraMind's scale
   - You've pre-normalized tiles
   - Avoid double-normalization

2. **Custom normalization scheme**
   - Using domain-specific statistics
   - Experimenting with alternative preprocessing

3. **Debugging/analysis**
   - Want to work with raw data values
   - Comparing with non-standardized methods

## Impact on Training

### With Standardization (Recommended)

**Advantages:**
- ✅ Stable gradients
- ✅ Faster convergence
- ✅ Better final quality
- ✅ Consistent with pretrained weights
- ✅ Less sensitive to input scale

**Example output:**
```
Building TerraMind generator (standardize=True)...
✓ Model loaded
Standardization: Enabled
  Using TerraMind pretraining statistics
  S1 mean: [-12.619, -20.229]
  S1 std:  [5.115, 5.642]
  S2 mean: [0.074, 0.090, 0.096, 0.255]
  S2 std:  [0.068, 0.076, 0.089, 0.143]
```

### Without Standardization

**Disadvantages:**
- ❌ May diverge or train slowly
- ❌ Gradient instability possible
- ❌ Pretrained weights less effective
- ❌ Final quality likely reduced

**Example output:**
```
Building TerraMind generator (standardize=False)...
✓ Model loaded
⚠️  WARNING: Standardization disabled!
   Model will receive raw data values
   This may reduce quality if pretrained weights expect standardized inputs
```

## Technical Details

### Data Flow

**With `--standardize`**:
```
Raw Tile → Standardize → Model → Denormalize → Loss → Backprop
   ↓           ↓            ↓          ↓
  S1      (x-μ)/σ     Prediction   Restore   Compare
  S2                   in std       to        with
                       space        original  target
```

**Without `--no-standardize`**:
```
Raw Tile → Model → Loss → Backprop
   ↓         ↓       ↓
  S1    Prediction  Compare
  S2    (raw)       directly
```

### Implementation

The standardization is handled by TerraMind's internal preprocessing:

```python
# In build_terramind_generator()
model = build_terramind_generator(
    input_modalities=("S1GRD",),
    output_modalities=("S2L2A",),
    standardize=True  # or False
)
```

When `standardize=True`:
1. **Input preprocessing**: SAR data standardized using S1 stats
2. **Output preprocessing**: Optical targets standardized using S2 stats
3. **Model training**: Operates in standardized space
4. **Loss computation**: In standardized space (fair comparison)

### Statistics Source

From `axs_lib/stdz.py`:

```python
# Sentinel-1 (SAR) - dB scale
TERRAMIND_S1_STATS = StandardizationStats(
    means=np.array([-12.619, -20.229], dtype=np.float32),  # VV, VH
    stds=np.array([5.115, 5.642], dtype=np.float32)
)

# Sentinel-2 (Optical) - reflectance [0,1]
TERRAMIND_S2_STATS = StandardizationStats(
    means=np.array([0.074, 0.090, 0.096, 0.255], dtype=np.float32),  # B02, B03, B04, B08
    stds=np.array([0.068, 0.076, 0.089, 0.143], dtype=np.float32)
)
```

## Configuration File

In `configs/train/stage1.lowvr.yaml`:

```yaml
model:
  timesteps: 10
  
  # Standardization using TerraMind pretraining statistics
  # IMPORTANT: Keep enabled (true) for best results
  # Only disable if you have custom-normalized data
  standardize: true
```

## Examples

### Example 1: Standard Training (Recommended)

```bash
python scripts/train_stage1.py \
  --data-dir tiles/ \
  --output-dir runs/standard/ \
  --timesteps 10 \
  --steps 45000
# Uses --standardize by default
```

### Example 2: Explicit Standardization

```bash
python scripts/train_stage1.py \
  --data-dir tiles/ \
  --output-dir runs/explicit_std/ \
  --standardize \
  --timesteps 10 \
  --steps 45000
```

### Example 3: Custom Normalization

```bash
# Only if your data is already normalized!
python scripts/train_stage1.py \
  --data-dir prenormalized_tiles/ \
  --output-dir runs/custom_norm/ \
  --no-standardize \
  --timesteps 10 \
  --steps 45000
```

### Example 4: Debug Without Standardization

```bash
# For debugging/analysis only
python scripts/train_stage1.py \
  --data-dir tiles/ \
  --output-dir debug/ \
  --no-standardize \
  --max-tiles 10 \
  --steps 100
```

## Inference

The inference script (`scripts/infer_stage1.py`) should use the **same standardization setting** as training:

```bash
# If trained with --standardize (default)
python scripts/infer_stage1.py \
  --input-dir tiles/ \
  --checkpoint runs/standard/best_model.pt
  # Automatically uses standardization

# If trained with --no-standardize
python scripts/infer_stage1.py \
  --input-dir tiles/ \
  --checkpoint runs/custom_norm/best_model.pt \
  --no-standardize
```

## FAQ

### Q: My model isn't learning, should I disable standardization?

**A:** No! Standardization usually **helps** training. Check instead:
- Learning rate (try 1e-4 or 2e-4)
- Data quality (verify tiles have valid data)
- Batch size / gradient accumulation
- Loss weights

### Q: Can I change standardization mid-training?

**A:** No. The model is trained with specific preprocessing. Changing it would break the learned weights.

### Q: My data is in [0, 1], do I need standardization?

**A:** Yes! TerraMind's statistics transform [0, 1] optical data to ~N(0,1) distribution. This is still beneficial even if data starts in [0, 1].

### Q: How do I know if my data is already standardized?

**A:** Check your tile statistics:
```python
import numpy as np
tile = np.load('tiles/tile001.npz')
s1_vv = tile['s1_vv']

print(f"Mean: {s1_vv.mean():.3f}")
print(f"Std:  {s1_vv.std():.3f}")

# If mean ≈ 0 and std ≈ 1, might be pre-standardized
# If mean ≈ -12.6 and std ≈ 5.1, it's raw S1 VV data
```

### Q: Does inference automatically use the same standardization as training?

**A:** The checkpoint should remember the setting, but it's safest to explicitly specify:
```bash
# Match training configuration
python scripts/infer_stage1.py --checkpoint model.pt --standardize  # or --no-standardize
```

## Troubleshooting

### Problem: Model produces poor results

**Check:**
1. Verify standardization matches training:
   ```python
   ckpt = torch.load('model.pt')
   print(ckpt.get('standardize', 'not recorded'))
   ```

2. Try switching to standard mode:
   ```bash
   python scripts/train_stage1.py --standardize  # Enable explicitly
   ```

### Problem: Gradients exploding/vanishing

**Solution**: Enable standardization (if disabled)
```bash
python scripts/train_stage1.py --standardize  # Provides numerical stability
```

### Problem: "Model expects standardized inputs" warning

**Cause**: Training with `--no-standardize` but using pretrained weights

**Solution**: 
- **Option A**: Use standardization (recommended)
  ```bash
  python scripts/train_stage1.py --standardize
  ```
  
- **Option B**: Train from scratch (not recommended)
  ```bash
  python scripts/train_stage1.py --no-standardize --no-pretrained
  ```

## Performance Impact

### Training Time

Standardization has **negligible** performance impact:
- Preprocessing: < 0.1% overhead
- No impact on forward/backward pass
- Same convergence speed (actually faster!)

### Quality Impact

Based on TerraMind paper and our experiments:

| Configuration | Quality | Convergence |
|--------------|---------|-------------|
| With standardization | **High** ✅ | **Fast** ✅ |
| Without standardization | Lower ❌ | Slower ❌ |

## Best Practices

1. **Always use `--standardize`** unless you have a specific reason not to
2. **Document your choice** in experiment logs
3. **Match train/inference** settings
4. **Check checkpoint metadata** before inference
5. **Verify data statistics** if training fails

## See Also

- `axs_lib/stdz.py` - Standardization statistics
- `axs_lib/models.py` - Model building with standardization
- `docs/reproducibility.md` - Reproducible training
- `docs/early_stopping.md` - Training optimization

## References

- TerraMind Technical Report (IBM/ESA, 2024)
- Batch Normalization Paper (Ioffe & Szegedy, 2015)
- Deep Learning Book Chapter 8 (Goodfellow et al., 2016)
