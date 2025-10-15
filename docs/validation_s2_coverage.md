# Validation S2 Coverage Warning

## Overview

The training script now monitors and warns when validation tiles are missing S2 (optical) truth data. This helps identify potential issues with validation quality while maintaining SAR-edge validation as a reliable fallback.

## Key Features

### 1. **Automatic Detection**
- Checks each validation batch for valid S2 truth data
- Considers S2 data missing if:
  - All values are near zero (< 1e-6)
  - Contains NaN values

### 2. **Warning Threshold**
- Triggers warning if **>50%** of validation batches lack S2 truth
- Displayed during validation runs
- Does not stop training

### 3. **SAR-Edge Validation**
- **Always available** even without S2 truth
- Uses SAR structure consistency as validation metric
- Reliable fallback for cloud-covered regions

## Example Output

### Normal Case (Good Coverage)
```
[Step 500] Running validation...
Val Loss: 0.2345
  (95/100 batches with S2 truth)
  L1: 0.1234, MS-SSIM: 0.8765
  SAR-Edge Agreement: 0.0456
```

### Warning Case (Low Coverage)
```
[Step 500] Running validation...

⚠️  WARNING: 65.0% of validation batches are missing S2 truth!
   (65/100 batches without S2 data)
   Validation will rely primarily on SAR-edge agreement.
   Consider adding more tiles with optical coverage.

Val Loss: 0.3456
  (35/100 batches with S2 truth)
  L1: 0.1876, MS-SSIM: 0.8234
  SAR-Edge Agreement: 0.0523
```

## Why This Matters

### Impact on Validation

#### With Sufficient S2 Coverage (>50%)
✅ **Full validation metrics available:**
- L1 loss (pixel accuracy)
- MS-SSIM (structural similarity)
- LPIPS (perceptual quality)
- SAR-edge agreement (geometric consistency)

#### With Low S2 Coverage (<50%)
⚠️ **Limited validation metrics:**
- L1, MS-SSIM, LPIPS: Less reliable (fewer samples)
- SAR-edge agreement: ✅ Still fully valid
- Model selection: Relies more on SAR-edge metric

### Why SAR-Edge Works Without S2 Truth

The SAR-edge agreement metric:
- Compares edge structures between input SAR and generated optical
- Does not require ground truth optical data
- Validates geometric consistency
- Detects hallucinations and artifacts

**Formula:**
```python
sar_edges = sobel_filter(sar_input)
optical_edges = sobel_filter(generated_optical)
agreement = 1 - abs(sar_edges - optical_edges).mean()
```

## Common Scenarios

### Scenario 1: Cloud-Covered Validation Region
**Problem:** Validation tiles from cloudy period

**Solution:**
```bash
# Option A: Use different validation period
python scripts/train_stage1.py \
  --data-dir tiles_clear_season/ \
  --val-split 0.1

# Option B: Rely on SAR-edge validation
# Training proceeds normally, early stopping uses SAR-edge metric
```

### Scenario 2: Sparse Dataset
**Problem:** Limited optical coverage overall

**Impact:**
- Warning appears but training continues
- SAR-edge agreement becomes primary validation metric
- Model quality may be harder to assess

**Recommendation:**
- Collect more tiles with optical coverage
- Use SAR-edge agreement for model selection
- Validate final model on external test set with full coverage

### Scenario 3: Intentional Design
**Problem:** None - you're training on SAR-only regions

**Action:**
- Warning is expected and can be ignored
- Trust SAR-edge agreement metric
- Consider disabling other metrics (set weights to 0)

## Configuration

### Adjusting Warning Threshold

The 50% threshold is hardcoded but can be modified in `scripts/train_stage1.py`:

```python
# In validate() function
if s2_missing_rate > 0.5:  # Change this threshold
    print(f"\n⚠️  WARNING: {s2_missing_rate*100:.1f}% of validation batches...")
```

### Disabling Non-SAR Metrics for Low Coverage

If S2 coverage is known to be low, reduce reliance on optical-based metrics:

```bash
python scripts/train_stage1.py \
  --data-dir tiles/ \
  --ms-ssim-weight 0.0 \
  --lpips-weight 0.0 \
  --sar-structure-weight 1.0  # Increase SAR-edge importance
```

## Batch-Level vs Tile-Level

### Current Implementation: Batch-Level
- Checks entire batch for S2 validity
- Fast and efficient
- Batch considered "without S2" if any sample is missing

### Alternative: Tile-Level (Not Implemented)
Could track individual tiles:
```python
# Per-tile tracking (future enhancement)
tiles_with_s2 = sum(has_valid_s2(tile) for tile in batch)
tiles_without_s2 = batch_size - tiles_with_s2
```

## Metrics Interpretation with Low Coverage

### When S2 Coverage < 50%

**Reliable Metrics:**
- ✅ **SAR-edge agreement**: Fully reliable, no S2 truth needed
- ✅ **Training loss**: Still valid for samples with S2

**Less Reliable Metrics:**
- ⚠️ **Validation L1/MS-SSIM/LPIPS**: Based on smaller sample
- ⚠️ **Best model selection**: May not generalize to unseen data

**Recommendations:**
1. Use SAR-edge agreement for early stopping
2. Validate final model on separate test set with full coverage
3. Check generated samples visually
4. Compare with baseline methods

## Integration with Early Stopping

The early stopping mechanism uses SAR-edge agreement by default:

```python
# In train_stage1.py
early_stopping_obj = EarlyStopping(
    patience=5,
    min_delta=1e-4,
    mode='min',  # Lower SAR-edge agreement is better
    verbose=True
)

# Triggered on SAR-edge plateau
if early_stopping(sar_edge_agreement, step):
    # Training stops
```

**This works regardless of S2 coverage!** ✅

## Technical Details

### S2 Validity Check
```python
# Check if S2 truth is available
has_s2_truth = (
    torch.abs(s2_target).max() > 1e-6  # Not all zeros
    and not torch.isnan(s2_target).any()  # No NaNs
)
```

### Batch Counting
```python
# Track batches during validation
if has_s2_truth:
    num_batches_with_s2 += 1
else:
    num_batches_without_s2 += 1

# Compute missing rate
s2_missing_rate = num_batches_without_s2 / num_batches
```

### Warning Logic
```python
# Warn if more than 50% missing
if s2_missing_rate > 0.5:
    print(f"\n⚠️  WARNING: {s2_missing_rate*100:.1f}% of validation batches...")
    print(f"   Validation will rely primarily on SAR-edge agreement.")
```

## Comparison with Other Approaches

### Approach 1: Skip Validation Without S2 ❌
**Problem:** Loses all validation capability

### Approach 2: Use SAR-Only Metrics ⚠️
**Problem:** May miss optical quality issues

### Approach 3: Warn + Keep SAR-Edge ✅ (Our Approach)
**Benefits:**
- User is informed about coverage issues
- SAR-edge validation always works
- Training continues normally
- Model selection remains possible

## FAQ

### Q: What if 100% of validation batches lack S2 truth?

**A:** Training continues, but you'll see:
```
⚠️  WARNING: 100.0% of validation batches are missing S2 truth!
   (100/100 batches without S2 data)
   Validation will rely primarily on SAR-edge agreement.
```

L1/MS-SSIM/LPIPS will be computed but less meaningful. SAR-edge agreement remains fully valid.

### Q: Should I increase validation split to get more S2 coverage?

**A:** Only if your dataset has spatial/temporal variations in coverage. If the entire dataset has low coverage, increasing val_split won't help.

### Q: Can I disable the warning?

**A:** Yes, comment out the warning in `validate()` function, but this is not recommended - the warning provides valuable information.

### Q: Does this affect training?

**A:** No. Training uses the full training set and is not affected. Only validation metrics are impacted.

### Q: How do I know if my model is good with low S2 coverage?

**A:** 
1. Check SAR-edge agreement (should be low/improving)
2. Visually inspect generated samples
3. Test on external validation set with full coverage
4. Compare with baseline methods

## Best Practices

1. ✅ **Aim for >50% S2 coverage** in validation set
2. ✅ **Trust SAR-edge agreement** as primary metric when S2 is limited
3. ✅ **Monitor the warning** - if it appears, investigate your data
4. ✅ **Use external test set** for final evaluation
5. ✅ **Keep early stopping enabled** - works with SAR-edge only

## See Also

- `docs/early_stopping.md` - Early stopping using SAR-edge agreement
- `axs_lib/losses.py` - Loss functions including SAR-structure
- `scripts/train_stage1.py` - Main training script
- `docs/validation.md` - General validation guide
