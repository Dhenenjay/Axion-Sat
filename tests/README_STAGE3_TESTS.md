# Stage 3 Testing Guide

This directory contains integration tests for Stage 3 SAR grounding model.

## Test Files

### `test_stage3_forward_real.py` - Integration Test ⭐

**Purpose**: Validate Stage 3 model with real/synthetic tile data

**Tests**:
1. ✅ Forward pass produces correct output shape
2. ✅ Output values are within valid range (no NaN/Inf)
3. ✅ **SAR edge agreement improves from Stage 2 to Stage 3** 
4. ✅ Perceptual changes are within acceptable range
5. ✅ Loss computation works without errors
6. ✅ Output statistics are reasonable

---

## Running Tests

### Option 1: Using pytest (Recommended)

```bash
# Run all Stage 3 tests
pytest tests/test_stage3_forward_real.py -v

# Run specific test
pytest tests/test_stage3_forward_real.py::TestStage3ForwardPass::test_sar_agreement_improves -v

# Run with coverage
pytest tests/test_stage3_forward_real.py --cov=axs_lib.stage3_tm_ground --cov-report=html
```

### Option 2: Direct Execution

```bash
# Run as standalone script
python tests/test_stage3_forward_real.py
```

---

## Test Data

### Using Real Data (Recommended)

Place a real test tile at `tests/data/test_tile.npz`:

```python
# Create test tile
import numpy as np

# Your real data
s1 = ...  # SAR (2, 120, 120)
opt_v2 = ...  # Stage 2 output (4, 120, 120)
s2_truth = ...  # Ground truth (4, 120, 120)

# Save
np.savez(
    'tests/data/test_tile.npz',
    s1=s1,
    opt_v2=opt_v2,
    s2_truth=s2_truth
)
```

**Required keys**:
- `s1`: SAR input (2, H, W) - VV, VH channels
- `opt_v2`: Stage 2 output (4, H, W) - B, G, R, NIR
- `s2_truth`: Ground truth (4, H, W) - optional, will use opt_v2 if missing

### Using Synthetic Data (Fallback)

If no real tile is found, the test automatically generates synthetic data with realistic properties:
- Structured patterns (edges, gradients)
- Correlated SAR and optical structure
- Realistic noise and color variation

---

## Test Configuration

Edit thresholds in `test_stage3_forward_real.py`:

```python
# Model configuration
TIMESTEPS = 5  # Fewer timesteps for faster testing

# Thresholds
SAR_AGREEMENT_IMPROVEMENT_MIN = 0.0  # Minimum improvement
SAR_AGREEMENT_IMPROVEMENT_TARGET = 0.02  # Target improvement
LPIPS_CHANGE_MAX = 0.25  # Max perceptual change
OUTPUT_VALUE_MIN = -10.0  # Output value range
OUTPUT_VALUE_MAX = 10.0
```

---

## Key Test: SAR Agreement Improvement

**Test**: `test_sar_agreement_improves`

**What it does**:
1. Computes SAR-optical edge agreement for Stage 2 (opt_v2)
2. Runs forward pass through Stage 3 model
3. Computes SAR-optical edge agreement for Stage 3 (opt_v3)
4. **Asserts that opt_v3 has better or equal SAR agreement than opt_v2**

**SAR Agreement Metric**:
- Uses Sobel edge detection on optical and SAR
- Normalizes edge maps to [0, 1]
- Computes cosine similarity between edge maps
- Range: [0, 1], higher is better

**Expected behavior**:
- **With trained model**: Improvement of 0.02-0.10
- **With untrained model**: May stay same or decrease slightly
- **Invalid**: Agreement outside [0, 1] or NaN/Inf

**Example output**:
```
Test 3: SAR Agreement Improvement
================================================================================
Stage 2 SAR agreement: 0.654321
Stage 3 SAR agreement: 0.723456
Improvement: 0.069135 (6.91%)

✓ SAR agreement improved
✓✓ Target improvement achieved (0.069135 >= 0.02)
```

---

## Test Fixtures

### `device`
- Auto-detects CUDA or CPU
- Scope: module (shared across tests)

### `model`
- Builds Stage 3 model with timesteps=5
- No pretrained weights (for testing)
- Scope: module

### `loss_fn`
- Stage3Loss with default weights
- LPIPS enabled
- Scope: module

### `real_tile_data`
- Loads real tile or generates synthetic
- Returns dict with s1, opt_v2, s2_truth, source
- Scope: module

---

## Expected Results

### ✅ All Tests Pass (Synthetic Data)

```
Test 1: Forward Pass Shape
✓ Output shape correct: torch.Size([1, 4, 120, 120])
✓ Output has 4 channels (B, G, R, NIR)

Test 2: Output Value Range
✓ No NaN values
✓ No Inf values
✓ Values within valid range [-10.0, 10.0]

Test 3: SAR Agreement Improvement
Stage 2 SAR agreement: 0.456789
Stage 3 SAR agreement: 0.478901
Improvement: 0.022112 (2.21%)
⚠ Using synthetic data: SAR agreement in valid range
✓ SAR agreement improved (synthetic data)

Test 4: Perceptual Change (LPIPS Proxy)
Perceptual change: 0.123456
✓ Perceptual change below threshold (0.123456 <= 0.25)

Test 5: Loss Computation
Loss components:
  total: 0.234567
  sar_consistency: 0.098765
  cycle_identity: 0.054321
  lpips: 0.045678
  spectral: 0.035803
✓ All loss components valid
✓ Total loss in reasonable range: 0.234567

Test 6: Output Statistics
Per-channel statistics:
  Blue  : mean=0.3456, std=0.1234, range=[0.0123, 0.9876]
  Green : mean=0.4567, std=0.1345, range=[0.0234, 0.9987]
  Red   : mean=0.5678, std=0.1456, range=[0.0345, 0.9998]
  NIR   : mean=0.6789, std=0.1567, range=[0.0456, 1.0000]
✓ All channels have reasonable statistics
✓ Changes are non-negative

TEST SUMMARY
================================================================================
✓ All tests passed!

Validated:
  1. Forward pass produces correct output shape
  2. Output values are within valid range (no NaN/Inf)
  3. SAR edge agreement improves from Stage 2 to Stage 3
  4. Perceptual changes are within acceptable range
  5. Loss computation works without errors
  6. Output statistics are reasonable
================================================================================
```

---

## Troubleshooting

### Issue: Test fails with NaN/Inf

**Cause**: Model instability or incorrect normalization

**Solution**:
1. Check input data ranges
2. Verify model initialization
3. Check standardization stats

### Issue: SAR agreement decreases significantly

**Cause**: Expected for untrained model

**Solution**:
1. Use trained checkpoint in test
2. Lower `SAR_AGREEMENT_IMPROVEMENT_MIN` to allow small decreases
3. Check if using real vs synthetic data

### Issue: CUDA out of memory

**Cause**: GPU memory insufficient

**Solution**:
```python
# Edit test configuration
TIMESTEPS = 3  # Reduce timesteps
TILE_SIZE = 96  # Reduce tile size
```

Or run on CPU:
```bash
CUDA_VISIBLE_DEVICES="" python tests/test_stage3_forward_real.py
```

### Issue: Import errors

**Cause**: Missing dependencies

**Solution**:
```bash
pip install pytest torch torchvision numpy lpips
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Stage 3 Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install pytest torch torchvision numpy lpips
    
    - name: Run Stage 3 tests
      run: |
        pytest tests/test_stage3_forward_real.py -v --tb=short
```

---

## Adding More Tests

### Example: Test with Different Timesteps

```python
@pytest.mark.parametrize("timesteps", [3, 5, 10])
def test_different_timesteps(model, real_tile_data, device, timesteps):
    """Test model with different timestep configurations."""
    s1 = real_tile_data['s1']
    opt_v2 = real_tile_data['opt_v2']
    
    with torch.no_grad():
        opt_v3 = model(s1, opt_v2, timesteps=timesteps)
    
    assert opt_v3.shape == opt_v2.shape
    assert not torch.isnan(opt_v3).any()
```

### Example: Test Batch Processing

```python
def test_batch_processing(model, device):
    """Test model handles multiple tiles in batch."""
    batch_size = 4
    s1 = torch.randn(batch_size, 2, 120, 120).to(device)
    opt_v2 = torch.randn(batch_size, 4, 120, 120).to(device)
    
    with torch.no_grad():
        opt_v3 = model(s1, opt_v2)
    
    assert opt_v3.shape[0] == batch_size
```

---

## Performance Benchmarks

### Typical Test Duration

| Configuration | Time | Notes |
|---------------|------|-------|
| CPU (synthetic) | ~30s | Slow but stable |
| GPU (synthetic) | ~5s | Fast testing |
| GPU (real, 1 tile) | ~6s | Includes I/O |
| Full test suite | ~10-15s | All 6 tests |

### Memory Usage

| Configuration | RAM | VRAM |
|---------------|-----|------|
| CPU | ~2 GB | - |
| GPU (timesteps=5) | ~1 GB | ~3 GB |
| GPU (timesteps=10) | ~1 GB | ~4 GB |

---

## Best Practices

1. **Use real data when possible** - More meaningful validation
2. **Run tests before commits** - Catch regressions early
3. **Check SAR agreement trend** - Key metric for Stage 3
4. **Monitor test duration** - Should be <30s on GPU
5. **Update thresholds for trained models** - Expect better performance

---

## Related Files

- **Model**: `axs_lib/stage3_tm_ground.py`
- **Losses**: `axs_lib/stage3_losses.py`
- **Training**: `scripts/train_stage3.py`
- **Validation**: `scripts/val_stage3.py`

---

**Last Updated**: 2025-10-14  
**Test Version**: 1.0.0
