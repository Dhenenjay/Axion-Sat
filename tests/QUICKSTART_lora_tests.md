# LoRA Tests - Quick Start

## Run Tests

```bash
# Run all tests
python tests/test_lora.py

# Run with verbose output
python -m unittest tests.test_lora -v

# Run specific test class
python -m unittest tests.test_lora.TestFrozenParameters

# Run single test
python -m unittest tests.test_lora.TestFrozenParameters.test_frozen_params_unchanged_after_training_step
```

## What Gets Tested

### ✅ Parameter Counts
- **131,712** parameters in base model
- **5,120** trainable with LoRA (3.7%)
- **26x reduction** in trainable parameters

### ✅ Frozen Parameters
- Base model weights **NEVER change**
- Only LoRA adapters update
- Verified with exact tensor comparison

### ✅ Training Dynamics
- LoRA parameters update correctly
- Gradients computed only for trainable params
- Forward pass produces expected outputs

## Test Results

```
Ran 8 tests in 1.221s

OK

✓ All LoRA tests passed!
```

## Why This Matters

**LoRA enables training Prithvi-EO-2.0-600M on low VRAM:**
- Full model: ~600M parameters to train
- With LoRA: ~2-5M parameters to train (1%)
- **100x+ memory savings**
- Base model knowledge preserved

## Common Test Failures

### Frozen params changed
```python
AssertionError: Frozen parameter fc1.linear.weight changed during training!
```
**Fix**: Ensure `param.requires_grad = False` for frozen params

### No LoRA updates
```python
AssertionError: LoRA parameters did not update!
```
**Fix**: Pass only trainable params to optimizer:
```python
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.001
)
```

## Integration

These tests validate the LoRA used in:
- **Stage 2 Training**: `scripts/train_stage2.py`
- **Prithvi Refiner**: `axs_lib/stage2_prithvi_refine.py`

## Questions?

See full documentation: `tests/README_lora_tests.md`
