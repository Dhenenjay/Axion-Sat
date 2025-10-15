# LoRA Unit Tests

## Overview

Comprehensive unit tests for LoRA (Low-Rank Adaptation) implementation in the Axion-Sat project. These tests verify that LoRA correctly reduces trainable parameters while keeping the base model frozen during training.

## Test Coverage

### 1. **Parameter Count Tests** (`TestLoRAParameterCounts`)

#### `test_original_model_params`
- **Purpose**: Verify baseline parameter counting
- **Checks**: 
  - Total parameter count matches expected
  - All parameters are trainable initially
- **Expected**: 131,712 parameters (all trainable)

#### `test_lora_reduces_trainable_params`
- **Purpose**: Verify LoRA dramatically reduces trainable parameters
- **Checks**:
  - LoRA adds correct number of trainable parameters
  - Frozen parameters outnumber trainable parameters
  - Reduction ratio < 10% of original
- **Expected**: 
  - Total: 136,832 params
  - Frozen: 131,712 (96.3%)
  - Trainable: 5,120 (3.7%)
  - **~26x parameter reduction!**

#### `test_parameter_shapes`
- **Purpose**: Verify LoRA matrix dimensions
- **Checks**: Shape of lora_A and lora_B matrices for each layer
- **Expected**:
  - `lora_A`: (rank, in_features)
  - `lora_B`: (out_features, rank)

### 2. **Frozen Parameter Tests** (`TestFrozenParameters`)

#### `test_frozen_params_marked_correctly`
- **Purpose**: Verify requires_grad flags are set correctly
- **Checks**:
  - Base model parameters: `requires_grad = False`
  - LoRA parameters: `requires_grad = True`
- **Critical**: Ensures optimizer won't update frozen weights

#### `test_frozen_params_unchanged_after_training_step`
- **Purpose**: **PRIMARY TEST** - Verify frozen params stay frozen
- **Checks**:
  - Store all frozen parameter values before training
  - Perform forward + backward + optimizer step
  - Compare frozen parameters after training
  - **Parameters must be EXACTLY identical**
- **Why Important**: Core guarantee of LoRA - base model unchanging

#### `test_lora_params_updated_after_training_step`
- **Purpose**: Verify LoRA parameters DO update
- **Checks**:
  - Store LoRA parameter values before training
  - Perform training step
  - Verify at least some LoRA params changed
- **Why Important**: Ensures training is actually happening

#### `test_gradients_only_for_trainable_params`
- **Purpose**: Verify gradient computation
- **Checks**:
  - Trainable params have gradients
  - Frozen params have no gradients (or zero gradients)
- **Why Important**: Prevents unnecessary gradient computation

### 3. **Forward Pass Tests** (`TestLoRAForwardPass`)

#### `test_lora_changes_output`
- **Purpose**: Verify LoRA adaptation affects model output
- **Checks**:
  - Initial output matches original (LoRA B initialized to zeros)
  - After modifying LoRA params, output differs significantly
- **Why Important**: Ensures LoRA actually adapts the model

## Running Tests

### Run All Tests
```bash
python tests/test_lora.py
```

### Run Specific Test Class
```bash
python -m unittest tests.test_lora.TestFrozenParameters
```

### Run Single Test
```bash
python -m unittest tests.test_lora.TestFrozenParameters.test_frozen_params_unchanged_after_training_step
```

## Test Results

```
================================================================================
LoRA Parameter Count and Frozen Parameter Tests
================================================================================
test_lora_reduces_trainable_params ... ok
  Total params:     136,832
  Frozen params:    131,712 (96.3%)
  Trainable params: 5,120 (3.7%)
  Reduction ratio:  3.89% of original

test_frozen_params_unchanged_after_training_step ... ok
  Checked 6 frozen parameters

test_lora_params_updated_after_training_step ... ok
  3/6 LoRA parameters changed

test_gradients_only_for_trainable_params ... ok
  3/6 trainable params have non-zero gradients

--------------------------------------------------------------------------------
Ran 8 tests in 1.221s

OK

✓ All LoRA tests passed!
```

## Key Metrics

| Metric | Value | Significance |
|--------|-------|--------------|
| **Parameter Reduction** | 96.3% frozen | ~26x fewer trainable params |
| **Memory Efficiency** | 3.7% trainable | Enables fine-tuning on low VRAM |
| **Frozen Integrity** | 100% unchanged | Base model preserved |
| **Training Active** | 50%+ params update | LoRA learning confirmed |

## Implementation Details

### LoRA Architecture

```python
# Original linear layer (frozen)
output = W_frozen @ x

# LoRA adaptation
output = W_frozen @ x + (B @ A) @ x * (alpha / rank)

# Where:
#   A: (rank, in_features)   - Down-projection
#   B: (out_features, rank)  - Up-projection
#   rank << min(in_features, out_features)
```

### Parameter Count Formula

For a linear layer with `in_features × out_features`:

**Original parameters**: `in_features × out_features + bias`

**LoRA parameters**: `rank × (in_features + out_features)`

**Reduction factor**: `(in × out) / (rank × (in + out))`

Example (256 × 256 layer, rank=4):
- Original: 65,536 + 256 = 65,792
- LoRA: 4 × (256 + 256) = 2,048
- Reduction: **32x fewer parameters**

## Test Design Philosophy

### 1. **Lightweight & Fast**
- No external dependencies beyond PyTorch
- Simple mock models for testing
- Runs in ~1 second

### 2. **Comprehensive Coverage**
- Parameter counting ✓
- Frozen parameter integrity ✓
- Training dynamics ✓
- Forward pass correctness ✓

### 3. **Clear Assertions**
- Each test has a single, clear purpose
- Descriptive error messages
- Quantitative metrics in output

### 4. **Production-Ready**
- Can be integrated into CI/CD
- Exit code indicates pass/fail
- Verbose output for debugging

## Common Issues Detected

### ❌ Frozen Parameters Modified
```
AssertionError: Frozen parameter fc1.linear.weight changed during training!
```
**Cause**: `requires_grad` not set to False
**Fix**: Ensure `param.requires_grad = False` for all frozen params

### ❌ No LoRA Updates
```
AssertionError: LoRA parameters did not update!
```
**Cause**: LoRA params not in optimizer, or learning rate = 0
**Fix**: Pass only trainable params to optimizer

### ❌ Incorrect Parameter Count
```
AssertionError: Expected 5120 trainable params, got 131712
```
**Cause**: LoRA not actually applied, all params trainable
**Fix**: Verify LoRA layer wrapping is working

## Integration with Stage 2 Training

These tests verify the LoRA implementation that Stage 2 training relies on:

```python
# In Stage 2 training (scripts/train_stage2.py)
from axs_lib.stage2_prithvi_refine import build_prithvi_refiner

# Build model with LoRA
prithvi_refiner = build_prithvi_refiner(config, device)

# Verify only LoRA params are trainable
trainable_params = [p for p in model.parameters() if p.requires_grad]

# These tests ensure:
# 1. Prithvi backbone stays frozen ✓
# 2. Only LoRA adapters train ✓
# 3. Memory footprint is minimal ✓
```

## Future Enhancements

Potential additional tests:

- [ ] LoRA rank sensitivity (rank=1,2,4,8,16)
- [ ] Alpha scaling effects
- [ ] Multi-step training convergence
- [ ] Gradient norm magnitudes
- [ ] Memory profiling
- [ ] Checkpoint save/load with LoRA

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685): "LoRA: Low-Rank Adaptation of Large Language Models"
- [Hugging Face PEFT](https://github.com/huggingface/peft): Reference implementation
- Stage 2 Training: `scripts/train_stage2.py`
- LoRA Implementation: `axs_lib/stage2_prithvi_refine.py`
