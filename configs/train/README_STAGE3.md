# Stage 3 Training Configuration Guide

This directory contains training configurations for Stage 3 SAR grounding with TerraMind.

## Available Configurations

### `stage3.lowvr.yaml` - Low VRAM Setup ⭐

**Purpose**: Train Stage 3 on low VRAM GPUs (6-8 GB)

**Key Features**:
- LoRA fine-tuning (only cross-attention trainable)
- Mixed precision (FP16)
- Gradient accumulation (effective batch = 8)
- Frozen base model (except LoRA adapters)

**Expected VRAM**: 5-6 GB  
**Training Time**: 2-3 hours (RTX 3060)  
**Steps**: 15k-30k

---

## Using the Configuration

### Basic Usage

```bash
python scripts/train_stage3.py \
  --config configs/train/stage3.lowvr.yaml \
  --data-dir tiles/ \
  --output-dir checkpoints/stage3/
```

### Override Parameters

```bash
# Override learning rate
python scripts/train_stage3.py \
  --config configs/train/stage3.lowvr.yaml \
  --lr 5e-5

# Override LoRA rank
python scripts/train_stage3.py \
  --config configs/train/stage3.lowvr.yaml \
  --lora-rank 16

# Override max steps
python scripts/train_stage3.py \
  --config configs/train/stage3.lowvr.yaml \
  --max-steps 20000
```

---

## Configuration Structure

### Model Configuration

```yaml
model:
  timesteps: 10              # Diffusion timesteps
  pretrained: true           # Use pretrained TerraMind
  
  lora:
    enabled: true
    rank: 8                  # LoRA rank (4, 8, 16)
    alpha: 16                # Scaling factor
    freeze_base_model: true  # ✅ FREEZE EVERYTHING EXCEPT LoRA
```

### Training Configuration

```yaml
training:
  batch_size: 1              # Per-GPU batch size
  gradient_accumulation_steps: 8  # Effective batch = 8
  
  max_steps: 30000           # Training duration
  min_steps: 15000           # Minimum before early stop
  
  mixed_precision: true      # FP16 AMP
  gradient_clip_norm: 1.0    # Gradient clipping
```

### Optimizer Configuration

```yaml
optimizer:
  name: "adamw"
  lr: 1.0e-4                # Learning rate
  weight_decay: 1.0e-5       # Weight decay
  
  scheduler:
    name: "cosine"           # Cosine annealing
    warmup_steps: 500        # Linear warmup
    min_lr: 1.0e-6           # Final LR
```

### Loss Configuration

```yaml
loss:
  sar_consistency_weight: 1.0   # SAR edge + texture
  cycle_weight: 0.5             # Cycle consistency
  identity_weight: 0.3          # Identity preservation
  lpips_weight: 0.1             # Perceptual loss
  spectral_weight: 1.0          # Pixel-level L1
```

---

## Key Parameters Explained

### `freeze_base_model: true` ✅

**Purpose**: Freeze all parameters except LoRA adapters

**Effect**:
- Base model parameters: `requires_grad = False`
- LoRA adapters: `requires_grad = True`
- Only ~1-2% parameters trainable
- Dramatically reduces memory usage

**When to use**: Always for low VRAM setups

### `timesteps: 10`

**Purpose**: Number of diffusion denoising steps

**Options**:
- `5`: Fast, lower quality (~0.5s/sample)
- `10`: Balanced (default) (~1s/sample)
- `20`: High quality, slower (~2s/sample)

**Recommendation**: Start with 10, increase to 20 if quality issues

### `max_steps: 30000`

**Purpose**: Maximum training iterations

**Guidelines**:
- **15k steps**: Quick training, may underfit
- **30k steps**: Standard (recommended)
- **50k steps**: Longer training, diminishing returns

**Formula**: `steps = samples / (batch_size × grad_accum)`

Example: 30k steps = 30k × 8 = 240k training samples

### `lr: 1.0e-4`

**Purpose**: Learning rate for LoRA adapters

**Guidelines**:
- `1e-4`: Standard, stable
- `5e-5`: Conservative, if loss unstable
- `2e-4`: Aggressive, faster convergence

**Recommendation**: Start with 1e-4, reduce if unstable

### `lora.rank: 8`

**Purpose**: Rank of LoRA decomposition matrices

**Options**:
- `rank=4`: Fewer parameters, faster, may underfit
- `rank=8`: Balanced (default)
- `rank=16`: More parameters, better quality, slower

**Effect on parameters**:
- rank=4: ~0.5% trainable
- rank=8: ~1% trainable
- rank=16: ~2% trainable

---

## Memory Budget

### VRAM Breakdown (batch=1, rank=8)

| Component | VRAM | Notes |
|-----------|------|-------|
| Base model (frozen) | ~3.5 GB | TerraMind parameters |
| LoRA adapters | ~50 MB | Trainable parameters |
| Gradients (LoRA only) | ~200 MB | Only for LoRA |
| Optimizer state | ~100 MB | AdamW momentum |
| Activations | ~1-1.5 GB | Forward pass |
| **Total** | **~5-6 GB** | |

### Reducing VRAM Usage

If you still run out of memory:

1. **Reduce tile size**: `tile_size: 96` (saves ~1 GB)
2. **Reduce LoRA rank**: `rank: 4` (saves ~100 MB)
3. **Reduce timesteps**: `timesteps: 5` (saves ~500 MB)
4. **Disable LPIPS**: `use_lpips: false` (saves ~1 GB)

---

## Training Duration

### Expected Training Time

| Steps | RTX 3060 | RTX 3070 | RTX 3080 |
|-------|----------|----------|----------|
| 15k | ~1 hour | ~40 min | ~30 min |
| 30k | ~2-3 hours | ~1.5 hours | ~1 hour |
| 50k | ~4-5 hours | ~2.5 hours | ~1.5 hours |

**Throughput**: ~150-200 samples/minute  
**Steps/hour**: ~2000-2500

### Monitoring Progress

Track these metrics during training:

1. **SAR Agreement**: Should increase over time
   - Target: +0.05 to +0.10 improvement
   - Check every 500 steps

2. **Total Loss**: Should decrease and stabilize
   - Target: 0.02-0.03 at convergence
   - Watch for instability

3. **LPIPS Change**: Should stay low
   - Target: < 0.15
   - Monitor on validation set

---

## Expected Results

### Good Training Run

✅ **Metrics**:
- SAR agreement improvement: +0.05 to +0.10
- Final SAR agreement: > 0.70
- LPIPS change: 0.08-0.12 (below threshold)
- Training loss: 0.02-0.03
- Validation loss: 0.025-0.035

✅ **Observations**:
- Loss decreases smoothly
- SAR agreement increases consistently
- No overfitting (train/val gap small)

### Problematic Training Run

⚠️ **Symptoms**:
- SAR agreement improvement < 0.01
- LPIPS change > 0.20
- Loss unstable (spikes)
- Large train/val gap

⚠️ **Solutions**:
- Reduce learning rate: `lr: 5e-5`
- Increase SAR weight: `sar_consistency_weight: 1.5`
- Add more regularization: `weight_decay: 1e-4`

---

## Hyperparameter Tuning

### Loss Weight Tuning

**Problem**: SAR agreement not improving
```yaml
loss:
  sar_consistency_weight: 1.5  # ← Increase
  cycle_weight: 0.3            # ← Decrease
  identity_weight: 0.2         # ← Decrease
```

**Problem**: Changes too drastic (high LPIPS)
```yaml
loss:
  sar_consistency_weight: 0.8  # ← Decrease
  cycle_weight: 0.7            # ← Increase
  identity_weight: 0.5         # ← Increase
```

**Problem**: Poor perceptual quality
```yaml
loss:
  lpips_weight: 0.2            # ← Increase
  spectral_weight: 0.5         # ← Decrease
```

### LoRA Tuning

**For faster training** (may sacrifice quality):
```yaml
lora:
  rank: 4                      # ← Reduce
  alpha: 8                     # ← Reduce (2×rank)
```

**For better quality** (slower, more memory):
```yaml
lora:
  rank: 16                     # ← Increase
  alpha: 32                    # ← Increase (2×rank)
```

### Learning Rate Tuning

**Training converges too slowly**:
```yaml
optimizer:
  lr: 2.0e-4                   # ← Increase
  scheduler:
    warmup_steps: 1000         # ← Increase warmup
```

**Loss is unstable/spiky**:
```yaml
optimizer:
  lr: 5.0e-5                   # ← Decrease
  gradient_clip_norm: 0.5      # ← Tighter clipping
```

---

## Advanced Options

### Step-based Training (Recommended)

```yaml
training:
  max_steps: 30000             # Use steps instead of epochs
  epochs: null                 # Disable epoch-based
```

**Advantages**:
- More predictable training time
- Easier to compare across datasets
- Better for large datasets

### Epoch-based Training

```yaml
training:
  max_steps: null              # Disable step-based
  epochs: 50                   # Use epochs
```

**When to use**: Small datasets (<5k samples)

### Resume Training

```yaml
resume:
  checkpoint_path: "checkpoints/stage3/checkpoint_15000.pt"
  resume_training: true
  load_optimizer: true
  load_scheduler: true
```

**Usage**:
```bash
python scripts/train_stage3.py \
  --config configs/train/stage3.lowvr.yaml \
  --resume checkpoints/stage3/checkpoint_15000.pt
```

---

## Debugging

### Quick Overfit Test

Test if model can learn on single batch:

```yaml
debug:
  enabled: true
  overfit_single_batch: true
  max_samples: 8
```

**Expected**: Loss → 0 after ~100 steps

### Anomaly Detection

Debug NaN/Inf issues:

```yaml
debug:
  detect_anomaly: true
```

**Warning**: Very slow, only for debugging

---

## Config Validation

### Check Your Config

```bash
python scripts/train_stage3.py \
  --config configs/train/stage3.lowvr.yaml \
  --validate-config
```

### Print All Settings

```bash
python scripts/train_stage3.py \
  --config configs/train/stage3.lowvr.yaml \
  --print-config
```

---

## FAQ

### Q: How do I know if LoRA is working?

**A**: Check trainable parameters count:
```
✓ Total parameters: 450,000,000
✓ Trainable parameters: 4,500,000 (1.0%)
```

If trainable % > 5%, LoRA may not be applied correctly.

### Q: What if I have more VRAM?

**A**: Increase batch size or LoRA rank:
```yaml
training:
  batch_size: 2              # If >8 GB VRAM
  gradient_accumulation_steps: 4

lora:
  rank: 16                   # Better quality
```

### Q: Can I train on CPU?

**A**: Yes, but very slow:
```yaml
hardware:
  device: "cpu"
```

Expect ~10-20× slower training.

### Q: How to disable early stopping?

**A**:
```yaml
training:
  early_stopping:
    enabled: false
```

---

## Related Files

- **Training script**: `scripts/train_stage3.py`
- **Loss functions**: `axs_lib/stage3_losses.py`
- **Model**: `axs_lib/stage3_tm_ground.py`
- **Training guide**: `docs/STAGE3_TRAINING.md`

---

## Support

For issues or questions:
1. Check `docs/STAGE3_TRAINING.md` for detailed guide
2. Review training logs in `logs/stage3_lowvr/`
3. Open an issue on the Axion-Sat repository

---

**Last Updated**: 2025-10-14  
**Config Version**: 1.0.0
