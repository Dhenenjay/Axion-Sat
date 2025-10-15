# Stage 3 Training Guide

This guide explains how to train Stage 3 of the Axion-Sat pipeline, which uses TerraMind's conditional generator to ground Stage 2 optical outputs with SAR features.

## Overview

**Stage 3** refines Stage 2 outputs by incorporating SAR structural information through a conditional generative model. The model uses TerraMind's diffusion-based generator with LoRA fine-tuning on cross-attention layers.

### Key Features

- **LoRA Fine-tuning**: Parameter-efficient training (only ~1-2% of parameters are trainable)
- **Mixed Precision (AMP fp16)**: Memory-efficient training with automatic mixed precision
- **Gradient Accumulation**: Effective batch size of 8 with batch size 1
- **SAR Agreement Metric**: Custom metric for monitoring SAR-optical consistency
- **Early Stopping**: Stops training when SAR agreement plateaus

### Model Architecture

```
Input: SAR (S1GRD, 2 channels) + Stage 2 Optical (opt_v2, 4 channels)
       ↓
TerraMind Encoder (frozen, except LoRA adapters)
       ↓
Cross-Attention Layers (LoRA fine-tuned)
       ↓
Diffusion Denoiser (frozen, except LoRA adapters)
       ↓
Output: Grounded Optical (opt_v3, 4 channels: B, G, R, NIR)
```

### Loss Function

The training uses a multi-component loss:

```
Total Loss = α·SAR_consistency + β·Cycle + γ·Identity + δ·LPIPS + ε·Spectral

Where:
- SAR_consistency: Edge alignment + texture correlation (weighted by backscatter)
- Cycle: Minimize changes from Stage 2 to Stage 3
- Identity: Preserve Stage 2 output in low-SAR-confidence regions
- LPIPS: Perceptual similarity (feature-level realism)
- Spectral: Pixel-level L1 loss with ground truth
```

Default weights: `α=1.0, β=0.5, γ=0.3, δ=0.1, ε=1.0`

---

## Prerequisites

### 1. Data Preparation

Ensure you have NPZ tiles with the following structure:

```python
tile.npz:
  - s1_vv: Sentinel-1 VV polarization (H, W)
  - s1_vh: Sentinel-1 VH polarization (H, W)
  - s2_b2: Sentinel-2 Blue band (H, W)
  - s2_b3: Sentinel-2 Green band (H, W)
  - s2_b4: Sentinel-2 Red band (H, W)
  - s2_b8: Sentinel-2 NIR band (H, W)
```

Tiles should be organized with split metadata:

```
tiles/
├── train/
│   ├── tile_001.npz
│   ├── tile_001.json  # {"split": "train", ...}
│   └── ...
├── val/
│   ├── tile_100.npz
│   ├── tile_100.json  # {"split": "val", ...}
│   └── ...
└── test/
    └── ...
```

### 2. Stage 2 Outputs (Optional)

If you have pre-computed Stage 2 outputs (opt_v2), organize them:

```
stage2_outputs/
├── tile_001.npz  # Contains 'opt_v2' key
├── tile_002.npz
└── ...
```

If not available, the script will use a noisy version of ground truth as a proxy for initial training.

### 3. Environment Setup

Install required packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install lpips tqdm numpy pathlib
```

Install LPIPS for perceptual loss:

```bash
pip install lpips
```

---

## Training

### Basic Usage

```bash
python scripts/train_stage3.py \
  --data-dir tiles/ \
  --output-dir checkpoints/stage3/ \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 1e-4 \
  --epochs 50
```

### With Pre-computed Stage 2 Outputs

```bash
python scripts/train_stage3.py \
  --data-dir tiles/ \
  --stage2-dir stage2_outputs/ \
  --output-dir checkpoints/stage3/ \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 1e-4 \
  --epochs 50
```

### Resume Training

```bash
python scripts/train_stage3.py \
  --data-dir tiles/ \
  --output-dir checkpoints/stage3/ \
  --resume checkpoints/stage3/checkpoint_epoch_20.pt \
  --epochs 50
```

---

## Command-Line Arguments

### Data Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | str | **required** | Directory containing NPZ tiles |
| `--stage2-dir` | str | `None` | Directory with Stage 2 opt_v2 outputs |
| `--output-dir` | str | **required** | Output directory for checkpoints |

### Model Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--timesteps` | int | `10` | Number of diffusion timesteps |
| `--lora-rank` | int | `8` | LoRA rank (lower = fewer params) |
| `--lora-alpha` | int | `16` | LoRA alpha scaling factor |
| `--pretrained` | flag | `True` | Use pretrained TerraMind weights |

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch-size` | int | `1` | Batch size per GPU |
| `--grad-accum` | int | `8` | Gradient accumulation steps |
| `--epochs` | int | `50` | Number of training epochs |
| `--lr` | float | `1e-4` | Learning rate |
| `--weight-decay` | float | `1e-5` | Weight decay for AdamW |

**Effective batch size** = `batch_size × grad_accum` = `1 × 8 = 8`

### Loss Weights

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--sar-weight` | float | `1.0` | SAR consistency loss weight |
| `--cycle-weight` | float | `0.5` | Cycle loss weight |
| `--identity-weight` | float | `0.3` | Identity preservation weight |
| `--lpips-weight` | float | `0.1` | LPIPS perceptual loss weight |
| `--spectral-weight` | float | `1.0` | Spectral L1 loss weight |

### Early Stopping

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--early-stop-patience` | int | `5` | Patience for early stopping |
| `--early-stop-delta` | float | `1e-4` | Minimum improvement threshold |

**Early stopping** triggers when SAR agreement metric stops improving for `patience` consecutive validations.

### Other Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num-workers` | int | `auto` | DataLoader workers (auto-detected) |
| `--max-samples` | int | `None` | Max samples for debugging |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--resume` | str | `None` | Resume from checkpoint path |

---

## Output Structure

After training, your output directory will contain:

```
checkpoints/stage3/
├── config.json                    # Training configuration
├── checkpoint_epoch_1.pt          # Checkpoint after epoch 1
├── checkpoint_epoch_2.pt          # Checkpoint after epoch 2
├── ...
├── checkpoint_epoch_N.pt          # Final checkpoint
└── best_model.pt                  # Best model (highest SAR agreement)
```

### Checkpoint Contents

Each checkpoint includes:

```python
{
    'epoch': int,                      # Epoch number
    'model_state_dict': dict,          # Model weights
    'optimizer_state_dict': dict,      # Optimizer state
    'scheduler_state_dict': dict,      # LR scheduler state
    'scaler_state_dict': dict,         # AMP scaler state
    'train_stats': dict,               # Training loss statistics
    'val_stats': dict,                 # Validation loss and metrics
    'best_sar_agreement': float,       # Best SAR agreement so far
    'config': dict                     # Training configuration
}
```

---

## Monitoring Training

### Training Progress

During training, you'll see progress bars with:

```
Epoch 1: 100%|████████| 256/256 [02:15<00:00, 1.89it/s, loss=0.0234]
Validation 1: 100%|████████| 64/64 [00:30<00:00, 2.13it/s, loss=0.0198, sar_agr=0.7234]
```

### Epoch Summary

After each epoch:

```
Epoch 1 Summary:
  Train loss: 0.023456
  Val loss: 0.019876
  Val SAR agreement: 0.723456
  Learning rate: 1.00e-04
  Time: 165.23s
```

### SAR Agreement Metric

- **Range**: [0, 1] (higher is better)
- **Interpretation**:
  - `< 0.5`: Poor SAR-optical alignment
  - `0.5 - 0.7`: Moderate alignment
  - `> 0.7`: Good alignment
  - `> 0.8`: Excellent alignment

### Early Stopping Messages

```
✓ SAR agreement improved to 0.734567
```
or
```
No improvement in SAR agreement (3/5)
```

When early stopping triggers:

```
⚠ Early stopping triggered!
  Best SAR agreement: 0.765432 at epoch 23
```

---

## Memory Optimization

### Memory Usage

Estimated memory usage per configuration:

| Config | Model | Gradients | Activations | Total |
|--------|-------|-----------|-------------|-------|
| Batch=1, fp16 | ~3.5 GB | ~0.2 GB | ~1.5 GB | ~5-6 GB |
| Batch=1, fp32 | ~7 GB | ~0.4 GB | ~3 GB | ~10-11 GB |

**LoRA reduces memory** by:
- Only computing gradients for ~1-2% of parameters
- Freezing most of the model weights
- Allowing larger batch sizes or higher resolution

### Tips for Low VRAM

If you encounter OOM errors:

1. **Reduce timesteps**: `--timesteps 5` (faster, less accurate)
2. **Disable LPIPS**: Set `--lpips-weight 0` (saves ~1 GB)
3. **Use smaller LoRA rank**: `--lora-rank 4` (fewer trainable params)
4. **Reduce tile size**: Resize tiles to 96×96 (requires data preprocessing)

---

## Advanced Usage

### Custom Loss Weights

Emphasize SAR consistency:

```bash
python scripts/train_stage3.py \
  --data-dir tiles/ \
  --output-dir checkpoints/stage3_sar_focus/ \
  --sar-weight 2.0 \
  --cycle-weight 0.3 \
  --identity-weight 0.2
```

Emphasize photorealism (perceptual loss):

```bash
python scripts/train_stage3.py \
  --data-dir tiles/ \
  --output-dir checkpoints/stage3_perceptual/ \
  --lpips-weight 0.5 \
  --sar-weight 0.8
```

### Aggressive Early Stopping

For faster experimentation:

```bash
python scripts/train_stage3.py \
  --data-dir tiles/ \
  --output-dir checkpoints/stage3_fast/ \
  --early-stop-patience 3 \
  --early-stop-delta 5e-4 \
  --epochs 30
```

### Debugging with Small Dataset

```bash
python scripts/train_stage3.py \
  --data-dir tiles/ \
  --output-dir checkpoints/stage3_debug/ \
  --max-samples 100 \
  --epochs 5 \
  --batch-size 1 \
  --grad-accum 2
```

---

## Troubleshooting

### Issue: OOM (Out of Memory)

**Solution**: Reduce memory usage
```bash
# Option 1: Reduce timesteps
python scripts/train_stage3.py ... --timesteps 5

# Option 2: Disable LPIPS
python scripts/train_stage3.py ... --lpips-weight 0

# Option 3: Smaller LoRA rank
python scripts/train_stage3.py ... --lora-rank 4 --lora-alpha 8
```

### Issue: Training too slow

**Solution**: Increase gradient accumulation or reduce timesteps
```bash
python scripts/train_stage3.py ... --grad-accum 16 --timesteps 8
```

### Issue: SAR agreement not improving

**Possible causes**:
1. SAR weight too low → Increase `--sar-weight`
2. Cycle/identity weight too high → Decrease `--cycle-weight` and `--identity-weight`
3. Learning rate too high/low → Try `--lr 5e-5` or `--lr 2e-4`

**Solution**: Adjust loss weights
```bash
python scripts/train_stage3.py \
  --sar-weight 1.5 \
  --cycle-weight 0.3 \
  --identity-weight 0.2 \
  --lr 5e-5
```

### Issue: Training diverges (loss increases)

**Solution**: Lower learning rate
```bash
python scripts/train_stage3.py ... --lr 5e-5 --weight-decay 1e-4
```

### Issue: Stage 2 outputs not available

**Solution**: Use proxy (automatically handled)

The script automatically uses a noisy version of ground truth as a proxy for `opt_v2` if `--stage2-dir` is not provided. This allows initial training without Stage 2 outputs.

For production, run Stage 2 first:
```bash
# 1. Train Stage 2
python scripts/train_stage2.py ...

# 2. Generate Stage 2 outputs
python scripts/inference_stage2.py --output-dir stage2_outputs/

# 3. Train Stage 3 with Stage 2 outputs
python scripts/train_stage3.py --stage2-dir stage2_outputs/ ...
```

---

## Performance Metrics

### Training Speed

On NVIDIA RTX 3060 (12GB):
- **Batch size 1, grad_accum 8**: ~2 minutes/epoch (256 batches)
- **Effective throughput**: ~2 samples/second
- **Total training time**: ~1.5 hours for 50 epochs (with early stopping)

### Expected Results

After training (50 epochs):
- **SAR agreement**: 0.75 - 0.85
- **Val loss**: < 0.02
- **SAR consistency loss**: 0.01 - 0.03
- **LPIPS**: 0.05 - 0.15

---

## Next Steps

After training Stage 3:

1. **Evaluate** the model on test set:
   ```bash
   python scripts/evaluate_stage3.py \
     --checkpoint checkpoints/stage3/best_model.pt \
     --data-dir tiles/test/ \
     --output-dir results/stage3/
   ```

2. **Run inference** on new data:
   ```bash
   python scripts/inference_stage3.py \
     --checkpoint checkpoints/stage3/best_model.pt \
     --input-dir new_tiles/ \
     --output-dir outputs/stage3/
   ```

3. **Visualize** results:
   ```bash
   python tools/visualize_stage3.py \
     --results results/stage3/ \
     --output-dir visualizations/
   ```

---

## References

- **LoRA**: [Hu et al., 2021 - LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **LPIPS**: [Zhang et al., 2018 - The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924)
- **TerraMind**: IBM's foundation model for geospatial data

---

## Contact

For issues or questions, please open an issue on the Axion-Sat repository.
