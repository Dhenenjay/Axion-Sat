# Stage 2: Prithvi Refinement Training Guide

## Overview

Stage 2 refines TerraMind's "mental images" (opt_v1) into high-quality optical features using:
- **Prithvi-EO-2.0-600M** backbone with 8-bit quantization
- **LoRA** (Low-Rank Adaptation) for parameter-efficient fine-tuning
- **ConvNeXt** refinement head for spatial enhancement
- **Metadata conditioning** (month, biome) for context-aware refinement
- **Multi-component loss** (spectral, identity/edge-guard, adversarial)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Stage 2 Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Input: opt_v1 (4 ch) + month + biome                       │
│     ↓                                                         │
│  Metadata Embedding                                          │
│     ↓                                                         │
│  Fusion: opt_v1 + metadata features → 8 ch → 4 ch           │
│     ↓                                                         │
│  Prithvi Backbone (8-bit + LoRA)                            │
│     ↓                                                         │
│  ConvNeXt Refinement Head                                    │
│     ↓                                                         │
│  Output: refined optical (4 ch: B02, B03, B04, B08)         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Loss Functions

### 1. Spectral Plausibility Loss (weight: 1.0)
- **NDVI RMSE**: Ensures vegetation index consistency
- **EVI RMSE**: Enhanced vegetation index validation
- **SAM**: Spectral Angle Mapper for spectral similarity

### 2. Identity/Edge-Guard Loss (weight: 0.5)
- **Identity Loss**: L1 distance between input and output
- **Edge Preservation**: Sobel gradient matching

### 3. PatchGAN Adversarial Loss (weight: 0.05)
- **Texture Detail**: Lightweight discriminator for realistic high-frequency content
- **LSGAN**: Least-squares GAN for training stability

## Memory Optimization

**Target: 8-12 GB VRAM systems**

| Technique | Memory Savings | Implementation |
|-----------|----------------|----------------|
| 8-bit quantization | ~75% | bitsandbytes for Prithvi |
| LoRA fine-tuning | ~99% trainable params | rank=8, alpha=16 |
| Gradient accumulation | 50% per step | batch=1, accum=8 |
| Mixed precision (FP16) | ~50% | PyTorch AMP |
| **Total Model Memory** | **~200-250 MB** | vs ~2.5 GB full precision |

## Data Requirements

### Input Files (NPZ tiles)
```python
{
    's2_b2': np.ndarray,      # Blue band (120, 120)
    's2_b3': np.ndarray,      # Green band (120, 120)
    's2_b4': np.ndarray,      # Red band (120, 120)
    's2_b8': np.ndarray,      # NIR band (120, 120)
    'month': int,             # Acquisition month (1-12)
    'biome_code': int,        # Biome classification (0-16)
}
```

### Expected Directory Structure
```
data/tiles/benv2_catalog/
├── train/
│   ├── tile_0001.npz
│   ├── tile_0001.json  # Optional metadata
│   ├── tile_0002.npz
│   └── ...
├── val/
│   ├── tile_0001.npz
│   └── ...
└── test/
    └── ...
```

### Optional: Pre-computed Stage 1 Outputs
```
data/stage1_outputs/
├── tile_0001.npz  # Contains 'opt_v1' key
├── tile_0002.npz
└── ...
```

## Training

### Quick Start

```bash
# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install bitsandbytes peft transformers
pip install pyyaml tqdm numpy

# Train with default settings
python scripts/train_stage2.py \
    --config configs/hardware.lowvr.yaml \
    --data_dir data/tiles/benv2_catalog \
    --output_dir outputs/stage2
```

### With Pre-computed Stage 1 Outputs

```bash
python scripts/train_stage2.py \
    --config configs/hardware.lowvr.yaml \
    --data_dir data/tiles/benv2_catalog \
    --stage1_dir data/stage1_outputs \
    --output_dir outputs/stage2
```

### Resume Training

```bash
python scripts/train_stage2.py \
    --config configs/hardware.lowvr.yaml \
    --data_dir data/tiles/benv2_catalog \
    --resume outputs/stage2/checkpoint_epoch_020.pt \
    --output_dir outputs/stage2
```

### Debug Mode (Small Dataset)

```bash
python scripts/train_stage2.py \
    --config configs/hardware.lowvr.yaml \
    --data_dir data/tiles/benv2_catalog \
    --max_samples 100 \
    --epochs 5 \
    --output_dir outputs/stage2_debug
```

## Configuration

### Key Parameters (configs/hardware.lowvr.yaml)

```yaml
stage2:
  model:
    lora_r: 8                    # LoRA rank (4, 8, 16, 32)
    lora_alpha: 16               # LoRA scaling factor
    lora_dropout: 0.1            # LoRA dropout rate
    freeze_base: true            # Freeze Prithvi backbone
    use_lora: true               # Enable LoRA

training:
  batch_size: 1                  # Batch size per GPU
  grad_accum_steps: 8            # Gradient accumulation (effective batch = 8)
  precision: fp16                # Mixed precision training
  learning_rate: 1.0e-4          # Initial learning rate
  epochs: 100                    # Total training epochs
  max_grad_norm: 1.0             # Gradient clipping threshold
  
  checkpoint_interval: 10        # Save every N epochs
  validation_interval: 5         # Validate every N epochs
```

### Loss Weights

```python
criterion = Stage2Loss(
    spectral_weight=1.0,         # Spectral plausibility
    identity_weight=0.5,         # Identity/edge-guard
    adversarial_weight=0.05      # PatchGAN (keep low!)
)
```

## Monitoring

### Training Logs

```bash
# View live training progress
tail -f outputs/stage2/training_log.jsonl | jq .

# Extract validation losses
cat outputs/stage2/training_log.jsonl | jq 'select(.val.loss != null) | {epoch, val_loss: .val.loss}'
```

### TensorBoard (TODO)

```bash
tensorboard --logdir outputs/stage2/tensorboard
```

## Outputs

### Checkpoints

```
outputs/stage2/
├── best.pt                      # Best validation checkpoint
├── checkpoint_epoch_010.pt      # Periodic checkpoints
├── checkpoint_epoch_020.pt
├── ...
├── config.yaml                  # Training configuration
└── training_log.jsonl           # Training metrics
```

### Checkpoint Contents

```python
{
    'model_state_dict': {...},    # Model weights
    'optimizer_state_dict': {...}, # Optimizer state
    'epoch': int,                  # Training epoch
    'metrics': {...},              # Validation metrics
    'config': {...}                # Training config
}
```

## Inference

### Load Trained Model

```python
from axs_lib.stage2_prithvi_refine import build_prithvi_refiner
from axs_lib.io import load_checkpoint
import torch

# Build model
prithvi_refiner = build_prithvi_refiner(
    config=config,
    device='cuda'
)

model = Stage2ModelWithMetadata(
    prithvi_refiner=prithvi_refiner,
    num_months=12,
    num_biomes=16
)

# Load checkpoint
checkpoint = load_checkpoint('outputs/stage2/best.pt', device='cuda')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    refined = model(opt_v1, month, biome)
```

## Performance Expectations

### Memory Usage
- **Training**: 8-10 GB VRAM (batch=1, grad_accum=8)
- **Inference**: 4-6 GB VRAM

### Training Speed
- **RTX 3060 (12 GB)**: ~50 samples/sec
- **RTX 4060 Ti (16 GB)**: ~80 samples/sec
- **Full epoch (160k samples)**: ~45-60 minutes

### Quality Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| NDVI RMSE | < 0.05 | Vegetation index consistency |
| EVI RMSE | < 0.05 | Enhanced vegetation index |
| SAM | < 0.1 rad | Spectral angle similarity |
| Identity L1 | < 0.05 | Geometric consistency |
| Edge L1 | < 0.02 | Edge preservation |

## Troubleshooting

### Out of Memory (OOM)

**Automatic CPU Fallback**: Stage 2 includes automatic CPU fallback for the refinement head when GPU runs out of memory. This allows training to continue with a performance penalty instead of crashing.

**When CPU Fallback Occurs**:
```
======================================================================
GPU OUT OF MEMORY - ConvNeXt Head Fallback
======================================================================
Moving refinement head to CPU. This will be slower but allows
processing to continue. Consider:
  1. Reducing batch size
  2. Reducing tile size
  3. Disabling discriminator
  4. Reducing num_convnext_blocks
======================================================================

CPU fallback runtime: 0.125s (expected GPU: ~0.013s, penalty: ~0.113s)

⚠️  CPU fallback occurred 5 times this epoch
   Consider reducing memory usage to avoid performance penalty.
```

**To Prevent OOM**:

1. **Reduce batch size**: Already at 1? Can't go lower.
2. **Increase gradient accumulation**: Try `grad_accum_steps=16`
3. **Reduce tile size**: `tile_size: 96` (from 120)
4. **Disable discriminator**: Set `adversarial_weight=0.0`
5. **Reduce LoRA rank**: `lora_r: 4` (from 8)
6. **Reduce ConvNeXt blocks**: `num_convnext_blocks: 2` (from 4)

### Slow Training

1. **Increase batch size** (if VRAM allows): `batch_size=2`
2. **Reduce workers**: `num_workers=1` (less CPU overhead)
3. **Disable augmentation** (during debugging): `use_augmentation=False`
4. **Profile bottlenecks**: Use PyTorch profiler

### Poor Quality

1. **Increase LoRA rank**: `lora_r=16` or `lora_r=32`
2. **Unfreeze more layers**: `unfreeze_last_n_blocks=4`
3. **Increase adversarial weight**: `adversarial_weight=0.1`
4. **Train longer**: `epochs=200`
5. **Check data quality**: Verify S2 targets are clean

### NaN Losses

1. **Reduce learning rate**: `learning_rate: 1.0e-5`
2. **Increase gradient clipping**: `max_grad_norm: 0.5`
3. **Check for invalid data**: NaN/Inf in inputs
4. **Disable mixed precision**: Use `fp32` (slower but more stable)

## Best Practices

### Data Preparation

✓ **DO**: Use high-quality S2/HLS targets  
✓ **DO**: Balance splits by geography and season  
✓ **DO**: Filter out cloudy/invalid tiles  
✗ **DON'T**: Mix different resolutions  
✗ **DON'T**: Skip metadata validation

### Training Strategy

✓ **DO**: Start with small dataset to verify pipeline  
✓ **DO**: Monitor all loss components separately  
✓ **DO**: Save checkpoints frequently (disk is cheap)  
✓ **DO**: Validate on diverse geographic regions  
✗ **DON'T**: Increase all loss weights simultaneously  
✗ **DON'T**: Skip validation to save time

### Hyperparameter Tuning

1. **Start with defaults** from `hardware.lowvr.yaml`
2. **Tune one parameter at a time**
3. **Focus on loss component weights first**
4. **Then tune LoRA rank and learning rate**
5. **Finally, adjust batch size and augmentation**

## Integration with Stage 1 and 3

### From Stage 1
```python
# Stage 1 produces opt_v1 (mental images)
opt_v1 = terramind_generator(sar_input)  # (B, 4, H, W)

# Stage 2 refines opt_v1
refined = stage2_model(opt_v1, month, biome)
```

### To Stage 3
```python
# Stage 2 output feeds into Stage 3 conditional model
refined_features = stage2_model(opt_v1, month, biome)

# Stage 3 performs final grounding
segmentation = stage3_model(refined_features, sar_input)
```

## References

- **Prithvi Foundation Model**: [NASA-IBM Geospatial AI](https://huggingface.co/ibm-nasa-geospatial)
- **LoRA**: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- **8-bit Quantization**: [Dettmers et al., 2022](https://arxiv.org/abs/2208.07339)
- **PatchGAN**: [Isola et al., 2017](https://arxiv.org/abs/1611.07004)

## Support

For issues or questions:
1. Check this documentation first
2. Review training logs: `outputs/stage2/training_log.jsonl`
3. Consult Stage 1 and Stage 3 documentation
4. Open an issue on the project repository

---

*Last Updated: 2025-10-14*  
*Version: 1.0.0*
