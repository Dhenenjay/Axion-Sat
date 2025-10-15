# Cosine Learning Rate Scheduler with Warmup

## Overview

The Axion-Sat project now implements a cosine annealing learning rate scheduler with linear warmup for both Stage 1 and Stage 2 training. This scheduler combines the benefits of:

1. **Linear Warmup**: Gradual LR increase to stabilize early training
2. **Cosine Annealing**: Smooth LR decay for better convergence

## Features

### Two Scheduler Types

#### 1. Step-based Scheduler (`CosineAnnealingWarmupLR`)
- Updates LR every training step (batch or accumulated step)
- Ideal for gradient accumulation training
- Used in Stage 2 training with grad accumulation

#### 2. Epoch-based Scheduler (`CosineAnnealingWarmupEpochLR`)
- Updates LR every epoch
- Simpler for standard training loops
- Useful when training without gradient accumulation

### Learning Rate Schedule

```
LR  │
    │     ╱────╮
    │    ╱      ╲
    │   ╱        ╲
    │  ╱          ╲___
    │ ╱                ╲___
    │╱                     ╲___
    └─────────────────────────────► Steps
    ←warmup→←cosine annealing───→
```

**Phases:**
1. **Warmup** (0 → warmup_steps): Linear increase from 0 to base_lr
2. **Annealing** (warmup_steps → total_steps): Cosine decay from base_lr to eta_min

## Usage

### In Training Script

The scheduler is already integrated into `scripts/train_stage2.py`:

```bash
# Use default warmup (10% of total steps)
python scripts/train_stage2.py \
    --data_dir data/tiles/benv2_catalog \
    --output_dir outputs/stage2

# Specify custom warmup steps
python scripts/train_stage2.py \
    --data_dir data/tiles/benv2_catalog \
    --warmup_steps 500 \
    --output_dir outputs/stage2

# Override learning rate
python scripts/train_stage2.py \
    --data_dir data/tiles/benv2_catalog \
    --warmup_steps 1000 \
    --lr 5e-4 \
    --output_dir outputs/stage2
```

### In Custom Code

```python
from axs_lib.lr_scheduler import CosineAnnealingWarmupLR

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Create scheduler
scheduler = CosineAnnealingWarmupLR(
    optimizer,
    warmup_steps=1000,      # 1000 steps of linear warmup
    total_steps=10000,      # 10000 total training steps
    eta_min=1e-6            # minimum LR
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward + backward
        loss = model(batch)
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Scheduler step (AFTER optimizer.step())
        scheduler.step()
```

### With Gradient Accumulation

```python
# Calculate total steps accounting for gradient accumulation
steps_per_epoch = len(train_loader) // grad_accum_steps
total_steps = steps_per_epoch * num_epochs

scheduler = CosineAnnealingWarmupLR(
    optimizer,
    warmup_steps=int(0.1 * total_steps),
    total_steps=total_steps,
    eta_min=1e-6
)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        # Forward + backward
        loss = model(batch) / grad_accum_steps
        loss.backward()
        
        # Update every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  # Step AFTER accumulated gradients
```

## Command-Line Arguments

### `--warmup_steps`

**Description**: Number of warmup steps for linear LR increase

**Default**: 10% of total training steps

**Examples**:
```bash
# Short warmup (5% of training)
--warmup_steps 500  # for 10000 total steps

# Standard warmup (10%)
--warmup_steps 1000

# Long warmup (20%)
--warmup_steps 2000
```

**Recommendations**:
- **Small datasets**: 10-20% warmup
- **Large datasets**: 5-10% warmup
- **Fine-tuning**: 5% warmup
- **Training from scratch**: 10-15% warmup

### `--lr`

**Description**: Base learning rate (maximum LR after warmup)

**Default**: Loaded from config file

**Examples**:
```bash
# Lower LR for fine-tuning
--lr 1e-4

# Higher LR for training from scratch
--lr 5e-4

# Very low LR for final refinement
--lr 1e-5
```

## Configuration File

You can also set warmup steps in the config YAML:

```yaml
training:
  learning_rate: 1e-4
  lr_scheduler:
    min_lr: 1e-6
    warmup_steps: 1000  # Specify warmup steps here
```

Priority order:
1. CLI argument (`--warmup_steps`)
2. Config file (`training.lr_scheduler.warmup_steps`)
3. Default (10% of total steps)

## Benefits of Warmup

### 1. **Training Stability**
- Prevents large gradient updates in early training
- Reduces risk of divergence with high learning rates
- Especially important for:
  - Large batch sizes
  - Gradient accumulation
  - AdamW optimizer

### 2. **Better Convergence**
- Allows model to explore parameter space gradually
- Reduces sensitivity to initialization
- Improves final model performance

### 3. **Compatibility with Large LR**
- Can use higher base learning rates safely
- Faster training without instability
- Better exploration of loss landscape

## Comparison with Other Schedulers

| Scheduler | Warmup | Smooth Decay | Easy Tuning | Best For |
|-----------|--------|--------------|-------------|----------|
| **Cosine w/ Warmup** | ✓ | ✓ | ✓ | Most cases |
| StepLR | ✗ | ✗ | ✓ | Simple baselines |
| ExponentialLR | ✗ | ✓ | ✗ | Specific tuning |
| ReduceLROnPlateau | ✗ | ✗ | ✓ | Validation-driven |

## Visualization

Generate LR schedule plots:

```bash
python examples/visualize_lr_schedule.py
```

This creates:
- `reports/lr_schedule_10pct_warmup.png`: Standard 10% warmup
- `reports/lr_schedule_20pct_warmup.png`: Longer 20% warmup
- `reports/lr_schedule_comparison.png`: Comparison of different warmup ratios

## Testing

Run scheduler tests:

```bash
python examples/test_lr_scheduler.py
```

Tests verify:
- Linear warmup behavior
- Cosine annealing correctness
- Step-based and epoch-based schedulers
- Edge cases

## Example Schedules

### Training Configuration

| Scenario | Epochs | Batch Size | Grad Accum | Total Steps | Warmup Steps | Base LR | Min LR |
|----------|--------|------------|------------|-------------|--------------|---------|--------|
| Stage 2 (low VRAM) | 20 | 1 | 8 | 5,000 | 500 (10%) | 1e-4 | 1e-6 |
| Stage 2 (high VRAM) | 20 | 8 | 1 | 5,000 | 500 (10%) | 5e-4 | 1e-6 |
| Fine-tuning | 10 | 4 | 2 | 2,500 | 125 (5%) | 1e-4 | 1e-7 |

### LR Values at Key Points

For **10,000 total steps** with **1,000 warmup steps**, base LR **1e-3**, min LR **1e-6**:

| Step | Phase | LR | Relative |
|------|-------|------------|----------|
| 0 | Warmup start | 0.00e+00 | 0% |
| 500 | Mid warmup | 5.00e-04 | 50% |
| 1,000 | Warmup end | 1.00e-03 | 100% |
| 2,500 | Early annealing | 9.34e-04 | 93% |
| 5,000 | Mid annealing | 5.89e-04 | 59% |
| 7,500 | Late annealing | 1.81e-04 | 18% |
| 10,000 | Training end | 1.00e-06 | 0.1% |

## Best Practices

### 1. **Choose Warmup Duration**
- Rule of thumb: 5-10% of total training
- Longer warmup for:
  - Large learning rates
  - Unstable training
  - Complex architectures

### 2. **Monitor Early Training**
- Check loss doesn't explode during warmup
- Verify LR increases smoothly
- Watch for NaN gradients

### 3. **Tune Min LR**
- Set eta_min = 0.001 * base_lr (typical)
- Lower eta_min for longer training
- Can set to 0 if training long enough

### 4. **Integration with Other Techniques**
- Compatible with:
  - Gradient accumulation ✓
  - Mixed precision (AMP) ✓
  - Gradient clipping ✓
  - Weight decay ✓

## Troubleshooting

### Training Unstable During Warmup
**Solution**: Increase warmup_steps or reduce base_lr

### LR Not Updating
**Solution**: Ensure `scheduler.step()` is called after `optimizer.step()`

### Premature Convergence
**Solution**: Reduce min_lr or extend total_steps

### Loss Plateaus Early
**Solution**: Increase base_lr or reduce warmup duration

## Implementation Details

### Mathematical Formula

**Warmup phase** (step < warmup_steps):
```
lr = base_lr * (step / warmup_steps)
```

**Annealing phase** (step >= warmup_steps):
```
progress = (step - warmup_steps) / (total_steps - warmup_steps)
lr = eta_min + (base_lr - eta_min) * 0.5 * (1 + cos(π * progress))
```

### Code Location

- **Scheduler implementation**: `axs_lib/lr_scheduler.py`
- **Stage 2 integration**: `scripts/train_stage2.py`
- **Tests**: `examples/test_lr_scheduler.py`
- **Visualization**: `examples/visualize_lr_schedule.py`

## References

- [Goyal et al. (2017): Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677)
- [Loshchilov & Hutter (2016): SGDR](https://arxiv.org/abs/1608.03983)
- [PyTorch LR Scheduler Docs](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

## Questions?

For issues or questions:
1. Check `examples/test_lr_scheduler.py` for usage examples
2. Run `examples/visualize_lr_schedule.py` to see schedules
3. Review training logs for LR values per epoch
