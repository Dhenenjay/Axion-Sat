# Early Stopping & Top-K Checkpoint Management

## Overview

The Stage 1 training script (`scripts/train_stage1.py`) now includes advanced features for robust training on constrained hardware:

1. **Early Stopping on SAR-Edge Agreement Plateau** - Automatically stops training when the model stops improving
2. **Top-K Checkpoint Management** - Keeps only the best K checkpoints based on SAR-edge agreement metric
3. **GPU Memory Logging** - Tracks VRAM usage throughout training
4. **Automatic OOM Recovery** - Reduces diffusion timesteps on out-of-memory errors

## Features

### 1. Early Stopping

Early stopping monitors the **SAR-edge agreement** metric (which corresponds to the `sar_structure` loss component) and stops training if no improvement is seen for a specified number of validation runs.

**How it works:**
- Monitors SAR-edge agreement (lower is better)
- Tracks best score and counts validations without improvement
- Stops training after N validations without improvement (configurable patience)
- Requires minimum improvement threshold (min_delta) to count as progress

**Benefits:**
- Prevents overfitting
- Saves compute time on 6GB GPUs
- Automatically finds optimal stopping point

### 2. Top-K Checkpoint Management

Instead of keeping all checkpoints, the system maintains only the **top-3 best** checkpoints based on SAR-edge agreement.

**How it works:**
- Saves checkpoint only if it's in top-K by SAR-edge agreement
- Automatically removes worse checkpoints when K is exceeded
- Stores in separate `checkpoints/top_k/` directory
- Filenames include metric score for easy identification

**Example filename:** `best_sar_edge_agreement_0.123456_step_5000.pt`

**Benefits:**
- Saves disk space (important for limited storage)
- Easy to identify best models
- Automatic cleanup of suboptimal checkpoints

### 3. SAR-Edge Agreement Metric

This metric measures how well the generated optical imagery preserves edge structures from the input SAR data.

- **Lower values = better** edge preservation
- Computed as part of the `sar_structure` loss component
- More reliable than validation loss for SAR-to-optical quality

## Usage

### Command Line Arguments

```bash
python scripts/train_stage1.py \
  --data-dir tiles/ \
  --output-dir runs/stage1_lowvr/ \
  --timesteps 10 \
  --steps 45000 \
  --lora-rank 8 \
  --early-stopping \
  --early-stopping-patience 5 \
  --early-stopping-min-delta 1e-4 \
  --keep-top-k 3
```

#### Early Stopping Parameters

- `--early-stopping`: Enable early stopping (flag)
- `--early-stopping-patience <int>`: Number of validations without improvement before stopping (default: 5)
- `--early-stopping-min-delta <float>`: Minimum change to count as improvement (default: 1e-4)

#### Checkpoint Management

- `--keep-top-k <int>`: Number of best checkpoints to keep (default: 3)

### Configuration File

The `configs/train/stage1.lowvr.yaml` includes these settings:

```yaml
# Early Stopping
early_stopping:
  enabled: true
  metric: sar_edge_agreement
  patience: 5          # Stop after 5 validations without improvement
  min_delta: 0.0001    # 1e-4 threshold
  mode: min            # Lower is better

# Top-K checkpoints by SAR-edge agreement
top_k_checkpoints:
  k: 3                 # Keep best 3
  save_dir: checkpoints/top_k
```

## Training Output

### Console Output

During training, you'll see:

```
Training from step 0 to 45000...
Validation every 1000 steps, saving every 2500 steps
Initial timesteps: 10
Early stopping: patience=5, min_delta=0.0001
Top-3 checkpoint management enabled
================================================================================

[Step 1000] Running validation...
Val Loss: 0.2345
  L1: 0.1234, MS-SSIM: 0.0567
  SAR-Edge Agreement: 0.0890

✓ Metric improved to 0.089000
✓ Saved top-K checkpoint: best_sar_edge_agreement_0.089000_step_1000.pt
```

### Early Stopping Triggered

```
[Step 8000] Running validation...
Val Loss: 0.2123
  L1: 0.1100, MS-SSIM: 0.0534
  SAR-Edge Agreement: 0.0875

⚠️  No improvement for 5/5 validations (best: 0.087200 at step 3000)

🛑 Early stopping triggered! No improvement for 5 validations.
   Best score: 0.087200 at step 3000
================================================================================
Early stopping triggered - training complete!
Best SAR-edge agreement: 0.087200 at step 3000
================================================================================
```

## Output Files

### Directory Structure

```
runs/stage1_lowvr/
├── checkpoints/
│   ├── checkpoint_step_2500.pt
│   ├── checkpoint_step_5000.pt
│   ├── checkpoint_step_7500.pt
│   └── top_k/
│       ├── best_sar_edge_agreement_0.087200_step_3000.pt
│       ├── best_sar_edge_agreement_0.088150_step_2500.pt
│       └── best_sar_edge_agreement_0.089000_step_1000.pt
├── best_model.pt         # Best by validation loss
└── training.jsonl        # Training logs
```

### Checkpoint Contents

Each top-K checkpoint contains:
```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'scaler_state_dict': ...,
    'step': 3000,
    'sar_edge_agreement': 0.087200,
    'val_loss': 0.2123
}
```

## Best Practices

### For 6GB GPUs

1. **Enable early stopping** - Saves hours of training time
2. **Use patience=5** - Good balance for 1000-step validation intervals
3. **Keep top-3 checkpoints** - Enough to compare, not too much storage
4. **Monitor SAR-edge agreement** - More stable than validation loss

### Hyperparameter Tuning

#### Aggressive Early Stopping (faster experimentation)
```bash
--early-stopping-patience 3 --early-stopping-min-delta 5e-4
```

#### Conservative Early Stopping (longer training)
```bash
--early-stopping-patience 10 --early-stopping-min-delta 1e-5
```

### Resuming Training

To resume from a top-K checkpoint:

```bash
python scripts/train_stage1.py \
  --data-dir tiles/ \
  --output-dir runs/stage1_lowvr/ \
  --resume runs/stage1_lowvr/checkpoints/top_k/best_sar_edge_agreement_0.087200_step_3000.pt \
  --early-stopping \
  --keep-top-k 3
```

## Monitoring

### GPU Memory Logging

Every 100 steps, you'll see:
```
[Step 100 - Before Forward] GPU Memory: Allocated=4.23GB, Reserved=4.50GB, Free=1.77GB, Total=6.00GB
[Step 100 - After Backward] GPU Memory: Allocated=4.45GB, Reserved=4.75GB, Free=1.55GB, Total=6.00GB
```

### Training Logs

The `training.jsonl` file includes:
```json
{
  "step": 1000,
  "val_loss": 0.2345,
  "sar_edge_agreement": 0.0890,
  "l1": 0.1234,
  "ms_ssim": 0.0567,
  "lpips": 0.0234,
  "sar_structure": 0.0890
}
```

## Troubleshooting

### Early Stopping Too Aggressive

**Symptom:** Training stops very early (< 5000 steps)

**Solution:** Increase patience or decrease min_delta:
```bash
--early-stopping-patience 10 --early-stopping-min-delta 1e-5
```

### Not Enough Checkpoints Saved

**Symptom:** Only 1-2 checkpoints in top_k directory

**Solution:** Model might be improving slowly. This is normal - only truly better checkpoints are saved.

### Too Many Checkpoints

**Symptom:** Many checkpoints filling disk

**Solution:** Reduce --keep-top-k:
```bash
--keep-top-k 2
```

## Performance Impact

- **Early stopping overhead:** < 1% (just comparison operations)
- **Top-K checkpoint overhead:** Minimal (only saves when in top-K)
- **Memory logging overhead:** Negligible (< 0.1%)
- **OOM handling overhead:** Only when OOM occurs (automatic recovery)

## Technical Details

### SAR-Edge Agreement Calculation

Computed in `CombinedLoss` as the `sar_structure` component:
1. Extract edges from input SAR using Sobel filter
2. Extract edges from generated optical
3. Compute L1 distance between edge maps
4. Lower distance = better edge preservation

### Early Stopping Algorithm

```python
if score < (best_score - min_delta):
    # Improvement detected
    best_score = score
    counter = 0
else:
    # No improvement
    counter += 1
    if counter >= patience:
        stop_training()
```

### Top-K Management

```python
if len(top_k) < k or score < worst_in_top_k:
    save_checkpoint()
    if len(top_k) > k:
        remove_worst_checkpoint()
```

## Examples

### Quick Test Run

```bash
python scripts/train_stage1.py \
  --data-dir tiles/ \
  --output-dir test/ \
  --max-tiles 100 \
  --steps 1000 \
  --val-every 100 \
  --early-stopping \
  --early-stopping-patience 3 \
  --keep-top-k 2
```

### Production Training

```bash
python scripts/train_stage1.py \
  --data-dir tiles/ \
  --output-dir runs/stage1_production/ \
  --steps 60000 \
  --val-every 1000 \
  --timesteps 10 \
  --lora-rank 8 \
  --early-stopping \
  --early-stopping-patience 5 \
  --early-stopping-min-delta 1e-4 \
  --keep-top-k 3
```

### Without Early Stopping

```bash
python scripts/train_stage1.py \
  --data-dir tiles/ \
  --output-dir runs/stage1_full/ \
  --steps 45000 \
  --keep-top-k 3
  # Note: --early-stopping flag omitted
```

## See Also

- `configs/train/stage1.lowvr.yaml` - Full configuration example
- `scripts/train_stage1.py` - Training script source
- `axs_lib/losses.py` - Loss function implementations
