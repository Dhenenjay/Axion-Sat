# Validation Metrics Integration for train_stage1.py

This document shows how to integrate comprehensive validation metrics (SSIM, LPIPS, SAR-edge agreement) with CSV logging into `train_stage1.py`.

## Changes Required

### 1. Add imports at the top of train_stage1.py

```python
# Add to existing imports
from validation_metrics import ValidationMetrics, ValidationCSVLogger, compute_validation_metrics_batch
```

### 2. Update the `validate()` function (around line 492)

Replace the existing `validate()` function with this enhanced version:

```python
@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    timesteps: int = 12,
    compute_detailed_metrics: bool = True,
    max_batches: Optional[int] = 50
) -> Dict[str, float]:
    """Validate the model with comprehensive metrics."""
    
    model.eval()
    
    # Loss tracking
    total_loss = 0.0
    component_losses = {
        'l1': 0.0,
        'ms_ssim': 0.0,
        'lpips': 0.0,
        'sar_structure': 0.0
    }
    
    # Detailed metrics tracking
    if compute_detailed_metrics:
        metrics_computer = ValidationMetrics(device)
        all_metrics = {
            'ssim': [],
            'lpips': [],
            'psnr': [],
            'mae': [],
            'sar_edge_agreement': []
        }
    
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        s1 = batch['s1'].to(device)
        s2_target = batch['s2'].to(device)
        
        # Generate synthetic optical
        s2_pred = tm_sar2opt(
            model,
            s1,
            timesteps=timesteps,
            denormalize=False,
            clip_range=None
        )
        
        # Compute loss
        loss, loss_dict = criterion(s2_pred, s2_target, sar=s1)
        
        total_loss += loss.item()
        for key in component_losses:
            if key in loss_dict:
                component_losses[key] += loss_dict[key]
        
        # Compute detailed metrics
        if compute_detailed_metrics:
            batch_metrics = compute_validation_metrics_batch(
                s2_pred, s2_target, s1, metrics_computer
            )
            for key in all_metrics:
                all_metrics[key].extend(batch_metrics.get(key, []))
        
        num_batches += 1
    
    # Average losses
    avg_loss = total_loss / num_batches
    for key in component_losses:
        component_losses[key] /= num_batches
    
    results = {
        'loss': avg_loss,
        **component_losses
    }
    
    # Add detailed metrics with std
    if compute_detailed_metrics:
        for key in all_metrics:
            if all_metrics[key]:
                results[key] = np.mean(all_metrics[key])
                results[f'{key}_std'] = np.std(all_metrics[key])
    
    return results
```

### 3. Update validation call in `train_step_based()` (around line 430)

Replace the validation section with:

```python
# Validation
if step % val_every == 0 and step > 0 and val_dataloader is not None:
    print(f"\n[Step {step}] Running validation...")
    val_metrics = validate(
        model=model,
        dataloader=val_dataloader,
        criterion=criterion,
        device=device,
        timesteps=timesteps,
        compute_detailed_metrics=True,
        max_batches=50  # Limit batches for speed
    )
    print(f"Val Loss: {val_metrics['loss']:.4f}")
    print(f"  L1: {val_metrics['l1']:.4f}, MS-SSIM: {val_metrics['ms_ssim']:.4f}")
    if 'ssim' in val_metrics:
        print(f"  SSIM: {val_metrics['ssim']:.4f} ± {val_metrics['ssim_std']:.4f}")
        print(f"  LPIPS: {val_metrics['lpips']:.4f} ± {val_metrics['lpips_std']:.4f}")
        print(f"  PSNR: {val_metrics['psnr']:.2f} ± {val_metrics['psnr_std']:.2f} dB")
        print(f"  SAR-Edge: {val_metrics['sar_edge_agreement']:.4f} ± {val_metrics['sar_edge_agreement_std']:.4f}\n")
    
    # Log to JSONL
    if logger is not None:
        logger.log_validation(
            step=step,
            val_loss=val_metrics['loss'],
            **val_metrics
        )
    
    # Log to CSV
    if val_csv_logger is not None:
        val_csv_logger.log(step, val_metrics)
    
    # Save best model
    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        if output_dir is not None:
            best_path = output_dir / 'best_model.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': step,
                'val_loss': val_metrics['loss']
            }, best_path)
            print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
    
    model.train()
```

### 4. Add CSV logger initialization in `main()` (around line 808)

Add after the JSONL logger initialization:

```python
# Initialize validation CSV logger
val_csv_path = output_dir / 'logs' / 'stage1_val.csv'
val_csv_logger = ValidationCSVLogger(str(val_csv_path))
print(f"Validation metrics will be logged to: {val_csv_path}")
```

### 5. Pass CSV logger to training function (around line 954)

Add `val_csv_logger` parameter to the `train_step_based()` call:

```python
results = train_step_based(
    model=model,
    dataloader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    device=device,
    grad_accum_steps=args.grad_accum_steps,
    timesteps=args.timesteps,
    total_steps=total_steps,
    val_every=args.val_every,
    save_every=args.save_every,
    start_step=start_step,
    val_dataloader=val_loader,
    logger=logger,
    ckpt_manager=ckpt_manager,
    output_dir=output_dir,
    val_csv_logger=val_csv_logger  # ADD THIS LINE
)
```

### 6. Update `train_step_based()` signature (around line 339)

Add the parameter to the function signature:

```python
def train_step_based(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    grad_accum_steps: int = 8,
    timesteps: int = 12,
    total_steps: int = 10000,
    val_every: int = 500,
    save_every: int = 1000,
    start_step: int = 0,
    val_dataloader: Optional[DataLoader] = None,
    logger: Optional[TrainingLogger] = None,
    ckpt_manager: Optional[CheckpointManager] = None,
    output_dir: Optional[Path] = None,
    val_csv_logger: Optional[ValidationCSVLogger] = None  # ADD THIS LINE
) -> Dict[str, float]:
```

## Output Format

The CSV file `logs/stage1_val.csv` will contain:

```csv
step,timestamp,loss,l1,ms_ssim,lpips_loss,sar_structure,ssim,ssim_std,lpips,lpips_std,psnr,psnr_std,mae,mae_std,sar_edge_agreement,sar_edge_agreement_std
500,2025-10-13T16:30:00.123456,0.1234,0.0567,0.0234,0.0123,0.0310,0.8765,0.0234,0.1234,0.0123,28.45,1.23,0.0567,0.0089,0.7654,0.0234
1000,2025-10-13T16:35:00.234567,0.1123,0.0523,0.0212,0.0112,0.0288,0.8892,0.0212,0.1145,0.0112,29.12,1.15,0.0523,0.0082,0.7823,0.0212
...
```

## Quick Integration

Run this command to apply all changes automatically:

```bash
# Copy the validation_metrics.py to scripts/
cp scripts/validation_metrics.py scripts/

# Then manually apply the changes listed above to train_stage1.py
```

## Usage

After integration, training will automatically log detailed validation metrics every `--val-every` steps:

```bash
python scripts/train_stage1.py \
    --data-dir tiles/ \
    --output-dir runs/exp1/ \
    --val-every 500 \
    --steps 10000
```

Metrics will be logged to:
- JSONL: `runs/exp1/training.jsonl` (detailed per-step logs)
- CSV: `runs/exp1/logs/stage1_val.csv` (validation metrics only)
