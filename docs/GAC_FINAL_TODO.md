# GAC Implementation - Final Checklist

## ✅ COMPLETED

### 1. Spectral Module (`axs_lib/spectral.py`)
- ✅ Added `compute_spectral_rmse()` function
- ✅ Computes average RMSE of NDVI and EVI
- ✅ Ready for use in validation loops

### 2. Losses Module (`axs_lib/losses.py`)
- ✅ Added `ColorConsistencyLoss` class
  - Computes L1 distance between band ratios (R/G, R/B, NIR/R)
  - Ensures physical plausibility of spectral relationships
- ✅ Updated `CombinedLoss` to include `color_consistency_weight` parameter
- ✅ Added color consistency to forward method
- ✅ Noted that SAR-structure should use weight=0.0 for GAC Stage 1

---

## 🔧 REMAINING TASKS

### Stage 1 Training Script (`scripts/train_stage1.py`)

#### Line 1364-1370: Update loss initialization for GAC
**Current**:
```python
criterion = CombinedLoss(
    l1_weight=args.l1_weight,
    ms_ssim_weight=args.ms_ssim_weight,
    lpips_weight=args.lpips_weight,
    sar_structure_weight=args.sar_structure_weight
).to(device)
```

**Update to**:
```python
criterion = CombinedLoss(
    l1_weight=args.l1_weight,
    ms_ssim_weight=args.ms_ssim_weight,
    lpips_weight=args.lpips_weight,
    color_consistency_weight=args.color_consistency_weight,  # NEW: GAC Stage 1
    sar_structure_weight=0.0  # GAC: Move to Stage 3 only
).to(device)
```

#### Line 1196-1204: Add CLI flag for color consistency
**Add after** `--sar-structure-weight`:
```python
parser.add_argument('--color-consistency-weight', type=float, default=0.5,
                    help='Color consistency loss weight (default: 0.5, GAC Stage 1)')
```

#### Line 879-1020: Add GAC-Score to validation function
**Add after computing SAR-edge agreement** (around line 745):
```python
# Compute GAC components
from axs_lib.spectral import compute_spectral_rmse
from axs_lib.metrics import GACScore

# Compute spectral RMSE
spectral_rmse = 0.0
if len(results) > 0:
    # Compute on accumulated predictions
    all_preds = torch.cat(all_s2_preds, dim=0)  # Need to track this
    all_targets = torch.cat(all_s2_targets, dim=0)  # Need to track this
    spectral_rmse = compute_spectral_rmse(all_preds, all_targets)

# Compute GAC-Score
gac_metric = GACScore(include_lpips=(args.lpips_weight > 0))
gac_score_val = 0.0
if len(results) > 0:
    gac_score_val, gac_components = gac_metric(
        all_preds, all_targets, all_s1,
        return_components=True
    )
    gac_score_val = gac_score_val.item()

result = {
    'loss': avg_loss,
    'sar_agreement': sar_edge_agreement,
    'spectral_rmse': spectral_rmse,  # NEW
    'gac_score': gac_score_val,  # NEW
    **component_losses
}
```

#### Line 756-770: Save best model by GAC-Score
**Add after saving best by val_loss**:
```python
# Save best model by GAC-Score (GAC criterion)
if val_metrics.get('gac_score', 0) > best_gac_score:
    best_gac_score = val_metrics['gac_score']
    if output_dir is not None:
        best_gac_path = output_dir / 'best_model_gac.pt'
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'step': step,
            'val_loss': val_metrics['loss'],
            'sar_agreement': val_metrics['sar_agreement'],
            'spectral_rmse': val_metrics.get('spectral_rmse', 0),
            'gac_score': val_metrics['gac_score']
        }
        add_seed_to_checkpoint(checkpoint, seed_info)
        torch.save(checkpoint, best_gac_path)
        print(f"✓ Saved best GAC model (score: {best_gac_score:.4f})")
```

---

### Stage 2 Training Script (`scripts/train_stage2.py`)

#### Verify edge-guard loss exists in `axs_lib/stage2_losses.py`
**Check lines 200-435** in `stage2_losses.py` for edge-guard implementation.

If missing, add to `Stage2Loss`:
```python
def _edge_guard_loss(self, pred: torch.Tensor, opt_v1: torch.Tensor) -> torch.Tensor:
    """
    Edge-guard loss: penalize Sobel edge drift from Stage 1.
    Ensures geometry doesn't change, only spectral polish.
    """
    # Convert to grayscale
    pred_gray = (0.2989 * pred[:, 2] + 0.5870 * pred[:, 1] + 0.1140 * pred[:, 0]).unsqueeze(1)
    v1_gray = (0.2989 * opt_v1[:, 2] + 0.5870 * opt_v1[:, 1] + 0.1140 * opt_v1[:, 0]).unsqueeze(1)
    
    # Sobel edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    
    # Compute edges
    pred_edges = torch.sqrt(
        F.conv2d(pred_gray, sobel_x, padding=1)**2 + 
        F.conv2d(pred_gray, sobel_y, padding=1)**2 + 1e-8
    )
    v1_edges = torch.sqrt(
        F.conv2d(v1_gray, sobel_x, padding=1)**2 + 
        F.conv2d(v1_gray, sobel_y, padding=1)**2 + 1e-8
    )
    
    return F.l1_loss(pred_edges, v1_edges)
```

#### Add GAC-Score to validation (lines 592-645)
```python
# Compute GAC components for Stage 2
# Use edge preservation instead of SAR agreement
def compute_edge_preservation(pred, opt_v1):
    # Compute edges
    pred_gray = (0.2989 * pred[:, 2] + 0.5870 * pred[:, 1] + 0.1140 * pred[:, 0])
    v1_gray = (0.2989 * opt_v1[:, 2] + 0.5870 * opt_v1[:, 1] + 0.1140 * opt_v1[:, 0])
    
    # Normalize and compute correlation
    pred_edges = torch.gradient(pred_gray)
    v1_edges = torch.gradient(v1_gray)
    
    correlation = F.cosine_similarity(
        pred_edges.flatten(1),
        v1_edges.flatten(1),
        dim=1
    ).mean()
    
    return correlation.item()

edge_preservation = compute_edge_preservation(pred, opt_v1)
spectral_rmse = compute_spectral_rmse(pred, target)
lpips_score = lpips_loss(pred, target).mean().item()

# GAC-Score for Stage 2
gac_score = (
    0.5 * edge_preservation +
    0.3 * (1.0 - min(spectral_rmse / 0.5, 1.0)) +
    0.2 * (1.0 - min(lpips_score, 1.0))
)

result = {
    'loss': avg_loss,
    'edge_preservation': edge_preservation,
    'spectral_rmse': spectral_rmse,
    'lpips': lpips_score,
    'gac_score': gac_score,
    **component_losses
}
```

---

### Stage 3 Training Script (`scripts/train_stage3.py`)

#### Verify cycle/identity losses exist in `axs_lib/stage3_losses.py`
Read lines 200-600 to check for:
- `cycle_loss` component
- `identity_loss` component

#### Verify CLI flags for urban/DEM weights
Check for:
```python
parser.add_argument('--urban-weight', type=float, default=0.0,
                    help='Urban area SAR weighting boost')
parser.add_argument('--dem-weight', type=float, default=0.0,
                    help='DEM slope SAR weighting')
```

#### Add GAC-Score to validation
```python
# Compute GAC components
sar_agreement = compute_sar_agreement(opt_v3, s1)  # Already exists
spectral_rmse = compute_spectral_rmse(opt_v3, s2_truth if s2_truth else opt_v2)
lpips_score = lpips_loss(opt_v3, opt_v2).mean().item()

# Normalize and compute GAC-Score
sar_norm = sar_agreement
spectral_norm = 1.0 - min(spectral_rmse / 0.5, 1.0)
lpips_norm = 1.0 - min(lpips_score, 1.0)

gac_score = (
    0.5 * sar_norm +
    0.3 * spectral_norm +
    0.2 * lpips_norm
)

result = {
    'loss': avg_loss,
    'sar_agreement': sar_agreement,
    'spectral_rmse': spectral_rmse,
    'lpips': lpips_score,
    'gac_score': gac_score,
    **component_losses
}
```

---

### Export Scripts (NEW)

Create three scripts:

#### `scripts/export_stage1_best.py`
```python
"""Export best Stage 1 checkpoint by GAC-Score."""
import argparse
from pathlib import Path
import torch
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Directory containing checkpoints')
    parser.add_argument('--output', type=str, default='best_stage1_gac.pt',
                       help='Output filename')
    args = parser.parse_args()
    
    ckpt_dir = Path(args.checkpoint_dir)
    
    # Find all checkpoints
    ckpts = list(ckpt_dir.glob('*.pt'))
    
    if not ckpts:
        print(f"❌ No checkpoints found in {ckpt_dir}")
        return 1
    
    # Load and find best by GAC-Score
    best_ckpt = None
    best_gac = -1
    
    print(f"Evaluating {len(ckpts)} checkpoints...")
    for ckpt_path in ckpts:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            gac_score = ckpt.get('gac_score', -1)
            
            if gac_score > best_gac:
                best_gac = gac_score
                best_ckpt = ckpt_path
                
            print(f"  {ckpt_path.name}: GAC={gac_score:.4f}")
        except Exception as e:
            print(f"  ⚠️  Skipping {ckpt_path.name}: {e}")
    
    if best_ckpt:
        print(f"\n✓ Best checkpoint: {best_ckpt.name}")
        print(f"  GAC-Score: {best_gac:.4f}")
        
        # Copy to output
        shutil.copy(best_ckpt, args.output)
        print(f"✓ Exported to: {args.output}")
        return 0
    else:
        print("❌ No valid checkpoints with GAC-Score found")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
```

Create similar scripts for Stage 2 and Stage 3.

---

## Summary

### ✅ Already Done:
1. `compute_spectral_rmse()` added to `axs_lib/spectral.py`
2. `ColorConsistencyLoss` added to `axs_lib/losses.py`
3. `CombinedLoss` updated with color consistency

### 🔧 Need to Update (Estimated ~2 hours):
1. **Stage 1 training script** - Add color consistency flag, GAC-Score logging (~30 min)
2. **Stage 2 training script** - Verify edge-guard, add GAC-Score logging (~30 min)
3. **Stage 3 training script** - Verify components, add GAC-Score logging (~20 min)
4. **Export scripts** - Create 3 simple scripts (~30 min)
5. **Testing** - Quick smoke test (~10 min)

### Next Steps:
1. Run through each training script and apply the changes listed above
2. Test with a small dataset to verify GAC-Score is computed correctly
3. Create the export scripts
4. Update documentation

All infrastructure is in place - just needs final integration! 🎉
