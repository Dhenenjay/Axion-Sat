# GAC Training Specifications

This document specifies the exact GAC (Grounded Axion-Sat Composite) requirements for each training stage, ensuring consistency with the final architecture.

---

## Overview: GAC-Score

**GAC-Score** is a composite metric for model selection across all stages:

```python
GAC_Score = 0.5 * SAR_agreement + 0.3 * (1 - spectral_RMSE_normalized) + 0.2 * (1 - LPIPS_normalized)
```

Where:
- **SAR_agreement**: Edge correlation between optical and SAR (0-1, higher is better)
- **spectral_RMSE_normalized**: Normalized RMSE of spectral indices (NDVI, EVI, etc.), scaled to [0,1]
- **LPIPS_normalized**: LPIPS perceptual loss, scaled to [0,1]

All stages log these three components and compute GAC-Score for model selection.

---

## Stage 1: TerraMind SAR-to-Optical (S1 → opt_v1)

### Status
✅ **MOSTLY COMPLETE** - Minor updates needed

### Current State
The script `scripts/train_stage1.py` is already well-implemented with:
- LoRA on cross-modal attention and denoiser projections ✅
- AMP fp16 ✅
- Gradient accumulation ✅
- Batch size = 1 ✅
- Configurable timesteps ✅
- OOM auto-retry (reduces timesteps by 2) ✅
- SAR-edge agreement tracking ✅

### Required Updates

#### 1. Loss Function (axs_lib/losses.py - CombinedLoss)
**Current**: L1 + MS-SSIM + LPIPS + SAR-structure
**Required**: Diffusion + Recons + Color consistency (NO SAR-hard constraint)

```python
class Stage1Loss(nn.Module):
    """
    Stage 1 loss: Diffusion + Reconstruction + Color consistency
    NO SAR-hard constraint (that comes in Stage 3)
    """
    
    def __init__(
        self,
        recons_weight: float = 1.0,        # L1/L2 reconstruction
        color_weight: float = 0.5,          # Color consistency
        ms_ssim_weight: float = 0.5,        # MS-SSIM structural
        lpips_weight: float = 0.1           # Perceptual
    ):
        super().__init__()
        
        # Reconstruction losses
        self.l1_loss = nn.L1Loss()
        self.ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=4)
        self.lpips_loss = lpips.LPIPS(net='alex').eval()
        
        # Weights
        self.recons_weight = recons_weight
        self.color_weight = color_weight
        self.ms_ssim_weight = ms_ssim_weight
        self.lpips_weight = lpips_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sar: Optional[torch.Tensor] = None  # SAR for logging only, NOT in loss
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Stage 1 loss.
        
        Args:
            pred: Predicted optical (B, 4, H, W)
            target: Ground truth optical (B, 4, H, W)
            sar: SAR input (B, 2, H, W) - for logging only
            
        Returns:
            (total_loss, loss_dict)
        """
        # Reconstruction loss
        l1 = self.l1_loss(pred, target)
        ms_ssim_val = 1 - self.ms_ssim(pred, target)
        
        # LPIPS (needs 3-channel RGB)
        pred_rgb = pred[:, [2, 1, 0], :, :]  # R, G, B
        target_rgb = target[:, [2, 1, 0], :, :]
        lpips_val = self.lpips_loss(pred_rgb, target_rgb).mean()
        
        # Color consistency (RGB ratios preserved)
        color_loss = self._color_consistency_loss(pred, target)
        
        # Total loss
        total_loss = (
            self.recons_weight * l1 +
            self.ms_ssim_weight * ms_ssim_val +
            self.lpips_weight * lpips_val +
            self.color_weight * color_loss
        )
        
        loss_dict = {
            'l1': l1.item(),
            'ms_ssim': ms_ssim_val.item(),
            'lpips': lpips_val.item(),
            'color_consistency': color_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _color_consistency_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Color consistency: preserve RGB ratios.
        
        Computes relative differences in band ratios to ensure
        natural color relationships are maintained.
        """
        # Band ratios (avoid division by zero)
        eps = 1e-6
        
        # Red/Green ratio
        pred_rg = pred[:, 2, :, :] / (pred[:, 1, :, :] + eps)
        target_rg = target[:, 2, :, :] / (target[:, 1, :, :] + eps)
        
        # Red/Blue ratio
        pred_rb = pred[:, 2, :, :] / (pred[:, 0, :, :] + eps)
        target_rb = target[:, 2, :, :] / (target[:, 0, :, :] + eps)
        
        # NIR/Red ratio (vegetation)
        pred_nr = pred[:, 3, :, :] / (pred[:, 2, :, :] + eps)
        target_nr = target[:, 3, :, :] / (target[:, 2, :, :] + eps)
        
        # L1 on ratios
        loss = (
            F.l1_loss(pred_rg, target_rg) +
            F.l1_loss(pred_rb, target_rb) +
            F.l1_loss(pred_nr, target_nr)
        ) / 3.0
        
        return loss
```

#### 2. GAC-Score Logging
Add GAC-Score computation to validation:

```python
def validate(...):
    # ... existing validation code ...
    
    # Compute GAC components
    sar_agreement = compute_sar_edge_agreement(s2_pred, s1)  # Already exists
    spectral_rmse = compute_spectral_rmse(s2_pred, s2_target)  # NEW
    lpips_score = lpips_loss(s2_pred, s2_target).mean().item()  # Existing
    
    # Normalize to [0, 1] (higher is better)
    sar_agreement_norm = sar_agreement  # Already in [0, 1]
    spectral_rmse_norm = 1.0 - min(spectral_rmse / 0.5, 1.0)  # Assume max RMSE = 0.5
    lpips_norm = 1.0 - min(lpips_score, 1.0)  # LPIPS in [0, 1+]
    
    # Compute GAC-Score
    gac_score = (
        0.5 * sar_agreement_norm +
        0.3 * spectral_rmse_norm +
        0.2 * lpips_norm
    )
    
    result = {
        'loss': avg_loss,
        'sar_agreement': sar_agreement,
        'spectral_rmse': spectral_rmse,
        'lpips': lpips_score,
        'gac_score': gac_score,  # NEW: composite metric
        **component_losses
    }
    
    return result
```

#### 3. Checkpoint Management
Update to track GAC-Score:

```python
# Save best model by GAC-Score (not just val_loss)
if val_metrics['gac_score'] > best_gac_score:
    best_gac_score = val_metrics['gac_score']
    if output_dir is not None:
        best_path = output_dir / 'best_model_gac.pt'
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'step': step,
            'val_loss': val_metrics['loss'],
            'gac_score': val_metrics['gac_score'],
            'sar_agreement': val_metrics['sar_agreement'],
            'spectral_rmse': val_metrics['spectral_rmse'],
            'lpips': val_metrics['lpips']
        }
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best GAC model (score: {best_gac_score:.4f})")
```

---

## Stage 2: Prithvi Refinement (opt_v1 → opt_v2)

### Status
🔧 **NEEDS SIGNIFICANT UPDATES**

### Current State
The script `scripts/train_stage2.py` has basic structure but needs GAC-specific components.

### Required Architecture

#### 1. Model Configuration
```python
# Prithvi-EO-2.0-600M with 8-bit/4-bit quantization
prithvi_refiner = build_prithvi_refiner(
    model_name='ibm-nasa-geospatial/Prithvi-EO-2.0-600M',
    load_in_8bit=True,  # Or load_in_4bit=True for extreme low VRAM
    lora_config={
        'r': 16,  # LoRA rank
        'lora_alpha': 32,
        'target_modules': ['transformer.layers.-4:', 'transformer.layers.-3:', 
                          'transformer.layers.-2:', 'transformer.layers.-1:'],  # Last 4 blocks
        'lora_dropout': 0.05
    },
    refiner_head_config={
        'type': 'convnext_nano',  # Shallow ConvNeXt head
        'in_channels': 768,  # Prithvi hidden size
        'out_channels': 4,  # BGRNIR
        'num_blocks': 2
    }
)
```

#### 2. Loss Function (NEW: axs_lib/stage2_losses.py - Spectral + Edge-Guard)
```python
class Stage2Loss(nn.Module):
    """
    Stage 2 loss: Spectral plausibility + LPIPS/SSIM + Edge-guard
    
    Focus on spectral/texture polish, NO SAR input here.
    """
    
    def __init__(
        self,
        spectral_weight: float = 1.0,
        edge_guard_weight: float = 0.5,
        lpips_weight: float = 0.3,
        ssim_weight: float = 0.2
    ):
        super().__init__()
        
        self.spectral_weight = spectral_weight
        self.edge_guard_weight = edge_guard_weight
        self.lpips_weight = lpips_weight
        self.ssim_weight = ssim_weight
        
        self.lpips_loss = lpips.LPIPS(net='alex').eval()
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=4)
    
    def forward(
        self,
        pred: torch.Tensor,      # opt_v2 (B, 4, H, W)
        target: torch.Tensor,    # ground truth S2 (B, 4, H, W)
        opt_v1: torch.Tensor,    # Stage 1 input (B, 4, H, W)
        mode: str = 'generator'
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Stage 2 loss.
        
        Args:
            pred: Refined optical v2 (B, 4, H, W)
            target: Ground truth optical (B, 4, H, W)
            opt_v1: Stage 1 input (for edge-guard) (B, 4, H, W)
            mode: 'generator' or 'discriminator'
        """
        # 1. Spectral plausibility
        spectral_loss = self._spectral_plausibility_loss(pred, target)
        
        # 2. Edge-guard: penalize geometry drift from v1
        edge_guard_loss = self._edge_guard_loss(pred, opt_v1)
        
        # 3. LPIPS perceptual loss
        pred_rgb = pred[:, [2, 1, 0], :, :]
        target_rgb = target[:, [2, 1, 0], :, :]
        lpips_val = self.lpips_loss(pred_rgb, target_rgb).mean()
        
        # 4. SSIM structural similarity
        ssim_val = 1 - self.ssim(pred, target)
        
        # Total loss
        total_loss = (
            self.spectral_weight * spectral_loss +
            self.edge_guard_weight * edge_guard_loss +
            self.lpips_weight * lpips_val +
            self.ssim_weight * ssim_val
        )
        
        loss_dict = {
            'spectral_loss': spectral_loss.item(),
            'edge_guard_loss': edge_guard_loss.item(),
            'lpips': lpips_val.item(),
            'ssim': ssim_val.item()
        }
        
        return total_loss, loss_dict
    
    def _spectral_plausibility_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Spectral plausibility: NDVI/EVI RMSE + SAM (Spectral Angle Mapper).
        
        Ensures vegetation indices and spectral angles are accurate.
        """
        # Extract bands (B, G, R, NIR)
        pred_r = pred[:, 2, :, :]
        pred_nir = pred[:, 3, :, :]
        target_r = target[:, 2, :, :]
        target_nir = target[:, 3, :, :]
        
        eps = 1e-6
        
        # NDVI = (NIR - Red) / (NIR + Red)
        pred_ndvi = (pred_nir - pred_r) / (pred_nir + pred_r + eps)
        target_ndvi = (target_nir - target_r) / (target_nir + target_r + eps)
        ndvi_rmse = torch.sqrt(F.mse_loss(pred_ndvi, target_ndvi))
        
        # EVI (simplified) = 2.5 * (NIR - Red) / (NIR + 6*Red + 1)
        pred_evi = 2.5 * (pred_nir - pred_r) / (pred_nir + 6 * pred_r + 1 + eps)
        target_evi = 2.5 * (target_nir - target_r) / (target_nir + 6 * target_r + 1 + eps)
        evi_rmse = torch.sqrt(F.mse_loss(pred_evi, target_evi))
        
        # Spectral Angle Mapper (SAM)
        # Measures angle between spectral vectors
        pred_flat = pred.flatten(2)  # (B, 4, H*W)
        target_flat = target.flatten(2)
        
        # Normalize to unit vectors
        pred_norm = F.normalize(pred_flat, p=2, dim=1)
        target_norm = F.normalize(target_flat, p=2, dim=1)
        
        # Cosine similarity
        cos_sim = (pred_norm * target_norm).sum(dim=1)  # (B, H*W)
        cos_sim = torch.clamp(cos_sim, -1, 1)
        
        # Spectral angle in radians
        angles = torch.acos(cos_sim)
        sam = angles.mean()
        
        # Combine
        spectral_loss = ndvi_rmse + evi_rmse + sam
        
        return spectral_loss
    
    def _edge_guard_loss(
        self,
        pred: torch.Tensor,
        opt_v1: torch.Tensor
    ) -> torch.Tensor:
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
        
        # Edges for pred
        pred_edges_x = F.conv2d(pred_gray, sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred_gray, sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edges_x ** 2 + pred_edges_y ** 2 + 1e-8)
        
        # Edges for v1
        v1_edges_x = F.conv2d(v1_gray, sobel_x, padding=1)
        v1_edges_y = F.conv2d(v1_gray, sobel_y, padding=1)
        v1_edges = torch.sqrt(v1_edges_x ** 2 + v1_edges_y ** 2 + 1e-8)
        
        # L1 difference
        edge_guard = F.l1_loss(pred_edges, v1_edges)
        
        return edge_guard
```

#### 3. GAC-Score Computation
```python
def validate(...):
    # ... existing validation ...
    
    # Compute GAC components
    # For Stage 2, SAR agreement is not directly measurable (no SAR input)
    # Instead, use edge preservation from v1 as a proxy
    edge_preservation = compute_edge_preservation(pred, opt_v1)  # NEW
    spectral_rmse = compute_spectral_rmse(pred, target)
    lpips_score = lpips_loss(pred, target).mean().item()
    
    # Normalize
    edge_pres_norm = edge_preservation  # In [0, 1]
    spectral_rmse_norm = 1.0 - min(spectral_rmse / 0.5, 1.0)
    lpips_norm = 1.0 - min(lpips_score, 1.0)
    
    # GAC-Score for Stage 2 (use edge preservation instead of SAR agreement)
    gac_score = (
        0.5 * edge_pres_norm +  # Edge preservation from v1
        0.3 * spectral_rmse_norm +
        0.2 * lpips_norm
    )
    
    result = {
        'loss': avg_loss,
        'edge_preservation': edge_preservation,
        'spectral_rmse': spectral_rmse,
        'lpips': lpips_score,
        'gac_score': gac_score,
        **component_losses
    }
    
    return result
```

---

## Stage 3: SAR Grounding (S1 + opt_v2 → opt_v3)

### Status
🔧 **NEEDS GAC-SPECIFIC UPDATES**

### Current State
The script `scripts/train_stage3.py` has good structure but needs GAC specifics.

### Required Updates

#### 1. Loss Function (Update axs_lib/stage3_losses.py)
Already exists but ensure it includes:

```python
class Stage3Loss(nn.Module):
    """
    Stage 3 loss: SAR-consistency + Cycle/Identity + LPIPS + Spectral
    """
    
    def __init__(
        self,
        sar_consistency_weight: float = 1.0,
        cycle_weight: float = 0.5,
        identity_weight: float = 0.3,
        lpips_weight: float = 0.2,
        spectral_weight: float = 0.2,
        urban_weight: float = 0.0,  # EXISTING: urban boost
        dem_weight: float = 0.0     # EXISTING: DEM-aware weighting
    ):
        super().__init__()
        
        self.sar_consistency_weight = sar_consistency_weight
        self.cycle_weight = cycle_weight
        self.identity_weight = identity_weight
        self.lpips_weight = lpips_weight
        self.spectral_weight = spectral_weight
        
        # SAR consistency loss with urban/DEM weighting
        self.sar_consistency_loss = SARConsistencyLoss(
            edge_weight=0.6,
            texture_weight=0.4,
            urban_weight=urban_weight,
            dem_weight=dem_weight
        )
        
        self.cycle_loss = nn.L1Loss()
        self.identity_loss = nn.L1Loss()
        self.lpips_loss = lpips.LPIPS(net='alex').eval()
    
    def forward(
        self,
        opt_v3: torch.Tensor,    # Final output (B, 4, H, W)
        opt_v2: torch.Tensor,    # Stage 2 input (B, 4, H, W)
        s1: torch.Tensor,        # SAR input (B, 2, H, W)
        s2_truth: Optional[torch.Tensor] = None,  # Optional ground truth
        sar_backscatter: Optional[torch.Tensor] = None,  # For urban weighting
        dem: Optional[torch.Tensor] = None  # For DEM weighting
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Stage 3 loss.
        """
        # 1. SAR consistency (edge/phase congruency)
        sar_loss = self.sar_consistency_loss(
            opt_v3, s1,
            backscatter=sar_backscatter,
            dem=dem
        )
        
        # 2. Cycle consistency (v3 should be close to v2 when SAR is consistent)
        cycle_loss = self.cycle_loss(opt_v3, opt_v2)
        
        # 3. Identity preservation (spectral features from v2)
        identity_loss = self._identity_preservation_loss(opt_v3, opt_v2)
        
        # 4. LPIPS perceptual
        pred_rgb = opt_v3[:, [2, 1, 0], :, :]
        v2_rgb = opt_v2[:, [2, 1, 0], :, :]
        lpips_val = self.lpips_loss(pred_rgb, v2_rgb).mean()
        
        # 5. Spectral preservation
        spectral_loss = self._spectral_preservation_loss(opt_v3, opt_v2)
        
        # Total loss
        total_loss = (
            self.sar_consistency_weight * sar_loss +
            self.cycle_weight * cycle_loss +
            self.identity_weight * identity_loss +
            self.lpips_weight * lpips_val +
            self.spectral_weight * spectral_loss
        )
        
        loss_dict = {
            'sar_consistency': sar_loss.item(),
            'cycle': cycle_loss.item(),
            'identity': identity_loss.item(),
            'lpips': lpips_val.item(),
            'spectral': spectral_loss.item()
        }
        
        return total_loss, loss_dict
```

#### 2. CLI Flags
```python
parser.add_argument('--urban-weight', type=float, default=0.0,
                    help='Urban area SAR weighting boost (default: 0.0)')
parser.add_argument('--dem-weight', type=float, default=0.0,
                    help='DEM slope SAR weighting (default: 0.0)')
```

#### 3. GAC-Score Computation
```python
def validate(...):
    # ... existing validation ...
    
    # Compute GAC components
    sar_agreement = compute_sar_edge_agreement(opt_v3, s1)  # Existing
    spectral_rmse = compute_spectral_rmse(opt_v3, s2_truth if s2_truth else opt_v2)
    lpips_score = lpips_loss(opt_v3, opt_v2).mean().item()
    
    # Normalize
    sar_agreement_norm = sar_agreement
    spectral_rmse_norm = 1.0 - min(spectral_rmse / 0.5, 1.0)
    lpips_norm = 1.0 - min(lpips_score, 1.0)
    
    # GAC-Score
    gac_score = (
        0.5 * sar_agreement_norm +
        0.3 * spectral_rmse_norm +
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
    
    return result
```

---

## Common Utilities

### 1. Spectral RMSE Computation
Add to `axs_lib/metrics.py`:

```python
def compute_spectral_rmse(
    pred: torch.Tensor,
    target: torch.Tensor
) -> float:
    """
    Compute RMSE of spectral indices (NDVI, EVI).
    
    Args:
        pred: Predicted optical (B, 4, H, W) - channels [B, G, R, NIR]
        target: Target optical (B, 4, H, W)
        
    Returns:
        Average RMSE of spectral indices
    """
    eps = 1e-6
    
    # Extract Red and NIR
    pred_r = pred[:, 2, :, :]
    pred_nir = pred[:, 3, :, :]
    target_r = target[:, 2, :, :]
    target_nir = target[:, 3, :, :]
    
    # NDVI
    pred_ndvi = (pred_nir - pred_r) / (pred_nir + pred_r + eps)
    target_ndvi = (target_nir - target_r) / (target_nir + target_r + eps)
    ndvi_rmse = torch.sqrt(F.mse_loss(pred_ndvi, target_ndvi)).item()
    
    # EVI
    pred_evi = 2.5 * (pred_nir - pred_r) / (pred_nir + 6 * pred_r + 1 + eps)
    target_evi = 2.5 * (target_nir - target_r) / (target_nir + 6 * target_r + 1 + eps)
    evi_rmse = torch.sqrt(F.mse_loss(pred_evi, target_evi)).item()
    
    # Average
    return (ndvi_rmse + evi_rmse) / 2.0
```

### 2. GAC-Score Computation
Add to `axs_lib/metrics.py`:

```python
def compute_gac_score(
    sar_agreement: float,
    spectral_rmse: float,
    lpips: float,
    spectral_rmse_max: float = 0.5,
    lpips_max: float = 1.0
) -> float:
    """
    Compute GAC-Score: composite metric for model selection.
    
    Args:
        sar_agreement: SAR-optical edge agreement [0, 1] (higher is better)
        spectral_rmse: Spectral index RMSE (lower is better)
        lpips: LPIPS perceptual loss (lower is better)
        spectral_rmse_max: Maximum expected RMSE (for normalization)
        lpips_max: Maximum expected LPIPS (for normalization)
        
    Returns:
        GAC-Score [0, 1] (higher is better)
    """
    # Normalize to [0, 1] (higher is better)
    sar_norm = sar_agreement  # Already in [0, 1]
    spectral_norm = 1.0 - min(spectral_rmse / spectral_rmse_max, 1.0)
    lpips_norm = 1.0 - min(lpips / lpips_max, 1.0)
    
    # Weighted combination
    gac_score = (
        0.5 * sar_norm +
        0.3 * spectral_norm +
        0.2 * lpips_norm
    )
    
    return gac_score
```

---

## Export Scripts

Create three export scripts that select the best checkpoint by GAC-Score:

### `scripts/export_stage1_best.py`
```python
"""
Export best Stage 1 checkpoint by GAC-Score.

Usage:
    python scripts/export_stage1_best.py \
        --checkpoint-dir checkpoints/stage1/top_k/ \
        --output best_stage1_gac.pt
"""

import argparse
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='best_stage1_gac.pt')
    args = parser.parse_args()
    
    ckpt_dir = Path(args.checkpoint_dir)
    
    # Find all checkpoints
    ckpts = list(ckpt_dir.glob('best_*.pt'))
    
    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir}")
        return
    
    # Load and find best by GAC-Score
    best_ckpt = None
    best_gac = -1
    
    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        gac_score = ckpt.get('gac_score', -1)
        
        if gac_score > best_gac:
            best_gac = gac_score
            best_ckpt = ckpt_path
    
    if best_ckpt:
        print(f"Best checkpoint: {best_ckpt.name}")
        print(f"  GAC-Score: {best_gac:.4f}")
        
        # Copy to output
        import shutil
        shutil.copy(best_ckpt, args.output)
        print(f"✓ Exported to: {args.output}")
    else:
        print("No valid checkpoints with GAC-Score found")

if __name__ == '__main__':
    main()
```

Create similar scripts for `export_stage2_best.py` and `export_stage3_best.py`.

---

## Summary of Changes

### Stage 1 (train_stage1.py)
- ✅ LoRA on cross-modal + denoiser: **Already implemented**
- ✅ AMP fp16, grad_accum, batch=1: **Already implemented**
- ✅ Configurable timesteps: **Already implemented**
- ✅ OOM auto-retry: **Already implemented**
- 🔧 **Update loss**: Remove SAR-structure, add color consistency
- 🔧 **Add**: Spectral RMSE computation
- 🔧 **Add**: GAC-Score logging and best model selection

### Stage 2 (train_stage2.py)
- ✅ Prithvi-EO-2.0-600M + LoRA: **Already implemented**
- ✅ 8-bit quantization: **Already implemented**
- ✅ AMP: **Already implemented**
- 🔧 **Update loss**: Add spectral plausibility (NDVI/EVI RMSE, SAM)
- 🔧 **Add**: Edge-guard loss (Sobel difference vs v1)
- 🔧 **Add**: GAC-Score logging (edge preservation + spectral + LPIPS)

### Stage 3 (train_stage3.py)
- ✅ LoRA on cross-modal: **Already implemented**
- ✅ SAR-consistency loss: **Already implemented**
- ✅ Urban/DEM weighting: **Already implemented**
- ✅ AMP, grad_accum, OOM retry: **Already implemented**
- 🔧 **Verify**: Cycle + Identity losses present
- 🔧 **Add**: GAC-Score logging
- 🔧 **Verify**: CLI flags for urban-weight and dem-weight

### New Files
1. `axs_lib/metrics.py` - Add `compute_spectral_rmse()` and `compute_gac_score()`
2. `scripts/export_stage1_best.py` - Export best Stage 1 by GAC-Score
3. `scripts/export_stage2_best.py` - Export best Stage 2 by GAC-Score
4. `scripts/export_stage3_best.py` - Export best Stage 3 by GAC-Score

---

## Implementation Priority

1. **HIGH**: Add spectral RMSE and GAC-Score utilities to `axs_lib/metrics.py`
2. **HIGH**: Update Stage 1 loss (remove SAR-structure, add color consistency)
3. **HIGH**: Update Stage 2 loss (add spectral plausibility + edge-guard)
4. **MEDIUM**: Add GAC-Score logging to all three stages
5. **MEDIUM**: Create export scripts for best model selection
6. **LOW**: Add documentation and examples

All three stages should log these metrics at validation time:
- `sar_agreement` (or `edge_preservation` for Stage 2)
- `spectral_rmse`
- `lpips`
- `gac_score` (composite)

The best models should be saved based on `gac_score`, not just `val_loss`.
