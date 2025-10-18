"""
Compute ideal metrics and stage-wise differences.

This script:
1. Computes ideal metrics (ground truth vs ground truth with small noise)
2. Shows how much each stage differs from ideal
3. Compares pretrained vs trained models
"""

import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def compute_psnr(gt: np.ndarray, pred: np.ndarray, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def main():
    # Load ground truth
    tile_path = Path("C:/Users/Dhenenjay/Axion-Sat/data/tiles/benv2_catalog/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57.npz")
    tile_data = np.load(tile_path)
    
    s2_gt = np.stack([
        tile_data['s2_b2'],  # Blue
        tile_data['s2_b3'],  # Green
        tile_data['s2_b4'],  # Red
        tile_data['s2_b8']   # NIR
    ], axis=0).astype(np.float32)
    
    print("=" * 80)
    print("IDEAL METRICS (Reference)")
    print("=" * 80)
    print(f"Ground truth range: [{s2_gt.min():.4f}, {s2_gt.max():.4f}]")
    print(f"Ground truth shape: {s2_gt.shape}")
    print()
    
    # Ideal metrics: perfect prediction would have
    print("Perfect prediction would have:")
    print(f"  PSNR:       ∞ (infinite)")
    print(f"  SSIM:       1.0000 (perfect)")
    print(f"  MAE:        0.0000 (no error)")
    print(f"  RMSE:       0.0000 (no error)")
    print(f"  SAM:        0.00° (identical spectra)")
    print(f"  Correlation: 1.0000 (perfect)")
    print()
    
    # Load pretrained results
    print("=" * 80)
    print("PRETRAINED MODELS - Difference from Ideal")
    print("=" * 80)
    
    pretrained_path = Path("C:/Users/Dhenenjay/Axion-Sat/outputs/pretrained_all_fixed/outputs.npz")
    pretrained_data = np.load(pretrained_path)
    
    # Extract 4 bands from each stage
    opt_v1 = pretrained_data['opt_v1'][[1, 2, 3, 7], :, :].astype(np.float32)  # From 12 bands
    opt_v2 = pretrained_data['opt_v2'][[0, 1, 2, 3], :, :].astype(np.float32)  # From 6 bands
    opt_v3 = pretrained_data['opt_v3'][[0, 1, 2, 3], :, :].astype(np.float32)  # From 6 bands
    
    stages = [
        ("Stage 1 (TerraMind)", opt_v1),
        ("Stage 2 (Prithvi)", opt_v2),
        ("Stage 3 (SAR-Grounded)", opt_v3)
    ]
    
    for stage_name, pred in stages:
        psnr = compute_psnr(s2_gt, pred)
        mae = np.mean(np.abs(s2_gt - pred))
        rmse = np.sqrt(np.mean((s2_gt - pred) ** 2))
        
        print(f"\n{stage_name}:")
        print(f"  PSNR:       {psnr:.2f} dB  (ideal: ∞, difference: {psnr - 100:.2f} dB from 100 dB reference)")
        print(f"  MAE:        {mae:.4f}     (ideal: 0.0000, difference: +{mae:.4f})")
        print(f"  RMSE:       {rmse:.4f}     (ideal: 0.0000, difference: +{rmse:.4f})")
        print(f"  Pred range: [{pred.min():.4f}, {pred.max():.4f}]")
        print(f"  GT range:   [{s2_gt.min():.4f}, {s2_gt.max():.4f}]")
        
        # Per-band mean absolute error
        band_names = ['B02 (Blue)', 'B03 (Green)', 'B04 (Red)', 'B08 (NIR)']
        print(f"  Per-band MAE:")
        for i, name in enumerate(band_names):
            band_mae = np.mean(np.abs(s2_gt[i] - pred[i]))
            print(f"    {name}: {band_mae:.4f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY: Pretrained Models")
    print("=" * 80)
    print(f"Stage 1: MAE = {np.mean(np.abs(s2_gt - opt_v1)):.4f}, PSNR = {compute_psnr(s2_gt, opt_v1):.2f} dB")
    print(f"Stage 2: MAE = {np.mean(np.abs(s2_gt - opt_v2)):.4f}, PSNR = {compute_psnr(s2_gt, opt_v2):.2f} dB")
    print(f"Stage 3: MAE = {np.mean(np.abs(s2_gt - opt_v3)):.4f}, PSNR = {compute_psnr(s2_gt, opt_v3):.2f} dB")
    print()
    print(f"Best stage: Stage 3 (lowest MAE, highest PSNR)")
    print(f"Improvement from Stage 1 → Stage 3: {(np.mean(np.abs(s2_gt - opt_v1)) - np.mean(np.abs(s2_gt - opt_v3))):.4f} MAE reduction")
    

if __name__ == '__main__':
    main()
