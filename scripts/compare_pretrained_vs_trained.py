"""
Comprehensive comparison: Pretrained vs Trained models.
"""

import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def compute_metrics(gt, pred):
    """Compute basic metrics."""
    psnr = 20 * np.log10(1.0 / np.sqrt(np.mean((gt - pred) ** 2))) if np.mean((gt - pred) ** 2) > 0 else float('inf')
    mae = np.mean(np.abs(gt - pred))
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    return {'psnr': psnr, 'mae': mae, 'rmse': rmse}


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
    
    # Load pretrained results
    pretrained_path = Path("C:/Users/Dhenenjay/Axion-Sat/outputs/pretrained_all_fixed/outputs.npz")
    pretrained_data = np.load(pretrained_path)
    
    # Load trained results
    trained_path = Path("C:/Users/Dhenenjay/Axion-Sat/outputs/trained_models/outputs.npz")
    trained_data = np.load(trained_path)
    
    # Extract 4 bands from each stage
    def extract_bands(data):
        opt_v1 = data['opt_v1'][[1, 2, 3, 7], :, :].astype(np.float32)  # From 12 bands
        opt_v2 = data['opt_v2'][[0, 1, 2, 3], :, :].astype(np.float32)  # From 6 bands
        opt_v3 = data['opt_v3'][[0, 1, 2, 3], :, :].astype(np.float32)  # From 6 bands
        return opt_v1, opt_v2, opt_v3
    
    pre_v1, pre_v2, pre_v3 = extract_bands(pretrained_data)
    train_v1, train_v2, train_v3 = extract_bands(trained_data)
    
    print("=" * 100)
    print("COMPREHENSIVE COMPARISON: PRETRAINED vs TRAINED MODELS")
    print("=" * 100)
    print()
    
    print("IDEAL METRICS (Perfect Prediction):")
    print("  PSNR:       ∞ (infinite)")
    print("  MAE:        0.0000 (no error)")
    print("  RMSE:       0.0000 (no error)")
    print()
    print("=" * 100)
    
    # Stage 1 (same for both, uses pretrained TerraMind)
    print("\nSTAGE 1: TERRAMIND BASELINE (Pretrained - Same for Both)")
    print("-" * 100)
    metrics_v1 = compute_metrics(s2_gt, pre_v1)
    print(f"  PSNR:       {metrics_v1['psnr']:.2f} dB")
    print(f"  MAE:        {metrics_v1['mae']:.4f}  (ideal: 0.0000, difference: +{metrics_v1['mae']:.4f})")
    print(f"  RMSE:       {metrics_v1['rmse']:.4f}  (ideal: 0.0000, difference: +{metrics_v1['rmse']:.4f})")
    print(f"  Range:      [{pre_v1.min():.4f}, {pre_v1.max():.4f}]  (GT: [{s2_gt.min():.4f}, {s2_gt.max():.4f}])")
    
    # Stage 2 comparison
    print("\n" + "=" * 100)
    print("STAGE 2: PRITHVI REFINEMENT")
    print("=" * 100)
    
    pre_metrics_v2 = compute_metrics(s2_gt, pre_v2)
    train_metrics_v2 = compute_metrics(s2_gt, train_v2)
    
    print("\nPRETRAINED (No Fine-tuning):")
    print("-" * 100)
    print(f"  PSNR:       {pre_metrics_v2['psnr']:.2f} dB")
    print(f"  MAE:        {pre_metrics_v2['mae']:.4f}  (difference from ideal: +{pre_metrics_v2['mae']:.4f})")
    print(f"  RMSE:       {pre_metrics_v2['rmse']:.4f}  (difference from ideal: +{pre_metrics_v2['rmse']:.4f})")
    print(f"  Range:      [{pre_v2.min():.4f}, {pre_v2.max():.4f}]")
    
    print("\nTRAINED (Fine-tuned on SAR→Optical):")
    print("-" * 100)
    print(f"  PSNR:       {train_metrics_v2['psnr']:.2f} dB")
    print(f"  MAE:        {train_metrics_v2['mae']:.4f}  (difference from ideal: +{train_metrics_v2['mae']:.4f})")
    print(f"  RMSE:       {train_metrics_v2['rmse']:.4f}  (difference from ideal: +{train_metrics_v2['rmse']:.4f})")
    print(f"  Range:      [{train_v2.min():.4f}, {train_v2.max():.4f}]")
    
    print("\nIMPROVEMENT (Pretrained → Trained):")
    print("-" * 100)
    psnr_improvement = train_metrics_v2['psnr'] - pre_metrics_v2['psnr']
    mae_improvement = pre_metrics_v2['mae'] - train_metrics_v2['mae']
    rmse_improvement = pre_metrics_v2['rmse'] - train_metrics_v2['rmse']
    
    print(f"  PSNR:       {psnr_improvement:+.2f} dB  {'✓ Better' if psnr_improvement > 0 else '✗ Worse'}")
    print(f"  MAE:        {mae_improvement:+.4f}  {'✓ Better (lower error)' if mae_improvement > 0 else '✗ Worse (higher error)'}")
    print(f"  RMSE:       {rmse_improvement:+.4f}  {'✓ Better (lower error)' if rmse_improvement > 0 else '✗ Worse (higher error)'}")
    
    # Stage 3 comparison
    print("\n" + "=" * 100)
    print("STAGE 3: SAR-GROUNDED FINAL OUTPUT")
    print("=" * 100)
    
    pre_metrics_v3 = compute_metrics(s2_gt, pre_v3)
    train_metrics_v3 = compute_metrics(s2_gt, train_v3)
    
    print("\nPRETRAINED (Untrained Decoder):")
    print("-" * 100)
    print(f"  PSNR:       {pre_metrics_v3['psnr']:.2f} dB")
    print(f"  MAE:        {pre_metrics_v3['mae']:.4f}  (difference from ideal: +{pre_metrics_v3['mae']:.4f})")
    print(f"  RMSE:       {pre_metrics_v3['rmse']:.4f}  (difference from ideal: +{pre_metrics_v3['rmse']:.4f})")
    print(f"  Range:      [{pre_v3.min():.4f}, {pre_v3.max():.4f}]")
    
    print("\nTRAINED (Fine-tuned Decoder):")
    print("-" * 100)
    print(f"  PSNR:       {train_metrics_v3['psnr']:.2f} dB")
    print(f"  MAE:        {train_metrics_v3['mae']:.4f}  (difference from ideal: +{train_metrics_v3['mae']:.4f})")
    print(f"  RMSE:       {train_metrics_v3['rmse']:.4f}  (difference from ideal: +{train_metrics_v3['rmse']:.4f})")
    print(f"  Range:      [{train_v3.min():.4f}, {train_v3.max():.4f}]")
    
    print("\nIMPROVEMENT (Pretrained → Trained):")
    print("-" * 100)
    psnr_improvement = train_metrics_v3['psnr'] - pre_metrics_v3['psnr']
    mae_improvement = pre_metrics_v3['mae'] - train_metrics_v3['mae']
    rmse_improvement = pre_metrics_v3['rmse'] - train_metrics_v3['rmse']
    
    print(f"  PSNR:       {psnr_improvement:+.2f} dB  {'✓ Better' if psnr_improvement > 0 else '✗ Worse'}")
    print(f"  MAE:        {mae_improvement:+.4f}  {'✓ Better (lower error)' if mae_improvement > 0 else '✗ Worse (higher error)'}")
    print(f"  RMSE:       {rmse_improvement:+.4f}  {'✓ Better (lower error)' if rmse_improvement > 0 else '✗ Worse (higher error)'}")
    
    # Final summary
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    
    print("\nBEST PRETRAINED STAGE:")
    best_pre = min([
        (pre_metrics_v2['mae'], "Stage 2", pre_metrics_v2),
        (pre_metrics_v3['mae'], "Stage 3", pre_metrics_v3)
    ], key=lambda x: x[0])
    print(f"  {best_pre[1]}: MAE = {best_pre[2]['mae']:.4f}, PSNR = {best_pre[2]['psnr']:.2f} dB")
    
    print("\nBEST TRAINED STAGE:")
    best_train = min([
        (train_metrics_v2['mae'], "Stage 2", train_metrics_v2),
        (train_metrics_v3['mae'], "Stage 3", train_metrics_v3)
    ], key=lambda x: x[0])
    print(f"  {best_train[1]}: MAE = {best_train[2]['mae']:.4f}, PSNR = {best_train[2]['psnr']:.2f} dB")
    
    print("\nOVERALL BEST:")
    overall_best = min([
        (metrics_v1['mae'], "Stage 1 (Baseline)", metrics_v1),
        (train_metrics_v2['mae'], "Stage 2 (Trained)", train_metrics_v2),
        (train_metrics_v3['mae'], "Stage 3 (Trained)", train_metrics_v3)
    ], key=lambda x: x[0])
    print(f"  {overall_best[1]}: MAE = {overall_best[2]['mae']:.4f}, PSNR = {overall_best[2]['psnr']:.2f} dB")
    
    print("\n" + "=" * 100)


if __name__ == '__main__':
    main()
