"""
Compare GAC pipeline outputs with ground truth Sentinel-2 optical data.

This script computes quantitative metrics comparing the predicted optical
imagery from the GAC pipeline with the actual Sentinel-2 observations.

Metrics computed:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- SAM (Spectral Angle Mapper)
- Per-band correlation
- NDVI accuracy

Usage:
    python scripts/compare_with_ground_truth.py \
        --tile-path data/tiles/benv2_catalog/tile.npz \
        --pipeline-output outputs/real_tile_final/outputs.npz \
        --output-dir outputs/comparison
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import sys

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_data(tile_path: Path, pipeline_output: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load ground truth and predictions.
    
    Handles band extraction:
    - Ground truth: 4 bands (B02, B03, B04, B08)
    - Stage 1: 12 bands (full S2L2A) → extract 4 bands
    - Stage 2: 6 bands (HLS) → extract 4 bands
    - Stage 3: 6 bands (HLS) → extract 4 bands
    """
    # Load original tile (ground truth)
    tile_data = np.load(tile_path)
    s2_gt = np.stack([
        tile_data['s2_b2'],  # Blue
        tile_data['s2_b3'],  # Green
        tile_data['s2_b4'],  # Red
        tile_data['s2_b8']   # NIR
    ], axis=0)  # Shape: (4, 120, 120)
    
    # Load pipeline outputs
    pipeline_data = np.load(pipeline_output)
    
    # Stage 1: Extract 4 matching bands from 12 S2L2A bands
    # S2L2A order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
    # We want indices [1, 2, 3, 7] for B02, B03, B04, B08
    opt_v1_full = pipeline_data['opt_v1']  # (12, H, W)
    if opt_v1_full.shape[0] == 12:
        opt_v1 = opt_v1_full[[1, 2, 3, 7], :, :]  # Extract B02, B03, B04, B08
    else:
        opt_v1 = opt_v1_full[:4, :, :]  # Fallback
    
    # Stage 2: Extract 4 matching bands from 6 HLS bands
    # HLS order: Blue(B02), Green(B03), Red(B04), NIR(B08), SWIR1(B11), SWIR2(B12)
    # We want indices [0, 1, 2, 3] for B02, B03, B04, B08
    opt_v2_full = pipeline_data['opt_v2']  # (6, H, W)
    if opt_v2_full.shape[0] == 6:
        opt_v2 = opt_v2_full[[0, 1, 2, 3], :, :]  # Extract first 4 bands
    else:
        opt_v2 = opt_v2_full[:4, :, :]  # Fallback
    
    # Stage 3: Extract 4 matching bands from 6 HLS bands (same as Stage 2)
    opt_v3_full = pipeline_data['opt_v3']  # (6, H, W)
    if opt_v3_full.shape[0] == 6:
        opt_v3 = opt_v3_full[[0, 1, 2, 3], :, :]  # Extract first 4 bands
    else:
        opt_v3 = opt_v3_full[:4, :, :]  # Fallback
    
    print(f"  Ground truth range before normalization: [{s2_gt.min():.2f}, {s2_gt.max():.2f}]")
    
    # Normalize ground truth if needed (check if already normalized)
    if s2_gt.max() > 1.5:
        print(f"  Normalizing ground truth from DN to [0,1] range...")
        s2_gt = np.clip(s2_gt / 10000.0, 0, 1)
    else:
        print(f"  Ground truth already in [0, 1] range")
    
    print(f"  Predictions range: v1=[{opt_v1.min():.4f}, {opt_v1.max():.4f}], "
          f"v2=[{opt_v2.min():.4f}, {opt_v2.max():.4f}], "
          f"v3=[{opt_v3.min():.4f}, {opt_v3.max():.4f}]")
    
    # Convert all arrays to float32 for compatibility with scipy functions
    s2_gt = s2_gt.astype(np.float32)
    opt_v1 = opt_v1.astype(np.float32)
    opt_v2 = opt_v2.astype(np.float32)
    opt_v3 = opt_v3.astype(np.float32)
    
    return s2_gt, opt_v1, opt_v2, opt_v3


def compute_psnr(gt: np.ndarray, pred: np.ndarray, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim(gt: np.ndarray, pred: np.ndarray, window_size: int = 11) -> float:
    """Compute Structural Similarity Index (simplified version)."""
    from scipy.ndimage import uniform_filter
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = uniform_filter(gt, window_size)
    mu2 = uniform_filter(pred, window_size)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = uniform_filter(gt ** 2, window_size) - mu1_sq
    sigma2_sq = uniform_filter(pred ** 2, window_size) - mu2_sq
    sigma12 = uniform_filter(gt * pred, window_size) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)


def compute_sam(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Spectral Angle Mapper (in degrees)."""
    # Flatten spatial dimensions
    gt_flat = gt.reshape(gt.shape[0], -1)  # (C, H*W)
    pred_flat = pred.reshape(pred.shape[0], -1)  # (C, H*W)
    
    # Compute spectral angle for each pixel
    dot_product = np.sum(gt_flat * pred_flat, axis=0)
    norm_gt = np.linalg.norm(gt_flat, axis=0)
    norm_pred = np.linalg.norm(pred_flat, axis=0)
    
    # Avoid division by zero
    valid = (norm_gt > 0) & (norm_pred > 0)
    cos_angle = np.zeros_like(dot_product)
    cos_angle[valid] = dot_product[valid] / (norm_gt[valid] * norm_pred[valid])
    
    # Clip to [-1, 1] to avoid numerical issues
    cos_angle = np.clip(cos_angle, -1, 1)
    
    # Convert to degrees
    angles = np.arccos(cos_angle) * 180 / np.pi
    
    return np.mean(angles)


def compute_ndvi(bands: np.ndarray) -> np.ndarray:
    """Compute NDVI from bands (B, G, R, NIR)."""
    nir = bands[3]  # B08
    red = bands[2]  # B04
    
    # Avoid division by zero
    denominator = nir + red
    ndvi = np.zeros_like(nir)
    valid = denominator > 1e-6
    ndvi[valid] = (nir[valid] - red[valid]) / denominator[valid]
    
    return ndvi


def compute_metrics(gt: np.ndarray, pred: np.ndarray, stage_name: str) -> Dict[str, float]:
    """Compute all metrics for a prediction."""
    metrics = {
        'stage': stage_name,
        'psnr': compute_psnr(gt, pred),
        'mae': np.mean(np.abs(gt - pred)),
        'rmse': np.sqrt(np.mean((gt - pred) ** 2)),
        'sam': compute_sam(gt, pred),
    }
    
    # Per-band SSIM
    band_names = ['B02 (Blue)', 'B03 (Green)', 'B04 (Red)', 'B08 (NIR)']
    ssim_scores = []
    for i in range(gt.shape[0]):
        ssim = compute_ssim(gt[i], pred[i])
        ssim_scores.append(ssim)
        metrics[f'ssim_{band_names[i]}'] = ssim
    
    metrics['ssim_mean'] = np.mean(ssim_scores)
    
    # Per-band correlation
    for i, name in enumerate(band_names):
        gt_flat = gt[i].flatten()
        pred_flat = pred[i].flatten()
        corr = np.corrcoef(gt_flat, pred_flat)[0, 1]
        metrics[f'corr_{name}'] = corr
    
    # NDVI accuracy
    ndvi_gt = compute_ndvi(gt)
    ndvi_pred = compute_ndvi(pred)
    metrics['ndvi_mae'] = np.mean(np.abs(ndvi_gt - ndvi_pred))
    metrics['ndvi_rmse'] = np.sqrt(np.mean((ndvi_gt - ndvi_pred) ** 2))
    
    return metrics


def create_comparison_plots(gt: np.ndarray, opt_v1: np.ndarray, opt_v2: np.ndarray, 
                            opt_v3: np.ndarray, output_dir: Path):
    """Create comprehensive comparison visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. RGB Comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Ground truth RGB
    gt_rgb = np.transpose(gt[[2, 1, 0], :, :], (1, 2, 0))  # RGB
    gt_rgb = np.clip(gt_rgb, 0, 1)
    axes[0, 0].imshow(gt_rgb)
    axes[0, 0].set_title('Ground Truth S2', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Stage 1
    v1_rgb = np.transpose(opt_v1[[2, 1, 0], :, :], (1, 2, 0))
    v1_rgb = np.clip(v1_rgb, 0, 1)
    axes[0, 1].imshow(v1_rgb)
    axes[0, 1].set_title('Stage 1: TerraMind', fontsize=14)
    axes[0, 1].axis('off')
    
    # Stage 2
    v2_rgb = np.transpose(opt_v2[[2, 1, 0], :, :], (1, 2, 0))
    v2_rgb = np.clip(v2_rgb, 0, 1)
    axes[0, 2].imshow(v2_rgb)
    axes[0, 2].set_title('Stage 2: Prithvi Refined', fontsize=14)
    axes[0, 2].axis('off')
    
    # Stage 3
    v3_rgb = np.transpose(opt_v3[[2, 1, 0], :, :], (1, 2, 0))
    v3_rgb = np.clip(v3_rgb, 0, 1)
    axes[1, 0].imshow(v3_rgb)
    axes[1, 0].set_title('Stage 3: SAR-Grounded', fontsize=14)
    axes[1, 0].axis('off')
    
    # Error maps (Stage 2)
    error_v2 = np.mean(np.abs(gt - opt_v2), axis=0)
    im = axes[1, 1].imshow(error_v2, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 1].set_title('Stage 2 Error Map (MAE)', fontsize=14)
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    
    # Error maps (Stage 3)
    error_v3 = np.mean(np.abs(gt - opt_v3), axis=0)
    im = axes[1, 2].imshow(error_v3, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 2].set_title('Stage 3 Error Map (MAE)', fontsize=14)
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rgb_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. NDVI Comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    ndvi_gt = compute_ndvi(gt)
    ndvi_v1 = compute_ndvi(opt_v1)
    ndvi_v2 = compute_ndvi(opt_v2)
    ndvi_v3 = compute_ndvi(opt_v3)
    
    im = axes[0, 0].imshow(ndvi_gt, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[0, 0].set_title('NDVI: Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
    
    im = axes[0, 1].imshow(ndvi_v2, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[0, 1].set_title('NDVI: Stage 2', fontsize=14)
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
    
    im = axes[1, 0].imshow(ndvi_v3, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1, 0].set_title('NDVI: Stage 3', fontsize=14)
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    
    # NDVI error
    ndvi_error = np.abs(ndvi_gt - ndvi_v3)
    im = axes[1, 1].imshow(ndvi_error, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 1].set_title('NDVI Error (Stage 3)', fontsize=14)
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ndvi_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare GAC pipeline with ground truth")
    parser.add_argument('--tile-path', type=str, required=True,
                       help='Path to original tile NPZ with ground truth')
    parser.add_argument('--pipeline-output', type=str, required=True,
                       help='Path to pipeline output NPZ')
    parser.add_argument('--output-dir', type=str, default='outputs/comparison',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    tile_path = Path(args.tile_path)
    pipeline_output = Path(args.pipeline_output)
    output_dir = Path(args.output_dir)
    
    print("=" * 80)
    print("GAC Pipeline Accuracy Evaluation")
    print("=" * 80)
    print(f"Ground truth: {tile_path}")
    print(f"Predictions:  {pipeline_output}")
    print()
    
    # Load data
    print("Loading data...")
    s2_gt, opt_v1, opt_v2, opt_v3 = load_data(tile_path, pipeline_output)
    print(f"  Ground truth shape: {s2_gt.shape}")
    print(f"  Predictions shape:  {opt_v2.shape}")
    print()
    
    # Compute metrics for each stage
    print("Computing metrics...")
    metrics_v1 = compute_metrics(s2_gt, opt_v1, "Stage 1 (TerraMind)")
    metrics_v2 = compute_metrics(s2_gt, opt_v2, "Stage 2 (Prithvi Refined)")
    metrics_v3 = compute_metrics(s2_gt, opt_v3, "Stage 3 (SAR-Grounded)")
    
    # Print results
    print("\n" + "=" * 80)
    print("QUANTITATIVE METRICS")
    print("=" * 80)
    
    for metrics in [metrics_v1, metrics_v2, metrics_v3]:
        print(f"\n{metrics['stage']}")
        print("-" * 80)
        print(f"  PSNR:       {metrics['psnr']:.2f} dB")
        print(f"  SSIM:       {metrics['ssim_mean']:.4f}")
        print(f"  MAE:        {metrics['mae']:.4f}")
        print(f"  RMSE:       {metrics['rmse']:.4f}")
        print(f"  SAM:        {metrics['sam']:.2f}°")
        print(f"  NDVI MAE:   {metrics['ndvi_mae']:.4f}")
        print(f"  NDVI RMSE:  {metrics['ndvi_rmse']:.4f}")
        print()
        print("  Per-band correlation:")
        for key in ['corr_B02 (Blue)', 'corr_B03 (Green)', 'corr_B04 (Red)', 'corr_B08 (NIR)']:
            if key in metrics:
                print(f"    {key}: {metrics[key]:.4f}")
    
    # Save metrics to file
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write("GAC Pipeline Accuracy Evaluation\n")
        f.write("=" * 80 + "\n\n")
        for metrics in [metrics_v1, metrics_v2, metrics_v3]:
            f.write(f"{metrics['stage']}\n")
            f.write("-" * 80 + "\n")
            for key, value in metrics.items():
                if key != 'stage':
                    f.write(f"{key}: {value}\n")
            f.write("\n")
    
    print(f"\n✓ Metrics saved to: {output_dir / 'metrics.txt'}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_comparison_plots(s2_gt, opt_v1, opt_v2, opt_v3, output_dir)
    
    print("\n" + "=" * 80)
    print("✓ Evaluation Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
