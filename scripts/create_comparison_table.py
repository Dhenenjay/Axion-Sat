"""
Create comprehensive comparison table: Ground Truth vs Untrained vs Trained models.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_and_normalize(npz_path):
    """Load NPZ and normalize to [0, 1]."""
    data = np.load(npz_path)
    
    # For pipeline outputs
    if 'opt_v3' in data:
        opt = data['opt_v3'].astype(np.float32)
    # For ground truth
    elif 's2_b2' in data:
        opt = np.stack([
            data['s2_b2'],
            data['s2_b3'],
            data['s2_b4'],
            data['s2_b8']
        ], axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unknown data format in {npz_path}")
    
    return np.clip(opt, 0, 1)


def compute_metrics(gt, pred):
    """Compute all metrics."""
    mae = np.mean(np.abs(gt - pred))
    mse = np.mean((gt - pred) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    # NDVI
    nir_gt, red_gt = gt[3], gt[2]
    nir_pred, red_pred = pred[3], pred[2]
    
    denom_gt = nir_gt + red_gt
    denom_pred = nir_pred + red_pred
    
    ndvi_gt = np.where(denom_gt > 1e-6, (nir_gt - red_gt) / denom_gt, 0)
    ndvi_pred = np.where(denom_pred > 1e-6, (nir_pred - red_pred) / denom_pred, 0)
    
    ndvi_mae = np.mean(np.abs(ndvi_gt - ndvi_pred))
    
    return {
        'PSNR (dB)': f"{psnr:.2f}",
        'MAE': f"{mae:.4f}",
        'NDVI MAE': f"{ndvi_mae:.4f}"
    }


def create_rgb(data):
    """Create RGB composite."""
    # R, G, B = channels 2, 1, 0
    rgb = np.transpose(data[[2, 1, 0], :, :], (1, 2, 0))
    return np.clip(rgb, 0, 1)


def main():
    print("=" * 80)
    print("COMPREHENSIVE COMPARISON: Ground Truth vs Models")
    print("=" * 80)
    
    # Load data
    tile_path = Path("data/tiles/benv2_catalog/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57.npz")
    trained_path = Path("outputs/final_test/outputs.npz")
    untrained_path = Path("outputs/untrained_baseline/outputs.npz")
    
    print("\nLoading data...")
    gt = load_and_normalize(tile_path)
    trained = load_and_normalize(trained_path)
    untrained = load_and_normalize(untrained_path)
    
    print(f"  Ground Truth: {gt.shape}")
    print(f"  Trained Model: {trained.shape}")
    print(f"  Untrained Model: {untrained.shape}")
    
    # Handle batch dimension
    if trained.ndim == 4:
        trained = trained[0]
    if untrained.ndim == 4:
        untrained = untrained[0]
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics_trained = compute_metrics(gt, trained)
    metrics_untrained = compute_metrics(gt, untrained)
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("QUANTITATIVE COMPARISON TABLE")
    print("=" * 80)
    print(f"\n{'Metric':<15} {'Ground Truth':<20} {'Untrained':<20} {'Your Trained':<20}")
    print("-" * 80)
    
    for key in metrics_trained.keys():
        print(f"{key:<15} {'-':<20} {metrics_untrained[key]:<20} {metrics_trained[key]:<20}")
    
    # Calculate improvements
    psnr_untrained = float(metrics_untrained['PSNR (dB)'].split()[0])
    psnr_trained = float(metrics_trained['PSNR (dB)'].split()[0])
    improvement = psnr_trained - psnr_untrained
    
    print("\n" + "=" * 80)
    print(f"IMPROVEMENT: {improvement:+.2f} dB PSNR")
    if improvement > 0:
        print(f"Your training IMPROVED the model by {improvement:.2f} dB!")
    else:
        print(f"Training decreased performance by {abs(improvement):.2f} dB (needs investigation)")
    print("=" * 80)
    
    # Create visual comparison
    print("\nCreating visualization...")
    output_dir = Path("outputs/final_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top row: RGB composites
    gt_rgb = create_rgb(gt)
    axes[0, 0].imshow(gt_rgb)
    axes[0, 0].set_title('Ground Truth\n(Real Sentinel-2)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    untrained_rgb = create_rgb(untrained)
    axes[0, 1].imshow(untrained_rgb)
    axes[0, 1].set_title(f'Untrained Models\nPSNR: {metrics_untrained["PSNR (dB)"]}', fontsize=14)
    axes[0, 1].axis('off')
    
    trained_rgb = create_rgb(trained)
    axes[0, 2].imshow(trained_rgb)
    axes[0, 2].set_title(f'Your Trained Models\nPSNR: {metrics_trained["PSNR (dB)"]}', fontsize=14)
    axes[0, 2].axis('off')
    
    # Bottom row: Error maps
    error_untrained = np.mean(np.abs(gt - untrained), axis=0)
    im1 = axes[1, 0].imshow(error_untrained, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 0].set_title(f'Untrained Error\nMAE: {metrics_untrained["MAE"]}', fontsize=14)
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    error_trained = np.mean(np.abs(gt - trained), axis=0)
    im2 = axes[1, 1].imshow(error_trained, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 1].set_title(f'Trained Error\nMAE: {metrics_trained["MAE"]}', fontsize=14)
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    
    # Difference in errors (improvement map)
    error_diff = error_untrained - error_trained  # Positive = improvement
    im3 = axes[1, 2].imshow(error_diff, cmap='RdYlGn', vmin=-0.2, vmax=0.2)
    axes[1, 2].set_title(f'Error Reduction\n(Green = Better with Training)', fontsize=14)
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to: {output_dir}")
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nGround Truth: Real Sentinel-2 optical imagery")
    print(f"Untrained:    Pretrained models without your training")
    print(f"Trained:      Your models after training on BigEarthNet v2")
    print(f"\nResult: {'IMPROVEMENT' if improvement > 0 else 'DEGRADATION'} of {abs(improvement):.2f} dB")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
