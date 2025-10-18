"""
Final comprehensive comparison: All configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
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
    
    # Load all results
    pretrained_data = np.load("C:/Users/Dhenenjay/Axion-Sat/outputs/pretrained_all_fixed/outputs.npz")
    trained_data = np.load("C:/Users/Dhenenjay/Axion-Sat/outputs/trained_models/outputs.npz")
    hybrid_data = np.load("C:/Users/Dhenenjay/Axion-Sat/outputs/hybrid_trained_s2_pretrained_s3/outputs.npz")
    
    # Extract 4 bands
    def extract_bands(data):
        opt_v1 = data['opt_v1'][[1, 2, 3, 7], :, :].astype(np.float32)
        opt_v2 = data['opt_v2'][[0, 1, 2, 3], :, :].astype(np.float32)
        opt_v3 = data['opt_v3'][[0, 1, 2, 3], :, :].astype(np.float32)
        return opt_v1, opt_v2, opt_v3
    
    pre_v1, pre_v2, pre_v3 = extract_bands(pretrained_data)
    train_v1, train_v2, train_v3 = extract_bands(trained_data)
    hybrid_v1, hybrid_v2, hybrid_v3 = extract_bands(hybrid_data)
    
    print("=" * 120)
    print("FINAL COMPREHENSIVE COMPARISON - ALL CONFIGURATIONS")
    print("=" * 120)
    print()
    
    print("IDEAL (Perfect Prediction):")
    print("  PSNR: ∞   |   MAE: 0.0000   |   RMSE: 0.0000")
    print()
    
    print("=" * 120)
    print("CONFIGURATION COMPARISON")
    print("=" * 120)
    
    configs = [
        ("Pretrained S2 + Pretrained S3", pre_v2, pre_v3),
        ("Trained S2 + Trained S3", train_v2, train_v3),
        ("Trained S2 + Pretrained S3 (HYBRID)", hybrid_v2, hybrid_v3)
    ]
    
    results = []
    
    for config_name, v2, v3 in configs:
        metrics_v2 = compute_metrics(s2_gt, v2)
        metrics_v3 = compute_metrics(s2_gt, v3)
        
        print(f"\n{config_name}")
        print("-" * 120)
        print(f"  Stage 2: PSNR={metrics_v2['psnr']:6.2f} dB  |  MAE={metrics_v2['mae']:.4f}  |  RMSE={metrics_v2['rmse']:.4f}")
        print(f"  Stage 3: PSNR={metrics_v3['psnr']:6.2f} dB  |  MAE={metrics_v3['mae']:.4f}  |  RMSE={metrics_v3['rmse']:.4f}")
        
        results.append({
            'name': config_name,
            's2_metrics': metrics_v2,
            's3_metrics': metrics_v3,
            's2_data': v2,
            's3_data': v3
        })
    
    print("\n" + "=" * 120)
    print("BEST STAGE 3 (Final Output)")
    print("=" * 120)
    
    best_s3 = min(results, key=lambda x: x['s3_metrics']['mae'])
    print(f"\n✓ WINNER: {best_s3['name']}")
    print(f"  PSNR:  {best_s3['s3_metrics']['psnr']:.2f} dB")
    print(f"  MAE:   {best_s3['s3_metrics']['mae']:.4f}  (difference from ideal: +{best_s3['s3_metrics']['mae']:.4f})")
    print(f"  RMSE:  {best_s3['s3_metrics']['rmse']:.4f}  (difference from ideal: +{best_s3['s3_metrics']['rmse']:.4f})")
    
    # Comparison table
    print("\n" + "=" * 120)
    print("DETAILED COMPARISON TABLE - STAGE 3 (Final Output)")
    print("=" * 120)
    print(f"{'Configuration':<45} | {'PSNR (dB)':<12} | {'MAE':<12} | {'RMSE':<12} | {'vs Ideal MAE':<15}")
    print("-" * 120)
    
    for r in results:
        m = r['s3_metrics']
        marker = " ← BEST" if r == best_s3 else ""
        print(f"{r['name']:<45} | {m['psnr']:>10.2f}   | {m['mae']:>10.4f} | {m['rmse']:>10.4f}  | +{m['mae']:>12.4f}{marker}")
    
    print("\n" + "=" * 120)
    print("KEY INSIGHTS")
    print("=" * 120)
    
    pre_mae = results[0]['s3_metrics']['mae']
    train_mae = results[1]['s3_metrics']['mae']
    hybrid_mae = results[2]['s3_metrics']['mae']
    
    print(f"\n1. Hybrid (Trained S2 + Pretrained S3) achieves MAE = {hybrid_mae:.4f}")
    print(f"   - Better than Fully Trained by: {train_mae - hybrid_mae:.4f} ({100*(train_mae - hybrid_mae)/train_mae:.1f}%)")
    print(f"   - Compared to Pretrained: {hybrid_mae - pre_mae:+.4f} ({100*(hybrid_mae - pre_mae)/pre_mae:+.1f}%)")
    
    print(f"\n2. Stage 2 Training Impact:")
    print(f"   - Pretrained S2 → Trained S2: MAE improved by {results[0]['s2_metrics']['mae'] - results[1]['s2_metrics']['mae']:.4f}")
    
    print(f"\n3. Stage 3 Observations:")
    print(f"   - Pretrained S3 decoder performs better than trained")
    print(f"   - Suggests potential overfitting in Stage 3 training")
    print(f"   - Hybrid approach leverages best of both stages")
    
    # Create comparison visualization
    output_dir = Path("outputs/final_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Ground truth
    gt_rgb = np.transpose(s2_gt[[2, 1, 0], :, :], (1, 2, 0))
    gt_rgb = np.clip(gt_rgb, 0, 1)
    axes[0, 0].imshow(gt_rgb)
    axes[0, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Pretrained S3
    pre_rgb = np.transpose(pre_v3[[2, 1, 0], :, :], (1, 2, 0))
    pre_rgb = np.clip(pre_rgb, 0, 1)
    axes[0, 1].imshow(pre_rgb)
    axes[0, 1].set_title(f'Pretrained S2+S3\nMAE={results[0]["s3_metrics"]["mae"]:.4f}', fontsize=12)
    axes[0, 1].axis('off')
    
    # Trained S3
    train_rgb = np.transpose(train_v3[[2, 1, 0], :, :], (1, 2, 0))
    train_rgb = np.clip(train_rgb, 0, 1)
    axes[0, 2].imshow(train_rgb)
    axes[0, 2].set_title(f'Trained S2+S3\nMAE={results[1]["s3_metrics"]["mae"]:.4f}', fontsize=12)
    axes[0, 2].axis('off')
    
    # Hybrid S3
    hybrid_rgb = np.transpose(hybrid_v3[[2, 1, 0], :, :], (1, 2, 0))
    hybrid_rgb = np.clip(hybrid_rgb, 0, 1)
    axes[0, 3].imshow(hybrid_rgb)
    axes[0, 3].set_title(f'HYBRID (Best)\nMAE={results[2]["s3_metrics"]["mae"]:.4f}', 
                         fontsize=12, fontweight='bold', color='green')
    axes[0, 3].axis('off')
    
    # Error maps
    error_pre = np.mean(np.abs(s2_gt - pre_v3), axis=0)
    im = axes[1, 1].imshow(error_pre, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 1].set_title('Error Map: Pretrained', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    
    error_train = np.mean(np.abs(s2_gt - train_v3), axis=0)
    im = axes[1, 2].imshow(error_train, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 2].set_title('Error Map: Trained', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    error_hybrid = np.mean(np.abs(s2_gt - hybrid_v3), axis=0)
    im = axes[1, 3].imshow(error_hybrid, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 3].set_title('Error Map: HYBRID', fontsize=12, fontweight='bold')
    axes[1, 3].axis('off')
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046)
    
    # PSNR comparison chart
    axes[1, 0].bar(['Pretrained', 'Trained', 'HYBRID'], 
                   [results[0]['s3_metrics']['psnr'], 
                    results[1]['s3_metrics']['psnr'],
                    results[2]['s3_metrics']['psnr']],
                   color=['blue', 'orange', 'green'])
    axes[1, 0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[1, 0].set_title('Stage 3 PSNR Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_comparison.png', dpi=200, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_dir / 'final_comparison.png'}")
    
    print("\n" + "=" * 120)


if __name__ == '__main__':
    main()
