"""
Visualize Stage 1 (TerraMind) output with SAR input and ground truth comparison.
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


def compute_ndvi(bands):
    """Compute NDVI from 4-band data (B, G, R, NIR)."""
    nir = bands[3]  # B08
    red = bands[2]  # B04
    denominator = nir + red
    ndvi = np.zeros_like(nir)
    valid = denominator > 1e-6
    ndvi[valid] = (nir[valid] - red[valid]) / denominator[valid]
    return ndvi


def main():
    # Load ground truth tile
    tile_path = Path("C:/Users/Dhenenjay/Axion-Sat/data/tiles/benv2_catalog/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57.npz")
    tile_data = np.load(tile_path)
    
    # Extract SAR
    sar_vv = tile_data['s1_vv']
    sar_vh = tile_data['s1_vh']
    
    # Extract ground truth optical (4 bands)
    s2_gt = np.stack([
        tile_data['s2_b2'],  # Blue
        tile_data['s2_b3'],  # Green
        tile_data['s2_b4'],  # Red
        tile_data['s2_b8']   # NIR
    ], axis=0).astype(np.float32)
    
    # Load Stage 1 output (use pretrained output)
    stage1_data = np.load("C:/Users/Dhenenjay/Axion-Sat/outputs/pretrained_all_fixed/outputs.npz")
    
    # Extract 4 bands from 12-band Stage 1 output
    opt_v1 = stage1_data['opt_v1'][[1, 2, 3, 7], :, :].astype(np.float32)
    
    # Compute metrics
    metrics = compute_metrics(s2_gt, opt_v1)
    
    # Compute NDVI
    ndvi_gt = compute_ndvi(s2_gt)
    ndvi_v1 = compute_ndvi(opt_v1)
    
    print("=" * 100)
    print("STAGE 1 (TerraMind) EVALUATION")
    print("=" * 100)
    print()
    print(f"SAR Input:")
    print(f"  VV range: [{sar_vv.min():.4f}, {sar_vv.max():.4f}]")
    print(f"  VH range: [{sar_vh.min():.4f}, {sar_vh.max():.4f}]")
    print(f"  Shape: {sar_vv.shape}")
    print()
    print(f"Ground Truth (Sentinel-2 Optical):")
    print(f"  Range: [{s2_gt.min():.4f}, {s2_gt.max():.4f}]")
    print(f"  Shape: {s2_gt.shape}")
    print(f"  Bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)")
    print()
    print(f"Stage 1 Output (TerraMind Generated):")
    print(f"  Range: [{opt_v1.min():.4f}, {opt_v1.max():.4f}]")
    print(f"  Shape: {opt_v1.shape}")
    print()
    print("=" * 100)
    print("METRICS: Stage 1 vs Ground Truth")
    print("=" * 100)
    print(f"  PSNR:  {metrics['psnr']:.2f} dB  (ideal: ∞)")
    print(f"  MAE:   {metrics['mae']:.4f}     (ideal: 0.0000, difference: +{metrics['mae']:.4f})")
    print(f"  RMSE:  {metrics['rmse']:.4f}     (ideal: 0.0000, difference: +{metrics['rmse']:.4f})")
    print()
    
    # Per-band metrics
    band_names = ['B02 (Blue)', 'B03 (Green)', 'B04 (Red)', 'B08 (NIR)']
    print("Per-band MAE:")
    for i, name in enumerate(band_names):
        band_mae = np.mean(np.abs(s2_gt[i] - opt_v1[i]))
        print(f"  {name}: {band_mae:.4f}")
    
    print()
    print(f"NDVI Accuracy:")
    print(f"  NDVI MAE:  {np.mean(np.abs(ndvi_gt - ndvi_v1)):.4f}")
    print(f"  NDVI RMSE: {np.sqrt(np.mean((ndvi_gt - ndvi_v1) ** 2)):.4f}")
    
    # Create comprehensive visualization
    output_dir = Path("outputs/stage1_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid: 3 rows, 4 columns
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: SAR inputs
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(sar_vv, cmap='gray', vmin=sar_vv.min(), vmax=sar_vv.max())
    ax1.set_title('SAR Input: VV Polarization', fontsize=14, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(sar_vh, cmap='gray', vmin=sar_vh.min(), vmax=sar_vh.max())
    ax2.set_title('SAR Input: VH Polarization', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # SAR RGB composite (VV=R, VH=G, VV/VH=B)
    ax3 = fig.add_subplot(gs[0, 2])
    sar_rgb = np.stack([
        (sar_vv - sar_vv.min()) / (sar_vv.max() - sar_vv.min() + 1e-8),
        (sar_vh - sar_vh.min()) / (sar_vh.max() - sar_vh.min() + 1e-8),
        np.clip(sar_vv / (sar_vh + 0.1), 0, 1)
    ], axis=-1).astype(np.float32)
    ax3.imshow(sar_rgb)
    ax3.set_title('SAR RGB Composite\n(R=VV, G=VH, B=VV/VH)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Add arrow
    ax_arrow = fig.add_subplot(gs[0, 3])
    ax_arrow.text(0.5, 0.5, '↓\nStage 1\nAxion-Sat\nSAR→Optical', 
                  ha='center', va='center', fontsize=16, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax_arrow.axis('off')
    
    # Row 2: Optical RGB comparisons
    ax4 = fig.add_subplot(gs[1, 0])
    gt_rgb = np.transpose(s2_gt[[2, 1, 0], :, :], (1, 2, 0))  # RGB
    gt_rgb = np.clip(gt_rgb, 0, 1)
    ax4.imshow(gt_rgb)
    ax4.set_title('Ground Truth (Sentinel-2)\nTrue Color RGB', fontsize=14, fontweight='bold', color='green')
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 1])
    v1_rgb = np.transpose(opt_v1[[2, 1, 0], :, :], (1, 2, 0))  # RGB
    v1_rgb = np.clip(v1_rgb, 0, 1)
    ax5.imshow(v1_rgb)
    ax5.set_title(f'Stage 1 Output (Axion-Sat)\nMAE={metrics["mae"]:.4f}', fontsize=14, fontweight='bold')
    ax5.axis('off')
    
    # Error map
    ax6 = fig.add_subplot(gs[1, 2])
    error_map = np.mean(np.abs(s2_gt - opt_v1), axis=0)
    im = ax6.imshow(error_map, cmap='hot', vmin=0, vmax=0.3)
    ax6.set_title('Absolute Error Map\n(Per-pixel MAE)', fontsize=14, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    
    # Metrics text
    ax7 = fig.add_subplot(gs[1, 3])
    metrics_text = f"""
QUANTITATIVE METRICS

PSNR:  {metrics['psnr']:.2f} dB
MAE:   {metrics['mae']:.4f}
RMSE:  {metrics['rmse']:.4f}

Per-band MAE:
  Blue:  {np.mean(np.abs(s2_gt[0] - opt_v1[0])):.4f}
  Green: {np.mean(np.abs(s2_gt[1] - opt_v1[1])):.4f}
  Red:   {np.mean(np.abs(s2_gt[2] - opt_v1[2])):.4f}
  NIR:   {np.mean(np.abs(s2_gt[3] - opt_v1[3])):.4f}

Difference from Ideal:
  MAE:  +{metrics['mae']:.4f}
  RMSE: +{metrics['rmse']:.4f}
"""
    ax7.text(0.05, 0.95, metrics_text, transform=ax7.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax7.axis('off')
    
    # Row 3: NDVI comparison
    ax8 = fig.add_subplot(gs[2, 0])
    im = ax8.imshow(ndvi_gt, cmap='RdYlGn', vmin=-1, vmax=1)
    ax8.set_title('Ground Truth NDVI', fontsize=14, fontweight='bold')
    ax8.axis('off')
    plt.colorbar(im, ax=ax8, fraction=0.046, pad=0.04)
    
    ax9 = fig.add_subplot(gs[2, 1])
    im = ax9.imshow(ndvi_v1, cmap='RdYlGn', vmin=-1, vmax=1)
    ax9.set_title('Stage 1 NDVI', fontsize=14, fontweight='bold')
    ax9.axis('off')
    plt.colorbar(im, ax=ax9, fraction=0.046, pad=0.04)
    
    ax10 = fig.add_subplot(gs[2, 2])
    ndvi_error = np.abs(ndvi_gt - ndvi_v1)
    im = ax10.imshow(ndvi_error, cmap='hot', vmin=0, vmax=1)
    ax10.set_title('NDVI Error Map', fontsize=14, fontweight='bold')
    ax10.axis('off')
    plt.colorbar(im, ax=ax10, fraction=0.046, pad=0.04)
    
    # Histogram comparison
    ax11 = fig.add_subplot(gs[2, 3])
    ax11.hist(s2_gt.flatten(), bins=50, alpha=0.5, label='Ground Truth', color='green', density=True)
    ax11.hist(opt_v1.flatten(), bins=50, alpha=0.5, label='Stage 1 Output', color='blue', density=True)
    ax11.set_xlabel('Reflectance', fontsize=12)
    ax11.set_ylabel('Density', fontsize=12)
    ax11.set_title('Reflectance Distribution', fontsize=14, fontweight='bold')
    ax11.legend()
    ax11.grid(alpha=0.3)
    
    plt.suptitle('STAGE 1: Axion-Sat SAR-to-Optical Translation', fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_dir / 'stage1_complete_visualization.png', dpi=200, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_dir / 'stage1_complete_visualization.png'}")
    print()
    print("=" * 100)


if __name__ == '__main__':
    main()
