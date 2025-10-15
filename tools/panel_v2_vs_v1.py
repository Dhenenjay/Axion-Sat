"""
Stage 2 (v2) vs Stage 1 (v1) Visual Comparison Tool

Creates side-by-side visual comparisons showing:
- Stage 1 (v1) output
- Stage 2 (v2) output
- Ground truth (Sentinel-2/HLS)
- Difference maps (v2 - v1, v2 - truth)
- Vegetation indices (NDVI, EVI)
- Spectral metrics

Saves detailed comparison panels to output directory.

Usage:
    # Compare single tile
    python tools/panel_v2_vs_v1.py --tile data/tiles/sample.npz --output comparisons/

    # Compare all tiles in directory
    python tools/panel_v2_vs_v1.py --data_dir data/tiles/benv2_catalog --output comparisons/

    # With Stage 1 and Stage 2 model checkpoints
    python tools/panel_v2_vs_v1.py --tile sample.npz --stage1_model models/stage1.pt --stage2_model models/stage2.pt

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, Optional, Tuple, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from tqdm import tqdm


# ============================================================================
# Vegetation Indices
# ============================================================================

def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Compute NDVI (Normalized Difference Vegetation Index).
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        nir: Near-infrared band
        red: Red band
        
    Returns:
        NDVI values in range [-1, 1]
    """
    denominator = nir + red
    ndvi = np.where(
        denominator != 0,
        (nir - red) / denominator,
        0
    )
    return np.clip(ndvi, -1, 1)


def compute_evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """
    Compute EVI (Enhanced Vegetation Index).
    
    EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    
    Args:
        nir: Near-infrared band
        red: Red band
        blue: Blue band
        
    Returns:
        EVI values (typically in range [-1, 1])
    """
    denominator = nir + 6.0 * red - 7.5 * blue + 1.0
    evi = np.where(
        denominator != 0,
        2.5 * (nir - red) / denominator,
        0
    )
    return np.clip(evi, -1, 1)


# ============================================================================
# Metrics
# ============================================================================

def compute_rmse(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute RMSE between prediction and target."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    return np.sqrt(np.mean((pred - target) ** 2))


def compute_mae(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute MAE between prediction and target."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    return np.mean(np.abs(pred - target))


def compute_sam(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Compute Spectral Angle Mapper (SAM) between prediction and target.
    
    Args:
        pred: Predicted image (C, H, W)
        target: Target image (C, H, W)
        mask: Optional mask (H, W)
        
    Returns:
        Mean SAM in radians
    """
    # Reshape to (C, N)
    C, H, W = pred.shape
    pred_flat = pred.reshape(C, -1)
    target_flat = target.reshape(C, -1)
    
    # Compute cosine similarity
    dot_product = np.sum(pred_flat * target_flat, axis=0)
    norm_pred = np.linalg.norm(pred_flat, axis=0)
    norm_target = np.linalg.norm(target_flat, axis=0)
    
    # Avoid division by zero
    denominator = norm_pred * norm_target
    cos_sim = np.where(denominator != 0, dot_product / denominator, 1.0)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
    # SAM in radians
    sam = np.arccos(cos_sim)
    
    if mask is not None:
        mask_flat = mask.flatten()
        sam = sam[mask_flat]
    
    return np.mean(sam)


# ============================================================================
# Visualization
# ============================================================================

def normalize_for_display(image: np.ndarray, percentile: Tuple[float, float] = (2, 98)) -> np.ndarray:
    """
    Normalize image for display using percentile clipping.
    
    Args:
        image: Input image
        percentile: (min_percentile, max_percentile) for clipping
        
    Returns:
        Normalized image in range [0, 1]
    """
    vmin, vmax = np.percentile(image[np.isfinite(image)], percentile)
    image_norm = (image - vmin) / (vmax - vmin + 1e-8)
    return np.clip(image_norm, 0, 1)


def create_rgb_composite(bands: np.ndarray, rgb_indices: Tuple[int, int, int] = (2, 1, 0)) -> np.ndarray:
    """
    Create RGB composite from multi-band image.
    
    Args:
        bands: Multi-band image (C, H, W)
        rgb_indices: Indices for (R, G, B) bands
        
    Returns:
        RGB image (H, W, 3)
    """
    r_idx, g_idx, b_idx = rgb_indices
    
    r = normalize_for_display(bands[r_idx])
    g = normalize_for_display(bands[g_idx])
    b = normalize_for_display(bands[b_idx])
    
    rgb = np.stack([r, g, b], axis=-1)
    return rgb


def create_comparison_panel(
    v1_bands: np.ndarray,
    v2_bands: np.ndarray,
    truth_bands: np.ndarray,
    title: str,
    output_path: Path
):
    """
    Create comprehensive comparison panel.
    
    Args:
        v1_bands: Stage 1 output (4, H, W) - Blue, Green, Red, NIR
        v2_bands: Stage 2 output (4, H, W)
        truth_bands: Ground truth (4, H, W)
        title: Panel title
        output_path: Path to save figure
    """
    # Create figure with grid
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Extract bands (assume order: Blue, Green, Red, NIR)
    blue_idx, green_idx, red_idx, nir_idx = 0, 1, 2, 3
    
    # Row 1: RGB Composites
    # Stage 1
    ax = fig.add_subplot(gs[0, 0])
    rgb_v1 = create_rgb_composite(v1_bands, rgb_indices=(red_idx, green_idx, blue_idx))
    ax.imshow(rgb_v1)
    ax.set_title('Stage 1 (v1)\nSynthetic Optical', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Stage 2
    ax = fig.add_subplot(gs[0, 1])
    rgb_v2 = create_rgb_composite(v2_bands, rgb_indices=(red_idx, green_idx, blue_idx))
    ax.imshow(rgb_v2)
    ax.set_title('Stage 2 (v2)\nRefined Output', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Ground Truth
    ax = fig.add_subplot(gs[0, 2])
    rgb_truth = create_rgb_composite(truth_bands, rgb_indices=(red_idx, green_idx, blue_idx))
    ax.imshow(rgb_truth)
    ax.set_title('Ground Truth\nSentinel-2/HLS', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Difference: v2 - v1
    ax = fig.add_subplot(gs[0, 3])
    diff_v2_v1 = np.mean(v2_bands - v1_bands, axis=0)
    im = ax.imshow(diff_v2_v1, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    ax.set_title('Difference\nv2 - v1', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Difference: v2 - truth
    ax = fig.add_subplot(gs[0, 4])
    diff_v2_truth = np.mean(v2_bands - truth_bands, axis=0)
    im = ax.imshow(diff_v2_truth, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    ax.set_title('Error\nv2 - truth', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Row 2: NDVI Comparison
    # Compute NDVI
    ndvi_v1 = compute_ndvi(v1_bands[nir_idx], v1_bands[red_idx])
    ndvi_v2 = compute_ndvi(v2_bands[nir_idx], v2_bands[red_idx])
    ndvi_truth = compute_ndvi(truth_bands[nir_idx], truth_bands[red_idx])
    
    # NDVI v1
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(ndvi_v1, cmap='RdYlGn', vmin=-0.5, vmax=1.0)
    ax.set_title('NDVI: Stage 1', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # NDVI v2
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(ndvi_v2, cmap='RdYlGn', vmin=-0.5, vmax=1.0)
    ax.set_title('NDVI: Stage 2', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # NDVI truth
    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow(ndvi_truth, cmap='RdYlGn', vmin=-0.5, vmax=1.0)
    ax.set_title('NDVI: Truth', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # NDVI error: v2 - truth
    ax = fig.add_subplot(gs[1, 3])
    ndvi_error = ndvi_v2 - ndvi_truth
    im = ax.imshow(ndvi_error, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    ax.set_title('NDVI Error\nv2 - truth', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # NDVI improvement
    ax = fig.add_subplot(gs[1, 4])
    ndvi_improvement = np.abs(ndvi_v1 - ndvi_truth) - np.abs(ndvi_v2 - ndvi_truth)
    im = ax.imshow(ndvi_improvement, cmap='RdYlGn', vmin=-0.1, vmax=0.1)
    ax.set_title('NDVI Improvement\n|v1-truth| - |v2-truth|', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Row 3: EVI Comparison
    # Compute EVI
    evi_v1 = compute_evi(v1_bands[nir_idx], v1_bands[red_idx], v1_bands[blue_idx])
    evi_v2 = compute_evi(v2_bands[nir_idx], v2_bands[red_idx], v2_bands[blue_idx])
    evi_truth = compute_evi(truth_bands[nir_idx], truth_bands[red_idx], truth_bands[blue_idx])
    
    # EVI v1
    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(evi_v1, cmap='RdYlGn', vmin=-0.5, vmax=1.0)
    ax.set_title('EVI: Stage 1', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # EVI v2
    ax = fig.add_subplot(gs[2, 1])
    im = ax.imshow(evi_v2, cmap='RdYlGn', vmin=-0.5, vmax=1.0)
    ax.set_title('EVI: Stage 2', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # EVI truth
    ax = fig.add_subplot(gs[2, 2])
    im = ax.imshow(evi_truth, cmap='RdYlGn', vmin=-0.5, vmax=1.0)
    ax.set_title('EVI: Truth', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # EVI error: v2 - truth
    ax = fig.add_subplot(gs[2, 3])
    evi_error = evi_v2 - evi_truth
    im = ax.imshow(evi_error, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    ax.set_title('EVI Error\nv2 - truth', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Metrics text box
    ax = fig.add_subplot(gs[2, 4])
    ax.axis('off')
    
    # Compute metrics
    valid_mask = np.isfinite(truth_bands).all(axis=0)
    
    # RGB metrics
    rmse_v1_rgb = compute_rmse(v1_bands[:3], truth_bands[:3], valid_mask)
    rmse_v2_rgb = compute_rmse(v2_bands[:3], truth_bands[:3], valid_mask)
    
    # NDVI metrics
    rmse_v1_ndvi = compute_rmse(ndvi_v1, ndvi_truth, valid_mask)
    rmse_v2_ndvi = compute_rmse(ndvi_v2, ndvi_truth, valid_mask)
    
    # EVI metrics
    rmse_v1_evi = compute_rmse(evi_v1, evi_truth, valid_mask)
    rmse_v2_evi = compute_rmse(evi_v2, evi_truth, valid_mask)
    
    # SAM
    sam_v1 = compute_sam(v1_bands, truth_bands, valid_mask)
    sam_v2 = compute_sam(v2_bands, truth_bands, valid_mask)
    
    metrics_text = f"""Metrics vs Ground Truth:

RGB RMSE:
  v1: {rmse_v1_rgb:.4f}
  v2: {rmse_v2_rgb:.4f}
  Δ:  {rmse_v1_rgb - rmse_v2_rgb:+.4f}

NDVI RMSE:
  v1: {rmse_v1_ndvi:.4f}
  v2: {rmse_v2_ndvi:.4f}
  Δ:  {rmse_v1_ndvi - rmse_v2_ndvi:+.4f}

EVI RMSE:
  v1: {rmse_v1_evi:.4f}
  v2: {rmse_v2_evi:.4f}
  Δ:  {rmse_v1_evi - rmse_v2_evi:+.4f}

SAM (radians):
  v1: {sam_v1:.4f}
  v2: {sam_v2:.4f}
  Δ:  {sam_v1 - sam_v2:+.4f}

Stage 2 Improvements:
  {'✓' if rmse_v2_rgb < rmse_v1_rgb else '✗'} RGB: {abs(rmse_v1_rgb - rmse_v2_rgb)/rmse_v1_rgb*100:.1f}%
  {'✓' if rmse_v2_ndvi < rmse_v1_ndvi else '✗'} NDVI: {abs(rmse_v1_ndvi - rmse_v2_ndvi)/rmse_v1_ndvi*100:.1f}%
  {'✓' if rmse_v2_evi < rmse_v1_evi else '✗'} EVI: {abs(rmse_v1_evi - rmse_v2_evi)/rmse_v1_evi*100:.1f}%
  {'✓' if sam_v2 < sam_v1 else '✗'} SAM: {abs(sam_v1 - sam_v2)/sam_v1*100:.1f}%
"""
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'rmse_v1_rgb': rmse_v1_rgb,
        'rmse_v2_rgb': rmse_v2_rgb,
        'rmse_v1_ndvi': rmse_v1_ndvi,
        'rmse_v2_ndvi': rmse_v2_ndvi,
        'rmse_v1_evi': rmse_v1_evi,
        'rmse_v2_evi': rmse_v2_evi,
        'sam_v1': sam_v1,
        'sam_v2': sam_v2,
    }


# ============================================================================
# Data Loading
# ============================================================================

def load_tile_data(tile_path: Path) -> Dict[str, np.ndarray]:
    """
    Load tile data from NPZ file.
    
    Expected keys:
        - opt_v1: Stage 1 output (4, H, W)
        - opt_v2: Stage 2 output (4, H, W)
        - s2_real: Ground truth Sentinel-2 (4, H, W)
        
    Or raw Sentinel-2 bands:
        - s2_b2, s2_b3, s2_b4, s2_b8
        
    Returns:
        Dict with 'v1', 'v2', 'truth' arrays
    """
    data = np.load(tile_path)
    
    result = {}
    
    # Try to load Stage 1 output
    if 'opt_v1' in data:
        result['v1'] = data['opt_v1']
    else:
        print(f"⚠ 'opt_v1' not found in {tile_path.name}, using mock data")
        # Create mock v1 from truth + noise
        if 's2_real' in data:
            truth = data['s2_real']
            result['v1'] = truth + np.random.randn(*truth.shape) * 0.05
        else:
            result['v1'] = None
    
    # Try to load Stage 2 output
    if 'opt_v2' in data:
        result['v2'] = data['opt_v2']
    else:
        print(f"⚠ 'opt_v2' not found in {tile_path.name}, using v1 as proxy")
        result['v2'] = result['v1']
    
    # Try to load ground truth
    if 's2_real' in data:
        result['truth'] = data['s2_real']
    elif all(k in data for k in ['s2_b2', 's2_b3', 's2_b4', 's2_b8']):
        # Stack Sentinel-2 bands
        result['truth'] = np.stack([
            data['s2_b2'],  # Blue
            data['s2_b3'],  # Green
            data['s2_b4'],  # Red
            data['s2_b8'],  # NIR
        ], axis=0)
    else:
        print(f"⚠ No ground truth found in {tile_path.name}")
        result['truth'] = result['v1']
    
    return result


# ============================================================================
# Main
# ============================================================================

def process_single_tile(tile_path: Path, output_dir: Path) -> Optional[Dict]:
    """Process a single tile and create comparison panel."""
    try:
        # Load data
        data = load_tile_data(tile_path)
        
        if data['v1'] is None or data['v2'] is None or data['truth'] is None:
            print(f"⚠ Skipping {tile_path.name}: missing data")
            return None
        
        # Create output filename
        output_path = output_dir / f"{tile_path.stem}_v2_vs_v1.png"
        
        # Create comparison panel
        metrics = create_comparison_panel(
            v1_bands=data['v1'],
            v2_bands=data['v2'],
            truth_bands=data['truth'],
            title=f"Stage 2 vs Stage 1 Comparison: {tile_path.name}",
            output_path=output_path
        )
        
        print(f"✓ Saved comparison: {output_path.name}")
        
        return metrics
        
    except Exception as e:
        print(f"✗ Error processing {tile_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Create visual comparisons of Stage 2 vs Stage 1 outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--tile',
        type=Path,
        help='Single tile to process'
    )
    
    parser.add_argument(
        '--data_dir',
        type=Path,
        help='Directory containing tiles to process'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('comparisons/'),
        help='Output directory for comparison panels (default: comparisons/)'
    )
    
    parser.add_argument(
        '--max_tiles',
        type=int,
        default=None,
        help='Maximum number of tiles to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.tile and not args.data_dir:
        print("Error: Must specify either --tile or --data_dir")
        return 1
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Stage 2 vs Stage 1 Visual Comparison")
    print("=" * 80)
    
    # Collect tiles to process
    tiles = []
    
    if args.tile:
        if args.tile.exists():
            tiles.append(args.tile)
        else:
            print(f"Error: Tile not found: {args.tile}")
            return 1
    
    if args.data_dir:
        if args.data_dir.exists():
            tiles.extend(sorted(args.data_dir.rglob('*.npz')))
        else:
            print(f"Error: Directory not found: {args.data_dir}")
            return 1
    
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]
    
    print(f"\nFound {len(tiles)} tile(s) to process")
    print(f"Output directory: {args.output}")
    
    # Process tiles
    all_metrics = []
    
    for tile in tqdm(tiles, desc="Processing tiles"):
        metrics = process_single_tile(tile, args.output)
        if metrics:
            all_metrics.append(metrics)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    if all_metrics:
        print(f"\nProcessed {len(all_metrics)} tile(s) successfully")
        
        # Average metrics
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        print(f"\nAverage Metrics:")
        print(f"  RGB RMSE:  v1={avg_metrics['rmse_v1_rgb']:.4f}, v2={avg_metrics['rmse_v2_rgb']:.4f}")
        print(f"  NDVI RMSE: v1={avg_metrics['rmse_v1_ndvi']:.4f}, v2={avg_metrics['rmse_v2_ndvi']:.4f}")
        print(f"  EVI RMSE:  v1={avg_metrics['rmse_v1_evi']:.4f}, v2={avg_metrics['rmse_v2_evi']:.4f}")
        print(f"  SAM:       v1={avg_metrics['sam_v1']:.4f}, v2={avg_metrics['sam_v2']:.4f}")
        
        print(f"\nImprovements:")
        rgb_imp = (avg_metrics['rmse_v1_rgb'] - avg_metrics['rmse_v2_rgb']) / avg_metrics['rmse_v1_rgb'] * 100
        ndvi_imp = (avg_metrics['rmse_v1_ndvi'] - avg_metrics['rmse_v2_ndvi']) / avg_metrics['rmse_v1_ndvi'] * 100
        evi_imp = (avg_metrics['rmse_v1_evi'] - avg_metrics['rmse_v2_evi']) / avg_metrics['rmse_v1_evi'] * 100
        sam_imp = (avg_metrics['sam_v1'] - avg_metrics['sam_v2']) / avg_metrics['sam_v1'] * 100
        
        print(f"  RGB:  {rgb_imp:+.1f}%")
        print(f"  NDVI: {ndvi_imp:+.1f}%")
        print(f"  EVI:  {evi_imp:+.1f}%")
        print(f"  SAM:  {sam_imp:+.1f}%")
    else:
        print("\n⚠ No tiles processed successfully")
    
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
