"""
Spectral Angle Mapper (SAM) Metric Tool

This tool computes the Spectral Angle Mapper (SAM) metric between predicted and
target multispectral images, with support for vegetation masking to focus analysis
on areas with significant spectral variability.

SAM measures the spectral similarity between two spectra by computing the angle
between them in n-dimensional space (where n = number of bands). It is invariant
to illumination intensity and is commonly used in remote sensing for:
- Material identification
- Change detection
- Spectral similarity assessment

Key Features:
1. Per-pixel SAM computation
2. Vegetation masking (NDVI threshold)
3. Biome-specific analysis
4. Histogram visualization
5. Spatial SAM maps

Usage:
    # Basic SAM computation
    python tools/sam_metric.py --pred opt_v1.npz --target s2_real.npz --output reports/sam_analysis.png

    # With vegetation masking
    python tools/sam_metric.py --pred opt_v1.npz --target s2_real.npz --ndvi_threshold 0.3 --output reports/sam_veg.png

    # Batch processing
    python tools/sam_metric.py --data_dir data/tiles --output_dir reports/sam_analysis

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# SAM Computation
# ============================================================================

def compute_sam_per_pixel(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Compute Spectral Angle Mapper (SAM) per pixel.
    
    SAM(pred, target) = arccos(dot(pred, target) / (||pred|| * ||target||))
    
    The spectral angle is measured in radians. Smaller angles indicate
    higher spectral similarity.
    
    Args:
        pred: Predicted image, shape (C, H, W) where C = number of bands
        target: Target image, shape (C, H, W)
        eps: Small constant for numerical stability
        
    Returns:
        SAM map, shape (H, W), values in radians [0, π]
        
    Notes:
        - SAM = 0: Identical spectra
        - SAM = π/2: Orthogonal spectra (90 degrees)
        - SAM = π: Opposite spectra (180 degrees)
    """
    C, H, W = pred.shape
    assert target.shape == pred.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
    
    # Reshape to (C, N) where N = H*W
    pred_flat = pred.reshape(C, -1)  # (C, H*W)
    target_flat = target.reshape(C, -1)  # (C, H*W)
    
    # Compute dot product
    dot_product = np.sum(pred_flat * target_flat, axis=0)  # (H*W,)
    
    # Compute norms
    pred_norm = np.linalg.norm(pred_flat, axis=0) + eps  # (H*W,)
    target_norm = np.linalg.norm(target_flat, axis=0) + eps  # (H*W,)
    
    # Compute cosine similarity
    cos_angle = dot_product / (pred_norm * target_norm)
    
    # Clamp to valid range for arccos [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Compute angle in radians
    angle_rad = np.arccos(cos_angle)
    
    # Reshape back to (H, W)
    sam_map = angle_rad.reshape(H, W)
    
    return sam_map


def compute_ndvi(red: np.ndarray, nir: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute NDVI: (NIR - Red) / (NIR + Red)
    
    Args:
        red: Red band (B04), shape (H, W)
        nir: NIR band (B08), shape (H, W)
        eps: Small constant for numerical stability
        
    Returns:
        NDVI map, shape (H, W), values in [-1, 1]
    """
    numerator = nir - red
    denominator = nir + red + eps
    return numerator / denominator


def create_vegetation_mask(
    ndvi: np.ndarray,
    ndvi_threshold: float = 0.3,
    morphology_filter: bool = True
) -> np.ndarray:
    """
    Create vegetation mask from NDVI.
    
    Args:
        ndvi: NDVI map, shape (H, W)
        ndvi_threshold: Minimum NDVI value to consider as vegetation (default: 0.3)
        morphology_filter: Apply morphological operations to clean up mask
        
    Returns:
        Binary mask, shape (H, W), dtype bool
        
    Notes:
        NDVI thresholds:
        - < 0.0: Water, bare soil
        - 0.0-0.2: Sparse vegetation
        - 0.2-0.4: Moderate vegetation
        - 0.4-0.8: Dense vegetation
        - > 0.8: Very dense vegetation
    """
    # Create binary mask
    mask = ndvi >= ndvi_threshold
    
    # Optional: morphological filtering to remove noise
    if morphology_filter:
        try:
            from scipy.ndimage import binary_erosion, binary_dilation
            # Erosion to remove small isolated pixels
            mask = binary_erosion(mask, iterations=1)
            # Dilation to restore boundaries
            mask = binary_dilation(mask, iterations=1)
        except ImportError:
            # Skip morphological filtering if scipy not available
            pass
    
    return mask


# ============================================================================
# Data Loading
# ============================================================================

def load_image_data(file_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Load image data from NPZ file.
    
    Expected keys:
        - 'pred' or 'opt_v1' or 'opt_v2': Predicted image (C, H, W)
        - 'target' or 's2_real' or 'real': Target image (C, H, W)
        - 'metadata' (optional): Dictionary with additional info
        
    Returns:
        Dict with 'pred', 'target', 'metadata' (optional)
    """
    try:
        data = np.load(file_path)
        result = {}
        
        # Try to find predicted image
        for key in ['pred', 'opt_v1', 'opt_v2', 'stage1', 'stage2']:
            if key in data:
                result['pred'] = data[key].astype(np.float32)
                break
        
        # Try to find target image
        for key in ['target', 's2_real', 'real', 's2', 'baseline']:
            if key in data:
                result['target'] = data[key].astype(np.float32)
                break
        
        # Check if we have both
        if 'pred' not in result or 'target' not in result:
            return None
        
        # Validate shapes
        if result['pred'].ndim != 3 or result['target'].ndim != 3:
            return None
        
        if result['pred'].shape != result['target'].shape:
            return None
        
        # Metadata (optional)
        if 'metadata' in data:
            result['metadata'] = data['metadata'].item() if data['metadata'].ndim == 0 else dict(data['metadata'])
        
        return result
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


# ============================================================================
# Visualization
# ============================================================================

def plot_sam_analysis(
    sam_map: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    vegetation_mask: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    show_rgb: bool = True
):
    """
    Create comprehensive SAM analysis visualization.
    
    Args:
        sam_map: SAM values, shape (H, W), in radians
        pred: Predicted image, shape (C, H, W)
        target: Target image, shape (C, H, W)
        vegetation_mask: Optional vegetation mask, shape (H, W)
        output_path: Path to save figure
        title: Optional custom title
        show_rgb: Whether to show RGB composites
    """
    # Setup figure
    if show_rgb:
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Convert SAM to degrees for easier interpretation
    sam_deg = np.rad2deg(sam_map)
    
    # Apply vegetation mask if provided
    if vegetation_mask is not None:
        sam_masked = sam_map.copy()
        sam_masked[~vegetation_mask] = np.nan
        sam_deg_masked = np.rad2deg(sam_masked)
    else:
        sam_masked = sam_map
        sam_deg_masked = sam_deg
    
    # -----------------------------------------------------------------
    # Subplot 1: SAM Map (All pixels)
    # -----------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    
    im1 = ax1.imshow(sam_deg, cmap='RdYlGn_r', vmin=0, vmax=30)
    ax1.set_title('SAM Map (All Pixels)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Spectral Angle (degrees)', rotation=270, labelpad=20)
    
    # -----------------------------------------------------------------
    # Subplot 2: SAM Map (Vegetation Only)
    # -----------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    
    if vegetation_mask is not None:
        im2 = ax2.imshow(sam_deg_masked, cmap='RdYlGn_r', vmin=0, vmax=30)
        ax2.set_title(f'SAM Map (Vegetation, {vegetation_mask.sum()} px)', fontsize=12, fontweight='bold')
    else:
        im2 = ax2.imshow(sam_deg, cmap='RdYlGn_r', vmin=0, vmax=30)
        ax2.set_title('SAM Map (No Mask)', fontsize=12, fontweight='bold')
    
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Spectral Angle (degrees)', rotation=270, labelpad=20)
    
    # -----------------------------------------------------------------
    # Subplot 3: Vegetation Mask
    # -----------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    
    if vegetation_mask is not None:
        ax3.imshow(vegetation_mask, cmap='Greens', vmin=0, vmax=1)
        veg_percent = 100 * vegetation_mask.sum() / vegetation_mask.size
        ax3.set_title(f'Vegetation Mask ({veg_percent:.1f}%)', fontsize=12, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No Mask Applied', ha='center', va='center', fontsize=14)
        ax3.set_title('Vegetation Mask', fontsize=12, fontweight='bold')
    
    ax3.axis('off')
    
    # -----------------------------------------------------------------
    # Subplot 4: SAM Histogram (All)
    # -----------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    
    sam_flat = sam_deg.flatten()
    sam_flat = sam_flat[np.isfinite(sam_flat)]
    
    ax4.hist(sam_flat, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax4.axvline(sam_flat.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {sam_flat.mean():.2f}°')
    ax4.axvline(np.median(sam_flat), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(sam_flat):.2f}°')
    ax4.set_xlabel('Spectral Angle (degrees)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('SAM Distribution (All Pixels)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # -----------------------------------------------------------------
    # Subplot 5: SAM Histogram (Vegetation)
    # -----------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    
    if vegetation_mask is not None:
        sam_veg_flat = sam_deg_masked.flatten()
        sam_veg_flat = sam_veg_flat[np.isfinite(sam_veg_flat)]
        
        ax5.hist(sam_veg_flat, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax5.axvline(sam_veg_flat.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {sam_veg_flat.mean():.2f}°')
        ax5.axvline(np.median(sam_veg_flat), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(sam_veg_flat):.2f}°')
        ax5.set_xlabel('Spectral Angle (degrees)', fontsize=11)
        ax5.set_ylabel('Frequency', fontsize=11)
        ax5.set_title('SAM Distribution (Vegetation)', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No Vegetation Mask', ha='center', va='center', fontsize=14)
        ax5.set_title('SAM Distribution (Vegetation)', fontsize=12, fontweight='bold')
    
    # -----------------------------------------------------------------
    # Subplot 6: Statistics Table
    # -----------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Compute statistics
    stats_data = []
    
    # All pixels
    sam_mean_all = sam_flat.mean()
    sam_std_all = sam_flat.std()
    sam_median_all = np.median(sam_flat)
    sam_p90_all = np.percentile(sam_flat, 90)
    
    stats_data.append([
        'All Pixels',
        f'{sam_mean_all:.3f}°',
        f'{sam_std_all:.3f}°',
        f'{sam_median_all:.3f}°',
        f'{sam_p90_all:.3f}°',
        f'{len(sam_flat):,}'
    ])
    
    # Vegetation pixels
    if vegetation_mask is not None and len(sam_veg_flat) > 0:
        sam_mean_veg = sam_veg_flat.mean()
        sam_std_veg = sam_veg_flat.std()
        sam_median_veg = np.median(sam_veg_flat)
        sam_p90_veg = np.percentile(sam_veg_flat, 90)
        
        stats_data.append([
            'Vegetation',
            f'{sam_mean_veg:.3f}°',
            f'{sam_std_veg:.3f}°',
            f'{sam_median_veg:.3f}°',
            f'{sam_p90_veg:.3f}°',
            f'{len(sam_veg_flat):,}'
        ])
        
        # Difference
        stats_data.append([
            'Δ (Veg - All)',
            f'{sam_mean_veg - sam_mean_all:+.3f}°',
            f'{sam_std_veg - sam_std_all:+.3f}°',
            f'{sam_median_veg - sam_median_all:+.3f}°',
            '-',
            '-'
        ])
    
    col_labels = ['Region', 'Mean', 'Std', 'Median', 'P90', 'Pixels']
    
    table = ax6.table(
        cellText=stats_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0.3, 1, 0.6]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#E0E0E0')
        table[(0, i)].set_text_props(weight='bold')
    
    # Style rows
    for i in range(1, len(stats_data) + 1):
        if 'Δ' in stats_data[i-1][0]:
            for j in range(len(col_labels)):
                table[(i, j)].set_facecolor('#FFF9C4')
        else:
            color = '#F5F5F5' if i % 2 == 0 else 'white'
            for j in range(len(col_labels)):
                table[(i, j)].set_facecolor(color)
    
    ax6.set_title('SAM Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # -----------------------------------------------------------------
    # Optional: RGB Composites
    # -----------------------------------------------------------------
    if show_rgb and pred.shape[0] >= 4:
        # Extract RGB bands (assuming [B02, B03, B04, B08])
        # RGB = [B04, B03, B02] for true color
        
        def make_rgb(img):
            # img shape: (C, H, W)
            if img.shape[0] >= 4:
                rgb = np.stack([img[2], img[1], img[0]], axis=-1)  # R, G, B
            else:
                rgb = img[:3].transpose(1, 2, 0)
            
            # Normalize to [0, 1]
            rgb = np.clip(rgb, 0, 1)
            
            # Stretch for better visualization
            p2, p98 = np.percentile(rgb, (2, 98))
            rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)
            
            return rgb
        
        pred_rgb = make_rgb(pred)
        target_rgb = make_rgb(target)
        
        # Predicted RGB
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.imshow(pred_rgb)
        ax7.set_title('Predicted (RGB)', fontsize=12, fontweight='bold')
        ax7.axis('off')
        
        # Target RGB
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.imshow(target_rgb)
        ax8.set_title('Target (RGB)', fontsize=12, fontweight='bold')
        ax8.axis('off')
        
        # Difference (enhanced)
        ax9 = fig.add_subplot(gs[2, 2])
        diff_rgb = np.abs(pred_rgb - target_rgb)
        diff_rgb = np.clip(diff_rgb * 5, 0, 1)  # Enhance differences
        ax9.imshow(diff_rgb)
        ax9.set_title('Absolute Difference (×5)', fontsize=12, fontweight='bold')
        ax9.axis('off')
    
    # Save or show
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_single_image(
    pred_path: Path,
    target_path: Path,
    output_path: Optional[Path] = None,
    ndvi_threshold: float = 0.3,
    show_rgb: bool = True
):
    """
    Analyze SAM for a single image pair.
    
    Args:
        pred_path: Path to predicted image NPZ
        target_path: Path to target image NPZ
        output_path: Path to save visualization
        ndvi_threshold: NDVI threshold for vegetation mask
        show_rgb: Whether to show RGB composites
    """
    print(f"Analyzing single image pair:")
    print(f"  Predicted: {pred_path.name}")
    print(f"  Target: {target_path.name}")
    
    # Load predicted
    pred_data = np.load(pred_path)
    if 'image' in pred_data:
        pred = pred_data['image']
    else:
        pred = pred_data[list(pred_data.keys())[0]]
    pred = pred.astype(np.float32)
    
    # Load target
    target_data = np.load(target_path)
    if 'image' in target_data:
        target = target_data['image']
    else:
        target = target_data[list(target_data.keys())[0]]
    target = target.astype(np.float32)
    
    # Validate shapes
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
    assert pred.ndim == 3, f"Expected 3D array, got {pred.ndim}D"
    
    # Compute SAM
    print("  Computing SAM...")
    sam_map = compute_sam_per_pixel(pred, target)
    
    # Create vegetation mask
    if ndvi_threshold > 0:
        print(f"  Creating vegetation mask (NDVI >= {ndvi_threshold})...")
        # Assuming band order: [B02, B03, B04, B08]
        if pred.shape[0] >= 4:
            red = pred[2]
            nir = pred[3]
            ndvi = compute_ndvi(red, nir)
            vegetation_mask = create_vegetation_mask(ndvi, ndvi_threshold)
        else:
            print("  Warning: Not enough bands for NDVI, skipping mask")
            vegetation_mask = None
    else:
        vegetation_mask = None
    
    # Compute statistics
    sam_deg = np.rad2deg(sam_map)
    print(f"\n  SAM Statistics (All Pixels):")
    print(f"    Mean: {sam_deg.mean():.3f}°")
    print(f"    Std:  {sam_deg.std():.3f}°")
    print(f"    Median: {np.median(sam_deg):.3f}°")
    print(f"    P90: {np.percentile(sam_deg, 90):.3f}°")
    
    if vegetation_mask is not None:
        sam_veg = sam_deg[vegetation_mask]
        print(f"\n  SAM Statistics (Vegetation):")
        print(f"    Mean: {sam_veg.mean():.3f}°")
        print(f"    Std:  {sam_veg.std():.3f}°")
        print(f"    Median: {np.median(sam_veg):.3f}°")
        print(f"    P90: {np.percentile(sam_veg, 90):.3f}°")
        print(f"    Pixels: {len(sam_veg):,} ({100*len(sam_veg)/sam_deg.size:.1f}%)")
    
    # Visualize
    print("\n  Generating visualization...")
    plot_sam_analysis(
        sam_map,
        pred,
        target,
        vegetation_mask,
        output_path,
        title=f"SAM Analysis: {pred_path.stem}",
        show_rgb=show_rgb
    )


def analyze_batch(
    data_dir: Path,
    output_dir: Path,
    ndvi_threshold: float = 0.3,
    max_files: Optional[int] = None,
    pattern: str = '*.npz'
):
    """
    Analyze SAM for multiple image pairs in a directory.
    
    Args:
        data_dir: Directory containing NPZ files with pred/target pairs
        output_dir: Directory to save visualizations
        ndvi_threshold: NDVI threshold for vegetation mask
        max_files: Maximum number of files to process
        pattern: File pattern to match
    """
    print("=" * 80)
    print("SAM Batch Analysis")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"NDVI threshold: {ndvi_threshold}")
    print()
    
    # Find files
    files = sorted(data_dir.glob(pattern))
    if max_files:
        files = files[:max_files]
    
    print(f"Found {len(files)} files")
    print()
    
    if len(files) == 0:
        print("❌ No files found!")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate statistics
    all_sam_mean = []
    all_sam_veg_mean = []
    
    # Process each file
    for file_path in tqdm(files, desc="Processing files"):
        data = load_image_data(file_path)
        
        if data is None:
            continue
        
        pred = data['pred']
        target = data['target']
        
        # Compute SAM
        sam_map = compute_sam_per_pixel(pred, target)
        
        # Vegetation mask
        if ndvi_threshold > 0 and pred.shape[0] >= 4:
            red = pred[2]
            nir = pred[3]
            ndvi = compute_ndvi(red, nir)
            vegetation_mask = create_vegetation_mask(ndvi, ndvi_threshold)
        else:
            vegetation_mask = None
        
        # Statistics
        sam_deg = np.rad2deg(sam_map)
        all_sam_mean.append(sam_deg.mean())
        
        if vegetation_mask is not None:
            sam_veg = sam_deg[vegetation_mask]
            if len(sam_veg) > 0:
                all_sam_veg_mean.append(sam_veg.mean())
        
        # Save visualization
        output_path = output_dir / f"sam_analysis_{file_path.stem}.png"
        plot_sam_analysis(
            sam_map,
            pred,
            target,
            vegetation_mask,
            output_path,
            title=f"SAM Analysis: {file_path.stem}",
            show_rgb=False  # Faster for batch processing
        )
    
    # Summary
    print("\n" + "=" * 80)
    print("Batch Analysis Summary")
    print("=" * 80)
    print(f"Files processed: {len(all_sam_mean)}")
    print(f"\nSAM (All Pixels):")
    print(f"  Mean: {np.mean(all_sam_mean):.3f}° ± {np.std(all_sam_mean):.3f}°")
    print(f"  Range: [{np.min(all_sam_mean):.3f}°, {np.max(all_sam_mean):.3f}°]")
    
    if all_sam_veg_mean:
        print(f"\nSAM (Vegetation):")
        print(f"  Mean: {np.mean(all_sam_veg_mean):.3f}° ± {np.std(all_sam_veg_mean):.3f}°")
        print(f"  Range: [{np.min(all_sam_veg_mean):.3f}°, {np.max(all_sam_veg_mean):.3f}°]")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute Spectral Angle Mapper (SAM) metric',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image pair
  python tools/sam_metric.py --pred opt_v1.npz --target s2_real.npz --output reports/sam.png

  # With vegetation masking
  python tools/sam_metric.py --pred opt_v1.npz --target s2_real.npz --ndvi_threshold 0.3

  # Batch processing
  python tools/sam_metric.py --data_dir data/tiles --output_dir reports/sam_analysis

  # Batch with custom threshold
  python tools/sam_metric.py --data_dir data/tiles --output_dir reports/sam_veg --ndvi_threshold 0.4
        """
    )
    
    parser.add_argument(
        '--pred',
        type=Path,
        help='Predicted image NPZ file (for single image mode)'
    )
    
    parser.add_argument(
        '--target',
        type=Path,
        help='Target image NPZ file (for single image mode)'
    )
    
    parser.add_argument(
        '--data_dir',
        type=Path,
        help='Directory with pred/target pairs (for batch mode)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Output path for single image visualization'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('reports/sam_analysis'),
        help='Output directory for batch visualizations'
    )
    
    parser.add_argument(
        '--ndvi_threshold',
        type=float,
        default=0.3,
        help='NDVI threshold for vegetation mask (default: 0.3, 0 to disable)'
    )
    
    parser.add_argument(
        '--max_files',
        type=int,
        help='Maximum files to process in batch mode'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.npz',
        help='File pattern for batch mode (default: *.npz)'
    )
    
    parser.add_argument(
        '--show_rgb',
        action='store_true',
        help='Show RGB composites in visualization'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.data_dir is None and (args.pred is None or args.target is None):
        parser.error("Must specify either --data_dir or (--pred and --target)")
    
    # Run appropriate mode
    if args.data_dir is not None:
        # Batch mode
        if not args.data_dir.exists():
            print(f"❌ Data directory not found: {args.data_dir}")
            return 1
        
        analyze_batch(
            args.data_dir,
            args.output_dir,
            ndvi_threshold=args.ndvi_threshold,
            max_files=args.max_files,
            pattern=args.pattern
        )
    else:
        # Single image mode
        if not args.pred.exists():
            print(f"❌ Predicted image not found: {args.pred}")
            return 1
        
        if not args.target.exists():
            print(f"❌ Target image not found: {args.target}")
            return 1
        
        analyze_single_image(
            args.pred,
            args.target,
            output_path=args.output,
            ndvi_threshold=args.ndvi_threshold,
            show_rgb=args.show_rgb
        )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
