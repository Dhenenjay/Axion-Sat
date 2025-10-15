"""
Panel Visualization: Synthetic Optical (v1) vs Ground Truth S2 + SAR Edge Overlay

Creates side-by-side comparison panels showing:
- Left: Generated synthetic optical (opt_v1)
- Middle: Ground truth Sentinel-2 optical
- Right: SAR edge overlay on synthetic optical

This helps visually assess:
1. Overall color/texture quality of generated imagery
2. Alignment with ground truth
3. Edge preservation from SAR input

Usage:
    # Process all tiles in a directory
    python tools/panel_v1_vs_truth.py --tile-dir tiles/ --output-dir panels/
    
    # Process specific tiles
    python tools/panel_v1_vs_truth.py --tile-dir tiles/ --output-dir panels/ --tile-ids tile001 tile002
    
    # Adjust visualization
    python tools/panel_v1_vs_truth.py --tile-dir tiles/ --output-dir panels/ --edge-threshold 0.1 --dpi 150
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axs_lib.stdz import TERRAMIND_S1_STATS, TERRAMIND_S2_STATS


def load_tile_data(tile_path: Path) -> dict:
    """Load tile data from NPZ file.
    
    Args:
        tile_path: Path to NPZ tile file
        
    Returns:
        Dictionary with s1_vv, s1_vh, s2_b2, s2_b3, s2_b4, s2_b8, opt_v1 (if exists)
    """
    data = np.load(tile_path)
    
    result = {
        's1_vv': data['s1_vv'],
        's1_vh': data['s1_vh'],
        's2_b2': data['s2_b2'],
        's2_b3': data['s2_b3'],
        's2_b4': data['s2_b4'],
        's2_b8': data['s2_b8']
    }
    
    # Check for opt_v1 (may be in separate file or same file)
    if 'opt_v1_b2' in data:
        result['opt_v1_b2'] = data['opt_v1_b2']
        result['opt_v1_b3'] = data['opt_v1_b3']
        result['opt_v1_b4'] = data['opt_v1_b4']
        result['opt_v1_b8'] = data['opt_v1_b8']
    
    return result


def load_opt_v1(tile_path: Path, opt_v1_suffix: str = '_opt_v1') -> Optional[np.ndarray]:
    """Load synthetic optical v1 from separate file.
    
    Args:
        tile_path: Path to original tile file
        opt_v1_suffix: Suffix for opt_v1 file (default: '_opt_v1')
        
    Returns:
        Array of shape (4, H, W) with B02, B03, B04, B08, or None if not found
    """
    # Try loading from separate file
    opt_v1_path = tile_path.parent / f"{tile_path.stem}{opt_v1_suffix}.npz"
    
    if not opt_v1_path.exists():
        return None
    
    try:
        opt_data = np.load(opt_v1_path)
        
        # Check for band keys
        if all(k in opt_data for k in ['opt_v1_b2', 'opt_v1_b3', 'opt_v1_b4', 'opt_v1_b8']):
            return np.stack([
                opt_data['opt_v1_b2'],
                opt_data['opt_v1_b3'],
                opt_data['opt_v1_b4'],
                opt_data['opt_v1_b8']
            ])
        elif 'opt_v1' in opt_data:
            return opt_data['opt_v1']
        else:
            return None
    except Exception as e:
        print(f"Warning: Could not load opt_v1 from {opt_v1_path}: {e}")
        return None


def extract_sar_edges(s1_vv: np.ndarray, s1_vh: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Extract edge map from SAR imagery.
    
    Args:
        s1_vv: VV polarization (H, W)
        s1_vh: VH polarization (H, W)
        threshold: Edge detection threshold (default: 0.1)
        
    Returns:
        Binary edge map (H, W) with values in [0, 1]
    """
    # Normalize SAR to [0, 1] for edge detection
    vv_norm = (s1_vv - s1_vv.min()) / (s1_vv.max() - s1_vv.min() + 1e-8)
    vh_norm = (s1_vh - s1_vh.min()) / (s1_vh.max() - s1_vh.min() + 1e-8)
    
    # Combine polarizations
    sar_combined = 0.6 * vv_norm + 0.4 * vh_norm
    
    # Sobel edge detection
    sobel_x = cv2.Sobel(sar_combined, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(sar_combined, cv2.CV_64F, 0, 1, ksize=3)
    
    # Edge magnitude
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize and threshold
    edges = edges / (edges.max() + 1e-8)
    edges_binary = (edges > threshold).astype(np.float32)
    
    return edges_binary


def create_rgb_composite(b2: np.ndarray, b3: np.ndarray, b4: np.ndarray, 
                         percentile_clip: Tuple[float, float] = (2, 98)) -> np.ndarray:
    """Create RGB composite from Sentinel-2 bands.
    
    Args:
        b2: Blue band (already normalized to [0, 1])
        b3: Green band
        b4: Red band
        percentile_clip: Percentile clipping for contrast (default: (2, 98))
        
    Returns:
        RGB image (H, W, 3) with values in [0, 1]
    """
    # Stack RGB (R=B4, G=B3, B=B2)
    rgb = np.stack([b4, b3, b2], axis=-1)
    
    # Clip to [0, 1] range (data should already be normalized)
    rgb = np.clip(rgb, 0, 1)
    
    # Optional: Apply percentile stretching for better contrast
    if percentile_clip is not None:
        p_low, p_high = percentile_clip
        for i in range(3):
            vmin, vmax = np.percentile(rgb[..., i], [p_low, p_high])
            rgb[..., i] = np.clip((rgb[..., i] - vmin) / (vmax - vmin + 1e-8), 0, 1)
    
    return rgb


def overlay_edges_on_rgb(rgb: np.ndarray, edges: np.ndarray, 
                         edge_color: Tuple[float, float, float] = (1.0, 1.0, 0.0),
                         alpha: float = 0.5) -> np.ndarray:
    """Overlay SAR edges on RGB image.
    
    Args:
        rgb: RGB image (H, W, 3) in [0, 1]
        edges: Binary edge map (H, W) in [0, 1]
        edge_color: RGB color for edges (default: yellow)
        alpha: Blending factor for edges (default: 0.5)
        
    Returns:
        RGB image with edge overlay (H, W, 3)
    """
    overlay = rgb.copy()
    
    # Create colored edge overlay
    edge_mask = edges > 0.5
    for i in range(3):
        overlay[edge_mask, i] = (1 - alpha) * rgb[edge_mask, i] + alpha * edge_color[i]
    
    return overlay


def create_comparison_panel(
    tile_data: dict,
    opt_v1: np.ndarray,
    tile_id: str,
    edge_threshold: float = 0.1,
    figsize: Tuple[int, int] = (18, 6),
    dpi: int = 100
) -> plt.Figure:
    """Create side-by-side comparison panel.
    
    Args:
        tile_data: Dictionary with SAR and S2 bands
        opt_v1: Synthetic optical array (4, H, W) - B02, B03, B04, B08
        tile_id: Tile identifier for title
        edge_threshold: Threshold for SAR edge detection
        figsize: Figure size in inches
        dpi: Figure DPI
        
    Returns:
        Matplotlib figure
    """
    # Extract data
    s1_vv = tile_data['s1_vv']
    s1_vh = tile_data['s1_vh']
    s2_b2 = tile_data['s2_b2']
    s2_b3 = tile_data['s2_b3']
    s2_b4 = tile_data['s2_b4']
    
    # Extract SAR edges
    sar_edges = extract_sar_edges(s1_vv, s1_vh, threshold=edge_threshold)
    
    # Create RGB composites
    opt_v1_rgb = create_rgb_composite(opt_v1[0], opt_v1[1], opt_v1[2])
    s2_truth_rgb = create_rgb_composite(s2_b2, s2_b3, s2_b4)
    
    # Create edge overlay
    opt_v1_edges = overlay_edges_on_rgb(opt_v1_rgb, sar_edges, edge_color=(1.0, 1.0, 0.0), alpha=0.6)
    
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(1, 3, figure=fig, wspace=0.05, hspace=0.05)
    
    # Panel 1: Synthetic Optical (v1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(opt_v1_rgb)
    ax1.set_title('Synthetic Optical (v1)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Ground Truth S2
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(s2_truth_rgb)
    ax2.set_title('Ground Truth S2', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Panel 3: v1 + SAR Edge Overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(opt_v1_edges)
    ax3.set_title('v1 + SAR Edges', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Add legend for edge overlay
    edge_patch = mpatches.Patch(color=(1.0, 1.0, 0.0), label='SAR Edges')
    ax3.legend(handles=[edge_patch], loc='upper right', fontsize=8)
    
    # Overall title
    fig.suptitle(f'Tile: {tile_id} - Quality Comparison', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Add info text
    info_text = (
        f'Image size: {opt_v1_rgb.shape[0]}×{opt_v1_rgb.shape[1]} | '
        f'Edge threshold: {edge_threshold:.3f} | '
        f'RGB bands: R=B04, G=B03, B=B02'
    )
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    return fig


def process_tiles(
    tile_dir: Path,
    output_dir: Path,
    tile_ids: Optional[List[str]] = None,
    opt_v1_suffix: str = '_opt_v1',
    edge_threshold: float = 0.1,
    figsize: Tuple[int, int] = (18, 6),
    dpi: int = 100,
    file_format: str = 'png'
) -> int:
    """Process tiles and generate comparison panels.
    
    Args:
        tile_dir: Directory containing tile NPZ files
        output_dir: Directory to save panel images
        tile_ids: Optional list of specific tile IDs to process
        opt_v1_suffix: Suffix for opt_v1 files
        edge_threshold: SAR edge detection threshold
        figsize: Figure size
        dpi: Figure DPI
        file_format: Output image format (png, jpg, pdf)
        
    Returns:
        Number of panels generated
    """
    tile_dir = Path(tile_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all tile files
    if tile_ids is not None:
        tile_paths = [tile_dir / f"{tid}.npz" for tid in tile_ids]
        tile_paths = [p for p in tile_paths if p.exists()]
    else:
        tile_paths = sorted(list(tile_dir.glob("*.npz")))
        # Exclude opt_v1 files
        tile_paths = [p for p in tile_paths if opt_v1_suffix not in p.stem]
    
    if not tile_paths:
        print(f"No tiles found in {tile_dir}")
        return 0
    
    print(f"Found {len(tile_paths)} tiles to process")
    print(f"Output directory: {output_dir}")
    print(f"Edge threshold: {edge_threshold}")
    print(f"Figure size: {figsize[0]}×{figsize[1]} inches @ {dpi} DPI")
    print("=" * 80)
    
    processed = 0
    skipped = 0
    
    for tile_path in tile_paths:
        tile_id = tile_path.stem
        
        try:
            # Load tile data
            tile_data = load_tile_data(tile_path)
            
            # Load opt_v1
            opt_v1 = load_opt_v1(tile_path, opt_v1_suffix)
            
            if opt_v1 is None:
                print(f"⚠️  {tile_id}: opt_v1 not found, skipping")
                skipped += 1
                continue
            
            # Create panel
            print(f"Processing {tile_id}... ", end='', flush=True)
            
            fig = create_comparison_panel(
                tile_data=tile_data,
                opt_v1=opt_v1,
                tile_id=tile_id,
                edge_threshold=edge_threshold,
                figsize=figsize,
                dpi=dpi
            )
            
            # Save figure
            output_path = output_dir / f"{tile_id}_panel.{file_format}"
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight', format=file_format)
            plt.close(fig)
            
            print(f"✓ Saved to {output_path.name}")
            processed += 1
            
        except Exception as e:
            print(f"❌ {tile_id}: Error - {e}")
            skipped += 1
            continue
    
    print("=" * 80)
    print(f"Completed: {processed} panels generated, {skipped} skipped")
    
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Generate side-by-side comparison panels: v1 vs S2 truth + SAR edges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all tiles
  python tools/panel_v1_vs_truth.py --tile-dir tiles/ --output-dir panels/
  
  # Process specific tiles
  python tools/panel_v1_vs_truth.py --tile-dir tiles/ --output-dir panels/ --tile-ids tile001 tile002
  
  # High-quality output
  python tools/panel_v1_vs_truth.py --tile-dir tiles/ --output-dir panels/ --dpi 300 --format pdf
  
  # Adjust edge sensitivity
  python tools/panel_v1_vs_truth.py --tile-dir tiles/ --output-dir panels/ --edge-threshold 0.15
        """
    )
    
    # Required arguments
    parser.add_argument('--tile-dir', type=str, required=True,
                        help='Directory containing tile NPZ files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save panel images')
    
    # Optional arguments
    parser.add_argument('--tile-ids', type=str, nargs='+', default=None,
                        help='Specific tile IDs to process (default: all tiles)')
    parser.add_argument('--opt-v1-suffix', type=str, default='_opt_v1',
                        help='Suffix for opt_v1 files (default: _opt_v1)')
    parser.add_argument('--edge-threshold', type=float, default=0.1,
                        help='SAR edge detection threshold (default: 0.1)')
    parser.add_argument('--figsize', type=int, nargs=2, default=[18, 6],
                        help='Figure size in inches (width height, default: 18 6)')
    parser.add_argument('--dpi', type=int, default=100,
                        help='Output DPI (default: 100)')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'jpg', 'jpeg', 'pdf', 'svg'],
                        help='Output image format (default: png)')
    
    args = parser.parse_args()
    
    # Process tiles
    num_processed = process_tiles(
        tile_dir=args.tile_dir,
        output_dir=args.output_dir,
        tile_ids=args.tile_ids,
        opt_v1_suffix=args.opt_v1_suffix,
        edge_threshold=args.edge_threshold,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        file_format=args.format
    )
    
    if num_processed == 0:
        print("\nNo panels generated. Please check:")
        print("  1. Tile directory exists and contains NPZ files")
        print("  2. opt_v1 files exist (run inference first)")
        print("  3. Tile IDs are correct (if specified)")
        sys.exit(1)
    
    print(f"\n✓ Successfully generated {num_processed} comparison panels!")


if __name__ == '__main__':
    main()
