"""
Panel Visualization: Stage 3 Final vs Stage 2

Creates comprehensive side-by-side comparison panels showing:
1. Stage 2 (opt_v2) RGB composite
2. Stage 3 (opt_final) RGB composite
3. Difference heatmap (absolute changes)
4. SAR grayscale
5. Optical edges (Stage 2)
6. Optical edges (Stage 3)
7. SAR edges
8. Edge alignment overlay

This tool helps visualize the improvements from Stage 3 grounding.

Usage:
    # Single tile comparison
    python tools/panel_final_vs_v2.py \\
        --opt-v2 stage2_outputs/tile_001.npz \\
        --opt-final stage3_outputs/tile_001.npz \\
        --s1 tiles/s1/tile_001.npz \\
        --output comparison_tile_001.png

    # Batch processing
    python tools/panel_final_vs_v2.py \\
        --opt-v2-dir stage2_outputs/ \\
        --opt-final-dir stage3_outputs/ \\
        --s1-dir tiles/s1/ \\
        --output-dir comparisons/

    # With metrics display
    python tools/panel_final_vs_v2.py \\
        --opt-v2 stage2_outputs/tile_001.npz \\
        --opt-final stage3_outputs/tile_001.npz \\
        --s1 tiles/s1/tile_001.npz \\
        --output comparison.png \\
        --show-metrics

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
import os
from pathlib import Path
import argparse
from typing import Dict, Optional, Tuple, List
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Data Loading
# ============================================================================

def load_tile(tile_path: Path, keys: List[str]) -> Optional[np.ndarray]:
    """
    Load tile from NPZ file, trying multiple possible keys.
    
    Args:
        tile_path: Path to NPZ file
        keys: List of possible keys to try
        
    Returns:
        Array (C, H, W) or None if loading fails
    """
    try:
        data = np.load(tile_path)
        
        # Try each key
        for key in keys:
            if key in data:
                arr = data[key]
                if arr.ndim == 2:
                    # Single channel, expand based on expected type
                    if 's1' in keys or 'vv' in keys:
                        arr = np.stack([arr, arr], axis=0)  # 2 channels for SAR
                    else:
                        arr = np.stack([arr] * 4, axis=0)  # 4 channels for optical
                return arr.astype(np.float32)
        
        # Try loading individual bands for optical
        if all(f's2_b{b}' in data for b in [2, 3, 4, 8]):
            arr = np.stack([data['s2_b2'], data['s2_b3'], data['s2_b4'], data['s2_b8']], axis=0)
            return arr.astype(np.float32)
        
        # Try loading SAR bands
        if 's1_vv' in data and 's1_vh' in data:
            arr = np.stack([data['s1_vv'], data['s1_vh']], axis=0)
            return arr.astype(np.float32)
        
        warnings.warn(f"Could not find valid data in {tile_path}")
        return None
        
    except Exception as e:
        warnings.warn(f"Error loading {tile_path}: {e}")
        return None


# ============================================================================
# Image Processing
# ============================================================================

def make_rgb_composite(img: np.ndarray, percentile: float = 2) -> np.ndarray:
    """
    Create RGB composite from multi-channel image.
    
    Args:
        img: Input image (4, H, W) - assumes B, G, R, NIR
        percentile: Percentile for contrast stretching
        
    Returns:
        RGB image (H, W, 3) in [0, 1]
    """
    # Extract RGB channels (indices 2, 1, 0 = R, G, B)
    rgb = img[[2, 1, 0], :, :]
    rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)
    
    # Contrast stretching
    p_low = np.percentile(rgb, percentile)
    p_high = np.percentile(rgb, 100 - percentile)
    rgb = np.clip((rgb - p_low) / (p_high - p_low + 1e-8), 0, 1)
    
    return rgb


def compute_edges(img: np.ndarray) -> np.ndarray:
    """
    Compute edge magnitude using Sobel operator.
    
    Args:
        img: Grayscale image (H, W)
        
    Returns:
        Edge magnitude (H, W)
    """
    edges = ndimage.sobel(img)
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
    return edges


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale.
    
    Args:
        img: RGB image (3, H, W) or (4, H, W)
        
    Returns:
        Grayscale image (H, W)
    """
    if img.shape[0] >= 3:
        # Use standard RGB to grayscale conversion
        gray = 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]
    else:
        gray = img.mean(axis=0)
    
    return gray


def create_difference_heatmap(
    img1: np.ndarray,
    img2: np.ndarray,
    mode: str = 'absolute'
) -> np.ndarray:
    """
    Create difference heatmap between two images.
    
    Args:
        img1: First image (C, H, W)
        img2: Second image (C, H, W)
        mode: 'absolute' or 'signed'
        
    Returns:
        Difference map (H, W)
    """
    if mode == 'absolute':
        diff = np.abs(img2 - img1).mean(axis=0)
    else:
        diff = (img2 - img1).mean(axis=0)
    
    return diff


def create_edge_overlay(
    rgb: np.ndarray,
    edges: np.ndarray,
    color: Tuple[float, float, float] = (0, 1, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay edges on RGB image.
    
    Args:
        rgb: RGB image (H, W, 3)
        edges: Edge map (H, W)
        color: Edge color (R, G, B)
        alpha: Edge opacity
        
    Returns:
        RGB image with edge overlay (H, W, 3)
    """
    overlay = rgb.copy()
    
    # Create colored edge map
    edge_colored = np.zeros_like(rgb)
    for i in range(3):
        edge_colored[:, :, i] = edges * color[i]
    
    # Blend with original
    mask = edges[:, :, np.newaxis] > 0.1  # Threshold
    overlay = np.where(mask, (1 - alpha) * rgb + alpha * edge_colored, rgb)
    
    return np.clip(overlay, 0, 1)


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_sar_edge_agreement(opt: np.ndarray, s1: np.ndarray) -> float:
    """
    Compute SAR-optical edge agreement.
    
    Args:
        opt: Optical image (4, H, W)
        s1: SAR image (2, H, W)
        
    Returns:
        Agreement score [0, 1]
    """
    # Convert to grayscale
    opt_gray = to_grayscale(opt)
    s1_gray = s1.mean(axis=0)
    
    # Compute edges
    opt_edges = compute_edges(opt_gray)
    s1_edges = compute_edges(s1_gray)
    
    # Compute correlation
    opt_flat = opt_edges.flatten()
    s1_flat = s1_edges.flatten()
    
    # Normalize
    opt_flat = opt_flat / (np.linalg.norm(opt_flat) + 1e-8)
    s1_flat = s1_flat / (np.linalg.norm(s1_flat) + 1e-8)
    
    # Cosine similarity
    agreement = np.dot(opt_flat, s1_flat)
    
    return float(agreement)


def compute_metrics(
    opt_v2: np.ndarray,
    opt_final: np.ndarray,
    s1: np.ndarray
) -> Dict[str, float]:
    """
    Compute comparison metrics.
    
    Args:
        opt_v2: Stage 2 output (4, H, W)
        opt_final: Stage 3 output (4, H, W)
        s1: SAR input (2, H, W)
        
    Returns:
        Dict with metrics
    """
    metrics = {}
    
    # SAR edge agreement
    metrics['sar_agreement_v2'] = compute_sar_edge_agreement(opt_v2, s1)
    metrics['sar_agreement_final'] = compute_sar_edge_agreement(opt_final, s1)
    metrics['sar_improvement'] = metrics['sar_agreement_final'] - metrics['sar_agreement_v2']
    
    # Change statistics
    diff = np.abs(opt_final - opt_v2)
    metrics['mean_change'] = float(diff.mean())
    metrics['max_change'] = float(diff.max())
    metrics['std_change'] = float(diff.std())
    
    # Per-channel changes
    channel_names = ['blue', 'green', 'red', 'nir']
    for i, name in enumerate(channel_names):
        metrics[f'{name}_change'] = float(np.abs(opt_final[i] - opt_v2[i]).mean())
    
    return metrics


# ============================================================================
# Visualization
# ============================================================================

def create_comparison_panel(
    opt_v2: np.ndarray,
    opt_final: np.ndarray,
    s1: np.ndarray,
    tile_name: str = "Tile",
    show_metrics: bool = True,
    figsize: Tuple[int, int] = (20, 12)
) -> plt.Figure:
    """
    Create comprehensive comparison panel.
    
    Layout (3 rows × 3 columns):
        Row 1: RGB composites and difference
        Row 2: SAR and edge maps
        Row 3: Edge overlays and metrics
    
    Args:
        opt_v2: Stage 2 output (4, H, W)
        opt_final: Stage 3 output (4, H, W)
        s1: SAR input (2, H, W)
        tile_name: Name for title
        show_metrics: Whether to show metrics panel
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    # Compute all necessary data
    rgb_v2 = make_rgb_composite(opt_v2)
    rgb_final = make_rgb_composite(opt_final)
    
    gray_v2 = to_grayscale(opt_v2)
    gray_final = to_grayscale(opt_final)
    gray_s1 = s1.mean(axis=0)
    
    edges_v2 = compute_edges(gray_v2)
    edges_final = compute_edges(gray_final)
    edges_s1 = compute_edges(gray_s1)
    
    diff_map = create_difference_heatmap(opt_v2, opt_final, mode='absolute')
    
    # Compute metrics
    if show_metrics:
        metrics = compute_metrics(opt_v2, opt_final, s1)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ========================================================================
    # Row 1: RGB Composites and Difference
    # ========================================================================
    
    # Stage 2 RGB
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_v2)
    ax1.set_title('Stage 2 (opt_v2)\nRGB Composite', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Stage 3 RGB
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rgb_final)
    ax2.set_title('Stage 3 (opt_final)\nRGB Composite', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Difference heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(diff_map, cmap='hot', vmin=0, vmax=np.percentile(diff_map, 95))
    ax3.set_title(f'Absolute Difference\nMean: {diff_map.mean():.4f}, Max: {diff_map.max():.4f}',
                  fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # ========================================================================
    # Row 2: SAR and Edge Maps
    # ========================================================================
    
    # SAR grayscale
    ax4 = fig.add_subplot(gs[1, 0])
    s1_vis = (gray_s1 - gray_s1.min()) / (gray_s1.max() - gray_s1.min() + 1e-8)
    ax4.imshow(s1_vis, cmap='gray')
    ax4.set_title('SAR (S1)\nMean of VV & VH', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Edges comparison (V2 and Final)
    ax5 = fig.add_subplot(gs[1, 1])
    # Create RGB visualization: R=edges_v2, G=edges_final, B=overlap
    edge_rgb = np.stack([
        edges_v2,  # Red: Stage 2 edges
        edges_final,  # Green: Stage 3 edges
        np.minimum(edges_v2, edges_final)  # Blue: Overlap
    ], axis=-1)
    ax5.imshow(edge_rgb)
    ax5.set_title('Edge Comparison\nRed: V2, Green: Final, Blue: Overlap',
                  fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # SAR edges
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(edges_s1, cmap='viridis')
    ax6.set_title('SAR Edges\nSobel Magnitude', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # ========================================================================
    # Row 3: Edge Overlays and Metrics
    # ========================================================================
    
    # Stage 2 with SAR edges overlay
    ax7 = fig.add_subplot(gs[2, 0])
    overlay_v2 = create_edge_overlay(rgb_v2, edges_s1, color=(1, 0, 0), alpha=0.6)
    ax7.imshow(overlay_v2)
    if show_metrics:
        ax7.set_title(f'Stage 2 + SAR Edges\nSAR Agr: {metrics["sar_agreement_v2"]:.4f}',
                      fontsize=12, fontweight='bold')
    else:
        ax7.set_title('Stage 2 + SAR Edges (Red)', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    # Stage 3 with SAR edges overlay
    ax8 = fig.add_subplot(gs[2, 1])
    overlay_final = create_edge_overlay(rgb_final, edges_s1, color=(0, 1, 0), alpha=0.6)
    ax8.imshow(overlay_final)
    if show_metrics:
        ax8.set_title(f'Stage 3 + SAR Edges\nSAR Agr: {metrics["sar_agreement_final"]:.4f}',
                      fontsize=12, fontweight='bold')
    else:
        ax8.set_title('Stage 3 + SAR Edges (Green)', fontsize=12, fontweight='bold')
    ax8.axis('off')
    
    # Metrics panel
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    if show_metrics:
        metrics_text = f"""
Tile: {tile_name}

SAR Edge Agreement:
  Stage 2:     {metrics['sar_agreement_v2']:.6f}
  Stage 3:     {metrics['sar_agreement_final']:.6f}
  Improvement: {metrics['sar_improvement']:.6f}
  {'↑' if metrics['sar_improvement'] > 0 else '↓'} {abs(metrics['sar_improvement']*100):.2f}%

Change Statistics:
  Mean:  {metrics['mean_change']:.6f}
  Max:   {metrics['max_change']:.6f}
  Std:   {metrics['std_change']:.6f}

Per-Channel Changes:
  Blue:  {metrics['blue_change']:.6f}
  Green: {metrics['green_change']:.6f}
  Red:   {metrics['red_change']:.6f}
  NIR:   {metrics['nir_change']:.6f}

Status:
  {'✓ SAR agreement improved' if metrics['sar_improvement'] > 0 else '⚠ SAR agreement decreased'}
  {'✓ Changes moderate' if metrics['mean_change'] < 0.1 else '⚠ Large changes'}
        """
        
        ax9.text(0.05, 0.95, metrics_text.strip(),
                transform=ax9.transAxes,
                fontsize=10,
                verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    else:
        ax9.text(0.5, 0.5, f'Tile: {tile_name}',
                transform=ax9.transAxes,
                fontsize=14,
                ha='center',
                va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # ========================================================================
    # Main Title
    # ========================================================================
    
    title_color = 'green' if (show_metrics and metrics['sar_improvement'] > 0) else 'black'
    
    if show_metrics:
        fig.suptitle(
            f"Stage 3 Final vs Stage 2 Comparison: {tile_name}\n"
            f"SAR Agreement: {metrics['sar_agreement_v2']:.3f} → {metrics['sar_agreement_final']:.3f} "
            f"({'↑' if metrics['sar_improvement'] > 0 else '↓'} {abs(metrics['sar_improvement']):.3f})",
            fontsize=16,
            fontweight='bold',
            color=title_color
        )
    else:
        fig.suptitle(
            f"Stage 3 Final vs Stage 2 Comparison: {tile_name}",
            fontsize=16,
            fontweight='bold'
        )
    
    return fig


# ============================================================================
# Batch Processing
# ============================================================================

def find_matching_tiles(
    opt_v2_dir: Path,
    opt_final_dir: Path,
    s1_dir: Path
) -> List[Dict[str, Path]]:
    """
    Find matching tiles across directories.
    
    Returns:
        List of dicts with tile paths
    """
    opt_v2_tiles = {p.name: p for p in opt_v2_dir.rglob('*.npz')}
    opt_final_tiles = {p.name: p for p in opt_final_dir.rglob('*.npz')}
    s1_tiles = {p.name: p for p in s1_dir.rglob('*.npz')}
    
    # Find common tiles
    common_names = set(opt_v2_tiles.keys()) & set(opt_final_tiles.keys()) & set(s1_tiles.keys())
    
    tiles = []
    for name in sorted(common_names):
        tiles.append({
            'name': name,
            'opt_v2': opt_v2_tiles[name],
            'opt_final': opt_final_tiles[name],
            's1': s1_tiles[name]
        })
    
    return tiles


def process_batch(
    opt_v2_dir: Path,
    opt_final_dir: Path,
    s1_dir: Path,
    output_dir: Path,
    show_metrics: bool = True,
    max_tiles: Optional[int] = None
):
    """
    Process multiple tiles in batch.
    
    Args:
        opt_v2_dir: Directory with Stage 2 outputs
        opt_final_dir: Directory with Stage 3 outputs
        s1_dir: Directory with SAR inputs
        output_dir: Output directory for comparisons
        show_metrics: Show metrics panel
        max_tiles: Maximum number of tiles to process
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find matching tiles
    tiles = find_matching_tiles(opt_v2_dir, opt_final_dir, s1_dir)
    
    if len(tiles) == 0:
        print("❌ No matching tiles found")
        return
    
    if max_tiles:
        tiles = tiles[:max_tiles]
    
    print(f"\nProcessing {len(tiles)} tiles...")
    
    for i, tile_dict in enumerate(tiles, 1):
        tile_name = tile_dict['name']
        print(f"  [{i}/{len(tiles)}] Processing {tile_name}...")
        
        # Load data
        opt_v2 = load_tile(tile_dict['opt_v2'], ['opt_v2', 's2_b2'])
        opt_final = load_tile(tile_dict['opt_final'], ['opt_final', 'opt_v3'])
        s1 = load_tile(tile_dict['s1'], ['s1', 's1_vv'])
        
        if opt_v2 is None or opt_final is None or s1 is None:
            print(f"    ⚠ Skipping {tile_name} (loading failed)")
            continue
        
        # Create panel
        try:
            fig = create_comparison_panel(
                opt_v2, opt_final, s1,
                tile_name=tile_name,
                show_metrics=show_metrics
            )
            
            # Save
            output_path = output_dir / tile_name.replace('.npz', '_comparison.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"    ✓ Saved to {output_path}")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    print(f"\n✓ Batch processing complete: {len(tiles)} tiles")
    print(f"  Output directory: {output_dir}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create comparison panels for Stage 3 final vs Stage 2"
    )
    
    # Single tile mode
    parser.add_argument('--opt-v2', type=str, help='Stage 2 output NPZ file')
    parser.add_argument('--opt-final', type=str, help='Stage 3 output NPZ file')
    parser.add_argument('--s1', type=str, help='SAR input NPZ file')
    parser.add_argument('--output', type=str, help='Output PNG file')
    
    # Batch mode
    parser.add_argument('--opt-v2-dir', type=str, help='Directory with Stage 2 outputs')
    parser.add_argument('--opt-final-dir', type=str, help='Directory with Stage 3 outputs')
    parser.add_argument('--s1-dir', type=str, help='Directory with SAR inputs')
    parser.add_argument('--output-dir', type=str, help='Output directory for comparisons')
    
    # Options
    parser.add_argument('--show-metrics', action='store_true', default=True,
                       help='Show metrics panel (default: True)')
    parser.add_argument('--no-metrics', action='store_true',
                       help='Hide metrics panel')
    parser.add_argument('--max-tiles', type=int, default=None,
                       help='Maximum number of tiles to process in batch mode')
    parser.add_argument('--figsize', type=str, default='20,12',
                       help='Figure size as "width,height" (default: 20,12)')
    
    args = parser.parse_args()
    
    # Parse figure size
    figsize = tuple(map(int, args.figsize.split(',')))
    
    # Determine show_metrics
    show_metrics = args.show_metrics and not args.no_metrics
    
    # Single tile mode
    if args.opt_v2 and args.opt_final and args.s1 and args.output:
        print("\n" + "=" * 80)
        print("Creating Comparison Panel (Single Tile)")
        print("=" * 80)
        
        # Load data
        print(f"\nLoading data...")
        opt_v2 = load_tile(Path(args.opt_v2), ['opt_v2', 's2_b2'])
        opt_final = load_tile(Path(args.opt_final), ['opt_final', 'opt_v3'])
        s1 = load_tile(Path(args.s1), ['s1', 's1_vv'])
        
        if opt_v2 is None or opt_final is None or s1 is None:
            print("❌ Failed to load one or more input files")
            return 1
        
        print(f"  opt_v2: {opt_v2.shape}")
        print(f"  opt_final: {opt_final.shape}")
        print(f"  s1: {s1.shape}")
        
        # Create panel
        print(f"\nCreating comparison panel...")
        tile_name = Path(args.opt_v2).stem
        
        fig = create_comparison_panel(
            opt_v2, opt_final, s1,
            tile_name=tile_name,
            show_metrics=show_metrics,
            figsize=figsize
        )
        
        # Save
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ Saved to {output_path}")
        print("=" * 80)
        
        return 0
    
    # Batch mode
    elif args.opt_v2_dir and args.opt_final_dir and args.s1_dir and args.output_dir:
        print("\n" + "=" * 80)
        print("Creating Comparison Panels (Batch Mode)")
        print("=" * 80)
        
        process_batch(
            Path(args.opt_v2_dir),
            Path(args.opt_final_dir),
            Path(args.s1_dir),
            Path(args.output_dir),
            show_metrics=show_metrics,
            max_tiles=args.max_tiles
        )
        
        print("=" * 80)
        
        return 0
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
