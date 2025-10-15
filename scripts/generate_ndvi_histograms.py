"""
Generate NDVI/EVI Histogram Comparisons

This script generates histogram plots comparing:
- Stage 1 (opt_v1): TerraMind SAR→Optical output
- Stage 2 (opt_v2): Prithvi-refined output
- Baseline (S2/HLS): Real Sentinel-2 or HLS imagery

Histograms are saved to reports/ndvi_hist_* and reports/evi_hist_*

The visualizations help assess:
1. How well Stage 1 approximates real spectral distributions
2. Whether Stage 2 refinement brings distributions closer to reality
3. Seasonal/biome-specific performance

Usage:
    python scripts/generate_ndvi_histograms.py --data_dir data/tiles/benv2_catalog --output_dir reports

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from axs_lib.stage2_losses import compute_ndvi, compute_evi
    LOSSES_AVAILABLE = True
except ImportError:
    LOSSES_AVAILABLE = False
    print("Warning: Could not import stage2_losses. Using fallback implementations.")


# ============================================================================
# Fallback NDVI/EVI Computation
# ============================================================================

def compute_ndvi_fallback(red: np.ndarray, nir: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute NDVI: (NIR - Red) / (NIR + Red)"""
    numerator = nir - red
    denominator = nir + red + eps
    return numerator / denominator


def compute_evi_fallback(
    blue: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0,
    eps: float = 1e-8
) -> np.ndarray:
    """Compute EVI: G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)"""
    numerator = G * (nir - red)
    denominator = nir + C1 * red - C2 * blue + L + eps
    return numerator / denominator


# ============================================================================
# Data Loading
# ============================================================================

def load_tile_data(tile_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Load tile data from NPZ file.
    
    Expected structure:
        - opt_v1: Stage 1 output (4, H, W) - [B02, B03, B04, B08]
        - opt_v2: Stage 2 output (4, H, W) - [B02, B03, B04, B08]
        - s2_real: Real S2 imagery (4, H, W) - [B02, B03, B04, B08]
        
    Returns:
        Dict with 'opt_v1', 'opt_v2', 's2_real' arrays, or None if loading fails
    """
    try:
        data = np.load(tile_path)
        
        # Extract arrays
        result = {}
        
        # Try different naming conventions
        if 'opt_v1' in data:
            result['opt_v1'] = data['opt_v1'].astype(np.float32)
        elif 'stage1' in data:
            result['opt_v1'] = data['stage1'].astype(np.float32)
        else:
            print(f"  Warning: No opt_v1 data in {tile_path.name}")
            return None
        
        if 'opt_v2' in data:
            result['opt_v2'] = data['opt_v2'].astype(np.float32)
        elif 'stage2' in data:
            result['opt_v2'] = data['stage2'].astype(np.float32)
        else:
            print(f"  Warning: No opt_v2 data in {tile_path.name}")
            # Stage 2 might not exist yet
            result['opt_v2'] = None
        
        if 's2_real' in data:
            result['s2_real'] = data['s2_real'].astype(np.float32)
        elif 'real' in data:
            result['s2_real'] = data['real'].astype(np.float32)
        elif 's2' in data:
            result['s2_real'] = data['s2'].astype(np.float32)
        else:
            print(f"  Warning: No S2 baseline data in {tile_path.name}")
            result['s2_real'] = None
        
        # Validate shapes
        for key, arr in result.items():
            if arr is not None:
                if arr.ndim != 3 or arr.shape[0] != 4:
                    print(f"  Warning: Invalid shape for {key}: {arr.shape}")
                    return None
        
        return result
        
    except Exception as e:
        print(f"  Error loading {tile_path}: {e}")
        return None


def extract_bands(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract bands from (4, H, W) image.
    
    Returns:
        (blue, green, red, nir) as (H, W) arrays
    """
    assert img.shape[0] == 4, f"Expected 4 channels, got {img.shape[0]}"
    
    blue = img[0]  # B02
    green = img[1]  # B03
    red = img[2]  # B04
    nir = img[3]  # B08
    
    return blue, green, red, nir


# ============================================================================
# Histogram Generation
# ============================================================================

def compute_indices_for_images(
    opt_v1: Optional[np.ndarray],
    opt_v2: Optional[np.ndarray],
    s2_real: Optional[np.ndarray]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute NDVI and EVI for all available images.
    
    Returns:
        Dict[image_name][index_name] = flattened array
    """
    results = {}
    
    for name, img in [('opt_v1', opt_v1), ('opt_v2', opt_v2), ('s2_real', s2_real)]:
        if img is None:
            continue
        
        # Extract bands
        blue, green, red, nir = extract_bands(img)
        
        # Compute indices
        if LOSSES_AVAILABLE:
            # Use torch implementation
            img_torch = torch.from_numpy(img).unsqueeze(0)  # Add batch dim
            ndvi_torch = compute_ndvi(img_torch[:, 2:3], img_torch[:, 3:4])
            evi_torch = compute_evi(img_torch[:, 0:1], img_torch[:, 2:3], img_torch[:, 3:4])
            
            ndvi = ndvi_torch.squeeze().numpy()
            evi = evi_torch.squeeze().numpy()
        else:
            # Use fallback numpy implementation
            ndvi = compute_ndvi_fallback(red, nir)
            evi = compute_evi_fallback(blue, red, nir)
        
        # Flatten and filter valid values
        ndvi_flat = ndvi.flatten()
        evi_flat = evi.flatten()
        
        # Remove NaN/Inf
        ndvi_flat = ndvi_flat[np.isfinite(ndvi_flat)]
        evi_flat = evi_flat[np.isfinite(evi_flat)]
        
        # Clip to valid range [-1, 1]
        ndvi_flat = np.clip(ndvi_flat, -1.0, 1.0)
        evi_flat = np.clip(evi_flat, -1.0, 1.0)
        
        results[name] = {
            'ndvi': ndvi_flat,
            'evi': evi_flat
        }
    
    return results


def plot_histogram_comparison(
    indices_data: Dict[str, Dict[str, np.ndarray]],
    index_name: str,
    output_path: Path,
    title: Optional[str] = None
):
    """
    Plot histogram comparison for a single index (NDVI or EVI).
    
    Args:
        indices_data: Dict[image_name][index_name] = flattened values
        index_name: 'ndvi' or 'evi'
        output_path: Path to save figure
        title: Optional custom title
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # Extract data
    v1_data = indices_data.get('opt_v1', {}).get(index_name, np.array([]))
    v2_data = indices_data.get('opt_v2', {}).get(index_name, np.array([]))
    s2_data = indices_data.get('s2_real', {}).get(index_name, np.array([]))
    
    has_v1 = len(v1_data) > 0
    has_v2 = len(v2_data) > 0
    has_s2 = len(s2_data) > 0
    
    # Colors
    color_v1 = '#FF6B6B'  # Red
    color_v2 = '#4ECDC4'  # Teal
    color_s2 = '#45B7D1'  # Blue
    
    # Set title
    if title is None:
        title = f"{index_name.upper()} Distribution Comparison"
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Bins
    bins = np.linspace(-1.0, 1.0, 80)
    
    # -----------------------------------------------------------------
    # Subplot 1: Overlaid histograms (all three)
    # -----------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    
    if has_v1:
        ax1.hist(v1_data, bins=bins, alpha=0.5, color=color_v1, label='Stage 1 (opt_v1)', density=True)
    if has_v2:
        ax1.hist(v2_data, bins=bins, alpha=0.5, color=color_v2, label='Stage 2 (opt_v2)', density=True)
    if has_s2:
        ax1.hist(s2_data, bins=bins, alpha=0.6, color=color_s2, label='S2/HLS Baseline', density=True, 
                 histtype='step', linewidth=2)
    
    ax1.set_xlabel(f'{index_name.upper()} Value', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Overlaid Distributions', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # -----------------------------------------------------------------
    # Subplot 2: Stage 1 vs Baseline
    # -----------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    
    if has_v1:
        ax2.hist(v1_data, bins=bins, alpha=0.6, color=color_v1, label='Stage 1', density=True)
    if has_s2:
        ax2.hist(s2_data, bins=bins, alpha=0.6, color=color_s2, label='Baseline', density=True,
                 histtype='step', linewidth=2)
    
    ax2.set_xlabel(f'{index_name.upper()} Value', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Stage 1 vs Baseline', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # -----------------------------------------------------------------
    # Subplot 3: Stage 2 vs Baseline
    # -----------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    
    if has_v2:
        ax3.hist(v2_data, bins=bins, alpha=0.6, color=color_v2, label='Stage 2', density=True)
    if has_s2:
        ax3.hist(s2_data, bins=bins, alpha=0.6, color=color_s2, label='Baseline', density=True,
                 histtype='step', linewidth=2)
    
    ax3.set_xlabel(f'{index_name.upper()} Value', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Stage 2 vs Baseline', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # -----------------------------------------------------------------
    # Subplot 4: Statistics Table
    # -----------------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Compute statistics
    stats_data = []
    
    if has_v1:
        stats_data.append([
            'Stage 1 (opt_v1)',
            f'{v1_data.mean():.4f}',
            f'{v1_data.std():.4f}',
            f'{np.percentile(v1_data, 25):.4f}',
            f'{np.percentile(v1_data, 50):.4f}',
            f'{np.percentile(v1_data, 75):.4f}',
            f'{v1_data.min():.4f}',
            f'{v1_data.max():.4f}'
        ])
    
    if has_v2:
        stats_data.append([
            'Stage 2 (opt_v2)',
            f'{v2_data.mean():.4f}',
            f'{v2_data.std():.4f}',
            f'{np.percentile(v2_data, 25):.4f}',
            f'{np.percentile(v2_data, 50):.4f}',
            f'{np.percentile(v2_data, 75):.4f}',
            f'{v2_data.min():.4f}',
            f'{v2_data.max():.4f}'
        ])
    
    if has_s2:
        stats_data.append([
            'S2/HLS Baseline',
            f'{s2_data.mean():.4f}',
            f'{s2_data.std():.4f}',
            f'{np.percentile(s2_data, 25):.4f}',
            f'{np.percentile(s2_data, 50):.4f}',
            f'{np.percentile(s2_data, 75):.4f}',
            f'{s2_data.min():.4f}',
            f'{s2_data.max():.4f}'
        ])
    
    # Add difference rows if both Stage 1/2 and baseline exist
    if has_v1 and has_s2:
        mean_diff = v1_data.mean() - s2_data.mean()
        std_diff = v1_data.std() - s2_data.std()
        stats_data.append([
            'Δ (Stage 1 - Baseline)',
            f'{mean_diff:+.4f}',
            f'{std_diff:+.4f}',
            '-', '-', '-', '-', '-'
        ])
    
    if has_v2 and has_s2:
        mean_diff = v2_data.mean() - s2_data.mean()
        std_diff = v2_data.std() - s2_data.std()
        stats_data.append([
            'Δ (Stage 2 - Baseline)',
            f'{mean_diff:+.4f}',
            f'{std_diff:+.4f}',
            '-', '-', '-', '-', '-'
        ])
    
    # Create table
    col_labels = ['Source', 'Mean', 'Std', 'Q25', 'Median', 'Q75', 'Min', 'Max']
    
    table = ax4.table(
        cellText=stats_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#E0E0E0')
        table[(0, i)].set_text_props(weight='bold')
    
    # Style rows
    for i in range(1, len(stats_data) + 1):
        if 'Δ' in stats_data[i-1][0]:
            # Difference rows in light yellow
            for j in range(len(col_labels)):
                table[(i, j)].set_facecolor('#FFF9C4')
        else:
            # Alternate row colors
            color = '#F5F5F5' if i % 2 == 0 else 'white'
            for j in range(len(col_labels)):
                table[(i, j)].set_facecolor(color)
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# Aggregate Analysis
# ============================================================================

def generate_aggregate_histograms(
    data_dir: Path,
    output_dir: Path,
    max_tiles: Optional[int] = None,
    pattern: str = '*.npz'
):
    """
    Generate aggregate histograms across multiple tiles.
    
    Args:
        data_dir: Directory containing tile NPZ files
        output_dir: Directory to save histogram plots
        max_tiles: Maximum number of tiles to process (None = all)
        pattern: File pattern to match (default: *.npz)
    """
    print("=" * 80)
    print("NDVI/EVI Histogram Generation")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Find tile files
    tile_files = sorted(data_dir.glob(pattern))
    
    if max_tiles is not None:
        tile_files = tile_files[:max_tiles]
    
    print(f"Found {len(tile_files)} tile files")
    print()
    
    if len(tile_files) == 0:
        print("❌ No tile files found!")
        return
    
    # Aggregate data
    print("Loading tiles and computing indices...")
    
    all_v1_ndvi = []
    all_v1_evi = []
    all_v2_ndvi = []
    all_v2_evi = []
    all_s2_ndvi = []
    all_s2_evi = []
    
    loaded_count = 0
    
    for tile_file in tqdm(tile_files, desc="Processing tiles"):
        tile_data = load_tile_data(tile_file)
        
        if tile_data is None:
            continue
        
        # Compute indices
        indices = compute_indices_for_images(
            tile_data.get('opt_v1'),
            tile_data.get('opt_v2'),
            tile_data.get('s2_real')
        )
        
        # Aggregate
        if 'opt_v1' in indices:
            all_v1_ndvi.append(indices['opt_v1']['ndvi'])
            all_v1_evi.append(indices['opt_v1']['evi'])
        
        if 'opt_v2' in indices:
            all_v2_ndvi.append(indices['opt_v2']['ndvi'])
            all_v2_evi.append(indices['opt_v2']['evi'])
        
        if 's2_real' in indices:
            all_s2_ndvi.append(indices['s2_real']['ndvi'])
            all_s2_evi.append(indices['s2_real']['evi'])
        
        loaded_count += 1
    
    print(f"\n✓ Successfully processed {loaded_count}/{len(tile_files)} tiles")
    print()
    
    if loaded_count == 0:
        print("❌ No valid tiles loaded!")
        return
    
    # Concatenate all data
    aggregated = {
        'opt_v1': {
            'ndvi': np.concatenate(all_v1_ndvi) if all_v1_ndvi else np.array([]),
            'evi': np.concatenate(all_v1_evi) if all_v1_evi else np.array([])
        },
        'opt_v2': {
            'ndvi': np.concatenate(all_v2_ndvi) if all_v2_ndvi else np.array([]),
            'evi': np.concatenate(all_v2_evi) if all_v2_evi else np.array([])
        },
        's2_real': {
            'ndvi': np.concatenate(all_s2_ndvi) if all_s2_ndvi else np.array([]),
            'evi': np.concatenate(all_s2_evi) if all_s2_evi else np.array([])
        }
    }
    
    # Generate histogram plots
    print("Generating histogram plots...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # NDVI histogram
    ndvi_path = output_dir / f'ndvi_hist_aggregate_{loaded_count}tiles.png'
    plot_histogram_comparison(
        aggregated,
        'ndvi',
        ndvi_path,
        title=f'NDVI Distribution Comparison ({loaded_count} tiles)'
    )
    
    # EVI histogram
    evi_path = output_dir / f'evi_hist_aggregate_{loaded_count}tiles.png'
    plot_histogram_comparison(
        aggregated,
        'evi',
        evi_path,
        title=f'EVI Distribution Comparison ({loaded_count} tiles)'
    )
    
    print()
    print("=" * 80)
    print("✓ Histogram generation complete!")
    print("=" * 80)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 80)
    
    for source, data in aggregated.items():
        if len(data['ndvi']) > 0:
            print(f"\n{source}:")
            print(f"  NDVI: mean={data['ndvi'].mean():.4f}, std={data['ndvi'].std():.4f}")
            print(f"  EVI:  mean={data['evi'].mean():.4f}, std={data['evi'].std():.4f}")
            print(f"  Samples: {len(data['ndvi']):,} pixels")


# ============================================================================
# Per-Tile Analysis
# ============================================================================

def generate_per_tile_histograms(
    tile_path: Path,
    output_dir: Path,
    tile_id: Optional[str] = None
):
    """
    Generate histograms for a single tile.
    
    Args:
        tile_path: Path to tile NPZ file
        output_dir: Directory to save histogram plots
        tile_id: Optional tile identifier for filename
    """
    print(f"Processing single tile: {tile_path.name}")
    
    # Load tile
    tile_data = load_tile_data(tile_path)
    
    if tile_data is None:
        print("❌ Failed to load tile data")
        return
    
    # Compute indices
    indices = compute_indices_for_images(
        tile_data.get('opt_v1'),
        tile_data.get('opt_v2'),
        tile_data.get('s2_real')
    )
    
    # Generate output filename
    if tile_id is None:
        tile_id = tile_path.stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # NDVI histogram
    ndvi_path = output_dir / f'ndvi_hist_{tile_id}.png'
    plot_histogram_comparison(
        indices,
        'ndvi',
        ndvi_path,
        title=f'NDVI Distribution: {tile_id}'
    )
    
    # EVI histogram
    evi_path = output_dir / f'evi_hist_{tile_id}.png'
    plot_histogram_comparison(
        indices,
        'evi',
        evi_path,
        title=f'EVI Distribution: {tile_id}'
    )
    
    print(f"✓ Histograms saved to {output_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate NDVI/EVI histogram comparisons',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate histograms from all tiles in directory
  python scripts/generate_ndvi_histograms.py --data_dir data/tiles/benv2_catalog --output_dir reports

  # Process first 100 tiles
  python scripts/generate_ndvi_histograms.py --data_dir data/tiles/benv2_catalog --output_dir reports --max_tiles 100

  # Single tile
  python scripts/generate_ndvi_histograms.py --tile data/tiles/sample_tile.npz --output_dir reports
        """
    )
    
    parser.add_argument(
        '--data_dir',
        type=Path,
        help='Directory containing tile NPZ files (for aggregate analysis)'
    )
    
    parser.add_argument(
        '--tile',
        type=Path,
        help='Single tile NPZ file (for per-tile analysis)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('reports'),
        help='Output directory for histogram plots (default: reports/)'
    )
    
    parser.add_argument(
        '--max_tiles',
        type=int,
        help='Maximum number of tiles to process (default: all)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.npz',
        help='File pattern to match (default: *.npz)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.data_dir is None and args.tile is None:
        parser.error("Must specify either --data_dir or --tile")
    
    if args.data_dir is not None and args.tile is not None:
        parser.error("Cannot specify both --data_dir and --tile")
    
    # Run appropriate analysis
    if args.tile is not None:
        # Per-tile analysis
        if not args.tile.exists():
            print(f"❌ Tile file not found: {args.tile}")
            return 1
        
        generate_per_tile_histograms(args.tile, args.output_dir)
    
    else:
        # Aggregate analysis
        if not args.data_dir.exists():
            print(f"❌ Data directory not found: {args.data_dir}")
            return 1
        
        generate_aggregate_histograms(
            args.data_dir,
            args.output_dir,
            max_tiles=args.max_tiles,
            pattern=args.pattern
        )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
