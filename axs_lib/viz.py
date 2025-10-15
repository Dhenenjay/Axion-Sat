"""
axs_lib/viz.py - Visualization Utilities

Provides visualization functions for satellite imagery tiles, including
alignment verification, quality control plots, and exploratory data analysis.

Features:
    - SAR-optical alignment verification
    - Edge detection overlay visualization
    - Multi-band RGB composites
    - Tile quality metrics visualization
    - Batch tile comparison

Usage:
    >>> from axs_lib.viz import show_alignment
    >>> show_alignment('data/tiles/tile_r00000_c00000.npz')

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available - install with: pip install matplotlib")

try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter, sobel
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available - install with: pip install scipy")


# ============================================================================
# Edge Detection
# ============================================================================

def compute_sar_edges(sar_data: np.ndarray, sigma: float = 1.0, threshold: float = 0.1) -> np.ndarray:
    """
    Compute edge map from SAR intensity using Sobel filter.
    
    SAR edges highlight strong backscatter boundaries (e.g., water/land,
    forest/field, urban structures).
    
    Args:
        sar_data: SAR intensity array (dB scale or linear)
        sigma: Gaussian smoothing sigma before edge detection
        threshold: Edge strength threshold (0-1), higher = fewer edges
        
    Returns:
        Binary edge map (True = edge)
        
    Example:
        >>> vv = tile['s1_vv']
        >>> edges = compute_sar_edges(vv, sigma=1.0, threshold=0.15)
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required for edge detection. Install with: pip install scipy")
    
    # Handle NaN values
    valid_mask = ~np.isnan(sar_data)
    data_clean = np.copy(sar_data)
    data_clean[~valid_mask] = np.nanmedian(sar_data)  # Fill NaN with median
    
    # Smooth to reduce speckle
    smoothed = gaussian_filter(data_clean, sigma=sigma)
    
    # Compute gradients
    grad_x = sobel(smoothed, axis=1)
    grad_y = sobel(smoothed, axis=0)
    
    # Edge magnitude
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to [0, 1]
    edge_magnitude = edge_magnitude / (np.max(edge_magnitude) + 1e-10)
    
    # Threshold to binary
    edges = edge_magnitude > threshold
    
    # Mask invalid regions
    edges[~valid_mask] = False
    
    return edges


def compute_optical_luminance(rgb_bands: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute luminance from RGB bands using standard weights.
    
    Luminance represents perceived brightness and is useful for
    visualizing grayscale optical imagery.
    
    Args:
        rgb_bands: Dictionary with keys 's2_b4' (red), 's2_b3' (green), 's2_b2' (blue)
        
    Returns:
        Luminance array (0-1 normalized)
        
    Example:
        >>> lum = compute_optical_luminance({
        ...     's2_b4': tile['s2_b4'],
        ...     's2_b3': tile['s2_b3'],
        ...     's2_b2': tile['s2_b2']
        ... })
    """
    # Extract bands
    red = rgb_bands.get('s2_b4', np.zeros_like(list(rgb_bands.values())[0]))
    green = rgb_bands.get('s2_b3', np.zeros_like(list(rgb_bands.values())[0]))
    blue = rgb_bands.get('s2_b2', np.zeros_like(list(rgb_bands.values())[0]))
    
    # ITU-R BT.709 luminance weights
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    
    # Normalize to [0, 1]
    valid_mask = ~np.isnan(luminance)
    if np.any(valid_mask):
        lum_min = np.nanpercentile(luminance[valid_mask], 2)
        lum_max = np.nanpercentile(luminance[valid_mask], 98)
        luminance_norm = np.clip((luminance - lum_min) / (lum_max - lum_min + 1e-10), 0, 1)
    else:
        luminance_norm = luminance
    
    return luminance_norm


# ============================================================================
# Main Visualization Function
# ============================================================================

def show_alignment(
    tile_npz_path: Union[str, Path],
    sar_band: str = 's1_vv',
    edge_sigma: float = 1.0,
    edge_threshold: float = 0.12,
    edge_color: str = 'cyan',
    edge_alpha: float = 0.7,
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True
) -> Optional[plt.Figure]:
    """
    Visualize SAR-optical alignment by overlaying SAR edges on optical luminance.
    
    This function helps verify that SAR and optical bands are properly co-registered
    by showing SAR-derived edges (boundaries) overlaid on optical grayscale imagery.
    Good alignment means edges should align with optical features.
    
    Args:
        tile_npz_path: Path to tile NPZ file
        sar_band: SAR band to use ('s1_vv' or 's1_vh')
        edge_sigma: Gaussian smoothing for edge detection (higher = smoother)
        edge_threshold: Edge detection threshold (0.05-0.3, higher = fewer edges)
        edge_color: Color for edge overlay ('cyan', 'red', 'yellow', 'magenta')
        edge_alpha: Transparency of edge overlay (0-1)
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save figure
        show_plot: Whether to display plot interactively
        
    Returns:
        matplotlib Figure object (if not shown)
        
    Example:
        >>> # Visual inspection of alignment
        >>> show_alignment('data/tiles/nairobi_2024-01-15/tile_r00000_c00000.npz')
        >>> 
        >>> # Fine-tune edge detection
        >>> show_alignment(
        ...     'tile.npz',
        ...     edge_threshold=0.15,  # More selective
        ...     edge_color='red',
        ...     save_path='alignment_check.png'
        ... )
        
    Interpretation:
        - **Good alignment**: SAR edges coincide with optical features
          (roads, buildings, field boundaries, water edges)
        - **Misalignment**: SAR edges offset from optical features
          (indicates reprojection issues)
        - **No edges**: Try lower threshold or check if data is valid
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required. Install with: pip install matplotlib")
    
    # Load tile
    tile_path = Path(tile_npz_path)
    if not tile_path.exists():
        raise FileNotFoundError(f"Tile not found: {tile_path}")
    
    print(f"Loading tile: {tile_path.name}")
    tile = np.load(tile_path)
    
    # Check available bands
    available_bands = list(tile.files)
    print(f"Available bands: {available_bands}")
    
    # Extract SAR band
    if sar_band not in available_bands:
        raise ValueError(
            f"SAR band '{sar_band}' not found in tile. "
            f"Available: {available_bands}"
        )
    
    sar_data = tile[sar_band]
    print(f"SAR band: {sar_band}, shape: {sar_data.shape}")
    
    # Extract optical bands
    optical_bands = {}
    for band in ['s2_b4', 's2_b3', 's2_b2']:
        if band in available_bands:
            optical_bands[band] = tile[band]
    
    if not optical_bands:
        raise ValueError(
            f"No optical bands found in tile. "
            f"Expected: s2_b4, s2_b3, s2_b2"
        )
    
    print(f"Optical bands: {list(optical_bands.keys())}")
    
    # Compute luminance
    print("Computing optical luminance...")
    luminance = compute_optical_luminance(optical_bands)
    
    # Compute SAR edges
    print(f"Computing SAR edges (sigma={edge_sigma}, threshold={edge_threshold})...")
    edges = compute_sar_edges(sar_data, sigma=edge_sigma, threshold=edge_threshold)
    edge_count = np.sum(edges)
    edge_percent = (edge_count / edges.size) * 100
    print(f"Detected {edge_count} edge pixels ({edge_percent:.2f}% of image)")
    
    if edge_count == 0:
        warnings.warn(
            f"No edges detected! Try lowering threshold (current: {edge_threshold})"
        )
    
    # Create visualization
    print("Creating visualization...")
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.05], hspace=0.3, wspace=0.3)
    
    # 1. Optical luminance
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(luminance, cmap='gray', vmin=0, vmax=1)
    ax1.set_title(f'Optical Luminance\n(RGB composite)', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # 2. SAR intensity
    ax2 = fig.add_subplot(gs[0, 1])
    sar_valid = sar_data[~np.isnan(sar_data)]
    if len(sar_valid) > 0:
        vmin, vmax = np.percentile(sar_valid, [2, 98])
    else:
        vmin, vmax = -30, 0
    im2 = ax2.imshow(sar_data, cmap='gray', vmin=vmin, vmax=vmax)
    ax2.set_title(f'SAR {sar_band.upper()}\n(dB scale)', fontsize=11, fontweight='bold')
    ax2.axis('off')
    
    # 3. Alignment overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(luminance, cmap='gray', vmin=0, vmax=1)
    
    # Create RGBA edge overlay
    edge_overlay = np.zeros((*edges.shape, 4))
    edge_overlay[edges, :3] = plt.cm.colors.to_rgb(edge_color)
    edge_overlay[edges, 3] = edge_alpha
    
    ax3.imshow(edge_overlay)
    ax3.set_title(
        f'Alignment Check\n(SAR edges on optical)',
        fontsize=11,
        fontweight='bold'
    )
    ax3.axis('off')
    
    # Add legend
    edge_patch = mpatches.Patch(
        color=edge_color,
        alpha=edge_alpha,
        label=f'SAR edges ({edge_percent:.1f}%)'
    )
    ax3.legend(handles=[edge_patch], loc='upper right', fontsize=9)
    
    # Colorbars
    cbar1 = plt.colorbar(im1, cax=fig.add_subplot(gs[1, 0]), orientation='horizontal')
    cbar1.set_label('Normalized intensity', fontsize=9)
    
    cbar2 = plt.colorbar(im2, cax=fig.add_subplot(gs[1, 1]), orientation='horizontal')
    cbar2.set_label('Backscatter (dB)', fontsize=9)
    
    # Overall title
    fig.suptitle(
        f'SAR-Optical Alignment Verification: {tile_path.stem}',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    # Add metadata text
    metadata_text = (
        f"Tile shape: {sar_data.shape}\n"
        f"Edge detection: σ={edge_sigma}, threshold={edge_threshold}\n"
        f"Edge pixels: {edge_count:,} ({edge_percent:.2f}%)"
    )
    fig.text(
        0.98, 0.02, metadata_text,
        ha='right', va='bottom',
        fontsize=8,
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    # Show or return
    if show_plot:
        plt.show()
        return None
    else:
        return fig


# ============================================================================
# Additional Visualization Functions
# ============================================================================

def show_rgb_composite(
    tile_npz_path: Union[str, Path],
    bands: Tuple[str, str, str] = ('s2_b4', 's2_b3', 's2_b2'),
    percentile_stretch: Tuple[float, float] = (2, 98),
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[Union[str, Path]] = None
):
    """
    Display RGB composite from tile.
    
    Args:
        tile_npz_path: Path to tile NPZ
        bands: Tuple of (red, green, blue) band names
        percentile_stretch: Percentile range for contrast stretch
        figsize: Figure size
        save_path: Optional save path
        
    Example:
        >>> # True color
        >>> show_rgb_composite('tile.npz', bands=('s2_b4', 's2_b3', 's2_b2'))
        >>> 
        >>> # False color (NIR-R-G)
        >>> show_rgb_composite('tile.npz', bands=('s2_b8', 's2_b4', 's2_b3'))
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")
    
    tile = np.load(tile_npz_path)
    
    # Extract and stack bands
    rgb = []
    for band_name in bands:
        if band_name not in tile.files:
            raise ValueError(f"Band {band_name} not found in tile")
        band = tile[band_name]
        
        # Percentile stretch
        valid = band[~np.isnan(band)]
        if len(valid) > 0:
            vmin, vmax = np.nanpercentile(band, percentile_stretch)
            band_norm = np.clip((band - vmin) / (vmax - vmin + 1e-10), 0, 1)
        else:
            band_norm = np.zeros_like(band)
        
        rgb.append(band_norm)
    
    rgb = np.stack(rgb, axis=-1)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb)
    ax.set_title(f'RGB Composite: {bands}', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def show_tile_overview(
    tile_npz_path: Union[str, Path],
    figsize: Tuple[int, int] = (16, 10)
):
    """
    Show comprehensive overview of all bands in a tile.
    
    Args:
        tile_npz_path: Path to tile NPZ
        figsize: Figure size
        
    Example:
        >>> show_tile_overview('tile.npz')
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")
    
    tile = np.load(tile_npz_path)
    bands = tile.files
    
    # Determine grid size
    n_bands = len(bands)
    n_cols = min(4, n_bands)
    n_rows = (n_bands + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    
    for i, band_name in enumerate(bands):
        ax = axes[i]
        data = tile[band_name]
        
        # Auto-scale
        valid = data[~np.isnan(data)]
        if len(valid) > 0:
            vmin, vmax = np.nanpercentile(data, [2, 98])
        else:
            vmin, vmax = 0, 1
        
        im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(band_name, fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(n_bands, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(f'Tile Overview: {Path(tile_npz_path).stem}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================================
# Utility Functions
# ============================================================================

def check_alignment_quality(
    tile_npz_path: Union[str, Path],
    sar_band: str = 's1_vv',
    edge_threshold: float = 0.12
) -> Dict[str, float]:
    """
    Compute quantitative alignment quality metrics.
    
    Returns metrics including edge count, spatial statistics, etc.
    
    Args:
        tile_npz_path: Path to tile NPZ
        sar_band: SAR band name
        edge_threshold: Edge detection threshold
        
    Returns:
        Dictionary of quality metrics
    """
    tile = np.load(tile_npz_path)
    
    # Compute edges
    sar_data = tile[sar_band]
    edges = compute_sar_edges(sar_data, threshold=edge_threshold)
    
    # Metrics
    metrics = {
        'edge_count': int(np.sum(edges)),
        'edge_percent': float(np.sum(edges) / edges.size * 100),
        'sar_valid_percent': float(np.sum(~np.isnan(sar_data)) / sar_data.size * 100),
    }
    
    # Optical coverage
    for band in ['s2_b4', 's2_b3', 's2_b2']:
        if band in tile.files:
            valid = ~np.isnan(tile[band])
            metrics[f'{band}_valid_percent'] = float(np.sum(valid) / valid.size * 100)
    
    return metrics


# ============================================================================
# Main / Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 79)
    print("VISUALIZATION UTILITIES DEMO")
    print("=" * 79)
    print()
    
    print("This module provides visualization functions for satellite imagery tiles.")
    print()
    print("Example usage:")
    print("-" * 79)
    print()
    print("from axs_lib.viz import show_alignment")
    print()
    print("# Verify SAR-optical alignment")
    print("show_alignment('data/tiles/tile_r00000_c00000.npz')")
    print()
    print("# Adjust edge detection sensitivity")
    print("show_alignment(")
    print("    'tile.npz',")
    print("    edge_threshold=0.15,  # Higher = fewer edges")
    print("    edge_color='red',")
    print("    save_path='alignment_check.png'")
    print(")")
    print()
    print("# RGB composite")
    print("from axs_lib.viz import show_rgb_composite")
    print("show_rgb_composite('tile.npz')")
    print()
    print("# Full tile overview")
    print("from axs_lib.viz import show_tile_overview")
    print("show_tile_overview('tile.npz')")
    print()
    print("=" * 79)
    print()
    
    # Check dependencies
    print("Dependency check:")
    print(f"  matplotlib: {'✓' if HAS_MATPLOTLIB else '✗ (pip install matplotlib)'}")
    print(f"  scipy:      {'✓' if HAS_SCIPY else '✗ (pip install scipy)'}")
    print()
    
    if not HAS_MATPLOTLIB:
        print("⚠ Install matplotlib to use visualization functions")
    if not HAS_SCIPY:
        print("⚠ Install scipy for edge detection")
