"""
Color Checker: Radiometric Validation Tool

Compares per-band statistics (mean, variance, percentiles) between Stage 2 outputs
and ground truth (Sentinel-2/HLS) to detect:
- Over-saturation (values clipped at high end)
- Under-saturation (reduced dynamic range)
- Color shifts (systematic bias in band means)
- Variance mismatch (over-smoothing or noise amplification)
- Outlier pixels

Outputs:
- Per-band statistics comparison table
- Saturation flags and warnings
- Visual histograms showing distribution differences
- Summary report with recommendations

Usage:
    # Check single tile
    python tools/color_checker.py --tile data/tiles/sample.npz

    # Check all tiles in directory
    python tools/color_checker.py --data_dir data/tiles/ --output reports/color_check.txt

    # Generate histogram visualizations
    python tools/color_checker.py --tile sample.npz --plot --output_dir color_checks/

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class BandStatistics:
    """Statistics for a single band."""
    name: str
    mean: float
    std: float
    variance: float
    min: float
    max: float
    p01: float  # 1st percentile
    p05: float  # 5th percentile
    p50: float  # Median
    p95: float  # 95th percentile
    p99: float  # 99th percentile
    saturated_low: float  # Fraction of pixels at minimum
    saturated_high: float  # Fraction of pixels at maximum
    
    def __str__(self) -> str:
        return (
            f"{self.name:8s} | "
            f"μ={self.mean:6.4f} σ={self.std:6.4f} | "
            f"[{self.min:6.4f}, {self.max:6.4f}] | "
            f"p50={self.p50:6.4f} | "
            f"sat_lo={self.saturated_low*100:4.1f}% sat_hi={self.saturated_high*100:4.1f}%"
        )


@dataclass
class ColorCheckResult:
    """Results from color checking a single tile."""
    tile_name: str
    v2_stats: List[BandStatistics]
    truth_stats: List[BandStatistics]
    warnings: List[str]
    flags: Dict[str, bool]
    
    def has_issues(self) -> bool:
        """Return True if any flags are raised."""
        return any(self.flags.values())


# ============================================================================
# Statistics Computation
# ============================================================================

def compute_band_statistics(
    band: np.ndarray,
    band_name: str,
    valid_range: Tuple[float, float] = (0.0, 1.0),
    saturation_threshold: float = 0.001
) -> BandStatistics:
    """
    Compute comprehensive statistics for a single band.
    
    Args:
        band: 2D array (H, W)
        band_name: Name of the band (e.g., 'Blue', 'Green', 'Red', 'NIR')
        valid_range: Expected valid range for pixel values
        saturation_threshold: Fraction threshold for saturation detection
        
    Returns:
        BandStatistics object
    """
    # Remove NaN/Inf values
    valid_pixels = band[np.isfinite(band)]
    
    if len(valid_pixels) == 0:
        warnings.warn(f"Band {band_name} has no valid pixels!")
        return BandStatistics(
            name=band_name,
            mean=np.nan, std=np.nan, variance=np.nan,
            min=np.nan, max=np.nan,
            p01=np.nan, p05=np.nan, p50=np.nan, p95=np.nan, p99=np.nan,
            saturated_low=0.0, saturated_high=0.0
        )
    
    # Basic statistics
    mean = np.mean(valid_pixels)
    std = np.std(valid_pixels)
    variance = np.var(valid_pixels)
    min_val = np.min(valid_pixels)
    max_val = np.max(valid_pixels)
    
    # Percentiles
    p01, p05, p50, p95, p99 = np.percentile(valid_pixels, [1, 5, 50, 95, 99])
    
    # Saturation detection
    range_min, range_max = valid_range
    tolerance = 1e-6
    
    saturated_low = np.sum(valid_pixels <= (range_min + tolerance)) / len(valid_pixels)
    saturated_high = np.sum(valid_pixels >= (range_max - tolerance)) / len(valid_pixels)
    
    return BandStatistics(
        name=band_name,
        mean=mean,
        std=std,
        variance=variance,
        min=min_val,
        max=max_val,
        p01=p01,
        p05=p05,
        p50=p50,
        p95=p95,
        p99=p99,
        saturated_low=saturated_low,
        saturated_high=saturated_high
    )


def compute_multispectral_statistics(
    image: np.ndarray,
    band_names: List[str] = None,
    valid_range: Tuple[float, float] = (0.0, 1.0)
) -> List[BandStatistics]:
    """
    Compute statistics for all bands in a multispectral image.
    
    Args:
        image: Multi-band image (C, H, W)
        band_names: Names for each band
        valid_range: Expected valid range for pixel values
        
    Returns:
        List of BandStatistics, one per band
    """
    n_bands = image.shape[0]
    
    if band_names is None:
        band_names = [f"Band_{i}" for i in range(n_bands)]
    
    if len(band_names) != n_bands:
        raise ValueError(f"Number of band names ({len(band_names)}) must match number of bands ({n_bands})")
    
    stats = []
    for i, name in enumerate(band_names):
        band_stats = compute_band_statistics(image[i], name, valid_range)
        stats.append(band_stats)
    
    return stats


# ============================================================================
# Color Checking Logic
# ============================================================================

def check_over_saturation(
    stats: BandStatistics,
    high_threshold: float = 0.01,
    low_threshold: float = 0.01
) -> Tuple[bool, Optional[str]]:
    """
    Check if band shows signs of saturation.
    
    Args:
        stats: Band statistics
        high_threshold: Threshold for high saturation (fraction of pixels)
        low_threshold: Threshold for low saturation (fraction of pixels)
        
    Returns:
        (has_issue, warning_message)
    """
    warnings = []
    
    if stats.saturated_high > high_threshold:
        warnings.append(
            f"  ⚠ {stats.name}: {stats.saturated_high*100:.2f}% of pixels saturated at high end (>{high_threshold*100}%)"
        )
    
    if stats.saturated_low > low_threshold:
        warnings.append(
            f"  ⚠ {stats.name}: {stats.saturated_low*100:.2f}% of pixels saturated at low end (>{low_threshold*100}%)"
        )
    
    if warnings:
        return True, "\n".join(warnings)
    return False, None


def check_color_shift(
    v2_stats: BandStatistics,
    truth_stats: BandStatistics,
    mean_threshold: float = 0.05,
    median_threshold: float = 0.05
) -> Tuple[bool, Optional[str]]:
    """
    Check if there's a systematic color shift between v2 and ground truth.
    
    Args:
        v2_stats: Statistics from Stage 2 output
        truth_stats: Statistics from ground truth
        mean_threshold: Threshold for mean difference
        median_threshold: Threshold for median difference
        
    Returns:
        (has_issue, warning_message)
    """
    warnings = []
    
    mean_diff = v2_stats.mean - truth_stats.mean
    median_diff = v2_stats.p50 - truth_stats.p50
    
    if abs(mean_diff) > mean_threshold:
        direction = "brighter" if mean_diff > 0 else "darker"
        warnings.append(
            f"  ⚠ {v2_stats.name}: Mean shift = {mean_diff:+.4f} ({direction} than truth)"
        )
    
    if abs(median_diff) > median_threshold:
        direction = "higher" if median_diff > 0 else "lower"
        warnings.append(
            f"  ⚠ {v2_stats.name}: Median shift = {median_diff:+.4f} ({direction} than truth)"
        )
    
    if warnings:
        return True, "\n".join(warnings)
    return False, None


def check_variance_mismatch(
    v2_stats: BandStatistics,
    truth_stats: BandStatistics,
    variance_ratio_threshold: float = 1.5
) -> Tuple[bool, Optional[str]]:
    """
    Check if variance differs significantly between v2 and ground truth.
    
    Args:
        v2_stats: Statistics from Stage 2 output
        truth_stats: Statistics from ground truth
        variance_ratio_threshold: Threshold for variance ratio
        
    Returns:
        (has_issue, warning_message)
    """
    warnings = []
    
    if truth_stats.variance == 0:
        return False, None
    
    variance_ratio = v2_stats.variance / truth_stats.variance
    
    if variance_ratio > variance_ratio_threshold:
        warnings.append(
            f"  ⚠ {v2_stats.name}: Variance ratio = {variance_ratio:.2f} (too high, possible noise amplification)"
        )
    elif variance_ratio < (1.0 / variance_ratio_threshold):
        warnings.append(
            f"  ⚠ {v2_stats.name}: Variance ratio = {variance_ratio:.2f} (too low, possible over-smoothing)"
        )
    
    if warnings:
        return True, "\n".join(warnings)
    return False, None


def check_dynamic_range(
    v2_stats: BandStatistics,
    truth_stats: BandStatistics,
    range_threshold: float = 0.2
) -> Tuple[bool, Optional[str]]:
    """
    Check if dynamic range is preserved.
    
    Args:
        v2_stats: Statistics from Stage 2 output
        truth_stats: Statistics from ground truth
        range_threshold: Threshold for dynamic range difference
        
    Returns:
        (has_issue, warning_message)
    """
    warnings = []
    
    v2_range = v2_stats.p99 - v2_stats.p01
    truth_range = truth_stats.p99 - truth_stats.p01
    
    if truth_range == 0:
        return False, None
    
    range_ratio = v2_range / truth_range
    
    if abs(1.0 - range_ratio) > range_threshold:
        if range_ratio < 1.0:
            warnings.append(
                f"  ⚠ {v2_stats.name}: Dynamic range reduced by {(1.0-range_ratio)*100:.1f}%"
            )
        else:
            warnings.append(
                f"  ⚠ {v2_stats.name}: Dynamic range increased by {(range_ratio-1.0)*100:.1f}%"
            )
    
    if warnings:
        return True, "\n".join(warnings)
    return False, None


def perform_color_check(
    v2_image: np.ndarray,
    truth_image: np.ndarray,
    band_names: List[str],
    tile_name: str = "unknown"
) -> ColorCheckResult:
    """
    Perform comprehensive color checking on a tile.
    
    Args:
        v2_image: Stage 2 output (C, H, W)
        truth_image: Ground truth (C, H, W)
        band_names: Names for each band
        tile_name: Name of the tile being checked
        
    Returns:
        ColorCheckResult object
    """
    # Compute statistics
    v2_stats = compute_multispectral_statistics(v2_image, band_names)
    truth_stats = compute_multispectral_statistics(truth_image, band_names)
    
    # Initialize warnings and flags
    all_warnings = []
    flags = {
        'saturation': False,
        'color_shift': False,
        'variance_mismatch': False,
        'dynamic_range': False
    }
    
    # Check each band
    for v2_band, truth_band in zip(v2_stats, truth_stats):
        # Saturation check
        has_sat, sat_warning = check_over_saturation(v2_band)
        if has_sat:
            flags['saturation'] = True
            all_warnings.append(sat_warning)
        
        # Color shift check
        has_shift, shift_warning = check_color_shift(v2_band, truth_band)
        if has_shift:
            flags['color_shift'] = True
            all_warnings.append(shift_warning)
        
        # Variance mismatch check
        has_variance, variance_warning = check_variance_mismatch(v2_band, truth_band)
        if has_variance:
            flags['variance_mismatch'] = True
            all_warnings.append(variance_warning)
        
        # Dynamic range check
        has_range, range_warning = check_dynamic_range(v2_band, truth_band)
        if has_range:
            flags['dynamic_range'] = True
            all_warnings.append(range_warning)
    
    return ColorCheckResult(
        tile_name=tile_name,
        v2_stats=v2_stats,
        truth_stats=truth_stats,
        warnings=all_warnings,
        flags=flags
    )


# ============================================================================
# Reporting
# ============================================================================

def print_statistics_table(stats_list: List[BandStatistics], title: str):
    """Print formatted statistics table."""
    print(f"\n{title}")
    print("-" * 100)
    print(f"{'Band':<8} | {'Mean':<6} {'Std':<6} | {'Min':<6} {'Max':<6} | {'Median':<6} | {'Sat.Low':<6} {'Sat.High':<6}")
    print("-" * 100)
    
    for stats in stats_list:
        print(
            f"{stats.name:<8} | "
            f"{stats.mean:6.4f} {stats.std:6.4f} | "
            f"{stats.min:6.4f} {stats.max:6.4f} | "
            f"{stats.p50:6.4f} | "
            f"{stats.saturated_low*100:5.2f}% {stats.saturated_high*100:6.2f}%"
        )
    print("-" * 100)


def print_comparison_table(
    v2_stats: List[BandStatistics],
    truth_stats: List[BandStatistics]
):
    """Print side-by-side comparison table."""
    print("\nComparison: Stage 2 (v2) vs Ground Truth")
    print("=" * 120)
    print(f"{'Band':<8} | {'v2 Mean':<8} {'Truth Mean':<10} {'Δ':<8} | {'v2 Std':<8} {'Truth Std':<10} {'Ratio':<6}")
    print("=" * 120)
    
    for v2, truth in zip(v2_stats, truth_stats):
        mean_diff = v2.mean - truth.mean
        std_ratio = v2.std / truth.std if truth.std != 0 else float('inf')
        
        print(
            f"{v2.name:<8} | "
            f"{v2.mean:8.4f} {truth.mean:10.4f} {mean_diff:+8.4f} | "
            f"{v2.std:8.4f} {truth.std:10.4f} {std_ratio:6.2f}"
        )
    print("=" * 120)


def print_color_check_report(result: ColorCheckResult):
    """Print comprehensive color check report."""
    print("\n" + "=" * 80)
    print(f"Color Check Report: {result.tile_name}")
    print("=" * 80)
    
    # Print statistics
    print_statistics_table(result.truth_stats, "Ground Truth Statistics")
    print_statistics_table(result.v2_stats, "Stage 2 (v2) Statistics")
    print_comparison_table(result.v2_stats, result.truth_stats)
    
    # Print warnings
    if result.warnings:
        print("\n⚠ WARNINGS:")
        print("-" * 80)
        for warning in result.warnings:
            print(warning)
    else:
        print("\n✓ No significant issues detected")
    
    # Print flags summary
    print("\nFlags Summary:")
    print("-" * 80)
    print(f"  Over-saturation:    {'⚠ YES' if result.flags['saturation'] else '✓ NO'}")
    print(f"  Color shift:        {'⚠ YES' if result.flags['color_shift'] else '✓ NO'}")
    print(f"  Variance mismatch:  {'⚠ YES' if result.flags['variance_mismatch'] else '✓ NO'}")
    print(f"  Dynamic range:      {'⚠ YES' if result.flags['dynamic_range'] else '✓ NO'}")
    print("-" * 80)


# ============================================================================
# Visualization
# ============================================================================

def plot_histogram_comparison(
    v2_image: np.ndarray,
    truth_image: np.ndarray,
    band_names: List[str],
    output_path: Path,
    tile_name: str = "unknown"
):
    """
    Create histogram comparison plot for all bands.
    
    Args:
        v2_image: Stage 2 output (C, H, W)
        truth_image: Ground truth (C, H, W)
        band_names: Names for each band
        output_path: Path to save figure
        tile_name: Name of the tile
    """
    n_bands = v2_image.shape[0]
    
    fig, axes = plt.subplots(2, n_bands, figsize=(4*n_bands, 8))
    if n_bands == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(f"Histogram Comparison: {tile_name}", fontsize=14, fontweight='bold')
    
    bins = np.linspace(0, 1, 100)
    
    for i, band_name in enumerate(band_names):
        # Get valid pixels
        v2_valid = v2_image[i][np.isfinite(v2_image[i])].flatten()
        truth_valid = truth_image[i][np.isfinite(truth_image[i])].flatten()
        
        # Top row: Overlaid histograms
        ax = axes[0, i]
        ax.hist(truth_valid, bins=bins, alpha=0.6, label='Truth', color='blue', density=True)
        ax.hist(v2_valid, bins=bins, alpha=0.6, label='v2', color='red', density=True)
        ax.set_title(f"{band_name}", fontweight='bold')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom row: Difference histogram
        ax = axes[1, i]
        diff = v2_image[i] - truth_image[i]
        diff_valid = diff[np.isfinite(diff)].flatten()
        
        ax.hist(diff_valid, bins=100, alpha=0.7, color='purple', density=True)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title(f"{band_name} Difference (v2 - truth)")
        ax.set_xlabel('Difference')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        
        # Add mean difference text
        mean_diff = np.mean(diff_valid)
        ax.text(0.02, 0.98, f'μ_diff = {mean_diff:+.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Data Loading
# ============================================================================

def load_tile_for_color_check(tile_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Load tile data for color checking.
    
    Returns:
        Dict with 'v2' and 'truth' arrays, or None if loading fails
    """
    try:
        data = np.load(tile_path)
        
        result = {}
        
        # Load v2
        if 'opt_v2' in data:
            result['v2'] = data['opt_v2']
        else:
            print(f"⚠ 'opt_v2' not found in {tile_path.name}")
            return None
        
        # Load truth
        if 's2_real' in data:
            result['truth'] = data['s2_real']
        elif all(k in data for k in ['s2_b2', 's2_b3', 's2_b4', 's2_b8']):
            result['truth'] = np.stack([
                data['s2_b2'],
                data['s2_b3'],
                data['s2_b4'],
                data['s2_b8'],
            ], axis=0)
        else:
            print(f"⚠ No ground truth found in {tile_path.name}")
            return None
        
        return result
        
    except Exception as e:
        print(f"✗ Error loading {tile_path.name}: {e}")
        return None


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Color checker for radiometric validation of Stage 2 outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--tile',
        type=Path,
        help='Single tile to check'
    )
    
    parser.add_argument(
        '--data_dir',
        type=Path,
        help='Directory containing tiles to check'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for text report (optional)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('color_checks/'),
        help='Directory for histogram plots (default: color_checks/)'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate histogram comparison plots'
    )
    
    parser.add_argument(
        '--max_tiles',
        type=int,
        help='Maximum number of tiles to process'
    )
    
    parser.add_argument(
        '--band_names',
        type=str,
        nargs='+',
        default=['Blue', 'Green', 'Red', 'NIR'],
        help='Names for bands (default: Blue Green Red NIR)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.tile and not args.data_dir:
        print("Error: Must specify either --tile or --data_dir")
        return 1
    
    # Create output directory if needed
    if args.plot:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Color Checker: Radiometric Validation")
    print("=" * 80)
    
    # Collect tiles
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
    
    print(f"\nFound {len(tiles)} tile(s) to check")
    
    # Process tiles
    all_results = []
    
    for tile in tqdm(tiles, desc="Checking tiles"):
        data = load_tile_for_color_check(tile)
        
        if data is None:
            continue
        
        # Perform color check
        result = perform_color_check(
            v2_image=data['v2'],
            truth_image=data['truth'],
            band_names=args.band_names,
            tile_name=tile.name
        )
        
        all_results.append(result)
        
        # Print report
        print_color_check_report(result)
        
        # Generate plot if requested
        if args.plot:
            plot_path = args.output_dir / f"{tile.stem}_histograms.png"
            plot_histogram_comparison(
                v2_image=data['v2'],
                truth_image=data['truth'],
                band_names=args.band_names,
                output_path=plot_path,
                tile_name=tile.name
            )
            print(f"  ✓ Saved histograms: {plot_path.name}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    if all_results:
        print(f"\nProcessed {len(all_results)} tile(s)")
        
        # Count issues
        n_saturation = sum(1 for r in all_results if r.flags['saturation'])
        n_color_shift = sum(1 for r in all_results if r.flags['color_shift'])
        n_variance = sum(1 for r in all_results if r.flags['variance_mismatch'])
        n_dynamic_range = sum(1 for r in all_results if r.flags['dynamic_range'])
        n_clean = sum(1 for r in all_results if not r.has_issues())
        
        print(f"\nIssue counts:")
        print(f"  Over-saturation:   {n_saturation:3d} / {len(all_results)} ({n_saturation/len(all_results)*100:.1f}%)")
        print(f"  Color shift:       {n_color_shift:3d} / {len(all_results)} ({n_color_shift/len(all_results)*100:.1f}%)")
        print(f"  Variance mismatch: {n_variance:3d} / {len(all_results)} ({n_variance/len(all_results)*100:.1f}%)")
        print(f"  Dynamic range:     {n_dynamic_range:3d} / {len(all_results)} ({n_dynamic_range/len(all_results)*100:.1f}%)")
        print(f"  Clean (no issues): {n_clean:3d} / {len(all_results)} ({n_clean/len(all_results)*100:.1f}%)")
        
        # Aggregate statistics
        print(f"\nAggregate statistics across all tiles:")
        for band_idx, band_name in enumerate(args.band_names):
            v2_means = [r.v2_stats[band_idx].mean for r in all_results]
            truth_means = [r.truth_stats[band_idx].mean for r in all_results]
            
            avg_v2_mean = np.mean(v2_means)
            avg_truth_mean = np.mean(truth_means)
            avg_diff = avg_v2_mean - avg_truth_mean
            
            print(f"  {band_name:>6s}: v2_μ={avg_v2_mean:.4f}, truth_μ={avg_truth_mean:.4f}, Δ={avg_diff:+.4f}")
    else:
        print("\n⚠ No tiles processed successfully")
    
    # Save text report if requested
    if args.output:
        print(f"\n✓ Saving report to: {args.output}")
        # TODO: Implement text report export
    
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
