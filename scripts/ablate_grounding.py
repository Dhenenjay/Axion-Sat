"""
Ablation Study: SAR Grounding Impact

Compares Stage 2 (no physics grounding) vs Stage 3 (SAR-grounded) outputs
to quantify the impact of physics-based refinement on SAR edge alignment.

This script:
1. Loads Stage 2 outputs (opt_v2) - baseline without grounding
2. Loads Stage 3 outputs (opt_final) - with SAR grounding
3. Loads SAR observations (s1) - ground truth physics
4. Computes SAR edge agreement for both
5. Reports mean improvement and statistical significance

Key Metrics:
- SAR Agreement: Correlation between optical and SAR edges
- Delta: Improvement from Stage 2 to Stage 3
- Effect Size: Normalized improvement magnitude

Usage:
    # Basic ablation on 50 tiles
    python scripts/ablate_grounding.py \
        --opt-v2-dir stage2_outputs/ \
        --opt-final-dir stage3_outputs/ \
        --s1-dir tiles/s1/ \
        --n-tiles 50

    # Full dataset with visualization
    python scripts/ablate_grounding.py \
        --opt-v2-dir stage2_outputs/ \
        --opt-final-dir stage3_outputs/ \
        --s1-dir tiles/s1/ \
        --n-tiles -1 \
        --save-results ablation_results.json \
        --plot ablation_plot.png

    # Select specific tiles
    python scripts/ablate_grounding.py \
        --opt-v2-dir stage2_outputs/ \
        --opt-final-dir stage3_outputs/ \
        --s1-dir tiles/s1/ \
        --tile-list tiles_to_test.txt

Output:
    Prints summary statistics including:
    - Mean SAR agreement for Stage 2 (no grounding)
    - Mean SAR agreement for Stage 3 (grounded)
    - Mean delta (improvement)
    - Standard deviation
    - Statistical significance (paired t-test)
    - Effect size (Cohen's d)

Author: Axion-Sat Project
Version: 1.0.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
from scipy import ndimage, stats
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Data Loading
# ============================================================================

def load_tile(tile_path: Path, expected_channels: int = 4) -> Optional[np.ndarray]:
    """
    Load tile from NPZ file.
    
    Args:
        tile_path: Path to NPZ file
        expected_channels: Expected number of channels
        
    Returns:
        Array (C, H, W) or None if loading fails
    """
    try:
        data = np.load(tile_path)
        
        # Try common keys
        possible_keys = [
            'opt_v2', 'opt_final', 'opt_v3', 'opt', 's1',
            's2_b2', 's2_b3', 's2_b4', 's2_b8',
            's1_vv', 's1_vh'
        ]
        
        for key in possible_keys:
            if key in data:
                arr = data[key]
                if arr.ndim == 2:
                    # Single channel, expand
                    arr = np.stack([arr] * expected_channels, axis=0)
                return arr.astype(np.float32)
        
        # Try loading individual bands for optical
        if expected_channels == 4 and all(f's2_b{b}' in data for b in [2, 3, 4, 8]):
            arr = np.stack([data['s2_b2'], data['s2_b3'], data['s2_b4'], data['s2_b8']], axis=0)
            return arr.astype(np.float32)
        
        # Try loading SAR bands
        if expected_channels == 2 and 's1_vv' in data and 's1_vh' in data:
            arr = np.stack([data['s1_vv'], data['s1_vh']], axis=0)
            return arr.astype(np.float32)
        
        warnings.warn(f"Could not find valid data in {tile_path}")
        return None
        
    except Exception as e:
        warnings.warn(f"Error loading {tile_path}: {e}")
        return None


def find_matching_tiles(
    opt_v2_dir: Path,
    opt_final_dir: Path,
    s1_dir: Path,
    max_tiles: int = -1
) -> List[Tuple[Path, Path, Path]]:
    """
    Find matching triplets of tiles.
    
    Args:
        opt_v2_dir: Stage 2 output directory
        opt_final_dir: Stage 3 output directory
        s1_dir: SAR input directory
        max_tiles: Maximum number of tiles (-1 for all)
        
    Returns:
        List of (opt_v2_path, opt_final_path, s1_path) tuples
    """
    opt_v2_tiles = {p.name: p for p in opt_v2_dir.rglob('*.npz')}
    opt_final_tiles = {p.name: p for p in opt_final_dir.rglob('*.npz')}
    s1_tiles = {p.name: p for p in s1_dir.rglob('*.npz')}
    
    # Find common tiles
    common_names = set(opt_v2_tiles.keys()) & set(opt_final_tiles.keys()) & set(s1_tiles.keys())
    
    triplets = []
    for name in sorted(common_names):
        triplets.append((
            opt_v2_tiles[name],
            opt_final_tiles[name],
            s1_tiles[name]
        ))
    
    if max_tiles > 0:
        triplets = triplets[:max_tiles]
    
    return triplets


# ============================================================================
# Edge Detection and SAR Agreement
# ============================================================================

def compute_edges(img: np.ndarray) -> np.ndarray:
    """
    Compute edge magnitude using Sobel operator.
    
    Args:
        img: Grayscale image (H, W)
        
    Returns:
        Edge magnitude (H, W)
    """
    # Sobel edge detection
    edges = ndimage.sobel(img)
    
    # Normalize to [0, 1]
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
    
    return edges


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert multi-channel image to grayscale.
    
    Args:
        img: Multi-channel image (C, H, W)
        
    Returns:
        Grayscale image (H, W)
    """
    if img.shape[0] >= 3:
        # Use standard RGB to grayscale conversion
        # Assuming channel order is B, G, R, NIR
        # Use R, G, B (indices 2, 1, 0)
        gray = 0.2989 * img[2] + 0.5870 * img[1] + 0.1140 * img[0]
    else:
        # For SAR or other, just average
        gray = img.mean(axis=0)
    
    return gray


def compute_sar_agreement(opt: np.ndarray, s1: np.ndarray) -> float:
    """
    Compute SAR edge agreement.
    
    This is the primary physics grounding metric. Higher values indicate
    better alignment between optical and SAR-derived physical boundaries.
    
    Args:
        opt: Optical image (4, H, W)
        s1: SAR image (2, H, W)
        
    Returns:
        Agreement score [0, 1], higher is better
    """
    # Convert to grayscale
    opt_gray = to_grayscale(opt)
    s1_gray = s1.mean(axis=0)
    
    # Normalize
    opt_gray = (opt_gray - opt_gray.min()) / (opt_gray.max() - opt_gray.min() + 1e-8)
    s1_gray = (s1_gray - s1_gray.min()) / (s1_gray.max() - s1_gray.min() + 1e-8)
    
    # Compute edges
    opt_edges = compute_edges(opt_gray)
    s1_edges = compute_edges(s1_gray)
    
    # Flatten
    opt_flat = opt_edges.flatten()
    s1_flat = s1_edges.flatten()
    
    # Normalize to unit vectors
    opt_norm = opt_flat / (np.linalg.norm(opt_flat) + 1e-8)
    s1_norm = s1_flat / (np.linalg.norm(s1_flat) + 1e-8)
    
    # Compute correlation (cosine similarity)
    agreement = np.dot(opt_norm, s1_norm)
    
    # Clip to [0, 1]
    agreement = np.clip(agreement, 0, 1)
    
    return float(agreement)


# ============================================================================
# Ablation Analysis
# ============================================================================

def run_ablation(
    tile_triplets: List[Tuple[Path, Path, Path]],
    verbose: bool = True
) -> Dict:
    """
    Run ablation study on tile triplets.
    
    Args:
        tile_triplets: List of (opt_v2, opt_final, s1) path tuples
        verbose: Show progress bar
        
    Returns:
        Dict with results
    """
    results = {
        'tiles': [],
        'sar_agreement_v2': [],
        'sar_agreement_final': [],
        'delta': []
    }
    
    # Process tiles
    if verbose:
        pbar = tqdm(tile_triplets, desc="Processing tiles")
    else:
        pbar = tile_triplets
    
    for opt_v2_path, opt_final_path, s1_path in pbar:
        tile_name = opt_v2_path.name
        
        # Load tiles
        opt_v2 = load_tile(opt_v2_path, expected_channels=4)
        opt_final = load_tile(opt_final_path, expected_channels=4)
        s1 = load_tile(s1_path, expected_channels=2)
        
        if opt_v2 is None or opt_final is None or s1 is None:
            if verbose:
                warnings.warn(f"Skipping {tile_name} (loading failed)")
            continue
        
        # Ensure spatial dimensions match
        if opt_v2.shape[1:] != s1.shape[1:]:
            if verbose:
                warnings.warn(f"Skipping {tile_name} (spatial mismatch)")
            continue
        
        if opt_final.shape[1:] != s1.shape[1:]:
            if verbose:
                warnings.warn(f"Skipping {tile_name} (spatial mismatch)")
            continue
        
        # Compute SAR agreement
        try:
            sar_v2 = compute_sar_agreement(opt_v2, s1)
            sar_final = compute_sar_agreement(opt_final, s1)
            delta = sar_final - sar_v2
            
            # Store results
            results['tiles'].append(tile_name)
            results['sar_agreement_v2'].append(sar_v2)
            results['sar_agreement_final'].append(sar_final)
            results['delta'].append(delta)
            
        except Exception as e:
            if verbose:
                warnings.warn(f"Error processing {tile_name}: {e}")
    
    if verbose and hasattr(pbar, 'close'):
        pbar.close()
    
    return results


def compute_statistics(results: Dict) -> Dict:
    """
    Compute summary statistics from results.
    
    Args:
        results: Results dict from run_ablation
        
    Returns:
        Statistics dict
    """
    sar_v2 = np.array(results['sar_agreement_v2'])
    sar_final = np.array(results['sar_agreement_final'])
    delta = np.array(results['delta'])
    
    stats_dict = {
        'n_tiles': len(results['tiles']),
        
        # Stage 2 (no grounding) statistics
        'stage2_no_grounding': {
            'mean': float(np.mean(sar_v2)),
            'std': float(np.std(sar_v2)),
            'min': float(np.min(sar_v2)),
            'max': float(np.max(sar_v2)),
            'median': float(np.median(sar_v2))
        },
        
        # Stage 3 (grounded) statistics
        'stage3_grounded': {
            'mean': float(np.mean(sar_final)),
            'std': float(np.std(sar_final)),
            'min': float(np.min(sar_final)),
            'max': float(np.max(sar_final)),
            'median': float(np.median(sar_final))
        },
        
        # Delta (improvement) statistics
        'improvement': {
            'mean_delta': float(np.mean(delta)),
            'std_delta': float(np.std(delta)),
            'min_delta': float(np.min(delta)),
            'max_delta': float(np.max(delta)),
            'median_delta': float(np.median(delta)),
            'percent_improved': float(np.sum(delta > 0) / len(delta) * 100),
            'percent_degraded': float(np.sum(delta < 0) / len(delta) * 100)
        }
    }
    
    # Paired t-test
    if len(sar_v2) > 1:
        t_stat, p_value = stats.ttest_rel(sar_final, sar_v2)
        stats_dict['statistical_test'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_0.05': bool(p_value < 0.05),
            'significant_at_0.01': bool(p_value < 0.01)
        }
    
    # Effect size (Cohen's d for paired samples)
    if len(delta) > 1:
        cohens_d = np.mean(delta) / (np.std(delta) + 1e-8)
        stats_dict['effect_size'] = {
            'cohens_d': float(cohens_d),
            'interpretation': interpret_cohens_d(cohens_d)
        }
    
    return stats_dict


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


# ============================================================================
# Visualization
# ============================================================================

def plot_ablation_results(results: Dict, output_path: Path):
    """
    Create visualization of ablation results.
    
    Args:
        results: Results dict
        output_path: Output image path
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        warnings.warn("Matplotlib not available, skipping plot")
        return
    
    sar_v2 = np.array(results['sar_agreement_v2'])
    sar_final = np.array(results['sar_agreement_final'])
    delta = np.array(results['delta'])
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Histogram comparison
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1, 30)
    ax1.hist(sar_v2, bins=bins, alpha=0.6, label='Stage 2 (No Grounding)', color='red', edgecolor='black')
    ax1.hist(sar_final, bins=bins, alpha=0.6, label='Stage 3 (Grounded)', color='green', edgecolor='black')
    ax1.axvline(np.mean(sar_v2), color='red', linestyle='--', linewidth=2, label=f'V2 Mean: {np.mean(sar_v2):.4f}')
    ax1.axvline(np.mean(sar_final), color='green', linestyle='--', linewidth=2, label=f'Final Mean: {np.mean(sar_final):.4f}')
    ax1.set_xlabel('SAR Edge Agreement', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of SAR Agreement\n(Stage 2 vs Stage 3)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 2. Delta histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(delta, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='black', linestyle='-', linewidth=2)
    ax2.axvline(np.mean(delta), color='red', linestyle='--', linewidth=2, label=f'Mean Δ: {np.mean(delta):.4f}')
    ax2.set_xlabel('Δ SAR Agreement (Final - V2)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Improvement Distribution\n(Physics Grounding Effect)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # 3. Scatter plot
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(sar_v2, sar_final, alpha=0.5, s=50)
    min_val = min(sar_v2.min(), sar_final.min())
    max_val = max(sar_v2.max(), sar_final.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='No Change Line')
    ax3.set_xlabel('Stage 2 SAR Agreement', fontsize=12)
    ax3.set_ylabel('Stage 3 SAR Agreement', fontsize=12)
    ax3.set_title('Per-Tile Comparison\n(Above line = Improved)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    
    # 4. Box plot comparison
    ax4 = fig.add_subplot(gs[1, 0])
    box_data = [sar_v2, sar_final]
    bp = ax4.boxplot(box_data, labels=['Stage 2\n(No Grounding)', 'Stage 3\n(Grounded)'],
                     patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('red')
    bp['boxes'][1].set_facecolor('green')
    for patch in bp['boxes']:
        patch.set_alpha(0.6)
    ax4.set_ylabel('SAR Edge Agreement', fontsize=12)
    ax4.set_title('Statistical Comparison', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Cumulative distribution
    ax5 = fig.add_subplot(gs[1, 1])
    sorted_v2 = np.sort(sar_v2)
    sorted_final = np.sort(sar_final)
    cumulative_v2 = np.arange(1, len(sorted_v2) + 1) / len(sorted_v2)
    cumulative_final = np.arange(1, len(sorted_final) + 1) / len(sorted_final)
    ax5.plot(sorted_v2, cumulative_v2, label='Stage 2 (No Grounding)', color='red', linewidth=2)
    ax5.plot(sorted_final, cumulative_final, label='Stage 3 (Grounded)', color='green', linewidth=2)
    ax5.set_xlabel('SAR Edge Agreement', fontsize=12)
    ax5.set_ylabel('Cumulative Probability', fontsize=12)
    ax5.set_title('Cumulative Distribution\n(Rightward shift = Better)', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)
    
    # 6. Statistics panel
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    stats_text = f"""
ABLATION STUDY RESULTS
{'=' * 40}

Dataset:
  Total tiles analyzed: {len(results['tiles'])}

Stage 2 (No Physics Grounding):
  Mean SAR Agreement:  {np.mean(sar_v2):.6f}
  Std Dev:             {np.std(sar_v2):.6f}
  Median:              {np.median(sar_v2):.6f}
  Range:               [{np.min(sar_v2):.4f}, {np.max(sar_v2):.4f}]

Stage 3 (SAR Grounded):
  Mean SAR Agreement:  {np.mean(sar_final):.6f}
  Std Dev:             {np.std(sar_final):.6f}
  Median:              {np.median(sar_final):.6f}
  Range:               [{np.min(sar_final):.4f}, {np.max(sar_final):.4f}]

Improvement (Δ):
  Mean Δ:              {np.mean(delta):.6f} {'✓' if np.mean(delta) > 0 else '✗'}
  Std Dev:             {np.std(delta):.6f}
  Median Δ:            {np.median(delta):.6f}
  % Tiles Improved:    {np.sum(delta > 0) / len(delta) * 100:.1f}%
  % Tiles Degraded:    {np.sum(delta < 0) / len(delta) * 100:.1f}%

Statistical Significance:
  Paired t-test p-value: {stats.ttest_rel(sar_final, sar_v2)[1]:.4e}
  Significant (α=0.05): {'Yes ✓' if stats.ttest_rel(sar_final, sar_v2)[1] < 0.05 else 'No ✗'}

Effect Size:
  Cohen's d:           {np.mean(delta) / (np.std(delta) + 1e-8):.4f}
  Interpretation:      {interpret_cohens_d(np.mean(delta) / (np.std(delta) + 1e-8))}

Conclusion:
  {'Physics grounding IMPROVES SAR alignment!' if np.mean(delta) > 0 else 'No improvement observed.'}
    """
    
    ax6.text(0.05, 0.95, stats_text.strip(),
            transform=ax6.transAxes,
            fontsize=9,
            verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Main title
    improvement_pct = (np.mean(sar_final) - np.mean(sar_v2)) / np.mean(sar_v2) * 100
    fig.suptitle(
        f'Ablation Study: Impact of SAR Physics Grounding\n'
        f'Mean SAR Agreement: {np.mean(sar_v2):.4f} → {np.mean(sar_final):.4f} '
        f'(Δ = {np.mean(delta):.4f}, +{improvement_pct:.2f}%)',
        fontsize=16,
        fontweight='bold'
    )
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ablation study: Compare no-ground (Stage 2) vs grounded (Stage 3)"
    )
    
    # Required arguments
    parser.add_argument('--opt-v2-dir', type=str, required=True,
                       help='Directory with Stage 2 outputs (no grounding)')
    parser.add_argument('--opt-final-dir', type=str, required=True,
                       help='Directory with Stage 3 outputs (grounded)')
    parser.add_argument('--s1-dir', type=str, required=True,
                       help='Directory with SAR inputs')
    
    # Tile selection
    parser.add_argument('--n-tiles', type=int, default=50,
                       help='Number of tiles to analyze (-1 for all, default: 50)')
    parser.add_argument('--tile-list', type=str,
                       help='Text file with tile names to analyze (one per line)')
    
    # Output options
    parser.add_argument('--save-results', type=str,
                       help='Save detailed results to JSON file')
    parser.add_argument('--plot', type=str,
                       help='Save visualization plot to file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Print header
    if verbose:
        print("\n" + "=" * 80)
        print("Ablation Study: SAR Physics Grounding Impact")
        print("=" * 80)
        print("\nConfiguration:")
        print(f"  Stage 2 dir (no grounding): {args.opt_v2_dir}")
        print(f"  Stage 3 dir (grounded):     {args.opt_final_dir}")
        print(f"  SAR dir:                    {args.s1_dir}")
        print(f"  N tiles:                    {args.n_tiles if args.n_tiles > 0 else 'all'}")
        print()
    
    # Find tile triplets
    if verbose:
        print("Finding matching tiles...")
    
    tile_triplets = find_matching_tiles(
        Path(args.opt_v2_dir),
        Path(args.opt_final_dir),
        Path(args.s1_dir),
        max_tiles=args.n_tiles
    )
    
    if len(tile_triplets) == 0:
        print("❌ No matching tiles found!")
        return 1
    
    if verbose:
        print(f"✓ Found {len(tile_triplets)} matching tile triplets")
        print()
    
    # Run ablation
    if verbose:
        print("Running ablation analysis...")
        print()
    
    results = run_ablation(tile_triplets, verbose=verbose)
    
    if len(results['tiles']) == 0:
        print("❌ No tiles successfully processed!")
        return 1
    
    # Compute statistics
    stats_dict = compute_statistics(results)
    
    # Print results
    print("\n" + "=" * 80)
    print("ABLATION RESULTS")
    print("=" * 80)
    
    print(f"\nDataset:")
    print(f"  Tiles analyzed: {stats_dict['n_tiles']}")
    
    print(f"\nStage 2 (No Physics Grounding):")
    print(f"  Mean SAR Agreement: {stats_dict['stage2_no_grounding']['mean']:.6f}")
    print(f"  Std Dev:            {stats_dict['stage2_no_grounding']['std']:.6f}")
    print(f"  Range:              [{stats_dict['stage2_no_grounding']['min']:.4f}, "
          f"{stats_dict['stage2_no_grounding']['max']:.4f}]")
    
    print(f"\nStage 3 (SAR Grounded):")
    print(f"  Mean SAR Agreement: {stats_dict['stage3_grounded']['mean']:.6f}")
    print(f"  Std Dev:            {stats_dict['stage3_grounded']['std']:.6f}")
    print(f"  Range:              [{stats_dict['stage3_grounded']['min']:.4f}, "
          f"{stats_dict['stage3_grounded']['max']:.4f}]")
    
    print(f"\n>>> MEAN SAR-AGREEMENT DELTA: {stats_dict['improvement']['mean_delta']:.6f} <<<")
    
    improvement_pct = (stats_dict['stage3_grounded']['mean'] - stats_dict['stage2_no_grounding']['mean']) / \
                      stats_dict['stage2_no_grounding']['mean'] * 100
    
    print(f"\nImprovement:")
    print(f"  Absolute:           {stats_dict['improvement']['mean_delta']:.6f}")
    print(f"  Relative:           {improvement_pct:+.2f}%")
    print(f"  Std Dev:            {stats_dict['improvement']['std_delta']:.6f}")
    print(f"  Tiles Improved:     {stats_dict['improvement']['percent_improved']:.1f}%")
    print(f"  Tiles Degraded:     {stats_dict['improvement']['percent_degraded']:.1f}%")
    
    if 'statistical_test' in stats_dict:
        print(f"\nStatistical Significance:")
        print(f"  Paired t-test:")
        print(f"    t-statistic:    {stats_dict['statistical_test']['t_statistic']:.4f}")
        print(f"    p-value:        {stats_dict['statistical_test']['p_value']:.4e}")
        print(f"    Significant:    {'Yes ✓' if stats_dict['statistical_test']['significant_at_0.05'] else 'No ✗'} (α=0.05)")
        
        if stats_dict['statistical_test']['significant_at_0.01']:
            print(f"                    Yes ✓✓ (α=0.01)")
    
    if 'effect_size' in stats_dict:
        print(f"\nEffect Size:")
        print(f"  Cohen's d:          {stats_dict['effect_size']['cohens_d']:.4f}")
        print(f"  Interpretation:     {stats_dict['effect_size']['interpretation']}")
    
    print(f"\nConclusion:")
    if stats_dict['improvement']['mean_delta'] > 0:
        print(f"  ✓ Physics grounding IMPROVES SAR edge alignment")
        print(f"  ✓ Mean improvement of {stats_dict['improvement']['mean_delta']:.6f} ({improvement_pct:+.2f}%)")
        if 'statistical_test' in stats_dict and stats_dict['statistical_test']['significant_at_0.05']:
            print(f"  ✓ Improvement is statistically significant")
    else:
        print(f"  ✗ No improvement observed from physics grounding")
    
    print("=" * 80)
    
    # Save results
    if args.save_results:
        output_data = {
            'statistics': stats_dict,
            'per_tile_results': {
                'tiles': results['tiles'],
                'sar_agreement_v2': results['sar_agreement_v2'],
                'sar_agreement_final': results['sar_agreement_final'],
                'delta': results['delta']
            }
        }
        
        with open(args.save_results, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        if verbose:
            print(f"\n✓ Detailed results saved to: {args.save_results}")
    
    # Create plot
    if args.plot:
        if verbose:
            print(f"\nGenerating visualization...")
        
        plot_ablation_results(results, Path(args.plot))
        
        if verbose:
            print(f"✓ Plot saved to: {args.plot}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
