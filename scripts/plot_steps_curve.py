"""
Plot SAR-Agreement vs Timesteps Curve

This script evaluates how SAR edge agreement varies with the number of 
diffusion timesteps in Stage 3 inference. It helps determine the optimal
timesteps setting for balancing quality and inference speed.

The curve shows:
- X-axis: Number of diffusion timesteps (2, 5, 10, 15, 20)
- Y-axis: Mean SAR edge agreement across test tiles
- ARS-S v1 threshold line at 0.6

Usage:
    # Basic usage with default settings
    python scripts/plot_steps_curve.py \
        --checkpoint checkpoints/stage3/best_model.pt \
        --opt-v2-dir stage2_outputs/ \
        --s1-dir tiles/s1/ \
        --output reports/stage3_steps_curve.png

    # Custom timestep range
    python scripts/plot_steps_curve.py \
        --checkpoint checkpoints/stage3/best_model.pt \
        --opt-v2-dir stage2_outputs/ \
        --s1-dir tiles/s1/ \
        --output reports/stage3_steps_curve.png \
        --timesteps 2 5 8 10 12 15 20

    # Use subset of tiles for faster evaluation
    python scripts/plot_steps_curve.py \
        --checkpoint checkpoints/stage3/best_model.pt \
        --opt-v2-dir stage2_outputs/ \
        --s1-dir tiles/s1/ \
        --output reports/stage3_steps_curve.png \
        --max-tiles 20

Author: Axion-Sat Project
Version: 1.0.0
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
import json

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axs_lib.stage3_tm_ground import build_stage3_model
from axs_lib.io import load_checkpoint


# ============================================================================
# Data Loading (copied from infer_stage3.py)
# ============================================================================

def find_tile_pairs(
    opt_v2_dir: Path,
    s1_dir: Path,
    pattern: str = '*.npz',
    max_tiles: int = -1
) -> List[Tuple[Path, Path]]:
    """Find matching pairs of opt_v2 and S1 tiles."""
    opt_v2_dir = Path(opt_v2_dir)
    s1_dir = Path(s1_dir)
    
    opt_v2_tiles = {p.name: p for p in opt_v2_dir.rglob(pattern)}
    
    pairs = []
    for s1_tile in s1_dir.rglob(pattern):
        tile_name = s1_tile.name
        if tile_name in opt_v2_tiles:
            pairs.append((opt_v2_tiles[tile_name], s1_tile))
    
    pairs = sorted(pairs)
    
    if max_tiles > 0:
        pairs = pairs[:max_tiles]
    
    return pairs


def load_opt_v2_tile(tile_path: Path) -> np.ndarray:
    """Load opt_v2 tile from NPZ file."""
    try:
        data = np.load(tile_path)
        
        if 'opt_v2' in data:
            opt_v2 = data['opt_v2']
            if opt_v2.ndim == 2:
                opt_v2 = np.stack([opt_v2] * 4, axis=0)
            return opt_v2.astype(np.float32)
        
        if all(f's2_b{b}' in data for b in [2, 3, 4, 8]):
            opt_v2 = np.stack([
                data['s2_b2'], data['s2_b3'], data['s2_b4'], data['s2_b8']
            ], axis=0)
            return opt_v2.astype(np.float32)
        
        warnings.warn(f"Could not find opt_v2 data in {tile_path}")
        return None
        
    except Exception as e:
        warnings.warn(f"Error loading opt_v2 from {tile_path}: {e}")
        return None


def load_s1_tile(tile_path: Path) -> np.ndarray:
    """Load S1 SAR tile from NPZ file."""
    try:
        data = np.load(tile_path)
        
        if 's1' in data:
            s1 = data['s1']
            if s1.ndim == 2:
                s1 = np.stack([s1, s1], axis=0)
            return s1.astype(np.float32)
        
        if 's1_vv' in data and 's1_vh' in data:
            s1 = np.stack([data['s1_vv'], data['s1_vh']], axis=0)
            return s1.astype(np.float32)
        
        warnings.warn(f"Could not find S1 data in {tile_path}")
        return None
        
    except Exception as e:
        warnings.warn(f"Error loading S1 from {tile_path}: {e}")
        return None


# ============================================================================
# SAR Agreement Computation
# ============================================================================

def compute_sar_agreement_numpy(opt_final: np.ndarray, s1: np.ndarray) -> float:
    """
    Compute SAR edge agreement for numpy arrays.
    
    Args:
        opt_final: Optical output (4, H, W)
        s1: SAR input (2, H, W)
        
    Returns:
        Agreement score [0, 1], higher is better
    """
    # Convert to grayscale
    opt_gray = 0.2989 * opt_final[2] + 0.5870 * opt_final[1] + 0.1140 * opt_final[0]
    s1_gray = s1.mean(axis=0)
    
    # Normalize
    opt_gray = (opt_gray - opt_gray.min()) / (opt_gray.max() - opt_gray.min() + 1e-8)
    s1_gray = (s1_gray - s1_gray.min()) / (s1_gray.max() - s1_gray.min() + 1e-8)
    
    # Compute edges using Sobel
    opt_edges = ndimage.sobel(opt_gray)
    s1_edges = ndimage.sobel(s1_gray)
    
    # Normalize edges
    opt_edges = (opt_edges - opt_edges.min()) / (opt_edges.max() - opt_edges.min() + 1e-8)
    s1_edges = (s1_edges - s1_edges.min()) / (s1_edges.max() - s1_edges.min() + 1e-8)
    
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
# Timesteps Evaluation
# ============================================================================

def evaluate_timesteps(
    model: torch.nn.Module,
    tile_pairs: List[Tuple[Path, Path]],
    timesteps_list: List[int],
    device: torch.device,
    verbose: bool = True
) -> Dict[int, List[float]]:
    """
    Evaluate SAR agreement across different timestep counts.
    
    Args:
        model: Stage 3 model
        tile_pairs: List of (opt_v2_path, s1_path) tuples
        timesteps_list: List of timestep values to evaluate
        device: Device to run on
        verbose: Show progress
        
    Returns:
        Dict mapping timesteps to list of SAR agreement scores
    """
    results = {t: [] for t in timesteps_list}
    
    if verbose:
        print(f"\nEvaluating {len(timesteps_list)} timestep settings on {len(tile_pairs)} tiles...")
    
    model.eval()
    
    for timesteps in timesteps_list:
        if verbose:
            print(f"\n  Timesteps = {timesteps:2d}:")
            pbar = tqdm(tile_pairs, desc=f"    Processing", leave=False)
        else:
            pbar = tile_pairs
        
        for opt_v2_path, s1_path in pbar:
            # Load tiles
            opt_v2 = load_opt_v2_tile(opt_v2_path)
            s1 = load_s1_tile(s1_path)
            
            if opt_v2 is None or s1 is None:
                continue
            
            # Ensure same spatial size
            if opt_v2.shape[1:] != s1.shape[1:]:
                s1_tensor = torch.from_numpy(s1).unsqueeze(0)
                s1_tensor = F.interpolate(
                    s1_tensor,
                    size=opt_v2.shape[1:],
                    mode='bilinear',
                    align_corners=False
                )
                s1 = s1_tensor.squeeze(0).numpy()
            
            # Run inference with specific timesteps
            try:
                with torch.no_grad():
                    opt_v2_tensor = torch.from_numpy(opt_v2).unsqueeze(0).to(device)
                    s1_tensor = torch.from_numpy(s1).unsqueeze(0).to(device)
                    
                    opt_final = model(s1_tensor, opt_v2_tensor, timesteps=timesteps)
                    
                    opt_final = opt_final.squeeze(0).cpu().numpy()
                
                # Compute SAR agreement
                sar_agreement = compute_sar_agreement_numpy(opt_final, s1)
                results[timesteps].append(sar_agreement)
                
            except Exception as e:
                warnings.warn(f"Error processing {opt_v2_path.name} with timesteps={timesteps}: {e}")
                continue
        
        if verbose and hasattr(pbar, 'close'):
            pbar.close()
        
        if len(results[timesteps]) > 0:
            mean_agreement = np.mean(results[timesteps])
            if verbose:
                print(f"    Mean SAR agreement: {mean_agreement:.4f} (n={len(results[timesteps])})")
        else:
            if verbose:
                print(f"    No valid results")
    
    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_steps_curve(
    results: Dict[int, List[float]],
    output_path: Path,
    title: str = "SAR Agreement vs Diffusion Timesteps"
):
    """
    Create comprehensive plot of SAR agreement vs timesteps.
    
    Args:
        results: Dict mapping timesteps to SAR agreement scores
        output_path: Output path for plot
        title: Plot title
    """
    # Compute statistics
    timesteps = sorted(results.keys())
    means = []
    stds = []
    mins = []
    maxs = []
    
    for t in timesteps:
        scores = results[t]
        if len(scores) > 0:
            means.append(np.mean(scores))
            stds.append(np.std(scores))
            mins.append(np.min(scores))
            maxs.append(np.max(scores))
        else:
            means.append(np.nan)
            stds.append(np.nan)
            mins.append(np.nan)
            maxs.append(np.nan)
    
    means = np.array(means)
    stds = np.array(stds)
    mins = np.array(mins)
    maxs = np.array(maxs)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # ========================================================================
    # Main curve with error bars
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot mean with error bars (std)
    ax1.errorbar(timesteps, means, yerr=stds, 
                marker='o', markersize=8, linewidth=2, capsize=5,
                label='Mean ± Std', color='blue', alpha=0.8)
    
    # Plot min/max range
    ax1.fill_between(timesteps, mins, maxs, alpha=0.2, color='blue', label='Min-Max Range')
    
    # ARS-S v1 threshold line
    ax1.axhline(y=0.6, color='red', linestyle='--', linewidth=2, label='ARS-S v1 Threshold (0.6)')
    ax1.axhline(y=0.7, color='orange', linestyle=':', linewidth=1.5, label='Good Threshold (0.7)')
    ax1.axhline(y=0.8, color='green', linestyle=':', linewidth=1.5, label='Excellent Threshold (0.8)')
    
    ax1.set_xlabel('Number of Diffusion Timesteps', fontsize=14, fontweight='bold')
    ax1.set_ylabel('SAR Edge Agreement', fontsize=14, fontweight='bold')
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='lower right')
    ax1.set_ylim([0, 1.0])
    
    # Annotate optimal point
    optimal_idx = np.nanargmax(means)
    optimal_timesteps = timesteps[optimal_idx]
    optimal_agreement = means[optimal_idx]
    
    ax1.annotate(
        f'Optimal: {optimal_timesteps} steps\n{optimal_agreement:.4f}',
        xy=(optimal_timesteps, optimal_agreement),
        xytext=(optimal_timesteps + 2, optimal_agreement - 0.1),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='green', lw=2),
        fontsize=12,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7)
    )
    
    # ========================================================================
    # Bar chart comparison
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    
    colors = ['green' if m >= 0.8 else 'orange' if m >= 0.7 else 'red' if m >= 0.6 else 'darkred' 
              for m in means]
    
    bars = ax2.bar(timesteps, means, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (t, m) in enumerate(zip(timesteps, means)):
        if not np.isnan(m):
            ax2.text(t, m + 0.02, f'{m:.3f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    ax2.axhline(y=0.6, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Timesteps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean SAR Agreement', fontsize=12, fontweight='bold')
    ax2.set_title('Mean SAR Agreement by Timesteps', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim([0, 1.0])
    
    # ========================================================================
    # Statistics table
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    # Build statistics text
    stats_lines = [
        "TIMESTEPS EVALUATION SUMMARY",
        "=" * 45,
        "",
        f"Timesteps evaluated:  {len(timesteps)}",
        f"Range:                {min(timesteps)} to {max(timesteps)}",
        f"Tiles per setting:    {len(results[timesteps[0]])}",
        "",
        "Results:",
        "-" * 45
    ]
    
    for t, m, s, mn, mx in zip(timesteps, means, stds, mins, maxs):
        if not np.isnan(m):
            status = "✓✓" if m >= 0.8 else "✓" if m >= 0.7 else "○" if m >= 0.6 else "✗"
            stats_lines.append(
                f"  {t:2d} steps: {m:.4f} ± {s:.4f}  [{mn:.3f}, {mx:.3f}]  {status}"
            )
    
    stats_lines.extend([
        "",
        "-" * 45,
        f"Optimal:              {optimal_timesteps} steps ({optimal_agreement:.4f})",
        "",
        "Legend:",
        "  ✓✓  Excellent (≥ 0.8)",
        "  ✓   Good (≥ 0.7)",
        "  ○   Acceptable (≥ 0.6, ARS-S v1)",
        "  ✗   Below threshold (< 0.6)",
        "",
        "Recommendation:",
    ])
    
    # Add recommendation
    if optimal_agreement >= 0.8:
        stats_lines.append(f"  Use {optimal_timesteps} steps for best quality.")
    elif optimal_agreement >= 0.7:
        stats_lines.append(f"  Use {optimal_timesteps} steps for good quality.")
        # Check if fewer steps give acceptable quality
        for t, m in zip(timesteps, means):
            if t < optimal_timesteps and m >= 0.6:
                stats_lines.append(f"  Or {t} steps for faster inference (SAR {m:.3f}).")
                break
    else:
        stats_lines.append(f"  ⚠ All settings below optimal threshold.")
        stats_lines.append(f"  Consider retraining model.")
    
    stats_text = "\n".join(stats_lines)
    
    ax3.text(0.05, 0.95, stats_text,
            transform=ax3.transAxes,
            fontsize=9,
            verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # ========================================================================
    # Save
    # ========================================================================
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot SAR agreement vs diffusion timesteps curve"
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to Stage 3 checkpoint')
    parser.add_argument('--opt-v2-dir', type=str, required=True,
                       help='Directory with Stage 2 outputs')
    parser.add_argument('--s1-dir', type=str, required=True,
                       help='Directory with SAR inputs')
    parser.add_argument('--output', type=str, default='reports/stage3_steps_curve.png',
                       help='Output path for plot (default: reports/stage3_steps_curve.png)')
    
    # Evaluation options
    parser.add_argument('--timesteps', type=int, nargs='+', default=[2, 5, 10, 15, 20],
                       help='Timestep values to evaluate (default: 2 5 10 15 20)')
    parser.add_argument('--max-tiles', type=int, default=-1,
                       help='Maximum tiles to evaluate (-1 for all, default: -1)')
    
    # Other options
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, default: auto)')
    parser.add_argument('--save-results', type=str,
                       help='Save raw results to JSON file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Print header
    if verbose:
        print("\n" + "=" * 80)
        print("SAR Agreement vs Timesteps Curve")
        print("=" * 80)
        print("\nConfiguration:")
        print(f"  Checkpoint:     {args.checkpoint}")
        print(f"  opt_v2 dir:     {args.opt_v2_dir}")
        print(f"  S1 dir:         {args.s1_dir}")
        print(f"  Output:         {args.output}")
        print(f"  Timesteps:      {args.timesteps}")
        print(f"  Max tiles:      {args.max_tiles if args.max_tiles > 0 else 'all'}")
        print()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"Using device: {device}")
        print()
    
    # Load model
    if verbose:
        print("Loading Stage 3 model...")
    
    checkpoint = load_checkpoint(Path(args.checkpoint))
    
    model = build_stage3_model(
        timesteps=max(args.timesteps),  # Use max for initialization
        standardize=True,
        pretrained=False,
        device=device
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if verbose:
        print(f"✓ Model loaded")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print()
    
    # Find tile pairs
    if verbose:
        print("Finding tile pairs...")
    
    tile_pairs = find_tile_pairs(
        Path(args.opt_v2_dir),
        Path(args.s1_dir),
        max_tiles=args.max_tiles
    )
    
    if len(tile_pairs) == 0:
        print("❌ No matching tile pairs found!")
        return 1
    
    if verbose:
        print(f"✓ Found {len(tile_pairs)} matching tile pairs")
    
    # Evaluate timesteps
    results = evaluate_timesteps(
        model=model,
        tile_pairs=tile_pairs,
        timesteps_list=args.timesteps,
        device=device,
        verbose=verbose
    )
    
    # Check if any results
    if all(len(scores) == 0 for scores in results.values()):
        print("❌ No valid results obtained!")
        return 1
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Plot results
    if verbose:
        print(f"\nGenerating plot...")
    
    plot_steps_curve(results, output_path)
    
    if verbose:
        print(f"✓ Plot saved to: {output_path}")
    
    # Save raw results if requested
    if args.save_results:
        results_data = {
            'timesteps': args.timesteps,
            'results': {str(k): v for k, v in results.items()},
            'statistics': {
                str(t): {
                    'mean': float(np.mean(results[t])) if len(results[t]) > 0 else None,
                    'std': float(np.std(results[t])) if len(results[t]) > 0 else None,
                    'min': float(np.min(results[t])) if len(results[t]) > 0 else None,
                    'max': float(np.max(results[t])) if len(results[t]) > 0 else None,
                    'n_tiles': len(results[t])
                }
                for t in args.timesteps
            },
            'optimal_timesteps': int(args.timesteps[np.argmax([np.mean(results[t]) for t in args.timesteps])]),
            'optimal_agreement': float(max([np.mean(results[t]) for t in args.timesteps]))
        }
        
        with open(args.save_results, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        if verbose:
            print(f"✓ Raw results saved to: {args.save_results}")
    
    # Print summary
    if verbose:
        print("\n" + "=" * 80)
        print("Evaluation Complete!")
        print("=" * 80)
        
        print("\nSummary:")
        for t in args.timesteps:
            if len(results[t]) > 0:
                mean_agr = np.mean(results[t])
                status = "✓" if mean_agr >= 0.6 else "✗"
                print(f"  {t:2d} steps: {mean_agr:.4f} {status}")
        
        optimal_idx = np.argmax([np.mean(results[t]) for t in args.timesteps])
        optimal_t = args.timesteps[optimal_idx]
        optimal_agr = np.mean(results[optimal_t])
        
        print(f"\nOptimal: {optimal_t} steps (SAR agreement: {optimal_agr:.4f})")
        print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
