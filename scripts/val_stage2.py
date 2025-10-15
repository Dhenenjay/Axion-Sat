"""
Stage 2 Validation Script: opt_v2 vs opt_v1 Quality Assessment

This script validates Stage 2 refinement by comparing opt_v2 (Prithvi refined)
against opt_v1 (TerraMind "mental images") to ensure:

Expected Improvements:
- NDVI RMSE ↓ (closer to ground truth vegetation)
- EVI RMSE ↓ (better canopy sensitivity)
- SAM ↓ (improved spectral similarity)

Expected Preservation:
- LPIPS stable (perceptual similarity maintained)
- Edge displacement ≤ 2px (spatial structure preserved)

Outputs:
- CSV: Per-tile metrics
- Summary CSV: Aggregate statistics
- Plots: Distribution histograms, scatter plots, example tiles

Usage:
    python scripts/val_stage2.py \\
        --v1_dir data/stage1_outputs \\
        --v2_dir data/stage2_outputs \\
        --gt_dir data/ground_truth \\
        --output_dir outputs/stage2_validation

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
from pathlib import Path
import argparse
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Windows

try:
    import torch
    from scipy import ndimage
    from scipy.spatial.distance import cosine
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    warnings.warn(f"Required dependencies not available: {e}")


# ============================================================================
# Vegetation Index Computation
# ============================================================================

def compute_ndvi(optical: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute NDVI from 4-band optical.
    
    Args:
        optical: Array (4, H, W) - [B02, B03, B04, B08]
        eps: Small constant for numerical stability
        
    Returns:
        NDVI array (H, W) in range [-1, 1]
    """
    red = optical[2]  # B04
    nir = optical[3]  # B08
    return (nir - red) / (nir + red + eps)


def compute_evi(optical: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute EVI from 4-band optical.
    
    Args:
        optical: Array (4, H, W) - [B02, B03, B04, B08]
        eps: Small constant for numerical stability
        
    Returns:
        EVI array (H, W) in range [-1, 1]
    """
    blue = optical[0]  # B02
    red = optical[2]   # B04
    nir = optical[3]   # B08
    
    G = 2.5
    C1 = 6.0
    C2 = 7.5
    L = 1.0
    
    numerator = G * (nir - red)
    denominator = nir + C1 * red - C2 * blue + L + eps
    
    return numerator / denominator


# ============================================================================
# Spectral Angle Mapper (SAM)
# ============================================================================

def compute_sam(spec1: np.ndarray, spec2: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute Spectral Angle Mapper (SAM) between two spectra.
    
    Args:
        spec1: First spectrum (4, H, W) or (C, H*W)
        spec2: Second spectrum (4, H, W) or (C, H*W)
        eps: Small constant for numerical stability
        
    Returns:
        Mean SAM angle in radians
    """
    # Flatten spatial dimensions
    spec1_flat = spec1.reshape(spec1.shape[0], -1)  # (C, N)
    spec2_flat = spec2.reshape(spec2.shape[0], -1)  # (C, N)
    
    # Compute dot product
    dot_product = (spec1_flat * spec2_flat).sum(axis=0)  # (N,)
    
    # Compute norms
    norm1 = np.linalg.norm(spec1_flat, axis=0) + eps  # (N,)
    norm2 = np.linalg.norm(spec2_flat, axis=0) + eps  # (N,)
    
    # Compute cosine similarity
    cos_angle = dot_product / (norm1 * norm2)
    
    # Clamp to valid range for arccos
    cos_angle = np.clip(cos_angle, -1.0 + eps, 1.0 - eps)
    
    # Compute angle
    angle = np.arccos(cos_angle)
    
    # Return mean angle
    return angle.mean()


# ============================================================================
# LPIPS (Learned Perceptual Image Patch Similarity)
# ============================================================================

def compute_lpips_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute simplified LPIPS-like metric using feature differences.
    
    Note: This is a simplified version without the full LPIPS network.
    For production, use the actual lpips library.
    
    Args:
        img1: First image (C, H, W)
        img2: Second image (C, H, W)
        
    Returns:
        Perceptual distance (lower is better)
    """
    # Compute multi-scale feature distances
    scales = [1, 2, 4]
    distances = []
    
    for scale in scales:
        if scale > 1:
            # Downsample
            h, w = img1.shape[1] // scale, img1.shape[2] // scale
            img1_scaled = ndimage.zoom(img1, (1, 1/scale, 1/scale), order=1)
            img2_scaled = ndimage.zoom(img2, (1, 1/scale, 1/scale), order=1)
        else:
            img1_scaled = img1
            img2_scaled = img2
        
        # Compute L2 distance
        dist = np.mean((img1_scaled - img2_scaled) ** 2)
        distances.append(dist)
    
    # Weighted average
    return np.mean(distances)


def compute_lpips_torch(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute LPIPS using PyTorch (if available).
    
    Args:
        img1: First image (C, H, W)
        img2: Second image (C, H, W)
        
    Returns:
        LPIPS distance
    """
    try:
        import lpips
        
        # Convert to torch tensors
        img1_torch = torch.from_numpy(img1).unsqueeze(0).float()
        img2_torch = torch.from_numpy(img2).unsqueeze(0).float()
        
        # Normalize to [-1, 1]
        img1_torch = img1_torch * 2 - 1
        img2_torch = img2_torch * 2 - 1
        
        # Create LPIPS model
        loss_fn = lpips.LPIPS(net='alex')
        
        # Compute LPIPS
        with torch.no_grad():
            distance = loss_fn(img1_torch, img2_torch)
        
        return distance.item()
        
    except ImportError:
        # Fallback to simple version
        return compute_lpips_simple(img1, img2)


# ============================================================================
# Edge Displacement
# ============================================================================

def compute_edge_displacement(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute maximum edge displacement between two images.
    
    Args:
        img1: First image (C, H, W)
        img2: Second image (C, H, W)
        
    Returns:
        Maximum edge displacement in pixels
    """
    # Use NIR band for edge detection (most informative)
    nir1 = img1[3]
    nir2 = img2[3]
    
    # Compute edges using Sobel
    edges1_x = ndimage.sobel(nir1, axis=0)
    edges1_y = ndimage.sobel(nir1, axis=1)
    edges1 = np.sqrt(edges1_x**2 + edges1_y**2)
    
    edges2_x = ndimage.sobel(nir2, axis=0)
    edges2_y = ndimage.sobel(nir2, axis=1)
    edges2 = np.sqrt(edges2_x**2 + edges2_y**2)
    
    # Threshold edges
    threshold = 0.1
    edges1_binary = edges1 > threshold
    edges2_binary = edges2 > threshold
    
    # Compute distance transform
    dist1 = ndimage.distance_transform_edt(~edges1_binary)
    dist2 = ndimage.distance_transform_edt(~edges2_binary)
    
    # Find maximum displacement
    # For each edge pixel in img2, find nearest edge in img1
    displacement1 = dist1[edges2_binary].max() if edges2_binary.any() else 0
    displacement2 = dist2[edges1_binary].max() if edges1_binary.any() else 0
    
    max_displacement = max(displacement1, displacement2)
    
    return max_displacement


# ============================================================================
# Tile-Level Metrics
# ============================================================================

def compute_tile_metrics(
    v1: np.ndarray,
    v2: np.ndarray,
    gt: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all metrics for a single tile.
    
    Args:
        v1: opt_v1 array (4, H, W)
        v2: opt_v2 array (4, H, W)
        gt: Ground truth array (4, H, W), optional
        
    Returns:
        Dict of metrics
    """
    metrics = {}
    
    # Reference for comparison (use GT if available, else v1)
    reference = gt if gt is not None else v1
    
    # ========================================================================
    # Vegetation Indices (compare against reference)
    # ========================================================================
    
    # NDVI
    ndvi_v1 = compute_ndvi(v1)
    ndvi_v2 = compute_ndvi(v2)
    ndvi_ref = compute_ndvi(reference)
    
    metrics['ndvi_v1_rmse'] = np.sqrt(np.mean((ndvi_v1 - ndvi_ref)**2))
    metrics['ndvi_v2_rmse'] = np.sqrt(np.mean((ndvi_v2 - ndvi_ref)**2))
    metrics['ndvi_improvement'] = metrics['ndvi_v1_rmse'] - metrics['ndvi_v2_rmse']
    
    # EVI
    evi_v1 = compute_evi(v1)
    evi_v2 = compute_evi(v2)
    evi_ref = compute_evi(reference)
    
    metrics['evi_v1_rmse'] = np.sqrt(np.mean((evi_v1 - evi_ref)**2))
    metrics['evi_v2_rmse'] = np.sqrt(np.mean((evi_v2 - evi_ref)**2))
    metrics['evi_improvement'] = metrics['evi_v1_rmse'] - metrics['evi_v2_rmse']
    
    # ========================================================================
    # Spectral Angle Mapper (compare against reference)
    # ========================================================================
    
    metrics['sam_v1'] = compute_sam(v1, reference)
    metrics['sam_v2'] = compute_sam(v2, reference)
    metrics['sam_improvement'] = metrics['sam_v1'] - metrics['sam_v2']
    
    # ========================================================================
    # LPIPS (compare v2 vs v1 - should be stable)
    # ========================================================================
    
    metrics['lpips_v2_vs_v1'] = compute_lpips_torch(v2, v1)
    
    # ========================================================================
    # Edge Displacement (v2 vs v1)
    # ========================================================================
    
    metrics['edge_displacement'] = compute_edge_displacement(v1, v2)
    
    # ========================================================================
    # Additional Quality Metrics
    # ========================================================================
    
    # RMSE (raw pixel values)
    metrics['rmse_v1_vs_ref'] = np.sqrt(np.mean((v1 - reference)**2))
    metrics['rmse_v2_vs_ref'] = np.sqrt(np.mean((v2 - reference)**2))
    
    # PSNR
    if gt is not None:
        mse_v2 = np.mean((v2 - gt)**2)
        if mse_v2 > 0:
            metrics['psnr_v2'] = 10 * np.log10(1.0 / mse_v2)
        else:
            metrics['psnr_v2'] = 100.0
    
    return metrics


# ============================================================================
# Dataset Processing
# ============================================================================

def load_tile(tile_path: Path) -> Optional[np.ndarray]:
    """Load tile from NPZ file."""
    try:
        data = np.load(tile_path)
        
        # Try different key names
        if 'opt_v2' in data:
            return data['opt_v2']
        elif 'opt_v1' in data:
            return data['opt_v1']
        elif 's2_b2' in data:
            # Reconstruct from bands
            return np.stack([
                data['s2_b2'], data['s2_b3'],
                data['s2_b4'], data['s2_b8']
            ], axis=0)
        else:
            return None
    except Exception as e:
        warnings.warn(f"Error loading {tile_path}: {e}")
        return None


def validate_dataset(
    v1_dir: Path,
    v2_dir: Path,
    gt_dir: Optional[Path],
    output_dir: Path,
    max_tiles: Optional[int] = None
):
    """
    Validate Stage 2 refinement across dataset.
    
    Args:
        v1_dir: Directory with opt_v1 tiles
        v2_dir: Directory with opt_v2 tiles
        gt_dir: Directory with ground truth tiles (optional)
        output_dir: Directory to save results
        max_tiles: Maximum number of tiles to process
    """
    print("="*70)
    print("Stage 2 Validation: opt_v2 vs opt_v1")
    print("="*70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find matching tiles
    print("\nFinding tiles...")
    v1_tiles = sorted(v1_dir.glob('*.npz'))
    v2_tiles_dict = {t.name: t for t in v2_dir.glob('*.npz')}
    
    if gt_dir:
        gt_tiles_dict = {t.name: t for t in gt_dir.glob('*.npz')}
    else:
        gt_tiles_dict = {}
    
    # Match tiles
    matched_tiles = []
    for v1_tile in v1_tiles:
        if v1_tile.name in v2_tiles_dict:
            matched_tiles.append({
                'v1': v1_tile,
                'v2': v2_tiles_dict[v1_tile.name],
                'gt': gt_tiles_dict.get(v1_tile.name)
            })
    
    print(f"Found {len(v1_tiles)} opt_v1 tiles")
    print(f"Found {len(v2_tiles_dict)} opt_v2 tiles")
    print(f"Found {len(gt_tiles_dict)} ground truth tiles")
    print(f"Matched: {len(matched_tiles)} tiles")
    
    if not matched_tiles:
        print("⚠️  No matching tiles found!")
        return
    
    if max_tiles:
        matched_tiles = matched_tiles[:max_tiles]
        print(f"Processing first {max_tiles} tiles")
    
    # ========================================================================
    # Process Tiles
    # ========================================================================
    
    print(f"\nProcessing {len(matched_tiles)} tiles...\n")
    
    all_metrics = []
    failed_tiles = []
    
    for tile_info in tqdm(matched_tiles, desc="Validating"):
        try:
            # Load tiles
            v1 = load_tile(tile_info['v1'])
            v2 = load_tile(tile_info['v2'])
            gt = load_tile(tile_info['gt']) if tile_info['gt'] else None
            
            if v1 is None or v2 is None:
                failed_tiles.append(tile_info['v1'].name)
                continue
            
            # Ensure same shape
            if v1.shape != v2.shape:
                failed_tiles.append(tile_info['v1'].name)
                continue
            
            # Compute metrics
            metrics = compute_tile_metrics(v1, v2, gt)
            metrics['tile_name'] = tile_info['v1'].name
            metrics['has_gt'] = gt is not None
            
            all_metrics.append(metrics)
            
        except Exception as e:
            failed_tiles.append(tile_info['v1'].name)
            warnings.warn(f"Error processing {tile_info['v1'].name}: {e}")
    
    print(f"\nSuccessfully processed: {len(all_metrics)} tiles")
    if failed_tiles:
        print(f"Failed: {len(failed_tiles)} tiles")
    
    # ========================================================================
    # Save Per-Tile Metrics CSV
    # ========================================================================
    
    csv_path = output_dir / 'tile_metrics.csv'
    print(f"\nSaving per-tile metrics to: {csv_path}")
    
    if all_metrics:
        fieldnames = list(all_metrics[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
    
    # ========================================================================
    # Compute Summary Statistics
    # ========================================================================
    
    print("\nComputing summary statistics...")
    
    summary = {}
    
    # Extract metric values
    metric_names = [k for k in all_metrics[0].keys() 
                    if k not in ['tile_name', 'has_gt']]
    
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics 
                  if metric_name in m and not np.isnan(m[metric_name])]
        
        if values:
            summary[f'{metric_name}_mean'] = np.mean(values)
            summary[f'{metric_name}_std'] = np.std(values)
            summary[f'{metric_name}_median'] = np.median(values)
            summary[f'{metric_name}_min'] = np.min(values)
            summary[f'{metric_name}_max'] = np.max(values)
    
    # Save summary
    summary_path = output_dir / 'summary_statistics.csv'
    print(f"Saving summary statistics to: {summary_path}")
    
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'value'])
        writer.writeheader()
        for k, v in summary.items():
            writer.writerow({'metric': k, 'value': v})
    
    # ========================================================================
    # Print Summary
    # ========================================================================
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    print("\n📊 Vegetation Indices (Lower RMSE = Better)")
    print("-" * 70)
    print(f"NDVI RMSE v1: {summary['ndvi_v1_rmse_mean']:.6f} ± {summary['ndvi_v1_rmse_std']:.6f}")
    print(f"NDVI RMSE v2: {summary['ndvi_v2_rmse_mean']:.6f} ± {summary['ndvi_v2_rmse_std']:.6f}")
    print(f"Improvement:  {summary['ndvi_improvement_mean']:.6f} ± {summary['ndvi_improvement_std']:.6f}")
    
    if summary['ndvi_improvement_mean'] > 0:
        print("✓ NDVI improved (v2 closer to reference)")
    else:
        print("⚠️  NDVI degraded (v2 further from reference)")
    
    print(f"\nEVI RMSE v1: {summary['evi_v1_rmse_mean']:.6f} ± {summary['evi_v1_rmse_std']:.6f}")
    print(f"EVI RMSE v2: {summary['evi_v2_rmse_mean']:.6f} ± {summary['evi_v2_rmse_std']:.6f}")
    print(f"Improvement: {summary['evi_improvement_mean']:.6f} ± {summary['evi_improvement_std']:.6f}")
    
    if summary['evi_improvement_mean'] > 0:
        print("✓ EVI improved (v2 closer to reference)")
    else:
        print("⚠️  EVI degraded (v2 further from reference)")
    
    print("\n📐 Spectral Angle Mapper (Lower = Better)")
    print("-" * 70)
    print(f"SAM v1: {summary['sam_v1_mean']:.6f} rad ± {summary['sam_v1_std']:.6f}")
    print(f"SAM v2: {summary['sam_v2_mean']:.6f} rad ± {summary['sam_v2_std']:.6f}")
    print(f"Improvement: {summary['sam_improvement_mean']:.6f} ± {summary['sam_improvement_std']:.6f}")
    
    if summary['sam_improvement_mean'] > 0:
        print("✓ SAM improved (better spectral similarity)")
    else:
        print("⚠️  SAM degraded (worse spectral similarity)")
    
    print("\n👁️  Perceptual Similarity")
    print("-" * 70)
    print(f"LPIPS (v2 vs v1): {summary['lpips_v2_vs_v1_mean']:.6f} ± {summary['lpips_v2_vs_v1_std']:.6f}")
    print(f"Target: < 0.2 (perceptual similarity maintained)")
    
    if summary['lpips_v2_vs_v1_mean'] < 0.2:
        print("✓ LPIPS acceptable (visual similarity maintained)")
    else:
        print("⚠️  LPIPS high (significant visual changes)")
    
    print("\n📏 Edge Displacement")
    print("-" * 70)
    print(f"Max displacement: {summary['edge_displacement_mean']:.2f} px ± {summary['edge_displacement_std']:.2f}")
    print(f"Target: ≤ 2.0 px (spatial structure preserved)")
    
    if summary['edge_displacement_mean'] <= 2.0:
        print("✓ Edge displacement acceptable (< 2px)")
    else:
        print("⚠️  Edge displacement high (> 2px)")
    
    # ========================================================================
    # Generate Plots
    # ========================================================================
    
    print("\n📈 Generating plots...")
    generate_plots(all_metrics, output_dir)
    
    print("\n" + "="*70)
    print("Validation complete!")
    print(f"Results saved to: {output_dir}")
    print("="*70)


# ============================================================================
# Plotting Functions
# ============================================================================

def generate_plots(all_metrics: List[Dict], output_dir: Path):
    """Generate validation plots."""
    
    # ========================================================================
    # Plot 1: NDVI/EVI RMSE Comparison
    # ========================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # NDVI
    ndvi_v1 = [m['ndvi_v1_rmse'] for m in all_metrics]
    ndvi_v2 = [m['ndvi_v2_rmse'] for m in all_metrics]
    
    axes[0].hist(ndvi_v1, bins=30, alpha=0.7, label='opt_v1', color='blue')
    axes[0].hist(ndvi_v2, bins=30, alpha=0.7, label='opt_v2', color='orange')
    axes[0].axvline(np.mean(ndvi_v1), color='blue', linestyle='--', linewidth=2, label=f'v1 mean: {np.mean(ndvi_v1):.4f}')
    axes[0].axvline(np.mean(ndvi_v2), color='orange', linestyle='--', linewidth=2, label=f'v2 mean: {np.mean(ndvi_v2):.4f}')
    axes[0].set_xlabel('NDVI RMSE')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('NDVI RMSE Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # EVI
    evi_v1 = [m['evi_v1_rmse'] for m in all_metrics]
    evi_v2 = [m['evi_v2_rmse'] for m in all_metrics]
    
    axes[1].hist(evi_v1, bins=30, alpha=0.7, label='opt_v1', color='blue')
    axes[1].hist(evi_v2, bins=30, alpha=0.7, label='opt_v2', color='orange')
    axes[1].axvline(np.mean(evi_v1), color='blue', linestyle='--', linewidth=2, label=f'v1 mean: {np.mean(evi_v1):.4f}')
    axes[1].axvline(np.mean(evi_v2), color='orange', linestyle='--', linewidth=2, label=f'v2 mean: {np.mean(evi_v2):.4f}')
    axes[1].set_xlabel('EVI RMSE')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('EVI RMSE Distribution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ndvi_evi_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # Plot 2: SAM Comparison
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sam_v1 = [m['sam_v1'] for m in all_metrics]
    sam_v2 = [m['sam_v2'] for m in all_metrics]
    
    ax.scatter(sam_v1, sam_v2, alpha=0.5, s=20)
    ax.plot([0, max(sam_v1)], [0, max(sam_v1)], 'r--', label='y=x (no change)')
    ax.set_xlabel('SAM opt_v1 (rad)')
    ax.set_ylabel('SAM opt_v2 (rad)')
    ax.set_title('Spectral Angle Mapper: v2 vs v1')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add statistics
    improvement_pct = (1 - np.mean(sam_v2) / np.mean(sam_v1)) * 100
    ax.text(0.05, 0.95, f'Mean improvement: {improvement_pct:.1f}%',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sam_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # Plot 3: Edge Displacement Distribution
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    edge_disp = [m['edge_displacement'] for m in all_metrics]
    
    ax.hist(edge_disp, bins=30, alpha=0.7, color='green')
    ax.axvline(np.mean(edge_disp), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(edge_disp):.2f} px')
    ax.axvline(2.0, color='orange', linestyle=':', linewidth=2, 
               label='Target: 2.0 px')
    ax.set_xlabel('Edge Displacement (pixels)')
    ax.set_ylabel('Frequency')
    ax.set_title('Edge Displacement Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add pass/fail statistics
    passing = sum(1 for d in edge_disp if d <= 2.0)
    pass_rate = passing / len(edge_disp) * 100
    ax.text(0.05, 0.95, f'Pass rate (≤2px): {pass_rate:.1f}%',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if pass_rate > 90 else 'lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'edge_displacement.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # Plot 4: LPIPS Distribution
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    lpips_vals = [m['lpips_v2_vs_v1'] for m in all_metrics]
    
    ax.hist(lpips_vals, bins=30, alpha=0.7, color='purple')
    ax.axvline(np.mean(lpips_vals), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(lpips_vals):.4f}')
    ax.axvline(0.2, color='orange', linestyle=':', linewidth=2,
               label='Target: < 0.2')
    ax.set_xlabel('LPIPS (v2 vs v1)')
    ax.set_ylabel('Frequency')
    ax.set_title('Perceptual Similarity Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lpips_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved 4 plots to {output_dir}")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 Validation: Compare opt_v2 vs opt_v1"
    )
    
    parser.add_argument('--v1_dir', type=str, required=True,
                        help='Directory with opt_v1 tiles')
    parser.add_argument('--v2_dir', type=str, required=True,
                        help='Directory with opt_v2 tiles')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Directory with ground truth tiles (optional)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save validation results')
    parser.add_argument('--max_tiles', type=int, default=None,
                        help='Maximum number of tiles to process')
    
    args = parser.parse_args()
    
    # Convert paths
    v1_dir = Path(args.v1_dir)
    v2_dir = Path(args.v2_dir)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    output_dir = Path(args.output_dir)
    
    # Validate paths
    if not v1_dir.exists():
        print(f"Error: opt_v1 directory not found: {v1_dir}")
        return 1
    
    if not v2_dir.exists():
        print(f"Error: opt_v2 directory not found: {v2_dir}")
        return 1
    
    if gt_dir and not gt_dir.exists():
        print(f"Warning: Ground truth directory not found: {gt_dir}")
        print("Continuing without ground truth (will use v1 as reference)")
        gt_dir = None
    
    # Run validation
    try:
        validate_dataset(
            v1_dir=v1_dir,
            v2_dir=v2_dir,
            gt_dir=gt_dir,
            output_dir=output_dir,
            max_tiles=args.max_tiles
        )
        return 0
    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR during validation:")
        print("="*70)
        print(str(e))
        
        import traceback
        traceback.print_exc()
        
        return 1


if __name__ == '__main__':
    exit(main())
