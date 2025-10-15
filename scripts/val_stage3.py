"""
Stage 3 Validation: SAR Agreement and LPIPS Metrics

This script validates Stage 3 outputs by computing:
1. SAR-edge agreement (↑ higher is better)
2. LPIPS change from Stage 2 (≤ threshold is better)

Outputs:
- CSV file with per-tile metrics
- Aggregate statistics JSON
- Example visualization images (best/worst/median cases)

Metrics:
- SAR Edge Agreement: Measures alignment between optical edges and SAR edges
  - Higher values indicate better SAR grounding
  - Range: [0, 1], target: > 0.7
  
- LPIPS Change: Perceptual distance from Stage 2 to Stage 3
  - Lower values indicate minimal perceptual changes
  - Range: [0, ∞], target: < 0.15

Usage:
    # Basic validation
    python scripts/val_stage3.py \\
        --opt-v2-dir stage2_outputs/ \\
        --opt-final-dir stage3_outputs/ \\
        --s1-dir tiles/s1/ \\
        --output-dir validation/stage3/

    # With ground truth for additional metrics
    python scripts/val_stage3.py \\
        --opt-v2-dir stage2_outputs/ \\
        --opt-final-dir stage3_outputs/ \\
        --s1-dir tiles/s1/ \\
        --s2-truth-dir tiles/ground_truth/ \\
        --output-dir validation/stage3/ \\
        --num-examples 10

Author: Axion-Sat Project
Version: 1.0.0
"""

import argparse
import os
import sys
from pathlib import Path
import json
import csv
import time
from typing import Dict, List, Optional, Tuple
import warnings

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Tile Loading
# ============================================================================

def load_tile(tile_path: Path, keys: List[str]) -> Optional[np.ndarray]:
    """
    Load tile from NPZ file, trying multiple possible keys.
    
    Args:
        tile_path: Path to NPZ file
        keys: List of possible keys to try
        
    Returns:
        Array or None if loading fails
    """
    try:
        data = np.load(tile_path)
        
        # Try each key
        for key in keys:
            if key in data:
                arr = data[key]
                if arr.ndim == 2:
                    # Expand to expected channels
                    if 's1' in keys or 'vv' in keys:
                        arr = np.stack([arr, arr], axis=0)  # 2 channels for S1
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
        
        return None
        
    except Exception as e:
        warnings.warn(f"Error loading {tile_path}: {e}")
        return None


def find_matching_tiles(
    opt_v2_dir: Path,
    opt_final_dir: Path,
    s1_dir: Path,
    s2_truth_dir: Optional[Path] = None
) -> List[Dict[str, Path]]:
    """
    Find matching tiles across all directories.
    
    Returns:
        List of dicts with paths for each tile
    """
    opt_v2_tiles = {p.name: p for p in opt_v2_dir.rglob('*.npz')}
    opt_final_tiles = {p.name: p for p in opt_final_dir.rglob('*.npz')}
    s1_tiles = {p.name: p for p in s1_dir.rglob('*.npz')}
    
    # Find common tiles
    common_names = set(opt_v2_tiles.keys()) & set(opt_final_tiles.keys()) & set(s1_tiles.keys())
    
    if s2_truth_dir:
        s2_truth_tiles = {p.name: p for p in s2_truth_dir.rglob('*.npz')}
        common_names = common_names & set(s2_truth_tiles.keys())
    
    # Build tile list
    tiles = []
    for name in sorted(common_names):
        tile_dict = {
            'name': name,
            'opt_v2': opt_v2_tiles[name],
            'opt_final': opt_final_tiles[name],
            's1': s1_tiles[name]
        }
        if s2_truth_dir:
            tile_dict['s2_truth'] = s2_truth_tiles[name]
        tiles.append(tile_dict)
    
    return tiles


# ============================================================================
# Metrics: SAR Edge Agreement
# ============================================================================

def compute_sar_edge_agreement(
    opt: torch.Tensor,
    s1: torch.Tensor,
    sobel_x: Optional[torch.Tensor] = None,
    sobel_y: Optional[torch.Tensor] = None
) -> float:
    """
    Compute SAR-optical edge agreement using Sobel edge detection.
    
    Args:
        opt: Optical image (4, H, W) or (B, 4, H, W)
        s1: SAR image (2, H, W) or (B, 2, H, W)
        sobel_x: Sobel X kernel (optional, will create if None)
        sobel_y: Sobel Y kernel (optional, will create if None)
        
    Returns:
        Edge agreement score [0, 1], higher is better
    """
    # Ensure batch dimension
    if opt.ndim == 3:
        opt = opt.unsqueeze(0)
    if s1.ndim == 3:
        s1 = s1.unsqueeze(0)
    
    device = opt.device
    
    # Create Sobel kernels if not provided
    if sobel_x is None:
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=device).view(1, 1, 3, 3)
    if sobel_y is None:
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    # Convert to grayscale
    opt_gray = (0.2989 * opt[:, 0, :, :] + 
                0.5870 * opt[:, 1, :, :] + 
                0.1140 * opt[:, 2, :, :]).unsqueeze(1)
    s1_gray = s1.mean(dim=1, keepdim=True)
    
    # Compute edges
    opt_edges_x = F.conv2d(opt_gray, sobel_x, padding=1)
    opt_edges_y = F.conv2d(opt_gray, sobel_y, padding=1)
    opt_edges = torch.sqrt(opt_edges_x ** 2 + opt_edges_y ** 2 + 1e-8)
    
    s1_edges_x = F.conv2d(s1_gray, sobel_x, padding=1)
    s1_edges_y = F.conv2d(s1_gray, sobel_y, padding=1)
    s1_edges = torch.sqrt(s1_edges_x ** 2 + s1_edges_y ** 2 + 1e-8)
    
    # Normalize edges to [0, 1]
    opt_edges_norm = opt_edges / (opt_edges.max() + 1e-8)
    s1_edges_norm = s1_edges / (s1_edges.max() + 1e-8)
    
    # Compute cosine similarity (edge alignment)
    correlation = F.cosine_similarity(
        opt_edges_norm.flatten(1),
        s1_edges_norm.flatten(1),
        dim=1
    )
    
    return correlation.mean().item()


# ============================================================================
# Metrics: LPIPS
# ============================================================================

class LPIPSMetric:
    """
    LPIPS (Learned Perceptual Image Patch Similarity) metric.
    
    Falls back to VGG feature distance if lpips package unavailable.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.use_lpips = False
        
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_model.eval()
            self.use_lpips = True
        except ImportError:
            warnings.warn(
                "lpips package not available. Install with: pip install lpips\n"
                "Falling back to VGG feature distance."
            )
            self._init_vgg_fallback()
    
    def _init_vgg_fallback(self):
        """Initialize VGG feature extractor as fallback."""
        from torchvision import models
        
        vgg = models.vgg16(pretrained=True).features.to(self.device)
        vgg.eval()
        
        # Use specific layers
        self.vgg_layers = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
        ]).to(self.device)
        
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
    
    def _extract_vgg_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract VGG features."""
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        
        # Use first 3 channels
        if x.shape[1] == 4:
            x = x[:, :3, :, :]
        
        x_norm = (x - mean) / std
        
        features = []
        for layer in self.vgg_layers:
            x_norm = layer(x_norm)
            features.append(x_norm)
        
        return features
    
    def compute(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Compute LPIPS distance.
        
        Args:
            img1: First image (4, H, W) or (B, 4, H, W), range [0, 1]
            img2: Second image (4, H, W) or (B, 4, H, W), range [0, 1]
            
        Returns:
            LPIPS distance [0, ∞], lower is better
        """
        # Ensure batch dimension
        if img1.ndim == 3:
            img1 = img1.unsqueeze(0)
        if img2.ndim == 3:
            img2 = img2.unsqueeze(0)
        
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        with torch.no_grad():
            if self.use_lpips:
                # LPIPS expects [-1, 1]
                img1_norm = img1 * 2.0 - 1.0
                img2_norm = img2 * 2.0 - 1.0
                
                # Use first 3 channels
                if img1_norm.shape[1] == 4:
                    img1_norm = img1_norm[:, :3, :, :]
                    img2_norm = img2_norm[:, :3, :, :]
                
                dist = self.lpips_model(img1_norm, img2_norm).mean().item()
            else:
                # VGG feature distance
                feat1 = self._extract_vgg_features(img1)
                feat2 = self._extract_vgg_features(img2)
                
                dist = 0.0
                for f1, f2 in zip(feat1, feat2):
                    dist += F.mse_loss(f1, f2).item()
                dist /= len(feat1)
        
        return dist


# ============================================================================
# Additional Metrics
# ============================================================================

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


def compute_ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute simplified SSIM (luminance only)."""
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1 = np.std(img1)
    sigma2 = np.std(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2))
    
    return float(ssim)


# ============================================================================
# Validation
# ============================================================================

def validate_tile(
    tile_dict: Dict[str, Path],
    lpips_metric: LPIPSMetric,
    device: torch.device
) -> Optional[Dict]:
    """
    Validate a single tile.
    
    Returns:
        Dict with metrics or None if validation fails
    """
    # Load tiles
    opt_v2 = load_tile(tile_dict['opt_v2'], ['opt_v2', 's2_b2'])
    opt_final = load_tile(tile_dict['opt_final'], ['opt_final', 'opt_v3'])
    s1 = load_tile(tile_dict['s1'], ['s1', 's1_vv'])
    
    if opt_v2 is None or opt_final is None or s1 is None:
        return None
    
    # Ensure same size
    if opt_v2.shape != opt_final.shape:
        return None
    
    # Convert to tensors
    opt_v2_tensor = torch.from_numpy(opt_v2).float()
    opt_final_tensor = torch.from_numpy(opt_final).float()
    s1_tensor = torch.from_numpy(s1).float()
    
    # Compute metrics
    metrics = {
        'tile_name': tile_dict['name'],
    }
    
    # 1. SAR Edge Agreement (Stage 2 vs Stage 3)
    sar_agreement_v2 = compute_sar_edge_agreement(opt_v2_tensor, s1_tensor)
    sar_agreement_final = compute_sar_edge_agreement(opt_final_tensor, s1_tensor)
    
    metrics['sar_agreement_v2'] = sar_agreement_v2
    metrics['sar_agreement_final'] = sar_agreement_final
    metrics['sar_agreement_improvement'] = sar_agreement_final - sar_agreement_v2
    
    # 2. LPIPS Change (Stage 2 → Stage 3)
    lpips_change = lpips_metric.compute(opt_v2_tensor, opt_final_tensor)
    metrics['lpips_change'] = lpips_change
    
    # 3. Additional metrics
    metrics['mean_change'] = float(np.mean(np.abs(opt_final - opt_v2)))
    metrics['max_change'] = float(np.max(np.abs(opt_final - opt_v2)))
    
    # Per-channel statistics
    channel_names = ['blue', 'green', 'red', 'nir']
    for i, name in enumerate(channel_names):
        metrics[f'{name}_mean'] = float(np.mean(opt_final[i]))
        metrics[f'{name}_std'] = float(np.std(opt_final[i]))
    
    # 4. Metrics vs ground truth (if available)
    if 's2_truth' in tile_dict:
        s2_truth = load_tile(tile_dict['s2_truth'], ['s2', 's2_b2'])
        if s2_truth is not None:
            # PSNR
            metrics['psnr_v2'] = compute_psnr(opt_v2, s2_truth)
            metrics['psnr_final'] = compute_psnr(opt_final, s2_truth)
            
            # Simple SSIM
            metrics['ssim_v2'] = compute_ssim_simple(opt_v2, s2_truth)
            metrics['ssim_final'] = compute_ssim_simple(opt_final, s2_truth)
            
            # LPIPS to ground truth
            s2_truth_tensor = torch.from_numpy(s2_truth).float()
            metrics['lpips_v2_truth'] = lpips_metric.compute(opt_v2_tensor, s2_truth_tensor)
            metrics['lpips_final_truth'] = lpips_metric.compute(opt_final_tensor, s2_truth_tensor)
    
    return metrics


# ============================================================================
# Visualization
# ============================================================================

def create_comparison_figure(
    tile_dict: Dict[str, Path],
    metrics: Dict,
    output_path: Path
):
    """
    Create a comparison figure showing all stages.
    
    Layout:
        Row 1: opt_v2 (Stage 2) | opt_final (Stage 3) | S1 SAR
        Row 2: Difference map   | Edge comparison     | Metrics
    """
    # Load data
    opt_v2 = load_tile(tile_dict['opt_v2'], ['opt_v2', 's2_b2'])
    opt_final = load_tile(tile_dict['opt_final'], ['opt_final', 'opt_v3'])
    s1 = load_tile(tile_dict['s1'], ['s1', 's1_vv'])
    
    if opt_v2 is None or opt_final is None or s1 is None:
        return
    
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Helper function to create RGB composite
    def make_rgb(img, percentile=2):
        rgb = img[[2, 1, 0], :, :]  # R, G, B
        rgb = np.transpose(rgb, (1, 2, 0))
        # Clip to percentiles for better visualization
        p_low = np.percentile(rgb, percentile)
        p_high = np.percentile(rgb, 100 - percentile)
        rgb = np.clip((rgb - p_low) / (p_high - p_low + 1e-8), 0, 1)
        return rgb
    
    # Row 1: RGB composites
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(make_rgb(opt_v2))
    ax1.set_title(f'Stage 2 (opt_v2)\nSAR Agr: {metrics["sar_agreement_v2"]:.3f}', 
                  fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(make_rgb(opt_final))
    ax2.set_title(f'Stage 3 (opt_final)\nSAR Agr: {metrics["sar_agreement_final"]:.3f}', 
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    s1_vis = s1.mean(axis=0)
    s1_vis = (s1_vis - s1_vis.min()) / (s1_vis.max() - s1_vis.min() + 1e-8)
    ax3.imshow(s1_vis, cmap='gray')
    ax3.set_title('SAR (S1)\nMean of VV & VH', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Row 2: Analysis
    # Difference map
    ax4 = fig.add_subplot(gs[1, 0])
    diff = np.mean(np.abs(opt_final - opt_v2), axis=0)
    im4 = ax4.imshow(diff, cmap='hot', vmin=0, vmax=np.percentile(diff, 95))
    ax4.set_title(f'Absolute Difference\nMean: {metrics["mean_change"]:.4f}, Max: {metrics["max_change"]:.4f}', 
                  fontsize=10)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # Edge comparison
    ax5 = fig.add_subplot(gs[1, 1])
    opt_gray = 0.2989 * opt_final[0] + 0.5870 * opt_final[1] + 0.1140 * opt_final[2]
    from scipy import ndimage
    edges = ndimage.sobel(opt_gray)
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
    ax5.imshow(edges, cmap='viridis')
    ax5.set_title('Edges (Stage 3)\nSobel Magnitude', fontsize=10)
    ax5.axis('off')
    
    # Metrics text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    metrics_text = f"""
    Tile: {tile_dict['name']}
    
    SAR Edge Agreement:
      Stage 2: {metrics['sar_agreement_v2']:.4f}
      Stage 3: {metrics['sar_agreement_final']:.4f}
      Improvement: {metrics['sar_agreement_improvement']:.4f}
    
    LPIPS Change (v2→final):
      {metrics['lpips_change']:.4f}
    
    Pixel Changes:
      Mean: {metrics['mean_change']:.4f}
      Max: {metrics['max_change']:.4f}
    
    Channel Means:
      Blue:  {metrics['blue_mean']:.3f}
      Green: {metrics['green_mean']:.3f}
      Red:   {metrics['red_mean']:.3f}
      NIR:   {metrics['nir_mean']:.3f}
    """
    
    if 'psnr_final' in metrics:
        metrics_text += f"""
    vs Ground Truth:
      PSNR: {metrics['psnr_final']:.2f} dB
      SSIM: {metrics['ssim_final']:.4f}
    """
    
    ax6.text(0.05, 0.95, metrics_text.strip(), 
             transform=ax6.transAxes, fontsize=9, verticalalignment='top',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Main title
    agreement_color = 'green' if metrics['sar_agreement_improvement'] > 0 else 'red'
    lpips_color = 'green' if metrics['lpips_change'] < 0.15 else 'orange'
    
    fig.suptitle(
        f"Stage 3 Validation: {tile_dict['name']}\n"
        f"SAR Agreement: {metrics['sar_agreement_final']:.3f} "
        f"({'↑' if metrics['sar_agreement_improvement'] > 0 else '↓'} "
        f"{abs(metrics['sar_agreement_improvement']):.3f}) | "
        f"LPIPS Change: {metrics['lpips_change']:.3f}",
        fontsize=14, fontweight='bold'
    )
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Validation Function
# ============================================================================

def run_validation(
    opt_v2_dir: Path,
    opt_final_dir: Path,
    s1_dir: Path,
    output_dir: Path,
    s2_truth_dir: Optional[Path] = None,
    num_examples: int = 5,
    lpips_threshold: float = 0.15,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict:
    """
    Run full validation on Stage 3 outputs.
    
    Returns:
        Summary dict with statistics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("\n" + "=" * 80)
        print("Stage 3 Validation")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  opt_v2 directory: {opt_v2_dir}")
        print(f"  opt_final directory: {opt_final_dir}")
        print(f"  S1 directory: {s1_dir}")
        print(f"  Ground truth directory: {s2_truth_dir if s2_truth_dir else 'None'}")
        print(f"  Output directory: {output_dir}")
        print(f"  LPIPS threshold: {lpips_threshold}")
        print(f"  Device: {device}")
        print()
    
    # Initialize LPIPS metric
    if verbose:
        print("Initializing LPIPS metric...")
    lpips_metric = LPIPSMetric(device)
    
    # Find matching tiles
    if verbose:
        print("Finding matching tiles...")
    tiles = find_matching_tiles(opt_v2_dir, opt_final_dir, s1_dir, s2_truth_dir)
    
    if len(tiles) == 0:
        raise ValueError("No matching tiles found")
    
    if verbose:
        print(f"✓ Found {len(tiles)} matching tiles\n")
    
    # Validate tiles
    start_time = time.time()
    all_metrics = []
    failed_tiles = []
    
    if verbose:
        pbar = tqdm(tiles, desc="Validating tiles")
    else:
        pbar = tiles
    
    for tile_dict in pbar:
        metrics = validate_tile(tile_dict, lpips_metric, device)
        
        if metrics:
            all_metrics.append(metrics)
        else:
            failed_tiles.append(tile_dict['name'])
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\n✓ Validated {len(all_metrics)} tiles in {total_time:.2f}s")
    
    # Compute aggregate statistics
    if len(all_metrics) == 0:
        raise ValueError("No tiles successfully validated")
    
    aggregate = {
        'total_tiles': len(tiles),
        'validated': len(all_metrics),
        'failed': len(failed_tiles),
        'validation_time_seconds': total_time,
    }
    
    # SAR agreement statistics
    sar_agr_v2 = [m['sar_agreement_v2'] for m in all_metrics]
    sar_agr_final = [m['sar_agreement_final'] for m in all_metrics]
    sar_agr_improvement = [m['sar_agreement_improvement'] for m in all_metrics]
    
    aggregate['sar_agreement'] = {
        'v2_mean': float(np.mean(sar_agr_v2)),
        'v2_std': float(np.std(sar_agr_v2)),
        'final_mean': float(np.mean(sar_agr_final)),
        'final_std': float(np.std(sar_agr_final)),
        'improvement_mean': float(np.mean(sar_agr_improvement)),
        'improvement_std': float(np.std(sar_agr_improvement)),
        'tiles_improved': int(sum(1 for x in sar_agr_improvement if x > 0)),
        'tiles_degraded': int(sum(1 for x in sar_agr_improvement if x < 0)),
    }
    
    # LPIPS change statistics
    lpips_changes = [m['lpips_change'] for m in all_metrics]
    
    aggregate['lpips_change'] = {
        'mean': float(np.mean(lpips_changes)),
        'std': float(np.std(lpips_changes)),
        'median': float(np.median(lpips_changes)),
        'min': float(np.min(lpips_changes)),
        'max': float(np.max(lpips_changes)),
        'threshold': lpips_threshold,
        'tiles_below_threshold': int(sum(1 for x in lpips_changes if x <= lpips_threshold)),
        'percent_below_threshold': float(sum(1 for x in lpips_changes if x <= lpips_threshold) / len(lpips_changes) * 100),
    }
    
    # Ground truth metrics (if available)
    if 'psnr_final' in all_metrics[0]:
        psnr_finals = [m['psnr_final'] for m in all_metrics]
        ssim_finals = [m['ssim_final'] for m in all_metrics]
        
        aggregate['ground_truth_metrics'] = {
            'psnr_mean': float(np.mean(psnr_finals)),
            'psnr_std': float(np.std(psnr_finals)),
            'ssim_mean': float(np.mean(ssim_finals)),
            'ssim_std': float(np.std(ssim_finals)),
        }
    
    # Save CSV
    csv_path = output_dir / 'validation_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        if len(all_metrics) > 0:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
    
    if verbose:
        print(f"✓ Saved metrics CSV to {csv_path}")
    
    # Save JSON summary
    json_path = output_dir / 'validation_summary.json'
    with open(json_path, 'w') as f:
        json.dump(aggregate, f, indent=2)
    
    if verbose:
        print(f"✓ Saved summary JSON to {json_path}")
    
    # Create example visualizations
    if num_examples > 0 and len(all_metrics) > 0:
        if verbose:
            print(f"\nCreating {num_examples} example visualizations...")
        
        examples_dir = output_dir / 'examples'
        examples_dir.mkdir(exist_ok=True)
        
        # Sort by SAR agreement improvement
        sorted_by_improvement = sorted(
            zip(all_metrics, tiles),
            key=lambda x: x[0]['sar_agreement_improvement'],
            reverse=True
        )
        
        # Best improvements
        n_best = min(num_examples // 2 + 1, len(sorted_by_improvement))
        for i, (metrics, tile_dict) in enumerate(sorted_by_improvement[:n_best]):
            output_path = examples_dir / f'best_{i+1}_{tile_dict["name"].replace(".npz", ".png")}'
            create_comparison_figure(tile_dict, metrics, output_path)
        
        # Worst improvements
        n_worst = min(num_examples // 2, len(sorted_by_improvement))
        for i, (metrics, tile_dict) in enumerate(sorted_by_improvement[-n_worst:]):
            output_path = examples_dir / f'worst_{i+1}_{tile_dict["name"].replace(".npz", ".png")}'
            create_comparison_figure(tile_dict, metrics, output_path)
        
        # Median
        median_idx = len(sorted_by_improvement) // 2
        metrics, tile_dict = sorted_by_improvement[median_idx]
        output_path = examples_dir / f'median_{tile_dict["name"].replace(".npz", ".png")}'
        create_comparison_figure(tile_dict, metrics, output_path)
        
        if verbose:
            print(f"✓ Saved {num_examples} example visualizations to {examples_dir}")
    
    # Print summary
    if verbose:
        print("\n" + "=" * 80)
        print("Validation Complete!")
        print("=" * 80)
        print(f"\nResults:")
        print(f"  Total tiles: {aggregate['total_tiles']}")
        print(f"  Validated: {aggregate['validated']}")
        print(f"  Failed: {aggregate['failed']}")
        print(f"  Validation time: {aggregate['validation_time_seconds']:.2f}s")
        
        print(f"\nSAR Edge Agreement:")
        print(f"  Stage 2 mean: {aggregate['sar_agreement']['v2_mean']:.4f} ± {aggregate['sar_agreement']['v2_std']:.4f}")
        print(f"  Stage 3 mean: {aggregate['sar_agreement']['final_mean']:.4f} ± {aggregate['sar_agreement']['final_std']:.4f}")
        print(f"  Improvement: {aggregate['sar_agreement']['improvement_mean']:.4f} ± {aggregate['sar_agreement']['improvement_std']:.4f}")
        print(f"  Tiles improved: {aggregate['sar_agreement']['tiles_improved']} ({aggregate['sar_agreement']['tiles_improved']/aggregate['validated']*100:.1f}%)")
        print(f"  Tiles degraded: {aggregate['sar_agreement']['tiles_degraded']} ({aggregate['sar_agreement']['tiles_degraded']/aggregate['validated']*100:.1f}%)")
        
        print(f"\nLPIPS Change (Stage 2 → Stage 3):")
        print(f"  Mean: {aggregate['lpips_change']['mean']:.4f} ± {aggregate['lpips_change']['std']:.4f}")
        print(f"  Median: {aggregate['lpips_change']['median']:.4f}")
        print(f"  Range: [{aggregate['lpips_change']['min']:.4f}, {aggregate['lpips_change']['max']:.4f}]")
        print(f"  Below threshold ({lpips_threshold}): {aggregate['lpips_change']['tiles_below_threshold']} ({aggregate['lpips_change']['percent_below_threshold']:.1f}%)")
        
        if 'ground_truth_metrics' in aggregate:
            print(f"\nGround Truth Metrics:")
            print(f"  PSNR: {aggregate['ground_truth_metrics']['psnr_mean']:.2f} ± {aggregate['ground_truth_metrics']['psnr_std']:.2f} dB")
            print(f"  SSIM: {aggregate['ground_truth_metrics']['ssim_mean']:.4f} ± {aggregate['ground_truth_metrics']['ssim_std']:.4f}")
        
        print(f"\nOutput:")
        print(f"  CSV: {csv_path}")
        print(f"  Summary: {json_path}")
        if num_examples > 0:
            print(f"  Examples: {examples_dir}")
        
        print("=" * 80)
    
    return aggregate


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 Validation: Compute SAR agreement and LPIPS metrics"
    )
    
    # Required arguments
    parser.add_argument('--opt-v2-dir', type=str, required=True,
                       help='Directory containing Stage 2 outputs (opt_v2)')
    parser.add_argument('--opt-final-dir', type=str, required=True,
                       help='Directory containing Stage 3 outputs (opt_final)')
    parser.add_argument('--s1-dir', type=str, required=True,
                       help='Directory containing S1 SAR tiles')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for validation results')
    
    # Optional arguments
    parser.add_argument('--s2-truth-dir', type=str, default=None,
                       help='Directory containing ground truth S2 tiles (optional)')
    parser.add_argument('--num-examples', type=int, default=5,
                       help='Number of example visualizations (default: 5)')
    parser.add_argument('--lpips-threshold', type=float, default=0.15,
                       help='LPIPS change threshold (default: 0.15)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device: cuda, cpu (default: auto)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Parse device
    device = None
    if args.device:
        device = torch.device(args.device)
    
    # Run validation
    try:
        summary = run_validation(
            opt_v2_dir=Path(args.opt_v2_dir),
            opt_final_dir=Path(args.opt_final_dir),
            s1_dir=Path(args.s1_dir),
            output_dir=Path(args.output_dir),
            s2_truth_dir=Path(args.s2_truth_dir) if args.s2_truth_dir else None,
            num_examples=args.num_examples,
            lpips_threshold=args.lpips_threshold,
            device=device,
            verbose=not args.quiet
        )
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error during validation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
