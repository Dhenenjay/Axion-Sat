"""
Stage 3 Inference: Generate Final SAR-Grounded Optical Outputs

This script runs Stage 3 inference to produce final optical outputs (opt_final)
by grounding Stage 2 outputs (opt_v2) with SAR features (S1).

Input:
    - opt_v2: Stage 2 refined optical tiles (4 channels: B, G, R, NIR)
    - S1: Sentinel-1 SAR tiles (2 channels: VV, VH)

Output:
    - opt_final (opt_v3): Final SAR-grounded optical tiles (4 channels)

Features:
    - Batch processing of tile folders
    - Automatic tile matching (opt_v2 ↔ S1)
    - Progress tracking with tqdm
    - Memory-efficient processing (batch size control)
    - Output validation and statistics
    - Optional visualization

Usage:
    # Basic inference
    python scripts/infer_stage3.py \\
        --checkpoint checkpoints/stage3/best_model.pt \\
        --opt-v2-dir stage2_outputs/ \\
        --s1-dir tiles/s1/ \\
        --output-dir outputs/stage3/

    # With custom settings
    python scripts/infer_stage3.py \\
        --checkpoint checkpoints/stage3/best_model.pt \\
        --opt-v2-dir stage2_outputs/ \\
        --s1-dir tiles/s1/ \\
        --output-dir outputs/stage3/ \\
        --batch-size 4 \\
        --timesteps 10 \\
        --save-intermediate

Author: Axion-Sat Project
Version: 1.0.0
"""

import argparse
import os
import sys
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Tuple
import warnings

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axs_lib.stage3_tm_ground import build_stage3_model, stage3_inference
from axs_lib.io import load_checkpoint
from axs_lib.path_utils import safe_path


# ============================================================================
# Tile Matching and Loading
# ============================================================================

def find_tile_pairs(
    opt_v2_dir: Path,
    s1_dir: Path,
    pattern: str = '*.npz'
) -> List[Tuple[Path, Path]]:
    """
    Find matching pairs of opt_v2 and S1 tiles.
    
    Matches tiles by filename (e.g., tile_001.npz in both directories).
    
    Args:
        opt_v2_dir: Directory containing opt_v2 tiles
        s1_dir: Directory containing S1 tiles
        pattern: File pattern to match
        
    Returns:
        List of (opt_v2_path, s1_path) tuples
    """
    opt_v2_dir = Path(opt_v2_dir)
    s1_dir = Path(s1_dir)
    
    # Find all opt_v2 tiles
    opt_v2_tiles = {p.name: p for p in opt_v2_dir.rglob(pattern)}
    
    # Find matching S1 tiles
    pairs = []
    for s1_tile in s1_dir.rglob(pattern):
        tile_name = s1_tile.name
        if tile_name in opt_v2_tiles:
            pairs.append((opt_v2_tiles[tile_name], s1_tile))
    
    return sorted(pairs)


def load_opt_v2_tile(tile_path: Path) -> np.ndarray:
    """
    Load opt_v2 tile from NPZ file.
    
    Args:
        tile_path: Path to NPZ file
        
    Returns:
        opt_v2 array (4, H, W) or None if loading fails
    """
    try:
        data = np.load(tile_path)
        
        # Try to load 'opt_v2' key directly
        if 'opt_v2' in data:
            opt_v2 = data['opt_v2']
            if opt_v2.ndim == 2:
                # Single channel, need to stack 4 times
                opt_v2 = np.stack([opt_v2] * 4, axis=0)
            return opt_v2.astype(np.float32)
        
        # Try to load individual bands
        if all(f's2_b{b}' in data for b in [2, 3, 4, 8]):
            opt_v2 = np.stack([
                data['s2_b2'],
                data['s2_b3'],
                data['s2_b4'],
                data['s2_b8']
            ], axis=0)
            return opt_v2.astype(np.float32)
        
        # Try alternative naming
        if all(f'opt_v2_b{b}' in data for b in [2, 3, 4, 8]):
            opt_v2 = np.stack([
                data['opt_v2_b2'],
                data['opt_v2_b3'],
                data['opt_v2_b4'],
                data['opt_v2_b8']
            ], axis=0)
            return opt_v2.astype(np.float32)
        
        # Last resort: use first 4 arrays found
        arrays = [v for k, v in data.items() if isinstance(v, np.ndarray) and v.ndim == 2]
        if len(arrays) >= 4:
            opt_v2 = np.stack(arrays[:4], axis=0)
            return opt_v2.astype(np.float32)
        
        warnings.warn(f"Could not find opt_v2 data in {tile_path}")
        return None
        
    except Exception as e:
        warnings.warn(f"Error loading opt_v2 from {tile_path}: {e}")
        return None


def load_s1_tile(tile_path: Path) -> np.ndarray:
    """
    Load S1 SAR tile from NPZ file.
    
    Args:
        tile_path: Path to NPZ file
        
    Returns:
        S1 array (2, H, W) or None if loading fails
    """
    try:
        data = np.load(tile_path)
        
        # Try to load 's1' key directly
        if 's1' in data:
            s1 = data['s1']
            if s1.ndim == 2:
                # Single channel, duplicate to 2 channels
                s1 = np.stack([s1, s1], axis=0)
            return s1.astype(np.float32)
        
        # Try to load VV and VH
        if 's1_vv' in data and 's1_vh' in data:
            s1 = np.stack([
                data['s1_vv'],
                data['s1_vh']
            ], axis=0)
            return s1.astype(np.float32)
        
        # Try alternative naming
        if 'vv' in data and 'vh' in data:
            s1 = np.stack([
                data['vv'],
                data['vh']
            ], axis=0)
            return s1.astype(np.float32)
        
        warnings.warn(f"Could not find S1 data in {tile_path}")
        return None
        
    except Exception as e:
        warnings.warn(f"Error loading S1 from {tile_path}: {e}")
        return None


# ============================================================================
# Batch Processing
# ============================================================================

def process_batch(
    model: torch.nn.Module,
    opt_v2_batch: torch.Tensor,
    s1_batch: torch.Tensor,
    device: torch.device,
    timesteps: Optional[int] = None,
    min_timesteps: int = 2
) -> Tuple[torch.Tensor, int]:
    """
    Process a batch of tiles through Stage 3 model with OOM auto-retry.
    
    If an OOM error occurs, automatically retries with reduced timesteps
    (decremented by 2 each attempt) until successful or min_timesteps is reached.
    
    Args:
        model: Stage 3 model
        opt_v2_batch: Batch of opt_v2 tiles (B, 4, H, W)
        s1_batch: Batch of S1 tiles (B, 2, H, W)
        device: Device to run on
        timesteps: Number of diffusion timesteps
        min_timesteps: Minimum timesteps to try (default: 2)
        
    Returns:
        Tuple of (opt_final batch (B, 4, H, W), actual_timesteps_used)
    """
    model.eval()
    
    current_timesteps = timesteps
    
    while True:
        try:
            with torch.no_grad():
                # Move to device
                opt_v2_batch = opt_v2_batch.to(device)
                s1_batch = s1_batch.to(device)
                
                # Run inference
                opt_final = model(s1_batch, opt_v2_batch, timesteps=current_timesteps)
                
                # Move back to CPU
                opt_final = opt_final.cpu()
            
            # Success - clear cache and return
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return opt_final, current_timesteps
            
        except RuntimeError as e:
            # Check if it's an OOM error
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Reduce timesteps
                if current_timesteps is not None and current_timesteps > min_timesteps:
                    old_timesteps = current_timesteps
                    current_timesteps = max(current_timesteps - 2, min_timesteps)
                    warnings.warn(
                        f"\n⚠ OOM detected! Reducing timesteps: {old_timesteps} → {current_timesteps}"
                    )
                    print(f"\n⚠ OOM AUTO-RETRY: Timesteps reduced from {old_timesteps} to {current_timesteps}")
                    # Retry with reduced timesteps
                    continue
                else:
                    # Can't reduce further
                    raise RuntimeError(
                        f"OOM error occurred even with minimum timesteps ({current_timesteps}). "
                        f"Try reducing batch size or using a smaller model."
                    ) from e
            else:
                # Not an OOM error, re-raise
                raise


# ============================================================================
# Output Saving
# ============================================================================

def save_tile_output(
    output_path: Path,
    opt_final: np.ndarray,
    metadata: Optional[Dict] = None,
    save_intermediate: bool = False,
    opt_v2: Optional[np.ndarray] = None,
    s1: Optional[np.ndarray] = None,
    quality_status: str = "production"
):
    """
    Save Stage 3 output to NPZ file with quality status tagging.
    
    Args:
        output_path: Output file path
        opt_final: Final optical output (4, H, W)
        metadata: Optional metadata dict
        save_intermediate: Whether to save intermediate inputs
        opt_v2: Optional Stage 2 output (4, H, W)
        s1: Optional S1 input (2, H, W)
        quality_status: Quality status tag ("production" or "preview-only")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data dict
    data = {
        'opt_final': opt_final.astype(np.float32),
        'opt_v3': opt_final.astype(np.float32),  # Alias
    }
    
    # Add intermediate inputs if requested
    if save_intermediate:
        if opt_v2 is not None:
            data['opt_v2'] = opt_v2.astype(np.float32)
        if s1 is not None:
            data['s1'] = s1.astype(np.float32)
    
    # Save NPZ
    np.savez_compressed(output_path, **data)
    
    # Save metadata as JSON if provided (always save if quality is preview-only)
    if metadata or quality_status == "preview-only":
        json_path = output_path.with_suffix('.json')
        
        if metadata is None:
            metadata = {}
        
        # Add quality status
        metadata['quality_status'] = quality_status
        metadata['ars_s_v1_certified'] = (quality_status == "production")
        
        # Add warning if preview-only
        if quality_status == "preview-only":
            metadata['warning'] = (
                "This output has SAR agreement < 0.6 and is PREVIEW-ONLY. "
                "It does not meet ARS-S v1 certification requirements for production use. "
                "Physics grounding may be insufficient. Use with caution."
            )
        
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)


# ============================================================================
# Statistics and Validation
# ============================================================================

def compute_sar_agreement_numpy(
    opt_final: np.ndarray,
    s1: np.ndarray
) -> float:
    """
    Compute SAR edge agreement for numpy arrays.
    
    This is the primary physics grounding metric. Higher values indicate
    better alignment between optical and SAR-derived physical boundaries.
    
    Args:
        opt_final: Optical output (4, H, W)
        s1: SAR input (2, H, W)
        
    Returns:
        Agreement score [0, 1], higher is better
    """
    from scipy import ndimage
    
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


def compute_tile_statistics(
    opt_final: np.ndarray,
    opt_v2: np.ndarray,
    s1: np.ndarray
) -> Dict[str, float]:
    """
    Compute statistics for a processed tile.
    
    Args:
        opt_final: Final output (4, H, W)
        opt_v2: Stage 2 output (4, H, W)
        s1: S1 input (2, H, W)
        
    Returns:
        Dict with statistics
    """
    stats = {}
    
    # Output statistics
    stats['opt_final_mean'] = float(np.mean(opt_final))
    stats['opt_final_std'] = float(np.std(opt_final))
    stats['opt_final_min'] = float(np.min(opt_final))
    stats['opt_final_max'] = float(np.max(opt_final))
    
    # Change from Stage 2
    diff = np.abs(opt_final - opt_v2)
    stats['change_mean'] = float(np.mean(diff))
    stats['change_max'] = float(np.max(diff))
    stats['change_std'] = float(np.std(diff))
    
    # SAR edge agreement (physics grounding metric)
    sar_agreement = compute_sar_agreement_numpy(opt_final, s1)
    stats['sar_agreement'] = sar_agreement
    
    # Per-channel statistics
    channel_names = ['blue', 'green', 'red', 'nir']
    for i, name in enumerate(channel_names):
        stats[f'{name}_mean'] = float(np.mean(opt_final[i]))
        stats[f'{name}_std'] = float(np.std(opt_final[i]))
    
    return stats


def validate_output(opt_final: np.ndarray) -> Tuple[bool, str]:
    """
    Validate output tile.
    
    Args:
        opt_final: Final output (4, H, W)
        
    Returns:
        (is_valid, message)
    """
    # Check for NaN
    if np.isnan(opt_final).any():
        return False, "Contains NaN values"
    
    # Check for Inf
    if np.isinf(opt_final).any():
        return False, "Contains Inf values"
    
    # Check range (should be normalized, typically [0, 1] or similar)
    if np.min(opt_final) < -10 or np.max(opt_final) > 10:
        return False, f"Values out of expected range: [{np.min(opt_final):.3f}, {np.max(opt_final):.3f}]"
    
    # Check shape
    if opt_final.ndim != 3 or opt_final.shape[0] != 4:
        return False, f"Invalid shape: {opt_final.shape}, expected (4, H, W)"
    
    return True, "Valid"


# ============================================================================
# Main Inference Function
# ============================================================================

def run_stage3_inference(
    checkpoint_path: Path,
    opt_v2_dir: Path,
    s1_dir: Path,
    output_dir: Path,
    batch_size: int = 1,
    timesteps: Optional[int] = None,
    device: Optional[torch.device] = None,
    save_intermediate: bool = False,
    save_stats: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run Stage 3 inference on a directory of tiles.
    
    Args:
        checkpoint_path: Path to trained Stage 3 checkpoint
        opt_v2_dir: Directory containing opt_v2 tiles
        s1_dir: Directory containing S1 tiles
        output_dir: Output directory for opt_final tiles
        batch_size: Batch size for processing
        timesteps: Number of diffusion timesteps (None = use default)
        device: Device to run on (None = auto-detect)
        save_intermediate: Save opt_v2 and s1 in output
        save_stats: Save statistics JSON
        verbose: Print progress
        
    Returns:
        Summary dict with statistics
    """
    # Setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("\n" + "=" * 80)
        print("Stage 3 Inference: SAR Grounding")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  opt_v2 directory: {opt_v2_dir}")
        print(f"  S1 directory: {s1_dir}")
        print(f"  Output directory: {output_dir}")
        print(f"  Batch size: {batch_size}")
        print(f"  Timesteps: {timesteps if timesteps else 'default'}")
        print(f"  Device: {device}")
        print()
    
    # Load model
    if verbose:
        print("Loading Stage 3 model...")
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Extract timesteps from checkpoint config if available
    if timesteps is None and 'config' in checkpoint:
        timesteps = checkpoint['config'].get('timesteps', 10)
    
    # Build model
    model = build_stage3_model(
        timesteps=timesteps or 10,
        standardize=True,
        pretrained=False,  # Will load from checkpoint
        device=device
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if verbose:
        print(f"✓ Model loaded from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Timesteps: {timesteps or 10}")
    
    # Find tile pairs
    if verbose:
        print("\nFinding tile pairs...")
    
    tile_pairs = find_tile_pairs(opt_v2_dir, s1_dir)
    
    if len(tile_pairs) == 0:
        raise ValueError(f"No matching tile pairs found between {opt_v2_dir} and {s1_dir}")
    
    if verbose:
        print(f"✓ Found {len(tile_pairs)} matching tile pairs")
    
    # Process tiles
    start_time = time.time()
    
    processed_tiles = []
    failed_tiles = []
    preview_only_tiles = []
    all_stats = []
    
    # Create progress bar
    if verbose:
        pbar = tqdm(total=len(tile_pairs), desc="Processing tiles")
    
    # Process in batches
    for i in range(0, len(tile_pairs), batch_size):
        batch_pairs = tile_pairs[i:i + batch_size]
        
        # Load batch
        opt_v2_list = []
        s1_list = []
        valid_pairs = []
        
        for opt_v2_path, s1_path in batch_pairs:
            opt_v2 = load_opt_v2_tile(opt_v2_path)
            s1 = load_s1_tile(s1_path)
            
            if opt_v2 is not None and s1 is not None:
                # Ensure same spatial size
                if opt_v2.shape[1:] != s1.shape[1:]:
                    # Resize S1 to match opt_v2
                    s1_tensor = torch.from_numpy(s1).unsqueeze(0)
                    s1_tensor = F.interpolate(
                        s1_tensor,
                        size=opt_v2.shape[1:],
                        mode='bilinear',
                        align_corners=False
                    )
                    s1 = s1_tensor.squeeze(0).numpy()
                
                opt_v2_list.append(opt_v2)
                s1_list.append(s1)
                valid_pairs.append((opt_v2_path, s1_path))
            else:
                failed_tiles.append((opt_v2_path, s1_path))
                if verbose:
                    pbar.update(1)
        
        if len(opt_v2_list) == 0:
            continue
        
        # Create batch tensors
        opt_v2_batch = torch.from_numpy(np.stack(opt_v2_list, axis=0))
        s1_batch = torch.from_numpy(np.stack(s1_list, axis=0))
        
        # Process batch (with OOM auto-retry)
        opt_final_batch, actual_timesteps = process_batch(
            model=model,
            opt_v2_batch=opt_v2_batch,
            s1_batch=s1_batch,
            device=device,
            timesteps=timesteps
        )
        
        # Log if timesteps were reduced
        if actual_timesteps != timesteps:
            warnings.warn(
                f"Batch processed with reduced timesteps: {actual_timesteps} (requested: {timesteps})"
            )
        
        # Save results
        for j, (opt_v2_path, s1_path) in enumerate(valid_pairs):
            opt_final = opt_final_batch[j].numpy()
            opt_v2 = opt_v2_list[j]
            s1 = s1_list[j]
            
            # Validate output
            is_valid, message = validate_output(opt_final)
            
            if not is_valid:
                warnings.warn(f"Invalid output for {opt_v2_path.name}: {message}")
                failed_tiles.append((opt_v2_path, s1_path))
                if verbose:
                    pbar.update(1)
                continue
            
            # Compute statistics (includes SAR agreement)
            stats = compute_tile_statistics(opt_final, opt_v2, s1)
            stats['tile_name'] = opt_v2_path.name
            stats['opt_v2_path'] = str(opt_v2_path)
            stats['s1_path'] = str(s1_path)
            
            # Sanity check: SAR agreement must be >= 0.6 for production
            sar_agreement = stats['sar_agreement']
            SAR_AGREEMENT_THRESHOLD = 0.6
            
            if sar_agreement < SAR_AGREEMENT_THRESHOLD:
                # Mark as preview-only
                quality_status = "preview-only"
                stats['quality_status'] = quality_status
                stats['ars_s_v1_certified'] = False
                preview_only_tiles.append(opt_v2_path.name)
                
                if verbose:
                    warnings.warn(
                        f"\n⚠ {opt_v2_path.name}: SAR agreement {sar_agreement:.4f} < {SAR_AGREEMENT_THRESHOLD:.2f} "
                        f"- tagged as PREVIEW-ONLY (not ARS-S v1 certified)"
                    )
            else:
                # Meets production quality
                quality_status = "production"
                stats['quality_status'] = quality_status
                stats['ars_s_v1_certified'] = True
            
            all_stats.append(stats)
            
            # Save output with quality status
            output_path = output_dir / opt_v2_path.name
            save_tile_output(
                output_path=output_path,
                opt_final=opt_final,
                metadata=stats if save_stats else None,
                save_intermediate=save_intermediate,
                opt_v2=opt_v2 if save_intermediate else None,
                s1=s1 if save_intermediate else None,
                quality_status=quality_status
            )
            
            processed_tiles.append(output_path)
            
            if verbose:
                pbar.update(1)
    
    if verbose:
        pbar.close()
    
    # Compute summary statistics
    total_time = time.time() - start_time
    
    # Count production vs preview-only tiles
    production_tiles = len(processed_tiles) - len(preview_only_tiles)
    
    summary = {
        'total_tiles': len(tile_pairs),
        'processed': len(processed_tiles),
        'production_certified': production_tiles,
        'preview_only': len(preview_only_tiles),
        'failed': len(failed_tiles),
        'success_rate': len(processed_tiles) / len(tile_pairs) if len(tile_pairs) > 0 else 0,
        'certification_rate': production_tiles / len(processed_tiles) if len(processed_tiles) > 0 else 0,
        'processing_time_seconds': total_time,
        'tiles_per_second': len(processed_tiles) / total_time if total_time > 0 else 0,
        'checkpoint': str(checkpoint_path),
        'output_dir': str(output_dir),
        'batch_size': batch_size,
        'timesteps': timesteps or 10,
        'sar_agreement_threshold': 0.6
    }
    
    # Aggregate tile statistics
    if len(all_stats) > 0:
        summary['aggregate_stats'] = {
            'opt_final_mean': float(np.mean([s['opt_final_mean'] for s in all_stats])),
            'opt_final_std': float(np.mean([s['opt_final_std'] for s in all_stats])),
            'change_mean': float(np.mean([s['change_mean'] for s in all_stats])),
            'change_max': float(np.max([s['change_max'] for s in all_stats])),
        }
    
    # Save summary
    summary_path = output_dir / 'inference_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed stats
    if save_stats and len(all_stats) > 0:
        stats_path = output_dir / 'tile_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
    
    # Print summary
    if verbose:
        print("\n" + "=" * 80)
        print("Inference Complete!")
        print("=" * 80)
        print(f"\nResults:")
        print(f"  Total tiles: {summary['total_tiles']}")
        print(f"  Processed: {summary['processed']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Success rate: {summary['success_rate']*100:.1f}%")
        print(f"\nQuality Status:")
        print(f"  Production (ARS-S v1 certified): {summary['production_certified']}")
        print(f"  Preview-only (SAR < 0.6):        {summary['preview_only']}")
        print(f"  Certification rate:              {summary['certification_rate']*100:.1f}%")
        print(f"\nPerformance:")
        print(f"  Processing time: {summary['processing_time_seconds']:.2f}s")
        print(f"  Throughput: {summary['tiles_per_second']:.2f} tiles/s")
        
        if 'aggregate_stats' in summary:
            print(f"\nAggregate Statistics:")
            print(f"  Mean output value: {summary['aggregate_stats']['opt_final_mean']:.4f}")
            print(f"  Mean output std: {summary['aggregate_stats']['opt_final_std']:.4f}")
            print(f"  Mean change from Stage 2: {summary['aggregate_stats']['change_mean']:.4f}")
            print(f"  Max change from Stage 2: {summary['aggregate_stats']['change_max']:.4f}")
        
        print(f"\nOutput saved to: {output_dir}")
        print(f"  Summary: {summary_path}")
        if save_stats:
            print(f"  Statistics: {stats_path}")
        
        if len(preview_only_tiles) > 0:
            print(f"\n⚠ Warning: {len(preview_only_tiles)} tiles tagged as PREVIEW-ONLY (SAR agreement < 0.6):")
            for tile_name in preview_only_tiles[:5]:
                print(f"  - {tile_name}")
            if len(preview_only_tiles) > 5:
                print(f"  ... and {len(preview_only_tiles) - 5} more")
            print(f"\n  These tiles do NOT meet ARS-S v1 certification requirements.")
            print(f"  See individual .json files for details and warnings.")
        
        if len(failed_tiles) > 0:
            print(f"\n⚠ Error: {len(failed_tiles)} tiles completely failed:")
            for opt_v2_path, s1_path in failed_tiles[:5]:
                print(f"  - {opt_v2_path.name}")
            if len(failed_tiles) > 5:
                print(f"  ... and {len(failed_tiles) - 5} more")
        
        print("=" * 80)
    
    return summary


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 Inference: Generate SAR-grounded optical outputs"
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained Stage 3 checkpoint')
    parser.add_argument('--opt-v2-dir', type=str, required=True,
                       help='Directory containing opt_v2 tiles (Stage 2 outputs)')
    parser.add_argument('--s1-dir', type=str, required=True,
                       help='Directory containing S1 SAR tiles')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for opt_final tiles')
    
    # Optional arguments
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for processing (default: 1)')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Number of diffusion timesteps (default: from checkpoint)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run on: cuda, cpu (default: auto)')
    
    # Output options
    parser.add_argument('--save-intermediate', action='store_true',
                       help='Save opt_v2 and s1 in output files')
    parser.add_argument('--no-stats', action='store_true',
                       help='Do not save statistics')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Parse device
    device = None
    if args.device:
        device = torch.device(args.device)
    
    # Run inference
    try:
        summary = run_stage3_inference(
            checkpoint_path=Path(args.checkpoint),
            opt_v2_dir=Path(args.opt_v2_dir),
            s1_dir=Path(args.s1_dir),
            output_dir=Path(args.output_dir),
            batch_size=args.batch_size,
            timesteps=args.timesteps,
            device=device,
            save_intermediate=args.save_intermediate,
            save_stats=not args.no_stats,
            verbose=not args.quiet
        )
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error during inference: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
