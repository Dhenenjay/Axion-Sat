"""
Pre-compute Stage 1 Outputs for GAC Pipeline

This script runs TerraMind inference on the entire tile dataset to generate
opt_v1 outputs, which will be used as inputs for Stage 2 (Prithvi) training.

Purpose:
    - Skip Stage 1 training (TerraMind uses non-differentiable sampling)
    - Use pretrained TerraMind for SAR→optical generation
    - Cache all opt_v1 outputs to disk for efficient Stage 2 training

Memory-efficient implementation:
    - Processes tiles in batches
    - Saves outputs incrementally
    - Clears GPU cache between batches

Usage:
    python scripts/00_precompute_stage1.py \\
        --data-dir data/tiles/benv2_catalog \\
        --output-dir data/stage1_outputs \\
        --batch-size 4 \\
        --timesteps 10

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axs_lib.models import build_terramind_generator
from axs_lib.stage1_tm_s2o import tm_sar2opt
from axs_lib.stdz import TERRAMIND_S1_STATS


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-compute Stage 1 (TerraMind) outputs for GAC pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Directory containing tile NPZ files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for opt_v1 files'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=10,
        help='Number of diffusion timesteps (lower = faster)'
    )
    parser.add_argument(
        '--max-tiles',
        type=int,
        default=None,
        help='Maximum number of tiles to process (for testing)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on'
    )
    parser.add_argument(
        '--tile-size',
        type=int,
        default=120,
        help='Expected tile size'
    )
    
    return parser.parse_args()


def find_tiles(data_dir: Path, max_tiles: Optional[int] = None) -> List[Path]:
    """Find all NPZ tile files in data directory."""
    tiles = sorted(data_dir.rglob('*.npz'))
    
    if max_tiles:
        tiles = tiles[:max_tiles]
    
    print(f"Found {len(tiles)} tiles in {data_dir}")
    return tiles


def load_sar_from_tile(tile_path: Path, tile_size: int = 120) -> torch.Tensor:
    """
    Load SAR (S1GRD) data from NPZ tile.
    
    Args:
        tile_path: Path to NPZ file
        tile_size: Expected tile size
        
    Returns:
        SAR tensor (1, 2, H, W) standardized and padded to valid size
    """
    data = np.load(tile_path)
    
    # Extract VV and VH channels
    vv = data.get('s1_vv', np.zeros((tile_size, tile_size), dtype=np.float32))
    vh = data.get('s1_vh', np.zeros((tile_size, tile_size), dtype=np.float32))
    
    # Stack to (2, H, W)
    s1 = np.stack([vv, vh], axis=0)
    
    # Replace NaN with zeros
    s1 = np.nan_to_num(s1, nan=0.0)
    
    # Convert to tensor and add batch dimension
    s1_tensor = torch.from_numpy(s1).unsqueeze(0)  # (1, 2, H, W)
    
    # Pad to nearest valid size (divisible by 16)
    # 120 → 128 (add 4 pixels on each side)
    h, w = s1_tensor.shape[2:]
    target_h = ((h + 15) // 16) * 16  # Round up to nearest multiple of 16
    target_w = ((w + 15) // 16) * 16
    
    if h != target_h or w != target_w:
        pad_h = target_h - h
        pad_w = target_w - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Pad with reflection to avoid edge artifacts
        # Note: reflection_pad2d only works with fp32, so convert temporarily
        original_dtype = s1_tensor.dtype
        if original_dtype != torch.float32:
            s1_tensor = s1_tensor.float()
        
        s1_tensor = torch.nn.functional.pad(
            s1_tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='reflect'
        )
        
        # Convert back to original dtype
        if original_dtype != torch.float32:
            s1_tensor = s1_tensor.to(original_dtype)
    
    # Standardize using TerraMind statistics
    mean = torch.tensor(TERRAMIND_S1_STATS.means, dtype=torch.float32).view(1, 2, 1, 1)
    std = torch.tensor(TERRAMIND_S1_STATS.stds, dtype=torch.float32).view(1, 2, 1, 1)
    s1_tensor = (s1_tensor - mean) / std
    
    return s1_tensor


def save_opt_v1(
    output_path: Path,
    opt_v1: torch.Tensor,
    tile_path: Path,
    original_size: int = 120
):
    """
    Save opt_v1 output to NPZ file.
    
    Args:
        output_path: Output file path
        opt_v1: Optical v1 tensor (1, 4, H, W) - may be padded
        tile_path: Original tile path (for metadata)
        original_size: Original tile size before padding
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Crop back to original size if padded
    h, w = opt_v1.shape[2:]
    if h != original_size or w != original_size:
        # Center crop
        crop_top = (h - original_size) // 2
        crop_left = (w - original_size) // 2
        opt_v1 = opt_v1[:, :, crop_top:crop_top+original_size, crop_left:crop_left+original_size]
    
    # Convert to numpy
    opt_v1_np = opt_v1.squeeze(0).cpu().numpy()  # (4, H, W)
    
    # Save to NPZ
    np.savez_compressed(
        output_path,
        opt_v1=opt_v1_np,
        source_tile=str(tile_path)
    )


def main():
    """Main inference loop."""
    args = parse_args()
    
    print("=" * 80)
    print("Pre-computing Stage 1 (TerraMind) Outputs")
    print("=" * 80)
    print()
    print(f"Data directory:   {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Timesteps:        {args.timesteps}")
    print(f"Device:           {args.device}")
    print()
    
    # Find tiles
    tile_paths = find_tiles(args.data_dir, args.max_tiles)
    
    if len(tile_paths) == 0:
        print("❌ No tiles found!")
        return
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build TerraMind generator
    print("\\nLoading TerraMind generator...")
    device = torch.device(args.device)
    
    generator = build_terramind_generator(
        input_modalities=('S1GRD',),
        output_modalities=('S2L2A',),
        timesteps=args.timesteps,
        standardize=True,
        pretrained=True
    )
    generator = generator.to(device)
    generator.eval()
    
    print(f"✓ TerraMind loaded on {device}")
    print()
    
    # Process tiles in batches
    print("Processing tiles...")
    num_processed = 0
    num_errors = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(tile_paths), args.batch_size)):
            batch_paths = tile_paths[i:i + args.batch_size]
            
            try:
                # Load batch
                s1_batch = []
                for tile_path in batch_paths:
                    s1 = load_sar_from_tile(tile_path, args.tile_size)
                    s1_batch.append(s1)
                
                # Stack into batch
                s1_batch = torch.cat(s1_batch, dim=0).to(device)  # (B, 2, H, W)
                
                # Run Stage 1 inference
                opt_v1_batch = tm_sar2opt(
                    generator,
                    s1_batch,
                    timesteps=args.timesteps,
                    denormalize=False,  # Keep standardized for Stage 2
                    clip_range=None
                )
                
                # Save individual outputs
                for j, tile_path in enumerate(batch_paths):
                    opt_v1 = opt_v1_batch[j:j+1]  # (1, 4, H, W)
                    
                    # Create output path (mirror directory structure)
                    rel_path = tile_path.relative_to(args.data_dir)
                    output_path = args.output_dir / rel_path
                    
                    save_opt_v1(output_path, opt_v1, tile_path, original_size=args.tile_size)
                    num_processed += 1
                
                # Clear GPU cache every 10 batches
                if (i // args.batch_size) % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"\\n⚠️  Error processing batch starting at tile {i}: {e}")
                num_errors += len(batch_paths)
                continue
    
    print()
    print("=" * 80)
    print(f"✓ Completed!")
    print(f"  Processed:  {num_processed}/{len(tile_paths)} tiles")
    print(f"  Errors:     {num_errors}")
    print(f"  Output dir: {args.output_dir}")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Train Stage 2: python scripts/train_stage2.py --stage1-dir", args.output_dir)
    print("  2. Train Stage 3: python scripts/train_stage3.py")
    print()


if __name__ == '__main__':
    main()
