"""
Ultra-Fast Stage 1 Precompute Script
=====================================

Optimizations:
- Large batch sizes (16-32)
- Minimal timesteps (2-4)
- torch.compile() for speed
- No intermediate saves
- Aggressive caching

Target: 200K tiles in ~2 hours (~28 tiles/second)
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
from tqdm import tqdm
import time

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from axs_lib.models import build_terramind_generator


def parse_args():
    parser = argparse.ArgumentParser(description="Ultra-Fast Stage 1 Precompute")
    parser.add_argument('--data-dir', type=str, required=True, help='Input tiles directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--timesteps', type=int, default=3, help='Diffusion timesteps (default: 3)')
    parser.add_argument('--max-tiles', type=int, default=None, help='Max tiles to process')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile()')
    parser.add_argument('--tile-size', type=int, default=120, help='Tile size')
    
    return parser.parse_args()


def load_tiles_fast(tile_paths, tile_size=120):
    """Fast batch tile loading."""
    sar_batch = []
    s2_batch = []
    valid_paths = []
    
    for tile_path in tile_paths:
        try:
            data = np.load(tile_path)
            s1_vv = data.get('s1_vv', np.zeros((tile_size, tile_size), dtype=np.float32))
            s1_vh = data.get('s1_vh', np.zeros((tile_size, tile_size), dtype=np.float32))
            
            # Stack SAR channels
            sar = np.stack([s1_vv, s1_vh], axis=0)
            sar = np.nan_to_num(sar, nan=0.0)
            
            # Pad if needed
            if sar.shape[1] != 128 or sar.shape[2] != 128:
                padded = np.zeros((2, 128, 128), dtype=np.float32)
                h, w = min(sar.shape[1], 128), min(sar.shape[2], 128)
                padded[:, :h, :w] = sar[:, :h, :w]
                sar = padded
            
            # Load S2 ground truth for loss computation
            s2_b2 = data.get('s2_b2', np.zeros((tile_size, tile_size), dtype=np.float32))
            s2_b3 = data.get('s2_b3', np.zeros((tile_size, tile_size), dtype=np.float32))
            s2_b4 = data.get('s2_b4', np.zeros((tile_size, tile_size), dtype=np.float32))
            s2_b8 = data.get('s2_b8', np.zeros((tile_size, tile_size), dtype=np.float32))
            s2 = np.stack([s2_b2, s2_b3, s2_b4, s2_b8], axis=0)
            s2 = np.nan_to_num(s2, nan=0.0)
            
            sar_batch.append(sar)
            s2_batch.append(s2)
            valid_paths.append(tile_path)
            
        except Exception:
            continue
    
    if sar_batch:
        return np.stack(sar_batch), np.stack(s2_batch), valid_paths
    return None, None, []


def main():
    args = parse_args()
    
    print("=" * 80)
    print("ULTRA-FAST Stage 1 Precompute")
    print("=" * 80)
    print(f"Batch size:   {args.batch_size}")
    print(f"Timesteps:    {args.timesteps}")
    print(f"Device:       {args.device}")
    print(f"Compile:      {args.compile}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all tiles
    data_dir = Path(args.data_dir)
    tile_paths = sorted(data_dir.rglob('*.npz'))
    
    if args.max_tiles:
        tile_paths = tile_paths[:args.max_tiles]
    
    print(f"Found {len(tile_paths)} tiles\n")
    
    # Load model
    print("Loading TerraMind (fast mode)...")
    device = torch.device(args.device)
    
    model = build_terramind_generator(
        input_modalities=('S1GRD',),
        output_modalities=('S2L2A',),
        pretrained=True,
        timesteps=args.timesteps
    )
    model = model.to(device)
    model.eval()
    
    # Compile for speed (PyTorch 2.0+)
    if args.compile:
        try:
            print("Compiling model with torch.compile()...")
            model = torch.compile(model, mode='reduce-overhead')
            print("✓ Model compiled")
        except Exception as e:
            print(f"⚠ Compilation failed: {e}")
    
    print("✓ Model loaded\n")
    
    # Process in batches
    total_processed = 0
    total_errors = 0
    total_loss = 0.0
    start_time = time.time()
    
    pbar = tqdm(total=len(tile_paths), desc="Processing", unit="tiles")
    
    for i in range(0, len(tile_paths), args.batch_size):
        batch_paths = tile_paths[i:i + args.batch_size]
        
        # Load batch
        sar_batch, s2_batch, valid_paths = load_tiles_fast(batch_paths, args.tile_size)
        
        if sar_batch is None or len(valid_paths) == 0:
            total_errors += len(batch_paths)
            pbar.update(len(batch_paths))
            continue
        
        # Convert to tensor
        sar_tensor = torch.from_numpy(sar_batch).float().to(device)
        
        # Generate
        try:
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(args.device=='cuda')):
                outputs = model(
                    **{'S1GRD': sar_tensor},
                    num_inference_steps=args.timesteps
                )
            
            opt_v1 = outputs['S2L2A'].cpu().numpy()
            
            # Crop back to original size if needed
            if opt_v1.shape[2] > args.tile_size or opt_v1.shape[3] > args.tile_size:
                opt_v1 = opt_v1[:, :, :args.tile_size, :args.tile_size]
            
            # Extract only the 4 bands we care about (B2, B3, B4, B8)
            # TerraMind outputs 12 bands, we need indices for B2=1, B3=2, B4=3, B8=7
            if opt_v1.shape[1] == 12:
                opt_v1_subset = opt_v1[:, [1, 2, 3, 7], :, :]
            else:
                opt_v1_subset = opt_v1
            
            # Compute loss (MSE vs ground truth S2)
            batch_mse = np.mean((opt_v1_subset - s2_batch) ** 2)
            total_loss += batch_mse * len(valid_paths)
            
            # Save outputs (save the 4-band subset, not all 12 bands)
            for j, tile_path in enumerate(valid_paths):
                output_path = output_dir / tile_path.name
                np.savez_compressed(
                    output_path,
                    opt_v1=opt_v1_subset[j],  # Save 4 bands, not 12
                    source_tile=str(tile_path)
                )
            
            total_processed += len(valid_paths)
            
        except Exception as e:
            print(f"\n⚠ Batch error: {e}")
            total_errors += len(valid_paths)
        
        pbar.update(len(batch_paths))
        
        # Update speed estimate
        elapsed = time.time() - start_time
        if total_processed > 0:
            tiles_per_sec = total_processed / elapsed
            remaining = len(tile_paths) - (i + len(batch_paths))
            eta_hours = (remaining / tiles_per_sec) / 3600 if tiles_per_sec > 0 else 0
            pbar.set_postfix({
                'speed': f'{tiles_per_sec:.1f} tiles/s',
                'ETA': f'{eta_hours:.1f}h'
            })
    
    pbar.close()
    
    # Summary
    elapsed_total = time.time() - start_time
    avg_loss = total_loss / total_processed if total_processed > 0 else 0
    
    print("\n" + "=" * 80)
    print("✓ Completed!")
    print(f"  Processed:    {total_processed}/{len(tile_paths)} tiles")
    print(f"  Errors:       {total_errors}")
    print(f"  Avg MSE Loss: {avg_loss:.6f}")
    print(f"  Time:         {elapsed_total/3600:.2f} hours")
    print(f"  Speed:        {total_processed/elapsed_total:.1f} tiles/second")
    print(f"  Output dir:   {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
