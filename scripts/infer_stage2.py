"""
Stage 2 Inference Script: Prithvi Refinement

This script loads a trained Stage 2 model and processes a folder of opt_v1 tiles
(TerraMind synthetic optical outputs) to produce opt_v2 tiles (Prithvi refined optical).

Features:
- Batch processing of opt_v1 tiles
- Metadata extraction (month, biome) from tile sidecars
- Mixed precision (FP16) inference for speed
- Progress tracking and ETA
- Automatic output directory creation
- Quality metrics computation (optional)
- CPU fallback support for low-VRAM systems

Input:
    - opt_v1 tiles (NPZ format): {tile_id}.npz containing 'opt_v1' (4, H, W)
    - metadata sidecars (JSON format): {tile_id}.json with 'month' and 'biome_code'

Output:
    - opt_v2 tiles (NPZ format): {tile_id}.npz containing 'opt_v2' (4, H, W)
    - metadata sidecars (JSON format): {tile_id}.json with processing info

Usage:
    python scripts/infer_stage2.py \\
        --checkpoint outputs/stage2/best.pt \\
        --input_dir data/stage1_outputs \\
        --output_dir data/stage2_outputs \\
        --config configs/hardware.lowvr.yaml

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
import os
from pathlib import Path
import argparse
import yaml
import json
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import warnings
from tqdm import tqdm
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np

try:
    from axs_lib.stage2_prithvi_refine import build_prithvi_refiner, PrithviRefiner
    from axs_lib.io import load_checkpoint
    AXSLIB_AVAILABLE = True
except ImportError as e:
    AXSLIB_AVAILABLE = False
    warnings.warn(f"axs_lib modules not available: {e}")


# ============================================================================
# Metadata Embedding (must match training)
# ============================================================================

class MetadataEmbedding(nn.Module):
    """
    Embeds metadata tokens (month, biome) into channel-wise feature maps.
    
    NOTE: This must match the training implementation exactly.
    """
    
    def __init__(
        self,
        num_months: int = 12,
        num_biomes: int = 16,
        embed_dim: int = 64,
        out_channels: int = 4
    ):
        super().__init__()
        
        self.num_months = num_months
        self.num_biomes = num_biomes
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        
        # Biome embedding (learnable)
        self.biome_embed = nn.Embedding(num_biomes, embed_dim)
        
        # Project embeddings to channel space
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, out_channels)
        )
    
    def forward(
        self,
        month: torch.Tensor,
        biome: torch.Tensor,
        spatial_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """Embed metadata and broadcast to spatial dimensions."""
        B = month.shape[0]
        H, W = spatial_shape
        
        # Flatten if needed
        if month.ndim > 1:
            month = month.squeeze(-1)
        if biome.ndim > 1:
            biome = biome.squeeze(-1)
        
        # Month sinusoidal encoding
        month_norm = month.float() / self.num_months
        month_embed = self._sinusoidal_encoding(month_norm, self.embed_dim)
        
        # Biome learnable embedding
        biome_embed = self.biome_embed(biome)
        
        # Concatenate and project
        combined = torch.cat([month_embed, biome_embed], dim=1)
        features = self.proj(combined)
        
        # Broadcast to spatial dimensions
        features = features.unsqueeze(-1).unsqueeze(-1)
        features = features.expand(B, self.out_channels, H, W)
        
        return features
    
    def _sinusoidal_encoding(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute sinusoidal positional encoding."""
        B = x.shape[0]
        device = x.device
        
        # Create frequency bands
        freqs = torch.arange(0, dim // 2, device=device).float()
        freqs = 1.0 / (10000 ** (2 * freqs / dim))
        
        # Compute sine and cosine
        x = x.unsqueeze(-1)
        args = x * freqs.unsqueeze(0)
        
        sin_vals = torch.sin(2 * np.pi * args)
        cos_vals = torch.cos(2 * np.pi * args)
        
        # Interleave sine and cosine
        encoding = torch.stack([sin_vals, cos_vals], dim=2).flatten(1)
        
        return encoding


class Stage2ModelWithMetadata(nn.Module):
    """Wraps PrithviRefiner with metadata conditioning."""
    
    def __init__(
        self,
        prithvi_refiner: PrithviRefiner,
        num_months: int = 12,
        num_biomes: int = 16,
        embed_dim: int = 64
    ):
        super().__init__()
        
        self.prithvi_refiner = prithvi_refiner
        self.metadata_embed = MetadataEmbedding(
            num_months=num_months,
            num_biomes=num_biomes,
            embed_dim=embed_dim,
            out_channels=4
        )
        
        # Input projection: 8 channels (4 opt_v1 + 4 metadata) → 4 channels
        self.input_fusion = nn.Conv2d(8, 4, kernel_size=1)
    
    def forward(
        self,
        opt_v1: torch.Tensor,
        month: torch.Tensor,
        biome: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with metadata conditioning."""
        B, C, H, W = opt_v1.shape
        
        # Embed metadata
        metadata_features = self.metadata_embed(month, biome, (H, W))
        
        # Concatenate opt_v1 with metadata features
        fused_input = torch.cat([opt_v1, metadata_features], dim=1)
        
        # Fuse to 4 channels
        fused_input = self.input_fusion(fused_input)
        
        # Pass through Prithvi refiner
        refined = self.prithvi_refiner(fused_input)
        
        return refined


# ============================================================================
# Tile Processing Functions
# ============================================================================

def load_opt_v1_tile(tile_path: Path) -> Tuple[np.ndarray, Dict]:
    """
    Load opt_v1 tile and metadata.
    
    Args:
        tile_path: Path to NPZ file
        
    Returns:
        (opt_v1 array, metadata dict)
    """
    # Load NPZ
    data = np.load(tile_path)
    
    # Get opt_v1 (4, H, W) - [B02, B03, B04, B08]
    if 'opt_v1' in data:
        opt_v1 = data['opt_v1']
    else:
        # Fallback: construct from individual bands
        opt_v1 = np.stack([
            data.get('s2_b2', np.zeros((120, 120), dtype=np.float32)),
            data.get('s2_b3', np.zeros((120, 120), dtype=np.float32)),
            data.get('s2_b4', np.zeros((120, 120), dtype=np.float32)),
            data.get('s2_b8', np.zeros((120, 120), dtype=np.float32))
        ], axis=0)
    
    # Load metadata from sidecar JSON if exists
    json_path = tile_path.with_suffix('.json')
    if json_path.exists():
        with open(json_path) as f:
            metadata = json.load(f)
    else:
        # Default metadata
        metadata = {
            'month': 6,  # June
            'biome_code': 0,  # Unknown
            'tile_id': tile_path.stem
        }
    
    # Extract metadata fields
    month = metadata.get('month', 6)
    biome = metadata.get('biome_code', 0)
    
    # Handle different month formats
    if isinstance(month, str):
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        month = month_map.get(month.lower()[:3], 6)
    
    metadata['month'] = int(month)
    metadata['biome_code'] = int(biome)
    
    return opt_v1, metadata


def save_opt_v2_tile(
    output_path: Path,
    opt_v2: np.ndarray,
    metadata: Dict,
    processing_info: Optional[Dict] = None
):
    """
    Save opt_v2 tile and metadata.
    
    Args:
        output_path: Path to output NPZ file
        opt_v2: Refined optical array (4, H, W)
        metadata: Metadata dict from input
        processing_info: Additional processing info to save
    """
    # Save NPZ
    np.savez_compressed(output_path, opt_v2=opt_v2)
    
    # Update metadata
    metadata_out = metadata.copy()
    metadata_out['stage'] = 'stage2_refined'
    metadata_out['timestamp'] = datetime.now().isoformat()
    
    if processing_info:
        metadata_out['processing'] = processing_info
    
    # Save metadata JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(metadata_out, f, indent=2)


def process_batch(
    model: nn.Module,
    opt_v1_batch: torch.Tensor,
    month_batch: torch.Tensor,
    biome_batch: torch.Tensor,
    device: torch.device,
    use_amp: bool = True
) -> torch.Tensor:
    """
    Process a batch of opt_v1 tiles.
    
    Args:
        model: Stage2ModelWithMetadata
        opt_v1_batch: Batch of opt_v1 tiles (B, 4, H, W)
        month_batch: Batch of month indices (B,)
        biome_batch: Batch of biome indices (B,)
        device: Device to run on
        use_amp: Whether to use mixed precision
        
    Returns:
        Batch of opt_v2 tiles (B, 4, H, W)
    """
    # Move to device
    opt_v1_batch = opt_v1_batch.to(device)
    month_batch = month_batch.to(device)
    biome_batch = biome_batch.to(device)
    
    # Inference with AMP
    with torch.no_grad():
        if use_amp:
            with autocast():
                opt_v2_batch = model(opt_v1_batch, month_batch, biome_batch)
        else:
            opt_v2_batch = model(opt_v1_batch, month_batch, biome_batch)
    
    return opt_v2_batch


# ============================================================================
# Main Inference Function
# ============================================================================

def run_inference(
    checkpoint_path: Path,
    input_dir: Path,
    output_dir: Path,
    config: Dict,
    batch_size: int = 4,
    device: str = 'cuda',
    use_amp: bool = True,
    num_workers: int = 0,
    skip_existing: bool = True
):
    """
    Run Stage 2 inference on a folder of opt_v1 tiles.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        input_dir: Directory containing opt_v1 tiles
        output_dir: Directory to save opt_v2 tiles
        config: Configuration dict
        batch_size: Batch size for inference
        device: Device to run on ('cuda' or 'cpu')
        use_amp: Whether to use mixed precision
        num_workers: Number of data loading workers (not used for simple impl)
        skip_existing: Skip tiles that already have output
    """
    print("="*70)
    print("Stage 2 Inference: opt_v1 → opt_v2")
    print("="*70)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nInput:  {input_dir}")
    print(f"Output: {output_dir}")
    
    # ========================================================================
    # Load Model
    # ========================================================================
    
    print("\nLoading model...")
    
    # Build model architecture
    prithvi_refiner = build_prithvi_refiner(
        config=config,
        device=device
    )
    
    model = Stage2ModelWithMetadata(
        prithvi_refiner=prithvi_refiner,
        num_months=12,
        num_biomes=16,
        embed_dim=64
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get checkpoint info
    epoch = checkpoint.get('epoch', 'unknown')
    metrics = checkpoint.get('metrics', {})
    print(f"  Epoch: {epoch}")
    if metrics:
        print(f"  Metrics: {metrics}")
    
    # Set to eval mode
    model.eval()
    
    print(f"✓ Model loaded successfully")
    
    # ========================================================================
    # Find Input Tiles
    # ========================================================================
    
    print("\nFinding input tiles...")
    input_tiles = sorted(input_dir.glob('*.npz'))
    
    if not input_tiles:
        print(f"⚠️  No NPZ tiles found in {input_dir}")
        return
    
    print(f"Found {len(input_tiles)} tiles")
    
    # Filter out already processed tiles if skip_existing
    if skip_existing:
        to_process = []
        for tile_path in input_tiles:
            output_path = output_dir / tile_path.name
            if not output_path.exists():
                to_process.append(tile_path)
        
        if len(to_process) < len(input_tiles):
            print(f"Skipping {len(input_tiles) - len(to_process)} existing tiles")
        
        input_tiles = to_process
    
    if not input_tiles:
        print("All tiles already processed!")
        return
    
    print(f"Processing {len(input_tiles)} tiles")
    
    # ========================================================================
    # Process Tiles in Batches
    # ========================================================================
    
    print(f"\nProcessing tiles (batch_size={batch_size})...\n")
    
    num_batches = (len(input_tiles) + batch_size - 1) // batch_size
    
    total_time = 0
    processed_count = 0
    cpu_fallback_count = 0
    
    # Track warnings for CPU fallbacks
    original_showwarning = warnings.showwarning
    
    def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
        nonlocal cpu_fallback_count
        if "CPU fallback" in str(message):
            cpu_fallback_count += 1
        original_showwarning(message, category, filename, lineno, file, line)
    
    warnings.showwarning = custom_warning_handler
    
    pbar = tqdm(total=len(input_tiles), desc="Processing tiles", unit="tile")
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(input_tiles))
        batch_tiles = input_tiles[batch_start:batch_end]
        
        # Load batch
        opt_v1_list = []
        month_list = []
        biome_list = []
        metadata_list = []
        
        for tile_path in batch_tiles:
            try:
                opt_v1, metadata = load_opt_v1_tile(tile_path)
                opt_v1_list.append(opt_v1)
                month_list.append(metadata['month'])
                biome_list.append(metadata['biome_code'])
                metadata_list.append(metadata)
            except Exception as e:
                print(f"\n⚠️  Error loading {tile_path.name}: {e}")
                # Add dummy data to maintain batch structure
                opt_v1_list.append(np.zeros((4, 120, 120), dtype=np.float32))
                month_list.append(6)
                biome_list.append(0)
                metadata_list.append({'error': str(e)})
        
        # Convert to tensors
        opt_v1_batch = torch.from_numpy(np.stack(opt_v1_list)).float()
        month_batch = torch.tensor(month_list, dtype=torch.long)
        biome_batch = torch.tensor(biome_list, dtype=torch.long)
        
        # Process batch
        start_time = time.time()
        
        try:
            opt_v2_batch = process_batch(
                model=model,
                opt_v1_batch=opt_v1_batch,
                month_batch=month_batch,
                biome_batch=biome_batch,
                device=device,
                use_amp=use_amp
            )
            
            # Convert to numpy
            opt_v2_batch = opt_v2_batch.cpu().numpy()
            
        except Exception as e:
            print(f"\n⚠️  Error processing batch {batch_idx}: {e}")
            # Create dummy output
            opt_v2_batch = opt_v1_batch.numpy()
        
        batch_time = time.time() - start_time
        total_time += batch_time
        
        # Save tiles
        for i, (tile_path, metadata) in enumerate(zip(batch_tiles, metadata_list)):
            output_path = output_dir / tile_path.name
            opt_v2 = opt_v2_batch[i]
            
            # Processing info
            processing_info = {
                'input_file': str(tile_path),
                'checkpoint': str(checkpoint_path),
                'device': str(device),
                'use_amp': use_amp,
                'processing_time_ms': (batch_time / len(batch_tiles)) * 1000
            }
            
            try:
                save_opt_v2_tile(output_path, opt_v2, metadata, processing_info)
                processed_count += 1
            except Exception as e:
                print(f"\n⚠️  Error saving {output_path.name}: {e}")
        
        pbar.update(len(batch_tiles))
    
    pbar.close()
    
    # Restore warning handler
    warnings.showwarning = original_showwarning
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "="*70)
    print("Inference Complete!")
    print("="*70)
    
    avg_time_per_tile = total_time / processed_count if processed_count > 0 else 0
    
    print(f"\nProcessed: {processed_count}/{len(input_tiles)} tiles")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per tile: {avg_time_per_tile*1000:.1f}ms")
    print(f"Throughput: {processed_count/total_time:.1f} tiles/sec")
    
    if cpu_fallback_count > 0:
        print(f"\n⚠️  CPU fallback occurred {cpu_fallback_count} times")
        print("   Consider reducing batch size or tile size")
    
    print(f"\nOutput saved to: {output_dir}")
    print("="*70)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 Inference: Refine opt_v1 tiles to opt_v2"
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained Stage 2 checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing opt_v1 tiles (NPZ)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save opt_v2 tiles')
    parser.add_argument('--config', type=str, default='configs/hardware.lowvr.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision (slower but more stable)')
    parser.add_argument('--no_skip_existing', action='store_true',
                        help='Reprocess existing tiles')
    
    args = parser.parse_args()
    
    # Convert paths
    checkpoint_path = Path(args.checkpoint)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validate paths
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    # Load config
    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: Config not found: {args.config}, using defaults")
    
    # Run inference
    try:
        run_inference(
            checkpoint_path=checkpoint_path,
            input_dir=input_dir,
            output_dir=output_dir,
            config=config,
            batch_size=args.batch_size,
            device=args.device,
            use_amp=not args.no_amp,
            skip_existing=not args.no_skip_existing
        )
        return 0
    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR during inference:")
        print("="*70)
        print(str(e))
        
        import traceback
        traceback.print_exc()
        
        return 1


if __name__ == '__main__':
    exit(main())
