"""
Stage 1 Inference: SAR to Optical Translation

This script performs batch inference using a trained TerraMind generator to
convert Sentinel-1 SAR tiles to synthetic Sentinel-2 optical imagery.

Features:
- Batch processing of SAR tiles from a directory
- Automatic checkpoint loading (best model or specific checkpoint)
- Output tiles saved alongside inputs with configurable suffix
- Progress tracking with optional visualization
- Memory-efficient processing with configurable batch size
- Support for both NPZ and GeoTIFF formats

Output Format:
- Band order: B02, B03, B04, B08 (Blue, Green, Red, NIR)
- Values: Normalized to [0, 1] range
- Format: NPZ or GeoTIFF (matching input format)

Usage:
    python scripts/infer_stage1.py --input-dir tiles/s1/ --checkpoint runs/exp1/best_model.pt
    python scripts/infer_stage1.py --input-dir tiles/ --checkpoint best --output-suffix _opt_synth
    python scripts/infer_stage1.py --help
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axs_lib.models import build_terramind_generator
from axs_lib.stage1_tm_s2o import tm_sar2opt
from axs_lib.stdz import TERRAMIND_S1_STATS

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    warnings.warn("rasterio not available, GeoTIFF I/O will be disabled")


# ============================================================================
# Tile I/O Functions
# ============================================================================

def load_s1_tile_npz(tile_path: Path) -> Tuple[torch.Tensor, Dict]:
    """
    Load Sentinel-1 tile from NPZ file.
    
    Args:
        tile_path: Path to NPZ file
        
    Returns:
        Tuple of (s1_tensor, metadata)
        s1_tensor shape: (1, 2, H, W) [VV, VH]
    """
    data = np.load(tile_path)
    
    # Extract SAR bands
    s1_vv = data['s1_vv']
    s1_vh = data['s1_vh']
    s1 = np.stack([s1_vv, s1_vh], axis=0).astype(np.float32)
    
    # Standardize
    means = torch.from_numpy(TERRAMIND_S1_STATS.means.astype(np.float32)).view(1, 2, 1, 1)
    stds = torch.from_numpy(TERRAMIND_S1_STATS.stds.astype(np.float32)).view(1, 2, 1, 1)
    
    s1_tensor = torch.from_numpy(s1).unsqueeze(0)  # Add batch dimension
    s1_tensor = (s1_tensor - means) / stds
    
    # Extract metadata if available
    metadata = {
        'shape': s1.shape,
        'tile_id': tile_path.stem
    }
    
    # Try to load geotransform from JSON sidecar
    json_path = tile_path.with_suffix('.json')
    if json_path.exists():
        import json
        with open(json_path, 'r') as f:
            tile_meta = json.load(f)
            metadata.update(tile_meta)
    
    return s1_tensor, metadata


def load_s1_tile_geotiff(tile_path: Path) -> Tuple[torch.Tensor, Dict]:
    """
    Load Sentinel-1 tile from GeoTIFF file.
    
    Expected format: 2-band GeoTIFF with VV and VH
    
    Args:
        tile_path: Path to GeoTIFF file
        
    Returns:
        Tuple of (s1_tensor, metadata)
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required to load GeoTIFF files")
    
    with rasterio.open(tile_path) as src:
        # Read bands (assumes band 1=VV, band 2=VH)
        s1_vv = src.read(1).astype(np.float32)
        s1_vh = src.read(2).astype(np.float32)
        s1 = np.stack([s1_vv, s1_vh], axis=0)
        
        # Extract metadata
        metadata = {
            'shape': s1.shape,
            'tile_id': tile_path.stem,
            'crs': str(src.crs),
            'transform': src.transform,
            'bounds': src.bounds,
            'nodata': src.nodata
        }
    
    # Standardize
    means = torch.from_numpy(TERRAMIND_S1_STATS.means.astype(np.float32)).view(1, 2, 1, 1)
    stds = torch.from_numpy(TERRAMIND_S1_STATS.stds.astype(np.float32)).view(1, 2, 1, 1)
    
    s1_tensor = torch.from_numpy(s1).unsqueeze(0)  # Add batch dimension
    s1_tensor = (s1_tensor - means) / stds
    
    return s1_tensor, metadata


def save_opt_tile_npz(
    opt_tensor: torch.Tensor,
    output_path: Path,
    metadata: Optional[Dict] = None
):
    """
    Save synthetic optical tile to NPZ file.
    
    Args:
        opt_tensor: Optical tensor (1, 4, H, W) [B02, B03, B04, B08]
        output_path: Path to save NPZ file
        metadata: Optional metadata to save
    """
    # Remove batch dimension and convert to numpy
    opt = opt_tensor.squeeze(0).cpu().numpy()
    
    # Split into bands
    s2_b2 = opt[0]
    s2_b3 = opt[1]
    s2_b4 = opt[2]
    s2_b8 = opt[3]
    
    # Save to NPZ
    np.savez_compressed(
        output_path,
        s2_b2=s2_b2,
        s2_b3=s2_b3,
        s2_b4=s2_b4,
        s2_b8=s2_b8
    )
    
    # Save metadata to JSON sidecar if provided
    if metadata is not None:
        import json
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            # Convert non-serializable types
            meta_copy = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    meta_copy[key] = value
                else:
                    meta_copy[key] = str(value)
            json.dump(meta_copy, f, indent=2)


def save_opt_tile_geotiff(
    opt_tensor: torch.Tensor,
    output_path: Path,
    metadata: Dict
):
    """
    Save synthetic optical tile to GeoTIFF file.
    
    Args:
        opt_tensor: Optical tensor (1, 4, H, W) [B02, B03, B04, B08]
        output_path: Path to save GeoTIFF file
        metadata: Metadata including CRS, transform, etc.
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required to save GeoTIFF files")
    
    # Remove batch dimension and convert to numpy
    opt = opt_tensor.squeeze(0).cpu().numpy()
    
    # Get spatial info from metadata
    transform = metadata.get('transform')
    crs = metadata.get('crs')
    
    # Write GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=opt.shape[1],
        width=opt.shape[2],
        count=4,
        dtype=opt.dtype,
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        # Write each band
        for i in range(4):
            dst.write(opt[i], i + 1)
        
        # Set band descriptions
        dst.set_band_description(1, 'B02 (Blue)')
        dst.set_band_description(2, 'B03 (Green)')
        dst.set_band_description(3, 'B04 (Red)')
        dst.set_band_description(4, 'B08 (NIR)')


# ============================================================================
# Inference Functions
# ============================================================================

def load_checkpoint(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract timesteps from checkpoint if available
    timesteps = checkpoint.get('timesteps', 12)
    
    # Build model
    model = build_terramind_generator(
        input_modalities=("S1GRD",),
        output_modalities=("S2L2A",),
        timesteps=timesteps,
        standardize=True,
        pretrained=False  # We'll load weights from checkpoint
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully (timesteps={timesteps})")
    
    return model


def infer_single_tile(
    model: nn.Module,
    s1_tensor: torch.Tensor,
    device: torch.device,
    timesteps: int = 12
) -> torch.Tensor:
    """
    Run inference on a single SAR tile.
    
    Args:
        model: Trained TerraMind generator
        s1_tensor: SAR input (1, 2, H, W)
        device: Device to run on
        timesteps: Number of diffusion timesteps
        
    Returns:
        Synthetic optical tensor (1, 4, H, W) in [0, 1] range
    """
    s1_tensor = s1_tensor.to(device)
    
    with torch.no_grad():
        opt_tensor = tm_sar2opt(
            model,
            s1_tensor,
            timesteps=timesteps,
            denormalize=True,
            clip_range=(0.0, 1.0)
        )
    
    return opt_tensor


def process_directory(
    model: nn.Module,
    input_dir: Path,
    output_dir: Optional[Path],
    device: torch.device,
    timesteps: int = 12,
    output_suffix: str = "_opt",
    file_format: str = "auto",
    batch_size: int = 1,
    max_tiles: Optional[int] = None,
    overwrite: bool = False
):
    """
    Process all SAR tiles in a directory.
    
    Args:
        model: Trained TerraMind generator
        input_dir: Directory containing input SAR tiles
        output_dir: Output directory (None = save next to inputs)
        device: Device to run on
        timesteps: Number of diffusion timesteps
        output_suffix: Suffix to add to output filenames
        file_format: Output format ('auto', 'npz', 'geotiff')
        batch_size: Batch size for processing (currently only 1 supported)
        max_tiles: Maximum number of tiles to process (None = all)
        overwrite: Whether to overwrite existing output files
    """
    # Find all SAR tiles
    npz_files = sorted(list(input_dir.glob("*.npz")))
    tif_files = sorted(list(input_dir.glob("*.tif"))) + sorted(list(input_dir.glob("*.tiff")))
    
    # Filter to S1 tiles only (heuristic: contains 's1' in name or has 2 bands)
    input_files = []
    for f in npz_files:
        if 's1' in f.stem.lower():
            input_files.append(('npz', f))
    
    for f in tif_files:
        if 's1' in f.stem.lower() and HAS_RASTERIO:
            input_files.append(('tif', f))
    
    if not input_files:
        print(f"No SAR tiles found in {input_dir}")
        print("Expected: NPZ files with 's1' in filename, or GeoTIFF files with 's1' in filename")
        return
    
    # Limit if specified
    if max_tiles is not None:
        input_files = input_files[:max_tiles]
    
    print(f"\nFound {len(input_files)} SAR tiles to process")
    print(f"Output suffix: '{output_suffix}'")
    print(f"Output format: {file_format}")
    if output_dir:
        print(f"Output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        print("Output directory: Same as input")
    print()
    
    # Process tiles
    successful = 0
    failed = 0
    skipped = 0
    
    for file_type, tile_path in tqdm(input_files, desc="Processing tiles"):
        try:
            # Determine output path
            if output_dir is not None:
                output_base = output_dir / tile_path.stem
            else:
                output_base = tile_path.parent / tile_path.stem
            
            # Add suffix
            output_stem = output_base.name + output_suffix
            output_base = output_base.parent / output_stem
            
            # Determine output format
            if file_format == 'auto':
                out_format = file_type
            else:
                out_format = file_format
            
            # Set output extension
            if out_format == 'npz':
                output_path = output_base.with_suffix('.npz')
            elif out_format in ['tif', 'geotiff']:
                output_path = output_base.with_suffix('.tif')
            else:
                raise ValueError(f"Unknown output format: {out_format}")
            
            # Check if output exists
            if output_path.exists() and not overwrite:
                skipped += 1
                continue
            
            # Load SAR tile
            if file_type == 'npz':
                s1_tensor, metadata = load_s1_tile_npz(tile_path)
            elif file_type == 'tif':
                s1_tensor, metadata = load_s1_tile_geotiff(tile_path)
            else:
                raise ValueError(f"Unknown file type: {file_type}")
            
            # Run inference
            opt_tensor = infer_single_tile(model, s1_tensor, device, timesteps)
            
            # Save output
            metadata['source_tile'] = str(tile_path)
            metadata['generated_by'] = 'TerraMind Stage 1'
            metadata['timesteps'] = timesteps
            
            if out_format == 'npz':
                save_opt_tile_npz(opt_tensor, output_path, metadata)
            elif out_format in ['tif', 'geotiff']:
                save_opt_tile_geotiff(opt_tensor, output_path, metadata)
            
            successful += 1
            
        except Exception as e:
            print(f"\n✗ Failed to process {tile_path.name}: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total: {len(input_files)}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 Inference: SAR to Optical Translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all tiles in a directory with best model
  python scripts/infer_stage1.py --input-dir tiles/s1/ --checkpoint runs/exp1/best_model.pt
  
  # Process with specific checkpoint and output directory
  python scripts/infer_stage1.py --input-dir tiles/s1/ --checkpoint runs/exp1/checkpoints/checkpoint_step_5000.pt --output-dir tiles/s2_synth/
  
  # Quick test with 10 tiles
  python scripts/infer_stage1.py --input-dir tiles/s1/ --checkpoint best --max-tiles 10
  
  # Custom output suffix and format
  python scripts/infer_stage1.py --input-dir tiles/ --checkpoint best --output-suffix _synthetic_opt --format geotiff
  
  # Process and overwrite existing outputs
  python scripts/infer_stage1.py --input-dir tiles/ --checkpoint best --overwrite
        """
    )
    
    # Input/Output
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing input SAR tiles (NPZ or GeoTIFF)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as input, with suffix)')
    parser.add_argument('--output-suffix', type=str, default='_opt',
                        help='Suffix to add to output filenames (default: _opt)')
    parser.add_argument('--format', type=str, default='auto',
                        choices=['auto', 'npz', 'geotiff', 'tif'],
                        help='Output format (default: auto = same as input)')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file or "best" to auto-find best_model.pt')
    parser.add_argument('--timesteps', type=int, default=12,
                        help='Number of diffusion timesteps (default: 12)')
    
    # Processing
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (default: 1, currently only 1 is supported)')
    parser.add_argument('--max-tiles', type=int, default=None,
                        help='Maximum number of tiles to process (default: all)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output files')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    print("=" * 80)
    print("Stage 1 Inference: SAR to Optical Translation")
    print("=" * 80)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Parse paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Find checkpoint
    if args.checkpoint.lower() == 'best':
        # Look for best_model.pt in common locations
        search_paths = [
            Path('best_model.pt'),
            Path('runs/best_model.pt'),
            Path('checkpoints/best_model.pt'),
            Path('output/best_model.pt')
        ]
        
        checkpoint_path = None
        for p in search_paths:
            if p.exists():
                checkpoint_path = p
                break
        
        if checkpoint_path is None:
            print("Error: Could not find best_model.pt. Please specify full checkpoint path.")
            sys.exit(1)
    else:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    
    # Load model
    model = load_checkpoint(checkpoint_path, device)
    
    # Process directory
    start_time = time.time()
    
    process_directory(
        model=model,
        input_dir=input_dir,
        output_dir=output_dir,
        device=device,
        timesteps=args.timesteps,
        output_suffix=args.output_suffix,
        file_format=args.format,
        batch_size=args.batch_size,
        max_tiles=args.max_tiles,
        overwrite=args.overwrite
    )
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f} seconds")
    
    print("\n" + "=" * 80)
    print("Inference complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
