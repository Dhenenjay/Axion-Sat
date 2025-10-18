"""
Stage 1 Inference: Production-Ready SAR to Optical Translation

This module provides a simple, robust interface for converting SAR imagery
to multispectral optical data using TerraMind with gamma correction.

Features:
- Accepts multiple input formats (numpy, torch, file paths)
- Automatic normalization and preprocessing
- Gamma correction for optimal brightness (γ=0.7)
- Returns 12-band S2L2A output ready for analysis

Usage:
    >>> from axs_lib.stage1_inference import infer_sar_to_optical
    >>> 
    >>> # From file
    >>> optical = infer_sar_to_optical('path/to/sar.tif')
    >>> 
    >>> # From numpy array
    >>> optical = infer_sar_to_optical(sar_array)  # (2, H, W) in dB
    >>> 
    >>> # With custom gamma
    >>> optical = infer_sar_to_optical(sar_array, gamma=0.7)

Author: Axion-Sat Project
Version: 1.0.0 (Production)
"""

import sys
from pathlib import Path
from typing import Union, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Global Model Cache
# ============================================================================

_TERRAMIND_MODEL = None
_DEVICE = None


def _get_model(device: Optional[torch.device] = None):
    """Load TerraMind model (cached for efficiency)."""
    global _TERRAMIND_MODEL, _DEVICE
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Return cached model if available and on same device
    if _TERRAMIND_MODEL is not None and _DEVICE == device:
        return _TERRAMIND_MODEL, _DEVICE
    
    try:
        from terratorch import FULL_MODEL_REGISTRY
        
        print(f"Loading TerraMind model on {device}...")
        model = FULL_MODEL_REGISTRY.build(
            'terramind_v1_large_generate',
            pretrained=True,
            modalities=['S1GRD'],
            output_modalities=['S2L2A'],
            timesteps=10,
            standardize=True
        )
        model = model.to(device)
        model.eval()
        
        _TERRAMIND_MODEL = model
        _DEVICE = device
        
        print(f"✓ TerraMind loaded successfully")
        return model, device
        
    except Exception as e:
        raise RuntimeError(f"Failed to load TerraMind model: {e}")


# ============================================================================
# Input Processing
# ============================================================================

def _load_sar_from_file(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load SAR data from file.
    
    Supports: .tif, .tiff, .npy, .npz
    
    Args:
        file_path: Path to SAR file
        
    Returns:
        SAR array (2, H, W) in dB scale
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"SAR file not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix in ['.tif', '.tiff']:
        try:
            import rasterio
            with rasterio.open(file_path) as src:
                sar = src.read()  # (C, H, W)
                if sar.shape[0] != 2:
                    raise ValueError(f"Expected 2 bands (VV, VH), got {sar.shape[0]}")
                return sar.astype(np.float32)
        except ImportError:
            raise ImportError("rasterio required for .tif files. Install: pip install rasterio")
    
    elif suffix == '.npy':
        sar = np.load(file_path)
        return sar.astype(np.float32)
    
    elif suffix == '.npz':
        data = np.load(file_path)
        # Try common key names
        for vv_key in ['s1_vv', 'vv', 'VV']:
            for vh_key in ['s1_vh', 'vh', 'VH']:
                if vv_key in data and vh_key in data:
                    sar = np.stack([data[vv_key], data[vh_key]], axis=0)
                    return sar.astype(np.float32)
        raise ValueError(f"Could not find VV/VH bands in .npz file. Available keys: {list(data.keys())}")
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .tif, .npy, or .npz")


def _normalize_sar_input(
    sar: np.ndarray,
    input_scale: str = 'auto'
) -> np.ndarray:
    """
    Normalize SAR input to dB scale.
    
    Args:
        sar: SAR array (2, H, W)
        input_scale: 'auto', 'db', 'linear', or 'normalized'
        
    Returns:
        SAR in dB scale (2, H, W)
    """
    if input_scale == 'auto':
        # Detect scale automatically
        sar_min, sar_max = sar.min(), sar.max()
        
        if sar_min >= -50 and sar_max <= 20:
            # Already in dB
            input_scale = 'db'
        elif sar_min >= 0 and sar_max <= 1:
            # Normalized [0, 1]
            input_scale = 'normalized'
        else:
            # Assume linear
            input_scale = 'linear'
        
        print(f"  Auto-detected input scale: {input_scale}")
    
    if input_scale == 'db':
        # Already in dB
        return sar
    
    elif input_scale == 'normalized':
        # Denormalize from [0, 1] to typical dB range [-25, 5]
        sar_db = sar * 30 - 25
        return sar_db
    
    elif input_scale == 'linear':
        # Convert from linear to dB (handle zeros)
        sar_db = 10 * np.log10(np.maximum(sar, 1e-10))
        return sar_db
    
    else:
        raise ValueError(f"Unknown input_scale: {input_scale}")


def _preprocess_sar(
    sar: Union[np.ndarray, torch.Tensor, str, Path],
    input_scale: str = 'auto',
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Preprocess SAR input for TerraMind.
    
    Args:
        sar: SAR input (multiple formats supported)
        input_scale: 'auto', 'db', 'linear', or 'normalized'
        device: Target device
        
    Returns:
        (padded_sar_tensor, (orig_h, orig_w))
    """
    # Load from file if needed
    if isinstance(sar, (str, Path)):
        print(f"Loading SAR from: {sar}")
        sar = _load_sar_from_file(sar)
    
    # Convert to numpy if torch
    if isinstance(sar, torch.Tensor):
        sar = sar.cpu().numpy()
    
    # Ensure numpy array
    sar = np.asarray(sar, dtype=np.float32)
    
    # Validate shape
    if sar.ndim == 2:
        # Single channel, assume VV, add dummy VH
        warnings.warn("Single-channel SAR detected, duplicating as VV=VH")
        sar = np.stack([sar, sar], axis=0)
    
    if sar.ndim != 3 or sar.shape[0] != 2:
        raise ValueError(f"Expected SAR shape (2, H, W), got {sar.shape}")
    
    # Normalize to dB
    sar_db = _normalize_sar_input(sar, input_scale)
    
    # Store original size
    orig_h, orig_w = sar_db.shape[1], sar_db.shape[2]
    
    # Convert to torch
    sar_tensor = torch.from_numpy(sar_db).float().unsqueeze(0)  # (1, 2, H, W)
    
    if device is not None:
        sar_tensor = sar_tensor.to(device)
    
    # Pad to 224x224 (TerraMind expects this)
    if orig_h != 224 or orig_w != 224:
        pad_h = max(0, 224 - orig_h)
        pad_w = max(0, 224 - orig_w)
        sar_tensor = nn.functional.pad(
            sar_tensor,
            (0, pad_w, 0, pad_h),
            mode='reflect'
        )
    
    return sar_tensor, (orig_h, orig_w)


# ============================================================================
# Main Inference Function
# ============================================================================

@torch.no_grad()
def infer_sar_to_optical(
    sar: Union[np.ndarray, torch.Tensor, str, Path],
    gamma: float = 0.7,
    input_scale: str = 'auto',
    return_all_bands: bool = True,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Convert SAR imagery to multispectral optical data.
    
    This is the main inference function for Stage 1. It accepts flexible
    input formats and returns calibrated optical imagery.
    
    Args:
        sar: SAR input, can be:
            - File path (str/Path): .tif, .npy, .npz
            - Numpy array: (2, H, W) SAR data
            - Torch tensor: (2, H, W) or (B, 2, H, W)
        gamma: Gamma correction for brightness (default: 0.7)
            - 0.5-0.6: Very bright
            - 0.7: Optimal balance (recommended)
            - 0.8-1.0: Darker, more accurate
        input_scale: Input data scale ('auto', 'db', 'linear', 'normalized')
            - 'auto': Detect automatically (recommended)
            - 'db': Already in dB scale [-50, 20]
            - 'linear': Linear backscatter values
            - 'normalized': Normalized to [0, 1]
        return_all_bands: If True, return all 12 S2L2A bands
                         If False, return 4 bands (B02, B03, B04, B08)
        device: Computation device (auto-detect if None)
        
    Returns:
        Optical imagery as numpy array:
        - If return_all_bands=True: (12, H, W) in DN [0, 10000]
        - If return_all_bands=False: (4, H, W) in reflectance [0, 1]
        
    Examples:
        >>> # Simple usage
        >>> optical = infer_sar_to_optical('sar.tif')
        >>> 
        >>> # From numpy with custom gamma
        >>> optical = infer_sar_to_optical(sar_array, gamma=0.7)
        >>> 
        >>> # Get 4-band RGB+NIR only
        >>> rgb_nir = infer_sar_to_optical(sar_array, return_all_bands=False)
    """
    # Load model
    model, device = _get_model(device)
    
    # Preprocess input
    print(f"Preprocessing SAR input...")
    sar_padded, (orig_h, orig_w) = _preprocess_sar(sar, input_scale, device)
    
    print(f"  Input size: {orig_h}x{orig_w}")
    print(f"  Padded size: {sar_padded.shape[2:4]}")
    
    # Run inference
    print(f"Running TerraMind inference...")
    outputs = model({'S1GRD': sar_padded})
    s2_padded = outputs['S2L2A']
    
    # Clamp to valid range
    s2_padded = torch.clamp(s2_padded, 0, 10000)
    
    # Crop back to original size
    if orig_h != 224 or orig_w != 224:
        pad_size_h = (224 - orig_h) // 2
        pad_size_w = (224 - orig_w) // 2
        s2_output = s2_padded[:, :, pad_size_h:pad_size_h+orig_h, pad_size_w:pad_size_w+orig_w]
    else:
        s2_output = s2_padded
    
    # Convert to numpy
    s2_numpy = s2_output.cpu().numpy()[0]  # (12, H, W)
    
    print(f"  Output size: {s2_numpy.shape[1:3]}")
    print(f"  Output range: [{s2_numpy.min():.1f}, {s2_numpy.max():.1f}] DN")
    
    # Apply gamma correction if requested
    if gamma != 1.0:
        print(f"Applying gamma correction (γ={gamma})...")
        # Convert to [0, 1], apply gamma, convert back
        s2_reflectance = s2_numpy / 10000.0
        s2_corrected = np.power(s2_reflectance, gamma)
        s2_numpy = (s2_corrected * 10000.0).astype(np.float32)
        s2_numpy = np.clip(s2_numpy, 0, 10000)
    
    # Return requested bands
    if return_all_bands:
        print(f"✓ Returning all 12 S2L2A bands")
        return s2_numpy
    else:
        # Extract 4 key bands (B02, B03, B04, B08 = indices 1,2,3,7)
        print(f"✓ Returning 4 bands (B02, B03, B04, B08)")
        s2_4band = s2_numpy[[1, 2, 3, 7], :, :]
        # Convert to reflectance [0, 1]
        return s2_4band / 10000.0


# ============================================================================
# Batch Inference
# ============================================================================

@torch.no_grad()
def infer_sar_to_optical_batch(
    sar_list: list,
    gamma: float = 0.7,
    input_scale: str = 'auto',
    return_all_bands: bool = True,
    device: Optional[torch.device] = None,
    batch_size: int = 8
) -> list:
    """
    Batch inference for multiple SAR images.
    
    Args:
        sar_list: List of SAR inputs (paths or arrays)
        gamma: Gamma correction value
        input_scale: Input scale type
        return_all_bands: Return 12 or 4 bands
        device: Computation device
        batch_size: Number of images per batch
        
    Returns:
        List of optical arrays
    """
    print(f"Processing {len(sar_list)} images in batches of {batch_size}...")
    
    results = []
    for i in range(0, len(sar_list), batch_size):
        batch = sar_list[i:i+batch_size]
        print(f"\nBatch {i//batch_size + 1}/{(len(sar_list)-1)//batch_size + 1}")
        
        for sar in batch:
            optical = infer_sar_to_optical(
                sar,
                gamma=gamma,
                input_scale=input_scale,
                return_all_bands=return_all_bands,
                device=device
            )
            results.append(optical)
    
    print(f"\n✓ Processed {len(results)} images")
    return results


# ============================================================================
# Utility Functions
# ============================================================================

def save_optical_output(
    optical: np.ndarray,
    output_path: Union[str, Path],
    format: str = 'geotiff'
):
    """
    Save optical output to file.
    
    Args:
        optical: Optical array from inference
        output_path: Where to save
        format: 'geotiff', 'npy', or 'npz'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'geotiff':
        try:
            import rasterio
            from rasterio.transform import from_bounds
            
            # Save as GeoTIFF
            height, width = optical.shape[1:3]
            count = optical.shape[0]
            
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=count,
                dtype=optical.dtype,
                compress='lzw'
            ) as dst:
                dst.write(optical)
            
            print(f"✓ Saved to: {output_path}")
            
        except ImportError:
            raise ImportError("rasterio required for GeoTIFF. Install: pip install rasterio")
    
    elif format == 'npy':
        np.save(output_path, optical)
        print(f"✓ Saved to: {output_path}")
    
    elif format == 'npz':
        np.savez_compressed(output_path, optical=optical)
        print(f"✓ Saved to: {output_path}")
    
    else:
        raise ValueError(f"Unknown format: {format}")


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Stage 1: SAR to Optical Translation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input', help='Input SAR file (.tif, .npy, .npz)')
    parser.add_argument('-o', '--output', help='Output path (optional)')
    parser.add_argument('-g', '--gamma', type=float, default=0.7,
                       help='Gamma correction (0.5=bright, 0.7=optimal, 1.0=raw)')
    parser.add_argument('--all-bands', action='store_true',
                       help='Return all 12 bands instead of 4')
    parser.add_argument('--format', choices=['geotiff', 'npy', 'npz'],
                       default='geotiff', help='Output format')
    
    args = parser.parse_args()
    
    # Run inference
    optical = infer_sar_to_optical(
        args.input,
        gamma=args.gamma,
        return_all_bands=args.all_bands
    )
    
    # Save if output specified
    if args.output:
        save_optical_output(optical, args.output, args.format)
    else:
        print(f"\n✓ Inference complete. Output shape: {optical.shape}")
        print("No output file specified. Use -o to save results.")
