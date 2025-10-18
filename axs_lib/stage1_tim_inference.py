"""
Stage 1 Inference with TiM + Tokenizers (Production v2.0)

This is the PROPER TerraMind implementation using:
1. Thinking-in-Modalities (TiM) - generates LULC internally for better features
2. FSQ-VAE Tokenizers - encode/decode for compression and quality
3. Gamma correction (γ=0.7) - balanced brightness/accuracy

This is the "major juice and sauce" - the full power of TerraMind!

Usage:
    >>> from axs_lib.stage1_tim_inference import infer_sar_to_optical_tim
    >>> optical = infer_sar_to_optical_tim('sar.tif')

Author: Axion-Sat Project  
Version: 2.0.0 (TiM + Tokenizers)
"""

import sys
from pathlib import Path
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Global Model & Tokenizer Cache
# ============================================================================

_TERRAMIND_TIM_MODEL = None
_TOKENIZERS = {}
_DEVICE = None


def _get_model_and_tokenizers(device: Optional[torch.device] = None):
    """Load TerraMind TiM model and tokenizers (cached)."""
    global _TERRAMIND_TIM_MODEL, _TOKENIZERS, _DEVICE
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Return cached if available
    if _TERRAMIND_TIM_MODEL is not None and _DEVICE == device:
        return _TERRAMIND_TIM_MODEL, _TOKENIZERS, _DEVICE
    
    try:
        from terratorch import FULL_MODEL_REGISTRY
        
        print(f"Loading TerraMind TiM model on {device}...")
        
        # Build TiM model - uses LULC as intermediate modality
        model = FULL_MODEL_REGISTRY.build(
            'terramind_v1_large_generate',
            pretrained=True,
            modalities=['S1GRD'],
            output_modalities=['S2L2A', 'LULC'],  # Generate both optical and LULC
            timesteps=10,
            standardize=True
        )
        model = model.to(device)
        model.eval()
        
        print(f"✓ TerraMind TiM loaded")
        
        # Load tokenizers
        print(f"Loading tokenizers...")
        tokenizers = {}
        
        for modality in ['s1grd', 's2l2a']:
            tokenizer_name = f'terramind_v1_tokenizer_{modality}'
            print(f"  Loading {tokenizer_name}...")
            tokenizer = FULL_MODEL_REGISTRY.build(
                tokenizer_name,
                pretrained=True
            )
            tokenizer = tokenizer.to(device)
            tokenizer.eval()
            tokenizers[modality] = tokenizer
        
        print(f"✓ Tokenizers loaded")
        
        _TERRAMIND_TIM_MODEL = model
        _TOKENIZERS = tokenizers
        _DEVICE = device
        
        return model, tokenizers, device
        
    except Exception as e:
        raise RuntimeError(f"Failed to load TerraMind TiM: {e}")


# ============================================================================
# Input Processing (reuse from stage1_inference.py)
# ============================================================================

def _load_sar_from_file(file_path: Union[str, Path]) -> np.ndarray:
    """Load SAR from file (.tif, .npy, .npz)."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"SAR file not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix in ['.tif', '.tiff']:
        try:
            import rasterio
            with rasterio.open(file_path) as src:
                sar = src.read()
                if sar.shape[0] != 2:
                    raise ValueError(f"Expected 2 bands (VV, VH), got {sar.shape[0]}")
                return sar.astype(np.float32)
        except ImportError:
            raise ImportError("rasterio required for .tif files")
    
    elif suffix == '.npy':
        return np.load(file_path).astype(np.float32)
    
    elif suffix == '.npz':
        data = np.load(file_path)
        for vv_key in ['s1_vv', 'vv', 'VV']:
            for vh_key in ['s1_vh', 'vh', 'VH']:
                if vv_key in data and vh_key in data:
                    return np.stack([data[vv_key], data[vh_key]], axis=0).astype(np.float32)
        raise ValueError(f"Could not find VV/VH in .npz. Keys: {list(data.keys())}")
    
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def _normalize_sar_input(sar: np.ndarray, input_scale: str = 'auto') -> np.ndarray:
    """Normalize SAR to dB scale."""
    if input_scale == 'auto':
        sar_min, sar_max = sar.min(), sar.max()
        
        if sar_min >= -50 and sar_max <= 20:
            input_scale = 'db'
        elif sar_min >= 0 and sar_max <= 1:
            input_scale = 'normalized'
        else:
            input_scale = 'linear'
        
        print(f"  Auto-detected scale: {input_scale}")
    
    if input_scale == 'db':
        return sar
    elif input_scale == 'normalized':
        return sar * 30 - 25
    elif input_scale == 'linear':
        return 10 * np.log10(np.maximum(sar, 1e-10))
    else:
        raise ValueError(f"Unknown input_scale: {input_scale}")


def _preprocess_sar(
    sar: Union[np.ndarray, torch.Tensor, str, Path],
    input_scale: str = 'auto',
    device: Optional[torch.device] = None
):
    """Preprocess SAR for TerraMind."""
    # Load from file
    if isinstance(sar, (str, Path)):
        print(f"Loading SAR from: {sar}")
        sar = _load_sar_from_file(sar)
    
    # Convert to numpy
    if isinstance(sar, torch.Tensor):
        sar = sar.cpu().numpy()
    
    sar = np.asarray(sar, dtype=np.float32)
    
    # Validate shape
    if sar.ndim == 2:
        import warnings
        warnings.warn("Single-channel SAR, duplicating as VV=VH")
        sar = np.stack([sar, sar], axis=0)
    
    if sar.ndim != 3 or sar.shape[0] != 2:
        raise ValueError(f"Expected (2, H, W), got {sar.shape}")
    
    # Normalize to dB
    sar_db = _normalize_sar_input(sar, input_scale)
    
    # Store original size
    orig_h, orig_w = sar_db.shape[1], sar_db.shape[2]
    
    # Convert to torch
    sar_tensor = torch.from_numpy(sar_db).float().unsqueeze(0)
    
    if device is not None:
        sar_tensor = sar_tensor.to(device)
    
    # Pad to 224x224
    if orig_h != 224 or orig_w != 224:
        pad_h = max(0, 224 - orig_h)
        pad_w = max(0, 224 - orig_w)
        sar_tensor = nn.functional.pad(sar_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    return sar_tensor, (orig_h, orig_w)


# ============================================================================
# Main TiM + Tokenizer Inference
# ============================================================================

@torch.no_grad()
def infer_sar_to_optical_tim(
    sar: Union[np.ndarray, torch.Tensor, str, Path],
    gamma: float = 0.7,  # Updated default
    input_scale: str = 'auto',
    return_all_bands: bool = True,
    use_tokenizers: bool = True,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Convert SAR to optical using TerraMind TiM + Tokenizers.
    
    This is the FULL TerraMind pipeline with:
    - TiM (Thinking-in-Modalities): Generates LULC internally
    - FSQ-VAE Tokenizers: Encode/decode for compression  
    - Gamma correction: γ=0.7 for balanced brightness/accuracy
    
    Args:
        sar: SAR input (file path, numpy, or torch tensor)
        gamma: Gamma correction (default: 0.7)
            - 0.5-0.6: Brighter
            - 0.7: Optimal balance (recommended)
            - 0.8-1.0: More accurate but darker
        input_scale: 'auto', 'db', 'linear', 'normalized'
        return_all_bands: Return 12 bands (True) or 4 bands (False)
        use_tokenizers: Use FSQ-VAE tokenizers (default: True)
        device: Computation device
        
    Returns:
        Optical imagery (12, H, W) or (4, H, W)
    """
    # Load model and tokenizers
    model, tokenizers, device = _get_model_and_tokenizers(device)
    
    # Preprocess
    print(f"Preprocessing SAR input...")
    sar_padded, (orig_h, orig_w) = _preprocess_sar(sar, input_scale, device)
    
    print(f"  Input size: {orig_h}x{orig_w}")
    
    # Use raw SAR input directly (tokenizers are internal to TerraMind)
    sar_input = sar_padded
    
    if use_tokenizers:
        print(f"Note: TerraMind uses FSQ-VAE tokenizers internally")
    
    # Run TerraMind TiM inference
    print(f"Running TerraMind TiM inference...")
    print(f"  This internally generates LULC modality for better features")
    
    outputs = model({'S1GRD': sar_input})
    s2_padded = outputs['S2L2A']
    
    # Clamp to valid range
    s2_padded = torch.clamp(s2_padded, 0, 10000)
    
    print(f"  TiM output: range=[{s2_padded.min():.1f}, {s2_padded.max():.1f}] DN")
    
    # Optionally pass through S2 tokenizer for refinement
    if use_tokenizers:
        print(f"Refining with S2L2A FSQ-VAE tokenizer...")
        tokenizer_s2 = tokenizers['s2l2a']
        
        # CRITICAL: Tokenizer expects normalized reflectance [0, 1]
        s2_normalized = s2_padded / 10000.0
        print(f"  Normalized input: [{s2_normalized.min():.4f}, {s2_normalized.max():.4f}]")
        
        # Encode + decode for FSQ-VAE refinement
        quantized_s2, _, tokens_s2 = tokenizer_s2.encode(s2_normalized)
        print(f"  FSQ tokens: {tokens_s2.shape}, codebook size: {tokens_s2.max().item() + 1}")
        
        # Decode back to reflectance
        s2_reconstructed = tokenizer_s2.decode_tokens(tokens_s2, timesteps=10, verbose=False)
        print(f"  Decoded output: [{s2_reconstructed.min():.4f}, {s2_reconstructed.max():.4f}]")
        
        # Scale back to DN [0, 10000]
        s2_padded = s2_reconstructed * 10000.0
        s2_padded = torch.clamp(s2_padded, 0, 10000)
        
        print(f"  Final DN range: [{s2_padded.min():.1f}, {s2_padded.max():.1f}]")
    
    # Crop back to original size
    if orig_h != 224 or orig_w != 224:
        pad_h = (224 - orig_h) // 2
        pad_w = (224 - orig_w) // 2
        s2_output = s2_padded[:, :, pad_h:pad_h+orig_h, pad_w:pad_w+orig_w]
    else:
        s2_output = s2_padded
    
    # Convert to numpy
    s2_numpy = s2_output.cpu().numpy()[0]
    
    print(f"  Output size: {s2_numpy.shape[1:3]}")
    print(f"  Output range: [{s2_numpy.min():.1f}, {s2_numpy.max():.1f}] DN")
    
    # Apply gamma correction
    if gamma != 1.0:
        print(f"Applying gamma correction (γ={gamma})...")
        s2_reflectance = s2_numpy / 10000.0
        s2_corrected = np.power(s2_reflectance, gamma)
        s2_numpy = (s2_corrected * 10000.0).astype(np.float32)
        s2_numpy = np.clip(s2_numpy, 0, 10000)
    
    # Return requested bands
    if return_all_bands:
        print(f"✓ Returning all 12 S2L2A bands (TiM + Tokenizers)")
        return s2_numpy
    else:
        print(f"✓ Returning 4 bands (B02, B03, B04, B08)")
        s2_4band = s2_numpy[[1, 2, 3, 7], :, :]
        return s2_4band / 10000.0


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Stage 1 with TiM + Tokenizers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input', help='Input SAR file')
    parser.add_argument('-o', '--output', help='Output path')
    parser.add_argument('-g', '--gamma', type=float, default=0.7,
                       help='Gamma correction')
    parser.add_argument('--no-tokenizers', action='store_true',
                       help='Disable tokenizers')
    parser.add_argument('--all-bands', action='store_true',
                       help='Return all 12 bands')
    
    args = parser.parse_args()
    
    optical = infer_sar_to_optical_tim(
        args.input,
        gamma=args.gamma,
        use_tokenizers=not args.no_tokenizers,
        return_all_bands=args.all_bands
    )
    
    if args.output:
        np.save(args.output, optical)
        print(f"✓ Saved to: {args.output}")
    else:
        print(f"\n✓ Inference complete: {optical.shape}")
