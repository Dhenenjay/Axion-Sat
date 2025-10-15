"""
Stage 1: TerraMind SAR-to-Optical Translation

This module implements the first stage of the Axion-Sat pipeline, using TerraMind's
generative foundation model to translate Sentinel-1 SAR imagery to synthetic 
Sentinel-2 optical imagery.

Pipeline Flow:
    Stage 1 (this module): SAR → TerraMind → Synthetic Optical "Mental Image"
    Stage 2: TerraMind latents → Prithvi → Dense Features
    Stage 3: Features → Conditional Model → Final Segmentation

Key Concepts:
    - TerraMind generates abstract "mental images" - latent representations that
      capture cross-modal correlations without being directly visualizable
    - These latents are in learned feature space, not pixel space
    - Output requires decoding (via FSQ-VAE) to become actual optical images
    - This module handles the full decode pipeline to S2 L2A normalized format

Output Format:
    - Band order: B02, B03, B04, B08 (Blue, Green, Red, NIR)
    - Normalized to [0, 1] range for downstream processing
    - Matches Sentinel-2 L2A surface reflectance specification
    - Compatible with standard optical image processing tools

Author: Axion-Sat Project
Version: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, Union
import warnings

try:
    from axs_lib.models import build_terramind_generator
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    warnings.warn("axs_lib.models not available, some functions will be limited")

try:
    from axs_lib.stdz import TERRAMIND_S2_STATS, S2_L2A_STATS
    STDZ_AVAILABLE = True
except ImportError:
    STDZ_AVAILABLE = False
    warnings.warn("axs_lib.stdz not available, using fallback statistics")


# ============================================================================
# Sentinel-2 Band Configuration
# ============================================================================

# Sentinel-2 L2A band ordering (10m resolution bands)
S2_L2A_BAND_ORDER = ['B02', 'B03', 'B04', 'B08']  # Blue, Green, Red, NIR

# Sentinel-2 L2A reflectance range (surface reflectance)
# Standard range: 0-10000 (unitless)
# We normalize to [0, 1] for neural network processing
S2_L2A_MAX_REFLECTANCE = 10000.0

# Fallback statistics if stdz module not available
FALLBACK_S2_MEANS = np.array([1369.42, 1303.07, 1249.54, 2633.78], dtype=np.float32)
FALLBACK_S2_STDS = np.array([68.88, 133.32, 188.04, 329.09], dtype=np.float32)


# ============================================================================
# Core Translation Function
# ============================================================================

def tm_sar2opt(
    generator: nn.Module,
    s1_tensor: torch.Tensor,
    timesteps: int = 12,
    return_raw: bool = False,
    denormalize: bool = True,
    clip_range: Tuple[float, float] = (0.0, 1.0),
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """
    Apply TerraMind's full SAR-to-optical translation pipeline.
    
    This function wraps TerraMind's generative model to produce synthetic optical
    imagery from SAR input. It handles:
    1. Input formatting (SAR tensor → modality dict)
    2. TerraMind generation (produces latent "mental image")
    3. Decoding (latent → pixel space via FSQ-VAE)
    4. Denormalization (standardized values → reflectance)
    5. Normalization ([0, 10000] → [0, 1])
    
    Args:
        generator: TerraMind generator model (from build_terramind_generator)
        s1_tensor: Sentinel-1 input tensor
            Shape: (B, 2, H, W) where 2 channels = [VV, VH]
            Expected: Pre-standardized (mean-centered, std-scaled)
        timesteps: Number of diffusion timesteps
            - 12: Fast inference with good quality (default)
            - 6-8: Very fast, acceptable quality
            - 25+: High quality, slower
        return_raw: If True, return raw model output without processing
            (useful for debugging or custom post-processing)
        denormalize: If True, denormalize from standardized values to reflectance
        clip_range: Range to clip output values to (default: [0, 1])
        device: Device to run on (defaults to same as input tensor)
        
    Returns:
        Tensor of synthetic optical imagery
        Shape: (B, 4, H, W) where 4 channels = [B02, B03, B04, B08]
        Values: Normalized to [0, 1] range (if denormalize=True)
        
    Example:
        >>> from axs_lib.models import build_terramind_generator
        >>> from axs_lib.stage1_tm_s2o import tm_sar2opt
        >>> 
        >>> # Load model
        >>> generator = build_terramind_generator(
        ...     input_modalities=("S1GRD",),
        ...     output_modalities=("S2L2A",),
        ...     timesteps=12,
        ...     pretrained=True
        ... )
        >>> generator.eval()
        >>> 
        >>> # Prepare SAR input (B=1, C=2, H=256, W=256)
        >>> s1_tensor = torch.randn(1, 2, 256, 256).cuda()
        >>> 
        >>> # Generate synthetic optical
        >>> with torch.no_grad():
        ...     opt_tensor = tm_sar2opt(generator, s1_tensor, timesteps=12)
        >>> 
        >>> # Output shape: (1, 4, 256, 256), range: [0, 1]
        >>> print(opt_tensor.shape, opt_tensor.min(), opt_tensor.max())
        
    Note:
        - Input must be pre-standardized using S1 statistics (see axs_lib.stdz)
        - Output is automatically denormalized and normalized to [0, 1]
        - TerraMind internally uses FSQ-VAE decoder for latent→pixel conversion
        - Lower timesteps = faster inference but potentially lower quality
        
    Raises:
        ValueError: If input tensor has incorrect shape
        RuntimeError: If generation fails
    """
    # Validate input
    if s1_tensor.ndim != 4:
        raise ValueError(
            f"s1_tensor must be 4D (B, C, H, W), got shape {s1_tensor.shape}"
        )
    
    if s1_tensor.shape[1] != 2:
        raise ValueError(
            f"s1_tensor must have 2 channels (VV, VH), got {s1_tensor.shape[1]}"
        )
    
    # Determine device
    if device is None:
        device = s1_tensor.device
    else:
        device = torch.device(device)
        s1_tensor = s1_tensor.to(device)
    
    # Ensure generator is on correct device
    generator = generator.to(device)
    
    # Format input as modality dictionary (TerraMind expects this format)
    input_dict = {
        'S1GRD': s1_tensor
    }
    
    # Generate synthetic optical "mental image" (latent representation)
    # TerraMind will:
    # 1. Encode SAR input to latent space
    # 2. Apply diffusion-based generation
    # 3. Decode to optical latent representation
    try:
        # Call generator - try different API methods
        # TerraMind may use forward() or generate() depending on version
        if hasattr(generator, 'generate'):
            raw_output = generator.generate(
                input_dict,
                timesteps=timesteps
            )
        elif hasattr(generator, 'forward'):
            # Use forward with timesteps as keyword arg
            raw_output = generator(
                input_dict,
                num_inference_steps=timesteps
            )
        else:
            raise AttributeError(
                f"Generator has no 'generate' or 'forward' method. "
                f"Available methods: {[m for m in dir(generator) if not m.startswith('_')]}"
            )
    except Exception as e:
        raise RuntimeError(f"TerraMind generation failed: {e}")
    
    # Extract optical output
    # TerraMind may return dict with modality keys or direct tensor
    if isinstance(raw_output, dict):
        if 'S2L2A' in raw_output:
            opt_tensor = raw_output['S2L2A']
        elif 'opt_v1' in raw_output:
            opt_tensor = raw_output['opt_v1']
        else:
            # Try to find any optical-like output
            available_keys = list(raw_output.keys())
            raise ValueError(
                f"Could not find optical output in generator result. "
                f"Available keys: {available_keys}"
            )
    else:
        # Assume direct tensor output
        opt_tensor = raw_output
    
    # Debug: Check output shape and fix if needed
    if opt_tensor.ndim == 4 and opt_tensor.shape[1] != 4:
        # TerraMind might return multiple timesteps or intermediate outputs
        # Expected shape: (B, 4, H, W) for S2 bands [B02, B03, B04, B08]
        if opt_tensor.shape[1] == 12:
            # Likely 3 timesteps * 4 channels = 12 channels
            # Take the last 4 channels (final timestep)
            opt_tensor = opt_tensor[:, -4:, :, :]
        elif opt_tensor.shape[1] > 4:
            # Take first 4 channels as fallback
            import warnings
            warnings.warn(
                f"TerraMind output has {opt_tensor.shape[1]} channels, expected 4. "
                f"Taking channels 0-3 as [B02, B03, B04, B08]. "
                f"Full shape: {opt_tensor.shape}"
            )
            opt_tensor = opt_tensor[:, :4, :, :]
    
    # Debug: Check if tensor requires grad
    if opt_tensor.requires_grad:
        print(f"✓ [DEBUG] Output tensor requires_grad=True")
    else:
        print(f"⚠ [DEBUG] Output tensor requires_grad=False - gradient flow broken!")
        print(f"   Model training mode: {generator.training}")
        print(f"   Input requires_grad: {s1_tensor.requires_grad}")
    
    # Return raw output if requested (for debugging/custom processing)
    if return_raw:
        return opt_tensor
    
    # Denormalize from standardized space to reflectance space
    if denormalize:
        opt_tensor = _denormalize_s2(opt_tensor)
    
    # Normalize to [0, 1] range
    opt_tensor = _normalize_to_01(opt_tensor)
    
    # Clip to valid range
    if clip_range is not None:
        opt_tensor = torch.clamp(opt_tensor, clip_range[0], clip_range[1])
    
    return opt_tensor


def _denormalize_s2(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize Sentinel-2 tensor from standardized space to reflectance.
    
    Converts from: (value - mean) / std
    Back to: value = (tensor * std) + mean
    
    Args:
        tensor: Standardized tensor (B, 4, H, W)
        
    Returns:
        Denormalized tensor in reflectance units (0-10000 range)
    """
    # Get statistics
    if STDZ_AVAILABLE:
        means = TERRAMIND_S2_STATS.means
        stds = TERRAMIND_S2_STATS.stds
    else:
        means = FALLBACK_S2_MEANS
        stds = FALLBACK_S2_STDS
    
    # Convert to tensors on same device as input
    means_t = torch.from_numpy(np.array(means, dtype=np.float32)).to(tensor.device)
    stds_t = torch.from_numpy(np.array(stds, dtype=np.float32)).to(tensor.device)
    
    # Reshape for broadcasting: (1, C, 1, 1)
    means_t = means_t.view(1, -1, 1, 1)
    stds_t = stds_t.view(1, -1, 1, 1)
    
    # Denormalize: x = (x_norm * std) + mean
    denorm_tensor = (tensor * stds_t) + means_t
    
    return denorm_tensor


def _normalize_to_01(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize Sentinel-2 reflectance to [0, 1] range.
    
    Converts from reflectance units (0-10000) to normalized [0, 1]
    
    Args:
        tensor: Reflectance tensor (B, 4, H, W)
        
    Returns:
        Normalized tensor in [0, 1] range
    """
    return tensor / S2_L2A_MAX_REFLECTANCE


def _unnormalize_from_01(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert from [0, 1] back to reflectance units.
    
    Inverse of _normalize_to_01. Useful for saving outputs in standard format.
    
    Args:
        tensor: Normalized tensor (B, 4, H, W) in [0, 1]
        
    Returns:
        Reflectance tensor (0-10000 range)
    """
    return tensor * S2_L2A_MAX_REFLECTANCE


# ============================================================================
# Batch Processing
# ============================================================================

def tm_sar2opt_batch(
    generator: nn.Module,
    s1_batch: torch.Tensor,
    timesteps: int = 12,
    batch_size: Optional[int] = None,
    show_progress: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    Process large batches with automatic sub-batching.
    
    For memory-constrained environments, this splits large batches into
    smaller chunks and processes them sequentially.
    
    Args:
        generator: TerraMind generator model
        s1_batch: Large SAR batch (B, 2, H, W)
        timesteps: Number of diffusion timesteps
        batch_size: Sub-batch size (None = process all at once)
        show_progress: Show progress bar (requires tqdm)
        **kwargs: Additional arguments passed to tm_sar2opt
        
    Returns:
        Synthetic optical batch (B, 4, H, W)
        
    Example:
        >>> # Process 100 samples in batches of 8
        >>> s1_large = torch.randn(100, 2, 256, 256).cuda()
        >>> opt_large = tm_sar2opt_batch(
        ...     generator, 
        ...     s1_large, 
        ...     batch_size=8,
        ...     show_progress=True
        ... )
    """
    total_samples = s1_batch.shape[0]
    
    # If no batch size specified or batch fits in memory, process all at once
    if batch_size is None or total_samples <= batch_size:
        return tm_sar2opt(generator, s1_batch, timesteps=timesteps, **kwargs)
    
    # Process in sub-batches
    results = []
    
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(
                range(0, total_samples, batch_size),
                desc="Processing batches"
            )
        except ImportError:
            iterator = range(0, total_samples, batch_size)
            print(f"Processing {total_samples} samples in batches of {batch_size}...")
    else:
        iterator = range(0, total_samples, batch_size)
    
    for i in iterator:
        batch_end = min(i + batch_size, total_samples)
        sub_batch = s1_batch[i:batch_end]
        
        with torch.no_grad():
            result = tm_sar2opt(generator, sub_batch, timesteps=timesteps, **kwargs)
        
        results.append(result)
    
    # Concatenate all results
    return torch.cat(results, dim=0)


# ============================================================================
# Utility Functions
# ============================================================================

def get_s2_band_info() -> Dict[str, Dict[str, Union[str, int, float]]]:
    """
    Get information about Sentinel-2 L2A bands.
    
    Returns:
        Dictionary with band metadata
        
    Example:
        >>> info = get_s2_band_info()
        >>> print(info['B04']['wavelength_nm'])
        665
    """
    return {
        'B02': {
            'name': 'Blue',
            'wavelength_nm': 490,
            'resolution_m': 10,
            'index': 0
        },
        'B03': {
            'name': 'Green',
            'wavelength_nm': 560,
            'resolution_m': 10,
            'index': 1
        },
        'B04': {
            'name': 'Red',
            'wavelength_nm': 665,
            'resolution_m': 10,
            'index': 2
        },
        'B08': {
            'name': 'NIR',
            'wavelength_nm': 842,
            'resolution_m': 10,
            'index': 3
        }
    }


def extract_rgb(opt_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract RGB channels from S2 L2A tensor.
    
    Args:
        opt_tensor: S2 L2A tensor (B, 4, H, W) in band order [B02, B03, B04, B08]
        
    Returns:
        RGB tensor (B, 3, H, W) in order [R, G, B] ready for visualization
        
    Example:
        >>> opt = tm_sar2opt(generator, s1_tensor)
        >>> rgb = extract_rgb(opt)  # Shape: (B, 3, H, W)
        >>> # Can now save as image or display
    """
    # Band order: [B02=Blue, B03=Green, B04=Red, B08=NIR]
    # RGB order: [Red, Green, Blue]
    # Extract indices: [2, 1, 0] for [Red, Green, Blue]
    
    blue = opt_tensor[:, 0:1, :, :]   # B02
    green = opt_tensor[:, 1:2, :, :]  # B03
    red = opt_tensor[:, 2:3, :, :]    # B04
    
    # Stack in RGB order
    rgb = torch.cat([red, green, blue], dim=1)
    
    return rgb


def compute_ndvi(opt_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute NDVI from synthetic optical imagery.
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        opt_tensor: S2 L2A tensor (B, 4, H, W)
        
    Returns:
        NDVI tensor (B, 1, H, W) in range [-1, 1]
        
    Example:
        >>> opt = tm_sar2opt(generator, s1_tensor)
        >>> ndvi = compute_ndvi(opt)
    """
    # Extract bands
    red = opt_tensor[:, 2:3, :, :]   # B04
    nir = opt_tensor[:, 3:4, :, :]   # B08
    
    # Compute NDVI with numerical stability
    numerator = nir - red
    denominator = nir + red + 1e-8  # Add small epsilon to avoid division by zero
    
    ndvi = numerator / denominator
    
    return ndvi


# ============================================================================
# Example / Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Stage 1: TerraMind SAR-to-Optical Translation")
    print("=" * 80)
    print()
    
    if not MODELS_AVAILABLE:
        print("⚠ axs_lib.models not available - cannot run full example")
        print("This is a demonstration of the module structure and API.")
        print()
        print("To use this module:")
        print("1. Install TerraTorch: pip install terratorch")
        print("2. Register models: python register_models.py")
        print("3. Import and use:")
        print()
        print(">>> from axs_lib.models import build_terramind_generator")
        print(">>> from axs_lib.stage1_tm_s2o import tm_sar2opt")
        print(">>> generator = build_terramind_generator(...)")
        print(">>> opt = tm_sar2opt(generator, s1_tensor, timesteps=12)")
    else:
        print("✓ Dependencies available")
        print()
        
        # Show band information
        print("Sentinel-2 L2A Band Information:")
        print("-" * 80)
        band_info = get_s2_band_info()
        for band_id, info in band_info.items():
            print(f"{band_id:4s} | {info['name']:6s} | "
                  f"{info['wavelength_nm']:4d} nm | "
                  f"{info['resolution_m']:2d}m resolution")
        print()
        
        # Demonstrate API usage
        print("Example Usage:")
        print("-" * 80)
        print("""
# Load TerraMind generator
from axs_lib.models import build_terramind_generator
from axs_lib.stage1_tm_s2o import tm_sar2opt, extract_rgb, compute_ndvi

generator = build_terramind_generator(
    input_modalities=("S1GRD",),
    output_modalities=("S2L2A",),
    timesteps=12,
    pretrained=True
)
generator.eval()

# Prepare SAR input (pre-standardized)
s1_tensor = torch.randn(1, 2, 256, 256).cuda()  # [VV, VH]

# Generate synthetic optical
with torch.no_grad():
    opt_tensor = tm_sar2opt(generator, s1_tensor, timesteps=12)

# Output: (1, 4, 256, 256) in [0, 1] range
# Bands: [B02, B03, B04, B08]

# Extract RGB for visualization
rgb = extract_rgb(opt_tensor)  # (1, 3, 256, 256) [R, G, B]

# Compute vegetation index
ndvi = compute_ndvi(opt_tensor)  # (1, 1, 256, 256) [-1, 1]
        """)
        
        print()
        print("=" * 80)
        print("Module loaded successfully!")
        print("=" * 80)
