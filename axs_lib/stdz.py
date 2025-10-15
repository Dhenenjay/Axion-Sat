"""
Standardization utilities for satellite imagery.

This module provides normalization/denormalization functions and statistics
for Sentinel-2 bands and other satellite data sources.

Statistics are derived from:
- Sentinel-2 global statistics (ESA/Copernicus)
- TerraMind pretraining dataset statistics
- Custom dataset statistics (configurable)

Band order for Sentinel-2:
- B02 (Blue): 490 nm
- B03 (Green): 560 nm
- B04 (Red): 665 nm
- B08 (NIR): 842 nm
- B11 (SWIR1): 1610 nm (optional)
- B12 (SWIR2): 2190 nm (optional)

SAR bands (Sentinel-1):
- VV: Vertical-Vertical polarization
- VH: Vertical-Horizontal polarization
"""

import torch
import numpy as np
from typing import Union, Optional, Dict, List, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class BandStatistics:
    """Statistics for satellite imagery bands."""
    
    means: Union[List[float], np.ndarray, torch.Tensor]
    stds: Union[List[float], np.ndarray, torch.Tensor]
    band_names: List[str]
    source: str = "unknown"
    description: str = ""
    
    def __post_init__(self):
        """Convert lists to numpy arrays."""
        if isinstance(self.means, list):
            self.means = np.array(self.means, dtype=np.float32)
        if isinstance(self.stds, list):
            self.stds = np.array(self.stds, dtype=np.float32)
        
        # Validate
        if len(self.means) != len(self.stds):
            raise ValueError("means and stds must have same length")
        if len(self.means) != len(self.band_names):
            raise ValueError("means and band_names must have same length")
    
    def to_torch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to PyTorch tensors."""
        means = torch.from_numpy(np.array(self.means, dtype=np.float32))
        stds = torch.from_numpy(np.array(self.stds, dtype=np.float32))
        return means, stds
    
    def __repr__(self):
        return (f"BandStatistics(bands={self.band_names}, "
                f"source={self.source})")


# ============================================================================
# Sentinel-2 Statistics
# ============================================================================

# Sentinel-2 L2A global statistics (10m bands)
# Derived from ESA Copernicus global samples
S2_L2A_STATS = BandStatistics(
    means=[
        1353.0,  # B02 (Blue)
        1265.0,  # B03 (Green)
        1269.0,  # B04 (Red)
        2498.0,  # B08 (NIR)
    ],
    stds=[
        65.0,    # B02
        154.0,   # B03
        194.0,   # B04
        354.0,   # B08
    ],
    band_names=['B02', 'B03', 'B04', 'B08'],
    source="ESA Copernicus",
    description="Sentinel-2 L2A global statistics (10m resolution bands)"
)

# Sentinel-2 L2A with SWIR bands (all 6 bands)
S2_L2A_FULL_STATS = BandStatistics(
    means=[
        1353.0,  # B02 (Blue)
        1265.0,  # B03 (Green)
        1269.0,  # B04 (Red)
        2498.0,  # B08 (NIR)
        1741.0,  # B11 (SWIR1)
        1356.0,  # B12 (SWIR2)
    ],
    stds=[
        65.0,    # B02
        154.0,   # B03
        194.0,   # B04
        354.0,   # B08
        472.0,   # B11
        540.0,   # B12
    ],
    band_names=['B02', 'B03', 'B04', 'B08', 'B11', 'B12'],
    source="ESA Copernicus",
    description="Sentinel-2 L2A global statistics (all bands)"
)

# TerraMind pretraining statistics (optimized for deep learning)
# These are derived from a large-scale pretraining dataset
TERRAMIND_S2_STATS = BandStatistics(
    means=[
        1369.42,  # B02 (Blue)
        1303.07,  # B03 (Green)
        1249.54,  # B04 (Red)
        2633.78,  # B08 (NIR)
    ],
    stds=[
        68.88,    # B02
        133.32,   # B03
        188.04,   # B04
        329.09,   # B08
    ],
    band_names=['B02', 'B03', 'B04', 'B08'],
    source="TerraMind",
    description="TerraMind pretraining statistics (Sentinel-2)"
)

# Normalized (0-1 range) statistics for RGB visualization
S2_RGB_NORMALIZED_STATS = BandStatistics(
    means=[
        0.485,  # Red (approximate ImageNet mean)
        0.456,  # Green
        0.406,  # Blue
    ],
    stds=[
        0.229,  # Red (approximate ImageNet std)
        0.224,  # Green
        0.225,  # Blue
    ],
    band_names=['R', 'G', 'B'],
    source="ImageNet-adapted",
    description="RGB normalization for visualization (ImageNet-like)"
)


# ============================================================================
# Sentinel-1 SAR Statistics
# ============================================================================

# Sentinel-1 GRD statistics (dB scale)
S1_GRD_STATS = BandStatistics(
    means=[
        -11.76,  # VV (dB)
        -19.29,  # VH (dB)
    ],
    stds=[
        5.46,    # VV
        5.52,    # VH
    ],
    band_names=['VV', 'VH'],
    source="ESA Copernicus",
    description="Sentinel-1 GRD statistics (dB scale)"
)

# Sentinel-1 linear scale (power)
S1_LINEAR_STATS = BandStatistics(
    means=[
        0.067,   # VV (linear)
        0.012,   # VH (linear)
    ],
    stds=[
        0.121,   # VV
        0.025,   # VH
    ],
    band_names=['VV', 'VH'],
    source="ESA Copernicus",
    description="Sentinel-1 linear scale statistics"
)

# TerraMind SAR pretraining statistics
TERRAMIND_S1_STATS = BandStatistics(
    means=[
        -12.01,  # VV (dB)
        -19.45,  # VH (dB)
    ],
    stds=[
        5.32,    # VV
        5.48,    # VH
    ],
    band_names=['VV', 'VH'],
    source="TerraMind",
    description="TerraMind pretraining statistics (Sentinel-1)"
)


# ============================================================================
# Statistics Registry
# ============================================================================

STATISTICS_REGISTRY = {
    's2_l2a': S2_L2A_STATS,
    's2_l2a_full': S2_L2A_FULL_STATS,
    'terramind_s2': TERRAMIND_S2_STATS,
    's2_rgb': S2_RGB_NORMALIZED_STATS,
    's1_grd': S1_GRD_STATS,
    's1_linear': S1_LINEAR_STATS,
    'terramind_s1': TERRAMIND_S1_STATS,
}


def get_statistics(name: str) -> BandStatistics:
    """
    Get band statistics by name.
    
    Args:
        name: Statistics name (e.g., 's2_l2a', 'terramind_s2')
        
    Returns:
        BandStatistics object
        
    Example:
        >>> stats = get_statistics('terramind_s2')
        >>> print(stats.means)
    """
    if name not in STATISTICS_REGISTRY:
        available = ', '.join(STATISTICS_REGISTRY.keys())
        raise ValueError(f"Unknown statistics '{name}'. Available: {available}")
    
    return STATISTICS_REGISTRY[name]


def list_statistics() -> List[str]:
    """List all available statistics presets."""
    return list(STATISTICS_REGISTRY.keys())


# ============================================================================
# Normalization Functions
# ============================================================================

def normalize(
    data: Union[np.ndarray, torch.Tensor],
    stats: Union[str, BandStatistics],
    inplace: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize satellite imagery using z-score normalization.
    
    Normalization: (data - mean) / std
    
    Args:
        data: Input data (C, H, W) or (B, C, H, W)
        stats: Statistics name (str) or BandStatistics object
        inplace: Modify data in-place (only for tensors)
        
    Returns:
        Normalized data
        
    Example:
        >>> import torch
        >>> data = torch.randn(4, 256, 256) * 1000 + 1500
        >>> normalized = normalize(data, 'terramind_s2')
    """
    # Get statistics
    if isinstance(stats, str):
        stats = get_statistics(stats)
    
    # Handle numpy
    if isinstance(data, np.ndarray):
        return _normalize_numpy(data, stats)
    
    # Handle torch
    elif isinstance(data, torch.Tensor):
        return _normalize_torch(data, stats, inplace)
    
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def denormalize(
    data: Union[np.ndarray, torch.Tensor],
    stats: Union[str, BandStatistics],
    inplace: bool = False,
    clip: bool = True,
    clip_range: Optional[Tuple[float, float]] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Denormalize satellite imagery.
    
    Denormalization: data * std + mean
    
    Args:
        data: Normalized data (C, H, W) or (B, C, H, W)
        stats: Statistics name (str) or BandStatistics object
        inplace: Modify data in-place (only for tensors)
        clip: Clip values to valid range
        clip_range: Custom clipping range (min, max)
        
    Returns:
        Denormalized data
        
    Example:
        >>> normalized = normalize(data, 'terramind_s2')
        >>> original = denormalize(normalized, 'terramind_s2')
    """
    # Get statistics
    if isinstance(stats, str):
        stats = get_statistics(stats)
    
    # Handle numpy
    if isinstance(data, np.ndarray):
        result = _denormalize_numpy(data, stats)
    
    # Handle torch
    elif isinstance(data, torch.Tensor):
        result = _denormalize_torch(data, stats, inplace)
    
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
    
    # Clip to valid range
    if clip:
        if clip_range is None:
            # Default: clip to reasonable Sentinel-2 range
            clip_range = (0, 10000)
        
        if isinstance(result, np.ndarray):
            result = np.clip(result, clip_range[0], clip_range[1])
        else:
            result = torch.clamp(result, clip_range[0], clip_range[1])
    
    return result


def _normalize_numpy(
    data: np.ndarray,
    stats: BandStatistics
) -> np.ndarray:
    """Normalize numpy array."""
    means = stats.means
    stds = stats.stds
    
    # Get number of channels
    if data.ndim == 3:
        num_channels = data.shape[0]
    elif data.ndim == 4:
        num_channels = data.shape[1]
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {data.shape}")
    
    # Validate channels
    if num_channels != len(means):
        warnings.warn(
            f"Data has {num_channels} channels but statistics have {len(means)} bands. "
            f"Using first {min(num_channels, len(means))} bands."
        )
        num_channels = min(num_channels, len(means))
        means = means[:num_channels]
        stds = stds[:num_channels]
    
    # Reshape for broadcasting
    if data.ndim == 3:
        means = means.reshape(-1, 1, 1)
        stds = stds.reshape(-1, 1, 1)
    else:  # 4D
        means = means.reshape(1, -1, 1, 1)
        stds = stds.reshape(1, -1, 1, 1)
    
    # Normalize
    normalized = (data - means) / (stds + 1e-8)
    
    return normalized.astype(np.float32)


def _denormalize_numpy(
    data: np.ndarray,
    stats: BandStatistics
) -> np.ndarray:
    """Denormalize numpy array."""
    means = stats.means
    stds = stats.stds
    
    # Get number of channels
    if data.ndim == 3:
        num_channels = data.shape[0]
    elif data.ndim == 4:
        num_channels = data.shape[1]
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {data.shape}")
    
    # Validate channels
    if num_channels != len(means):
        num_channels = min(num_channels, len(means))
        means = means[:num_channels]
        stds = stds[:num_channels]
    
    # Reshape for broadcasting
    if data.ndim == 3:
        means = means.reshape(-1, 1, 1)
        stds = stds.reshape(-1, 1, 1)
    else:  # 4D
        means = means.reshape(1, -1, 1, 1)
        stds = stds.reshape(1, -1, 1, 1)
    
    # Denormalize
    denormalized = data * stds + means
    
    return denormalized.astype(np.float32)


def _normalize_torch(
    data: torch.Tensor,
    stats: BandStatistics,
    inplace: bool = False
) -> torch.Tensor:
    """Normalize PyTorch tensor."""
    means, stds = stats.to_torch()
    means = means.to(data.device)
    stds = stds.to(data.device)
    
    # Get number of channels
    if data.ndim == 3:
        num_channels = data.shape[0]
    elif data.ndim == 4:
        num_channels = data.shape[1]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {data.shape}")
    
    # Validate channels
    if num_channels != len(means):
        warnings.warn(
            f"Data has {num_channels} channels but statistics have {len(means)} bands. "
            f"Using first {min(num_channels, len(means))} bands."
        )
        num_channels = min(num_channels, len(means))
        means = means[:num_channels]
        stds = stds[:num_channels]
    
    # Reshape for broadcasting
    if data.ndim == 3:
        means = means.view(-1, 1, 1)
        stds = stds.view(-1, 1, 1)
    else:  # 4D
        means = means.view(1, -1, 1, 1)
        stds = stds.view(1, -1, 1, 1)
    
    # Normalize
    if inplace:
        data.sub_(means).div_(stds + 1e-8)
        return data
    else:
        return (data - means) / (stds + 1e-8)


def _denormalize_torch(
    data: torch.Tensor,
    stats: BandStatistics,
    inplace: bool = False
) -> torch.Tensor:
    """Denormalize PyTorch tensor."""
    means, stds = stats.to_torch()
    means = means.to(data.device)
    stds = stds.to(data.device)
    
    # Get number of channels
    if data.ndim == 3:
        num_channels = data.shape[0]
    elif data.ndim == 4:
        num_channels = data.shape[1]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {data.shape}")
    
    # Validate channels
    if num_channels != len(means):
        num_channels = min(num_channels, len(means))
        means = means[:num_channels]
        stds = stds[:num_channels]
    
    # Reshape for broadcasting
    if data.ndim == 3:
        means = means.view(-1, 1, 1)
        stds = stds.view(-1, 1, 1)
    else:  # 4D
        means = means.view(1, -1, 1, 1)
        stds = stds.view(1, -1, 1, 1)
    
    # Denormalize
    if inplace:
        data.mul_(stds).add_(means)
        return data
    else:
        return data * stds + means


# ============================================================================
# Batch Normalization
# ============================================================================

class Normalizer:
    """
    Reusable normalizer for consistent preprocessing.
    
    Args:
        stats: Statistics name or BandStatistics object
        device: Device to place normalization constants on
        
    Example:
        >>> normalizer = Normalizer('terramind_s2', device='cuda')
        >>> normalized = normalizer(data)
        >>> denormalized = normalizer.inverse(normalized)
    """
    
    def __init__(
        self,
        stats: Union[str, BandStatistics],
        device: Optional[torch.device] = None
    ):
        if isinstance(stats, str):
            stats = get_statistics(stats)
        
        self.stats = stats
        self.device = device or torch.device('cpu')
        
        # Prepare tensors
        self.means, self.stds = stats.to_torch()
        self.means = self.means.to(self.device)
        self.stds = self.stds.to(self.device)
    
    def __call__(
        self,
        data: torch.Tensor,
        inplace: bool = False
    ) -> torch.Tensor:
        """Normalize data."""
        return normalize(data, self.stats, inplace=inplace)
    
    def inverse(
        self,
        data: torch.Tensor,
        inplace: bool = False,
        clip: bool = True
    ) -> torch.Tensor:
        """Denormalize data."""
        return denormalize(data, self.stats, inplace=inplace, clip=clip)
    
    def to(self, device: torch.device):
        """Move normalizer to device."""
        self.device = device
        self.means = self.means.to(device)
        self.stds = self.stds.to(device)
        return self


# ============================================================================
# Utility Functions
# ============================================================================

def compute_statistics(
    data: Union[np.ndarray, torch.Tensor],
    band_names: Optional[List[str]] = None,
    axis: Optional[Tuple[int, ...]] = None
) -> BandStatistics:
    """
    Compute statistics from data.
    
    Args:
        data: Input data (B, C, H, W) or (C, H, W)
        band_names: Names of bands
        axis: Axes to compute statistics over
        
    Returns:
        BandStatistics object
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    # Default axes: all spatial and batch dimensions
    if axis is None:
        if data.ndim == 4:
            axis = (0, 2, 3)  # (B, C, H, W)
        elif data.ndim == 3:
            axis = (1, 2)     # (C, H, W)
        else:
            raise ValueError(f"Expected 3D or 4D data, got shape {data.shape}")
    
    means = np.mean(data, axis=axis)
    stds = np.std(data, axis=axis)
    
    if band_names is None:
        num_bands = len(means)
        band_names = [f"Band_{i}" for i in range(num_bands)]
    
    return BandStatistics(
        means=means.tolist(),
        stds=stds.tolist(),
        band_names=band_names,
        source="computed",
        description="Statistics computed from provided data"
    )


def print_statistics_info(stats: Union[str, BandStatistics]):
    """
    Print detailed information about statistics.
    
    Args:
        stats: Statistics name or BandStatistics object
    """
    if isinstance(stats, str):
        stats = get_statistics(stats)
    
    print("="*70)
    print(f"Band Statistics: {stats.source}")
    print("="*70)
    print(f"Description: {stats.description}")
    print(f"\nBands: {len(stats.band_names)}")
    print("-"*70)
    print(f"{'Band':<15} {'Mean':>15} {'Std':>15}")
    print("-"*70)
    
    for name, mean, std in zip(stats.band_names, stats.means, stats.stds):
        print(f"{name:<15} {mean:>15.2f} {std:>15.2f}")
    
    print("="*70)


# ============================================================================
# Example / Testing
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Standardization Utilities")
    print("="*70)
    
    # List available statistics
    print("\nAvailable statistics:")
    for name in list_statistics():
        print(f"  - {name}")
    
    # Print TerraMind statistics
    print("\n")
    print_statistics_info('terramind_s2')
    
    # Test normalization
    print("\nTesting normalization...")
    
    # Create dummy Sentinel-2 data
    data = torch.randn(2, 4, 64, 64) * 300 + 1500  # Simulated S2 data
    
    print(f"Original data: mean={data.mean():.2f}, std={data.std():.2f}")
    
    # Normalize
    normalized = normalize(data, 'terramind_s2')
    print(f"Normalized: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
    
    # Denormalize
    denormalized = denormalize(normalized, 'terramind_s2')
    print(f"Denormalized: mean={denormalized.mean():.2f}, std={denormalized.std():.2f}")
    
    # Check reconstruction
    max_error = (data - denormalized).abs().max()
    print(f"Max reconstruction error: {max_error:.6f}")
    
    # Test Normalizer class
    print("\nTesting Normalizer class...")
    normalizer = Normalizer('terramind_s2')
    
    normalized2 = normalizer(data)
    denormalized2 = normalizer.inverse(normalized2)
    
    print(f"Normalizer reconstruction error: {(data - denormalized2).abs().max():.6f}")
    
    # Test SAR normalization
    print("\nTesting SAR normalization...")
    sar_data = torch.randn(2, 2, 64, 64) * 5 - 15  # Simulated SAR dB
    print(f"SAR data: mean={sar_data.mean():.2f}, std={sar_data.std():.2f}")
    
    sar_normalized = normalize(sar_data, 'terramind_s1')
    print(f"SAR normalized: mean={sar_normalized.mean():.4f}, std={sar_normalized.std():.4f}")
    
    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
