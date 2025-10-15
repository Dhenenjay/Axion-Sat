"""
Spectral indices computation for satellite imagery.

This module provides functions to compute vegetation indices and spectral metrics
from Sentinel-2 optical bands:
- B02 (Blue): 490 nm
- B03 (Green): 560 nm
- B04 (Red): 665 nm
- B08 (NIR): 842 nm

Indices:
- NDVI (Normalized Difference Vegetation Index)
- EVI (Enhanced Vegetation Index)
- SAVI (Soil-Adjusted Vegetation Index)
- SAM (Spectral Angle Mapper)
"""

import torch
import torch.nn.functional as F
from typing import Union, Tuple, Optional
import numpy as np


def extract_bands(
    optical: Union[torch.Tensor, np.ndarray],
    band_order: str = "B02_B03_B04_B08"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract Blue, Green, Red, and NIR bands from optical imagery.
    
    Args:
        optical: Optical imagery tensor/array (B, C, H, W) or (C, H, W)
        band_order: Order of bands in the input. Options:
            - "B02_B03_B04_B08": Blue, Green, Red, NIR (default)
            - "RGB_NIR": Red, Green, Blue, NIR
            - "RGBNIR": Red, Green, Blue, NIR (same as RGB_NIR)
    
    Returns:
        Tuple of (blue, green, red, nir) tensors, each with shape (B, 1, H, W) or (1, H, W)
    """
    # Convert to torch if numpy
    if isinstance(optical, np.ndarray):
        optical = torch.from_numpy(optical)
    
    # Add batch dimension if needed
    squeeze_batch = False
    if optical.ndim == 3:
        optical = optical.unsqueeze(0)
        squeeze_batch = True
    
    if optical.shape[1] < 4:
        raise ValueError(
            f"Expected at least 4 channels (B, G, R, NIR), got {optical.shape[1]}"
        )
    
    # Extract bands based on order
    if band_order == "B02_B03_B04_B08":
        blue = optical[:, 0:1, :, :]
        green = optical[:, 1:2, :, :]
        red = optical[:, 2:3, :, :]
        nir = optical[:, 3:4, :, :]
    elif band_order in ["RGB_NIR", "RGBNIR"]:
        red = optical[:, 0:1, :, :]
        green = optical[:, 1:2, :, :]
        blue = optical[:, 2:3, :, :]
        nir = optical[:, 3:4, :, :]
    else:
        raise ValueError(
            f"Unknown band_order: {band_order}. "
            f"Valid options: 'B02_B03_B04_B08', 'RGB_NIR', 'RGBNIR'"
        )
    
    if squeeze_batch:
        blue = blue.squeeze(0)
        green = green.squeeze(0)
        red = red.squeeze(0)
        nir = nir.squeeze(0)
    
    return blue, green, red, nir


def compute_ndvi(
    optical: Union[torch.Tensor, np.ndarray],
    band_order: str = "B02_B03_B04_B08",
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute NDVI (Normalized Difference Vegetation Index).
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    NDVI is sensitive to chlorophyll content and is widely used for vegetation
    monitoring. Values range from -1 to 1, with higher values indicating
    healthier vegetation.
    
    Typical ranges:
    - Dense vegetation: 0.6 to 0.9
    - Sparse vegetation: 0.2 to 0.5
    - Non-vegetation: -0.1 to 0.2
    - Water/clouds: < -0.1
    
    Args:
        optical: Optical imagery (B, C, H, W) or (C, H, W) with at least 4 bands
        band_order: Band ordering in input (see extract_bands)
        eps: Small epsilon to avoid division by zero
    
    Returns:
        NDVI tensor with shape (B, 1, H, W) or (1, H, W)
    """
    _, _, red, nir = extract_bands(optical, band_order)
    
    ndvi = (nir - red) / (nir + red + eps)
    
    return ndvi


def compute_evi(
    optical: Union[torch.Tensor, np.ndarray],
    band_order: str = "B02_B03_B04_B08",
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute EVI (Enhanced Vegetation Index).
    
    EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)
    
    EVI was developed to optimize the vegetation signal with improved sensitivity
    in high biomass regions and improved vegetation monitoring through a
    de-coupling of the canopy background signal and a reduction in atmosphere
    influences.
    
    Standard coefficients (MODIS):
    - G = 2.5 (gain factor)
    - C1 = 6.0 (red correction coefficient)
    - C2 = 7.5 (blue correction coefficient)
    - L = 1.0 (canopy background adjustment)
    
    Args:
        optical: Optical imagery (B, C, H, W) or (C, H, W) with at least 4 bands
        band_order: Band ordering in input (see extract_bands)
        G: Gain factor
        C1: Red correction coefficient
        C2: Blue correction coefficient
        L: Canopy background adjustment
        eps: Small epsilon to avoid division by zero
    
    Returns:
        EVI tensor with shape (B, 1, H, W) or (1, H, W)
    """
    blue, _, red, nir = extract_bands(optical, band_order)
    
    evi = G * (nir - red) / (nir + C1 * red - C2 * blue + L + eps)
    
    return evi


def compute_savi(
    optical: Union[torch.Tensor, np.ndarray],
    band_order: str = "B02_B03_B04_B08",
    L: float = 0.5,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute SAVI (Soil-Adjusted Vegetation Index).
    
    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    
    SAVI is a vegetation index that attempts to minimize soil brightness
    influences using a soil-brightness correction factor (L).
    
    L factor guidelines:
    - L = 0.0: Very high vegetation cover (equivalent to NDVI)
    - L = 0.5: Intermediate vegetation cover (default)
    - L = 1.0: Low vegetation cover
    
    Typical ranges:
    - Dense vegetation: 0.6 to 0.8
    - Sparse vegetation: 0.2 to 0.5
    - Bare soil: -0.1 to 0.2
    
    Args:
        optical: Optical imagery (B, C, H, W) or (C, H, W) with at least 4 bands
        band_order: Band ordering in input (see extract_bands)
        L: Soil brightness correction factor (0 to 1)
        eps: Small epsilon to avoid division by zero
    
    Returns:
        SAVI tensor with shape (B, 1, H, W) or (1, H, W)
    """
    _, _, red, nir = extract_bands(optical, band_order)
    
    savi = ((nir - red) / (nir + red + L + eps)) * (1.0 + L)
    
    return savi


def compute_spectral_angle(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    reduction: str = "mean",
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute Spectral Angle Mapper (SAM) between predicted and target spectra.
    
    SAM measures the spectral similarity between two spectra by calculating the
    angle between them when treated as vectors. It is insensitive to illumination
    variations and scaling differences.
    
    Lower angles indicate higher similarity:
    - 0 radians (0°): Identical spectra
    - π/2 radians (90°): Orthogonal spectra
    - π radians (180°): Opposite spectra
    
    Args:
        pred: Predicted optical imagery (B, C, H, W) or (C, H, W)
        target: Target optical imagery (B, C, H, W) or (C, H, W)
        reduction: Reduction method ('mean', 'sum', 'none')
        eps: Small epsilon for numerical stability
    
    Returns:
        Spectral angle(s) in radians:
        - If reduction='mean': scalar tensor (average angle)
        - If reduction='sum': scalar tensor (sum of angles)
        - If reduction='none': tensor with shape (B, H, W) or (H, W)
    """
    # Convert to torch if numpy
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    
    # Add batch dimension if needed
    squeeze_batch = False
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        squeeze_batch = True
    
    # Ensure shapes match
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape. "
            f"Got pred: {pred.shape}, target: {target.shape}"
        )
    
    # Compute dot product between spectral vectors
    # pred: (B, C, H, W), target: (B, C, H, W)
    dot_product = (pred * target).sum(dim=1)  # (B, H, W)
    
    # Compute magnitudes
    pred_magnitude = torch.sqrt((pred ** 2).sum(dim=1) + eps)  # (B, H, W)
    target_magnitude = torch.sqrt((target ** 2).sum(dim=1) + eps)  # (B, H, W)
    
    # Compute cosine similarity
    cos_angle = dot_product / (pred_magnitude * target_magnitude + eps)
    
    # Clamp to valid range for arccos
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    
    # Compute angle in radians
    angle = torch.acos(cos_angle)  # (B, H, W)
    
    if squeeze_batch:
        angle = angle.squeeze(0)
    
    # Apply reduction
    if reduction == "mean":
        return angle.mean()
    elif reduction == "sum":
        return angle.sum()
    elif reduction == "none":
        return angle
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Valid: 'mean', 'sum', 'none'")


def compute_spectral_angle_degrees(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    reduction: str = "mean",
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute Spectral Angle Mapper in degrees (convenience wrapper).
    
    Args:
        pred: Predicted optical imagery (B, C, H, W) or (C, H, W)
        target: Target optical imagery (B, C, H, W) or (C, H, W)
        reduction: Reduction method ('mean', 'sum', 'none')
        eps: Small epsilon for numerical stability
    
    Returns:
        Spectral angle(s) in degrees
    """
    angle_rad = compute_spectral_angle(pred, target, reduction=reduction, eps=eps)
    return torch.rad2deg(angle_rad)


def compute_all_indices(
    optical: Union[torch.Tensor, np.ndarray],
    band_order: str = "B02_B03_B04_B08",
    indices: Tuple[str, ...] = ("ndvi", "evi", "savi"),
    savi_L: float = 0.5
) -> dict:
    """
    Compute multiple spectral indices at once.
    
    Args:
        optical: Optical imagery (B, C, H, W) or (C, H, W) with at least 4 bands
        band_order: Band ordering in input (see extract_bands)
        indices: Tuple of indices to compute ('ndvi', 'evi', 'savi')
        savi_L: L parameter for SAVI computation
    
    Returns:
        Dictionary mapping index name to computed tensor
    """
    result = {}
    
    valid_indices = {"ndvi", "evi", "savi"}
    for idx in indices:
        if idx not in valid_indices:
            raise ValueError(f"Unknown index: {idx}. Valid: {valid_indices}")
    
    if "ndvi" in indices:
        result["ndvi"] = compute_ndvi(optical, band_order)
    
    if "evi" in indices:
        result["evi"] = compute_evi(optical, band_order)
    
    if "savi" in indices:
        result["savi"] = compute_savi(optical, band_order, L=savi_L)
    
    return result


def visualize_index(
    index: Union[torch.Tensor, np.ndarray],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colormap: str = "RdYlGn"
) -> np.ndarray:
    """
    Normalize spectral index for visualization.
    
    Args:
        index: Spectral index tensor (B, 1, H, W), (1, H, W), or (H, W)
        vmin: Minimum value for normalization (default: index.min())
        vmax: Maximum value for normalization (default: index.max())
        colormap: Colormap name (for reference, not applied)
    
    Returns:
        Normalized index as numpy array with values in [0, 1]
    """
    # Convert to numpy if torch
    if isinstance(index, torch.Tensor):
        index = index.detach().cpu().numpy()
    
    # Squeeze to (H, W)
    while index.ndim > 2:
        index = index.squeeze(0)
    
    # Determine value range
    if vmin is None:
        vmin = index.min()
    if vmax is None:
        vmax = index.max()
    
    # Normalize to [0, 1]
    if vmax > vmin:
        normalized = (index - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(index)
    
    normalized = np.clip(normalized, 0.0, 1.0)
    
    return normalized


def classify_vegetation(
    ndvi: Union[torch.Tensor, np.ndarray],
    thresholds: Optional[dict] = None
) -> torch.Tensor:
    """
    Classify vegetation based on NDVI thresholds.
    
    Default classification:
    - 0: Water/Snow (NDVI < 0.0)
    - 1: Bare soil/Rock (0.0 <= NDVI < 0.2)
    - 2: Sparse vegetation (0.2 <= NDVI < 0.5)
    - 3: Moderate vegetation (0.5 <= NDVI < 0.7)
    - 4: Dense vegetation (NDVI >= 0.7)
    
    Args:
        ndvi: NDVI tensor (B, 1, H, W) or (1, H, W) or (H, W)
        thresholds: Custom thresholds dict with keys: 'water', 'bare', 'sparse', 'moderate'
    
    Returns:
        Classification tensor with integer class labels
    """
    # Convert to torch if numpy
    if isinstance(ndvi, np.ndarray):
        ndvi = torch.from_numpy(ndvi)
    
    # Default thresholds
    if thresholds is None:
        thresholds = {
            'water': 0.0,
            'bare': 0.2,
            'sparse': 0.5,
            'moderate': 0.7
        }
    
    # Initialize classification tensor
    classification = torch.zeros_like(ndvi, dtype=torch.long)
    
    # Apply thresholds
    classification[ndvi < thresholds['water']] = 0  # Water/Snow
    classification[(ndvi >= thresholds['water']) & (ndvi < thresholds['bare'])] = 1  # Bare
    classification[(ndvi >= thresholds['bare']) & (ndvi < thresholds['sparse'])] = 2  # Sparse
    classification[(ndvi >= thresholds['sparse']) & (ndvi < thresholds['moderate'])] = 3  # Moderate
    classification[ndvi >= thresholds['moderate']] = 4  # Dense
    
    return classification


def compute_spectral_rmse(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    band_order: str = "B02_B03_B04_B08",
    eps: float = 1e-8
) -> float:
    """
    Compute RMSE of spectral indices (NDVI and EVI).
    
    This metric measures the accuracy of spectral relationships in predicted imagery
    compared to target imagery. It's useful for validating that refinement/generation
    processes maintain physically plausible vegetation signatures.
    
    Args:
        pred: Predicted optical imagery (B, C, H, W) or (C, H, W) with at least 4 bands
        target: Target optical imagery (B, C, H, W) or (C, H, W) with at least 4 bands
        band_order: Band ordering in input (see extract_bands)
        eps: Small epsilon for numerical stability
    
    Returns:
        Average RMSE of NDVI and EVI (lower is better)
        
    Example:
        >>> pred = torch.rand(1, 4, 256, 256)  # B, G, R, NIR
        >>> target = torch.rand(1, 4, 256, 256)
        >>> rmse = compute_spectral_rmse(pred, target)
        >>> print(f"Spectral RMSE: {rmse:.4f}")
    """
    # Convert to torch if numpy
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    
    # Compute NDVI for pred and target
    ndvi_pred = compute_ndvi(pred, band_order, eps)
    ndvi_target = compute_ndvi(target, band_order, eps)
    
    # Compute NDVI RMSE
    ndvi_mse = F.mse_loss(ndvi_pred, ndvi_target)
    ndvi_rmse = torch.sqrt(ndvi_mse).item()
    
    # Compute EVI for pred and target
    evi_pred = compute_evi(pred, band_order, eps=eps)
    evi_target = compute_evi(target, band_order, eps=eps)
    
    # Compute EVI RMSE
    evi_mse = F.mse_loss(evi_pred, evi_target)
    evi_rmse = torch.sqrt(evi_mse).item()
    
    # Return average RMSE
    return (ndvi_rmse + evi_rmse) / 2.0


# Convenience function for batch processing
def batch_compute_indices(
    optical_batch: torch.Tensor,
    band_order: str = "B02_B03_B04_B08",
    return_dict: bool = True
) -> Union[dict, Tuple[torch.Tensor, ...]]:
    """
    Compute NDVI, EVI, and SAVI for a batch of optical images.
    
    Args:
        optical_batch: Batch of optical imagery (B, C, H, W)
        band_order: Band ordering in input
        return_dict: If True, return dict; if False, return tuple (ndvi, evi, savi)
    
    Returns:
        Dictionary or tuple of spectral indices
    """
    ndvi = compute_ndvi(optical_batch, band_order)
    evi = compute_evi(optical_batch, band_order)
    savi = compute_savi(optical_batch, band_order)
    
    if return_dict:
        return {"ndvi": ndvi, "evi": evi, "savi": savi}
    else:
        return ndvi, evi, savi
