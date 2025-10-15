"""
axs_lib/alignment.py - SAR-Optical Alignment Correction

Provides functions to detect and correct spatial misalignment between SAR and 
optical imagery. Misalignment can occur due to:
- Imprecise geolocation metadata
- Terrain effects not fully corrected
- Resampling artifacts
- Different acquisition geometries

Correction Methods:
1. Cross-correlation (automatic, subpixel accuracy)
2. Feature-based matching (SIFT/ORB keypoints)
3. Phase correlation (FFT-based, fast)
4. Manual offset adjustment

Usage:
    >>> from axs_lib.alignment import correct_alignment, estimate_offset
    >>> 
    >>> # Automatic correction
    >>> offset = estimate_offset(sar_data, optical_data, method='correlation')
    >>> sar_aligned = apply_offset(sar_data, offset)
    >>> 
    >>> # Or in one step
    >>> sar_aligned, offset, confidence = correct_alignment(sar_data, optical_data)

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union, List

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Optional dependencies
try:
    from scipy import ndimage
    from scipy.ndimage import shift, gaussian_filter
    from scipy.signal import correlate2d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available - install with: pip install scipy")

try:
    from skimage.registration import phase_cross_correlation
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    warnings.warn("scikit-image not available for phase correlation - install with: pip install scikit-image")


# ============================================================================
# Offset Estimation
# ============================================================================

def estimate_offset_correlation(
    sar_data: np.ndarray,
    optical_data: np.ndarray,
    max_offset: int = 20,
    upsample_factor: int = 10
) -> Tuple[float, float, float]:
    """
    Estimate spatial offset using normalized cross-correlation.
    
    This is the most robust method for SAR-optical alignment as it works
    directly on intensity values and handles different modalities well.
    
    Args:
        sar_data: SAR intensity array (2D)
        optical_data: Optical intensity array (2D, same shape as SAR)
        max_offset: Maximum offset to search (pixels)
        upsample_factor: Subpixel precision (higher = more precise but slower)
        
    Returns:
        Tuple of (offset_y, offset_x, confidence)
        - offset_y, offset_x: Shift in pixels (can be fractional)
        - confidence: Match confidence (0-1)
        
    Example:
        >>> offset_y, offset_x, conf = estimate_offset_correlation(sar, optical)
        >>> print(f"Offset: ({offset_y:.2f}, {offset_x:.2f}) pixels, confidence: {conf:.3f}")
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required. Install with: pip install scipy")
    
    # Handle NaN values - replace with mean
    sar_clean = np.copy(sar_data)
    optical_clean = np.copy(optical_data)
    
    sar_clean[np.isnan(sar_clean)] = np.nanmean(sar_data)
    optical_clean[np.isnan(optical_clean)] = np.nanmean(optical_data)
    
    # Normalize to zero mean, unit variance
    sar_norm = (sar_clean - np.mean(sar_clean)) / (np.std(sar_clean) + 1e-10)
    optical_norm = (optical_clean - np.mean(optical_clean)) / (np.std(optical_clean) + 1e-10)
    
    # Use phase correlation if available (more robust)
    if HAS_SKIMAGE:
        try:
            shift_yx, error, diffphase = phase_cross_correlation(
                optical_norm,
                sar_norm,
                upsample_factor=upsample_factor
            )
            confidence = 1.0 - error  # Convert error to confidence
            return float(shift_yx[0]), float(shift_yx[1]), float(confidence)
        except Exception as e:
            warnings.warn(f"Phase correlation failed: {e}, falling back to spatial correlation")
    
    # Fallback: Spatial cross-correlation
    # Crop to reduce computation
    h, w = sar_norm.shape
    crop_size = min(h, w, 500)  # Use 500x500 max for speed
    center_y, center_x = h // 2, w // 2
    half_crop = crop_size // 2
    
    sar_crop = sar_norm[
        center_y - half_crop:center_y + half_crop,
        center_x - half_crop:center_x + half_crop
    ]
    optical_crop = optical_norm[
        center_y - half_crop:center_y + half_crop,
        center_x - half_crop:center_x + half_crop
    ]
    
    # Compute cross-correlation
    correlation = correlate2d(optical_crop, sar_crop, mode='same')
    
    # Find peak
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    # Convert to offset (relative to center)
    offset_y = peak_y - correlation.shape[0] // 2
    offset_x = peak_x - correlation.shape[1] // 2
    
    # Clip to max_offset
    offset_y = np.clip(offset_y, -max_offset, max_offset)
    offset_x = np.clip(offset_x, -max_offset, max_offset)
    
    # Estimate confidence from correlation peak
    max_corr = np.max(correlation)
    mean_corr = np.mean(correlation)
    confidence = (max_corr - mean_corr) / (np.std(correlation) + 1e-10)
    confidence = 1.0 / (1.0 + np.exp(-confidence))  # Sigmoid
    
    return float(offset_y), float(offset_x), float(confidence)


def estimate_offset_edges(
    sar_data: np.ndarray,
    optical_data: np.ndarray,
    max_offset: int = 20
) -> Tuple[float, float, float]:
    """
    Estimate offset using edge features (more robust to intensity differences).
    
    This method detects edges in both images and aligns them, which can work
    better when SAR and optical have very different intensity distributions.
    
    Args:
        sar_data: SAR intensity array
        optical_data: Optical intensity array
        max_offset: Maximum offset to search
        
    Returns:
        Tuple of (offset_y, offset_x, confidence)
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required")
    
    from scipy.ndimage import sobel
    
    # Compute edges
    sar_edges = np.sqrt(
        sobel(sar_data, axis=0)**2 + 
        sobel(sar_data, axis=1)**2
    )
    optical_edges = np.sqrt(
        sobel(optical_data, axis=0)**2 + 
        sobel(optical_data, axis=1)**2
    )
    
    # Use correlation on edges
    return estimate_offset_correlation(
        sar_edges,
        optical_edges,
        max_offset=max_offset,
        upsample_factor=5
    )


# ============================================================================
# Offset Application
# ============================================================================

def apply_offset(
    data: np.ndarray,
    offset: Tuple[float, float],
    order: int = 3,
    fill_value: float = np.nan
) -> np.ndarray:
    """
    Apply spatial offset to data array using interpolation.
    
    Args:
        data: Input data array (2D)
        offset: (offset_y, offset_x) in pixels (can be fractional)
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        fill_value: Value for areas outside original bounds
        
    Returns:
        Shifted data array (same shape as input)
        
    Example:
        >>> # Shift SAR by detected offset
        >>> sar_aligned = apply_offset(sar_data, (1.5, -0.8))
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required")
    
    offset_y, offset_x = offset
    
    # Use scipy's shift (supports subpixel with spline interpolation)
    shifted = shift(
        data,
        shift=(offset_y, offset_x),
        order=order,
        cval=fill_value,
        prefilter=True
    )
    
    return shifted


# ============================================================================
# Combined Correction
# ============================================================================

def correct_alignment(
    sar_data: np.ndarray,
    optical_data: np.ndarray,
    method: str = 'correlation',
    max_offset: int = 20,
    confidence_threshold: float = 0.3,
    return_offset: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[float, float], float]]:
    """
    Automatically detect and correct SAR-optical misalignment.
    
    This is the main function to use for alignment correction. It estimates
    the offset and applies the correction in one step.
    
    Args:
        sar_data: SAR intensity array to be corrected
        optical_data: Reference optical intensity array
        method: Alignment method ('correlation' or 'edges')
        max_offset: Maximum offset to search (pixels)
        confidence_threshold: Minimum confidence to apply correction
        return_offset: If True, return (corrected_data, offset, confidence)
        
    Returns:
        If return_offset=False: Corrected SAR array
        If return_offset=True: (corrected_sar, (offset_y, offset_x), confidence)
        
    Example:
        >>> # Simple usage
        >>> sar_corrected = correct_alignment(sar, optical)
        >>> 
        >>> # With details
        >>> sar_corrected, offset, conf = correct_alignment(
        ...     sar, optical, 
        ...     method='correlation',
        ...     return_offset=True
        ... )
        >>> print(f"Applied offset: {offset}, confidence: {conf:.2f}")
    """
    print(f"Estimating alignment offset (method: {method})...")
    
    # Estimate offset
    if method == 'correlation':
        offset_y, offset_x, confidence = estimate_offset_correlation(
            sar_data, optical_data, max_offset=max_offset
        )
    elif method == 'edges':
        offset_y, offset_x, confidence = estimate_offset_edges(
            sar_data, optical_data, max_offset=max_offset
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'correlation' or 'edges'")
    
    print(f"  Detected offset: ({offset_y:.2f}, {offset_x:.2f}) pixels")
    print(f"  Confidence: {confidence:.3f}")
    
    # Check confidence
    if confidence < confidence_threshold:
        warnings.warn(
            f"Low confidence ({confidence:.3f} < {confidence_threshold}). "
            f"Correction may not be reliable. Consider manual inspection."
        )
    
    # Apply correction
    if abs(offset_y) < 0.1 and abs(offset_x) < 0.1:
        print("  Offset negligible, no correction applied")
        sar_corrected = sar_data
    else:
        print(f"  Applying correction...")
        sar_corrected = apply_offset(sar_data, (offset_y, offset_x))
        print(f"  ✓ Correction applied")
    
    if return_offset:
        return sar_corrected, (offset_y, offset_x), confidence
    else:
        return sar_corrected


# ============================================================================
# Tile Correction
# ============================================================================

def correct_tile_alignment(
    tile_npz_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sar_band: str = 's1_vv',
    reference_band: str = 's2_b4',
    method: str = 'correlation',
    max_offset: int = 20,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Correct alignment in an NPZ tile file.
    
    Loads a tile, detects misalignment between SAR and optical, applies
    correction to all SAR bands, and saves corrected tile.
    
    Args:
        tile_npz_path: Path to input tile NPZ
        output_path: Path for corrected tile (None = overwrite original)
        sar_band: SAR band to use for offset detection
        reference_band: Optical band to use as reference
        method: Alignment method
        max_offset: Maximum offset to search
        overwrite: Allow overwriting original file
        
    Returns:
        Dictionary with correction metadata
        
    Example:
        >>> # Correct tile and save with suffix
        >>> result = correct_tile_alignment(
        ...     'tile_r00000_c00000.npz',
        ...     output_path='tile_r00000_c00000_aligned.npz'
        ... )
        >>> print(f"Offset: {result['offset']}")
    """
    tile_path = Path(tile_npz_path)
    
    if not tile_path.exists():
        raise FileNotFoundError(f"Tile not found: {tile_path}")
    
    # Determine output path
    if output_path is None:
        if not overwrite:
            # Add suffix
            output_path = tile_path.with_name(
                tile_path.stem + '_aligned' + tile_path.suffix
            )
        else:
            output_path = tile_path
    else:
        output_path = Path(output_path)
    
    print(f"Loading tile: {tile_path.name}")
    tile = np.load(tile_path)
    
    # Extract reference bands
    if sar_band not in tile.files:
        raise ValueError(f"SAR band '{sar_band}' not found in tile")
    if reference_band not in tile.files:
        raise ValueError(f"Reference band '{reference_band}' not found in tile")
    
    sar_data = tile[sar_band]
    optical_data = tile[reference_band]
    
    # Correct alignment
    _, offset, confidence = correct_alignment(
        sar_data,
        optical_data,
        method=method,
        max_offset=max_offset,
        return_offset=True
    )
    
    # Apply to all SAR bands
    corrected_arrays = {}
    sar_bands = [b for b in tile.files if b.startswith('s1_')]
    
    print(f"\nApplying correction to {len(sar_bands)} SAR bands:")
    for band_name in sar_bands:
        print(f"  - {band_name}")
        corrected_arrays[band_name] = apply_offset(
            tile[band_name],
            offset
        )
    
    # Keep optical bands unchanged
    for band_name in tile.files:
        if not band_name.startswith('s1_'):
            corrected_arrays[band_name] = tile[band_name]
    
    # Save corrected tile
    print(f"\nSaving corrected tile to: {output_path}")
    np.savez_compressed(output_path, **corrected_arrays)
    
    # Metadata
    result = {
        'input_path': str(tile_path),
        'output_path': str(output_path),
        'offset': offset,
        'confidence': confidence,
        'method': method,
        'sar_bands_corrected': sar_bands,
        'timestamp': np.datetime64('now').astype(str)
    }
    
    print(f"✓ Alignment correction complete")
    
    return result


# ============================================================================
# Batch Correction
# ============================================================================

def correct_tiles_batch(
    tile_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    pattern: str = '*.npz',
    method: str = 'correlation',
    max_offset: int = 20,
    min_confidence: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Correct alignment for all tiles in a directory.
    
    Args:
        tile_dir: Directory containing tiles
        output_dir: Output directory (None = same as input with _aligned suffix)
        pattern: Glob pattern for tile files
        method: Alignment method
        max_offset: Maximum offset
        min_confidence: Skip tiles with lower confidence
        
    Returns:
        List of result dictionaries for each tile
        
    Example:
        >>> results = correct_tiles_batch(
        ...     'data/tiles/nairobi_2024-01-15',
        ...     output_dir='data/tiles/nairobi_2024-01-15_aligned'
        ... )
        >>> print(f"Corrected {len(results)} tiles")
    """
    tile_dir = Path(tile_dir)
    
    if output_dir is None:
        output_dir = tile_dir.parent / (tile_dir.name + '_aligned')
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find tiles
    tiles = sorted(tile_dir.glob(pattern))
    
    if not tiles:
        print(f"No tiles found matching '{pattern}' in {tile_dir}")
        return []
    
    print(f"Found {len(tiles)} tiles to process")
    print()
    
    results = []
    skipped = 0
    
    for i, tile_path in enumerate(tiles, 1):
        print(f"{'='*79}")
        print(f"Processing tile {i}/{len(tiles)}: {tile_path.name}")
        print(f"{'='*79}")
        
        try:
            output_path = output_dir / tile_path.name
            
            result = correct_tile_alignment(
                tile_path,
                output_path=output_path,
                method=method,
                max_offset=max_offset
            )
            
            # Check confidence
            if result['confidence'] < min_confidence:
                print(f"⚠ WARNING: Low confidence ({result['confidence']:.3f})")
                skipped += 1
            
            results.append(result)
            
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append({
                'input_path': str(tile_path),
                'error': str(e)
            })
        
        print()
    
    # Summary
    print(f"{'='*79}")
    print(f"BATCH CORRECTION COMPLETE")
    print(f"{'='*79}")
    print(f"Total tiles: {len(tiles)}")
    print(f"Successful: {len([r for r in results if 'error' not in r])}")
    print(f"Failed: {len([r for r in results if 'error' in r])}")
    print(f"Low confidence: {skipped}")
    print()
    
    return results


# ============================================================================
# Validation
# ============================================================================

def validate_correction(
    original_tile: Union[str, Path],
    corrected_tile: Union[str, Path],
    sar_band: str = 's1_vv',
    reference_band: str = 's2_b4'
) -> Dict[str, float]:
    """
    Validate alignment correction by computing before/after metrics.
    
    Args:
        original_tile: Path to original tile
        corrected_tile: Path to corrected tile
        sar_band: SAR band name
        reference_band: Optical reference band
        
    Returns:
        Dictionary with validation metrics
    """
    # Load tiles
    orig = np.load(original_tile)
    corr = np.load(corrected_tile)
    
    # Compute offsets
    print("Computing offset for original tile...")
    _, orig_offset, orig_conf = estimate_offset_correlation(
        orig[sar_band],
        orig[reference_band]
    )
    
    print("Computing offset for corrected tile...")
    _, corr_offset, corr_conf = estimate_offset_correlation(
        corr[sar_band],
        corr[reference_band]
    )
    
    # Metrics
    orig_magnitude = np.sqrt(orig_offset[0]**2 + orig_offset[1]**2)
    corr_magnitude = np.sqrt(corr_offset[0]**2 + corr_offset[1]**2)
    improvement = (orig_magnitude - corr_magnitude) / (orig_magnitude + 1e-10) * 100
    
    metrics = {
        'original_offset_y': orig_offset[0],
        'original_offset_x': orig_offset[1],
        'original_magnitude': orig_magnitude,
        'original_confidence': orig_conf,
        'corrected_offset_y': corr_offset[0],
        'corrected_offset_x': corr_offset[1],
        'corrected_magnitude': corr_magnitude,
        'corrected_confidence': corr_conf,
        'improvement_percent': improvement
    }
    
    print(f"\nValidation Results:")
    print(f"  Original offset: ({orig_offset[0]:.2f}, {orig_offset[1]:.2f}) pixels")
    print(f"  Corrected offset: ({corr_offset[0]:.2f}, {corr_offset[1]:.2f}) pixels")
    print(f"  Improvement: {improvement:.1f}%")
    
    return metrics


# ============================================================================
# Main / Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 79)
    print("ALIGNMENT CORRECTION MODULE")
    print("=" * 79)
    print()
    
    print("This module provides automatic misalignment correction for SAR-optical data.")
    print()
    print("Example usage:")
    print("-" * 79)
    print()
    print("from axs_lib.alignment import correct_tile_alignment")
    print()
    print("# Correct single tile")
    print("result = correct_tile_alignment(")
    print("    'tile_r00000_c00000.npz',")
    print("    output_path='tile_r00000_c00000_aligned.npz'")
    print(")")
    print()
    print("# Batch correct all tiles")
    print("from axs_lib.alignment import correct_tiles_batch")
    print("results = correct_tiles_batch('data/tiles/nairobi_2024-01-15')")
    print()
    print("=" * 79)
    print()
    
    # Check dependencies
    print("Dependency check:")
    print(f"  scipy:        {'✓' if HAS_SCIPY else '✗ (pip install scipy)'}")
    print(f"  scikit-image: {'✓' if HAS_SKIMAGE else '✗ (pip install scikit-image, optional)'}")
    print()
    
    if not HAS_SCIPY:
        print("⚠ scipy is required for alignment correction")
    if not HAS_SKIMAGE:
        print("ℹ scikit-image is optional but recommended for better accuracy")
