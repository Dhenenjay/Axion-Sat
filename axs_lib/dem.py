"""
axs_lib/dem.py - Digital Elevation Model Utilities

Provides functions to fetch and cache Digital Elevation Model (DEM) data from
SRTM or Copernicus DEM sources. Computes terrain derivatives (slope, aspect)
for topographic correction and analysis.

Key features:
- Automatic download and caching of DEM tiles
- Support for SRTM 30m and Copernicus DEM 30m/90m
- Efficient local cache management
- Computation of slope and aspect from elevation
- Graceful fallback to zero-filled arrays when DEM unavailable

Usage:
    from axs_lib.dem import get_dem, compute_slope_aspect
    
    # Fetch DEM for bounding box
    dem = get_dem(bounds, resolution=30, crs='EPSG:4326')
    
    # Compute terrain derivatives
    slope, aspect = compute_slope_aspect(dem, pixel_size=30)

Author: Axion-Sat Project
Version: 1.0.0
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Literal
import hashlib
import json

import numpy as np

# Optional dependencies
try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    warnings.warn(
        "rasterio not available - DEM fetching disabled. "
        "Install with: pip install rasterio"
    )

try:
    from scipy.ndimage import sobel, gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import elevation
    HAS_ELEVATION = True
except ImportError:
    HAS_ELEVATION = False
    # Not critical - will try alternative methods


# ============================================================================
# Configuration
# ============================================================================

# Default cache directory for DEM tiles
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "axion_sat" / "dem"

# DEM source priorities (tried in order)
DEM_SOURCES = [
    "copernicus_30m",  # Copernicus DEM 30m (global, most recent)
    "copernicus_90m",  # Copernicus DEM 90m (global, coarser)
    "srtm_30m",        # SRTM 30m (limited to ±60° latitude)
    "srtm_90m",        # SRTM 90m (limited to ±60° latitude)
]

# Resolution mapping (meters)
DEM_RESOLUTIONS = {
    "copernicus_30m": 30,
    "copernicus_90m": 90,
    "srtm_30m": 30,
    "srtm_90m": 90,
}


# ============================================================================
# Cache Management
# ============================================================================

def _get_cache_dir() -> Path:
    """Get or create DEM cache directory."""
    cache_dir = os.environ.get("AXION_DEM_CACHE", DEFAULT_CACHE_DIR)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _cache_key(bounds: Tuple[float, float, float, float], 
               resolution: int,
               source: str) -> str:
    """
    Generate unique cache key for DEM tile.
    
    Args:
        bounds: (west, south, east, north) in decimal degrees
        resolution: Resolution in meters
        source: DEM source identifier
        
    Returns:
        Cache key string
    """
    # Round bounds to reduce cache fragmentation
    rounded = tuple(round(b, 4) for b in bounds)
    key_str = f"{source}_{resolution}m_{rounded}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cached_dem(cache_key: str) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    Retrieve DEM from cache if available.
    
    Args:
        cache_key: Cache key for the DEM tile
        
    Returns:
        Tuple of (elevation array, metadata) or None if not cached
    """
    cache_dir = _get_cache_dir()
    dem_file = cache_dir / f"{cache_key}.npz"
    meta_file = cache_dir / f"{cache_key}.json"
    
    if not dem_file.exists() or not meta_file.exists():
        return None
    
    try:
        # Load DEM data
        data = np.load(dem_file)
        elevation = data['elevation']
        
        # Load metadata
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        return elevation, metadata
    
    except Exception as e:
        warnings.warn(f"Failed to load cached DEM: {e}")
        return None


def _cache_dem(cache_key: str, 
               elevation: np.ndarray, 
               metadata: Dict) -> None:
    """
    Save DEM to cache.
    
    Args:
        cache_key: Cache key for the DEM tile
        elevation: Elevation array
        metadata: DEM metadata
    """
    cache_dir = _get_cache_dir()
    dem_file = cache_dir / f"{cache_key}.npz"
    meta_file = cache_dir / f"{cache_key}.json"
    
    try:
        # Save DEM data (compressed)
        np.savez_compressed(dem_file, elevation=elevation)
        
        # Save metadata
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    except Exception as e:
        warnings.warn(f"Failed to cache DEM: {e}")


# ============================================================================
# DEM Fetching
# ============================================================================

def _fetch_srtm(bounds: Tuple[float, float, float, float],
                resolution: int = 30) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    Fetch SRTM DEM using elevation package.
    
    Args:
        bounds: (west, south, east, north) in decimal degrees
        resolution: Resolution in meters (30 or 90)
        
    Returns:
        Tuple of (elevation array, metadata) or None if failed
    """
    if not HAS_ELEVATION:
        return None
    
    if not HAS_RASTERIO:
        return None
    
    west, south, east, north = bounds
    
    # Check latitude bounds for SRTM (±60°)
    if south < -60 or north > 60:
        warnings.warn(
            f"SRTM data not available for latitude {south:.2f}° to {north:.2f}°. "
            "SRTM covers ±60° only."
        )
        return None
    
    try:
        import tempfile
        
        # Create temporary file for DEM
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Determine product (SRTM1 = 30m, SRTM3 = 90m)
        product = 'SRTM1' if resolution <= 30 else 'SRTM3'
        
        # Download DEM using elevation package
        elevation.clip(bounds=(west, south, east, north),
                      output=tmp_path,
                      product=product)
        
        # Read DEM
        with rasterio.open(tmp_path) as src:
            dem_data = src.read(1)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
        
        # Clean up
        os.unlink(tmp_path)
        
        # Replace nodata with NaN
        if nodata is not None:
            dem_data = dem_data.astype(np.float32)
            dem_data[dem_data == nodata] = np.nan
        
        metadata = {
            'source': f'srtm_{resolution}m',
            'bounds': bounds,
            'resolution': resolution,
            'crs': str(crs),
            'transform': [transform.a, transform.b, transform.c,
                         transform.d, transform.e, transform.f],
            'shape': dem_data.shape
        }
        
        return dem_data, metadata
    
    except Exception as e:
        warnings.warn(f"SRTM fetch failed: {e}")
        return None


def _fetch_copernicus(bounds: Tuple[float, float, float, float],
                      resolution: int = 30) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    Fetch Copernicus DEM (placeholder - requires API access).
    
    Note: Copernicus DEM requires authentication and API access.
    This is a stub for future implementation.
    
    Args:
        bounds: (west, south, east, north) in decimal degrees
        resolution: Resolution in meters (30 or 90)
        
    Returns:
        Tuple of (elevation array, metadata) or None if failed
    """
    warnings.warn(
        "Copernicus DEM fetching not yet implemented. "
        "Falling back to SRTM or zero-filled DEM."
    )
    return None


def get_dem(bounds: Tuple[float, float, float, float],
            resolution: int = 30,
            crs: str = 'EPSG:4326',
            output_shape: Optional[Tuple[int, int]] = None,
            source: Optional[str] = None,
            use_cache: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Fetch Digital Elevation Model for specified bounds.
    
    Attempts to download DEM from available sources (SRTM, Copernicus) with
    automatic caching. Falls back to zero-filled array if unavailable.
    
    Args:
        bounds: Bounding box (west, south, east, north) in decimal degrees
        resolution: Target resolution in meters (default: 30)
        crs: Output coordinate reference system (default: EPSG:4326)
        output_shape: Optional output shape (height, width) for resampling
        source: Force specific DEM source (e.g., 'srtm_30m', 'copernicus_30m')
        use_cache: Use cached DEM if available
        
    Returns:
        Tuple of (elevation array, metadata dict)
        
    Example:
        >>> bounds = (-122.5, 37.5, -122.0, 38.0)  # San Francisco area
        >>> dem, meta = get_dem(bounds, resolution=30)
        >>> print(f"DEM shape: {dem.shape}, source: {meta['source']}")
    """
    west, south, east, north = bounds
    
    # Validate bounds
    if not (-180 <= west <= 180 and -180 <= east <= 180 and
            -90 <= south <= 90 and -90 <= north <= 90):
        raise ValueError(f"Invalid bounds: {bounds}")
    
    if west >= east or south >= north:
        raise ValueError(f"Invalid bounds: west >= east or south >= north")
    
    # Check cache
    if use_cache:
        source_key = source or 'auto'
        cache_key = _cache_key(bounds, resolution, source_key)
        cached = _get_cached_dem(cache_key)
        if cached is not None:
            elevation, metadata = cached
            
            # Resample if output shape requested
            if output_shape is not None and elevation.shape != output_shape:
                elevation = _resample_dem(elevation, output_shape)
                metadata['resampled'] = True
                metadata['output_shape'] = output_shape
            
            return elevation, metadata
    
    # Determine sources to try
    if source is not None:
        sources_to_try = [source]
    else:
        sources_to_try = DEM_SOURCES
    
    # Try each source in order
    elevation = None
    metadata = None
    
    for src in sources_to_try:
        if 'srtm' in src:
            res = 30 if '30m' in src else 90
            result = _fetch_srtm(bounds, resolution=res)
        elif 'copernicus' in src:
            res = 30 if '30m' in src else 90
            result = _fetch_copernicus(bounds, resolution=res)
        else:
            warnings.warn(f"Unknown DEM source: {src}")
            continue
        
        if result is not None:
            elevation, metadata = result
            break
    
    # Fallback to zero-filled array
    if elevation is None:
        warnings.warn(
            f"DEM not available for bounds {bounds}. "
            "Using zero-filled elevation array. "
            "Terrain correction will be skipped."
        )
        
        # Calculate default shape based on resolution
        if output_shape is not None:
            shape = output_shape
        else:
            # Approximate shape from bounds and resolution
            lat_dist = (north - south) * 111000  # meters
            lon_dist = (east - west) * 111000 * np.cos(np.radians((north + south) / 2))
            height = int(lat_dist / resolution)
            width = int(lon_dist / resolution)
            shape = (height, width)
        
        elevation = np.zeros(shape, dtype=np.float32)
        
        metadata = {
            'source': 'zero_filled',
            'bounds': bounds,
            'resolution': resolution,
            'crs': crs,
            'shape': shape,
            'warning': 'DEM unavailable - using zero elevation'
        }
    
    # Resample if needed
    if output_shape is not None and elevation.shape != output_shape:
        elevation = _resample_dem(elevation, output_shape)
        metadata['resampled'] = True
        metadata['output_shape'] = output_shape
    
    # Cache result
    if use_cache and metadata.get('source') != 'zero_filled':
        source_key = metadata.get('source', 'auto')
        cache_key = _cache_key(bounds, resolution, source_key)
        _cache_dem(cache_key, elevation, metadata)
    
    return elevation, metadata


def _resample_dem(dem: np.ndarray, 
                  output_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resample DEM to target shape using bilinear interpolation.
    
    Args:
        dem: Input elevation array
        output_shape: Target shape (height, width)
        
    Returns:
        Resampled elevation array
    """
    from scipy.ndimage import zoom
    
    zoom_factors = (output_shape[0] / dem.shape[0],
                   output_shape[1] / dem.shape[1])
    
    resampled = zoom(dem, zoom_factors, order=1)  # Bilinear
    return resampled


# ============================================================================
# Terrain Derivatives
# ============================================================================

def compute_slope_aspect(elevation: np.ndarray,
                        pixel_size: float = 30.0,
                        smooth: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute slope and aspect from elevation data.
    
    Slope is computed as the gradient magnitude in degrees.
    Aspect is computed as the gradient direction in degrees from north (0-360).
    
    Args:
        elevation: Elevation array in meters
        pixel_size: Pixel size in meters (default: 30)
        smooth: Apply Gaussian smoothing before gradient computation
        
    Returns:
        Tuple of (slope in degrees, aspect in degrees)
        
    Example:
        >>> dem, _ = get_dem(bounds, resolution=30)
        >>> slope, aspect = compute_slope_aspect(dem, pixel_size=30)
        >>> print(f"Max slope: {np.nanmax(slope):.1f}°")
    """
    if not HAS_SCIPY:
        warnings.warn(
            "scipy not available - returning zero slope/aspect. "
            "Install with: pip install scipy"
        )
        return (np.zeros_like(elevation), 
                np.zeros_like(elevation))
    
    # Apply smoothing to reduce noise
    if smooth:
        elevation_smooth = gaussian_filter(elevation, sigma=1.0)
    else:
        elevation_smooth = elevation
    
    # Compute gradients using Sobel operator
    # Note: Sobel returns gradient in pixel units
    dy = sobel(elevation_smooth, axis=0) / (8.0 * pixel_size)  # North-South gradient
    dx = sobel(elevation_smooth, axis=1) / (8.0 * pixel_size)  # East-West gradient
    
    # Compute slope (gradient magnitude) in radians, then convert to degrees
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    
    # Compute aspect (gradient direction)
    # arctan2(dx, dy) gives angle from north, clockwise
    aspect_rad = np.arctan2(dx, -dy)  # Note: -dy because y increases downward
    aspect_deg = np.degrees(aspect_rad)
    
    # Convert aspect to 0-360 range (0 = north, 90 = east, 180 = south, 270 = west)
    aspect_deg = (aspect_deg + 360) % 360
    
    # Set aspect to NaN where slope is near zero (flat areas)
    aspect_deg[slope_deg < 0.1] = np.nan
    
    return slope_deg, aspect_deg


def compute_hillshade(elevation: np.ndarray,
                     pixel_size: float = 30.0,
                     azimuth: float = 315.0,
                     altitude: float = 45.0) -> np.ndarray:
    """
    Compute hillshade (shaded relief) from elevation data.
    
    Simulates illumination of terrain from a light source at specified
    azimuth and altitude angles.
    
    Args:
        elevation: Elevation array in meters
        pixel_size: Pixel size in meters
        azimuth: Light source azimuth in degrees (0=north, 90=east)
        altitude: Light source altitude in degrees (0=horizon, 90=zenith)
        
    Returns:
        Hillshade array (0-255)
        
    Example:
        >>> dem, _ = get_dem(bounds)
        >>> hillshade = compute_hillshade(dem, azimuth=315, altitude=45)
    """
    # Compute slope and aspect
    slope_deg, aspect_deg = compute_slope_aspect(elevation, pixel_size, smooth=True)
    
    # Convert angles to radians
    slope_rad = np.radians(slope_deg)
    aspect_rad = np.radians(aspect_deg)
    azimuth_rad = np.radians(azimuth)
    altitude_rad = np.radians(altitude)
    
    # Compute hillshade using standard formula
    hillshade = (
        np.cos(altitude_rad) * np.cos(slope_rad) +
        np.sin(altitude_rad) * np.sin(slope_rad) * 
        np.cos(azimuth_rad - aspect_rad)
    )
    
    # Handle NaN values (flat areas)
    hillshade = np.nan_to_num(hillshade, nan=np.cos(altitude_rad))
    
    # Scale to 0-255
    hillshade = np.clip(hillshade * 255, 0, 255).astype(np.uint8)
    
    return hillshade


# ============================================================================
# Topographic Correction
# ============================================================================

def cosine_correction(image: np.ndarray,
                     slope: np.ndarray,
                     aspect: np.ndarray,
                     solar_azimuth: float,
                     solar_zenith: float) -> np.ndarray:
    """
    Apply cosine (Lambertian) topographic correction to optical imagery.
    
    Corrects for illumination variations due to terrain slope and aspect.
    
    Args:
        image: Input image array (any shape, last two dims are spatial)
        slope: Slope array in degrees
        aspect: Aspect array in degrees (0-360, 0=north)
        solar_azimuth: Solar azimuth angle in degrees (0=north, clockwise)
        solar_zenith: Solar zenith angle in degrees (0=zenith, 90=horizon)
        
    Returns:
        Topographically corrected image
        
    Reference:
        Teillet et al. (1982) - "On the Slope-Aspect Correction of 
        Multispectral Scanner Data"
    """
    # Convert angles to radians
    slope_rad = np.radians(slope)
    aspect_rad = np.radians(aspect)
    solar_az_rad = np.radians(solar_azimuth)
    solar_zen_rad = np.radians(solar_zenith)
    
    # Compute solar incidence angle (angle between sun and surface normal)
    cos_i = (
        np.cos(solar_zen_rad) * np.cos(slope_rad) +
        np.sin(solar_zen_rad) * np.sin(slope_rad) *
        np.cos(solar_az_rad - aspect_rad)
    )
    
    # Compute correction factor
    # correction = cos(solar_zenith) / cos(incidence_angle)
    cos_solar_zen = np.cos(solar_zen_rad)
    correction = cos_solar_zen / np.maximum(cos_i, 0.01)  # Avoid division by zero
    
    # Apply correction (broadcast across all bands)
    corrected = image * correction[..., np.newaxis] if image.ndim > 2 else image * correction
    
    # Clip to reasonable range
    corrected = np.clip(corrected, 0, np.nanmax(image))
    
    return corrected


# ============================================================================
# Utility Functions
# ============================================================================

def clear_cache(max_age_days: Optional[int] = None) -> int:
    """
    Clear DEM cache.
    
    Args:
        max_age_days: Only remove files older than this many days.
                     If None, remove all cached files.
        
    Returns:
        Number of files removed
    """
    cache_dir = _get_cache_dir()
    
    if not cache_dir.exists():
        return 0
    
    import time
    
    removed = 0
    now = time.time()
    
    for file in cache_dir.glob('*'):
        if max_age_days is not None:
            file_age_days = (now - file.stat().st_mtime) / 86400
            if file_age_days < max_age_days:
                continue
        
        try:
            file.unlink()
            removed += 1
        except Exception:
            pass
    
    return removed


def get_cache_size() -> Tuple[int, int]:
    """
    Get DEM cache statistics.
    
    Returns:
        Tuple of (number of cached tiles, total size in bytes)
    """
    cache_dir = _get_cache_dir()
    
    if not cache_dir.exists():
        return 0, 0
    
    files = list(cache_dir.glob('*.npz'))
    total_size = sum(f.stat().st_size for f in files)
    
    return len(files), total_size


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Fetch DEM for San Francisco Bay Area
    bounds = (-122.5, 37.5, -122.0, 38.0)
    
    print("Fetching DEM for San Francisco Bay Area...")
    dem, metadata = get_dem(bounds, resolution=30)
    
    print(f"DEM shape: {dem.shape}")
    print(f"Source: {metadata['source']}")
    print(f"Elevation range: {np.nanmin(dem):.1f} - {np.nanmax(dem):.1f} m")
    
    print("\nComputing slope and aspect...")
    slope, aspect = compute_slope_aspect(dem, pixel_size=30)
    
    print(f"Slope range: {np.nanmin(slope):.1f}° - {np.nanmax(slope):.1f}°")
    print(f"Mean slope: {np.nanmean(slope):.1f}°")
    
    print("\nComputing hillshade...")
    hillshade = compute_hillshade(dem, pixel_size=30)
    print(f"Hillshade range: {hillshade.min()} - {hillshade.max()}")
    
    # Cache statistics
    num_tiles, cache_size = get_cache_size()
    print(f"\nCache: {num_tiles} tiles, {cache_size / 1024**2:.1f} MB")
