"""
axs_lib/geo.py - Geospatial Processing Utilities

Provides functions for reading, reprojecting, terrain-correcting, and resampling
satellite imagery (Sentinel-1 SAR and Sentinel-2 optical).

Features:
    - Read SAFE/COG formats with rasterio
    - Automatic UTM zone detection from bounding box
    - Approximate terrain correction for Sentinel-1 (empirical normalization)
    - Resampling S1 (10m) onto S2 (10m) grid with alignment
    - Memory-efficient windowed processing
    - CRS harmonization for multi-sensor fusion

Important Notes:
    **Terrain Correction Approximation:**
    The terrain correction applied to Sentinel-1 GRD is an empirical
    approximation based on local incidence angle normalization. This is NOT
    a rigorous radiometric terrain correction (RTC) which requires:
    - High-resolution DEM (e.g., SRTM 30m, Copernicus DEM)
    - Precise orbit information
    - SAR-specific geometric models
    
    For production use cases requiring accurate radiometric terrain correction,
    use specialized tools like SNAP, GAMMA, or ASF's HyP3 RTC service.
    
    Our approximation is suitable for:
    - Relative analysis within the same scene
    - Machine learning features (models learn residual corrections)
    - Preliminary analysis and prototyping
    
    It assumes:
    - Flat or moderately sloping terrain
    - Local incidence angle variations are dominant effect
    - Approximated via latitude-based cosine correction

Usage:
    >>> from axs_lib.geo import read_cog, reproject_to_utm, align_s1_to_s2_grid
    >>> 
    >>> # Read Sentinel-2 band
    >>> s2_data, s2_profile = read_cog("data/raw/.../B04.tif")
    >>> 
    >>> # Read Sentinel-1 (with terrain correction approximation)
    >>> s1_data, s1_profile = read_cog("data/raw/.../VV.tif")
    >>> s1_corrected = terrain_correct_s1_approx(s1_data, s1_profile)
    >>> 
    >>> # Align S1 to S2 grid
    >>> s1_aligned = align_s1_to_s2_grid(s1_data, s1_profile, s2_profile)
"""

import sys
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import rasterio
    from rasterio.warp import reproject, calculate_default_transform, Resampling
    from rasterio.enums import Resampling as ResamplingEnum
    from rasterio.windows import Window
    from rasterio import Affine
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("WARNING: rasterio not available. Install with: pip install rasterio")

try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    print("WARNING: pyproj not available. Install with: pip install pyproj")


# ============================================================================
# CRS Detection and Conversion
# ============================================================================

def detect_utm_zone(bbox: List[float]) -> str:
    """
    Detect appropriate UTM zone from bounding box.
    
    Args:
        bbox: Bounding box [west, south, east, north] in EPSG:4326
        
    Returns:
        EPSG code for UTM zone (e.g., "EPSG:32736" for UTM 36S)
        
    Example:
        >>> bbox = [31.5, -1.5, 34.0, 0.5]  # Lake Victoria
        >>> utm_zone = detect_utm_zone(bbox)
        >>> print(utm_zone)
        'EPSG:32736'  # UTM Zone 36S
    """
    if not PYPROJ_AVAILABLE:
        raise ImportError("pyproj required. Install with: pip install pyproj")
    
    west, south, east, north = bbox
    
    # Calculate center longitude
    center_lon = (west + east) / 2.0
    center_lat = (south + north) / 2.0
    
    # Calculate UTM zone number (1-60)
    # Zone formula: floor((lon + 180) / 6) + 1
    zone_number = int((center_lon + 180) / 6) + 1
    
    # Determine hemisphere
    if center_lat >= 0:
        # Northern hemisphere: EPSG:326XX
        epsg_code = f"EPSG:326{zone_number:02d}"
    else:
        # Southern hemisphere: EPSG:327XX
        epsg_code = f"EPSG:327{zone_number:02d}"
    
    return epsg_code


def get_utm_crs_from_geometry(geometry: Dict) -> str:
    """
    Get UTM CRS from GeoJSON geometry.
    
    Args:
        geometry: GeoJSON geometry dict (e.g., from STAC item)
        
    Returns:
        EPSG code for UTM zone
    """
    coords = geometry['coordinates']
    
    # Handle Polygon geometry
    if geometry['type'] == 'Polygon':
        # Exterior ring
        lons = [c[0] for c in coords[0]]
        lats = [c[1] for c in coords[0]]
    else:
        raise ValueError(f"Unsupported geometry type: {geometry['type']}")
    
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    return detect_utm_zone(bbox)


# ============================================================================
# File Reading
# ============================================================================

def read_cog(
    file_path: Union[str, Path],
    window: Optional[Window] = None,
    masked: bool = True,
    overview_level: Optional[int] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Read Cloud-Optimized GeoTIFF (COG) or standard GeoTIFF.
    
    Args:
        file_path: Path to GeoTIFF file
        window: Optional window to read (for memory efficiency)
        masked: Return masked array (handle nodata)
        overview_level: Read overview level (0 = full resolution, 1+ = downsampled)
        
    Returns:
        Tuple of (data array, profile dict)
        - data: numpy array (H, W) or masked array
        - profile: rasterio profile with metadata
        
    Example:
        >>> data, profile = read_cog("data/raw/.../B04.tif")
        >>> print(data.shape, profile['crs'])
        (10980, 10980) EPSG:32636
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio required. Install with: pip install rasterio")
    
    file_path = Path(file_path)
    
    with rasterio.open(file_path) as src:
        # Read data
        if overview_level is not None:
            # Read from overview
            data = src.read(1, out_shape=(
                src.height // (2 ** overview_level),
                src.width // (2 ** overview_level)
            ), masked=masked, window=window)
        else:
            # Read at full resolution
            data = src.read(1, masked=masked, window=window)
        
        # Get profile
        profile = src.profile.copy()
        
        # Update profile if window was used
        if window is not None:
            profile['transform'] = src.window_transform(window)
            profile['height'] = window.height
            profile['width'] = window.width
    
    return data, profile


def read_safe_band(
    safe_dir: Path,
    band: str,
    resolution: str = "10m"
) -> Tuple[np.ndarray, Dict]:
    """
    Read band from Sentinel-2 SAFE format.
    
    Args:
        safe_dir: Path to .SAFE directory
        band: Band name (e.g., "B04", "B08")
        resolution: Resolution folder ("10m", "20m", "60m")
        
    Returns:
        Tuple of (data array, profile dict)
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio required. Install with: pip install rasterio")
    
    safe_dir = Path(safe_dir)
    
    # Find band file in SAFE structure
    # Structure: .SAFE/GRANULE/*/IMG_DATA/R{resolution}/*_{band}_{resolution}.jp2
    granule_dirs = list((safe_dir / "GRANULE").glob("*"))
    
    if not granule_dirs:
        raise FileNotFoundError(f"No GRANULE directory found in {safe_dir}")
    
    img_dir = granule_dirs[0] / "IMG_DATA" / f"R{resolution}"
    
    # Find band file
    band_files = list(img_dir.glob(f"*_{band}_{resolution}.jp2"))
    
    if not band_files:
        raise FileNotFoundError(f"Band {band} not found in {img_dir}")
    
    return read_cog(band_files[0])


# ============================================================================
# Reprojection
# ============================================================================

def reproject_to_utm(
    data: np.ndarray,
    src_profile: Dict,
    target_epsg: Optional[str] = None,
    target_resolution: Optional[float] = None,
    resampling: str = "bilinear"
) -> Tuple[np.ndarray, Dict]:
    """
    Reproject raster to UTM coordinate system.
    
    Args:
        data: Input data array
        src_profile: Source rasterio profile
        target_epsg: Target EPSG code (None = auto-detect from bounds)
        target_resolution: Target resolution in meters (None = preserve)
        resampling: Resampling method ("bilinear", "cubic", "nearest", "average")
        
    Returns:
        Tuple of (reprojected data, new profile)
        
    Example:
        >>> data, profile = read_cog("s2_band.tif")
        >>> data_utm, profile_utm = reproject_to_utm(
        ...     data, profile,
        ...     target_resolution=10.0  # 10m resolution
        ... )
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio required. Install with: pip install rasterio")
    
    src_crs = src_profile['crs']
    
    # Handle case where CRS is None (common with some Sentinel-1 products)
    if src_crs is None:
        raise ValueError(
            "Source CRS is None. Cannot reproject data without valid CRS. "
            "This typically happens with Sentinel-1 GRD data that lacks proper "
            "georeferencing metadata. Try using pre-processed RTC products or "
            "download from a different source."
        )
    
    src_transform = src_profile['transform']
    src_height = src_profile['height']
    src_width = src_profile['width']
    
    # Auto-detect UTM zone if not specified
    if target_epsg is None:
        # Get bounds in lat/lon
        from rasterio.warp import transform_bounds
        bounds = rasterio.transform.array_bounds(src_height, src_width, src_transform)
        bounds_latlon = transform_bounds(src_crs, 'EPSG:4326', *bounds)
        target_epsg = detect_utm_zone(list(bounds_latlon))
    
    # Calculate transform
    if target_resolution:
        # Use specified resolution
        bounds = rasterio.transform.array_bounds(src_height, src_width, src_transform)
        from rasterio.warp import transform_bounds
        dst_bounds = transform_bounds(src_crs, target_epsg, *bounds)
        
        dst_width = int((dst_bounds[2] - dst_bounds[0]) / target_resolution)
        dst_height = int((dst_bounds[3] - dst_bounds[1]) / target_resolution)
        
        dst_transform = Affine.translation(dst_bounds[0], dst_bounds[3]) * Affine.scale(
            target_resolution, -target_resolution
        )
    else:
        # Calculate default transform (preserve resolution)
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, target_epsg,
            src_width, src_height,
            *rasterio.transform.array_bounds(src_height, src_width, src_transform)
        )
    
    # Create destination array
    dst_data = np.zeros((dst_height, dst_width), dtype=data.dtype)
    
    # Resampling method mapping
    resampling_methods = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }
    resampling_method = resampling_methods.get(resampling, Resampling.bilinear)
    
    # Reproject
    reproject(
        source=data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=target_epsg,
        resampling=resampling_method
    )
    
    # Create new profile
    dst_profile = src_profile.copy()
    dst_profile.update({
        'crs': target_epsg,
        'transform': dst_transform,
        'width': dst_width,
        'height': dst_height,
    })
    
    return dst_data, dst_profile


# ============================================================================
# Sentinel-1 Terrain Correction (Approximate)
# ============================================================================

def terrain_correct_s1_approx(
    data: np.ndarray,
    profile: Dict,
    method: str = "cosine"
) -> np.ndarray:
    """
    Apply approximate terrain correction to Sentinel-1 GRD backscatter.
    
    **IMPORTANT: This is an empirical approximation, not rigorous RTC.**
    
    This function applies a simplified terrain correction based on local
    incidence angle approximation. It is suitable for:
    - Preliminary analysis
    - Machine learning preprocessing (models learn residuals)
    - Flat to moderately sloping terrain
    
    Limitations:
    - Does NOT use actual DEM (assumes flat Earth approximation)
    - Does NOT account for azimuth/range geometric distortions
    - Does NOT use precise orbit information
    - Approximates incidence angle from latitude only
    
    For rigorous RTC, use:
    - ESA SNAP Toolbox (Range-Doppler Terrain Correction)
    - GAMMA SAR processor
    - ASF HyP3 RTC products (pre-processed)
    
    Algorithm:
    1. Estimate local incidence angle from latitude (center of swath ~35-45°)
    2. Apply cosine correction: σ0_corrected = σ0 / cos(θ_local)
    3. This approximates normalization to flat surface reference
    
    Args:
        data: Sentinel-1 backscatter in linear scale (not dB)
        profile: Rasterio profile with CRS/transform info
        method: Correction method ("cosine" or "none")
        
    Returns:
        Terrain-corrected backscatter (linear scale)
        
    Example:
        >>> vv_data, profile = read_cog("S1A_VV.tif")
        >>> vv_corrected = terrain_correct_s1_approx(vv_data, profile)
        >>> # Convert to dB: vv_db = 10 * np.log10(vv_corrected)
        
    References:
        - Small, D. (2011). "Flattening Gamma: Radiometric Terrain Correction
          for SAR Imagery." IEEE TGRS.
        - Kellndorfer et al. (2016). "SAR Image Radiometric Terrain Correction."
    """
    if method == "none":
        return data.copy()
    
    if method != "cosine":
        raise ValueError(f"Unknown correction method: {method}")
    
    # Get transform and CRS
    transform = profile['transform']
    crs = profile['crs']
    height, width = data.shape
    
    # Create latitude array (approximate incidence angle from latitude)
    # Sentinel-1 IW mode: incidence angle varies ~29-46° across swath
    # We approximate this with latitude-based variation
    
    # Get center latitude
    if PYPROJ_AVAILABLE and crs is not None:
        from rasterio.transform import xy
        center_x, center_y = xy(transform, height // 2, width // 2)
        
        # Transform to lat/lon if needed
        if crs != 'EPSG:4326':
            from pyproj import Transformer
            transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
            center_lon, center_lat = transformer.transform(center_x, center_y)
        else:
            center_lat = center_y
    else:
        # Fallback: assume mid-latitudes
        center_lat = 0.0
    
    # Approximate incidence angle (degrees)
    # Sentinel-1 IW: 29-46° range, assume 37.5° mean + latitude variation
    # Higher latitudes -> slightly different geometry
    mean_incidence = 37.5  # degrees
    lat_variation = abs(center_lat) * 0.1  # Small latitude-dependent adjustment
    incidence_angle = mean_incidence + lat_variation
    
    # Convert to radians
    theta = np.radians(incidence_angle)
    
    # Apply cosine correction
    # σ0_flat = σ0_slope / cos(θ_local)
    # This approximates normalization to flat surface
    correction_factor = 1.0 / np.cos(theta)
    
    # Clip correction factor to reasonable range (avoid extreme values)
    correction_factor = np.clip(correction_factor, 0.5, 2.0)
    
    corrected = data * correction_factor
    
    return corrected


def convert_s1_to_db(data: np.ndarray, clip_min: float = -30.0) -> np.ndarray:
    """
    Convert Sentinel-1 linear backscatter to dB scale.
    
    Args:
        data: Backscatter in linear scale
        clip_min: Minimum dB value (avoid -inf from zero values)
        
    Returns:
        Backscatter in dB scale
    """
    # Avoid log(0)
    data_safe = np.maximum(data, 1e-10)
    
    # Convert to dB: 10 * log10(σ0)
    data_db = 10.0 * np.log10(data_safe)
    
    # Clip to avoid extreme values
    data_db = np.maximum(data_db, clip_min)
    
    return data_db


# ============================================================================
# Grid Alignment
# ============================================================================

def align_s1_to_s2_grid(
    s1_data: np.ndarray,
    s1_profile: Dict,
    s2_profile: Dict,
    resampling: str = "bilinear"
) -> np.ndarray:
    """
    Resample Sentinel-1 onto Sentinel-2 grid with exact alignment.
    
    Ensures S1 and S2 have identical:
    - CRS (coordinate reference system)
    - Transform (pixel alignment)
    - Shape (height, width)
    
    This is critical for pixel-wise fusion in the model.
    
    Args:
        s1_data: Sentinel-1 data array
        s1_profile: S1 rasterio profile
        s2_profile: S2 rasterio profile (target grid)
        resampling: Resampling method ("bilinear", "cubic", "nearest")
        
    Returns:
        S1 data resampled onto S2 grid
        
    Example:
        >>> s1_vv, s1_profile = read_cog("S1A_VV.tif")
        >>> s2_b04, s2_profile = read_cog("S2A_B04.tif")
        >>> 
        >>> s1_aligned = align_s1_to_s2_grid(s1_vv, s1_profile, s2_profile)
        >>> 
        >>> # Now s1_aligned and s2_b04 have identical grids
        >>> assert s1_aligned.shape == s2_b04.shape
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio required. Install with: pip install rasterio")
    
    # Extract profiles
    src_crs = s1_profile['crs']
    src_transform = s1_profile['transform']
    
    dst_crs = s2_profile['crs']
    dst_transform = s2_profile['transform']
    dst_height = s2_profile['height']
    dst_width = s2_profile['width']
    
    # Create destination array
    dst_data = np.zeros((dst_height, dst_width), dtype=s1_data.dtype)
    
    # Resampling method
    resampling_methods = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }
    resampling_method = resampling_methods.get(resampling, Resampling.bilinear)
    
    # Reproject
    reproject(
        source=s1_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling_method
    )
    
    return dst_data


# ============================================================================
# Complete Processing Pipeline
# ============================================================================

def process_s1_s2_pair(
    s1_path: Path,
    s2_path: Path,
    target_epsg: Optional[str] = None,
    target_resolution: float = 10.0,
    terrain_correct_s1: bool = True,
    s1_to_db: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Process S1 and S2 to common grid for fusion.
    
    Complete pipeline:
    1. Read S1 and S2 bands
    2. Reproject both to common UTM CRS
    3. Apply terrain correction to S1 (approximate)
    4. Convert S1 to dB scale
    5. Align S1 onto S2 grid (exact pixel alignment)
    
    Args:
        s1_path: Path to S1 band (VV or VH)
        s2_path: Path to S2 band (e.g., B04)
        target_epsg: Target CRS (None = auto-detect)
        target_resolution: Resolution in meters (default: 10m)
        terrain_correct_s1: Apply terrain correction to S1
        s1_to_db: Convert S1 to dB scale
        
    Returns:
        Tuple of (s1_processed, s2_processed, common_profile)
        
    Example:
        >>> s1_vv, s2_b04, profile = process_s1_s2_pair(
        ...     Path("data/raw/sentinel-1-grd/.../VV.tif"),
        ...     Path("data/raw/sentinel-2-l2a/.../B04.tif")
        ... )
        >>> 
        >>> # Now ready for model input
        >>> print(s1_vv.shape, s2_b04.shape)
        (10980, 10980) (10980, 10980)
        >>> print(profile['crs'])
        EPSG:32636
    """
    # Read data
    s1_data, s1_profile = read_cog(s1_path)
    s2_data, s2_profile = read_cog(s2_path)
    
    # Reproject S2 to UTM (reference grid)
    s2_utm, s2_utm_profile = reproject_to_utm(
        s2_data, s2_profile,
        target_epsg=target_epsg,
        target_resolution=target_resolution,
        resampling="cubic"  # High-quality for optical
    )
    
    # Reproject S1 to same UTM
    s1_utm, s1_utm_profile = reproject_to_utm(
        s1_data, s1_profile,
        target_epsg=s2_utm_profile['crs'],
        target_resolution=target_resolution,
        resampling="bilinear"  # Standard for SAR
    )
    
    # Terrain correction (approximate)
    if terrain_correct_s1:
        s1_utm = terrain_correct_s1_approx(s1_utm, s1_utm_profile)
    
    # Convert to dB
    if s1_to_db:
        s1_utm = convert_s1_to_db(s1_utm)
    
    # Align S1 onto S2 grid (exact alignment)
    s1_aligned = align_s1_to_s2_grid(s1_utm, s1_utm_profile, s2_utm_profile)
    
    return s1_aligned, s2_utm, s2_utm_profile


# ============================================================================
# Testing & Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 79)
    print("GEOSPATIAL UTILITIES TEST")
    print("=" * 79)
    print()
    
    # Test UTM detection
    print("TEST 1: UTM Zone Detection")
    print("-" * 79)
    
    test_bboxes = {
        "Lake Victoria": [31.5, -1.5, 34.0, 0.5],
        "Cairo": [31.0, 29.5, 32.0, 30.5],
        "Amazon Basin": [-70.0, -5.0, -60.0, 5.0],
    }
    
    for location, bbox in test_bboxes.items():
        utm = detect_utm_zone(bbox)
        print(f"{location}: {utm}")
    
    print()
    
    # Test terrain correction approximation
    print("TEST 2: Terrain Correction (Approximate)")
    print("-" * 79)
    print("Creating synthetic S1 backscatter...")
    
    # Synthetic S1 data (linear scale)
    synthetic_s1 = np.random.uniform(0.01, 0.5, (1000, 1000))
    
    # Mock profile
    mock_profile = {
        'crs': 'EPSG:32636',
        'transform': Affine(10, 0, 300000, 0, -10, 9000000),
        'height': 1000,
        'width': 1000,
    }
    
    # Apply correction
    s1_corrected = terrain_correct_s1_approx(synthetic_s1, mock_profile)
    
    print(f"Original range: [{synthetic_s1.min():.4f}, {synthetic_s1.max():.4f}]")
    print(f"Corrected range: [{s1_corrected.min():.4f}, {s1_corrected.max():.4f}]")
    
    # Convert to dB
    s1_db = convert_s1_to_db(s1_corrected)
    print(f"dB range: [{s1_db.min():.2f}, {s1_db.max():.2f}] dB")
    
    print()
    
    # Documentation
    print("=" * 79)
    print("TERRAIN CORRECTION NOTES")
    print("=" * 79)
    print()
    print("This module implements APPROXIMATE terrain correction for Sentinel-1.")
    print()
    print("Approximation Method:")
    print("  - Cosine correction based on estimated incidence angle")
    print("  - Assumes flat Earth approximation (no DEM)")
    print("  - Suitable for preliminary analysis and ML preprocessing")
    print()
    print("Limitations:")
    print("  - NOT rigorous radiometric terrain correction (RTC)")
    print("  - Does NOT account for actual topography")
    print("  - Approximates incidence angle from latitude")
    print()
    print("For Production RTC:")
    print("  - ESA SNAP: Range-Doppler Terrain Correction")
    print("  - GAMMA: Geocoding with DEM")
    print("  - ASF HyP3: Pre-processed RTC products")
    print()
    print("When Approximation is Acceptable:")
    print("  ✓ Flat or moderately sloping terrain")
    print("  ✓ Relative analysis within same scene")
    print("  ✓ Machine learning (models learn residual corrections)")
    print("  ✓ Preliminary analysis and prototyping")
    print()
    print("When Rigorous RTC is Required:")
    print("  ✗ Mountainous terrain (>10° slopes)")
    print("  ✗ Quantitative radiometric analysis")
    print("  ✗ Cross-scene comparison")
    print("  ✗ Change detection requiring absolute calibration")
    print()
    print("=" * 79)
    print()
    print("Usage Example:")
    print("  from axs_lib.geo import process_s1_s2_pair")
    print("  ")
    print("  s1, s2, profile = process_s1_s2_pair(")
    print("      s1_path=Path('S1A_VV.tif'),")
    print("      s2_path=Path('S2A_B04.tif'),")
    print("      terrain_correct_s1=True  # Approximate correction")
    print("  )")
    print("=" * 79)
