#!/usr/bin/env python3
"""
scripts/build_tiles.py - Complete Tile Building Pipeline

Orchestrates the full pipeline from place name or bbox to ready-to-train tiles:
1. Geocoding: Convert place name to bbox
2. STAC Search: Find satellite imagery
3. Download: Fetch imagery tiles
4. Align: Reproject to common grid
5. Mask: Apply cloud/shadow masks
6. Tile: Generate NPZ tiles with metadata
7. Split: Create train/val/test split index

Usage:
    # From place name
    python scripts/build_tiles.py \
        --place "Lake Victoria" \
        --date 2023-06-15 \
        --tile-size 256 \
        --overlap 32 \
        --max-cloud 30 \
        --split-ratio 0.7 0.15 0.15

    # From bbox
    python scripts/build_tiles.py \
        --bbox 31.5 -1.5 34.0 0.5 \
        --date 2023-06-15 \
        --tile-size 256 \
        --overlap 0 \
        --max-cloud 40 \
        --split-ratio 0.8 0.1 0.1

Output:
    - Tiles: data/tiles/{place}_{date}/tile_*.npz
    - Index: data/index/{place}_{date}_tiles.csv
    - Metadata: data/tiles/{place}_{date}/tile_*.json

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
import os
import argparse
import json
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from axs_lib
try:
    from axs_lib.geo import (
        read_cog, reproject_to_utm, detect_utm_zone, 
        align_s1_to_s2_grid, terrain_correct_s1_approx, convert_s1_to_db
    )
    from axs_lib.data import (
        create_tiles_from_arrays, TileConfig, TileMetadata
    )
    from axs_lib.cloudmask import (
        filter_by_cloud_cover, create_combined_mask, calculate_coverage
    )
except ImportError as e:
    print(f"ERROR: Failed to import axs_lib modules: {e}")
    print("Make sure axs_lib is in your Python path")
    sys.exit(1)

# Optional dependencies
try:
    import rasterio
    from rasterio.warp import transform_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    warnings.warn("rasterio not available - install with: pip install rasterio")

try:
    import pystac_client
    import planetary_computer
    HAS_STAC = True
except ImportError:
    HAS_STAC = False
    warnings.warn("STAC libraries not available - install with: pip install pystac-client planetary-computer")

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    HAS_GEOCODING = True
except ImportError:
    HAS_GEOCODING = False
    warnings.warn("geopy not available - install with: pip install geopy")

try:
    from scipy.ndimage import zoom as scipy_zoom
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available - install with: pip install scipy")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the tile building pipeline."""
    place: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    date: str = ""
    date_range_days: int = 7
    tile_size: int = 256
    overlap: int = 0
    max_cloud_percent: float = 40.0
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    output_dir: Path = Path("data/tiles")
    index_dir: Path = Path("data/index")
    cache_dir: Path = Path("cache")
    stac_provider: str = "planetary_computer"  # or "earthsearch"
    collections: List[str] = None
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        # Ensure we have either place or bbox
        if self.place is None and self.bbox is None:
            raise ValueError("Must provide either --place or --bbox")
        
        if self.place and self.bbox:
            raise ValueError("Provide either --place or --bbox, not both")
        
        # Validate split ratio
        if not np.isclose(sum(self.split_ratio), 1.0):
            raise ValueError(f"Split ratio must sum to 1.0, got {sum(self.split_ratio)}")
        
        # Default collections
        if self.collections is None:
            self.collections = ["sentinel-2-l2a", "sentinel-1-grd"]
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Step 1: Geocoding
# ============================================================================

def geocode_place(place: str) -> Tuple[float, float, float, float]:
    """
    Geocode place name to bounding box.
    
    Args:
        place: Place name (e.g., "Lake Victoria", "Paris, France")
        
    Returns:
        Bounding box tuple (west, south, east, north) in EPSG:4326
        
    Example:
        >>> bbox = geocode_place("Lake Victoria")
        >>> print(bbox)
        (31.5, -1.5, 34.0, 0.5)
    """
    if not HAS_GEOCODING:
        raise ImportError("geopy required for geocoding. Install with: pip install geopy")
    
    print(f"\n{'='*79}")
    print(f"STEP 1: GEOCODING")
    print(f"{'='*79}")
    print(f"Place: {place}")
    
    geolocator = Nominatim(user_agent="axion-sat-pipeline")
    
    try:
        location = geolocator.geocode(place, exactly_one=True, timeout=10)
        
        if location is None:
            raise ValueError(f"Could not geocode place: {place}")
        
        # Get bounding box
        if hasattr(location, 'raw') and 'boundingbox' in location.raw:
            # Nominatim returns [south, north, west, east]
            bb = location.raw['boundingbox']
            bbox = (float(bb[2]), float(bb[0]), float(bb[3]), float(bb[1]))  # Convert to [west, south, east, north]
        else:
            # Fallback: create small bbox around point
            lat, lon = location.latitude, location.longitude
            delta = 0.1  # ~11km buffer
            bbox = (lon - delta, lat - delta, lon + delta, lat + delta)
        
        print(f"✓ Geocoded to bbox: {bbox}")
        print(f"  Center: ({location.latitude:.4f}, {location.longitude:.4f})")
        print(f"  Address: {location.address}")
        
        return bbox
        
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        raise RuntimeError(f"Geocoding failed: {e}")


# ============================================================================
# Step 2: STAC Search
# ============================================================================

def search_stac(
    bbox: Tuple[float, float, float, float],
    date: str,
    date_range_days: int,
    collections: List[str],
    stac_provider: str = "planetary_computer",
    max_cloud: float = 40.0
) -> Dict[str, List[Any]]:
    """
    Search STAC catalog for imagery.
    
    Args:
        bbox: Bounding box (west, south, east, north)
        date: Center date (YYYY-MM-DD)
        date_range_days: Search window (days before/after)
        collections: STAC collection names
        stac_provider: "planetary_computer" or "earthsearch"
        max_cloud: Maximum cloud cover percentage
        
    Returns:
        Dictionary mapping collection names to lists of STAC items
    """
    if not HAS_STAC:
        raise ImportError("pystac-client required. Install with: pip install pystac-client planetary-computer")
    
    print(f"\n{'='*79}")
    print(f"STEP 2: STAC SEARCH")
    print(f"{'='*79}")
    print(f"Provider: {stac_provider}")
    print(f"Collections: {collections}")
    print(f"Date: {date} ± {date_range_days} days")
    print(f"Max cloud: {max_cloud}%")
    
    # Parse date
    center_date = datetime.fromisoformat(date)
    start_date = center_date - timedelta(days=date_range_days)
    end_date = center_date + timedelta(days=date_range_days)
    
    date_range = f"{start_date.date()}/{end_date.date()}"
    
    # Connect to STAC
    if stac_provider == "planetary_computer":
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace
        )
    elif stac_provider == "earthsearch":
        catalog = pystac_client.Client.open(
            "https://earth-search.aws.element84.com/v1"
        )
    else:
        raise ValueError(f"Unknown STAC provider: {stac_provider}")
    
    results = {}
    
    for collection in collections:
        print(f"\nSearching {collection}...")
        
        # Build query
        query_params = {
            "bbox": bbox,
            "datetime": date_range,
            "collections": [collection],
        }
        
        # Add cloud filter for optical
        if "sentinel-2" in collection.lower():
            query_params["query"] = {
                "eo:cloud_cover": {"lt": max_cloud}
            }
        
        # Search
        search = catalog.search(**query_params)
        items = list(search.items())
        
        results[collection] = items
        print(f"  Found {len(items)} items")
        
        # Print first few items
        for i, item in enumerate(items[:3]):
            print(f"    {i+1}. {item.id} ({item.datetime.date() if item.datetime else 'no date'})")
    
    return results


# ============================================================================
# Step 3: Download
# ============================================================================

def download_assets(
    stac_items: Dict[str, List[Any]],
    cache_dir: Path,
    bbox: Tuple[float, float, float, float]
) -> Dict[str, Path]:
    """
    Download assets from STAC items using COG windowed reading.
    
    Args:
        stac_items: Dictionary mapping collection to STAC items
        cache_dir: Cache directory
        bbox: Bounding box for windowed reading (west, south, east, north)
        
    Returns:
        Dictionary mapping band names to local file paths
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio required for downloading. Install with: pip install rasterio")
    
    print(f"\n{'='*79}")
    print(f"STEP 3: DOWNLOAD")
    print(f"{'='*79}")
    print(f"Cache dir: {cache_dir}")
    print(f"BBox: {bbox}")
    
    downloaded = {}
    
    # Process Sentinel-2 items
    if 'sentinel-2-l2a' in stac_items and stac_items['sentinel-2-l2a']:
        s2_items = stac_items['sentinel-2-l2a']
        print(f"\nProcessing Sentinel-2 ({len(s2_items)} items)")
        
        # Use the first item with lowest cloud cover
        best_item = min(s2_items, key=lambda x: x.properties.get('eo:cloud_cover', 100))
        print(f"  Selected: {best_item.id}")
        print(f"  Date: {best_item.datetime.date() if best_item.datetime else 'unknown'}")
        print(f"  Cloud cover: {best_item.properties.get('eo:cloud_cover', 'unknown')}%")
        
        # Download required bands
        s2_bands = {
            'B02': 's2_b2',  # Blue
            'B03': 's2_b3',  # Green
            'B04': 's2_b4',  # Red
            'B08': 's2_b8',  # NIR
            'SCL': 'scl',    # Scene Classification
        }
        
        for asset_key, band_name in s2_bands.items():
            if asset_key in best_item.assets:
                asset = best_item.assets[asset_key]
                href = asset.href
                
                # Create cache filename
                item_id_safe = best_item.id.replace('/', '_').replace(':', '_')
                cache_file = cache_dir / f"{item_id_safe}_{asset_key}.tif"
                
                print(f"  Downloading {asset_key}...", end=' ', flush=True)
                
                try:
                    # Download with windowed reading
                    _download_cog_windowed(href, cache_file, bbox)
                    downloaded[band_name] = cache_file
                    print("✓")
                except Exception as e:
                    print(f"✗ Error: {e}")
    
    # Process Sentinel-1 items
    if 'sentinel-1-grd' in stac_items and stac_items['sentinel-1-grd']:
        s1_items = stac_items['sentinel-1-grd']
        print(f"\nProcessing Sentinel-1 ({len(s1_items)} items)")
        
        # Use the first item
        if s1_items:
            best_item = s1_items[0]
            print(f"  Selected: {best_item.id}")
            print(f"  Date: {best_item.datetime.date() if best_item.datetime else 'unknown'}")
            
            # Download VV and VH polarizations
            s1_bands = {
                'vv': 's1_vv',
                'vh': 's1_vh',
            }
            
            for asset_key, band_name in s1_bands.items():
                if asset_key in best_item.assets:
                    asset = best_item.assets[asset_key]
                    href = asset.href
                    
                    # Create cache filename
                    item_id_safe = best_item.id.replace('/', '_').replace(':', '_')
                    cache_file = cache_dir / f"{item_id_safe}_{asset_key}.tif"
                    
                    print(f"  Downloading {asset_key}...", end=' ', flush=True)
                    
                    try:
                        _download_cog_windowed(href, cache_file, bbox)
                        downloaded[band_name] = cache_file
                        print("✓")
                    except Exception as e:
                        print(f"✗ Error: {e}")
    
    print(f"\n✓ Downloaded {len(downloaded)} assets")
    return downloaded


def _download_cog_windowed(
    href: str,
    output_path: Path,
    bbox: Tuple[float, float, float, float]
):
    """
    Download COG with windowed reading to save bandwidth and disk space.
    
    Args:
        href: COG URL (supports /vsicurl/ virtual filesystem)
        output_path: Local output path
        bbox: Bounding box (west, south, east, north) in EPSG:4326
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from rasterio.env import Env
    
    # Use /vsicurl/ for efficient COG reading
    if not href.startswith('/vsicurl/'):
        href = f'/vsicurl/{href}'
    
    # Configure rasterio for COG reading
    env_options = {
        'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif,.tiff',
        'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
        'CPL_VSIL_CURL_USE_HEAD': 'NO',
        'GDAL_HTTP_MAX_RETRY': '3',
        'GDAL_HTTP_RETRY_DELAY': '1',
    }
    
    with Env(**env_options):
        with rasterio.open(href) as src:
            # Check if CRS is valid
            if src.crs is None:
                # Some Sentinel-1 assets may not have CRS metadata
                # Download full extent without windowing
                print("Warning: CRS not found, downloading full extent")
                data = src.read(1)
                transform = src.transform
                width = src.width
                height = src.height
            else:
                # Transform bbox to source CRS
                src_bbox = transform_bounds('EPSG:4326', src.crs, *bbox)
                
                # Get window from bounds
                try:
                    window = from_bounds(*src_bbox, transform=src.transform)
                except Exception:
                    # If window is outside bounds, use full extent
                    window = None
                
                # Read data
                if window:
                    data = src.read(1, window=window)
                    transform = src.window_transform(window)
                    width = int(window.width)
                    height = int(window.height)
                else:
                    data = src.read(1)
                    transform = src.transform
                    width = src.width
                    height = src.height
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to local file
            profile = src.profile.copy()
            profile.update({
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'transform': transform,
                'compress': 'lzw',
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256,
            })
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)


# ============================================================================
# Step 4: Align
# ============================================================================

def align_bands(
    band_files: Dict[str, Path],
    target_epsg: Optional[str] = None,
    resolution: float = 10.0
) -> Tuple[Dict[str, np.ndarray], List[float], str]:
    """
    Align all bands to common grid.
    
    Args:
        band_files: Dictionary mapping band names to file paths
        target_epsg: Target CRS (None = auto-detect from S2)
        resolution: Target resolution in meters
        
    Returns:
        Tuple of (aligned arrays dict, geotransform, crs)
    """
    print(f"\n{'='*79}")
    print(f"STEP 4: ALIGN")
    print(f"{'='*79}")
    print(f"Target resolution: {resolution}m")
    
    if not band_files:
        raise ValueError("No band files to align")
    
    aligned_arrays = {}
    reference_profile = None
    
    # Step 1: Process Sentinel-2 bands (they are already aligned)
    s2_bands = ['s2_b2', 's2_b3', 's2_b4', 's2_b8']
    s2_reference = None
    
    for band_name in s2_bands:
        if band_name in band_files:
            print(f"  Reading {band_name}...", end=' ', flush=True)
            data, profile = read_cog(band_files[band_name])
            
            # Auto-detect target EPSG from first S2 band
            if target_epsg is None and reference_profile is None:
                target_epsg = str(profile['crs'])
                print(f"(detected CRS: {target_epsg})")
            else:
                print("✓")
            
            # Reproject to target resolution if needed
            if profile.get('transform')[0] != resolution:
                print(f"    Reprojecting to {resolution}m...")
                data, profile = reproject_to_utm(
                    data, profile,
                    target_epsg=target_epsg,
                    target_resolution=resolution
                )
            
            aligned_arrays[band_name] = data
            
            # Use first S2 band as reference
            if reference_profile is None:
                reference_profile = profile
                s2_reference = (data, profile)
    
    if reference_profile is None:
        raise ValueError("No Sentinel-2 bands found for alignment reference")
    
    # Step 2: Process Sentinel-1 bands (align to S2 grid)
    s1_bands = ['s1_vv', 's1_vh']
    
    for band_name in s1_bands:
        if band_name in band_files:
            print(f"  Reading {band_name}...", end=' ', flush=True)
            try:
                data, profile = read_cog(band_files[band_name])
                
                # Check if CRS is valid
                if profile.get('crs') is None:
                    print("⚠ Skipping (no CRS metadata)")
                    continue
                
                print("✓")
                
                # Reproject to target CRS and resolution
                print(f"    Reprojecting to {target_epsg} @ {resolution}m...")
                data, profile = reproject_to_utm(
                    data, profile,
                    target_epsg=target_epsg,
                    target_resolution=resolution
                )
            except Exception as e:
                print(f"✗ Error: {e}")
                print(f"    Skipping {band_name}")
                continue
            
            # Align to S2 grid
            print(f"    Aligning to S2 grid...")
            data = align_s1_to_s2_grid(data, profile, reference_profile)
            
            # Apply terrain correction approximation
            print(f"    Applying terrain correction...")
            data = terrain_correct_s1_approx(data, reference_profile)
            
            # Convert to dB scale
            print(f"    Converting to dB...")
            data = convert_s1_to_db(data, clip_min=-30.0)
            
            aligned_arrays[band_name] = data
    
    # Extract geotransform
    transform = reference_profile['transform']
    geotransform = [
        transform[2],  # x_origin
        transform[0],  # pixel_width
        transform[1],  # rotation (usually 0)
        transform[5],  # y_origin
        transform[3],  # rotation (usually 0)
        transform[4]   # pixel_height (negative)
    ]
    
    crs = str(reference_profile['crs'])
    
    print(f"\n✓ Aligned {len(aligned_arrays)} bands to common grid")
    print(f"  CRS: {crs}")
    print(f"  Resolution: {resolution}m")
    print(f"  Grid shape: {reference_profile['height']} x {reference_profile['width']}")
    
    return aligned_arrays, geotransform, crs


def _resize_nearest_numpy(array: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resize array using nearest neighbor interpolation (pure numpy).
    
    Args:
        array: Input 2D array
        target_shape: Target (height, width)
        
    Returns:
        Resized array
    """
    h_in, w_in = array.shape
    h_out, w_out = target_shape
    
    # Create coordinate arrays
    row_indices = np.round(np.linspace(0, h_in - 1, h_out)).astype(int)
    col_indices = np.round(np.linspace(0, w_in - 1, w_out)).astype(int)
    
    # Index array
    return array[row_indices[:, None], col_indices[None, :]]


# ============================================================================
# Step 5: Cloud Masking
# ============================================================================

def apply_cloud_mask(
    arrays: Dict[str, np.ndarray],
    scl_array: np.ndarray,
    max_cloud_percent: float
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Apply cloud masking to arrays.
    
    Args:
        arrays: Dictionary of aligned arrays
        scl_array: Sentinel-2 SCL band
        max_cloud_percent: Maximum acceptable cloud cover
        
    Returns:
        Tuple of (masked arrays, quality stats)
    """
    print(f"\n{'='*79}")
    print(f"STEP 5: CLOUD MASKING")
    print(f"{'='*79}")
    
    # Check quality
    is_valid, stats = filter_by_cloud_cover(
        scl_array,
        max_cloud_percent=max_cloud_percent,
        verbose=True
    )
    
    if not is_valid:
        print(f"⚠ WARNING: Scene exceeds cloud threshold")
    
    # Create combined mask
    bad_mask = create_combined_mask(
        scl_array,
        mask_clouds=True,
        mask_shadows=True,
        mask_invalid=True
    )
    
    # Apply mask to all arrays
    masked_arrays = {}
    for key, arr in arrays.items():
        masked_arrays[key] = np.where(bad_mask, np.nan, arr)
    
    print(f"✓ Applied mask ({np.sum(bad_mask)} / {bad_mask.size} pixels masked)")
    
    return masked_arrays, stats


# ============================================================================
# Step 6: Tiling
# ============================================================================

def create_tiles(
    arrays: Dict[str, np.ndarray],
    geotransform: List[float],
    crs: str,
    config: PipelineConfig,
    prefix: str
) -> List[Path]:
    """
    Create tiles from aligned arrays.
    
    Args:
        arrays: Dictionary of aligned masked arrays
        geotransform: GDAL geotransform
        crs: Coordinate reference system
        config: Pipeline configuration
        prefix: Tile filename prefix
        
    Returns:
        List of tile paths
    """
    print(f"\n{'='*79}")
    print(f"STEP 6: TILING")
    print(f"{'='*79}")
    print(f"Tile size: {config.tile_size}x{config.tile_size}")
    print(f"Overlap: {config.overlap} pixels")
    
    # Calculate stride
    stride = config.tile_size - config.overlap
    
    # Create tile config
    tile_config = TileConfig(
        tile_size=config.tile_size,
        stride=stride,
        min_valid_pixels=int(config.tile_size * config.tile_size * 0.5),  # Require 50% valid
        dtype='float32'
    )
    
    # Create output directory
    output_dir = config.output_dir / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tiles
    tile_paths = create_tiles_from_arrays(
        arrays=arrays,
        base_geotransform=geotransform,
        crs=crs,
        config=tile_config,
        output_dir=output_dir,
        prefix=prefix
    )
    
    print(f"✓ Created {len(tile_paths)} tiles in {output_dir}")
    
    return tile_paths


# ============================================================================
# Step 7: Train/Val/Test Split
# ============================================================================

def create_split_index(
    tile_paths: List[Path],
    config: PipelineConfig,
    output_name: str
) -> Path:
    """
    Create train/val/test split index CSV.
    
    Args:
        tile_paths: List of tile NPZ paths
        config: Pipeline configuration
        output_name: Output filename (without .csv)
        
    Returns:
        Path to index CSV file
    """
    print(f"\n{'='*79}")
    print(f"STEP 7: TRAIN/VAL/TEST SPLIT")
    print(f"{'='*79}")
    print(f"Split ratio: {config.split_ratio}")
    
    # Set random seed
    np.random.seed(config.random_seed)
    
    # Shuffle tiles
    tile_paths = list(tile_paths)
    np.random.shuffle(tile_paths)
    
    # Calculate split indices
    n_tiles = len(tile_paths)
    train_ratio, val_ratio, test_ratio = config.split_ratio
    
    n_train = int(n_tiles * train_ratio)
    n_val = int(n_tiles * val_ratio)
    n_test = n_tiles - n_train - n_val  # Remainder goes to test
    
    print(f"Total tiles: {n_tiles}")
    print(f"  Train: {n_train} ({n_train/n_tiles*100:.1f}%)")
    print(f"  Val:   {n_val} ({n_val/n_tiles*100:.1f}%)")
    print(f"  Test:  {n_test} ({n_test/n_tiles*100:.1f}%)")
    
    # Create index CSV
    index_path = config.index_dir / f"{output_name}_tiles.csv"
    
    with open(index_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tile_path', 'tile_id', 'split'])
        
        # Write train
        for i in range(n_train):
            tile_path = tile_paths[i]
            writer.writerow([str(tile_path), tile_path.stem, 'train'])
        
        # Write val
        for i in range(n_train, n_train + n_val):
            tile_path = tile_paths[i]
            writer.writerow([str(tile_path), tile_path.stem, 'val'])
        
        # Write test
        for i in range(n_train + n_val, n_tiles):
            tile_path = tile_paths[i]
            writer.writerow([str(tile_path), tile_path.stem, 'test'])
    
    print(f"✓ Saved split index to {index_path}")
    
    return index_path


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(config: PipelineConfig):
    """
    Run the complete tile building pipeline.
    
    Args:
        config: Pipeline configuration
    """
    print(f"\n{'#'*79}")
    print(f"# AXION-SAT TILE BUILDING PIPELINE")
    print(f"{'#'*79}")
    print(f"\nConfiguration:")
    print(f"  Place/BBox: {config.place or config.bbox}")
    print(f"  Date: {config.date}")
    print(f"  Tile size: {config.tile_size}")
    print(f"  Overlap: {config.overlap}")
    print(f"  Max cloud: {config.max_cloud_percent}%")
    print(f"  Split: {config.split_ratio}")
    
    # Step 1: Geocoding (if place provided)
    if config.place:
        bbox = geocode_place(config.place)
        output_name = config.place.replace(' ', '_').replace(',', '').lower()
    else:
        bbox = config.bbox
        output_name = f"bbox_{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}"
    
    # Add date to output name
    output_name = f"{output_name}_{config.date}"
    
    # Step 2: STAC Search
    stac_items = search_stac(
        bbox=bbox,
        date=config.date,
        date_range_days=config.date_range_days,
        collections=config.collections,
        stac_provider=config.stac_provider,
        max_cloud=config.max_cloud_percent
    )
    
    if not any(stac_items.values()):
        print("\n✗ ERROR: No imagery found for query")
        return
    
    # Step 3: Download
    downloaded = download_assets(
        stac_items=stac_items,
        cache_dir=config.cache_dir,
        bbox=bbox
    )
    
    if not downloaded:
        print("\n✗ ERROR: No assets downloaded")
        return
    
    # Step 4: Align
    arrays, geotransform, target_crs = align_bands(
        band_files=downloaded,
        target_epsg=None,  # Auto-detect from S2
        resolution=10.0
    )
    
    if not arrays:
        print("\n✗ ERROR: No bands aligned")
        return
    
    # Step 5: Cloud Masking
    if 'scl' in downloaded:
        print(f"\nLoading SCL for cloud masking...")
        scl_data, _ = read_cog(downloaded['scl'])
        
        # Reproject SCL to match other bands if needed
        target_shape = next(iter(arrays.values())).shape
        if scl_data.shape != target_shape:
            print(f"  Resizing SCL from {scl_data.shape} to {target_shape}")
            
            if HAS_SCIPY:
                zoom_factor = (
                    target_shape[0] / scl_data.shape[0],
                    target_shape[1] / scl_data.shape[1]
                )
                scl_data = scipy_zoom(scl_data, zoom_factor, order=0)  # Nearest neighbor for categorical
            else:
                # Fallback: simple nearest neighbor resize with numpy
                print(f"  ⚠ WARNING: scipy not available, using numpy resize")
                scl_data = _resize_nearest_numpy(scl_data, target_shape)
        
        masked_arrays, quality_stats = apply_cloud_mask(
            arrays=arrays,
            scl_array=scl_data,
            max_cloud_percent=config.max_cloud_percent
        )
    else:
        print(f"\n⚠ WARNING: No SCL available, skipping cloud masking")
        masked_arrays = arrays
        quality_stats = {'cloud_percent': 0.0}
    
    # Step 6: Tiling
    
    tile_paths = create_tiles(
        arrays=masked_arrays,
        geotransform=geotransform,
        crs=target_crs,
        config=config,
        prefix=output_name
    )
    
    if not tile_paths:
        print("\n✗ ERROR: No tiles created")
        return
    
    # Step 7: Train/Val/Test Split
    index_path = create_split_index(
        tile_paths=tile_paths,
        config=config,
        output_name=output_name
    )
    
    # Summary
    print(f"\n{'='*79}")
    print(f"PIPELINE COMPLETE ✓")
    print(f"{'='*79}")
    print(f"\nOutput:")
    print(f"  Tiles: {config.output_dir / output_name}")
    print(f"  Index: {index_path}")
    print(f"  Total tiles: {len(tile_paths)}")
    print(f"\nNext steps:")
    print(f"  1. Inspect tiles with visualization tools")
    print(f"  2. Train model using index CSV")
    print(f"  3. Evaluate on test split")
    print()


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build satellite imagery tiles with train/val/test split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From place name
  python scripts/build_tiles.py \\
      --place "Lake Victoria" \\
      --date 2023-06-15 \\
      --tile-size 256 \\
      --overlap 32 \\
      --max-cloud 30

  # From bbox
  python scripts/build_tiles.py \\
      --bbox 31.5 -1.5 34.0 0.5 \\
      --date 2023-06-15 \\
      --tile-size 512 \\
      --overlap 0 \\
      --split-ratio 0.8 0.1 0.1
        """
    )
    
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--place',
        type=str,
        help='Place name to geocode (e.g., "Lake Victoria", "Paris, France")'
    )
    input_group.add_argument(
        '--bbox',
        type=float,
        nargs=4,
        metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'),
        help='Bounding box in EPSG:4326 (west south east north)'
    )
    
    # Date
    parser.add_argument(
        '--date',
        type=str,
        required=True,
        help='Acquisition date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--date-range',
        type=int,
        default=7,
        metavar='DAYS',
        help='Search window ±days around date (default: 7)'
    )
    
    # Tiling
    parser.add_argument(
        '--tile-size',
        type=int,
        default=256,
        help='Tile size in pixels (default: 256)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=0,
        help='Overlap between tiles in pixels (default: 0)'
    )
    
    # Quality
    parser.add_argument(
        '--max-cloud',
        type=float,
        default=40.0,
        help='Maximum cloud cover percentage (default: 40.0)'
    )
    
    # Split
    parser.add_argument(
        '--split-ratio',
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=('TRAIN', 'VAL', 'TEST'),
        help='Train/val/test split ratios (default: 0.7 0.15 0.15)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for splitting (default: 42)'
    )
    
    # Directories
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/tiles'),
        help='Output directory for tiles (default: data/tiles)'
    )
    parser.add_argument(
        '--index-dir',
        type=Path,
        default=Path('data/index'),
        help='Output directory for index CSV (default: data/index)'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path('cache'),
        help='Cache directory for downloads (default: cache)'
    )
    
    # STAC
    parser.add_argument(
        '--stac-provider',
        type=str,
        choices=['planetary_computer', 'earthsearch'],
        default='planetary_computer',
        help='STAC provider (default: planetary_computer)'
    )
    parser.add_argument(
        '--collections',
        type=str,
        nargs='+',
        default=['sentinel-2-l2a', 'sentinel-1-grd'],
        help='STAC collections to search (default: sentinel-2-l2a sentinel-1-grd)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Create config
        config = PipelineConfig(
            place=args.place,
            bbox=tuple(args.bbox) if args.bbox else None,
            date=args.date,
            date_range_days=args.date_range,
            tile_size=args.tile_size,
            overlap=args.overlap,
            max_cloud_percent=args.max_cloud,
            split_ratio=tuple(args.split_ratio),
            output_dir=args.output_dir,
            index_dir=args.index_dir,
            cache_dir=args.cache_dir,
            stac_provider=args.stac_provider,
            collections=args.collections,
            random_seed=args.random_seed
        )
        
        # Run pipeline
        run_pipeline(config)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
