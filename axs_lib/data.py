"""
axs_lib/data.py - Data Tiling and Batching

Produces NPZ tiles from aligned multi-source satellite imagery arrays.
Tiles contain standardized keys for Sentinel-1, Sentinel-2, and ancillary data.

Features:
    - Tile generation from large aligned rasters
    - Overlap/stride control for seamless mosaicking
    - Geotransform tracking via JSON sidecars
    - Quality filtering integration
    - Memory-efficient windowed reading
    - Batch processing for datasets

Tile Keys:
    - s1_vv: Sentinel-1 VV polarization
    - s1_vh: Sentinel-1 VH polarization
    - s2_b2: Sentinel-2 Blue (10m)
    - s2_b3: Sentinel-2 Green (10m)
    - s2_b4: Sentinel-2 Red (10m)
    - s2_b8: Sentinel-2 NIR (10m)
    - month: Month of acquisition (1-12)
    - inc_angle: Sentinel-1 incidence angle
    - biome_code: Biome classification code
    - dem_slope: Terrain slope (degrees)
    - dem_aspect: Terrain aspect (degrees)

Usage:
    >>> from axs_lib.data import create_tiles_from_files
    >>> 
    >>> tile_paths = create_tiles_from_files(
    ...     s1_vv='path/to/s1_vv.tif',
    ...     s1_vh='path/to/s1_vh.tif',
    ...     s2_bands=['b2.tif', 'b3.tif', 'b4.tif', 'b8.tif'],
    ...     dem_slope='slope.tif',
    ...     dem_aspect='aspect.tif',
    ...     tile_size=256,
    ...     output_dir='tiles/'
    ... )

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TileMetadata:
    """Metadata for a single tile."""
    tile_id: str
    row: int
    col: int
    width: int
    height: int
    geotransform: List[float]
    crs: str
    timestamp: str
    source_files: Dict[str, str]
    tile_bounds: Dict[str, float]  # minx, miny, maxx, maxy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @staticmethod
    def compute_tile_geotransform(
        base_geotransform: List[float],
        col_offset: int,
        row_offset: int
    ) -> List[float]:
        """
        Compute geotransform for a tile given base geotransform and offsets.
        
        Args:
            base_geotransform: [x_origin, pixel_width, 0, y_origin, 0, -pixel_height]
            col_offset: Column offset in pixels
            row_offset: Row offset in pixels
            
        Returns:
            Tile geotransform with updated origin
        """
        x_origin = base_geotransform[0] + col_offset * base_geotransform[1]
        y_origin = base_geotransform[3] + row_offset * base_geotransform[5]
        
        return [
            x_origin,
            base_geotransform[1],
            base_geotransform[2],
            y_origin,
            base_geotransform[4],
            base_geotransform[5]
        ]
    
    @staticmethod
    def compute_tile_bounds(geotransform: List[float], width: int, height: int) -> Dict[str, float]:
        """
        Compute geographic bounds from geotransform.
        
        Args:
            geotransform: GDAL-style geotransform
            width: Tile width in pixels
            height: Tile height in pixels
            
        Returns:
            Dictionary with minx, miny, maxx, maxy
        """
        minx = geotransform[0]
        maxy = geotransform[3]
        maxx = minx + width * geotransform[1]
        miny = maxy + height * geotransform[5]  # negative pixel height
        
        return {
            'minx': minx,
            'miny': miny,
            'maxx': maxx,
            'maxy': maxy
        }


@dataclass
class TileConfig:
    """Configuration for tile generation."""
    tile_size: int = 256
    stride: Optional[int] = None  # If None, stride = tile_size (no overlap)
    min_valid_pixels: int = 0  # Minimum non-NaN pixels required
    edge_padding: int = 0  # Padding to add around each tile
    normalize: bool = False  # Normalize values to [0, 1]
    dtype: str = 'float32'  # Output data type
    
    def __post_init__(self):
        """Set defaults after initialization."""
        if self.stride is None:
            self.stride = self.tile_size


# ============================================================================
# Tiling Functions
# ============================================================================

def create_tile(
    arrays: Dict[str, np.ndarray],
    row: int,
    col: int,
    config: TileConfig
) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract a single tile from aligned arrays.
    
    Args:
        arrays: Dictionary of aligned arrays (all same shape)
        row: Starting row index
        col: Starting column index
        config: Tile configuration
        
    Returns:
        Dictionary of tile arrays, or None if insufficient valid pixels
        
    Example:
        >>> arrays = {
        ...     's2_b4': np.random.rand(1000, 1000),
        ...     's2_b8': np.random.rand(1000, 1000)
        ... }
        >>> config = TileConfig(tile_size=256)
        >>> tile = create_tile(arrays, row=0, col=0, config=config)
    """
    tile_data = {}
    
    # Extract tile from each array
    for key, array in arrays.items():
        # Handle edge cases where tile extends beyond array bounds
        row_end = min(row + config.tile_size, array.shape[0])
        col_end = min(col + config.tile_size, array.shape[1])
        
        tile = array[row:row_end, col:col_end].copy()
        
        # Pad if needed (for edge tiles)
        if tile.shape != (config.tile_size, config.tile_size):
            padded = np.full((config.tile_size, config.tile_size), np.nan, dtype=tile.dtype)
            padded[:tile.shape[0], :tile.shape[1]] = tile
            tile = padded
        
        tile_data[key] = tile
    
    # Check minimum valid pixels (using first band as reference)
    if config.min_valid_pixels > 0:
        first_band = next(iter(tile_data.values()))
        valid_pixels = np.sum(~np.isnan(first_band))
        if valid_pixels < config.min_valid_pixels:
            return None
    
    # Normalize if requested
    if config.normalize:
        for key in tile_data:
            tile_data[key] = normalize_array(tile_data[key])
    
    # Convert dtype
    for key in tile_data:
        tile_data[key] = tile_data[key].astype(config.dtype)
    
    return tile_data


def normalize_array(arr: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize array to [0, 1] range.
    
    Args:
        arr: Input array
        method: 'minmax' or 'percentile'
        
    Returns:
        Normalized array
    """
    if method == 'minmax':
        arr_min = np.nanmin(arr)
        arr_max = np.nanmax(arr)
        if arr_max > arr_min:
            return (arr - arr_min) / (arr_max - arr_min)
        return arr
    
    elif method == 'percentile':
        p2 = np.nanpercentile(arr, 2)
        p98 = np.nanpercentile(arr, 98)
        if p98 > p2:
            arr_norm = (arr - p2) / (p98 - p2)
            return np.clip(arr_norm, 0, 1)
        return arr
    
    return arr


def create_tiles_from_arrays(
    arrays: Dict[str, np.ndarray],
    base_geotransform: List[float],
    crs: str,
    config: TileConfig,
    output_dir: Union[str, Path],
    prefix: str = 'tile',
    source_files: Optional[Dict[str, str]] = None
) -> List[Path]:
    """
    Create tiles from aligned arrays and save as NPZ with JSON sidecars.
    
    Args:
        arrays: Dictionary of aligned arrays with standardized keys
        base_geotransform: GDAL geotransform of source data
        crs: Coordinate reference system (e.g., 'EPSG:32610')
        config: Tile configuration
        output_dir: Output directory for tiles
        prefix: Filename prefix for tiles
        source_files: Optional dict mapping keys to source file paths
        
    Returns:
        List of paths to created tile NPZ files
        
    Example:
        >>> arrays = {
        ...     's1_vv': np.random.rand(1000, 1000),
        ...     's2_b4': np.random.rand(1000, 1000),
        ...     'dem_slope': np.random.rand(1000, 1000)
        ... }
        >>> geotransform = [500000, 10, 0, 4500000, 0, -10]
        >>> config = TileConfig(tile_size=256, stride=128)
        >>> tiles = create_tiles_from_arrays(
        ...     arrays, geotransform, 'EPSG:32610', config, 'output/tiles/'
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get array shape (all arrays must be same shape)
    first_array = next(iter(arrays.values()))
    height, width = first_array.shape
    
    # Validate all arrays have same shape
    for key, arr in arrays.items():
        if arr.shape != (height, width):
            raise ValueError(f"Array '{key}' has shape {arr.shape}, expected {(height, width)}")
    
    tile_paths = []
    tile_count = 0
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    # Iterate over grid
    for row in range(0, height, config.stride):
        for col in range(0, width, config.stride):
            # Extract tile
            tile_data = create_tile(arrays, row, col, config)
            
            if tile_data is None:
                continue  # Skip tiles with insufficient valid pixels
            
            # Generate tile ID
            tile_id = f"{prefix}_r{row:05d}_c{col:05d}"
            
            # Compute tile geotransform
            tile_geotransform = TileMetadata.compute_tile_geotransform(
                base_geotransform, col, row
            )
            
            # Compute tile bounds
            tile_bounds = TileMetadata.compute_tile_bounds(
                tile_geotransform, config.tile_size, config.tile_size
            )
            
            # Create metadata
            metadata = TileMetadata(
                tile_id=tile_id,
                row=row,
                col=col,
                width=config.tile_size,
                height=config.tile_size,
                geotransform=tile_geotransform,
                crs=crs,
                timestamp=timestamp,
                source_files=source_files or {},
                tile_bounds=tile_bounds
            )
            
            # Save NPZ tile
            npz_path = output_dir / f"{tile_id}.npz"
            np.savez_compressed(npz_path, **tile_data)
            
            # Save JSON sidecar
            json_path = output_dir / f"{tile_id}.json"
            with open(json_path, 'w') as f:
                f.write(metadata.to_json())
            
            tile_paths.append(npz_path)
            tile_count += 1
    
    print(f"Created {tile_count} tiles in {output_dir}")
    return tile_paths


def create_tiles_from_files(
    s1_vv: Optional[Union[str, Path]] = None,
    s1_vh: Optional[Union[str, Path]] = None,
    s2_b2: Optional[Union[str, Path]] = None,
    s2_b3: Optional[Union[str, Path]] = None,
    s2_b4: Optional[Union[str, Path]] = None,
    s2_b8: Optional[Union[str, Path]] = None,
    month: Optional[Union[int, Union[str, Path]]] = None,
    inc_angle: Optional[Union[str, Path]] = None,
    biome_code: Optional[Union[str, Path]] = None,
    dem_slope: Optional[Union[str, Path]] = None,
    dem_aspect: Optional[Union[str, Path]] = None,
    config: Optional[TileConfig] = None,
    output_dir: Union[str, Path] = 'tiles/',
    prefix: str = 'tile'
) -> List[Path]:
    """
    Create tiles from raster files.
    
    This function reads aligned raster files and produces NPZ tiles.
    Requires axs_lib.geo module for raster I/O.
    
    Args:
        s1_vv: Sentinel-1 VV polarization file
        s1_vh: Sentinel-1 VH polarization file
        s2_b2: Sentinel-2 Blue band file
        s2_b3: Sentinel-2 Green band file
        s2_b4: Sentinel-2 Red band file
        s2_b8: Sentinel-2 NIR band file
        month: Month value (int) or raster file with month values
        inc_angle: Sentinel-1 incidence angle file
        biome_code: Biome classification file
        dem_slope: DEM slope file
        dem_aspect: DEM aspect file
        config: Tile configuration (uses defaults if None)
        output_dir: Output directory
        prefix: Tile filename prefix
        
    Returns:
        List of paths to created tile NPZ files
        
    Example:
        >>> tiles = create_tiles_from_files(
        ...     s1_vv='data/s1/vv.tif',
        ...     s1_vh='data/s1/vh.tif',
        ...     s2_b4='data/s2/B04.tif',
        ...     s2_b8='data/s2/B08.tif',
        ...     dem_slope='data/dem/slope.tif',
        ...     tile_size=256,
        ...     output_dir='tiles/'
        ... )
    """
    try:
        from axs_lib.geo import read_cog
    except ImportError:
        raise ImportError("axs_lib.geo module required for reading raster files")
    
    if config is None:
        config = TileConfig()
    
    arrays = {}
    source_files = {}
    geotransform = None
    crs = None
    
    # Map of parameter names to standardized keys
    file_map = {
        's1_vv': s1_vv,
        's1_vh': s1_vh,
        's2_b2': s2_b2,
        's2_b3': s2_b3,
        's2_b4': s2_b4,
        's2_b8': s2_b8,
        'inc_angle': inc_angle,
        'biome_code': biome_code,
        'dem_slope': dem_slope,
        'dem_aspect': dem_aspect
    }
    
    # Load raster files
    for key, file_path in file_map.items():
        if file_path is not None:
            data, profile = read_cog(file_path)
            arrays[key] = data
            source_files[key] = str(file_path)
            
            # Get geotransform and CRS from first file
            if geotransform is None:
                geotransform = [
                    profile['transform'][2],  # x_origin
                    profile['transform'][0],  # pixel_width
                    profile['transform'][1],  # rotation (usually 0)
                    profile['transform'][5],  # y_origin
                    profile['transform'][3],  # rotation (usually 0)
                    profile['transform'][4]   # pixel_height (negative)
                ]
                crs = profile.get('crs', 'EPSG:4326')
    
    # Handle month (can be int or file)
    if month is not None:
        if isinstance(month, int):
            # Create constant array
            shape = next(iter(arrays.values())).shape
            arrays['month'] = np.full(shape, month, dtype=np.float32)
            source_files['month'] = f'constant:{month}'
        else:
            # Load from file
            data, _ = read_cog(month)
            arrays['month'] = data
            source_files['month'] = str(month)
    
    if not arrays:
        raise ValueError("No input files provided")
    
    if geotransform is None:
        raise ValueError("Could not determine geotransform from input files")
    
    return create_tiles_from_arrays(
        arrays=arrays,
        base_geotransform=geotransform,
        crs=crs,
        config=config,
        output_dir=output_dir,
        prefix=prefix,
        source_files=source_files
    )


# ============================================================================
# Tile Loading and Batching
# ============================================================================

def load_tile(tile_path: Union[str, Path]) -> Tuple[Dict[str, np.ndarray], TileMetadata]:
    """
    Load a tile NPZ and its JSON metadata.
    
    Args:
        tile_path: Path to NPZ tile file
        
    Returns:
        Tuple of (tile_data_dict, metadata)
        
    Example:
        >>> data, metadata = load_tile('tiles/tile_r00000_c00000.npz')
        >>> print(data.keys())
        dict_keys(['s1_vv', 's2_b4', 'dem_slope'])
    """
    tile_path = Path(tile_path)
    
    # Load NPZ
    npz_data = np.load(tile_path)
    tile_data = {key: npz_data[key] for key in npz_data.files}
    
    # Load JSON sidecar
    json_path = tile_path.with_suffix('.json')
    if json_path.exists():
        with open(json_path, 'r') as f:
            metadata_dict = json.load(f)
        metadata = TileMetadata(**metadata_dict)
    else:
        # Create minimal metadata if JSON doesn't exist
        metadata = TileMetadata(
            tile_id=tile_path.stem,
            row=0,
            col=0,
            width=next(iter(tile_data.values())).shape[1],
            height=next(iter(tile_data.values())).shape[0],
            geotransform=[0, 1, 0, 0, 0, -1],
            crs='unknown',
            timestamp=datetime.utcnow().isoformat() + 'Z',
            source_files={},
            tile_bounds={'minx': 0, 'miny': 0, 'maxx': 0, 'maxy': 0}
        )
    
    return tile_data, metadata


def load_tiles_batch(
    tile_paths: List[Union[str, Path]],
    keys: Optional[List[str]] = None,
    stack: bool = True
) -> Union[Dict[str, np.ndarray], Dict[str, List[np.ndarray]]]:
    """
    Load multiple tiles as a batch.
    
    Args:
        tile_paths: List of tile NPZ paths
        keys: Specific keys to load (None = all keys)
        stack: If True, stack into single array; if False, return list
        
    Returns:
        Dictionary mapping keys to stacked arrays or lists of arrays
        
    Example:
        >>> paths = list(Path('tiles/').glob('*.npz'))[:10]
        >>> batch = load_tiles_batch(paths, keys=['s2_b4', 's2_b8'], stack=True)
        >>> print(batch['s2_b4'].shape)
        (10, 256, 256)
    """
    if not tile_paths:
        return {}
    
    # Load first tile to get keys
    first_data, _ = load_tile(tile_paths[0])
    
    if keys is None:
        keys = list(first_data.keys())
    
    # Initialize storage
    batch_data = {key: [] for key in keys}
    
    # Load all tiles
    for tile_path in tile_paths:
        tile_data, _ = load_tile(tile_path)
        for key in keys:
            if key in tile_data:
                batch_data[key].append(tile_data[key])
    
    # Stack if requested
    if stack:
        batch_data = {
            key: np.stack(arrays, axis=0) 
            for key, arrays in batch_data.items()
        }
    
    return batch_data


# ============================================================================
# Utility Functions
# ============================================================================

def list_tiles(tile_dir: Union[str, Path], pattern: str = '*.npz') -> List[Path]:
    """
    List all tiles in a directory.
    
    Args:
        tile_dir: Directory containing tiles
        pattern: Glob pattern for tile files
        
    Returns:
        List of tile paths
    """
    tile_dir = Path(tile_dir)
    return sorted(tile_dir.glob(pattern))


def get_tile_statistics(
    tile_path: Union[str, Path],
    keys: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for tile arrays.
    
    Args:
        tile_path: Path to tile NPZ
        keys: Keys to compute stats for (None = all)
        
    Returns:
        Dictionary mapping keys to stat dictionaries
        
    Example:
        >>> stats = get_tile_statistics('tiles/tile_r00000_c00000.npz')
        >>> print(stats['s2_b4'])
        {'mean': 0.42, 'std': 0.15, 'min': 0.0, 'max': 1.0, 'valid_pixels': 65536}
    """
    tile_data, _ = load_tile(tile_path)
    
    if keys is None:
        keys = list(tile_data.keys())
    
    stats = {}
    for key in keys:
        if key in tile_data:
            arr = tile_data[key]
            stats[key] = {
                'mean': float(np.nanmean(arr)),
                'std': float(np.nanstd(arr)),
                'min': float(np.nanmin(arr)),
                'max': float(np.nanmax(arr)),
                'valid_pixels': int(np.sum(~np.isnan(arr)))
            }
    
    return stats


def visualize_tile(
    tile_path: Union[str, Path],
    rgb_keys: Tuple[str, str, str] = ('s2_b4', 's2_b3', 's2_b2'),
    output_path: Optional[Union[str, Path]] = None
):
    """
    Create RGB visualization of a tile.
    
    Args:
        tile_path: Path to tile NPZ
        rgb_keys: Keys for R, G, B channels
        output_path: Optional path to save PNG (shows if None)
        
    Example:
        >>> visualize_tile(
        ...     'tiles/tile_r00000_c00000.npz',
        ...     rgb_keys=('s2_b4', 's2_b3', 's2_b2'),
        ...     output_path='preview.png'
        ... )
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")
    
    tile_data, metadata = load_tile(tile_path)
    
    # Extract RGB channels
    rgb = []
    for key in rgb_keys:
        if key in tile_data:
            channel = tile_data[key]
            # Normalize to [0, 1]
            channel = normalize_array(channel, method='percentile')
            rgb.append(channel)
        else:
            raise ValueError(f"Key '{key}' not found in tile")
    
    rgb = np.stack(rgb, axis=-1)
    
    # Create figure
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title(f"Tile: {metadata.tile_id}\n{metadata.crs}")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Main / Testing
# ============================================================================

def run_demo():
    """
    Run demo with synthetic data.
    """
    print("=" * 79)
    print("DATA TILING DEMO")
    print("=" * 79)
    print()
    
    # Create synthetic aligned arrays
    print("Creating synthetic data (1000x1000 pixels)...")
    height, width = 1000, 1000
    
    arrays = {
        's1_vv': np.random.randn(height, width).astype(np.float32) * 10 - 15,
        's1_vh': np.random.randn(height, width).astype(np.float32) * 10 - 20,
        's2_b2': np.random.randint(0, 10000, (height, width)).astype(np.float32),
        's2_b3': np.random.randint(0, 10000, (height, width)).astype(np.float32),
        's2_b4': np.random.randint(0, 10000, (height, width)).astype(np.float32),
        's2_b8': np.random.randint(0, 10000, (height, width)).astype(np.float32),
        'month': np.full((height, width), 6, dtype=np.float32),
        'inc_angle': np.random.uniform(20, 45, (height, width)).astype(np.float32),
        'biome_code': np.random.randint(1, 15, (height, width)).astype(np.float32),
        'dem_slope': np.random.uniform(0, 45, (height, width)).astype(np.float32),
        'dem_aspect': np.random.uniform(0, 360, (height, width)).astype(np.float32)
    }
    print(f"Created {len(arrays)} arrays")
    print()
    
    # Define geotransform (10m pixels, UTM Zone 10N)
    geotransform = [500000, 10, 0, 4500000, 0, -10]
    crs = 'EPSG:32610'
    
    # Create tiles
    print("Creating tiles (256x256, stride=256)...")
    config = TileConfig(
        tile_size=256,
        stride=256,
        min_valid_pixels=1000,
        dtype='float32'
    )
    
    output_dir = Path('output/demo_tiles')
    
    tile_paths = create_tiles_from_arrays(
        arrays=arrays,
        base_geotransform=geotransform,
        crs=crs,
        config=config,
        output_dir=output_dir,
        prefix='demo'
    )
    print()
    
    # Load and inspect first tile
    if tile_paths:
        print("Inspecting first tile...")
        first_tile = tile_paths[0]
        print(f"  Path: {first_tile}")
        
        tile_data, metadata = load_tile(first_tile)
        print(f"  Keys: {list(tile_data.keys())}")
        print(f"  Tile ID: {metadata.tile_id}")
        print(f"  Shape: {next(iter(tile_data.values())).shape}")
        print(f"  Geotransform: {metadata.geotransform}")
        print(f"  Bounds: {metadata.tile_bounds}")
        print()
        
        # Get statistics
        print("Computing statistics...")
        stats = get_tile_statistics(first_tile)
        for key in ['s2_b4', 'dem_slope']:
            if key in stats:
                s = stats[key]
                print(f"  {key}: mean={s['mean']:.2f}, std={s['std']:.2f}, "
                      f"range=[{s['min']:.2f}, {s['max']:.2f}]")
        print()
        
        # Load batch
        print("Loading batch of 4 tiles...")
        batch = load_tiles_batch(tile_paths[:4], keys=['s2_b4', 's2_b8'], stack=True)
        print(f"  Batch shapes:")
        for key, arr in batch.items():
            print(f"    {key}: {arr.shape}")
        print()
    
    print("=" * 79)
    print(f"Demo complete! Created {len(tile_paths)} tiles in {output_dir}")
    print("=" * 79)


if __name__ == "__main__":
    run_demo()
