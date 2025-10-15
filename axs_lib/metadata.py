"""
axs_lib/metadata.py - Metadata Handling for TerraMind Pipeline

Manages auxiliary metadata (month, incidence angle, etc.) throughout the pipeline:
- Stage 1: Metadata collection and storage in tiles
- Stage 2: Metadata preservation and tracking
- Stage 3: Metadata conditioning for final predictions

TerraMind Stage 1 Note:
    TerraMind 1.0's pretrained architecture does not support custom metadata 
    tokens. However, metadata is crucial for:
    - Understanding seasonal patterns (month)
    - Accounting for SAR viewing geometry (incidence angle)
    - Enabling conditional generation in Stage 3
    
    Therefore, we collect and preserve metadata throughout the pipeline for
    use in the Stage 3 conditional model.

Metadata Types:
    - Temporal: month (1-12), day_of_year (1-365)
    - Geometric: incidence_angle (degrees), azimuth_angle (degrees)
    - Environmental: biome_code, elevation, slope, aspect
    - Acquisition: satellite_id, orbit_direction, processing_level

Usage:
    >>> from axs_lib.metadata import MetadataManager
    >>> 
    >>> # Extract metadata from tiles
    >>> manager = MetadataManager()
    >>> metadata = manager.extract_from_tile('tiles/tile001.npz')
    >>> print(metadata)
    {'month': 7, 'incidence_angle': 38.5, 'biome': 'forest'}
    >>> 
    >>> # Save for Stage 3
    >>> manager.save_for_stage3(metadata, 'stage3_meta/tile001_meta.json')
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict, field
import numpy as np


# ============================================================================
# Metadata Data Classes
# ============================================================================

@dataclass
class TileMetadataExtract:
    """Extracted metadata from a single tile for Stage 3 conditioning."""
    
    # Tile identification
    tile_id: str
    tile_path: str
    
    # Temporal metadata
    month: Optional[int] = None  # 1-12
    day_of_year: Optional[int] = None  # 1-365
    year: Optional[int] = None
    
    # Geometric metadata (SAR)
    incidence_angle: Optional[float] = None  # degrees, typically 20-45
    azimuth_angle: Optional[float] = None  # degrees, 0-360
    orbit_direction: Optional[str] = None  # 'ascending' or 'descending'
    
    # Environmental metadata
    biome_code: Optional[int] = None
    biome_name: Optional[str] = None
    mean_elevation: Optional[float] = None  # meters
    mean_slope: Optional[float] = None  # degrees
    mean_aspect: Optional[float] = None  # degrees
    
    # Quality metadata
    cloud_coverage: Optional[float] = None  # 0-100%
    data_completeness: Optional[float] = None  # 0-100%
    
    # Source info
    s1_product_id: Optional[str] = None
    s2_product_id: Optional[str] = None
    
    # Additional custom metadata
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TileMetadataExtract':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TileMetadataExtract':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


# ============================================================================
# Metadata Manager
# ============================================================================

class MetadataManager:
    """Manages metadata extraction, storage, and retrieval for the pipeline."""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize metadata manager.
        
        Args:
            verbose: Print status messages
        """
        self.verbose = verbose
        self._metadata_cache = {}
    
    def extract_from_tile(
        self, 
        tile_path: Union[str, Path],
        include_arrays: bool = False
    ) -> TileMetadataExtract:
        """
        Extract metadata from NPZ tile file.
        
        Args:
            tile_path: Path to NPZ tile file
            include_arrays: If True, compute statistics from arrays (slower)
            
        Returns:
            Extracted metadata
            
        Example:
            >>> manager = MetadataManager()
            >>> meta = manager.extract_from_tile('tiles/tile001.npz')
            >>> print(meta.month, meta.incidence_angle)
            7 38.5
        """
        tile_path = Path(tile_path)
        
        if not tile_path.exists():
            raise FileNotFoundError(f"Tile not found: {tile_path}")
        
        # Load NPZ
        npz_data = np.load(tile_path)
        
        # Initialize metadata
        metadata = TileMetadataExtract(
            tile_id=tile_path.stem,
            tile_path=str(tile_path)
        )
        
        # Extract scalar metadata
        if 'month' in npz_data:
            month_arr = npz_data['month']
            # Month might be constant or varying
            metadata.month = int(np.nanmean(month_arr))
        
        if 'inc_angle' in npz_data:
            inc_arr = npz_data['inc_angle']
            metadata.incidence_angle = float(np.nanmean(inc_arr))
        
        if 'biome_code' in npz_data:
            biome_arr = npz_data['biome_code']
            metadata.biome_code = int(np.nanmean(biome_arr))
        
        # Extract from arrays if requested
        if include_arrays:
            if 'dem_elevation' in npz_data:
                metadata.mean_elevation = float(np.nanmean(npz_data['dem_elevation']))
            
            if 'dem_slope' in npz_data:
                metadata.mean_slope = float(np.nanmean(npz_data['dem_slope']))
            
            if 'dem_aspect' in npz_data:
                metadata.mean_aspect = float(np.nanmean(npz_data['dem_aspect']))
        
        # Try to load JSON sidecar for additional info
        json_path = tile_path.with_suffix('.json')
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    tile_info = json.load(f)
                    # Extract any additional metadata
                    if 'source_files' in tile_info:
                        metadata.custom['source_files'] = tile_info['source_files']
                    if 'geotransform' in tile_info:
                        metadata.custom['geotransform'] = tile_info['geotransform']
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"Could not load JSON sidecar: {e}")
        
        return metadata
    
    def extract_batch(
        self,
        tile_paths: List[Union[str, Path]],
        include_arrays: bool = False
    ) -> List[TileMetadataExtract]:
        """
        Extract metadata from multiple tiles.
        
        Args:
            tile_paths: List of tile paths
            include_arrays: Compute statistics from arrays
            
        Returns:
            List of metadata objects
        """
        metadata_list = []
        for tile_path in tile_paths:
            try:
                meta = self.extract_from_tile(tile_path, include_arrays=include_arrays)
                metadata_list.append(meta)
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"Failed to extract metadata from {tile_path}: {e}")
        
        return metadata_list
    
    def save_for_stage3(
        self,
        metadata: Union[TileMetadataExtract, List[TileMetadataExtract]],
        output_path: Union[str, Path]
    ):
        """
        Save metadata for Stage 3 conditioning.
        
        Args:
            metadata: Single metadata object or list of metadata
            output_path: Path to save JSON file
            
        Example:
            >>> manager = MetadataManager()
            >>> meta = manager.extract_from_tile('tiles/tile001.npz')
            >>> manager.save_for_stage3(meta, 'stage3_meta/tile001_meta.json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(metadata, list):
            data = [m.to_dict() for m in metadata]
        else:
            data = metadata.to_dict()
        
        with open(output_path, 'w') as f:
            json.dumps(data, f, indent=2)
        
        if self.verbose:
            print(f"Saved metadata to {output_path}")
    
    def load_from_stage3(self, metadata_path: Union[str, Path]) -> Union[TileMetadataExtract, List[TileMetadataExtract]]:
        """
        Load metadata saved for Stage 3.
        
        Args:
            metadata_path: Path to JSON metadata file
            
        Returns:
            Metadata object or list of objects
        """
        metadata_path = Path(metadata_path)
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return [TileMetadataExtract.from_dict(d) for d in data]
        else:
            return TileMetadataExtract.from_dict(data)
    
    def create_embedding_vector(
        self,
        metadata: TileMetadataExtract,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Create embedding vector from metadata for conditioning.
        
        This prepares metadata for use as conditioning input to Stage 3 models.
        
        Args:
            metadata: Metadata object
            normalize: Normalize values to reasonable ranges
            
        Returns:
            Numpy array of shape (n_features,)
            
        Features:
            - month_sin, month_cos: Cyclical encoding of month
            - incidence_angle_norm: Normalized incidence angle (0-1)
            - biome_onehot: One-hot encoded biome (if available)
            - elevation_norm: Normalized elevation
            - slope_norm: Normalized slope
        """
        features = []
        
        # Temporal features (cyclical encoding)
        if metadata.month is not None:
            month_rad = (metadata.month - 1) * (2 * np.pi / 12)
            features.extend([np.sin(month_rad), np.cos(month_rad)])
        else:
            features.extend([0.0, 0.0])
        
        # Geometric features
        if metadata.incidence_angle is not None:
            # Normalize to [0, 1] assuming range [20, 45] degrees
            inc_norm = (metadata.incidence_angle - 20) / 25 if normalize else metadata.incidence_angle
            features.append(inc_norm)
        else:
            features.append(0.0)
        
        # Elevation (normalize to typical range)
        if metadata.mean_elevation is not None:
            elev_norm = metadata.mean_elevation / 3000 if normalize else metadata.mean_elevation
            features.append(elev_norm)
        else:
            features.append(0.0)
        
        # Slope
        if metadata.mean_slope is not None:
            slope_norm = metadata.mean_slope / 45 if normalize else metadata.mean_slope
            features.append(slope_norm)
        else:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def get_metadata_summary(
        self,
        tile_dir: Union[str, Path],
        max_tiles: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics of metadata across tiles.
        
        Args:
            tile_dir: Directory containing tiles
            max_tiles: Maximum number of tiles to analyze
            
        Returns:
            Dictionary with summary statistics
        """
        tile_dir = Path(tile_dir)
        tile_paths = sorted(list(tile_dir.glob("*.npz")))
        
        if max_tiles is not None:
            tile_paths = tile_paths[:max_tiles]
        
        if not tile_paths:
            return {"error": "No tiles found"}
        
        # Extract metadata
        metadata_list = self.extract_batch(tile_paths, include_arrays=False)
        
        # Compute statistics
        months = [m.month for m in metadata_list if m.month is not None]
        incidence_angles = [m.incidence_angle for m in metadata_list if m.incidence_angle is not None]
        
        summary = {
            "n_tiles": len(metadata_list),
            "month": {
                "available": len(months),
                "min": int(np.min(months)) if months else None,
                "max": int(np.max(months)) if months else None,
                "mean": float(np.mean(months)) if months else None,
                "unique": sorted(list(set(months))) if months else []
            },
            "incidence_angle": {
                "available": len(incidence_angles),
                "min": float(np.min(incidence_angles)) if incidence_angles else None,
                "max": float(np.max(incidence_angles)) if incidence_angles else None,
                "mean": float(np.mean(incidence_angles)) if incidence_angles else None,
                "std": float(np.std(incidence_angles)) if incidence_angles else None
            }
        }
        
        return summary


# ============================================================================
# Stage 3 Preparation Utilities
# ============================================================================

def prepare_metadata_for_stage3(
    tile_dir: Union[str, Path],
    output_dir: Union[str, Path],
    include_arrays: bool = False
):
    """
    Prepare all metadata for Stage 3 by extracting and saving.
    
    Args:
        tile_dir: Directory containing tiles
        output_dir: Output directory for metadata JSON files
        include_arrays: Compute statistics from arrays
        
    Example:
        >>> prepare_metadata_for_stage3('tiles/', 'stage3_metadata/')
        Processed 100 tiles
        Metadata saved to stage3_metadata/
    """
    tile_dir = Path(tile_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manager = MetadataManager(verbose=True)
    
    # Find all tiles
    tile_paths = sorted(list(tile_dir.glob("*.npz")))
    tile_paths = [p for p in tile_paths if '_opt_v1' not in p.stem]  # Exclude generated tiles
    
    print(f"Found {len(tile_paths)} tiles")
    print(f"Extracting metadata...")
    
    # Extract metadata
    metadata_list = manager.extract_batch(tile_paths, include_arrays=include_arrays)
    
    # Save individual files
    for meta in metadata_list:
        output_path = output_dir / f"{meta.tile_id}_metadata.json"
        manager.save_for_stage3(meta, output_path)
    
    # Save summary
    summary_path = output_dir / "metadata_summary.json"
    summary = manager.get_metadata_summary(tile_dir)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Processed {len(metadata_list)} tiles")
    print(f"✓ Metadata saved to {output_dir}")
    print(f"✓ Summary saved to {summary_path}")


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract and manage metadata for Stage 3 conditioning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract metadata summary
  python axs_lib/metadata.py --tile-dir tiles/ --summary
  
  # Prepare metadata for Stage 3
  python axs_lib/metadata.py --tile-dir tiles/ --output-dir stage3_meta/ --prepare
  
  # Extract single tile
  python axs_lib/metadata.py --tile tiles/tile001.npz
        """
    )
    
    parser.add_argument('--tile-dir', type=str,
                        help='Directory containing tiles')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for metadata')
    parser.add_argument('--tile', type=str,
                        help='Single tile to extract metadata from')
    parser.add_argument('--summary', action='store_true',
                        help='Print metadata summary')
    parser.add_argument('--prepare', action='store_true',
                        help='Prepare all metadata for Stage 3')
    parser.add_argument('--include-arrays', action='store_true',
                        help='Compute statistics from arrays (slower)')
    
    args = parser.parse_args()
    
    manager = MetadataManager(verbose=True)
    
    if args.tile:
        # Extract single tile
        meta = manager.extract_from_tile(args.tile, include_arrays=args.include_arrays)
        print("\nExtracted Metadata:")
        print(json.dumps(meta.to_dict(), indent=2))
    
    elif args.summary and args.tile_dir:
        # Print summary
        summary = manager.get_metadata_summary(args.tile_dir)
        print("\nMetadata Summary:")
        print(json.dumps(summary, indent=2))
    
    elif args.prepare and args.tile_dir and args.output_dir:
        # Prepare for Stage 3
        prepare_metadata_for_stage3(
            args.tile_dir,
            args.output_dir,
            include_arrays=args.include_arrays
        )
    
    else:
        parser.print_help()
