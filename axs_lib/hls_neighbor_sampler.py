"""
HLS Neighbor Sampler with Validation

This module provides a data sampler for Stage 2 training that validates HLS neighbor
availability before including tiles in the training set. It ensures that each tile
has sufficient temporal and spatial neighbors for learning seasonality priors.

Key Features:
1. Validates HLS neighbor availability (spatial, temporal, biome)
2. Logs detailed skip reasons (missing neighbors, insufficient coverage, etc.)
3. Tracks statistics on skip patterns
4. Supports multiple skip criteria (configurable thresholds)
5. Caches validation results for efficiency

Skip Reasons:
- NO_HLS_NEIGHBORS: No HLS tiles found within spatial/temporal window
- INSUFFICIENT_NEIGHBORS: Too few neighbors (< min_neighbors threshold)
- TEMPORAL_GAP_TOO_LARGE: Nearest neighbor too far in time (> max_temporal_gap days)
- BIOME_MISMATCH: No neighbors in same biome
- CLOUD_COVER_HIGH: All neighbors have high cloud cover (> max_cloud_cover)
- NDVI_OUTLIER: Tile NDVI significantly different from neighbors (anomalous)
- MISSING_METADATA: Required metadata missing (month, location, biome)

Usage:
    from axs_lib.hls_neighbor_sampler import HLSNeighborSampler
    
    sampler = HLSNeighborSampler(
        tile_catalog_path='data/tiles/benv2_catalog.csv',
        hls_index_path='data/hls/hls_index.csv',
        min_neighbors=3,
        max_temporal_gap=15,  # days
        log_file='logs/sampler_skipped_tiles.log'
    )
    
    valid_tiles = sampler.filter_tiles()
    sampler.print_statistics()

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import json

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================================
# Skip Reason Enumeration
# ============================================================================

class SkipReason(Enum):
    """Enumeration of reasons why a tile is skipped."""
    
    NO_HLS_NEIGHBORS = "no_hls_neighbors"
    INSUFFICIENT_NEIGHBORS = "insufficient_neighbors"
    TEMPORAL_GAP_TOO_LARGE = "temporal_gap_too_large"
    BIOME_MISMATCH = "biome_mismatch"
    CLOUD_COVER_HIGH = "cloud_cover_high"
    NDVI_OUTLIER = "ndvi_outlier"
    MISSING_METADATA = "missing_metadata"
    SPATIAL_COVERAGE_LOW = "spatial_coverage_low"
    QUALITY_FLAGS_BAD = "quality_flags_bad"
    
    def __str__(self):
        return self.value


@dataclass
class SkipRecord:
    """Record of a skipped tile with reason and details."""
    
    tile_id: str
    reason: SkipReason
    details: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'tile_id': self.tile_id,
            'reason': str(self.reason),
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }
    
    def to_log_string(self) -> str:
        """Convert to formatted log string."""
        meta_str = json.dumps(self.metadata) if self.metadata else '{}'
        return f"{self.timestamp.isoformat()} | {self.tile_id} | {self.reason.value} | {self.details} | {meta_str}"


# ============================================================================
# HLS Neighbor Sampler
# ============================================================================

class HLSNeighborSampler:
    """
    Sampler that validates HLS neighbor availability for Stage 2 training.
    
    Validates each tile against multiple criteria:
    1. HLS neighbors exist within spatial radius
    2. HLS neighbors exist within temporal window
    3. Sufficient number of neighbors (min_neighbors)
    4. Neighbors have acceptable cloud cover
    5. Neighbors match biome classification
    6. Tile NDVI is not anomalous compared to neighbors
    """
    
    def __init__(
        self,
        tile_catalog_path: Optional[Path] = None,
        hls_index_path: Optional[Path] = None,
        min_neighbors: int = 3,
        max_temporal_gap: int = 15,  # days
        max_spatial_distance: float = 100.0,  # km
        max_cloud_cover: float = 0.3,  # 30%
        require_same_biome: bool = True,
        ndvi_outlier_threshold: float = 3.0,  # standard deviations
        log_file: Optional[Path] = None,
        cache_file: Optional[Path] = None,
        verbose: bool = True
    ):
        """
        Initialize HLS neighbor sampler.
        
        Args:
            tile_catalog_path: Path to tile catalog CSV with metadata
            hls_index_path: Path to HLS index CSV with available tiles
            min_neighbors: Minimum number of valid HLS neighbors required
            max_temporal_gap: Maximum days between tile and nearest neighbor
            max_spatial_distance: Maximum distance to HLS neighbor (km)
            max_cloud_cover: Maximum acceptable cloud cover fraction
            require_same_biome: Whether to require neighbors in same biome
            ndvi_outlier_threshold: NDVI z-score threshold for outlier detection
            log_file: Path to save skip log
            cache_file: Path to cache validation results
            verbose: Whether to print progress
        """
        self.tile_catalog_path = tile_catalog_path
        self.hls_index_path = hls_index_path
        self.min_neighbors = min_neighbors
        self.max_temporal_gap = max_temporal_gap
        self.max_spatial_distance = max_spatial_distance
        self.max_cloud_cover = max_cloud_cover
        self.require_same_biome = require_same_biome
        self.ndvi_outlier_threshold = ndvi_outlier_threshold
        self.log_file = log_file
        self.cache_file = cache_file
        self.verbose = verbose
        
        # Statistics
        self.skip_records: List[SkipRecord] = []
        self.valid_tiles: List[str] = []
        self.skipped_tiles: Set[str] = set()
        
        # Cached data
        self.tile_catalog: Optional[pd.DataFrame] = None
        self.hls_index: Optional[pd.DataFrame] = None
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger('HLSNeighborSampler')
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        # Console handler
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def load_data(self):
        """Load tile catalog and HLS index."""
        self.logger.info("Loading tile catalog and HLS index...")
        
        if self.tile_catalog_path and self.tile_catalog_path.exists():
            self.tile_catalog = pd.read_csv(self.tile_catalog_path)
            self.logger.info(f"  Loaded {len(self.tile_catalog)} tiles from catalog")
        else:
            self.logger.warning("  Tile catalog not found, using mock data")
            self.tile_catalog = self._create_mock_tile_catalog()
        
        if self.hls_index_path and self.hls_index_path.exists():
            self.hls_index = pd.read_csv(self.hls_index_path)
            self.logger.info(f"  Loaded {len(self.hls_index)} HLS tiles from index")
        else:
            self.logger.warning("  HLS index not found, using mock data")
            self.hls_index = self._create_mock_hls_index()
    
    def _create_mock_tile_catalog(self) -> pd.DataFrame:
        """Create mock tile catalog for testing."""
        return pd.DataFrame({
            'tile_id': [f'tile_{i:04d}' for i in range(100)],
            'date': pd.date_range('2020-01-01', periods=100, freq='3D'),
            'lat': np.random.uniform(40, 50, 100),
            'lon': np.random.uniform(-10, 10, 100),
            'biome': np.random.choice([5, 6, 7, 8], 100),
            'ndvi_mean': np.random.uniform(0.3, 0.7, 100),
            'cloud_cover': np.random.uniform(0, 0.5, 100)
        })
    
    def _create_mock_hls_index(self) -> pd.DataFrame:
        """Create mock HLS index for testing."""
        return pd.DataFrame({
            'hls_id': [f'HLS.S30.T30UXV.{2020000 + i}' for i in range(500)],
            'date': pd.date_range('2019-12-01', periods=500, freq='2D'),
            'lat': np.random.uniform(40, 50, 500),
            'lon': np.random.uniform(-10, 10, 500),
            'biome': np.random.choice([5, 6, 7, 8], 500),
            'ndvi_mean': np.random.uniform(0.2, 0.8, 500),
            'cloud_cover': np.random.uniform(0, 0.6, 500)
        })
    
    def validate_tile(self, tile_row: pd.Series) -> Tuple[bool, Optional[SkipRecord]]:
        """
        Validate a single tile for HLS neighbor availability.
        
        Args:
            tile_row: Row from tile catalog DataFrame
            
        Returns:
            (is_valid, skip_record)
            - is_valid: True if tile passes all checks
            - skip_record: SkipRecord if tile is skipped, None if valid
        """
        tile_id = tile_row['tile_id']
        
        # Check 1: Required metadata present
        required_fields = ['date', 'lat', 'lon', 'biome', 'ndvi_mean']
        missing_fields = [f for f in required_fields if pd.isna(tile_row.get(f))]
        
        if missing_fields:
            return False, SkipRecord(
                tile_id=tile_id,
                reason=SkipReason.MISSING_METADATA,
                details=f"Missing fields: {', '.join(missing_fields)}",
                metadata={'missing_fields': missing_fields}
            )
        
        # Extract tile info
        tile_date = pd.to_datetime(tile_row['date'])
        tile_lat = float(tile_row['lat'])
        tile_lon = float(tile_row['lon'])
        tile_biome = int(tile_row['biome'])
        tile_ndvi = float(tile_row['ndvi_mean'])
        
        # Check 2: Find HLS neighbors
        neighbors = self._find_hls_neighbors(
            tile_date, tile_lat, tile_lon, tile_biome
        )
        
        if len(neighbors) == 0:
            return False, SkipRecord(
                tile_id=tile_id,
                reason=SkipReason.NO_HLS_NEIGHBORS,
                details=f"No HLS tiles within {self.max_spatial_distance}km and {self.max_temporal_gap} days",
                metadata={
                    'spatial_radius': self.max_spatial_distance,
                    'temporal_window': self.max_temporal_gap
                }
            )
        
        # Check 3: Sufficient neighbors
        if len(neighbors) < self.min_neighbors:
            return False, SkipRecord(
                tile_id=tile_id,
                reason=SkipReason.INSUFFICIENT_NEIGHBORS,
                details=f"Found {len(neighbors)} neighbors, need {self.min_neighbors}",
                metadata={
                    'found_neighbors': len(neighbors),
                    'required_neighbors': self.min_neighbors
                }
            )
        
        # Check 4: Temporal gap
        temporal_gaps = [(n_date - tile_date).days for n_date, _, _ in neighbors]
        min_temporal_gap = min(abs(gap) for gap in temporal_gaps)
        
        if min_temporal_gap > self.max_temporal_gap:
            return False, SkipRecord(
                tile_id=tile_id,
                reason=SkipReason.TEMPORAL_GAP_TOO_LARGE,
                details=f"Nearest neighbor {min_temporal_gap} days away (max: {self.max_temporal_gap})",
                metadata={
                    'min_temporal_gap': min_temporal_gap,
                    'max_allowed': self.max_temporal_gap
                }
            )
        
        # Check 5: Cloud cover
        neighbor_clouds = [cloud for _, _, cloud in neighbors]
        if all(cloud > self.max_cloud_cover for cloud in neighbor_clouds):
            return False, SkipRecord(
                tile_id=tile_id,
                reason=SkipReason.CLOUD_COVER_HIGH,
                details=f"All {len(neighbors)} neighbors have cloud cover > {self.max_cloud_cover:.1%}",
                metadata={
                    'min_neighbor_cloud': min(neighbor_clouds),
                    'max_allowed': self.max_cloud_cover
                }
            )
        
        # Check 6: NDVI outlier detection
        neighbor_ndvis = self.hls_index[
            self.hls_index['hls_id'].isin([hls_id for _, hls_id, _ in neighbors])
        ]['ndvi_mean'].values
        
        if len(neighbor_ndvis) > 0:
            ndvi_mean = neighbor_ndvis.mean()
            ndvi_std = neighbor_ndvis.std()
            
            if ndvi_std > 0:
                ndvi_zscore = abs(tile_ndvi - ndvi_mean) / ndvi_std
                
                if ndvi_zscore > self.ndvi_outlier_threshold:
                    return False, SkipRecord(
                        tile_id=tile_id,
                        reason=SkipReason.NDVI_OUTLIER,
                        details=f"NDVI z-score {ndvi_zscore:.2f} > {self.ndvi_outlier_threshold}",
                        metadata={
                            'tile_ndvi': tile_ndvi,
                            'neighbor_ndvi_mean': ndvi_mean,
                            'neighbor_ndvi_std': ndvi_std,
                            'z_score': ndvi_zscore
                        }
                    )
        
        # All checks passed
        return True, None
    
    def _find_hls_neighbors(
        self,
        tile_date: datetime,
        tile_lat: float,
        tile_lon: float,
        tile_biome: int
    ) -> List[Tuple[datetime, str, float]]:
        """
        Find HLS neighbors for a given tile.
        
        Args:
            tile_date: Date of tile
            tile_lat: Latitude of tile
            tile_lon: Longitude of tile
            tile_biome: Biome classification
            
        Returns:
            List of (neighbor_date, neighbor_id, cloud_cover) tuples
        """
        if self.hls_index is None:
            return []
        
        # Temporal filtering
        date_min = tile_date - timedelta(days=self.max_temporal_gap)
        date_max = tile_date + timedelta(days=self.max_temporal_gap)
        
        self.hls_index['date_dt'] = pd.to_datetime(self.hls_index['date'])
        temporal_mask = (
            (self.hls_index['date_dt'] >= date_min) &
            (self.hls_index['date_dt'] <= date_max)
        )
        
        # Spatial filtering (approximate, using lat/lon box)
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 * cos(lat) km
        lat_delta = self.max_spatial_distance / 111.0
        lon_delta = self.max_spatial_distance / (111.0 * np.cos(np.radians(tile_lat)))
        
        spatial_mask = (
            (self.hls_index['lat'] >= tile_lat - lat_delta) &
            (self.hls_index['lat'] <= tile_lat + lat_delta) &
            (self.hls_index['lon'] >= tile_lon - lon_delta) &
            (self.hls_index['lon'] <= tile_lon + lon_delta)
        )
        
        # Biome filtering
        if self.require_same_biome:
            biome_mask = self.hls_index['biome'] == tile_biome
            mask = temporal_mask & spatial_mask & biome_mask
        else:
            mask = temporal_mask & spatial_mask
        
        # Get neighbors
        neighbors_df = self.hls_index[mask]
        
        neighbors = [
            (row['date_dt'], row['hls_id'], row['cloud_cover'])
            for _, row in neighbors_df.iterrows()
        ]
        
        return neighbors
    
    def filter_tiles(self) -> List[str]:
        """
        Filter tiles based on HLS neighbor availability.
        
        Returns:
            List of valid tile IDs
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting HLS Neighbor Validation")
        self.logger.info("=" * 80)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  Min neighbors: {self.min_neighbors}")
        self.logger.info(f"  Max temporal gap: {self.max_temporal_gap} days")
        self.logger.info(f"  Max spatial distance: {self.max_spatial_distance} km")
        self.logger.info(f"  Max cloud cover: {self.max_cloud_cover:.1%}")
        self.logger.info(f"  Require same biome: {self.require_same_biome}")
        self.logger.info(f"  NDVI outlier threshold: {self.ndvi_outlier_threshold} σ")
        self.logger.info("")
        
        # Load data if not already loaded
        if self.tile_catalog is None or self.hls_index is None:
            self.load_data()
        
        # Validate each tile
        self.valid_tiles = []
        self.skip_records = []
        self.skipped_tiles = set()
        
        for _, tile_row in tqdm(
            self.tile_catalog.iterrows(),
            total=len(self.tile_catalog),
            desc="Validating tiles",
            disable=not self.verbose
        ):
            is_valid, skip_record = self.validate_tile(tile_row)
            
            if is_valid:
                self.valid_tiles.append(tile_row['tile_id'])
            else:
                self.skipped_tiles.add(tile_row['tile_id'])
                self.skip_records.append(skip_record)
                
                # Log skip
                if self.log_file:
                    self._log_skip_record(skip_record)
        
        # Summary
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("Validation Complete")
        self.logger.info("=" * 80)
        self.logger.info(f"Valid tiles: {len(self.valid_tiles)} ({100*len(self.valid_tiles)/len(self.tile_catalog):.1f}%)")
        self.logger.info(f"Skipped tiles: {len(self.skipped_tiles)} ({100*len(self.skipped_tiles)/len(self.tile_catalog):.1f}%)")
        self.logger.info("")
        
        return self.valid_tiles
    
    def _log_skip_record(self, skip_record: SkipRecord):
        """Write skip record to log file."""
        if not self.log_file:
            return
        
        with open(self.log_file, 'a') as f:
            f.write(skip_record.to_log_string() + '\n')
    
    def print_statistics(self):
        """Print detailed statistics about skip reasons."""
        print("\n" + "=" * 80)
        print("HLS Neighbor Validation Statistics")
        print("=" * 80)
        
        print(f"\nTotal tiles: {len(self.tile_catalog)}")
        print(f"Valid tiles: {len(self.valid_tiles)} ({100*len(self.valid_tiles)/len(self.tile_catalog):.1f}%)")
        print(f"Skipped tiles: {len(self.skipped_tiles)} ({100*len(self.skipped_tiles)/len(self.tile_catalog):.1f}%)")
        
        if len(self.skip_records) == 0:
            print("\n✓ All tiles passed validation!")
            return
        
        # Count by reason
        reason_counts = defaultdict(int)
        for record in self.skip_records:
            reason_counts[record.reason] += 1
        
        print("\nSkip Reasons:")
        print("-" * 80)
        
        for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * count / len(self.skip_records)
            print(f"  {reason.value:30s} : {count:5d} ({pct:5.1f}%)")
        
        # Top skipped tiles
        print("\nTop 5 Most Recent Skips:")
        print("-" * 80)
        
        for record in self.skip_records[-5:]:
            print(f"  {record.tile_id:20s} | {record.reason.value:30s} | {record.details}")
        
        # Save summary
        if self.log_file:
            summary_file = self.log_file.parent / f"{self.log_file.stem}_summary.json"
            self._save_summary(summary_file)
            print(f"\n✓ Summary saved to: {summary_file}")
    
    def _save_summary(self, output_path: Path):
        """Save summary statistics to JSON."""
        reason_counts = defaultdict(int)
        for record in self.skip_records:
            reason_counts[str(record.reason)] += 1
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'min_neighbors': self.min_neighbors,
                'max_temporal_gap': self.max_temporal_gap,
                'max_spatial_distance': self.max_spatial_distance,
                'max_cloud_cover': self.max_cloud_cover,
                'require_same_biome': self.require_same_biome,
                'ndvi_outlier_threshold': self.ndvi_outlier_threshold
            },
            'results': {
                'total_tiles': len(self.tile_catalog),
                'valid_tiles': len(self.valid_tiles),
                'skipped_tiles': len(self.skipped_tiles),
                'valid_percentage': 100 * len(self.valid_tiles) / len(self.tile_catalog)
            },
            'skip_reasons': dict(reason_counts),
            'recent_skips': [
                record.to_dict() for record in self.skip_records[-10:]
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def export_valid_tiles(self, output_path: Path):
        """Export list of valid tile IDs to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for tile_id in self.valid_tiles:
                f.write(f"{tile_id}\n")
        
        self.logger.info(f"✓ Exported {len(self.valid_tiles)} valid tile IDs to: {output_path}")
    
    def export_skip_records(self, output_path: Path):
        """Export all skip records to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'min_neighbors': self.min_neighbors,
                'max_temporal_gap': self.max_temporal_gap,
                'max_spatial_distance': self.max_spatial_distance,
                'max_cloud_cover': self.max_cloud_cover
            },
            'skip_records': [record.to_dict() for record in self.skip_records]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"✓ Exported {len(self.skip_records)} skip records to: {output_path}")


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate HLS neighbor availability for Stage 2 training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--tile_catalog',
        type=Path,
        required=True,
        help='Path to tile catalog CSV'
    )
    
    parser.add_argument(
        '--hls_index',
        type=Path,
        required=True,
        help='Path to HLS index CSV'
    )
    
    parser.add_argument(
        '--min_neighbors',
        type=int,
        default=3,
        help='Minimum number of HLS neighbors (default: 3)'
    )
    
    parser.add_argument(
        '--max_temporal_gap',
        type=int,
        default=15,
        help='Maximum temporal gap in days (default: 15)'
    )
    
    parser.add_argument(
        '--max_spatial_distance',
        type=float,
        default=100.0,
        help='Maximum spatial distance in km (default: 100.0)'
    )
    
    parser.add_argument(
        '--max_cloud_cover',
        type=float,
        default=0.3,
        help='Maximum cloud cover fraction (default: 0.3)'
    )
    
    parser.add_argument(
        '--log_file',
        type=Path,
        default=Path('logs/sampler_skipped_tiles.log'),
        help='Path to save skip log'
    )
    
    parser.add_argument(
        '--output_valid_tiles',
        type=Path,
        default=Path('data/valid_tiles.txt'),
        help='Path to save valid tile IDs'
    )
    
    parser.add_argument(
        '--output_skip_records',
        type=Path,
        default=Path('data/skip_records.json'),
        help='Path to save skip records JSON'
    )
    
    args = parser.parse_args()
    
    # Create sampler
    sampler = HLSNeighborSampler(
        tile_catalog_path=args.tile_catalog,
        hls_index_path=args.hls_index,
        min_neighbors=args.min_neighbors,
        max_temporal_gap=args.max_temporal_gap,
        max_spatial_distance=args.max_spatial_distance,
        max_cloud_cover=args.max_cloud_cover,
        log_file=args.log_file,
        verbose=True
    )
    
    # Filter tiles
    valid_tiles = sampler.filter_tiles()
    
    # Print statistics
    sampler.print_statistics()
    
    # Export results
    sampler.export_valid_tiles(args.output_valid_tiles)
    sampler.export_skip_records(args.output_skip_records)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
