"""
axs_lib/split.py - Reproducible Dataset Splitting

Provides deterministic train/val/test splitting with seed tracking for 
reproducibility. Records split configuration in tile manifests and metadata.

Key features:
- Deterministic random splitting with seed control
- Temporal splitting by acquisition date
- Spatial splitting by geographic region
- Stratified splitting (balanced by region/class)
- Automatic seed recording in manifests and metadata
- Split verification and validation

Usage:
    from axs_lib.split import DatasetSplitter, split_tiles
    
    # Create splitter with reproducible seed
    splitter = DatasetSplitter(
        method='random',
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    # Split tiles and record seed
    assignments = splitter.split(tile_list)
    
    # Save manifest with split metadata
    splitter.save_manifest(assignments, output_path='data/index/tiles.csv')

Author: Axion-Sat Project
Version: 1.0.0
"""

import os
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Literal, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import hashlib

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    warnings.warn("pandas not available - install with: pip install pandas")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""
    
    method: Literal['random', 'temporal', 'spatial'] = 'random'
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: Optional[int] = 42
    
    # Temporal split parameters
    val_date: Optional[str] = None
    test_date: Optional[str] = None
    
    # Spatial split parameters
    regions: Optional[Dict[str, str]] = None  # region_name -> split assignment
    
    # Stratification
    stratify_by: Optional[str] = None  # Column name to stratify by
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    config_hash: str = field(default="")
    
    def __post_init__(self):
        """Validate and compute config hash."""
        # Validate ratios
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0 (got {total}). "
                f"train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}"
            )
        
        # Validate method
        if self.method not in ['random', 'temporal', 'spatial']:
            raise ValueError(f"Invalid split method: {self.method}")
        
        # Validate temporal parameters
        if self.method == 'temporal':
            if not self.val_date or not self.test_date:
                raise ValueError("Temporal split requires val_date and test_date")
        
        # Validate spatial parameters
        if self.method == 'spatial':
            if not self.regions:
                raise ValueError("Spatial split requires region definitions")
        
        # Compute config hash for reproducibility tracking
        self.config_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute deterministic hash of configuration."""
        config_str = (
            f"{self.method}|"
            f"{self.train_ratio}|{self.val_ratio}|{self.test_ratio}|"
            f"{self.random_seed}|"
            f"{self.val_date}|{self.test_date}|"
            f"{self.regions}|{self.stratify_by}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SplitConfig':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, config_dict: Dict) -> 'SplitConfig':
        """Create from YAML config dictionary (tiling.yaml format)."""
        split_config = config_dict.get('split', {})
        
        method = split_config.get('method', 'random')
        
        # Basic parameters
        config = cls(
            method=method,
            train_ratio=split_config.get('train_ratio', 0.70),
            val_ratio=split_config.get('val_ratio', 0.15),
            test_ratio=split_config.get('test_ratio', 0.15),
            random_seed=split_config.get('random_seed', 42),
        )
        
        # Temporal parameters
        if method == 'temporal':
            temporal_split = split_config.get('temporal_split', {})
            config.val_date = temporal_split.get('val_date')
            config.test_date = temporal_split.get('test_date')
        
        # Spatial parameters
        if method == 'spatial':
            spatial_split = split_config.get('spatial_split', {})
            regions_list = spatial_split.get('regions', [])
            config.regions = {r['name']: r['split'] for r in regions_list}
        
        # Stratification
        config.stratify_by = split_config.get('stratify_by', None)
        
        return config


# ============================================================================
# Dataset Splitter
# ============================================================================

class DatasetSplitter:
    """
    Handles reproducible dataset splitting with seed tracking.
    
    Example:
        >>> splitter = DatasetSplitter(method='random', random_seed=42)
        >>> assignments = splitter.split(tile_list)
        >>> print(f"Split seed: {splitter.config.random_seed}")
    """
    
    def __init__(
        self,
        method: Literal['random', 'temporal', 'spatial'] = 'random',
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: Optional[int] = 42,
        **kwargs
    ):
        """
        Initialize dataset splitter.
        
        Args:
            method: Split method ('random', 'temporal', or 'spatial')
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            random_seed: Random seed for reproducibility (None = non-deterministic)
            **kwargs: Additional method-specific parameters
        """
        self.config = SplitConfig(
            method=method,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
            **kwargs
        )
        
        # Set random seed for reproducibility
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
    
    def split(
        self,
        items: List[Dict],
        stratify_column: Optional[str] = None
    ) -> List[Dict]:
        """
        Split dataset into train/val/test.
        
        Args:
            items: List of item dictionaries (e.g., tile metadata)
            stratify_column: Column name to stratify by (e.g., 'region')
            
        Returns:
            List of items with 'split' key added
        """
        if len(items) == 0:
            return []
        
        # Dispatch to appropriate split method
        if self.config.method == 'random':
            return self._split_random(items, stratify_column)
        elif self.config.method == 'temporal':
            return self._split_temporal(items)
        elif self.config.method == 'spatial':
            return self._split_spatial(items)
        else:
            raise ValueError(f"Unknown split method: {self.config.method}")
    
    def _split_random(
        self,
        items: List[Dict],
        stratify_column: Optional[str] = None
    ) -> List[Dict]:
        """Random split with optional stratification."""
        n = len(items)
        
        if stratify_column and stratify_column in items[0]:
            # Stratified split
            return self._split_stratified(items, stratify_column)
        else:
            # Simple random split
            indices = np.arange(n)
            np.random.shuffle(indices)
            
            # Compute split indices
            n_train = int(n * self.config.train_ratio)
            n_val = int(n * self.config.val_ratio)
            
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]
            
            # Assign splits
            for i, item in enumerate(items):
                if i in train_idx:
                    item['split'] = 'train'
                elif i in val_idx:
                    item['split'] = 'val'
                else:
                    item['split'] = 'test'
            
            return items
    
    def _split_stratified(
        self,
        items: List[Dict],
        stratify_column: str
    ) -> List[Dict]:
        """Stratified random split (balanced across groups)."""
        # Group items by stratification column
        groups = {}
        for i, item in enumerate(items):
            key = item.get(stratify_column, 'unknown')
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        
        # Split each group
        train_indices = []
        val_indices = []
        test_indices = []
        
        for group_indices in groups.values():
            n = len(group_indices)
            shuffled = np.array(group_indices)
            np.random.shuffle(shuffled)
            
            n_train = int(n * self.config.train_ratio)
            n_val = int(n * self.config.val_ratio)
            
            train_indices.extend(shuffled[:n_train])
            val_indices.extend(shuffled[n_train:n_train + n_val])
            test_indices.extend(shuffled[n_train + n_val:])
        
        # Assign splits
        train_set = set(train_indices)
        val_set = set(val_indices)
        
        for i, item in enumerate(items):
            if i in train_set:
                item['split'] = 'train'
            elif i in val_set:
                item['split'] = 'val'
            else:
                item['split'] = 'test'
        
        return items
    
    def _split_temporal(self, items: List[Dict]) -> List[Dict]:
        """Temporal split by acquisition date."""
        if not self.config.val_date or not self.config.test_date:
            raise ValueError("Temporal split requires val_date and test_date")
        
        from datetime import datetime
        
        val_cutoff = datetime.fromisoformat(self.config.val_date)
        test_cutoff = datetime.fromisoformat(self.config.test_date)
        
        for item in items:
            # Get acquisition date
            date_str = item.get('date', item.get('acquisition_date', None))
            if not date_str:
                warnings.warn(f"Item missing date field: {item}")
                item['split'] = 'train'  # Default to train
                continue
            
            # Parse date
            try:
                item_date = datetime.fromisoformat(date_str.split('T')[0])
            except Exception as e:
                warnings.warn(f"Failed to parse date {date_str}: {e}")
                item['split'] = 'train'
                continue
            
            # Assign split
            if item_date < val_cutoff:
                item['split'] = 'train'
            elif item_date < test_cutoff:
                item['split'] = 'val'
            else:
                item['split'] = 'test'
        
        return items
    
    def _split_spatial(self, items: List[Dict]) -> List[Dict]:
        """Spatial split by geographic region."""
        if not self.config.regions:
            raise ValueError("Spatial split requires region definitions")
        
        for item in items:
            region = item.get('region', 'unknown')
            split = self.config.regions.get(region, 'train')  # Default to train
            item['split'] = split
        
        return items
    
    def get_split_stats(self, items: List[Dict]) -> Dict[str, int]:
        """Get statistics about split distribution."""
        stats = {'train': 0, 'val': 0, 'test': 0}
        
        for item in items:
            split = item.get('split', 'unknown')
            if split in stats:
                stats[split] += 1
        
        return stats
    
    def save_manifest(
        self,
        items: List[Dict],
        output_path: Union[str, Path],
        include_metadata: bool = True
    ) -> None:
        """
        Save manifest CSV with split assignments and metadata.
        
        Args:
            items: List of items with 'split' assignments
            output_path: Output CSV file path
            include_metadata: Include split metadata in CSV header comments
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required. Install with: pip install pandas")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(items)
        
        # Ensure split column exists
        if 'split' not in df.columns:
            warnings.warn("No 'split' column found in items")
            df['split'] = 'train'  # Default
        
        # Add reproducibility metadata columns
        if include_metadata:
            df['split_method'] = self.config.method
            df['split_seed'] = self.config.random_seed
            df['split_config_hash'] = self.config.config_hash
            df['split_timestamp'] = self.config.timestamp
        
        # Save CSV
        df.to_csv(output_path, index=False)
        
        # Save split configuration as sidecar JSON
        config_path = output_path.with_suffix('.split.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        print(f"✓ Saved manifest: {output_path}")
        print(f"✓ Saved split config: {config_path}")
        
        # Print statistics
        stats = self.get_split_stats(items)
        total = sum(stats.values())
        print(f"\nSplit distribution:")
        for split, count in stats.items():
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {split:<6}: {count:>6} ({pct:>5.1f}%)")
        print(f"  Total : {total:>6}")
        print(f"\nReproducibility:")
        print(f"  Method      : {self.config.method}")
        print(f"  Random seed : {self.config.random_seed}")
        print(f"  Config hash : {self.config.config_hash}")
    
    @staticmethod
    def load_manifest(
        manifest_path: Union[str, Path]
    ) -> Tuple[pd.DataFrame, Optional[SplitConfig]]:
        """
        Load manifest CSV and split configuration.
        
        Args:
            manifest_path: Path to manifest CSV
            
        Returns:
            Tuple of (DataFrame, SplitConfig or None)
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required. Install with: pip install pandas")
        
        manifest_path = Path(manifest_path)
        
        # Load CSV
        df = pd.read_csv(manifest_path)
        
        # Try to load split config
        config_path = manifest_path.with_suffix('.split.json')
        config = None
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                config = SplitConfig.from_dict(config_data)
        else:
            # Try to reconstruct from CSV metadata columns
            if 'split_seed' in df.columns:
                config = SplitConfig(
                    method=df['split_method'].iloc[0] if 'split_method' in df.columns else 'random',
                    random_seed=int(df['split_seed'].iloc[0]) if pd.notna(df['split_seed'].iloc[0]) else None,
                )
        
        return df, config
    
    @staticmethod
    def verify_split(
        manifest_path: Union[str, Path],
        expected_seed: Optional[int] = None
    ) -> bool:
        """
        Verify that manifest has valid split assignments and seed.
        
        Args:
            manifest_path: Path to manifest CSV
            expected_seed: Expected random seed (None = don't check)
            
        Returns:
            True if split is valid, False otherwise
        """
        try:
            df, config = DatasetSplitter.load_manifest(manifest_path)
            
            # Check split column exists
            if 'split' not in df.columns:
                print(f"✗ No 'split' column in manifest")
                return False
            
            # Check all splits are valid
            valid_splits = {'train', 'val', 'test'}
            invalid = df[~df['split'].isin(valid_splits)]
            if len(invalid) > 0:
                print(f"✗ Found {len(invalid)} invalid split values")
                return False
            
            # Check seed if expected
            if expected_seed is not None:
                if config is None or config.random_seed != expected_seed:
                    actual_seed = config.random_seed if config else None
                    print(f"✗ Seed mismatch: expected {expected_seed}, got {actual_seed}")
                    return False
            
            # Check split distribution
            stats = df['split'].value_counts()
            total = len(df)
            print(f"✓ Split validation passed ({total} tiles)")
            for split in ['train', 'val', 'test']:
                count = stats.get(split, 0)
                pct = (count / total * 100) if total > 0 else 0
                print(f"  {split}: {count} ({pct:.1f}%)")
            
            if config:
                print(f"  Seed: {config.random_seed}")
                print(f"  Hash: {config.config_hash}")
            
            return True
        
        except Exception as e:
            print(f"✗ Split verification failed: {e}")
            return False


# ============================================================================
# Convenience Functions
# ============================================================================

def split_tiles(
    tiles: List[Dict],
    method: str = 'random',
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: Optional[int] = 42,
    **kwargs
) -> Tuple[List[Dict], SplitConfig]:
    """
    Convenience function to split tiles with reproducible seed.
    
    Args:
        tiles: List of tile metadata dictionaries
        method: Split method ('random', 'temporal', 'spatial')
        train_ratio: Training set fraction
        val_ratio: Validation set fraction
        test_ratio: Test set fraction
        random_seed: Random seed for reproducibility
        **kwargs: Additional method-specific parameters
        
    Returns:
        Tuple of (tiles with split assignments, split configuration)
    
    Example:
        >>> tiles = [{'tile_path': 'tile_001.npz', 'date': '2023-01-01'}, ...]
        >>> split_tiles, config = split_tiles(tiles, random_seed=42)
        >>> print(f"Split with seed: {config.random_seed}")
    """
    splitter = DatasetSplitter(
        method=method,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        **kwargs
    )
    
    split_items = splitter.split(tiles)
    
    return split_items, splitter.config


def verify_reproducibility(
    manifest_path: Union[str, Path],
    tiles: List[Dict],
    random_seed: int
) -> bool:
    """
    Verify that split can be reproduced with the same seed.
    
    Args:
        manifest_path: Path to existing manifest
        tiles: Original tiles (before splitting)
        random_seed: Seed to test
        
    Returns:
        True if split is reproducible, False otherwise
    """
    # Load existing split
    df_existing, config = DatasetSplitter.load_manifest(manifest_path)
    
    if config is None:
        print("✗ No split config found in manifest")
        return False
    
    # Re-split with same seed
    splitter = DatasetSplitter(
        method=config.method,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=random_seed
    )
    
    resplit_tiles = splitter.split(tiles.copy())
    
    # Compare splits
    df_resplit = pd.DataFrame(resplit_tiles)
    
    # Check if splits match
    matches = (df_existing['split'] == df_resplit['split']).sum()
    total = len(df_existing)
    match_rate = (matches / total * 100) if total > 0 else 0
    
    print(f"\nReproducibility test:")
    print(f"  Matching splits: {matches}/{total} ({match_rate:.1f}%)")
    
    if match_rate == 100.0:
        print("  ✓ Split is fully reproducible")
        return True
    else:
        print("  ✗ Split differs from original")
        return False


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Basic random split
    print("=" * 80)
    print("Example 1: Random Split with Seed")
    print("=" * 80)
    
    tiles = [
        {'tile_path': f'tile_{i:03d}.npz', 'region': f'region_{i % 3}'}
        for i in range(100)
    ]
    
    splitter = DatasetSplitter(
        method='random',
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    split_tiles = splitter.split(tiles)
    stats = splitter.get_split_stats(split_tiles)
    
    print(f"Split {len(tiles)} tiles:")
    for split, count in stats.items():
        print(f"  {split}: {count}")
    
    print(f"\nSeed: {splitter.config.random_seed}")
    print(f"Config hash: {splitter.config.config_hash}")
    
    # Example 2: Verify reproducibility
    print("\n" + "=" * 80)
    print("Example 2: Verify Reproducibility")
    print("=" * 80)
    
    # Create second splitter with same seed
    splitter2 = DatasetSplitter(
        method='random',
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42  # Same seed
    )
    
    tiles2 = [{'tile_path': f'tile_{i:03d}.npz'} for i in range(100)]
    split_tiles2 = splitter2.split(tiles2)
    
    # Compare
    matches = sum(
        1 for t1, t2 in zip(split_tiles, split_tiles2)
        if t1['split'] == t2['split']
    )
    
    print(f"Matching splits: {matches}/{len(tiles)}")
    print(f"Reproducible: {matches == len(tiles)}")
