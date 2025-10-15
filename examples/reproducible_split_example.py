#!/usr/bin/env python3
"""
examples/reproducible_split_example.py - Reproducible Split Demo

Demonstrates how to create reproducible train/val/test splits with seed tracking
and verification. Shows how split configuration is recorded in manifests.

Usage:
    python examples/reproducible_split_example.py

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axs_lib.split import DatasetSplitter, split_tiles, verify_reproducibility
import numpy as np


def example_1_basic_split():
    """Example 1: Basic reproducible split with seed."""
    print("=" * 80)
    print("Example 1: Basic Reproducible Split")
    print("=" * 80)
    print()
    
    # Create sample tile list
    tiles = [
        {
            'tile_path': f'data/tiles/tile_{i:04d}.npz',
            'tile_id': f'tile_{i:04d}',
            'date': '2023-01-15',
            'region': f'region_{i % 5}'
        }
        for i in range(100)
    ]
    
    print(f"Created {len(tiles)} sample tiles")
    print()
    
    # Split with seed=42
    splitter = DatasetSplitter(
        method='random',
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    split_tiles_list = splitter.split(tiles)
    stats = splitter.get_split_stats(split_tiles_list)
    
    print("Split distribution:")
    for split_name, count in stats.items():
        pct = (count / len(tiles)) * 100
        print(f"  {split_name:<6}: {count:>3} tiles ({pct:>5.1f}%)")
    
    print()
    print("Reproducibility metadata:")
    print(f"  Seed        : {splitter.config.random_seed}")
    print(f"  Config hash : {splitter.config.config_hash}")
    print(f"  Timestamp   : {splitter.config.timestamp}")
    print()
    
    return split_tiles_list, splitter


def example_2_verify_reproducibility():
    """Example 2: Verify that same seed produces same split."""
    print("=" * 80)
    print("Example 2: Verify Reproducibility")
    print("=" * 80)
    print()
    
    # Create tiles
    tiles = [
        {'tile_path': f'tile_{i:04d}.npz', 'tile_id': i}
        for i in range(100)
    ]
    
    # First split with seed=42
    splitter1 = DatasetSplitter(random_seed=42)
    split1 = splitter1.split(tiles.copy())
    
    # Second split with same seed
    splitter2 = DatasetSplitter(random_seed=42)
    split2 = splitter2.split(tiles.copy())
    
    # Compare
    matches = sum(
        1 for t1, t2 in zip(split1, split2)
        if t1['split'] == t2['split']
    )
    
    print(f"Tiles in split 1: {len(split1)}")
    print(f"Tiles in split 2: {len(split2)}")
    print(f"Matching splits : {matches}")
    print(f"Match rate      : {(matches / len(split1)) * 100:.1f}%")
    print()
    
    if matches == len(split1):
        print("✓ Splits are identical - fully reproducible!")
    else:
        print("✗ Splits differ - not reproducible!")
    print()


def example_3_different_seeds():
    """Example 3: Different seeds produce different splits."""
    print("=" * 80)
    print("Example 3: Different Seeds = Different Splits")
    print("=" * 80)
    print()
    
    tiles = [
        {'tile_path': f'tile_{i:04d}.npz'}
        for i in range(50)
    ]
    
    seeds = [42, 123, 999]
    all_splits = []
    
    for seed in seeds:
        splitter = DatasetSplitter(random_seed=seed)
        split = splitter.split(tiles.copy())
        all_splits.append(split)
        
        stats = splitter.get_split_stats(split)
        print(f"Seed {seed:>3}: train={stats['train']}, val={stats['val']}, test={stats['test']}")
    
    print()
    
    # Compare first tile's assignment across seeds
    print("First tile assignment across seeds:")
    for i, seed in enumerate(seeds):
        first_tile_split = all_splits[i][0]['split']
        print(f"  Seed {seed:>3}: {first_tile_split}")
    
    print()
    
    # Check if all splits differ
    different = (
        all_splits[0][0]['split'] != all_splits[1][0]['split'] or
        all_splits[1][0]['split'] != all_splits[2][0]['split']
    )
    
    if different:
        print("✓ Different seeds produce different splits (as expected)")
    else:
        print("⚠ Coincidentally got same split (very unlikely)")
    print()


def example_4_stratified_split():
    """Example 4: Stratified split (balanced across regions)."""
    print("=" * 80)
    print("Example 4: Stratified Split")
    print("=" * 80)
    print()
    
    # Create tiles with regions
    tiles = []
    regions = ['north', 'south', 'east', 'west']
    
    for i in range(100):
        tiles.append({
            'tile_path': f'tile_{i:04d}.npz',
            'region': regions[i % len(regions)]
        })
    
    print(f"Created {len(tiles)} tiles across {len(regions)} regions")
    print()
    
    # Split with stratification
    splitter = DatasetSplitter(random_seed=42)
    split_tiles_list = splitter.split(tiles, stratify_column='region')
    
    # Analyze stratification
    print("Split distribution by region:")
    print()
    
    for region in regions:
        region_tiles = [t for t in split_tiles_list if t['region'] == region]
        region_stats = {}
        for t in region_tiles:
            split = t['split']
            region_stats[split] = region_stats.get(split, 0) + 1
        
        total = len(region_tiles)
        print(f"  {region:>6}: ", end='')
        for split in ['train', 'val', 'test']:
            count = region_stats.get(split, 0)
            pct = (count / total * 100) if total > 0 else 0
            print(f"{split}={count:>2} ({pct:>4.0f}%)  ", end='')
        print()
    
    print()
    print("✓ Stratification ensures balanced splits across regions")
    print()


def example_5_save_and_load():
    """Example 5: Save manifest with metadata and load it back."""
    print("=" * 80)
    print("Example 5: Save and Load Manifest")
    print("=" * 80)
    print()
    
    # Create tiles
    tiles = [
        {
            'tile_path': f'data/tiles/tile_{i:04d}.npz',
            'date': '2023-01-15',
            'cloud_cover': np.random.uniform(0, 30)
        }
        for i in range(50)
    ]
    
    # Split
    splitter = DatasetSplitter(random_seed=42)
    split_tiles_list = splitter.split(tiles)
    
    # Save manifest
    output_path = project_root / 'data' / 'index' / 'example_tiles.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Saving manifest...")
    splitter.save_manifest(split_tiles_list, output_path)
    print()
    
    # Load manifest back
    print("Loading manifest...")
    df, config = DatasetSplitter.load_manifest(output_path)
    
    print(f"✓ Loaded {len(df)} tiles")
    print(f"✓ Split config recovered:")
    print(f"    Method: {config.method}")
    print(f"    Seed  : {config.random_seed}")
    print(f"    Hash  : {config.config_hash}")
    print()
    
    # Verify split
    print("Verifying split...")
    is_valid = DatasetSplitter.verify_split(output_path, expected_seed=42)
    print()
    
    return output_path


def example_6_temporal_split():
    """Example 6: Temporal split by date."""
    print("=" * 80)
    print("Example 6: Temporal Split")
    print("=" * 80)
    print()
    
    from datetime import datetime, timedelta
    
    # Create tiles with different dates
    start_date = datetime(2020, 1, 1)
    tiles = []
    
    for i in range(100):
        date = start_date + timedelta(days=i * 10)
        tiles.append({
            'tile_path': f'tile_{i:04d}.npz',
            'date': date.strftime('%Y-%m-%d')
        })
    
    print(f"Created {len(tiles)} tiles from {tiles[0]['date']} to {tiles[-1]['date']}")
    print()
    
    # Temporal split
    splitter = DatasetSplitter(
        method='temporal',
        val_date='2021-01-01',
        test_date='2022-01-01',
        random_seed=None  # Not used for temporal split
    )
    
    split_tiles_list = splitter.split(tiles)
    stats = splitter.get_split_stats(split_tiles_list)
    
    print("Temporal split distribution:")
    print(f"  Train: {stats['train']} tiles (before 2021-01-01)")
    print(f"  Val  : {stats['val']} tiles (2021-01-01 to 2022-01-01)")
    print(f"  Test : {stats['test']} tiles (after 2022-01-01)")
    print()
    
    # Show date ranges
    for split_name in ['train', 'val', 'test']:
        split_dates = [t['date'] for t in split_tiles_list if t['split'] == split_name]
        if split_dates:
            print(f"  {split_name:>5}: {min(split_dates)} to {max(split_dates)}")
    
    print()


def main():
    """Run all examples."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "REPRODUCIBLE DATASET SPLITTING EXAMPLES" + " " * 23 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    # Run examples
    example_1_basic_split()
    input("Press Enter to continue to Example 2...")
    print()
    
    example_2_verify_reproducibility()
    input("Press Enter to continue to Example 3...")
    print()
    
    example_3_different_seeds()
    input("Press Enter to continue to Example 4...")
    print()
    
    example_4_stratified_split()
    input("Press Enter to continue to Example 5...")
    print()
    
    manifest_path = example_5_save_and_load()
    input("Press Enter to continue to Example 6...")
    print()
    
    example_6_temporal_split()
    
    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)
    print()
    print("Key takeaways:")
    print("  1. Use fixed random_seed (e.g., 42) for reproducible splits")
    print("  2. Seed is automatically recorded in manifests and metadata")
    print("  3. Same seed + same tiles = identical splits every time")
    print("  4. Verify reproducibility with DatasetSplitter.verify_split()")
    print("  5. Use stratification to balance splits across regions/classes")
    print()
    print(f"Example manifest saved to: {manifest_path}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
