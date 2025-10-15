#!/usr/bin/env python3
"""
scripts/warm_cache.py - Demo Data Cache Warmer

Fetches and prepares demo tiles from small AOIs for testing, examples, and
quick validation. Generates ~30-60 tiles from 2 diverse locations and 2 dates.

Purpose:
    - Quick testing of tile pipeline
    - Example data for documentation
    - Validation of alignment and quality
    - Demo for new users

Output:
    data/tiles/demo/
    ├── nairobi_kenya_2024-01-15/
    │   ├── tile_*.npz
    │   └── tile_*.json
    └── iowa_usa_2023-07-15/
        ├── tile_*.npz
        └── tile_*.json

Usage:
    python scripts/warm_cache.py
    python scripts/warm_cache.py --quick  # Faster, fewer tiles
    python scripts/warm_cache.py --full   # More tiles, all dates

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import build_tiles functions
try:
    from scripts.build_tiles import PipelineConfig, run_pipeline
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False
    warnings.warn("Could not import build_tiles pipeline")


# ============================================================================
# Demo AOI Configuration
# ============================================================================

DEMO_AOIS = {
    'nairobi_kenya': {
        'bbox': [36.75, -1.35, 36.85, -1.25],  # Nairobi city center
        'description': 'Urban area with clear features',
        'dates': ['2024-01-15', '2023-07-15'],
        'expected_tiles': 15,  # Small urban area
        'cloud_threshold': 30.0,
        'reasons': [
            '✓ Urban features (roads, buildings) good for alignment testing',
            '✓ Near equator (frequent satellite coverage)',
            '✓ Typically cloud-free in January',
            '✓ Mix of SAR and optical features'
        ]
    },
    'iowa_usa': {
        'bbox': [-93.2, 41.8, -93.0, 42.0],  # Central Iowa cropland
        'description': 'Agricultural area with field patterns',
        'dates': ['2023-07-15', '2023-01-15'],
        'expected_tiles': 20,  # Larger rural area
        'cloud_threshold': 40.0,
        'reasons': [
            '✓ Agricultural fields show clear boundaries',
            '✓ Good for testing vegetation indices',
            '✓ Regular grid patterns test alignment',
            '✓ Seasonal contrast (summer vs winter)'
        ]
    }
}


# ============================================================================
# Configuration Presets
# ============================================================================

def get_quick_config():
    """Quick mode: ~15-20 tiles, 1 AOI, 1 date."""
    return {
        'aois': ['nairobi_kenya'],
        'dates_per_aoi': 1,
        'tile_size': 256,
        'overlap': 0,
        'max_cloud': 40.0,
        'split_ratio': (0.7, 0.15, 0.15)
    }


def get_standard_config():
    """Standard mode: ~30-40 tiles, 2 AOIs, 1 date each."""
    return {
        'aois': ['nairobi_kenya', 'iowa_usa'],
        'dates_per_aoi': 1,
        'tile_size': 256,
        'overlap': 0,
        'max_cloud': 40.0,
        'split_ratio': (0.7, 0.15, 0.15)
    }


def get_full_config():
    """Full mode: ~60-80 tiles, 2 AOIs, 2 dates each."""
    return {
        'aois': ['nairobi_kenya', 'iowa_usa'],
        'dates_per_aoi': 2,
        'tile_size': 256,
        'overlap': 32,  # Some overlap for testing
        'max_cloud': 40.0,
        'split_ratio': (0.7, 0.15, 0.15)
    }


# ============================================================================
# Main Functions
# ============================================================================

def print_aoi_info(aoi_name: str, aoi_config: dict):
    """Print information about an AOI."""
    print(f"\n{aoi_name.upper().replace('_', ' ')}")
    print(f"  {aoi_config['description']}")
    print(f"  BBox: {aoi_config['bbox']}")
    print(f"  Dates: {', '.join(aoi_config['dates'][:2])}")
    print(f"  Expected tiles: ~{aoi_config['expected_tiles']}")
    print(f"  Reasons:")
    for reason in aoi_config['reasons']:
        print(f"    {reason}")


def generate_demo_tiles(mode: str = 'standard', output_dir: Path = None):
    """
    Generate demo tiles for testing and examples.
    
    Args:
        mode: 'quick', 'standard', or 'full'
        output_dir: Output directory (default: data/tiles/demo)
    """
    if not HAS_PIPELINE:
        print("✗ ERROR: Could not import build_tiles pipeline")
        print("  Make sure scripts/build_tiles.py exists and is importable")
        return 1
    
    # Get configuration
    if mode == 'quick':
        config = get_quick_config()
    elif mode == 'full':
        config = get_full_config()
    else:
        config = get_standard_config()
    
    if output_dir is None:
        output_dir = Path('data/tiles/demo')
    
    # Header
    print("=" * 79)
    print("DEMO TILE CACHE WARMER")
    print("=" * 79)
    print()
    print(f"Mode: {mode.upper()}")
    print(f"Output: {output_dir}")
    print()
    
    # Show what we'll fetch
    print("Demo AOIs selected:")
    print("-" * 79)
    for aoi_name in config['aois']:
        print_aoi_info(aoi_name, DEMO_AOIS[aoi_name])
    
    print()
    print(f"Configuration:")
    print(f"  Tile size: {config['tile_size']}x{config['tile_size']}")
    print(f"  Overlap: {config['overlap']} pixels")
    print(f"  Max cloud: {config['max_cloud']}%")
    print(f"  Dates per AOI: {config['dates_per_aoi']}")
    print()
    
    total_expected = sum(
        DEMO_AOIS[aoi]['expected_tiles'] * config['dates_per_aoi']
        for aoi in config['aois']
    )
    print(f"Expected total tiles: ~{total_expected}")
    print()
    
    # Confirm
    try:
        response = input("Proceed with download? [Y/n]: ").strip().lower()
        if response and response != 'y':
            print("Cancelled by user")
            return 0
    except KeyboardInterrupt:
        print("\nCancelled by user")
        return 0
    
    print()
    print("=" * 79)
    print("STARTING TILE GENERATION")
    print("=" * 79)
    
    # Process each AOI
    results = []
    total_tiles = 0
    
    for aoi_name in config['aois']:
        aoi_config = DEMO_AOIS[aoi_name]
        
        # Process dates for this AOI
        dates_to_process = aoi_config['dates'][:config['dates_per_aoi']]
        
        for date in dates_to_process:
            print()
            print("=" * 79)
            print(f"Processing: {aoi_name} - {date}")
            print("=" * 79)
            print()
            
            # Create pipeline config
            pipeline_config = PipelineConfig(
                bbox=tuple(aoi_config['bbox']),
                date=date,
                date_range_days=7,
                tile_size=config['tile_size'],
                overlap=config['overlap'],
                max_cloud_percent=config['max_cloud'],
                split_ratio=tuple(config['split_ratio']),
                output_dir=output_dir,
                index_dir=Path('data/index/demo'),
                cache_dir=Path('cache/demo'),
                stac_provider='planetary_computer',
                collections=['sentinel-2-l2a', 'sentinel-1-grd'],
                random_seed=42
            )
            
            try:
                # Run pipeline
                run_pipeline(pipeline_config)
                
                # Count generated tiles
                output_name = f"{aoi_name}_{date}"
                tile_dir = output_dir / output_name
                if tile_dir.exists():
                    tiles = list(tile_dir.glob('*.npz'))
                    n_tiles = len(tiles)
                    total_tiles += n_tiles
                    print(f"\n✓ Generated {n_tiles} tiles for {aoi_name} - {date}")
                    results.append({
                        'aoi': aoi_name,
                        'date': date,
                        'tiles': n_tiles,
                        'success': True
                    })
                else:
                    print(f"\n⚠ No tiles generated for {aoi_name} - {date}")
                    results.append({
                        'aoi': aoi_name,
                        'date': date,
                        'tiles': 0,
                        'success': False
                    })
                
            except Exception as e:
                print(f"\n✗ ERROR processing {aoi_name} - {date}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'aoi': aoi_name,
                    'date': date,
                    'error': str(e),
                    'success': False
                })
    
    # Summary
    print()
    print("=" * 79)
    print("CACHE WARMING COMPLETE")
    print("=" * 79)
    print()
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print("Summary:")
    print(f"  Total AOI-date combinations: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Total tiles generated: {total_tiles}")
    print()
    
    if successful:
        print("Successfully processed:")
        for r in successful:
            print(f"  ✓ {r['aoi']} - {r['date']}: {r['tiles']} tiles")
    
    if failed:
        print()
        print("Failed:")
        for r in failed:
            error = r.get('error', 'Unknown error')
            print(f"  ✗ {r['aoi']} - {r['date']}: {error}")
    
    print()
    print("Demo tiles location:")
    print(f"  {output_dir.absolute()}")
    print()
    
    # Next steps
    print("Next steps:")
    print("  1. Visualize alignment:")
    print(f"     python -c \"from axs_lib.viz import show_alignment; show_alignment(list(Path('{output_dir}').rglob('*.npz'))[0])\"")
    print()
    print("  2. Check tile quality:")
    print(f"     python examples/example_alignment_check.py")
    print()
    print("  3. Use for training:")
    print(f"     import pandas as pd")
    print(f"     index = pd.read_csv('data/index/demo/*.csv')")
    print()
    
    return 0 if failed == [] else 1


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Warm cache with demo satellite imagery tiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Demo AOIs:
  1. Nairobi, Kenya - Urban area with clear features
  2. Iowa, USA - Agricultural croplands with field patterns

Modes:
  --quick:    ~15-20 tiles, 1 AOI, 1 date (fastest, for testing)
  --standard: ~30-40 tiles, 2 AOIs, 1 date each (default, balanced)
  --full:     ~60-80 tiles, 2 AOIs, 2 dates each (comprehensive)

Examples:
  python scripts/warm_cache.py                    # Standard mode
  python scripts/warm_cache.py --quick            # Quick test
  python scripts/warm_cache.py --full             # Full dataset
  python scripts/warm_cache.py --output data/test # Custom location
        """
    )
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: ~15-20 tiles, 1 AOI, 1 date'
    )
    mode_group.add_argument(
        '--full',
        action='store_true',
        help='Full mode: ~60-80 tiles, 2 AOIs, 2 dates each'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/tiles/demo'),
        help='Output directory (default: data/tiles/demo)'
    )
    
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    parser.add_argument(
        '--list-aois',
        action='store_true',
        help='List available demo AOIs and exit'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # List AOIs if requested
    if args.list_aois:
        print("=" * 79)
        print("AVAILABLE DEMO AOIs")
        print("=" * 79)
        for aoi_name, aoi_config in DEMO_AOIS.items():
            print_aoi_info(aoi_name, aoi_config)
        print()
        return 0
    
    # Determine mode
    if args.quick:
        mode = 'quick'
    elif args.full:
        mode = 'full'
    else:
        mode = 'standard'
    
    # Override confirmation if --yes
    if args.yes:
        # Monkey patch input to always return 'y'
        import builtins
        builtins.input = lambda _: 'y'
    
    # Generate tiles
    try:
        return generate_demo_tiles(mode=mode, output_dir=args.output)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
