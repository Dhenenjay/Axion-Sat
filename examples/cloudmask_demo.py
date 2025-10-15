"""
Cloud Masking Demo - Example Usage

Demonstrates how to use axs_lib.cloudmask for quality filtering
of Sentinel-2 imagery in data processing pipelines.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axs_lib.cloudmask import (
    parse_scl,
    calculate_coverage,
    filter_by_cloud_cover,
    create_combined_mask
)


def demo_basic_usage():
    """Basic usage: Parse SCL and check quality."""
    print("=" * 79)
    print("DEMO 1: Basic Cloud Masking")
    print("=" * 79)
    print()
    
    # Simulate loading SCL band from a Sentinel-2 tile
    # In practice, you'd load this from a COG file:
    # from axs_lib.geo import read_cog
    # scl_data, profile = read_cog('path/to/SCL.tif')
    
    # For demo, create synthetic SCL data
    np.random.seed(42)
    scl_data = np.random.choice(
        [4, 5, 6, 8, 9, 3],  # vegetation, bare, water, cloud_med, cloud_high, shadow
        size=(512, 512),
        p=[0.5, 0.2, 0.15, 0.08, 0.05, 0.02]  # 13% cloud, 2% shadow
    )
    
    print(f"Tile size: {scl_data.shape}")
    print()
    
    # Step 1: Check if tile is usable
    is_valid, stats = filter_by_cloud_cover(
        scl_data,
        max_cloud_percent=40.0,
        verbose=True
    )
    print()
    
    if is_valid:
        print("✓ Tile quality OK, proceeding with processing")
        
        # Step 2: Parse SCL to masks
        cloud_mask, shadow_mask, valid_mask = parse_scl(scl_data)
        
        print(f"\nMask Summary:")
        print(f"  Cloud pixels: {cloud_mask.sum():,} ({cloud_mask.mean()*100:.2f}%)")
        print(f"  Shadow pixels: {shadow_mask.sum():,} ({shadow_mask.mean()*100:.2f}%)")
        print(f"  Clear pixels: {(valid_mask & ~cloud_mask & ~shadow_mask).sum():,}")
        
        # Step 3: Apply masks to spectral bands
        # In practice, you'd load actual band data:
        # red_data, _ = read_cog('path/to/B04.tif')
        # nir_data, _ = read_cog('path/to/B08.tif')
        
        # For demo, simulate band data
        red_band = np.random.randint(0, 10000, scl_data.shape, dtype=np.uint16)
        
        # Mask out clouds
        red_masked = np.where(cloud_mask, np.nan, red_band.astype(float))
        
        print(f"\nBand masking:")
        print(f"  Original valid pixels: {(~np.isnan(red_band)).sum():,}")
        print(f"  After cloud masking: {(~np.isnan(red_masked)).sum():,}")
        
    else:
        print("✗ Tile quality poor, skipping")
    
    print()


def demo_combined_mask():
    """Create a single mask for all bad pixels."""
    print("=" * 79)
    print("DEMO 2: Combined Quality Mask")
    print("=" * 79)
    print()
    
    # Simulate SCL with various quality issues
    scl_data = np.full((100, 100), 4, dtype=np.uint8)  # Start with all vegetation
    scl_data[0:20, :] = 8  # Top 20% is cloud
    scl_data[20:30, :] = 3  # Next 10% is shadow
    scl_data[30:35, :] = 0  # 5% no data
    
    print(f"Simulated tile: {scl_data.shape}")
    
    # Calculate stats
    stats = calculate_coverage(scl_data)
    print(f"\nQuality breakdown:")
    print(f"  Cloud: {stats['cloud_percent']:.1f}%")
    print(f"  Shadow: {stats['shadow_percent']:.1f}%")
    print(f"  Invalid: {stats['invalid_percent']:.1f}%")
    print(f"  Clear: {stats['clear_percent']:.1f}%")
    
    # Create combined mask
    bad_pixel_mask = create_combined_mask(
        scl_data,
        mask_clouds=True,
        mask_shadows=True,
        mask_invalid=True
    )
    
    print(f"\nCombined mask: {bad_pixel_mask.sum():,}/{bad_pixel_mask.size:,} bad pixels")
    print(f"  ({bad_pixel_mask.mean()*100:.1f}% masked)")
    
    # Use mask on multiple bands at once
    print("\nApplying to multi-band imagery:")
    
    # Simulate RGB bands
    r = np.random.randint(0, 255, scl_data.shape, dtype=np.uint8)
    g = np.random.randint(0, 255, scl_data.shape, dtype=np.uint8)
    b = np.random.randint(0, 255, scl_data.shape, dtype=np.uint8)
    
    # Stack and mask
    rgb = np.stack([r, g, b], axis=-1)
    rgb_masked = np.where(bad_pixel_mask[:, :, np.newaxis], 0, rgb)
    
    print(f"  RGB shape: {rgb.shape}")
    print(f"  Masked RGB: {(rgb_masked == 0).all(axis=2).sum():,} black pixels")
    
    print()


def demo_batch_filtering():
    """Filter multiple tiles by cloud cover."""
    print("=" * 79)
    print("DEMO 3: Batch Tile Filtering")
    print("=" * 79)
    print()
    
    # Simulate a batch of tiles with varying cloud cover
    np.random.seed(123)
    
    tiles = []
    true_cloud_percents = []
    
    print("Generating synthetic tile batch...")
    for i in range(10):
        # Each tile has random cloud cover
        cloud_percent = np.random.uniform(0, 80)
        true_cloud_percents.append(cloud_percent)
        
        # Generate SCL with that cloud percentage
        n_pixels = 1000 * 1000
        n_cloud = int(n_pixels * cloud_percent / 100)
        n_clear = n_pixels - n_cloud
        
        scl = np.concatenate([
            np.full(n_clear, 4, dtype=np.uint8),  # Clear
            np.full(n_cloud, 8, dtype=np.uint8)   # Cloud
        ])
        np.random.shuffle(scl)
        scl = scl.reshape(1000, 1000)
        
        tiles.append(scl)
    
    print(f"Created {len(tiles)} tiles")
    print()
    
    # Filter tiles
    print("Filtering with 40% cloud threshold:")
    print("-" * 79)
    
    valid_tiles = []
    rejected_tiles = []
    
    for i, scl in enumerate(tiles):
        is_valid, stats = filter_by_cloud_cover(
            scl,
            max_cloud_percent=40.0,
            verbose=False
        )
        
        status = "✓ KEEP" if is_valid else "✗ REJECT"
        print(f"  Tile {i+1:2d}: {stats['cloud_percent']:5.1f}% cloud  {status}")
        
        if is_valid:
            valid_tiles.append(i)
        else:
            rejected_tiles.append(i)
    
    print()
    print(f"Summary:")
    print(f"  Valid: {len(valid_tiles)}/{len(tiles)} ({len(valid_tiles)/len(tiles)*100:.0f}%)")
    print(f"  Rejected: {len(rejected_tiles)}/{len(tiles)}")
    print()


def demo_workflow():
    """Complete data processing workflow with cloud masking."""
    print("=" * 79)
    print("DEMO 4: Complete Workflow")
    print("=" * 79)
    print()
    
    print("Simulating Sentinel-2 tile processing workflow:")
    print("-" * 79)
    print()
    
    # Simulate tile metadata
    tile_id = "T10TEM_20240115T184751_B04_10m"
    print(f"Processing: {tile_id}")
    print()
    
    # Step 1: Load SCL
    print("Step 1: Load Scene Classification Layer (SCL)")
    np.random.seed(42)
    scl_data = np.random.choice(
        [4, 5, 6, 8, 9, 3],
        size=(10980, 10980),  # Full Sentinel-2 tile
        p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02]
    )
    print(f"  Loaded: {scl_data.shape} ({scl_data.nbytes / 1e6:.1f} MB)")
    print()
    
    # Step 2: Quick quality check
    print("Step 2: Quality assessment")
    is_valid, stats = filter_by_cloud_cover(
        scl_data,
        max_cloud_percent=40.0,
        max_shadow_percent=20.0,
        max_invalid_percent=5.0
    )
    
    print(f"  Cloud cover: {stats['cloud_percent']:.2f}%")
    print(f"  Shadow cover: {stats['shadow_percent']:.2f}%")
    print(f"  Clear area: {stats['clear_percent']:.2f}%")
    print(f"  Decision: {'PROCESS' if is_valid else 'SKIP'}")
    print()
    
    if not is_valid:
        print("✗ Tile rejected due to poor quality")
        return
    
    # Step 3: Parse masks
    print("Step 3: Generate quality masks")
    cloud_mask, shadow_mask, valid_mask = parse_scl(scl_data)
    print(f"  Cloud mask: {cloud_mask.shape}")
    print(f"  Shadow mask: {shadow_mask.shape}")
    print(f"  Valid mask: {valid_mask.shape}")
    print()
    
    # Step 4: Simulate band loading and masking
    print("Step 4: Apply masks to spectral bands")
    print("  Loading bands: B02, B03, B04, B08...")
    
    # In practice:
    # b02, _ = read_cog(f'{tile_path}/B02.tif')
    # b03, _ = read_cog(f'{tile_path}/B03.tif')
    # etc.
    
    # For demo, simulate
    b04 = np.random.randint(0, 10000, scl_data.shape, dtype=np.uint16)
    b08 = np.random.randint(0, 10000, scl_data.shape, dtype=np.uint16)
    
    # Apply cloud mask
    b04_masked = np.where(cloud_mask | shadow_mask, np.nan, b04.astype(float))
    b08_masked = np.where(cloud_mask | shadow_mask, np.nan, b08.astype(float))
    
    print(f"  B04 valid pixels: {(~np.isnan(b04_masked)).sum():,}")
    print(f"  B08 valid pixels: {(~np.isnan(b08_masked)).sum():,}")
    print()
    
    # Step 5: Calculate NDVI on clean pixels only
    print("Step 5: Calculate NDVI on clean pixels")
    ndvi = (b08_masked - b04_masked) / (b08_masked + b04_masked + 1e-8)
    valid_ndvi = ~np.isnan(ndvi)
    
    print(f"  NDVI range: [{np.nanmin(ndvi):.3f}, {np.nanmax(ndvi):.3f}]")
    print(f"  Valid NDVI pixels: {valid_ndvi.sum():,} ({valid_ndvi.mean()*100:.1f}%)")
    print()
    
    # Step 6: Summary
    print("Step 6: Processing complete")
    print(f"  Input quality: {stats['clear_percent']:.1f}% clear")
    print(f"  Output pixels: {valid_ndvi.sum():,}")
    print(f"  Status: ✓ SUCCESS")
    print()


if __name__ == "__main__":
    # Run all demos
    demo_basic_usage()
    demo_combined_mask()
    demo_batch_filtering()
    demo_workflow()
    
    print("=" * 79)
    print("ALL DEMOS COMPLETE")
    print("=" * 79)
    print()
    print("Next steps:")
    print("  1. Load real Sentinel-2 SCL data using axs_lib.geo.read_cog()")
    print("  2. Integrate quality filtering into your data pipeline")
    print("  3. Apply masks to spectral bands before analysis")
    print("  4. Track quality statistics for each processed tile")
