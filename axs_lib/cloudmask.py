"""
axs_lib/cloudmask.py - Cloud and Shadow Masking

Parses Sentinel-2 Scene Classification Layer (SCL) to create cloud and shadow
masks for data quality filtering.

Sentinel-2 SCL Classification:
    0  - No Data (missing data)
    1  - Saturated or defective pixel
    2  - Dark Area Pixels
    3  - Cloud Shadows
    4  - Vegetation
    5  - Not-vegetated
    6  - Water
    7  - Unclassified
    8  - Cloud Medium Probability
    9  - Cloud High Probability
    10 - Thin Cirrus
    11 - Snow / Ice

Features:
    - Parse SCL to binary masks
    - Calculate cloud/shadow coverage percentage
    - Filter tiles by quality threshold
    - Combine multiple quality criteria
    - Memory-efficient windowed processing

Usage:
    >>> from axs_lib.cloudmask import parse_scl, filter_by_cloud_cover
    >>> 
    >>> # Parse SCL to masks
    >>> cloud_mask, shadow_mask, valid_mask = parse_scl(scl_data)
    >>> 
    >>> # Check if tile is usable
    >>> is_valid, stats = filter_by_cloud_cover(
    ...     scl_data, 
    ...     max_cloud_percent=40.0
    ... )
    >>> 
    >>> if is_valid:
    ...     # Process tile
    ...     pass

References:
    - Sentinel-2 Level-2A Algorithm Theoretical Basis Document:
      https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# SCL Classification Constants
# ============================================================================

class SCLClass:
    """Sentinel-2 Scene Classification Layer class values."""
    NO_DATA = 0
    SATURATED_DEFECTIVE = 1
    DARK_AREA = 2
    CLOUD_SHADOW = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROB = 8
    CLOUD_HIGH_PROB = 9
    THIN_CIRRUS = 10
    SNOW_ICE = 11


# Class groupings for convenience
CLOUD_CLASSES = [
    SCLClass.CLOUD_MEDIUM_PROB,
    SCLClass.CLOUD_HIGH_PROB,
    SCLClass.THIN_CIRRUS
]

SHADOW_CLASSES = [
    SCLClass.CLOUD_SHADOW
]

INVALID_CLASSES = [
    SCLClass.NO_DATA,
    SCLClass.SATURATED_DEFECTIVE
]

CLEAR_CLASSES = [
    SCLClass.VEGETATION,
    SCLClass.NOT_VEGETATED,
    SCLClass.WATER,
    SCLClass.DARK_AREA  # Often clear but dark
]


# ============================================================================
# Mask Parsing Functions
# ============================================================================

def parse_scl(
    scl_data: np.ndarray,
    include_cirrus: bool = True,
    include_snow: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse Sentinel-2 SCL to binary masks.
    
    Args:
        scl_data: SCL array with classification values (0-11)
        include_cirrus: Include thin cirrus as cloud
        include_snow: Include snow/ice as cloud (for most use cases, exclude)
        
    Returns:
        Tuple of (cloud_mask, shadow_mask, valid_mask) as boolean arrays:
        - cloud_mask: True where clouds detected
        - shadow_mask: True where shadows detected
        - valid_mask: True where data is valid (not nodata/saturated)
        
    Example:
        >>> scl_data = np.array([[0, 4, 8], [3, 6, 9], [4, 5, 10]])
        >>> cloud_mask, shadow_mask, valid_mask = parse_scl(scl_data)
        >>> print(cloud_mask)
        [[False False  True]
         [False False  True]
         [False False  True]]
    """
    # Cloud mask
    cloud_classes = [SCLClass.CLOUD_MEDIUM_PROB, SCLClass.CLOUD_HIGH_PROB]
    if include_cirrus:
        cloud_classes.append(SCLClass.THIN_CIRRUS)
    if include_snow:
        cloud_classes.append(SCLClass.SNOW_ICE)
    
    cloud_mask = np.isin(scl_data, cloud_classes)
    
    # Shadow mask
    shadow_mask = np.isin(scl_data, SHADOW_CLASSES)
    
    # Valid data mask (not nodata or saturated)
    valid_mask = ~np.isin(scl_data, INVALID_CLASSES)
    
    return cloud_mask, shadow_mask, valid_mask


def calculate_coverage(
    scl_data: np.ndarray,
    include_cirrus: bool = True,
    include_snow: bool = False
) -> Dict[str, float]:
    """
    Calculate coverage statistics from SCL.
    
    Args:
        scl_data: SCL array with classification values
        include_cirrus: Include thin cirrus as cloud
        include_snow: Include snow/ice as cloud
        
    Returns:
        Dictionary with coverage percentages:
        {
            'cloud_percent': float,
            'shadow_percent': float,
            'invalid_percent': float,
            'clear_percent': float,
            'total_pixels': int
        }
        
    Example:
        >>> scl_data = np.array([[8, 8, 4], [3, 6, 4], [4, 4, 4]])
        >>> stats = calculate_coverage(scl_data)
        >>> print(f"Cloud cover: {stats['cloud_percent']:.1f}%")
        Cloud cover: 22.2%
    """
    total_pixels = scl_data.size
    
    # Parse masks
    cloud_mask, shadow_mask, valid_mask = parse_scl(
        scl_data, 
        include_cirrus=include_cirrus,
        include_snow=include_snow
    )
    
    # Count pixels
    cloud_pixels = np.sum(cloud_mask)
    shadow_pixels = np.sum(shadow_mask)
    invalid_pixels = np.sum(~valid_mask)
    
    # Calculate percentages
    cloud_percent = (cloud_pixels / total_pixels) * 100.0
    shadow_percent = (shadow_pixels / total_pixels) * 100.0
    invalid_percent = (invalid_pixels / total_pixels) * 100.0
    
    # Clear = valid and not cloud and not shadow
    clear_mask = valid_mask & ~cloud_mask & ~shadow_mask
    clear_pixels = np.sum(clear_mask)
    clear_percent = (clear_pixels / total_pixels) * 100.0
    
    return {
        'cloud_percent': cloud_percent,
        'shadow_percent': shadow_percent,
        'invalid_percent': invalid_percent,
        'clear_percent': clear_percent,
        'total_pixels': total_pixels,
        'cloud_pixels': int(cloud_pixels),
        'shadow_pixels': int(shadow_pixels),
        'invalid_pixels': int(invalid_pixels),
        'clear_pixels': int(clear_pixels),
    }


# ============================================================================
# Quality Filtering
# ============================================================================

def filter_by_cloud_cover(
    scl_data: np.ndarray,
    max_cloud_percent: float = 40.0,
    max_shadow_percent: Optional[float] = None,
    max_invalid_percent: float = 10.0,
    include_cirrus: bool = True,
    verbose: bool = False
) -> Tuple[bool, Dict[str, float]]:
    """
    Check if tile passes cloud cover quality threshold.
    
    Args:
        scl_data: SCL array with classification values
        max_cloud_percent: Maximum cloud coverage allowed (%)
        max_shadow_percent: Maximum shadow coverage allowed (%, None = no limit)
        max_invalid_percent: Maximum invalid data allowed (%)
        include_cirrus: Include thin cirrus as cloud
        verbose: Print filtering decision
        
    Returns:
        Tuple of (is_valid, stats):
        - is_valid: True if tile passes all quality thresholds
        - stats: Coverage statistics dictionary
        
    Example:
        >>> scl_data = np.random.randint(0, 12, (1000, 1000))
        >>> is_valid, stats = filter_by_cloud_cover(scl_data, max_cloud_percent=40.0)
        >>> 
        >>> if is_valid:
        ...     print(f"✓ Tile OK: {stats['cloud_percent']:.1f}% cloud")
        ... else:
        ...     print(f"✗ Tile rejected: {stats['cloud_percent']:.1f}% cloud")
    """
    # Calculate coverage
    stats = calculate_coverage(scl_data, include_cirrus=include_cirrus)
    
    # Check thresholds
    is_valid = True
    rejection_reasons = []
    
    if stats['cloud_percent'] > max_cloud_percent:
        is_valid = False
        rejection_reasons.append(
            f"cloud {stats['cloud_percent']:.1f}% > {max_cloud_percent}%"
        )
    
    if max_shadow_percent is not None and stats['shadow_percent'] > max_shadow_percent:
        is_valid = False
        rejection_reasons.append(
            f"shadow {stats['shadow_percent']:.1f}% > {max_shadow_percent}%"
        )
    
    if stats['invalid_percent'] > max_invalid_percent:
        is_valid = False
        rejection_reasons.append(
            f"invalid {stats['invalid_percent']:.1f}% > {max_invalid_percent}%"
        )
    
    # Verbose output
    if verbose:
        if is_valid:
            print(f"✓ Tile PASSED quality check")
            print(f"  Cloud: {stats['cloud_percent']:.1f}%")
            print(f"  Shadow: {stats['shadow_percent']:.1f}%")
            print(f"  Clear: {stats['clear_percent']:.1f}%")
        else:
            print(f"✗ Tile REJECTED: {', '.join(rejection_reasons)}")
    
    return is_valid, stats


def create_combined_mask(
    scl_data: np.ndarray,
    mask_clouds: bool = True,
    mask_shadows: bool = True,
    mask_invalid: bool = True,
    include_cirrus: bool = True
) -> np.ndarray:
    """
    Create combined binary mask for all bad pixels.
    
    Args:
        scl_data: SCL array with classification values
        mask_clouds: Include clouds in mask
        mask_shadows: Include shadows in mask
        mask_invalid: Include invalid pixels in mask
        include_cirrus: Include thin cirrus as cloud
        
    Returns:
        Boolean mask where True = bad pixel (should be masked out)
        
    Example:
        >>> scl_data = np.array([[8, 4, 3], [0, 6, 4]])
        >>> bad_mask = create_combined_mask(scl_data)
        >>> clean_data = np.where(bad_mask, np.nan, original_data)
    """
    cloud_mask, shadow_mask, valid_mask = parse_scl(
        scl_data,
        include_cirrus=include_cirrus
    )
    
    # Start with all False (no masking)
    combined_mask = np.zeros_like(scl_data, dtype=bool)
    
    if mask_clouds:
        combined_mask |= cloud_mask
    
    if mask_shadows:
        combined_mask |= shadow_mask
    
    if mask_invalid:
        combined_mask |= ~valid_mask
    
    return combined_mask


# ============================================================================
# Batch Processing
# ============================================================================

def batch_filter_tiles(
    scl_files: list,
    max_cloud_percent: float = 40.0,
    verbose: bool = True
) -> Tuple[list, list, Dict]:
    """
    Filter multiple tiles by cloud cover.
    
    Args:
        scl_files: List of SCL file paths or arrays
        max_cloud_percent: Maximum cloud coverage threshold
        verbose: Print progress
        
    Returns:
        Tuple of (valid_tiles, rejected_tiles, summary):
        - valid_tiles: List of tiles that passed
        - rejected_tiles: List of tiles that failed with reasons
        - summary: Overall statistics
    """
    valid_tiles = []
    rejected_tiles = []
    
    if verbose:
        print(f"Filtering {len(scl_files)} tiles (max cloud: {max_cloud_percent}%)")
    
    for i, scl_file in enumerate(scl_files):
        # Load if path, otherwise assume numpy array
        if isinstance(scl_file, (str, Path)):
            try:
                from axs_lib.geo import read_cog
                scl_data, _ = read_cog(scl_file)
            except Exception as e:
                rejected_tiles.append({
                    'file': scl_file,
                    'reason': f'Failed to load: {e}'
                })
                continue
        else:
            scl_data = scl_file
        
        # Filter
        is_valid, stats = filter_by_cloud_cover(
            scl_data,
            max_cloud_percent=max_cloud_percent,
            verbose=False
        )
        
        if is_valid:
            valid_tiles.append(scl_file)
        else:
            rejected_tiles.append({
                'file': scl_file,
                'stats': stats
            })
    
    # Summary
    summary = {
        'total': len(scl_files),
        'valid': len(valid_tiles),
        'rejected': len(rejected_tiles),
        'valid_percent': (len(valid_tiles) / len(scl_files)) * 100.0
    }
    
    if verbose:
        print(f"  Valid: {summary['valid']}/{summary['total']} "
              f"({summary['valid_percent']:.1f}%)")
    
    return valid_tiles, rejected_tiles, summary


# ============================================================================
# Unit Tests
# ============================================================================

def run_unit_tests():
    """
    Run unit tests for cloud masking functions.
    """
    print("=" * 79)
    print("CLOUD MASKING UNIT TESTS")
    print("=" * 79)
    print()
    
    # Test 1: Parse SCL to masks
    print("TEST 1: Parse SCL to masks")
    print("-" * 79)
    
    # Create synthetic SCL with known distribution
    scl_test = np.array([
        [0, 4, 8],   # No data, vegetation, cloud medium
        [3, 6, 9],   # Shadow, water, cloud high
        [4, 5, 10]   # Vegetation, not-vegetated, thin cirrus
    ])
    
    cloud_mask, shadow_mask, valid_mask = parse_scl(scl_test)
    
    # Expected: clouds at [0,2], [1,2], [2,2]
    expected_clouds = np.array([
        [False, False, True],
        [False, False, True],
        [False, False, True]
    ])
    
    # Expected: shadow at [1,0]
    expected_shadows = np.array([
        [False, False, False],
        [True, False, False],
        [False, False, False]
    ])
    
    # Expected: valid everywhere except [0,0] (no data)
    expected_valid = np.array([
        [False, True, True],
        [True, True, True],
        [True, True, True]
    ])
    
    assert np.array_equal(cloud_mask, expected_clouds), "Cloud mask mismatch"
    assert np.array_equal(shadow_mask, expected_shadows), "Shadow mask mismatch"
    assert np.array_equal(valid_mask, expected_valid), "Valid mask mismatch"
    
    print("✓ SCL parsing correct")
    print(f"  Clouds: {np.sum(cloud_mask)} pixels")
    print(f"  Shadows: {np.sum(shadow_mask)} pixels")
    print(f"  Valid: {np.sum(valid_mask)} pixels")
    print()
    
    # Test 2: Calculate coverage
    print("TEST 2: Calculate coverage statistics")
    print("-" * 79)
    
    stats = calculate_coverage(scl_test)
    
    expected_cloud_percent = (3 / 9) * 100  # 33.33%
    expected_shadow_percent = (1 / 9) * 100  # 11.11%
    
    assert abs(stats['cloud_percent'] - expected_cloud_percent) < 0.1, \
        f"Cloud percent mismatch: {stats['cloud_percent']}"
    assert abs(stats['shadow_percent'] - expected_shadow_percent) < 0.1, \
        f"Shadow percent mismatch: {stats['shadow_percent']}"
    
    print("✓ Coverage calculation correct")
    print(f"  Cloud: {stats['cloud_percent']:.1f}%")
    print(f"  Shadow: {stats['shadow_percent']:.1f}%")
    print(f"  Clear: {stats['clear_percent']:.1f}%")
    print()
    
    # Test 3: Filter by cloud cover (should reject)
    print("TEST 3: Filter by cloud cover (rejection case)")
    print("-" * 79)
    
    # 33% cloud should be rejected with 20% threshold
    is_valid, stats = filter_by_cloud_cover(scl_test, max_cloud_percent=20.0)
    
    assert not is_valid, "Should reject tile with 33% cloud (threshold 20%)"
    print(f"✓ Correctly rejected tile with {stats['cloud_percent']:.1f}% cloud")
    print()
    
    # Test 4: Filter by cloud cover (should accept)
    print("TEST 4: Filter by cloud cover (acceptance case)")
    print("-" * 79)
    
    # 33% cloud should be accepted with 40% threshold
    # Need to adjust invalid threshold since test data has 11.1% invalid
    is_valid, stats = filter_by_cloud_cover(
        scl_test, 
        max_cloud_percent=40.0,
        max_invalid_percent=15.0  # Allow for the no-data pixel
    )
    
    assert is_valid, "Should accept tile with 33% cloud (threshold 40%)"
    print(f"✓ Correctly accepted tile with {stats['cloud_percent']:.1f}% cloud")
    print()
    
    # Test 5: Create combined mask
    print("TEST 5: Create combined mask")
    print("-" * 79)
    
    bad_mask = create_combined_mask(
        scl_test,
        mask_clouds=True,
        mask_shadows=True,
        mask_invalid=True
    )
    
    # Should mask: no data [0,0], shadow [1,0], clouds [0,2],[1,2],[2,2]
    expected_bad = np.array([
        [True, False, True],
        [True, False, True],
        [False, False, True]
    ])
    
    assert np.array_equal(bad_mask, expected_bad), "Combined mask mismatch"
    print(f"✓ Combined mask correct ({np.sum(bad_mask)}/9 pixels masked)")
    print()
    
    # Test 6: Large tile simulation
    print("TEST 6: Large tile simulation (realistic)")
    print("-" * 79)
    
    # Create realistic tile: 1000x1000 pixels
    np.random.seed(42)
    large_scl = np.random.choice(
        [4, 5, 6, 8, 9],  # Mostly clear with some clouds
        size=(1000, 1000),
        p=[0.4, 0.3, 0.2, 0.05, 0.05]  # 10% cloud total
    )
    
    is_valid, stats = filter_by_cloud_cover(large_scl, max_cloud_percent=40.0)
    
    print(f"  Cloud cover: {stats['cloud_percent']:.2f}%")
    print(f"  Clear cover: {stats['clear_percent']:.2f}%")
    print(f"  Status: {'✓ ACCEPTED' if is_valid else '✗ REJECTED'}")
    
    assert is_valid, "Realistic tile should be accepted"
    assert stats['cloud_percent'] < 15.0, "Cloud percent should be ~10%"
    print()
    
    # Summary
    print("=" * 79)
    print("ALL TESTS PASSED ✓")
    print("=" * 79)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    run_unit_tests()
    
    print()
    print("Usage Example:")
    print("-" * 79)
    print("from axs_lib.cloudmask import parse_scl, filter_by_cloud_cover")
    print()
    print("# Read SCL band")
    print("scl_data, profile = read_cog('data/raw/.../SCL.tif')")
    print()
    print("# Check quality")
    print("is_valid, stats = filter_by_cloud_cover(scl_data, max_cloud_percent=40.0)")
    print()
    print("if is_valid:")
    print("    # Process tile")
    print("    cloud_mask, shadow_mask, valid_mask = parse_scl(scl_data)")
    print("    # Apply mask to bands")
    print("    masked_data = np.where(cloud_mask, np.nan, band_data)")
    print("else:")
    print("    print(f'Skipped: {stats[\"cloud_percent\"]:.1f}% cloud')")
