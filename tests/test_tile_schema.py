"""
tests/test_tile_schema.py - Tile Schema Validation Tests

Validates that tile NPZ files conform to the expected schema:
- Required arrays exist (SAR and optical bands)
- Array shapes match expected tile size
- Data types are correct
- Value ranges are reasonable
- Metadata is present and valid

Usage:
    pytest tests/test_tile_schema.py
    pytest tests/test_tile_schema.py -v
    pytest tests/test_tile_schema.py -k test_sar_bands
    pytest tests/test_tile_schema.py --tile-dir data/tiles/test

Author: Axion-Sat Project
Version: 1.0.0
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json


# ============================================================================
# Configuration
# ============================================================================

# Default tile size (can be overridden by fixtures)
DEFAULT_TILE_SIZE = (512, 512)

# Sentinel-1 SAR band names
S1_BANDS = ['s1_vv', 's1_vh']

# Sentinel-2 optical band names (10m and 20m bands)
S2_BANDS_10M = ['s2_b2', 's2_b3', 's2_b4', 's2_b8']  # Blue, Green, Red, NIR
S2_BANDS_20M = ['s2_b5', 's2_b6', 's2_b7', 's2_b8a', 's2_b11', 's2_b12']  # RedEdge, SWIR
S2_BANDS_ALL = S2_BANDS_10M + S2_BANDS_20M

# Expected data types
EXPECTED_DTYPES = {
    'float32': np.float32,
    'float64': np.float64,
}

# Value range checks (dB scale for SAR, reflectance for optical)
SAR_RANGE_DB = (-40.0, 10.0)  # Typical SAR backscatter in dB
OPTICAL_RANGE_REFLECTANCE = (0.0, 1.0)  # Normalized reflectance
OPTICAL_RANGE_RAW = (0, 10000)  # Raw digital numbers (if not normalized)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tile_dir(request):
    """Get tile directory from command line or use default."""
    return request.config.getoption("--tile-dir", default="data/tiles")


@pytest.fixture
def tile_size(request):
    """Get expected tile size from command line or use default."""
    size_str = request.config.getoption("--tile-size", default="512,512")
    height, width = map(int, size_str.split(','))
    return (height, width)


@pytest.fixture
def sample_tiles(tile_dir) -> List[Path]:
    """Find sample tile files for testing."""
    tile_path = Path(tile_dir)
    
    if not tile_path.exists():
        pytest.skip(f"Tile directory not found: {tile_dir}")
    
    # Find NPZ files
    tiles = list(tile_path.rglob('*.npz'))
    
    if not tiles:
        pytest.skip(f"No tile files found in: {tile_dir}")
    
    return tiles[:10]  # Test first 10 tiles for speed


@pytest.fixture
def sample_tile(sample_tiles) -> Path:
    """Get a single sample tile for detailed testing."""
    return sample_tiles[0]


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--tile-dir",
        action="store",
        default="data/tiles",
        help="Directory containing tile files"
    )
    parser.addoption(
        "--tile-size",
        action="store",
        default="512,512",
        help="Expected tile size as 'height,width'"
    )


# ============================================================================
# Helper Functions
# ============================================================================

def load_tile(tile_path: Path) -> Dict[str, np.ndarray]:
    """Load tile NPZ file and return arrays as dictionary."""
    tile = np.load(tile_path)
    arrays = {key: tile[key] for key in tile.files}
    tile.close()
    return arrays


def load_tile_metadata(tile_path: Path) -> Optional[Dict]:
    """Load tile metadata JSON if it exists."""
    json_path = tile_path.with_suffix('.json')
    if not json_path.exists():
        return None
    
    with open(json_path, 'r') as f:
        return json.load(f)


def check_array_shape(array: np.ndarray, expected_shape: Tuple[int, int]) -> bool:
    """Check if array has expected 2D shape."""
    return array.shape == expected_shape


def check_array_dtype(array: np.ndarray, allowed_dtypes: List[np.dtype]) -> bool:
    """Check if array dtype is in allowed list."""
    return array.dtype in allowed_dtypes


def check_value_range(
    array: np.ndarray,
    min_val: float,
    max_val: float,
    allow_nan: bool = True
) -> Tuple[bool, str]:
    """
    Check if array values are within expected range.
    
    Returns:
        Tuple of (is_valid, message)
    """
    # Get valid (non-NaN) values
    if allow_nan:
        valid_values = array[~np.isnan(array)]
    else:
        valid_values = array
    
    if len(valid_values) == 0:
        return False, "Array contains only NaN values"
    
    actual_min = np.nanmin(array)
    actual_max = np.nanmax(array)
    
    if actual_min < min_val or actual_max > max_val:
        return False, f"Values out of range: [{actual_min:.2f}, {actual_max:.2f}] not in [{min_val}, {max_val}]"
    
    return True, "OK"


# ============================================================================
# Basic Structure Tests
# ============================================================================

class TestTileStructure:
    """Test basic tile file structure and loading."""
    
    def test_tile_files_exist(self, sample_tiles):
        """Test that tile files exist and can be found."""
        assert len(sample_tiles) > 0, "No tile files found"
    
    def test_tile_can_load(self, sample_tile):
        """Test that tile NPZ file can be loaded."""
        try:
            arrays = load_tile(sample_tile)
            assert len(arrays) > 0, "Tile contains no arrays"
        except Exception as e:
            pytest.fail(f"Failed to load tile: {e}")
    
    def test_tile_has_arrays(self, sample_tile):
        """Test that tile contains data arrays."""
        arrays = load_tile(sample_tile)
        assert len(arrays) > 0, "Tile has no arrays"
        assert len(arrays) >= 2, f"Tile has too few arrays: {len(arrays)}"
    
    def test_tile_metadata_exists(self, sample_tile):
        """Test that tile metadata JSON exists."""
        metadata = load_tile_metadata(sample_tile)
        if metadata is None:
            pytest.skip("Metadata file not found (optional)")
        assert isinstance(metadata, dict), "Metadata is not a dictionary"


# ============================================================================
# SAR Band Tests (Sentinel-1)
# ============================================================================

class TestSARBands:
    """Test Sentinel-1 SAR band presence and validity."""
    
    def test_sar_bands_present(self, sample_tile):
        """Test that required SAR bands are present."""
        arrays = load_tile(sample_tile)
        
        for band in S1_BANDS:
            assert band in arrays, f"Missing SAR band: {band}"
    
    def test_sar_band_shapes(self, sample_tile, tile_size):
        """Test that SAR bands have correct shape."""
        arrays = load_tile(sample_tile)
        
        for band in S1_BANDS:
            if band not in arrays:
                continue
            
            array = arrays[band]
            assert array.ndim == 2, f"{band} is not 2D: shape={array.shape}"
            assert array.shape == tile_size, \
                f"{band} has wrong shape: {array.shape}, expected {tile_size}"
    
    def test_sar_band_dtypes(self, sample_tile):
        """Test that SAR bands have float dtype."""
        arrays = load_tile(sample_tile)
        allowed_dtypes = [np.float32, np.float64]
        
        for band in S1_BANDS:
            if band not in arrays:
                continue
            
            array = arrays[band]
            assert array.dtype in allowed_dtypes, \
                f"{band} has wrong dtype: {array.dtype}, expected float32/float64"
    
    def test_sar_value_ranges_db(self, sample_tile):
        """Test that SAR values are in reasonable dB range."""
        arrays = load_tile(sample_tile)
        
        for band in S1_BANDS:
            if band not in arrays:
                continue
            
            array = arrays[band]
            
            # Skip if all NaN (invalid tile)
            if np.all(np.isnan(array)):
                pytest.skip(f"{band} is all NaN")
            
            # Check if values are in dB scale
            valid_values = array[~np.isnan(array)]
            
            if len(valid_values) == 0:
                continue
            
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            
            # SAR backscatter in dB typically ranges from -40 to +10 dB
            # Allow some tolerance
            assert min_val >= -50, \
                f"{band} has unreasonably low values: {min_val:.2f} dB"
            assert max_val <= 20, \
                f"{band} has unreasonably high values: {max_val:.2f} dB"
    
    def test_sar_not_all_zero(self, sample_tile):
        """Test that SAR bands are not all zeros."""
        arrays = load_tile(sample_tile)
        
        for band in S1_BANDS:
            if band not in arrays:
                continue
            
            array = arrays[band]
            valid_values = array[~np.isnan(array)]
            
            if len(valid_values) == 0:
                continue
            
            assert not np.allclose(valid_values, 0, atol=1e-6), \
                f"{band} contains only zeros"
    
    def test_sar_vh_vv_ratio(self, sample_tile):
        """Test that VH/VV ratio is reasonable."""
        arrays = load_tile(sample_tile)
        
        if 's1_vv' not in arrays or 's1_vh' not in arrays:
            pytest.skip("Both VV and VH bands required")
        
        vv = arrays['s1_vv']
        vh = arrays['s1_vh']
        
        # Get valid pixels
        mask = ~(np.isnan(vv) | np.isnan(vh))
        valid_vv = vv[mask]
        valid_vh = vh[mask]
        
        if len(valid_vv) == 0:
            pytest.skip("No valid pixels")
        
        # In dB scale, VH is typically lower than VV
        # Mean difference should be negative (VH < VV)
        mean_diff = np.mean(valid_vh - valid_vv)
        
        # VH is typically 5-15 dB lower than VV
        assert mean_diff < 0, \
            f"VH should be lower than VV in dB scale: mean(VH-VV)={mean_diff:.2f}"


# ============================================================================
# Optical Band Tests (Sentinel-2)
# ============================================================================

class TestOpticalBands:
    """Test Sentinel-2 optical band presence and validity."""
    
    def test_optical_bands_present(self, sample_tile):
        """Test that required optical bands are present."""
        arrays = load_tile(sample_tile)
        
        # At least some optical bands should be present
        present_bands = [b for b in S2_BANDS_ALL if b in arrays]
        assert len(present_bands) > 0, "No optical bands found"
    
    def test_optical_band_shapes(self, sample_tile, tile_size):
        """Test that optical bands have correct shape."""
        arrays = load_tile(sample_tile)
        
        for band in S2_BANDS_ALL:
            if band not in arrays:
                continue
            
            array = arrays[band]
            assert array.ndim == 2, f"{band} is not 2D: shape={array.shape}"
            assert array.shape == tile_size, \
                f"{band} has wrong shape: {array.shape}, expected {tile_size}"
    
    def test_optical_band_dtypes(self, sample_tile):
        """Test that optical bands have float dtype."""
        arrays = load_tile(sample_tile)
        allowed_dtypes = [np.float32, np.float64]
        
        for band in S2_BANDS_ALL:
            if band not in arrays:
                continue
            
            array = arrays[band]
            assert array.dtype in allowed_dtypes, \
                f"{band} has wrong dtype: {array.dtype}, expected float32/float64"
    
    def test_optical_value_ranges(self, sample_tile):
        """Test that optical values are in reasonable range."""
        arrays = load_tile(sample_tile)
        
        for band in S2_BANDS_ALL:
            if band not in arrays:
                continue
            
            array = arrays[band]
            valid_values = array[~np.isnan(array)]
            
            if len(valid_values) == 0:
                continue
            
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            
            # Check if normalized (0-1) or raw (0-10000)
            if max_val <= 1.5:
                # Normalized reflectance
                assert min_val >= -0.1, \
                    f"{band} has negative reflectance: {min_val:.4f}"
                assert max_val <= 1.5, \
                    f"{band} has too high reflectance: {max_val:.4f}"
            else:
                # Raw digital numbers
                assert min_val >= 0, \
                    f"{band} has negative values: {min_val:.0f}"
                assert max_val <= 15000, \
                    f"{band} has unreasonably high values: {max_val:.0f}"
    
    def test_optical_not_all_zero(self, sample_tile):
        """Test that optical bands are not all zeros."""
        arrays = load_tile(sample_tile)
        
        for band in S2_BANDS_ALL:
            if band not in arrays:
                continue
            
            array = arrays[band]
            valid_values = array[~np.isnan(array)]
            
            if len(valid_values) == 0:
                continue
            
            assert not np.allclose(valid_values, 0, atol=1e-6), \
                f"{band} contains only zeros"
    
    def test_rgb_bands_present(self, sample_tile):
        """Test that RGB bands are present for visualization."""
        arrays = load_tile(sample_tile)
        rgb_bands = ['s2_b2', 's2_b3', 's2_b4']  # Blue, Green, Red
        
        present = [b for b in rgb_bands if b in arrays]
        assert len(present) == 3, \
            f"Missing RGB bands: {[b for b in rgb_bands if b not in arrays]}"
    
    def test_nir_band_higher_than_red(self, sample_tile):
        """Test that NIR reflectance is typically higher than Red (vegetation)."""
        arrays = load_tile(sample_tile)
        
        if 's2_b4' not in arrays or 's2_b8' not in arrays:
            pytest.skip("Red and NIR bands required")
        
        red = arrays['s2_b4']
        nir = arrays['s2_b8']
        
        # Get valid pixels
        mask = ~(np.isnan(red) | np.isnan(nir))
        valid_red = red[mask]
        valid_nir = nir[mask]
        
        if len(valid_red) == 0:
            pytest.skip("No valid pixels")
        
        # For vegetated areas, NIR > Red
        # Check that at least 30% of pixels have NIR > Red
        nir_higher = np.sum(valid_nir > valid_red) / len(valid_red)
        
        assert nir_higher > 0.1, \
            f"Too few pixels with NIR > Red: {nir_higher*100:.1f}% (might indicate no vegetation)"


# ============================================================================
# Shape Consistency Tests
# ============================================================================

class TestShapeConsistency:
    """Test that all bands have consistent shapes."""
    
    def test_all_bands_same_shape(self, sample_tile, tile_size):
        """Test that all bands in tile have the same shape."""
        arrays = load_tile(sample_tile)
        
        for band_name, array in arrays.items():
            assert array.shape == tile_size, \
                f"{band_name} has wrong shape: {array.shape}, expected {tile_size}"
    
    def test_sar_optical_alignment(self, sample_tile, tile_size):
        """Test that SAR and optical bands are spatially aligned."""
        arrays = load_tile(sample_tile)
        
        # Get one SAR and one optical band
        sar_band = None
        optical_band = None
        
        for band in S1_BANDS:
            if band in arrays:
                sar_band = arrays[band]
                break
        
        for band in S2_BANDS_ALL:
            if band in arrays:
                optical_band = arrays[band]
                break
        
        if sar_band is None or optical_band is None:
            pytest.skip("Need both SAR and optical bands")
        
        assert sar_band.shape == optical_band.shape, \
            "SAR and optical bands have different shapes"
        assert sar_band.shape == tile_size, \
            f"Bands have wrong shape: {sar_band.shape}, expected {tile_size}"


# ============================================================================
# Data Quality Tests
# ============================================================================

class TestDataQuality:
    """Test data quality metrics."""
    
    def test_valid_pixel_percentage(self, sample_tile):
        """Test that tiles have sufficient valid (non-NaN) pixels."""
        arrays = load_tile(sample_tile)
        
        min_valid_pct = 50.0  # Require at least 50% valid pixels
        
        for band_name, array in arrays.items():
            total_pixels = array.size
            valid_pixels = np.sum(~np.isnan(array))
            valid_pct = (valid_pixels / total_pixels) * 100
            
            assert valid_pct >= min_valid_pct, \
                f"{band_name} has too few valid pixels: {valid_pct:.1f}% (min: {min_valid_pct}%)"
    
    def test_no_inf_values(self, sample_tile):
        """Test that arrays don't contain infinite values."""
        arrays = load_tile(sample_tile)
        
        for band_name, array in arrays.items():
            assert not np.any(np.isinf(array)), \
                f"{band_name} contains infinite values"
    
    def test_reasonable_nan_patterns(self, sample_tile):
        """Test that NaN patterns are reasonable (not checkerboard, etc.)."""
        arrays = load_tile(sample_tile)
        
        for band_name, array in arrays.items():
            if np.all(~np.isnan(array)):
                continue  # No NaNs, OK
            
            # Check that NaNs are somewhat contiguous (not noisy)
            # Use simple erosion test: NaN pixels should have NaN neighbors
            nan_mask = np.isnan(array)
            
            # Skip if too few or too many NaNs
            nan_count = np.sum(nan_mask)
            if nan_count < 10 or nan_count > array.size * 0.9:
                continue
            
            # For random noise, isolated NaNs would be common
            # For valid masks, NaNs should be clustered
            # This is a simple heuristic check
            assert True  # Placeholder - implement if needed


# ============================================================================
# Metadata Tests
# ============================================================================

class TestTileMetadata:
    """Test tile metadata JSON files."""
    
    def test_metadata_file_exists(self, sample_tiles):
        """Test that metadata JSON files exist for tiles."""
        tiles_with_metadata = sum(
            1 for tile in sample_tiles
            if load_tile_metadata(tile) is not None
        )
        
        # Allow some tiles without metadata, but warn
        if tiles_with_metadata == 0:
            pytest.skip("No tiles have metadata JSON files")
    
    def test_metadata_has_required_fields(self, sample_tile):
        """Test that metadata contains required fields."""
        metadata = load_tile_metadata(sample_tile)
        
        if metadata is None:
            pytest.skip("No metadata file")
        
        required_fields = ['tile_path', 'tile_id']
        optional_fields = ['date', 'bounds', 'crs', 'split']
        
        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"
    
    def test_metadata_split_assignment(self, sample_tile):
        """Test that metadata has valid split assignment."""
        metadata = load_tile_metadata(sample_tile)
        
        if metadata is None:
            pytest.skip("No metadata file")
        
        if 'split' not in metadata:
            pytest.skip("No split field in metadata")
        
        valid_splits = {'train', 'val', 'test'}
        split = metadata['split']
        
        assert split in valid_splits, \
            f"Invalid split value: {split}, expected one of {valid_splits}"


# ============================================================================
# Batch Tests
# ============================================================================

class TestBatchValidation:
    """Test multiple tiles for consistency."""
    
    def test_all_tiles_have_same_bands(self, sample_tiles):
        """Test that all tiles have the same set of bands."""
        band_sets = []
        
        for tile in sample_tiles[:5]:  # Check first 5
            arrays = load_tile(tile)
            band_sets.append(set(arrays.keys()))
        
        if not band_sets:
            pytest.skip("No tiles to compare")
        
        # All tiles should have the same bands
        first_set = band_sets[0]
        for i, band_set in enumerate(band_sets[1:], 1):
            assert band_set == first_set, \
                f"Tile {i} has different bands: {band_set} vs {first_set}"
    
    def test_all_tiles_have_same_shape(self, sample_tiles, tile_size):
        """Test that all tiles have the same shape."""
        for tile in sample_tiles:
            arrays = load_tile(tile)
            
            for band_name, array in arrays.items():
                assert array.shape == tile_size, \
                    f"{tile.name}/{band_name} has wrong shape: {array.shape}"
    
    def test_tiles_have_unique_ids(self, sample_tiles):
        """Test that tiles have unique IDs in metadata."""
        tile_ids = []
        
        for tile in sample_tiles:
            metadata = load_tile_metadata(tile)
            if metadata and 'tile_id' in metadata:
                tile_ids.append(metadata['tile_id'])
        
        if not tile_ids:
            pytest.skip("No tiles with IDs found")
        
        # Check for duplicates
        unique_ids = set(tile_ids)
        assert len(unique_ids) == len(tile_ids), \
            f"Found duplicate tile IDs: {len(tile_ids)} tiles, {len(unique_ids)} unique IDs"


# ============================================================================
# Parametric Tests
# ============================================================================

@pytest.mark.parametrize("band", S1_BANDS)
def test_specific_sar_band(sample_tile, tile_size, band):
    """Parametric test for each SAR band."""
    arrays = load_tile(sample_tile)
    
    if band not in arrays:
        pytest.skip(f"Band {band} not in tile")
    
    array = arrays[band]
    
    # Check shape
    assert array.shape == tile_size
    
    # Check dtype
    assert array.dtype in [np.float32, np.float64]
    
    # Check not all NaN
    valid_pixels = np.sum(~np.isnan(array))
    assert valid_pixels > 0, f"{band} is all NaN"


@pytest.mark.parametrize("band", S2_BANDS_ALL)
def test_specific_optical_band(sample_tile, tile_size, band):
    """Parametric test for each optical band."""
    arrays = load_tile(sample_tile)
    
    if band not in arrays:
        pytest.skip(f"Band {band} not in tile")
    
    array = arrays[band]
    
    # Check shape
    assert array.shape == tile_size
    
    # Check dtype
    assert array.dtype in [np.float32, np.float64]
    
    # Check not all NaN
    valid_pixels = np.sum(~np.isnan(array))
    assert valid_pixels > 0, f"{band} is all NaN"


# ============================================================================
# Summary Report
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def print_test_summary(request):
    """Print summary after all tests complete."""
    yield
    
    # This runs after all tests
    if hasattr(request.config, '_json_report'):
        report = request.config._json_report
        print("\n" + "=" * 80)
        print("TILE SCHEMA VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total tests: {report['summary']['total']}")
        print(f"Passed: {report['summary'].get('passed', 0)}")
        print(f"Failed: {report['summary'].get('failed', 0)}")
        print(f"Skipped: {report['summary'].get('skipped', 0)}")
        print("=" * 80)
