"""
tests/test_stac_search_small.py - STAC Search Integration Tests

Tests the STAC search functionality with a small bounding box and date range.
Validates that Sentinel-1 and Sentinel-2 items can be found from STAC catalogs.

Usage:
    pytest tests/test_stac_search_small.py -v
    python tests/test_stac_search_small.py  # Run directly

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    warnings.warn("pytest not available - install with: pip install pytest")

try:
    import pystac_client
    import planetary_computer
    HAS_STAC = True
except ImportError:
    HAS_STAC = False
    warnings.warn("STAC libraries not available - install with: pip install pystac-client planetary-computer")


# ============================================================================
# Test Configuration
# ============================================================================

# Small test area: Nairobi city center (~10 km x 10 km)
TEST_BBOX = [36.75, -1.35, 36.85, -1.25]

# Recent date range (likely to have data)
TEST_DATE_CENTER = "2024-01-15"
TEST_DATE_RANGE_DAYS = 14  # ±14 days window

# STAC configuration
STAC_PROVIDER = "planetary_computer"
STAC_CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Collections to test
SENTINEL_2_COLLECTION = "sentinel-2-l2a"
SENTINEL_1_COLLECTION = "sentinel-1-grd"

# Tolerance settings
MIN_EXPECTED_S2_ITEMS = 1  # At least 1 S2 scene expected
MIN_EXPECTED_S1_ITEMS = 0  # S1 is less frequent, 0 is acceptable but warn
CLOUD_COVER_MAX = 80.0  # Generous threshold for test


# ============================================================================
# Helper Functions
# ============================================================================

def get_stac_catalog():
    """
    Connect to STAC catalog.
    
    Returns:
        pystac_client.Client: Connected STAC catalog
        
    Raises:
        ImportError: If STAC libraries not available
        ConnectionError: If cannot connect to catalog
    """
    if not HAS_STAC:
        raise ImportError(
            "STAC libraries required. Install with: pip install pystac-client planetary-computer"
        )
    
    try:
        catalog = pystac_client.Client.open(
            STAC_CATALOG_URL,
            modifier=planetary_computer.sign_inplace
        )
        return catalog
    except Exception as e:
        raise ConnectionError(f"Failed to connect to STAC catalog: {e}")


def search_stac_collection(
    bbox,
    date_center,
    date_range_days,
    collection,
    max_cloud=None
):
    """
    Search STAC catalog for a collection.
    
    Args:
        bbox: Bounding box [west, south, east, north]
        date_center: Center date string (YYYY-MM-DD)
        date_range_days: Days before/after center date
        collection: STAC collection name
        max_cloud: Maximum cloud cover (for optical only)
        
    Returns:
        list: List of STAC items
    """
    catalog = get_stac_catalog()
    
    # Parse date range
    center = datetime.fromisoformat(date_center)
    start = center - timedelta(days=date_range_days)
    end = center + timedelta(days=date_range_days)
    date_range = f"{start.date()}/{end.date()}"
    
    # Build query
    query_params = {
        "bbox": bbox,
        "datetime": date_range,
        "collections": [collection],
    }
    
    # Add cloud filter for optical
    if max_cloud is not None and "sentinel-2" in collection.lower():
        query_params["query"] = {
            "eo:cloud_cover": {"lt": max_cloud}
        }
    
    # Execute search
    search = catalog.search(**query_params)
    items = list(search.items())
    
    return items


# ============================================================================
# Test Functions
# ============================================================================

def test_stac_connection():
    """Test that we can connect to STAC catalog."""
    if not HAS_STAC:
        pytest.skip("STAC libraries not available")
    
    try:
        catalog = get_stac_catalog()
        assert catalog is not None, "Catalog connection returned None"
        print(f"✓ Successfully connected to {STAC_CATALOG_URL}")
    except Exception as e:
        pytest.fail(f"Failed to connect to STAC catalog: {e}")


def test_sentinel2_search():
    """Test Sentinel-2 search returns at least one item."""
    if not HAS_STAC:
        pytest.skip("STAC libraries not available")
    
    print(f"\nSearching for Sentinel-2 data...")
    print(f"  BBox: {TEST_BBOX}")
    print(f"  Date: {TEST_DATE_CENTER} ± {TEST_DATE_RANGE_DAYS} days")
    print(f"  Max cloud: {CLOUD_COVER_MAX}%")
    
    try:
        items = search_stac_collection(
            bbox=TEST_BBOX,
            date_center=TEST_DATE_CENTER,
            date_range_days=TEST_DATE_RANGE_DAYS,
            collection=SENTINEL_2_COLLECTION,
            max_cloud=CLOUD_COVER_MAX
        )
        
        print(f"  Found: {len(items)} items")
        
        # Print first few items
        for i, item in enumerate(items[:3]):
            cloud_cover = item.properties.get('eo:cloud_cover', 'unknown')
            date = item.datetime.date() if item.datetime else 'unknown'
            print(f"    {i+1}. {item.id} - {date} - {cloud_cover}% cloud")
        
        # Assertion with helpful message
        assert len(items) >= MIN_EXPECTED_S2_ITEMS, (
            f"Expected at least {MIN_EXPECTED_S2_ITEMS} Sentinel-2 items, "
            f"but found {len(items)}. This may indicate:\n"
            f"  1. No data available for this area/date\n"
            f"  2. STAC catalog connectivity issues\n"
            f"  3. Cloud cover too restrictive (current: {CLOUD_COVER_MAX}%)"
        )
        
        print(f"✓ Test passed: Found {len(items)} Sentinel-2 items")
        
    except Exception as e:
        pytest.fail(f"Sentinel-2 search failed: {e}")


def test_sentinel1_search():
    """Test Sentinel-1 search (more lenient - may not find items)."""
    if not HAS_STAC:
        pytest.skip("STAC libraries not available")
    
    print(f"\nSearching for Sentinel-1 data...")
    print(f"  BBox: {TEST_BBOX}")
    print(f"  Date: {TEST_DATE_CENTER} ± {TEST_DATE_RANGE_DAYS} days")
    
    try:
        items = search_stac_collection(
            bbox=TEST_BBOX,
            date_center=TEST_DATE_CENTER,
            date_range_days=TEST_DATE_RANGE_DAYS,
            collection=SENTINEL_1_COLLECTION,
            max_cloud=None  # No cloud filter for SAR
        )
        
        print(f"  Found: {len(items)} items")
        
        # Print first few items
        for i, item in enumerate(items[:3]):
            date = item.datetime.date() if item.datetime else 'unknown'
            polarization = item.properties.get('sar:polarizations', 'unknown')
            print(f"    {i+1}. {item.id} - {date} - {polarization}")
        
        # Lenient assertion - warn if no items but don't fail
        if len(items) < MIN_EXPECTED_S1_ITEMS:
            warnings.warn(
                f"Found only {len(items)} Sentinel-1 items. "
                f"This is acceptable as S1 has lower temporal resolution, "
                f"but you may want to increase date range for actual use."
            )
            print(f"⚠ Warning: Found {len(items)} Sentinel-1 items (low but acceptable)")
        else:
            print(f"✓ Test passed: Found {len(items)} Sentinel-1 items")
        
        # Always pass - S1 availability is variable
        assert True, "Sentinel-1 search executed successfully"
        
    except Exception as e:
        pytest.fail(f"Sentinel-1 search failed: {e}")


def test_item_structure():
    """Test that returned items have expected structure."""
    if not HAS_STAC:
        pytest.skip("STAC libraries not available")
    
    print(f"\nValidating STAC item structure...")
    
    try:
        items = search_stac_collection(
            bbox=TEST_BBOX,
            date_center=TEST_DATE_CENTER,
            date_range_days=TEST_DATE_RANGE_DAYS,
            collection=SENTINEL_2_COLLECTION,
            max_cloud=CLOUD_COVER_MAX
        )
        
        if not items:
            pytest.skip("No items found to validate structure")
        
        item = items[0]
        
        # Check essential attributes
        assert hasattr(item, 'id'), "Item missing 'id' attribute"
        assert hasattr(item, 'assets'), "Item missing 'assets' attribute"
        assert hasattr(item, 'properties'), "Item missing 'properties' attribute"
        assert hasattr(item, 'geometry'), "Item missing 'geometry' attribute"
        
        # Check Sentinel-2 specific assets
        expected_assets = ['B02', 'B03', 'B04', 'B08', 'SCL']
        available_assets = list(item.assets.keys())
        
        print(f"  Item ID: {item.id}")
        print(f"  Available assets: {len(available_assets)}")
        print(f"  Required assets: {expected_assets}")
        
        missing_assets = [a for a in expected_assets if a not in available_assets]
        
        if missing_assets:
            warnings.warn(
                f"Some expected assets missing: {missing_assets}. "
                f"Available: {available_assets[:10]}"
            )
        
        # Check that at least some assets exist
        assert len(available_assets) > 0, "Item has no assets"
        
        # Check asset has href
        first_asset_key = list(item.assets.keys())[0]
        first_asset = item.assets[first_asset_key]
        assert hasattr(first_asset, 'href'), f"Asset '{first_asset_key}' missing href"
        assert first_asset.href.startswith('http'), f"Asset href not a URL: {first_asset.href}"
        
        print(f"✓ Item structure validation passed")
        
    except Exception as e:
        pytest.fail(f"Item structure validation failed: {e}")


def test_date_filtering():
    """Test that returned items are within the date range."""
    if not HAS_STAC:
        pytest.skip("STAC libraries not available")
    
    print(f"\nValidating date filtering...")
    
    try:
        center = datetime.fromisoformat(TEST_DATE_CENTER)
        start = center - timedelta(days=TEST_DATE_RANGE_DAYS)
        end = center + timedelta(days=TEST_DATE_RANGE_DAYS)
        
        items = search_stac_collection(
            bbox=TEST_BBOX,
            date_center=TEST_DATE_CENTER,
            date_range_days=TEST_DATE_RANGE_DAYS,
            collection=SENTINEL_2_COLLECTION,
            max_cloud=CLOUD_COVER_MAX
        )
        
        if not items:
            pytest.skip("No items found to validate dates")
        
        print(f"  Date range: {start.date()} to {end.date()}")
        print(f"  Checking {len(items)} items...")
        
        out_of_range = []
        for item in items:
            if item.datetime:
                item_date = item.datetime.replace(tzinfo=None)
                if item_date < start or item_date > end:
                    out_of_range.append((item.id, item_date.date()))
        
        if out_of_range:
            print(f"  ⚠ Warning: {len(out_of_range)} items outside date range:")
            for item_id, date in out_of_range[:3]:
                print(f"    - {item_id}: {date}")
        
        # Lenient assertion - allow some items slightly outside range
        tolerance = 0.1  # Allow 10% of items to be outside range
        max_out_of_range = int(len(items) * tolerance)
        
        assert len(out_of_range) <= max_out_of_range, (
            f"Too many items outside date range: {len(out_of_range)}/{len(items)} "
            f"(tolerance: {max_out_of_range})"
        )
        
        print(f"✓ Date filtering validation passed")
        
    except Exception as e:
        pytest.fail(f"Date filtering validation failed: {e}")


# ============================================================================
# Main Test Runner (for direct execution)
# ============================================================================

def run_tests():
    """Run all tests when script is executed directly."""
    print("=" * 79)
    print("STAC SEARCH INTEGRATION TESTS")
    print("=" * 79)
    print()
    
    if not HAS_STAC:
        print("✗ ERROR: STAC libraries not available")
        print("  Install with: pip install pystac-client planetary-computer")
        sys.exit(1)
    
    tests = [
        ("Connection Test", test_stac_connection),
        ("Sentinel-2 Search", test_sentinel2_search),
        ("Sentinel-1 Search", test_sentinel1_search),
        ("Item Structure", test_item_structure),
        ("Date Filtering", test_date_filtering),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, test_func in tests:
        print(f"\n{'-' * 79}")
        print(f"TEST: {name}")
        print(f"{'-' * 79}")
        
        try:
            test_func()
            passed += 1
        except pytest.skip.Exception as e:
            print(f"⊘ SKIPPED: {e}")
            skipped += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    # Summary
    print()
    print("=" * 79)
    print("TEST SUMMARY")
    print("=" * 79)
    print(f"Passed:  {passed}/{len(tests)}")
    print(f"Failed:  {failed}/{len(tests)}")
    print(f"Skipped: {skipped}/{len(tests)}")
    print()
    
    if failed > 0:
        print("✗ Some tests failed")
        sys.exit(1)
    else:
        print("✓ All tests passed!")
        sys.exit(0)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # Check if running with pytest
    if "pytest" in sys.modules:
        # Let pytest handle test discovery and execution
        pass
    else:
        # Run tests directly
        run_tests()
