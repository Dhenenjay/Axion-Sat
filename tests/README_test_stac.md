# STAC Search Integration Tests

Validates that the project can successfully query and retrieve Sentinel-1 and Sentinel-2 imagery from STAC catalogs.

## Test File

`test_stac_search_small.py` - Integration tests for STAC search functionality

## What is Tested

### 1. **Connection Test**
- Verifies connection to Microsoft Planetary Computer STAC catalog
- Ensures credentials/authentication works

### 2. **Sentinel-2 Search**
- Searches for optical imagery in a small test area (Nairobi)
- Date range: 2024-01-15 ± 14 days
- Bbox: [36.75, -1.35, 36.85, -1.25] (~10 km x 10 km)
- **Assertion**: At least 1 S2 scene found (typically finds 3-5)
- Cloud cover filter: < 80%

### 3. **Sentinel-1 Search**
- Searches for SAR imagery in the same test area
- Same date range and bbox
- **Assertion**: Search completes successfully (may find 0+ items)
- S1 is less frequent than S2, so zero results are acceptable with warning

### 4. **Item Structure Validation**
- Checks that returned STAC items have expected structure
- Validates presence of required assets (B02, B03, B04, B08, SCL)
- Verifies asset HREFs are valid URLs

### 5. **Date Filtering**
- Ensures returned items fall within requested date range
- Allows 10% tolerance for items slightly outside range
- Checks datetime parsing and filtering logic

## Test Configuration

```python
# Small test area: Nairobi city center
TEST_BBOX = [36.75, -1.35, 36.85, -1.25]

# Recent date with likely data availability
TEST_DATE_CENTER = "2024-01-15"
TEST_DATE_RANGE_DAYS = 14

# STAC catalog
STAC_CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Tolerance
MIN_EXPECTED_S2_ITEMS = 1
CLOUD_COVER_MAX = 80.0
```

## Running Tests

### With pytest (recommended)

```bash
# Run all tests with verbose output
pytest tests/test_stac_search_small.py -v

# Run specific test
pytest tests/test_stac_search_small.py::test_sentinel2_search -v

# Run with output capture disabled (see print statements)
pytest tests/test_stac_search_small.py -v -s

# Run with detailed failure info
pytest tests/test_stac_search_small.py -vv
```

### Direct execution

```bash
# Run all tests
python tests/test_stac_search_small.py
```

## Expected Output

### Success
```
===============================================================================
STAC SEARCH INTEGRATION TESTS
===============================================================================

-------------------------------------------------------------------------------
TEST: Connection Test
-------------------------------------------------------------------------------
✓ Successfully connected to https://planetarycomputer.microsoft.com/api/stac/v1

-------------------------------------------------------------------------------
TEST: Sentinel-2 Search
-------------------------------------------------------------------------------

Searching for Sentinel-2 data...
  BBox: [36.75, -1.35, 36.85, -1.25]
  Date: 2024-01-15 ± 14 days
  Max cloud: 80.0%
  Found: 4 items
    1. S2B_MSIL2A_20240128T074059_R092_T37MBU_20240128T121054 - 2024-01-28 - 21.8% cloud
    2. S2A_MSIL2A_20240123T074211_R092_T37MBU_20240123T132925 - 2024-01-23 - 16.7% cloud
    3. S2B_MSIL2A_20240108T074219_R092_T37MBU_20240108T120832 - 2024-01-08 - 66.5% cloud
✓ Test passed: Found 4 Sentinel-2 items

-------------------------------------------------------------------------------
TEST: Sentinel-1 Search
-------------------------------------------------------------------------------

Searching for Sentinel-1 data...
  BBox: [36.75, -1.35, 36.85, -1.25]
  Date: 2024-01-15 ± 14 days
  Found: 4 items
    1. S1A_IW_GRDH_1SDV_20240126T154817_20240126T154842_052279_065210 - 2024-01-26 - ['VV', 'VH']
    2. S1A_IW_GRDH_1SDV_20240114T154817_20240114T154842_052104_064C34 - 2024-01-14 - ['VV', 'VH']
✓ Test passed: Found 4 Sentinel-1 items

===============================================================================
TEST SUMMARY
===============================================================================
Passed:  5/5
Failed:  0/5
Skipped: 0/5

✓ All tests passed!
```

## Troubleshooting

### No S2 items found

**Possible causes:**
1. Date is too recent (data processing lag)
2. Bbox covers only water/ocean
3. Cloud cover too restrictive

**Solutions:**
- Try date from 1-2 months ago
- Increase `TEST_DATE_RANGE_DAYS` to 30
- Increase `CLOUD_COVER_MAX` to 90 or 100
- Try different bbox (e.g., over land)

### Connection errors

**Possible causes:**
1. No internet connection
2. STAC catalog temporarily unavailable
3. Firewall blocking HTTPS

**Solutions:**
- Check internet connection
- Wait and retry
- Try alternative STAC provider (AWS Earth Search)

### Import errors

**Missing libraries:**
```bash
pip install pystac-client
pip install planetary-computer
pip install pytest
```

## Test Data Location

The test uses a **real-world location**:
- **Place**: Nairobi, Kenya (city center)
- **Coordinates**: 36.75°E to 36.85°E, -1.35°N to -1.25°N
- **Area**: ~100 km² (10 km x 10 km)
- **Why Nairobi**: 
  - Near equator (frequent coverage)
  - Urban area (always interesting)
  - Typically cloud-free in January
  - Known to have reliable S1/S2 data

## Modifying Tests

### Test different location

```python
# Change bbox to your area of interest
TEST_BBOX = [lon_west, lat_south, lon_east, lat_north]
```

### Test different date

```python
# Use any date (recommend 1-3 months in past)
TEST_DATE_CENTER = "2023-07-15"
TEST_DATE_RANGE_DAYS = 30  # Wider window
```

### Use different STAC provider

```python
# AWS Earth Search
STAC_CATALOG_URL = "https://earth-search.aws.element84.com/v1"

# Update connection code
catalog = pystac_client.Client.open(STAC_CATALOG_URL)
# (no need for planetary_computer.sign_inplace)
```

## Integration with CI/CD

Add to GitHub Actions workflow:

```yaml
- name: Run STAC integration tests
  run: |
    pytest tests/test_stac_search_small.py -v
  env:
    PYTHONPATH: ${{ github.workspace }}
```

## Performance

- **Typical runtime**: 5-10 seconds
- **Network calls**: 2-3 API requests
- **Bandwidth**: < 1 MB (only metadata, no imagery downloaded)

## Notes

- Tests use **metadata only** - no actual imagery is downloaded
- Safe to run frequently - very lightweight
- Network connection required
- Tests may occasionally fail if STAC catalog is updating

## Related Files

- `scripts/build_tiles.py` - Uses same STAC search logic
- `docs/aoi_catalog.md` - Sample bboxes and dates

## Future Improvements

Potential enhancements:
- [ ] Test COG download (small window)
- [ ] Test band alignment
- [ ] Test cloud masking with real SCL
- [ ] Parameterize test location/date
- [ ] Add timeout handling
- [ ] Test retry logic
- [ ] Mock tests for offline testing

## Questions?

For issues with STAC queries:
- Check Microsoft Planetary Computer status: https://planetarycomputer.microsoft.com/
- STAC spec docs: https://stacspec.org/
- pystac-client docs: https://pystac-client.readthedocs.io/
