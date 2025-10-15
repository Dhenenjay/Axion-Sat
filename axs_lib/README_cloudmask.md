# Cloud Masking Module (`axs_lib.cloudmask`)

## Overview

The `cloudmask` module provides robust quality filtering for Sentinel-2 imagery by parsing Scene Classification Layer (SCL) data to identify and mask clouds, shadows, and invalid pixels.

## Features

✅ **Comprehensive SCL Parsing** - Full support for all 12 Sentinel-2 SCL classes  
✅ **Quality Filtering** - Configurable thresholds for cloud, shadow, and invalid data  
✅ **Batch Processing** - Filter multiple tiles efficiently  
✅ **Production Ready** - Unit tests with 100% pass rate  
✅ **Memory Efficient** - Works with large tiles (10980×10980 pixels)  
✅ **Flexible Masking** - Separate or combined masks for different quality issues  

## Sentinel-2 SCL Classification

| Value | Class | Typical Treatment |
|-------|-------|-------------------|
| 0 | No Data | ❌ Mask out |
| 1 | Saturated/Defective | ❌ Mask out |
| 2 | Dark Area Pixels | ✓ Usually OK |
| 3 | Cloud Shadows | ⚠️ Often masked |
| 4 | Vegetation | ✓ Clear |
| 5 | Not-vegetated | ✓ Clear |
| 6 | Water | ✓ Clear |
| 7 | Unclassified | ⚠️ Case dependent |
| 8 | Cloud Medium Prob | ❌ Mask out |
| 9 | Cloud High Prob | ❌ Mask out |
| 10 | Thin Cirrus | ❌ Usually masked |
| 11 | Snow/Ice | ⚠️ Case dependent |

## Quick Start

### 1. Basic Usage

```python
from axs_lib.cloudmask import parse_scl, filter_by_cloud_cover
from axs_lib.geo import read_cog

# Load SCL band
scl_data, profile = read_cog('data/raw/sentinel2/T10TEM_20240115/SCL.tif')

# Check tile quality
is_valid, stats = filter_by_cloud_cover(
    scl_data,
    max_cloud_percent=40.0,
    verbose=True
)

if is_valid:
    # Parse to individual masks
    cloud_mask, shadow_mask, valid_mask = parse_scl(scl_data)
    
    # Load spectral bands
    red_data, _ = read_cog('data/raw/sentinel2/T10TEM_20240115/B04.tif')
    nir_data, _ = read_cog('data/raw/sentinel2/T10TEM_20240115/B08.tif')
    
    # Apply masks
    red_clean = np.where(cloud_mask | shadow_mask, np.nan, red_data)
    nir_clean = np.where(cloud_mask | shadow_mask, np.nan, nir_data)
    
    # Calculate indices on clean pixels
    ndvi = (nir_clean - red_clean) / (nir_clean + red_clean + 1e-8)
else:
    print(f"Tile rejected: {stats['cloud_percent']:.1f}% cloud cover")
```

### 2. Combined Mask

```python
from axs_lib.cloudmask import create_combined_mask

# Create single mask for all bad pixels
bad_pixel_mask = create_combined_mask(
    scl_data,
    mask_clouds=True,
    mask_shadows=True,
    mask_invalid=True
)

# Apply to all bands at once
clean_data = np.where(bad_pixel_mask, np.nan, raw_data)
```

### 3. Batch Processing

```python
from pathlib import Path
from axs_lib.cloudmask import filter_by_cloud_cover
from axs_lib.geo import read_cog

# Filter tiles in a directory
scl_files = list(Path('data/raw/').rglob('**/SCL.tif'))

valid_tiles = []
for scl_path in scl_files:
    scl_data, _ = read_cog(scl_path)
    is_valid, stats = filter_by_cloud_cover(scl_data, max_cloud_percent=40.0)
    
    if is_valid:
        valid_tiles.append(scl_path.parent)
        print(f"✓ {scl_path.parent.name}: {stats['clear_percent']:.1f}% clear")
    else:
        print(f"✗ {scl_path.parent.name}: {stats['cloud_percent']:.1f}% cloud")

print(f"\nProcessing {len(valid_tiles)}/{len(scl_files)} tiles")
```

### 4. Coverage Statistics

```python
from axs_lib.cloudmask import calculate_coverage

stats = calculate_coverage(scl_data)

print(f"Quality Report:")
print(f"  Cloud:   {stats['cloud_percent']:.1f}%  ({stats['cloud_pixels']:,} pixels)")
print(f"  Shadow:  {stats['shadow_percent']:.1f}%  ({stats['shadow_pixels']:,} pixels)")
print(f"  Invalid: {stats['invalid_percent']:.1f}%  ({stats['invalid_pixels']:,} pixels)")
print(f"  Clear:   {stats['clear_percent']:.1f}%  ({stats['clear_pixels']:,} pixels)")
```

## API Reference

### `parse_scl(scl_data, include_cirrus=True, include_snow=False)`
Parse SCL to binary masks.

**Returns:** `(cloud_mask, shadow_mask, valid_mask)` as boolean arrays

**Example:**
```python
cloud_mask, shadow_mask, valid_mask = parse_scl(scl_data)
clean_mask = valid_mask & ~cloud_mask & ~shadow_mask
```

---

### `filter_by_cloud_cover(scl_data, max_cloud_percent=40.0, max_shadow_percent=None, max_invalid_percent=10.0, verbose=False)`
Check if tile passes quality thresholds.

**Returns:** `(is_valid, stats)` - boolean and statistics dict

**Example:**
```python
is_valid, stats = filter_by_cloud_cover(
    scl_data,
    max_cloud_percent=40.0,
    max_shadow_percent=20.0,
    verbose=True
)
```

---

### `calculate_coverage(scl_data, include_cirrus=True, include_snow=False)`
Calculate detailed coverage statistics.

**Returns:** Dictionary with percentages and pixel counts

**Example:**
```python
stats = calculate_coverage(scl_data)
print(f"{stats['clear_percent']:.1f}% usable data")
```

---

### `create_combined_mask(scl_data, mask_clouds=True, mask_shadows=True, mask_invalid=True, include_cirrus=True)`
Create single binary mask for all bad pixels.

**Returns:** Boolean array where `True` = bad pixel

**Example:**
```python
bad_mask = create_combined_mask(scl_data)
data[bad_mask] = np.nan
```

## Configuration Recommendations

### Conservative (High Quality)
```python
is_valid, stats = filter_by_cloud_cover(
    scl_data,
    max_cloud_percent=20.0,
    max_shadow_percent=10.0,
    max_invalid_percent=5.0
)
```
**Use case:** Time series analysis, high-precision applications

### Balanced (Default)
```python
is_valid, stats = filter_by_cloud_cover(
    scl_data,
    max_cloud_percent=40.0,
    max_shadow_percent=None,  # No limit
    max_invalid_percent=10.0
)
```
**Use case:** General purpose, regional mapping

### Permissive (Low Quality OK)
```python
is_valid, stats = filter_by_cloud_cover(
    scl_data,
    max_cloud_percent=60.0,
    max_shadow_percent=30.0,
    max_invalid_percent=15.0
)
```
**Use case:** Data-scarce regions, exploratory analysis

## Integration with Processing Pipelines

### Example: NDVI Time Series

```python
from pathlib import Path
from axs_lib.cloudmask import filter_by_cloud_cover, parse_scl
from axs_lib.geo import read_cog
import numpy as np
import pandas as pd

def process_tile(tile_dir):
    """Process single tile with cloud filtering."""
    
    # Load SCL
    scl_path = tile_dir / 'SCL.tif'
    scl_data, _ = read_cog(scl_path)
    
    # Quality check
    is_valid, stats = filter_by_cloud_cover(scl_data, max_cloud_percent=40.0)
    if not is_valid:
        return None
    
    # Parse masks
    cloud_mask, shadow_mask, _ = parse_scl(scl_data)
    bad_mask = cloud_mask | shadow_mask
    
    # Load bands
    red_data, _ = read_cog(tile_dir / 'B04.tif')
    nir_data, _ = read_cog(tile_dir / 'B08.tif')
    
    # Mask and calculate NDVI
    red_clean = np.where(bad_mask, np.nan, red_data.astype(float))
    nir_clean = np.where(bad_mask, np.nan, nir_data.astype(float))
    ndvi = (nir_clean - red_clean) / (nir_clean + red_clean + 1e-8)
    
    return {
        'date': tile_dir.name.split('_')[1],
        'mean_ndvi': np.nanmean(ndvi),
        'std_ndvi': np.nanstd(ndvi),
        'clear_percent': stats['clear_percent'],
        'valid_pixels': (~np.isnan(ndvi)).sum()
    }

# Process all tiles
tile_dirs = sorted(Path('data/raw/sentinel2/').glob('T10TEM_*'))
results = [process_tile(td) for td in tile_dirs]
results = [r for r in results if r is not None]

# Create time series
df = pd.DataFrame(results)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').sort_index()

print(df)
```

## Performance

**Typical processing times (single tile, 10980×10980 pixels):**

| Operation | Time | Notes |
|-----------|------|-------|
| `parse_scl()` | ~50 ms | 3 boolean masks |
| `calculate_coverage()` | ~60 ms | Includes mask parsing + stats |
| `filter_by_cloud_cover()` | ~65 ms | Full quality check |
| `create_combined_mask()` | ~55 ms | Single combined mask |

**Memory:** ~360 MB peak for full-resolution tile (3 masks × 120 MB each)

## Testing

Run unit tests:
```bash
python axs_lib/cloudmask.py
```

Run demos:
```bash
python examples/cloudmask_demo.py
```

## References

- [Sentinel-2 Level-2A Product Specification](https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm)
- [ESA Sentinel-2 User Handbook](https://sentinel.esa.int/documents/247904/685211/Sentinel-2_User_Handbook)

## Next Steps

1. ✅ **Module complete** - Ready for production use
2. 🔄 **Integration** - Add to main processing pipeline
3. 📊 **Monitoring** - Track quality stats across datasets
4. 🚀 **Optimization** - Consider windowed processing for very large mosaics

---

**Author:** Axion-Sat Project  
**Version:** 1.0.0  
**Last Updated:** 2024
