# build_tiles.py - Satellite Tile Building Pipeline

Complete pipeline to generate training-ready satellite imagery tiles from STAC catalogs.

## Overview

This script orchestrates a 7-step pipeline:
1. **Geocoding** - Convert place names to bounding boxes
2. **STAC Search** - Find Sentinel-1 and Sentinel-2 imagery
3. **Download** - Fetch COG assets with windowed reading
4. **Align** - Reproject and align all bands to common grid
5. **Cloud Masking** - Apply quality filtering using SCL
6. **Tiling** - Generate NPZ tiles with metadata
7. **Train/Val/Test Split** - Create index CSV with splits

## Requirements

```bash
pip install rasterio
pip install pystac-client
pip install planetary-computer
pip install geopy
pip install scipy  # optional, for faster resizing
pip install pyproj
```

## Usage

### From Place Name

```bash
python scripts/build_tiles.py \
    --place "Lake Victoria" \
    --date 2023-06-15 \
    --tile-size 256 \
    --overlap 32 \
    --max-cloud 30 \
    --split-ratio 0.7 0.15 0.15
```

### From Bounding Box

```bash
python scripts/build_tiles.py \
    --bbox 31.5 -1.5 34.0 0.5 \
    --date 2023-06-15 \
    --tile-size 512 \
    --overlap 0 \
    --max-cloud 40 \
    --split-ratio 0.8 0.1 0.1
```

### Small Test Area

```bash
# Small bbox around Nairobi, Kenya
python scripts/build_tiles.py \
    --bbox 36.7 -1.35 36.9 -1.25 \
    --date 2024-01-15 \
    --tile-size 256 \
    --overlap 0 \
    --max-cloud 20 \
    --split-ratio 0.7 0.15 0.15
```

## Arguments

### Required
- `--place "Name"` OR `--bbox west south east north` - Area of interest
- `--date YYYY-MM-DD` - Acquisition date

### Optional
- `--date-range DAYS` - Search window (default: 7 days)
- `--tile-size PIXELS` - Tile size (default: 256)
- `--overlap PIXELS` - Overlap between tiles (default: 0)
- `--max-cloud PERCENT` - Maximum cloud cover (default: 40.0)
- `--split-ratio TRAIN VAL TEST` - Split ratios (default: 0.7 0.15 0.15)
- `--random-seed INT` - Random seed for splitting (default: 42)
- `--output-dir PATH` - Output directory (default: data/tiles)
- `--index-dir PATH` - Index CSV directory (default: data/index)
- `--cache-dir PATH` - Download cache (default: cache)
- `--stac-provider` - planetary_computer or earthsearch (default: planetary_computer)
- `--collections` - STAC collections (default: sentinel-2-l2a sentinel-1-grd)

## Output Structure

```
data/
├── tiles/
│   └── lake_victoria_2023-06-15/
│       ├── lake_victoria_2023-06-15_r00000_c00000.npz
│       ├── lake_victoria_2023-06-15_r00000_c00000.json
│       └── ...
└── index/
    └── lake_victoria_2023-06-15_tiles.csv

cache/
└── S2B_MSIL2A_20230615T074609_R135_T36NVH_*_B*.tif
```

### Tile NPZ Contents

Each `.npz` file contains:
- `s1_vv` - Sentinel-1 VV polarization (dB)
- `s1_vh` - Sentinel-1 VH polarization (dB)
- `s2_b2` - Sentinel-2 Blue (10m)
- `s2_b3` - Sentinel-2 Green (10m)
- `s2_b4` - Sentinel-2 Red (10m)
- `s2_b8` - Sentinel-2 NIR (10m)

All bands are aligned to a common 10m grid in UTM projection.

### Tile JSON Metadata

Each `.json` sidecar contains:
```json
{
  "tile_id": "lake_victoria_2023-06-15_r00000_c00000",
  "row": 0,
  "col": 0,
  "width": 256,
  "height": 256,
  "geotransform": [500000, 10, 0, 9000000, 0, -10],
  "crs": "EPSG:32636",
  "timestamp": "2025-10-13T14:00:00.000000Z",
  "tile_bounds": {
    "minx": 500000,
    "miny": 8997440,
    "maxx": 502560,
    "maxy": 9000000
  }
}
```

### Index CSV Format

```csv
tile_path,tile_id,split
data/tiles/lake_victoria_2023-06-15/tile_r00000_c00000.npz,tile_r00000_c00000,train
data/tiles/lake_victoria_2023-06-15/tile_r00256_c00000.npz,tile_r00256_c00000,train
data/tiles/lake_victoria_2023-06-15/tile_r00512_c00000.npz,tile_r00512_c00000,val
...
```

## Features

### 1. COG Windowed Reading
Downloads only the required bbox from Cloud-Optimized GeoTIFFs, saving bandwidth and disk space.

### 2. Automatic Alignment
- Sentinel-2 bands: Already aligned (same grid)
- Sentinel-1 bands: Reprojected and aligned to S2 grid
- Automatic UTM zone detection
- Terrain correction (approximate) for S1
- Conversion to dB scale for SAR

### 3. Cloud Quality Control
- Uses Sentinel-2 SCL (Scene Classification Layer)
- Filters scenes by cloud cover percentage
- Masks clouds, shadows, and invalid pixels
- Requires 50% valid pixels per tile

### 4. Reproducible Splits
- Random seed for reproducibility
- Configurable train/val/test ratios
- CSV index for easy dataset loading

## Example Workflow

```bash
# 1. Build tiles for a small test area
python scripts/build_tiles.py \
    --place "Nairobi, Kenya" \
    --date 2024-01-15 \
    --tile-size 256 \
    --max-cloud 20

# 2. Check output
ls data/tiles/nairobi_kenya_2024-01-15/
ls data/index/

# 3. Load in training script
import pandas as pd
import numpy as np

index = pd.read_csv('data/index/nairobi_kenya_2024-01-15_tiles.csv')
train_tiles = index[index['split'] == 'train']

for _, row in train_tiles.iterrows():
    tile = np.load(row['tile_path'])
    s2_rgb = np.stack([tile['s2_b4'], tile['s2_b3'], tile['s2_b2']], axis=-1)
    s1_vv = tile['s1_vv']
    # ... training code ...
```

## Troubleshooting

### No imagery found
- Try increasing `--date-range` (e.g., 14 or 30 days)
- Check cloud cover threshold with `--max-cloud 60`
- Verify bbox covers land (not open ocean)

### Download errors
- Check internet connection
- STAC catalogs may be temporarily unavailable
- Try switching providers: `--stac-provider earthsearch`

### Alignment warnings
- S1 and S2 may have slightly different footprints
- Overlap is handled automatically
- Areas outside overlap will be NaN-masked

### Memory issues
- Reduce `--tile-size` (e.g., 128 or 64)
- Process smaller bboxes
- Close other applications

## Advanced Usage

### Multiple Time Points
```bash
# Generate tiles for multiple dates
for date in 2023-06-15 2023-07-15 2023-08-15; do
    python scripts/build_tiles.py \
        --bbox 31.5 -1.5 34.0 0.5 \
        --date $date \
        --tile-size 256
done
```

### Custom Collections
```bash
# Only Sentinel-2 (no SAR)
python scripts/build_tiles.py \
    --place "Paris, France" \
    --date 2024-01-15 \
    --collections sentinel-2-l2a
```

### High Overlap for Prediction
```bash
# 50% overlap for seamless mosaicking
python scripts/build_tiles.py \
    --bbox 36.7 -1.35 36.9 -1.25 \
    --date 2024-01-15 \
    --tile-size 256 \
    --overlap 128
```

## Notes

- **Real Data Only**: This script downloads and processes real satellite data from STAC catalogs
- **Cloud-Optimized**: Uses COG windowed reading for efficiency
- **Production Ready**: Includes proper error handling, logging, and validation
- **Terrain Correction**: S1 correction is approximate (see `axs_lib/geo.py` for details)
- **Resolution**: All data resampled to 10m (Sentinel-2 native)

## References

- [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)
- [AWS Earth Search](https://registry.opendata.aws/sentinel-2/)
- [STAC Specification](https://stacspec.org/)
- [Sentinel-2 User Guide](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi)
- [Sentinel-1 User Guide](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar)
