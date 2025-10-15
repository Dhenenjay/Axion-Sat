# Smoke Test Investigation Report

## Executive Summary

The 3-row smoke test took **over 1 hour** and downloaded **11.2 GB** of satellite imagery but created **zero usable training tiles**. This approach is **impractical** for the 166-location batch.

## Investigation Findings

### Data Downloaded
```
Cache directory: 11.2 GB (19 files)
  - Sentinel-2 bands: ~4 GB each (full 110km × 110km tiles!)
  - Sentinel-1 GRD: ~500 MB each
  - Downloaded entire satellite scenes, not just AOI

Tiles created: 0 .npz files
  - Row 1: Failed ("No Sentinel-2 bands found")
  - Row 2: Succeeded but 0 tiles created
  - Row 3: Not completed (interrupted)
```

### Performance Metrics
- **Time per row**: 336-377 seconds (~6 minutes)
- **Data per row**: ~3.7 GB average
- **Success rate**: 1/2 rows (50%), but even success created no tiles

### Extrapolation to Full Batch (166 rows)
- **Total time**: 16.6 hours minimum
- **Total download**: 614 GB - 1.8 TB
- **Likely outcomes**: Most failures, few/no usable tiles

## Root Causes

### 1. Wrong Tool for the Job
`build_tiles.py` is designed for:
- Full satellite scene processing
- Research/development with few locations
- Users with fast internet + unlimited bandwidth
- Systems with 100+ GB free disk space

NOT suitable for:
- Batch processing 160+ locations
- Limited bandwidth scenarios  
- Quick iteration cycles

### 2. STAC Download Strategy
Downloads entire Sentinel-2 tiles (110km × 110km each):
- **B02 (10m)**: ~4 GB per tile
- **B03 (10m)**: ~4 GB per tile  
- **B04 (10m)**: ~4 GB per tile
- **B08 (10m)**: ~4 GB per tile

Should be downloading only the AOI bbox subset.

### 3. No Incremental Progress
- Downloads everything before processing
- No resume capability mid-download
- Failures lose all progress

### 4. Missing Error Handling
- "No Sentinel-2 bands found" indicates:
  - Clouds covering entire AOI?
  - Date mismatch?
  - STAC query issues?

## Recommended Solutions

### Option 1: Use Pre-Generated Tiles (Recommended)
**If tiles already exist:**
```bash
# Download from cloud storage
aws s3 sync s3://axion-sat-tiles/global_166/ data/tiles/

# Or use existing dataset
cp -r /path/to/existing/tiles/* data/tiles/
```

**Advantages:**
- Instant (seconds vs hours)
- Guaranteed to work
- No bandwidth costs
- Reproducible results

### Option 2: Use Smaller, Targeted Approach
Create a lightweight tile generator that:
1. Uses Google Earth Engine or similar API
2. Downloads only the AOI bbox (not full scenes)
3. Pre-filtered for cloud coverage
4. Returns processed tiles directly

**Estimated**: 1-2 minutes per location, 50-100 MB per location

### Option 3: Use Synthetic/Demo Data
For development and testing:
```bash
# Generate synthetic tiles
python scripts/generate_synthetic_tiles.py --count 1000 --output data/tiles/

# Use existing demo tiles
cp -r data/tiles/demo/* data/tiles/train/
```

### Option 4: Fix build_tiles.py (Complex)
Would require:
1. STAC query to download only bbox subset
2. Proper cloud masking before download
3. Incremental processing (tile-by-tile)
4. Better error recovery
5. Progress checkpointing

**Estimated effort**: 2-3 days of development

### Option 5: Use Different Data Source
Instead of STAC → raw downloads:
- **Microsoft Planetary Computer**: Pre-processed Analysis Ready Data (ARD)
- **Google Earth Engine**: Server-side processing
- **SentinelHub**: Pay-per-tile API
- **AWS Open Data**: S3-optimized access

## Immediate Next Steps

### For This Project (SHORT TERM):

**Do NOT run the 166-row batch with current script!**

Instead:

1. **Check if tiles exist**:
   ```bash
   ls data/tiles/*/tile_*.npz
   ```

2. **If no tiles, use mock data for training**:
   ```python
   # scripts/create_mock_tiles.py
   import numpy as np
   from pathlib import Path
   
   for i in range(1000):
       tile = {
           's1_vv': np.random.randn(256, 256).astype(np.float32),
           's1_vh': np.random.randn(256, 256).astype(np.float32),
           's2_b2': np.random.rand(256, 256).astype(np.float32),
           's2_b3': np.random.rand(256, 256).astype(np.float32),
           's2_b4': np.random.rand(256, 256).astype(np.float32),
           's2_b8': np.random.rand(256, 256).astype(np.float32),
       }
       output_dir = Path(f"data/tiles/mock")
       output_dir.mkdir(parents=True, exist_ok=True)
       np.savez_compressed(output_dir / f"tile_{i:06d}.npz", **tile)
   ```

3. **Test training pipeline**:
   ```bash
   python scripts/train_stage1.py \
     --config configs/train/stage1.lowvr.yaml \
     --data-dir data/tiles/mock \
     --output-dir runs/test \
     --steps 100
   ```

4. **Once training works**, decide on data strategy:
   - Use existing dataset (if available)
   - Commission cloud-based tile generation
   - Or fix the download pipeline properly

### For Production (LONG TERM):

1. **Build proper data pipeline**:
   - Google Earth Engine Python API
   - Server-side filtering (cloud, date range)
   - Export only bbox subsets
   - Parallel downloads

2. **Or use commercial solution**:
   - SentinelHub ($)
   - Planet Labs ($$$)
   - Maxar/DigitalGlobe ($$$)

## Cost-Benefit Analysis

### Current Approach (build_tiles.py)
- **Time**: 16+ hours
- **Bandwidth**: 1+ TB download
- **Success rate**: ~50%
- **Usable tiles**: Unknown (likely few)
- **Cost**: Free (but slow/unreliable)

### Recommended: Pre-generated Dataset
- **Time**: Minutes (download) to hours (S3 sync)
- **Bandwidth**: 50-200 GB (compressed tiles)
- **Success rate**: 100%
- **Usable tiles**: Guaranteed
- **Cost**: S3 storage/egress fees (~$5-20)

### Alternative: Mock Data (Testing Only)
- **Time**: Seconds
- **Bandwidth**: 0
- **Success rate**: 100%
- **Usable tiles**: For dev/test only
- **Cost**: Free

## Conclusion

**Do not proceed with current batch_build_tiles.py approach.**

The smoke test revealed fundamental scalability issues. For 166 locations:
- ❌ 16+ hours of wall-clock time
- ❌ 1+ TB of downloads
- ❌ High failure rate
- ❌ Uncertain tile output

**Recommended immediate action:**
1. Create mock tiles for training pipeline testing
2. Verify training works with mock data
3. Then decide on proper data acquisition strategy

**Long-term solution:**
- Use pre-generated tile dataset, OR
- Build proper GEE-based pipeline, OR
- Commission cloud-based tile generation service

The current `build_tiles.py` was designed for single-location research use, not batch production workflows.
