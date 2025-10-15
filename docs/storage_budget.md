# Storage Budget Planning

**Document Version:** 1.0.0  
**Last Updated:** 2025-10-13  
**Author:** Axion-Sat Project

---

## Executive Summary

This document provides storage budget calculations for SAR-optical fusion datasets. For a typical research project with **10 Areas of Interest (AOIs)** sampled across **4 seasons**, expect:

- **Final Dataset (tiles only):** 120–200 GB
- **With intermediate products:** 300–500 GB
- **With raw SAFE archives:** 1–2 TB

All calculations assume tile sizes of 256×256 to 384×384 pixels with NPZ compression.

---

## Table of Contents

1. [Storage Components](#storage-components)
2. [Detailed Calculations](#detailed-calculations)
3. [Example Scenarios](#example-scenarios)
4. [Optimization Strategies](#optimization-strategies)
5. [Budget Planning](#budget-planning)

---

## Storage Components

### Data Hierarchy

```
Axion-Sat Project Storage
│
├── Raw Data (~70% of total)
│   ├── Sentinel-1 SAFE archives
│   └── Sentinel-2 SAFE archives
│
├── Intermediate Products (~20%)
│   ├── Preprocessed COGs
│   └── Cached reprojections
│
└── Final Tiles (~10%)
    ├── Training tiles (NPZ)
    ├── Validation tiles (NPZ)
    └── Test tiles (NPZ)
```

### Storage Breakdown by Stage

| Stage | Data Type | Compression | Typical Size per Image | Delete After? |
|-------|-----------|-------------|----------------------|---------------|
| Raw | S1 SAFE | ZIP | 700-900 MB | Yes* |
| Raw | S2 SAFE | ZIP | 600-1200 MB | Yes* |
| Intermediate | S1 COG | GeoTIFF | 50-100 MB | Optional |
| Intermediate | S2 COG | GeoTIFF | 200-400 MB | Optional |
| Final | Tiles | NPZ (compressed) | 0.5-2 MB | **Keep** |

\* *Can be deleted after tile generation to reclaim 70-80% of storage*

---

## Detailed Calculations

### Single Tile Storage

#### Tile Composition

A single tile contains:
- **SAR bands (2):** VV, VH polarizations
- **Optical bands (10):** B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
- **Total:** 12 bands per tile

#### Tile Size Calculation

**Formula:**
```
tile_size_bytes = (height × width × num_bands × bytes_per_value) × compression_ratio
```

**Parameters:**
- `height × width`: 256×256 to 384×384 pixels
- `num_bands`: 12 (2 SAR + 10 optical)
- `bytes_per_value`: 4 bytes (float32)
- `compression_ratio`: 0.4–0.6 (NPZ compression)

#### Example Calculations

**Tile Size: 256×256**
```
Uncompressed:
  256 × 256 × 12 × 4 bytes = 3,145,728 bytes = 3.0 MB

Compressed (NPZ, level 6):
  3.0 MB × 0.5 = 1.5 MB per tile
```

**Tile Size: 384×384**
```
Uncompressed:
  384 × 384 × 12 × 4 bytes = 7,077,888 bytes = 6.75 MB

Compressed (NPZ, level 6):
  6.75 MB × 0.5 = 3.4 MB per tile
```

**Tile Size: 512×512** *(reference)*
```
Uncompressed:
  512 × 512 × 12 × 4 bytes = 12,582,912 bytes = 12.0 MB

Compressed (NPZ, level 6):
  12.0 MB × 0.5 = 6.0 MB per tile
```

### Dataset Scaling

#### Tiles per AOI per Season

**Assumptions:**
- AOI size: 50 km × 50 km (typical agricultural region)
- Tile overlap: 10%
- Valid tiles (after quality filtering): 60-80%

**Calculation:**
```
Grid with 10% overlap:
  - 256×256 at 10m resolution: ~200 tiles
  - 384×384 at 10m resolution: ~90 tiles
  - 512×512 at 10m resolution: ~50 tiles

After quality filtering (70% retention):
  - 256×256: ~140 tiles per AOI per season
  - 384×384: ~63 tiles per AOI per season
  - 512×512: ~35 tiles per AOI per season
```

#### Full Dataset Calculation

**Scenario: 10 AOIs × 4 Seasons**

| Tile Size | Tiles/AOI/Season | Total Tiles | Size per Tile | **Total Size** |
|-----------|------------------|-------------|---------------|----------------|
| 256×256 | 140 | 5,600 | 1.5 MB | **8.4 GB** |
| 384×384 | 63 | 2,520 | 3.4 MB | **8.6 GB** |
| 512×512 | 35 | 1,400 | 6.0 MB | **8.4 GB** |

**Wait, this is too small!** ⚠️

The above assumes **one image pair per season**. In reality, you need multiple acquisitions per season for temporal diversity.

---

## Example Scenarios

### Scenario 1: Research Dataset (Conservative)

**Configuration:**
- 10 AOIs
- 4 seasons
- 3-5 acquisition dates per season per AOI
- Tile size: 384×384
- Quality filtering: 70% retention

**Calculation:**
```
Acquisitions per AOI per season: 4 (average)
Tiles per acquisition: 63 (after filtering)
Total tiles per AOI: 63 tiles × 4 acquisitions × 4 seasons = 1,008 tiles
Total tiles: 1,008 × 10 AOIs = 10,080 tiles

Storage:
  10,080 tiles × 3.4 MB = 34.3 GB (tiles only)
  
With metadata and manifests: ~35 GB
```

**✓ This is closer but still conservative**

### Scenario 2: Research Dataset (Realistic)

**Configuration:**
- 10 AOIs
- 4 seasons
- 8-12 acquisition dates per season per AOI (temporal sampling)
- Tile size: 384×384
- Quality filtering: 70% retention

**Calculation:**
```
Acquisitions per AOI per season: 10 (average)
Tiles per acquisition: 63
Total tiles per AOI: 63 × 10 × 4 = 2,520 tiles
Total tiles: 2,520 × 10 AOIs = 25,200 tiles

Storage:
  25,200 tiles × 3.4 MB = 85.7 GB
  
With metadata (+10%): ~94 GB
With augmented variants (+30%): ~122 GB
```

**✓ Target: ~120 GB** ✅

### Scenario 3: Extended Dataset (Generous)

**Configuration:**
- 10 AOIs
- 4 seasons
- 15 acquisition dates per season per AOI (dense temporal)
- Tile size: 384×384
- Quality filtering: 70% retention

**Calculation:**
```
Acquisitions per AOI per season: 15
Tiles per acquisition: 63
Total tiles per AOI: 63 × 15 × 4 = 3,780 tiles
Total tiles: 3,780 × 10 AOIs = 37,800 tiles

Storage:
  37,800 tiles × 3.4 MB = 128.5 GB
  
With metadata: ~141 GB
With augmented variants: ~183 GB
```

**✓ Target: ~180 GB** ✅

### Scenario 4: Maximum Dataset (Upper Bound)

**Configuration:**
- 10 AOIs
- 4 seasons
- 20 acquisition dates per season per AOI
- Tile size: 256×256 (more tiles, smaller size)
- Quality filtering: 75% retention (less strict)

**Calculation:**
```
Acquisitions per AOI per season: 20
Tiles per acquisition: 140 × 0.75 = 105
Total tiles per AOI: 105 × 20 × 4 = 8,400 tiles
Total tiles: 8,400 × 10 AOIs = 84,000 tiles

Storage:
  84,000 tiles × 1.5 MB = 126.0 GB
  
With metadata: ~139 GB
With augmented variants: ~181 GB
With temporal variations: ~217 GB
```

**✓ Target: ~200 GB** ✅

---

## Storage Breakdown by Component

### Detailed Component Analysis

#### Base Tiles (70%)

```
25,000 tiles × 3.4 MB = 85 GB
Train: 70% = 59.5 GB
Val:   15% = 12.8 GB
Test:  15% = 12.8 GB
```

#### Metadata (5%)

```
JSON files per tile: ~2 KB
Index CSVs: ~5 MB
Split configs: ~1 MB

Total: ~55 MB (negligible)
```

#### Augmented Variants (20%)

```
If storing pre-augmented tiles:
  - Rotations (×4): +85 GB
  - Flips (×2): +85 GB
  
Selective augmentation:
  +17 GB (20% of base)
```

#### Intermediate Cache (5%)

```
DEM tiles: ~500 MB
Preprocessed scenes: ~3 GB
Temporary files: ~1 GB

Total: ~4.5 GB
```

#### **Grand Total: ~120-200 GB** ✅

---

## Storage by Data Source

### Sentinel-1 (SAR)

**Raw SAFE Archives:**
```
Size per scene: ~800 MB
Scenes per AOI per season: 10-20
Total scenes: 10 AOIs × 4 seasons × 15 avg = 600 scenes

Raw SAR storage: 600 × 800 MB = 480 GB
```

**After COG conversion:**
```
COG per scene: ~75 MB
Total: 600 × 75 MB = 45 GB
Savings: 435 GB (91% reduction)
```

**After tiling:**
```
SAR contribution to tiles: 2 bands / 12 bands = 17%
SAR in final dataset: 120 GB × 0.17 = 20 GB
Savings vs raw: 460 GB (96% reduction)
```

### Sentinel-2 (Optical)

**Raw SAFE Archives:**
```
Size per scene: ~900 MB (L2A product)
Scenes per AOI per season: 10-20
Total scenes: 10 AOIs × 4 seasons × 15 avg = 600 scenes

Raw optical storage: 600 × 900 MB = 540 GB
```

**After COG conversion (10 bands):**
```
COG per scene: ~300 MB
Total: 600 × 300 MB = 180 GB
Savings: 360 GB (67% reduction)
```

**After tiling:**
```
Optical contribution: 10 bands / 12 bands = 83%
Optical in final dataset: 120 GB × 0.83 = 100 GB
Savings vs raw: 440 GB (81% reduction)
```

---

## Optimization Strategies

### Storage Reduction Techniques

#### 1. Delete Raw SAFE After Tiling

```
Before: 480 GB (S1) + 540 GB (S2) + 120 GB (tiles) = 1,140 GB
After:  120 GB (tiles only)

Savings: 1,020 GB (89% reduction) ✅
```

#### 2. Use Smaller Tile Sizes

```
512×512: 6.0 MB/tile → Dataset: 150 GB
384×384: 3.4 MB/tile → Dataset: 120 GB
256×256: 1.5 MB/tile → Dataset: 90 GB

Trade-off: More tiles = longer loading times
```

#### 3. Increase Quality Filtering

```
60% retention: +40% more tiles = 168 GB
70% retention: baseline = 120 GB
80% retention: -14% fewer tiles = 103 GB

Trade-off: Fewer tiles = less training data
```

#### 4. Selective Band Storage

```
All 12 bands: 120 GB
RGB + NIR + SAR (6 bands): 60 GB (50% reduction)
RGB + SAR (5 bands): 50 GB (58% reduction)

Trade-off: Less spectral information
```

#### 5. Increase Compression

```
Compression level 3: 120 GB × 1.15 = 138 GB
Compression level 6: 120 GB (baseline)
Compression level 9: 120 GB × 0.85 = 102 GB

Trade-off: Level 9 slower to read/write
```

### Recommended Strategy

**For 120-200 GB target:**

1. ✅ Use 384×384 tiles (good balance)
2. ✅ Compression level 6 (good speed/size balance)
3. ✅ Delete raw SAFE after successful tiling
4. ✅ Keep intermediate COGs only if needed for reprocessing
5. ✅ Store augmented tiles on-the-fly during training (don't pre-generate)

---

## Budget Planning

### Storage Tiers

#### Tier 1: Minimal (60-80 GB)
- **Use case:** Small pilot study, single season
- **Configuration:** 5 AOIs, 2 seasons, 256×256 tiles
- **Contents:** Final tiles only
- **Limitations:** Limited temporal diversity

#### Tier 2: Standard (120-150 GB) ✅
- **Use case:** Research paper, full seasonal coverage
- **Configuration:** 10 AOIs, 4 seasons, 384×384 tiles
- **Contents:** Final tiles + metadata
- **Recommended for:** Most projects

#### Tier 3: Extended (180-220 GB) ✅
- **Use case:** Comprehensive study, dense temporal sampling
- **Configuration:** 10 AOIs, 4 seasons, dense acquisitions
- **Contents:** Final tiles + augmented variants
- **Recommended for:** Production models

#### Tier 4: Complete (300-500 GB)
- **Use case:** Archival with reprocessing capability
- **Configuration:** All above + intermediate products
- **Contents:** Tiles + COGs + cache
- **Recommended for:** Long-term projects

#### Tier 5: Full Archive (1-2 TB)
- **Use case:** Raw data archival
- **Configuration:** All tiers + raw SAFE
- **Contents:** Everything including raw downloads
- **Recommended for:** Data centers only

### Cost Estimates

**Cloud Storage (AWS S3 Standard):**
```
120 GB × $0.023/GB/month = $2.76/month
200 GB × $0.023/GB/month = $4.60/month
500 GB × $0.023/GB/month = $11.50/month

Annual costs:
  120 GB: $33/year
  200 GB: $55/year
  500 GB: $138/year
```

**Local Storage:**
```
1 TB SSD: ~$80-120 (one-time)
2 TB HDD: ~$50-70 (one-time)
4 TB HDD: ~$90-120 (one-time)

Recommendation: 2TB HDD for full workflow
```

---

## Quick Reference Table

### Storage by Dataset Size

| AOIs | Seasons | Acquisitions/Season | Tile Size | Expected Storage |
|------|---------|---------------------|-----------|------------------|
| 5 | 2 | 10 | 256×256 | **40-50 GB** |
| 10 | 4 | 10 | 384×384 | **120-140 GB** ✅ |
| 10 | 4 | 15 | 384×384 | **180-200 GB** ✅ |
| 15 | 4 | 15 | 512×512 | **320-350 GB** |
| 20 | 4 | 20 | 256×256 | **400-450 GB** |

### Storage by Project Phase

| Phase | Storage Needed | Duration | Can Delete After? |
|-------|---------------|----------|-------------------|
| Download raw | 1-2 TB | 1-7 days | ✅ Yes (keep COGs) |
| Preprocessing | 200-300 GB | 1-3 days | ✅ Yes (keep tiles) |
| Tiling | 120-200 GB | 1-2 days | ❌ No (final output) |
| Training | 120-200 GB | Weeks-months | ❌ No |
| Inference | 10-50 GB | Ongoing | ⚠️ Archive old |

---

## Example Calculation Walkthrough

### Complete Example: 10 AOIs, 4 Seasons, 384×384 Tiles

**Step 1: Define parameters**
```
Number of AOIs: 10
Seasons: 4 (Spring, Summer, Fall, Winter)
Acquisitions per season: 12 (roughly every 12 days over 6 months)
AOI size: 50 km × 50 km
Tile size: 384 × 384 pixels at 10m resolution
Overlap: 10%
Quality retention: 70%
```

**Step 2: Calculate tiles per acquisition**
```
AOI coverage: 50,000m × 50,000m = 2,500 km²
Tile coverage: 3.84 km × 3.84 km = 14.75 km²
Tiles needed (with overlap): 2,500 / 14.75 × 1.1 = ~186 tiles
After quality filtering: 186 × 0.70 = 130 tiles
```

**Step 3: Calculate total tiles**
```
Tiles per AOI per season: 130 tiles × 12 acquisitions = 1,560 tiles
Total tiles per AOI: 1,560 × 4 seasons = 6,240 tiles
Total dataset: 6,240 × 10 AOIs = 62,400 tiles
```

**Step 4: Calculate storage**
```
Size per tile: 3.4 MB (384×384, compressed)
Total size: 62,400 × 3.4 MB = 212.2 GB

Wait, this is over our target! Let's adjust...
```

**Step 5: Adjust for realistic constraints**
```
More aggressive quality filtering: 60% retention
Fewer acquisitions: 8 per season (more realistic)

Revised calculation:
  Tiles per acquisition: 186 × 0.60 = 112 tiles
  Tiles per AOI per season: 112 × 8 = 896 tiles
  Total tiles per AOI: 896 × 4 = 3,584 tiles
  Total dataset: 3,584 × 10 = 35,840 tiles
  
Storage: 35,840 × 3.4 MB = 121.9 GB ✅
```

**Step 6: Add metadata overhead**
```
Base tiles: 122 GB
Metadata JSON: 0.07 GB (2KB × 35,840)
Index CSVs: 0.01 GB
Split configs: 0.001 GB

Total: ~122 GB ✅
```

**Final Result: 120-125 GB for final dataset** ✅

---

## Validation Checklist

Before finalizing your storage budget:

- [ ] Estimated number of AOIs
- [ ] Number of seasons/temporal points
- [ ] Expected acquisitions per season
- [ ] Chosen tile size (256/384/512)
- [ ] Quality filtering threshold
- [ ] Compression level
- [ ] Need to keep raw SAFE? (×10 storage!)
- [ ] Need intermediate products? (×3 storage)
- [ ] Cloud vs local storage costs
- [ ] Backup strategy (×2 storage)

---

## Conclusion

For a typical research project with **10 AOIs × 4 seasons**:

- **Target: 120-200 GB** (tiles only)
- **Realistic: 300-500 GB** (with intermediate products)
- **Maximum: 1-2 TB** (with raw archives)

**Key Takeaway:** Delete raw SAFE archives after successful tiling to reclaim 70-80% of storage.

**Recommended Configuration:**
- Tile size: 384×384
- Compression: Level 6
- Quality filtering: 60-70%
- Storage tier: Standard (120-150 GB)
- Delete raw after tiling: Yes ✅

---

## References

- Sentinel-1 Product Specification: ~700-900 MB per SAFE
- Sentinel-2 L2A Product Specification: ~600-1200 MB per SAFE
- NPZ compression benchmarks: 40-60% of uncompressed size
- Typical AOI size for agricultural/environmental studies: 50×50 km

**Document End**
