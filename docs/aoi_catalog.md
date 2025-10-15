# AOI Catalog - Starter Training Locations

A curated collection of diverse Areas of Interest (AOIs) for building satellite imagery training datasets. Each AOI includes seasonal dates spanning all four meteorological seasons and sample bounding boxes for different scales.

## Table of Contents
- [AOI Overview](#aoi-overview)
- [Spatial Split Strategy](#spatial-split-strategy)
- [AOI Catalog](#aoi-catalog)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)

---

## AOI Overview

This catalog provides 10 geographically diverse locations representing different:
- **Biomes**: Tropical forests, savannas, temperate forests, croplands, urban areas, wetlands
- **Continents**: Africa, Asia, Europe, North America, South America, Australia
- **Land Cover Types**: Agriculture, forest, grassland, urban, water bodies
- **Topography**: Coastal, inland, mountainous, flat plains

### Seasonal Dates

For each AOI, four representative dates are provided:
- **DJF** (Dec-Jan-Feb): Boreal winter / Austral summer
- **MAM** (Mar-Apr-May): Boreal spring / Austral autumn
- **JJA** (Jun-Jul-Aug): Boreal summer / Austral winter
- **SON** (Sep-Oct-Nov): Boreal autumn / Austral spring

---

## Spatial Split Strategy

### Why Spatial Splitting?

When training geospatial models, **random pixel-level or tile-level splits are insufficient** because:
1. **Spatial Autocorrelation**: Nearby pixels are highly correlated
2. **Data Leakage**: Adjacent tiles may appear in both train and test sets
3. **Overfitted Performance**: Models memorize local patterns instead of generalizing

### Recommended Approach: Spatially Disjoint Splits

**Definition**: Training, validation, and test sets are separated geographically with no spatial overlap.

```
┌─────────────────────────────────────┐
│         Full Study Area             │
│                                     │
│  ┌────────┐    ┌──────┐  ┌──────┐ │
│  │ Train  │    │ Val  │  │ Test │ │
│  │        │    │      │  │      │ │
│  │        │    │      │  │      │ │
│  └────────┘    └──────┘  └──────┘ │
│                                     │
│     (Geographically separated)     │
└─────────────────────────────────────┘
```

### Implementation Strategies

#### Strategy 1: Regional Blocks (Recommended)
Split large AOIs into non-overlapping geographic regions:
```
Lake Victoria (36,000 km²)
├── Train:      Eastern region (Kenya)
├── Validation: Southern region (Tanzania)
└── Test:       Northern region (Uganda)
```

#### Strategy 2: Multiple AOIs
Use completely different geographic locations:
```
Train:      AOI 1, 2, 3, 5, 7, 9     (6 locations)
Validation: AOI 4, 8                 (2 locations)
Test:       AOI 6, 10                (2 locations)
```

#### Strategy 3: Temporal + Spatial
Combine temporal and spatial separation:
```
Amazon Basin
├── Train:      Western region, all seasons
├── Validation: Central region, JJA + SON only
└── Test:       Eastern region, DJF + MAM only
```

### Validation Metrics

For spatially disjoint splits, expect:
- **Lower validation accuracy** (10-20% drop vs random split)
- **More realistic generalization** to new areas
- **Better domain adaptation** performance

---

## AOI Catalog

### 1. Lake Victoria Basin (East Africa)

**Location**: Kenya, Tanzania, Uganda  
**Biome**: Tropical freshwater lake with agricultural surroundings  
**Land Cover**: Water, wetlands, croplands, settlements  
**Area**: ~36,000 km² (lake + buffer)

**Bounding Boxes**:
```python
# Full basin
bbox_full = [31.5, -1.5, 34.5, 0.5]

# Regional splits for training
bbox_train_east = [33.5, -1.5, 34.5, 0.5]      # Kenya side
bbox_val_south = [31.5, -3.0, 33.5, -1.5]      # Tanzania side
bbox_test_north = [31.5, 0.0, 33.5, 0.5]       # Uganda side
```

**Seasonal Dates** (2024):
- **DJF**: 2024-01-15 (dry season)
- **MAM**: 2024-04-15 (long rains)
- **JJA**: 2024-07-15 (dry season)
- **SON**: 2024-10-15 (short rains)

**Use Cases**: Water quality monitoring, agricultural productivity, wetland mapping

---

### 2. Amazon Rainforest (Brazil)

**Location**: Pará, Brazil (Tapajós River region)  
**Biome**: Tropical moist broadleaf forest  
**Land Cover**: Primary forest, secondary forest, deforestation, rivers  
**Area**: ~25,000 km²

**Bounding Boxes**:
```python
# Full region
bbox_full = [-56.0, -4.5, -54.0, -2.5]

# Regional splits
bbox_train_west = [-56.0, -4.5, -55.0, -2.5]   # Western forest
bbox_val_central = [-55.0, -4.5, -54.5, -2.5]  # Central mixed
bbox_test_east = [-54.5, -4.5, -54.0, -2.5]    # Eastern frontier
```

**Seasonal Dates** (2023):
- **DJF**: 2023-12-15 (wet season)
- **MAM**: 2023-03-15 (wet season)
- **JJA**: 2023-06-15 (dry season)
- **SON**: 2023-09-15 (dry season)

**Use Cases**: Deforestation detection, forest health, carbon stock estimation

---

### 3. US Midwest Croplands (Iowa)

**Location**: Central Iowa, USA  
**Biome**: Temperate grasslands converted to agriculture  
**Land Cover**: Corn, soybean rotation, some pasture  
**Area**: ~15,000 km²

**Bounding Boxes**:
```python
# Full region
bbox_full = [-93.5, 41.5, -91.5, 42.5]

# Regional splits
bbox_train_west = [-93.5, 41.5, -92.5, 42.5]   # Western farms
bbox_val_central = [-92.5, 41.5, -92.0, 42.5]  # Central
bbox_test_east = [-92.0, 41.5, -91.5, 42.5]    # Eastern farms
```

**Seasonal Dates** (2023):
- **DJF**: 2023-01-15 (winter dormant)
- **MAM**: 2023-05-15 (planting/emergence)
- **JJA**: 2023-07-15 (peak vegetation)
- **SON**: 2023-10-15 (harvest/senescence)

**Use Cases**: Crop type classification, yield prediction, phenology monitoring

---

### 4. Sahel Region (Niger)

**Location**: Southern Niger  
**Biome**: Semi-arid savanna/grassland  
**Land Cover**: Sparse vegetation, bare soil, seasonal agriculture  
**Area**: ~20,000 km²

**Bounding Boxes**:
```python
# Full region
bbox_full = [2.0, 13.0, 4.0, 14.0]

# Regional splits
bbox_train_west = [2.0, 13.0, 2.7, 14.0]       # Western Sahel
bbox_val_central = [2.7, 13.0, 3.3, 14.0]      # Central
bbox_test_east = [3.3, 13.0, 4.0, 14.0]        # Eastern
```

**Seasonal Dates** (2023):
- **DJF**: 2023-01-15 (dry season)
- **MAM**: 2023-04-15 (pre-monsoon)
- **JJA**: 2023-07-15 (wet season peak)
- **SON**: 2023-10-15 (end of rains)

**Use Cases**: Drought monitoring, food security, vegetation greening trends

---

### 5. Mekong Delta (Vietnam)

**Location**: Southern Vietnam  
**Biome**: Tropical river delta and wetlands  
**Land Cover**: Rice paddies, aquaculture, mangroves, rivers  
**Area**: ~18,000 km²

**Bounding Boxes**:
```python
# Full region
bbox_full = [105.5, 9.5, 106.5, 10.5]

# Regional splits
bbox_train_north = [105.5, 10.0, 106.5, 10.5]  # Northern delta
bbox_val_central = [105.5, 9.75, 106.5, 10.0]  # Central
bbox_test_south = [105.5, 9.5, 106.5, 9.75]    # Southern coast
```

**Seasonal Dates** (2023):
- **DJF**: 2023-01-15 (dry season, winter crop)
- **MAM**: 2023-04-15 (transition, harvest)
- **JJA**: 2023-07-15 (wet season, summer crop)
- **SON**: 2023-10-15 (peak flooding)

**Use Cases**: Flood mapping, rice monitoring, aquaculture detection

---

### 6. Australian Wheat Belt (Western Australia)

**Location**: Western Australia  
**Biome**: Mediterranean climate agriculture  
**Land Cover**: Wheat, barley, canola, sheep grazing  
**Area**: ~22,000 km²

**Bounding Boxes**:
```python
# Full region
bbox_full = [117.0, -32.0, 119.0, -31.0]

# Regional splits
bbox_train_west = [117.0, -32.0, 117.7, -31.0]  # Western farms
bbox_val_central = [117.7, -32.0, 118.3, -31.0] # Central
bbox_test_east = [118.3, -32.0, 119.0, -31.0]   # Eastern farms
```

**Seasonal Dates** (2023):
- **DJF**: 2023-12-15 (summer dry, harvest complete)
- **MAM**: 2023-05-15 (autumn planting)
- **JJA**: 2023-07-15 (winter growth)
- **SON**: 2023-10-15 (spring pre-harvest)

**Use Cases**: Crop yield forecasting, water use efficiency, soil moisture

---

### 7. European Mixed Forest (Germany)

**Location**: Bavaria, Germany  
**Biome**: Temperate mixed deciduous/coniferous forest  
**Land Cover**: Forest, agriculture, small towns  
**Area**: ~12,000 km²

**Bounding Boxes**:
```python
# Full region
bbox_full = [11.0, 48.0, 12.5, 49.0]

# Regional splits
bbox_train_west = [11.0, 48.0, 11.5, 49.0]     # Western forests
bbox_val_central = [11.5, 48.0, 12.0, 49.0]    # Central mixed
bbox_test_east = [12.0, 48.0, 12.5, 49.0]      # Eastern
```

**Seasonal Dates** (2023):
- **DJF**: 2023-01-15 (winter dormant)
- **MAM**: 2023-04-15 (spring leaf-out)
- **JJA**: 2023-07-15 (summer peak green)
- **SON**: 2023-10-15 (autumn senescence)

**Use Cases**: Forest health, tree species classification, phenology

---

### 8. Nairobi Urban Expansion (Kenya)

**Location**: Nairobi and surroundings, Kenya  
**Biome**: Urban/peri-urban with savanna background  
**Land Cover**: Dense urban, suburbs, agriculture, grassland  
**Area**: ~8,000 km²

**Bounding Boxes**:
```python
# Full region
bbox_full = [36.6, -1.5, 37.1, -1.1]

# Regional splits
bbox_train_urban = [36.75, -1.35, 37.1, -1.1]   # Core urban
bbox_val_west = [36.6, -1.35, 36.75, -1.1]      # Western suburbs
bbox_test_south = [36.75, -1.5, 37.1, -1.35]    # Southern sprawl
```

**Seasonal Dates** (2024):
- **DJF**: 2024-01-15 (dry season)
- **MAM**: 2024-04-15 (long rains)
- **JJA**: 2024-07-15 (dry/cool)
- **SON**: 2024-10-15 (short rains)

**Use Cases**: Urban growth mapping, informal settlement detection, green space

---

### 9. Great Plains Grasslands (Kansas/Nebraska)

**Location**: Central Great Plains, USA  
**Biome**: Temperate grasslands and rangelands  
**Land Cover**: Pasture, rangeland, some cropland  
**Area**: ~16,000 km²

**Bounding Boxes**:
```python
# Full region
bbox_full = [-99.5, 39.5, -97.5, 40.5]

# Regional splits
bbox_train_west = [-99.5, 39.5, -98.5, 40.5]   # Western rangeland
bbox_val_central = [-98.5, 39.5, -98.0, 40.5]  # Central
bbox_test_east = [-98.0, 39.5, -97.5, 40.5]    # Eastern mixed
```

**Seasonal Dates** (2023):
- **DJF**: 2023-01-15 (winter dormant)
- **MAM**: 2023-05-15 (spring green-up)
- **JJA**: 2023-07-15 (peak biomass)
- **SON**: 2023-10-15 (autumn senescence)

**Use Cases**: Grassland productivity, grazing management, soil carbon

---

### 10. Okavango Delta (Botswana)

**Location**: Northern Botswana  
**Biome**: Seasonal inland delta and floodplains  
**Land Cover**: Wetlands, seasonal flooding, grasslands, forest patches  
**Area**: ~14,000 km²

**Bounding Boxes**:
```python
# Full region
bbox_full = [22.5, -19.5, 23.5, -18.5]

# Regional splits
bbox_train_north = [22.5, -19.0, 23.5, -18.5]  # Northern delta
bbox_val_central = [22.5, -19.3, 23.5, -19.0]  # Central channels
bbox_test_south = [22.5, -19.5, 23.5, -19.3]   # Southern floodplain
```

**Seasonal Dates** (2023):
- **DJF**: 2023-01-15 (wet season, low flood)
- **MAM**: 2023-04-15 (flood arrival)
- **JJA**: 2023-07-15 (peak flood)
- **SON**: 2023-10-15 (receding flood)

**Use Cases**: Flood extent mapping, wetland monitoring, wildlife habitat

---

## Usage Examples

### Example 1: Single AOI with Spatial Splits

```bash
# Lake Victoria - Train split (Eastern Kenya side)
python scripts/build_tiles.py \
    --bbox 33.5 -1.5 34.5 0.5 \
    --date 2024-01-15 \
    --tile-size 256 \
    --overlap 0 \
    --max-cloud 30 \
    --output-dir data/tiles/lake_victoria/train \
    --split-ratio 1.0 0.0 0.0

# Lake Victoria - Validation split (Southern Tanzania side)
python scripts/build_tiles.py \
    --bbox 31.5 -3.0 33.5 -1.5 \
    --date 2024-01-15 \
    --tile-size 256 \
    --overlap 0 \
    --max-cloud 30 \
    --output-dir data/tiles/lake_victoria/val \
    --split-ratio 1.0 0.0 0.0

# Lake Victoria - Test split (Northern Uganda side)
python scripts/build_tiles.py \
    --bbox 31.5 0.0 33.5 0.5 \
    --date 2024-01-15 \
    --tile-size 256 \
    --overlap 0 \
    --max-cloud 30 \
    --output-dir data/tiles/lake_victoria/test \
    --split-ratio 1.0 0.0 0.0
```

### Example 2: Multiple AOIs for Global Model

```bash
# Training: 6 diverse AOIs
for aoi in lake_victoria amazon iowa_crops sahel mekong australia; do
    python scripts/build_tiles.py \
        --bbox <from catalog> \
        --date 2023-07-15 \
        --tile-size 256 \
        --output-dir data/tiles/global_train/${aoi} \
        --split-ratio 1.0 0.0 0.0
done

# Validation: 2 held-out AOIs
for aoi in germany nairobi; do
    python scripts/build_tiles.py \
        --bbox <from catalog> \
        --date 2023-07-15 \
        --tile-size 256 \
        --output-dir data/tiles/global_val/${aoi} \
        --split-ratio 1.0 0.0 0.0
done

# Test: 2 completely unseen AOIs
for aoi in great_plains okavango; do
    python scripts/build_tiles.py \
        --bbox <from catalog> \
        --date 2023-07-15 \
        --tile-size 256 \
        --output-dir data/tiles/global_test/${aoi} \
        --split-ratio 1.0 0.0 0.0
done
```

### Example 3: All Seasons for Single AOI

```bash
# Iowa crops - All 4 seasons for training
dates=("2023-01-15" "2023-05-15" "2023-07-15" "2023-10-15")
seasons=("DJF" "MAM" "JJA" "SON")

for i in {0..3}; do
    python scripts/build_tiles.py \
        --bbox -93.5 41.5 -91.5 42.5 \
        --date ${dates[$i]} \
        --tile-size 256 \
        --output-dir data/tiles/iowa_temporal/train_${seasons[$i]} \
        --split-ratio 0.8 0.1 0.1
done
```

### Example 4: Batch Processing with PowerShell

```powershell
# Define AOIs and dates
$aois = @{
    "lake_victoria" = @{bbox="31.5,-1.5,34.5,0.5"; dates=@("2024-01-15","2024-04-15","2024-07-15","2024-10-15")}
    "amazon" = @{bbox="-56.0,-4.5,-54.0,-2.5"; dates=@("2023-12-15","2023-03-15","2023-06-15","2023-09-15")}
    "iowa" = @{bbox="-93.5,41.5,-91.5,42.5"; dates=@("2023-01-15","2023-05-15","2023-07-15","2023-10-15")}
}

# Process all AOIs
foreach ($aoi_name in $aois.Keys) {
    $config = $aois[$aoi_name]
    foreach ($date in $config.dates) {
        Write-Host "Processing $aoi_name - $date"
        python scripts/build_tiles.py `
            --bbox $config.bbox.Split(",") `
            --date $date `
            --tile-size 256 `
            --max-cloud 30 `
            --output-dir "data/tiles/${aoi_name}" `
            --split-ratio 0.7 0.15 0.15
    }
}
```

---

## Best Practices

### 1. Spatial Split Design

**DO**:
- ✅ Separate train/val/test by at least 50-100 km
- ✅ Ensure each split has similar land cover diversity
- ✅ Document split boundaries in metadata
- ✅ Test on completely unseen geographic regions

**DON'T**:
- ❌ Split adjacent tiles randomly
- ❌ Use the same location with different dates as test set
- ❌ Overlap train and test regions at all
- ❌ Assume temporal differences alone prevent overfitting

### 2. AOI Selection Strategy

For a robust dataset:
1. **Diversity**: Include 6-10 different biomes/land covers
2. **Scale**: Mix small (~5,000 km²) and large (~30,000 km²) AOIs
3. **Balance**: ~60% train, ~20% val, ~20% test by area
4. **Seasons**: Include 2-4 dates per AOI when possible

### 3. Quality Control

Before training:
```python
# Check spatial overlap
import geopandas as gpd
from shapely.geometry import box

train_bbox = box(33.5, -1.5, 34.5, 0.5)
test_bbox = box(31.5, 0.0, 33.5, 0.5)

assert not train_bbox.intersects(test_bbox), "Train/test overlap!"

# Check tile count balance
import pandas as pd
train_df = pd.read_csv('data/index/train_tiles.csv')
test_df = pd.read_csv('data/index/test_tiles.csv')

print(f"Train: {len(train_df)} tiles")
print(f"Test: {len(test_df)} tiles")
print(f"Ratio: {len(train_df)/len(test_df):.1f}:1")
```

### 4. Combining Splits

After generating tiles separately:

```python
import pandas as pd
from pathlib import Path

# Load individual splits
train_df = pd.read_csv('data/tiles/lake_victoria/train/index.csv')
val_df = pd.read_csv('data/tiles/lake_victoria/val/index.csv')
test_df = pd.read_csv('data/tiles/lake_victoria/test/index.csv')

# Add split column
train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'

# Combine
combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
combined_df.to_csv('data/index/lake_victoria_spatially_split.csv', index=False)

print(f"Total tiles: {len(combined_df)}")
print(combined_df['split'].value_counts())
```

### 5. Documentation

Always document your split strategy:

```yaml
# dataset_metadata.yaml
name: "Lake Victoria Multi-Temporal Dataset"
version: "1.0"
date_created: "2024-10-13"

spatial_split_strategy: "Geographic regions"
splits:
  train:
    region: "Eastern basin (Kenya)"
    bbox: [33.5, -1.5, 34.5, 0.5]
    tiles: 1247
  validation:
    region: "Southern basin (Tanzania)"
    bbox: [31.5, -3.0, 33.5, -1.5]
    tiles: 318
  test:
    region: "Northern basin (Uganda)"
    bbox: [31.5, 0.0, 33.5, 0.5]
    tiles: 289

temporal_coverage:
  - "2024-01-15"  # DJF
  - "2024-04-15"  # MAM
  - "2024-07-15"  # JJA
  - "2024-10-15"  # SON
```

---

## References

### Scientific Basis
- **Roberts et al. (2017)**: "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure"
- **Meyer et al. (2018)**: "Importance of spatial predictor variable selection in machine learning applications"
- **Ploton et al. (2020)**: "Spatial validation reveals poor predictive performance of large-scale ecological mapping models"

### Tools
- [MLDATA Toolkit](https://github.com/environmentalinformatics-marburg/mldata) - Spatial cross-validation
- [BlockCV](https://github.com/rvalavi/blockCV) - Spatial and environmental blocking
- [sktime](https://www.sktime.net/) - Time series cross-validation

### Best Practices
- [Kaggle: Avoiding Leakage in Geospatial Data](https://www.kaggle.com/discussions/getting-started/203487)
- [Google Earth Engine: Large-Scale ML Best Practices](https://developers.google.com/earth-engine/guides/ml_intro)

---

## Quick Reference Table

| AOI | Continent | Biome | Area (km²) | Seasons | Primary Use Case |
|-----|-----------|-------|------------|---------|------------------|
| Lake Victoria | Africa | Tropical lake | 36,000 | All | Water/agriculture |
| Amazon | S. America | Rainforest | 25,000 | Wet/Dry | Deforestation |
| Iowa | N. America | Cropland | 15,000 | All | Crop monitoring |
| Sahel | Africa | Semi-arid | 20,000 | Wet/Dry | Drought |
| Mekong | Asia | River delta | 18,000 | All | Rice/flooding |
| Australia | Oceania | Med. agriculture | 22,000 | All | Wheat yield |
| Germany | Europe | Mixed forest | 12,000 | All | Forest health |
| Nairobi | Africa | Urban/savanna | 8,000 | All | Urban growth |
| Great Plains | N. America | Grassland | 16,000 | All | Rangeland |
| Okavango | Africa | Wetland | 14,000 | All | Flood mapping |

**Total Coverage**: ~186,000 km²  
**Global Distribution**: 6 continents, 10 biomes  
**Recommended Split**: 6 AOIs train, 2 AOIs val, 2 AOIs test

---

*Last Updated: 2025-10-13*  
*Coordinate System: All bboxes in EPSG:4326 (WGS84)*  
*Date Format: YYYY-MM-DD*
