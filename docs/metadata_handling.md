# Metadata Handling for TerraMind Pipeline

## Overview

This document explains how metadata (month, incidence angle, etc.) is handled throughout the Axion-Sat pipeline and prepared for Stage 3 conditional models.

## Key Finding: TerraMind Stage 1 Limitation

**TerraMind 1.0's pretrained architecture does NOT support custom metadata tokens** as conditioning inputs. The model was trained on standard modality inputs (S1GRD, S2L2A, etc.) without metadata conditioning.

**Solution**: Collect and preserve metadata for Stage 3, where it will be used for conditional generation.

## Metadata Types

### Temporal Metadata
- **Month**: 1-12, captures seasonal patterns
- **Day of Year**: 1-365, finer temporal resolution
- **Year**: Useful for multi-year datasets

### Geometric Metadata (SAR-specific)
- **Incidence Angle**: 20-45 degrees, affects SAR backscatter
- **Azimuth Angle**: 0-360 degrees, viewing direction  
- **Orbit Direction**: ascending/descending

### Environmental Metadata
- **Biome Code**: Land cover/ecosystem type
- **Elevation**: Mean elevation in meters
- **Slope**: Terrain slope in degrees
- **Aspect**: Terrain aspect in degrees

### Quality Metadata
- **Cloud Coverage**: 0-100%, for optical imagery
- **Data Completeness**: 0-100%, percentage of valid pixels

## Pipeline Stages

### Stage 1: TerraMind SAR-to-Optical

**Status**: ❌ Metadata NOT used as model input  
**Reason**: TerraMind 1.0 pretrained weights don't include metadata tokens  
**Action**: Metadata is stored in tiles but not fed to model

```python
# Stage 1 Training (current)
model = build_terramind_generator(
    input_modalities=("S1GRD",),
    output_modalities=("S2L2A",),
    timesteps=12
)

# Metadata is in tile NPZ but not used
tile_data = np.load('tiles/tile001.npz')
# tile_data contains: s1_vv, s1_vh, s2_*, month, inc_angle
# but only S1 is fed to model
```

### Stage 2: Optional Refinement

Metadata preserved but typically not used in intermediate refinement stages.

### Stage 3: Conditional Model (Future)

**Status**: ✅ Metadata WILL BE used for conditioning  
**Method**: Metadata embedding concatenated with model features  
**Benefits**: Season-aware, geometry-aware predictions

```python
# Stage 3 (future implementation)
from axs_lib.metadata import MetadataManager

manager = MetadataManager()
metadata = manager.extract_from_tile('tiles/tile001.npz')

# Create conditioning vector
meta_vector = manager.create_embedding_vector(metadata)
# Shape: (5,) with [month_sin, month_cos, inc_angle_norm, elev_norm, slope_norm]

# Feed to conditional model
prediction = conditional_model(
    features=stage2_output,
    metadata=meta_vector
)
```

## Usage

### Check Metadata in Tiles

```bash
# View metadata summary
python axs_lib/metadata.py --tile-dir tiles/ --summary
```

Output:
```json
{
  "n_tiles": 150,
  "month": {
    "available": 150,
    "min": 1,
    "max": 12,
    "mean": 6.5,
    "unique": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  },
  "incidence_angle": {
    "available": 150,
    "min": 29.3,
    "max": 45.8,
    "mean": 37.6,
    "std": 4.2
  }
}
```

### Extract Single Tile Metadata

```bash
# Extract from one tile
python axs_lib/metadata.py --tile tiles/tile001.npz
```

Output:
```json
{
  "tile_id": "tile001",
  "tile_path": "tiles/tile001.npz",
  "month": 7,
  "incidence_angle": 38.5,
  "biome_code": 4,
  "mean_elevation": 324.5,
  "mean_slope": 5.2,
  "mean_aspect": 135.7
}
```

### Prepare Metadata for Stage 3

```bash
# Extract and save all metadata for Stage 3
python axs_lib/metadata.py \
  --tile-dir tiles/ \
  --output-dir stage3_metadata/ \
  --prepare \
  --include-arrays
```

This creates:
```
stage3_metadata/
├── tile001_metadata.json
├── tile002_metadata.json
├── ...
└── metadata_summary.json
```

### Programmatic Usage

```python
from axs_lib.metadata import MetadataManager

# Initialize manager
manager = MetadataManager(verbose=True)

# Extract from tile
metadata = manager.extract_from_tile('tiles/tile001.npz')

print(f"Month: {metadata.month}")
print(f"Incidence Angle: {metadata.incidence_angle}°")

# Create embedding for Stage 3
embedding = manager.create_embedding_vector(metadata, normalize=True)
print(f"Embedding shape: {embedding.shape}")  # (5,)

# Save for Stage 3
manager.save_for_stage3(metadata, 'stage3_meta/tile001_meta.json')
```

## Metadata Embedding for Stage 3

The `create_embedding_vector()` method converts metadata into a numerical vector suitable for model conditioning:

### Features (5-dimensional vector)

1. **month_sin**: sin((month-1) * 2π/12) - Cyclical month encoding
2. **month_cos**: cos((month-1) * 2π/12) - Cyclical month encoding
3. **incidence_angle_norm**: (angle - 20) / 25 - Normalized to [0,1]
4. **elevation_norm**: elevation / 3000 - Normalized elevation
5. **slope_norm**: slope / 45 - Normalized slope

### Why Cyclical Encoding for Month?

Using sin/cos preserves the cyclical nature of months (December is close to January):
- Month 12 (December): sin ≈ 0, cos ≈ 1
- Month 1 (January): sin ≈ 0, cos ≈ 1
- Month 6 (June): sin ≈ 0, cos ≈ -1

Linear encoding (1, 2, 3, ..., 12) would incorrectly suggest December and January are far apart.

## Data Collection

Metadata is collected during tile generation:

```python
from axs_lib.data import create_tiles_from_files

tiles = create_tiles_from_files(
    s1_vv='data/s1_vv.tif',
    s1_vh='data/s1_vh.tif',
    s2_b2='data/s2_b2.tif',
    s2_b3='data/s2_b3.tif',
    s2_b4='data/s2_b4.tif',
    s2_b8='data/s2_b8.tif',
    month=7,  # July
    inc_angle='data/s1_inc_angle.tif',
    dem_slope='data/slope.tif',
    tile_size=256,
    output_dir='tiles/'
)
```

Each tile NPZ contains:
- `s1_vv`, `s1_vh`: SAR data (used in Stage 1)
- `s2_b2`, `s2_b3`, `s2_b4`, `s2_b8`: Optical data (target)
- `month`: Constant or raster (saved for Stage 3)
- `inc_angle`: SAR incidence angle raster (saved for Stage 3)
- `dem_slope`, `dem_aspect`: Terrain (saved for Stage 3)

## Why This Matters

### Without Metadata Conditioning (Current Stage 1)
- Model generates "generic" optical imagery
- Doesn't account for seasonal variations
- Ignores SAR geometry effects
- May produce inconsistent results across seasons/geometries

### With Metadata Conditioning (Future Stage 3)
- **Season-aware**: Different vegetation in summer vs. winter
- **Geometry-aware**: Accounts for incidence angle effects on SAR
- **Context-aware**: Uses terrain and biome information
- **Consistent**: Produces physically plausible results

## Stage 3 Integration Plan

When implementing Stage 3 conditional models:

```python
# Pseudo-code for Stage 3 model
class Stage3ConditionalModel(nn.Module):
    def __init__(self, feature_dim=256, metadata_dim=5):
        self.feature_encoder = ...
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        self.decoder = ...
    
    def forward(self, features, metadata_vector):
        # Encode features from Stage 1/2
        feat = self.feature_encoder(features)
        
        # Encode metadata
        meta = self.metadata_encoder(metadata_vector)
        
        # Concatenate and decode
        combined = torch.cat([feat, meta], dim=1)
        output = self.decoder(combined)
        
        return output
```

## Best Practices

1. **Always collect metadata** during tile generation
2. **Document metadata ranges** (min/max month, typical incidence angles)
3. **Validate metadata** before Stage 3 training
4. **Normalize consistently** using same ranges across train/val/test
5. **Handle missing values** gracefully (use defaults or masking)

## Troubleshooting

### No metadata in tiles

**Problem**: Tiles don't contain month or incidence_angle keys

**Solution**: Regenerate tiles with metadata:
```bash
python axs_lib/data.py \
  --s1-vv data/s1_vv.tif \
  --s1-vh data/s1_vh.tif \
  --month 7 \
  --inc-angle data/inc_angle.tif \
  --output-dir tiles/
```

### Metadata values seem wrong

**Problem**: Month is 0 or incidence angle is 90°

**Solution**: Check source data and units:
- Month should be 1-12, not 0-11
- Incidence angle should be 20-45° for Sentinel-1
- Check raster data with `gdalinfo`

### Stage 1 not using metadata

**Status**: This is EXPECTED BEHAVIOR  
**Reason**: TerraMind 1.0 doesn't support metadata tokens  
**Action**: No action needed - metadata saved for Stage 3

## Future Enhancements

### Stage 3 Model Architecture Options

1. **FiLM (Feature-wise Linear Modulation)**
   - Use metadata to modulate feature maps
   - Proven effective for conditional generation

2. **Cross-Attention**
   - Metadata as key/value, features as query
   - Allows model to attend to relevant metadata

3. **Concatenation (simplest)**
   - Directly concatenate metadata embedding
   - Works well for low-dimensional metadata

### Additional Metadata

Consider collecting for Stage 3:
- **Sun angle** (solar elevation/azimuth)
- **Orbit number** (for temporal ordering)
- **Processing baseline** (for change detection)
- **Weather conditions** (if available)

## See Also

- `axs_lib/metadata.py` - Metadata handling implementation
- `axs_lib/data.py` - Tile generation with metadata
- `docs/early_stopping.md` - Training optimization
- `docs/panel_visualization.md` - Visualization tools

## References

- FiLM: Visual Reasoning with Feature-wise Linear Modulation (Perez et al., 2018)
- Conditional GANs (Mirza & Osindero, 2014)
- TerraMind Technical Report (IBM/ESA, 2024)
