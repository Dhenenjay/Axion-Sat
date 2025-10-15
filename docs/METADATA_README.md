# Metadata Handling - Quick Reference

## TL;DR

**TerraMind Stage 1 does NOT support metadata tokens** (month, incidence angle).  
**Solution**: Metadata is collected and saved for Stage 3 conditional models.

## Status by Stage

| Stage | Metadata Usage | Status |
|-------|---------------|--------|
| **Stage 1** (TerraMind SAR→Optical) | ❌ NOT used | Stored in tiles |
| **Stage 2** (Optional Refinement) | ❌ NOT used | Preserved |
| **Stage 3** (Conditional Model) | ✅ WILL BE used | For conditioning |

## Quick Commands

### Check metadata in your tiles
```bash
python axs_lib/metadata.py --tile-dir tiles/ --summary
```

### Extract metadata from one tile
```bash
python axs_lib/metadata.py --tile tiles/tile001.npz
```

### Prepare all metadata for Stage 3
```bash
python axs_lib/metadata.py \
  --tile-dir tiles/ \
  --output-dir stage3_metadata/ \
  --prepare
```

## Metadata Types Collected

### Essential
- **Month** (1-12): Seasonal patterns
- **Incidence Angle** (20-45°): SAR viewing geometry

### Optional
- **Biome Code**: Land cover type
- **Elevation**: Terrain height
- **Slope/Aspect**: Terrain characteristics

## Usage in Code

```python
from axs_lib.metadata import MetadataManager

# Extract metadata
manager = MetadataManager()
meta = manager.extract_from_tile('tiles/tile001.npz')

print(f"Month: {meta.month}")
print(f"Incidence: {meta.incidence_angle}°")

# Create embedding for Stage 3
embedding = manager.create_embedding_vector(meta)
# Returns: [month_sin, month_cos, inc_angle_norm, elev_norm, slope_norm]
```

## Why This Matters

### Stage 1 (Current)
- Generates "generic" optical from SAR
- Same output regardless of season/geometry
- Good baseline, but limited

### Stage 3 (Future)
- **Season-aware**: Summer vs. winter vegetation
- **Geometry-aware**: Accounts for SAR viewing angle
- **Context-aware**: Uses terrain and biome info
- **Better quality**: More physically plausible

## Architecture Decision

**Q: Why not add metadata tokens to TerraMind Stage 1?**

**A:** TerraMind 1.0 pretrained weights don't include metadata conditioning. Options:
1. ❌ **Retrain from scratch** - Requires 100B+ tokens, impractical
2. ❌ **Fine-tune with new tokens** - Would break pretrained weights
3. ✅ **Save for Stage 3** - Leverage pretrained Stage 1, add conditioning later

## Stage 3 Integration (Future)

```python
# Pseudo-code for Stage 3
class ConditionalModel(nn.Module):
    def forward(self, stage1_features, metadata_embedding):
        # Encode metadata
        meta_feat = self.meta_encoder(metadata_embedding)
        
        # Combine with Stage 1 output
        combined = torch.cat([stage1_features, meta_feat], dim=1)
        
        # Generate final prediction
        output = self.decoder(combined)
        return output
```

## Full Documentation

See `docs/metadata_handling.md` for:
- Detailed metadata types
- Embedding strategies
- Stage 3 architecture options
- Best practices
- Troubleshooting

## Files Created

- `axs_lib/metadata.py` - Metadata extraction and management
- `docs/metadata_handling.md` - Comprehensive documentation
- `docs/METADATA_README.md` - This quick reference
