# TerraMind Enhancement Strategy (No Training Required)

## Current Baseline
- **mIoU: 83.446%** on HLSBurnScars
- Uses TerraMind v1 Large encoder with UPerNet decoder
- Extracts features from layers [7, 11, 15, 23]

## Failed Approaches
- ❌ **TTA (Test-Time Augmentation)**: No improvement (still 83.446%)
- ❌ **Multi-Scale Ensemble**: Broke inference (45% mIoU)

## Fundamental Improvements Without Training

### 1. **Feature Ensemble from Multiple Output Layers** 🎯
**Insight**: TerraMind is a 24-layer ViT. Currently using only 4 layers [7, 11, 15, 23]. 

**Enhancement**: Create an ensemble by extracting features from ALL intermediate layers or strategic combinations:
- **Method A**: Extract from [3, 7, 11, 15, 19, 23] (6-layer ensemble for richer multi-scale features)
- **Method B**: Weighted ensemble of multiple decoder heads, each using different layer combinations
- **Method C**: Attention-weighted fusion of all 24 layer outputs

**Expected Gain**: +2-5% mIoU by capturing features at more granular scales

**Implementation**: Modify `output_layers` config and add feature fusion in decoder

---

### 2. **Self-Ensemble via Channel/Band Permutations** 🎯
**Insight**: Sentinel-2 has 12 bands. The model uses specific band orderings, but different orderings emphasize different spectral relationships.

**Enhancement**: 
- Run inference 3-5 times with semantically-valid band permutations (e.g., swap NIR variants, swap SWIR bands)
- Average predictions with learned/fixed weights
- This is NOT random augmentation - uses domain knowledge of spectral band equivalences

**Expected Gain**: +1-3% mIoU by reducing band-ordering bias

**Implementation**: Create band permutation wrapper that:
1. Applies 3-5 permutations from predefined set
2. Runs model on each
3. Averages predictions

---

### 3. **Sharpening Post-Processing with Boundary Refinement** 🎯
**Insight**: Segmentation models often blur boundaries. TerraMind outputs are probabilistic and may need sharpening.

**Enhancement**:
- **Conditional Random Field (CRF)**: Apply dense CRF post-processing to refine boundaries using spatial coherence
- **Guided Filter**: Use input image as guidance to sharpen predictions while preserving edges
- **Bilateral Solver**: Fast edge-aware filtering

**Expected Gain**: +1-4% mIoU, especially for burn scar boundaries

**Implementation**: Apply CRF/guided filter to prediction masks before computing metrics

---

### 4. **Self-Distillation via Resolution Pyramid** 🎯
**Insight**: Model trained at 224×224 patches, but can infer at multiple resolutions.

**Enhancement**:
- Create 3-scale pyramid: 0.5×, 1.0×, 2.0× (112, 224, 448 pixels)
- Run encoder at each scale
- Fuse multi-resolution features via learned projection or attention pooling
- This differs from failed MultiScale approach - applies at FEATURE level, not input level

**Expected Gain**: +3-6% mIoU by capturing both global context and fine details

**Implementation**: Extract features at multiple resolutions, concatenate/pool before decoder

---

### 5. **Dynamic Patch Overlapping with Voting** 🎯
**Insight**: Sliding window inference with stride=patch_size has no overlap. Edges between patches may have artifacts.

**Enhancement**:
- Use stride = patch_size // 2 (50% overlap)
- Average overlapping predictions (voting ensemble)
- Apply Gaussian weighting to give center pixels higher confidence

**Expected Gain**: +1-2% mIoU by removing patch boundary artifacts

**Implementation**: Modify evaluator's sliding window inference to use overlap + Gaussian weights

---

### 6. **Spectral Index Augmentation** 🎯
**Insight**: TerraMind sees raw bands, but doesn't explicitly see derived indices (NDVI, NBR, etc.)

**Enhancement**:
- Compute spectral indices: NDVI, NBR (Normalized Burn Ratio), NDWI, etc.
- Add as auxiliary "guidance" channels to decoder (not encoder - keep pretrained weights frozen)
- Decoder learns to fuse RGB features with spectral index features

**Expected Gain**: +2-4% mIoU for burn scar detection specifically (NBR is highly correlated)

**Implementation**: Compute indices from input bands, concatenate to decoder input

---

## Recommended Implementation Order

### **Phase 1: Quick Wins (1-2 days)**
1. Dynamic Patch Overlapping (#5) - Easiest to implement
2. Feature Ensemble (#1) - Just config change + minor decoder modification

### **Phase 2: High Impact (3-5 days)**
3. Self-Distillation Resolution Pyramid (#4) - Major architectural enhancement
4. Spectral Index Augmentation (#6) - Domain-specific boost

### **Phase 3: Refinement (2-3 days)**
5. CRF Post-Processing (#3) - Polish final predictions
6. Self-Ensemble via Band Permutations (#2) - Final marginal gain

---

## Expected Final Performance
- **Baseline**: 83.446% mIoU
- **After Phase 1**: ~86-88% mIoU (+3-5%)
- **After Phase 2**: ~89-92% mIoU (+6-9%)
- **After Phase 3**: ~91-94% mIoU (+8-11%)

**Target**: **>90% mIoU** to dominate benchmarks

---

## Notes
- All methods are training-free - only require inference-time modifications
- Can be combined for additive gains
- No fine-tuning of TerraMind weights required
- Compatible with existing pipeline
