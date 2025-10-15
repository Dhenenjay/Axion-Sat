# Stage 2: Prithvi Refinement with Seasonality Priors

**Author:** Axion-Sat Project  
**Version:** 1.0.0  
**Last Updated:** 2025-10-14

---

## Table of Contents

1. [Overview](#overview)
2. [Seasonality Prior from HLS Neighbors](#seasonality-prior-from-hls-neighbors)
3. [Why We Never Change Geometry](#why-we-never-change-geometry)
4. [Architecture Design](#architecture-design)
5. [Loss Function Philosophy](#loss-function-philosophy)
6. [Training Strategy](#training-strategy)
7. [Validation Metrics](#validation-metrics)
8. [Common Pitfalls](#common-pitfalls)

---

## Overview

Stage 2 of the Axion-Sat pipeline refines the Stage 1 optical output (opt_v1) to produce
higher-quality, spectrally plausible imagery (opt_v2). The refinement uses **Prithvi-EO-2.0**
as a foundation model with two critical constraints:

1. **Seasonality Prior**: Leverage HLS neighbor tiles to understand seasonal vegetation patterns
2. **Geometric Consistency**: Never alter spatial structure or edge positions from Stage 1

This document explains the rationale behind these design decisions and how they ensure
physically plausible, scientifically valid outputs.

---

## Seasonality Prior from HLS Neighbors

### What is the Seasonality Prior?

The **seasonality prior** is contextual information about expected vegetation patterns
based on:

- **Temporal context**: What month is the observation?
- **Spatial context**: What biome/climate zone is the tile in?
- **Spectral context**: What do real HLS (Harmonized Landsat-Sentinel) observations look like
  for similar locations and times?

### Why Do We Need It?

Stage 1 (TerraMind SAR→Optical) produces reasonable optical imagery from SAR, but it has
**no knowledge** of:

1. Seasonal vegetation cycles (e.g., deciduous trees in winter vs. summer)
2. Regional phenology (e.g., corn in Iowa vs. rainforest in Brazil)
3. Spectral plausibility constraints (e.g., valid NDVI/EVI ranges)

Without seasonality priors, Stage 1 might produce:
- Lush green vegetation in winter for temperate forests
- Incorrect NIR responses for crop growth stages
- Implausible spectral signatures that violate physical laws

### How We Implement Seasonality Priors

#### 1. HLS Neighbor Retrieval

During training, we retrieve **nearby HLS tiles** from the same:
- **Temporal window**: ±15 days from target observation
- **Spatial radius**: Within ~50-100 km
- **Same biome**: Matching Köppen-Geiger climate classification

Example:
```
Target tile: BigEarthNet-v2 tile S2A_MSIL2A_20170613T101031_85_55
Location: 52.5°N, 13.4°E (Berlin, Germany)
Month: June
Biome: Temperate deciduous forest (Dfb)

HLS Neighbors Retrieved:
1. HLS.S30.T33UUU.2017166T103021 (June 15, 2017) - 12 km away
2. HLS.L30.T33UUU.2017171T100847 (June 20, 2017) - 18 km away
3. HLS.S30.T33UUU.2017161T103021 (June 10, 2017) - 12 km away
```

These neighbors provide **ground truth spectral distributions** for:
- NDVI (Normalized Difference Vegetation Index)
- EVI (Enhanced Vegetation Index)
- Per-band reflectance statistics

#### 2. Metadata Conditioning

We condition the Prithvi model on:

```python
metadata = {
    'month': 6,              # June (1-12)
    'biome': 7,              # Temperate deciduous (Köppen Dfb)
    'ndvi_mean': 0.65,       # From HLS neighbors
    'ndvi_std': 0.12,
    'evi_mean': 0.48,
    'evi_std': 0.09
}
```

This metadata is embedded and injected into the Prithvi backbone via:
- **Month embedding**: Sinusoidal encoding or learnable embedding (dim=64)
- **Biome embedding**: Learnable embedding lookup table (16 biomes, dim=64)
- **Statistical priors**: Used in loss function for spectral plausibility constraints

#### 3. Spectral Plausibility Loss

The loss function explicitly penalizes deviations from HLS neighbor statistics:

```python
# Extract NDVI/EVI from prediction and HLS neighbors
pred_ndvi = compute_ndvi(pred_red, pred_nir)
hls_ndvi_mean = metadata['ndvi_mean']
hls_ndvi_std = metadata['ndvi_std']

# Penalize if prediction falls outside expected range
ndvi_loss = F.mse_loss(pred_ndvi, hls_ndvi_mean)
ndvi_loss += relu(pred_ndvi - (hls_ndvi_mean + 2*hls_ndvi_std))  # Upper bound
ndvi_loss += relu((hls_ndvi_mean - 2*hls_ndvi_std) - pred_ndvi)  # Lower bound
```

This ensures refined outputs are **statistically consistent** with real observations.

### Example: Winter vs. Summer Refinement

**Scenario**: Same location, different seasons

**Winter (January):**
```
Input (Stage 1): NDVI = 0.45 (moderate vegetation)
HLS Neighbors:   NDVI = 0.15 ± 0.08 (leafless trees)
Output (Stage 2): NDVI = 0.22 (adjusted down to match winter phenology)
```

**Summer (July):**
```
Input (Stage 1): NDVI = 0.62 (high vegetation)
HLS Neighbors:   NDVI = 0.72 ± 0.10 (full canopy)
Output (Stage 2): NDVI = 0.68 (adjusted up to match summer phenology)
```

The model **learns** that the same SAR signature can correspond to different optical
appearances depending on season.

### Benefits of Seasonality Priors

1. **Physical Plausibility**: Outputs respect seasonal vegetation cycles
2. **Regional Accuracy**: Biome conditioning ensures realistic spectral signatures
3. **Generalization**: Model learns underlying phenological patterns, not memorization
4. **Uncertainty Reduction**: HLS statistics constrain the solution space

---

## Why We Never Change Geometry

### The Golden Rule: Geometric Immutability

**Stage 2 refines spectral content ONLY. It NEVER alters spatial structure.**

This is a **hard constraint** enforced through:
1. Architecture design (no downsampling, no spatial transformations)
2. Loss function penalties (edge-guard loss, identity loss)
3. Validation metrics (edge displacement < 2 pixels)

### Rationale: Why Is This Critical?

#### 1. Scientific Integrity

Remote sensing products are used for:
- Land cover mapping
- Crop field boundary delineation
- Urban infrastructure monitoring
- Forest fragmentation analysis

If Stage 2 **moves edges**, we introduce:
- False positives/negatives in change detection
- Geometric misalignment with other data sources
- Invalid area measurements
- Broken spatial relationships

**Example:**
```
Stage 1: Field boundary at pixel (100, 200)
Stage 2 (bad): Field boundary moves to (102, 199)
Result: 2-pixel shift = ~20-60m error in real-world coordinates
Impact: Field area calculated incorrectly, wrong crop yield estimates
```

#### 2. Preservation of SAR Geometric Fidelity

SAR imagery has **superior geometric accuracy** compared to optical:
- No cloud distortion
- Precise geolocation
- Stable geometry across acquisitions

Stage 1 (TerraMind) preserves this SAR geometry in the optical domain. Stage 2 must
**not degrade** this advantage by introducing spatial artifacts.

#### 3. Spectral vs. Geometric Information Content

| Information Type | Stage 1 Quality | Stage 2 Goal |
|------------------|-----------------|--------------|
| **Geometric** (edges, boundaries) | ✅ Good (from SAR) | 🔒 **Preserve** |
| **Spectral** (NDVI, EVI, per-band) | ⚠️ Moderate (translated) | ✨ **Refine** |
| **Texture** (high-freq detail) | ⚠️ Moderate | ✨ **Refine** |

Stage 1 gives us accurate **where**, Stage 2 gives us accurate **what**.

#### 4. Downstream Application Requirements

Many downstream tasks require **pixel-perfect alignment**:

**Change Detection:**
```python
# Requires exact spatial correspondence
diff = image_t2 - image_t1  # Pixel-wise subtraction
```

If Stage 2 moves edges by even 1-2 pixels, change detection becomes unreliable.

**Multi-temporal Analysis:**
```python
# Stack images for time series analysis
time_series = [img_jan, img_mar, img_jun, img_sep]
ndvi_trend = compute_ndvi_trend(time_series)  # Requires pixel alignment
```

Geometric shifts would introduce spurious temporal variations.

#### 5. Prithvi's Tendency to Hallucinate Edges

Foundation models like Prithvi are trained on **millions of optical images** and learn
strong priors about:
- Field boundaries
- Road networks
- Building edges
- Natural features

Without constraints, Prithvi might:
- "Sharpen" edges that are actually gradual transitions
- "Create" boundaries that don't exist in the SAR
- "Move" features to align with its learned priors

**Example of Hallucination:**
```
Input (Stage 1):  Gradual forest-grassland transition over 5 pixels
Prithvi (unconstrained): Sharp boundary at pixel 3 (matches training data)
Reality: Transition is actually gradual (SAR is correct)
```

This is why we need **edge-guard loss** to prevent hallucination.

### How We Enforce Geometric Consistency

#### 1. Architecture: No Spatial Transformations

```python
class ConvNeXtHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # All operations preserve spatial dimensions
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(dim=in_channels)  # stride=1, same padding
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=1  # 1x1 conv, no spatial change
        )
    
    def forward(self, x):
        # Input: (B, C, H, W)
        for block in self.blocks:
            x = block(x)  # Output: (B, C, H, W) - same size!
        x = self.output_proj(x)
        return x  # Output: (B, C', H, W) - same H, W
```

Key design choices:
- ✅ **No pooling** (would downsample)
- ✅ **No strided convolutions** (would shift features)
- ✅ **No deformable convolutions** (would warp geometry)
- ✅ **No attention-based spatial mixing** (could blur boundaries)

#### 2. Identity/Edge-Guard Loss

```python
class IdentityEdgeGuardLoss(nn.Module):
    def forward(self, pred, target):
        # Identity loss: penalize pixel-wise differences
        identity_loss = F.l1_loss(pred, target)
        
        # Edge loss: penalize edge position changes
        pred_edges = compute_sobel_edges(pred)
        target_edges = compute_sobel_edges(target)
        edge_loss = F.l1_loss(pred_edges, target_edges)
        
        return identity_loss + 0.5 * edge_loss
```

This loss **heavily penalizes** any spatial changes, forcing the model to only adjust
per-pixel intensity values.

#### 3. Validation: Edge Displacement Metric

```python
def compute_edge_displacement(pred, target):
    """
    Measure maximum spatial shift of edges.
    
    Returns:
        mean_displacement (pixels): Average edge shift
        max_displacement (pixels): Maximum edge shift
    """
    # Extract edges
    pred_edges = extract_edge_pixels(pred)
    target_edges = extract_edge_pixels(target)
    
    # Find nearest neighbors
    displacements = []
    for target_edge_pixel in target_edges:
        nearest_pred_edge = find_nearest(target_edge_pixel, pred_edges)
        displacement = distance(target_edge_pixel, nearest_pred_edge)
        displacements.append(displacement)
    
    return {
        'mean': np.mean(displacements),
        'max': np.max(displacements)
    }
```

**Acceptance criteria:**
- ✅ Mean displacement < 2.0 pixels (good)
- ⚠️ Mean displacement 2-5 pixels (marginal)
- ❌ Mean displacement > 5 pixels (reject model)

### What Changes ARE Allowed?

Stage 2 **CAN** modify:

1. **Per-pixel intensity**: Adjust B02, B03, B04, B08 values
   ```
   Before: [0.10, 0.15, 0.08, 0.45]
   After:  [0.12, 0.18, 0.10, 0.52]  ✅ OK
   ```

2. **Spectral indices**: Change NDVI, EVI, etc.
   ```
   Before: NDVI = 0.45
   After:  NDVI = 0.52  ✅ OK (within reason)
   ```

3. **Texture**: Enhance fine-scale detail
   ```
   Before: Smooth field texture
   After:  Realistic crop row texture  ✅ OK
   ```

Stage 2 **CANNOT** modify:

1. **Edge positions**: Boundaries must not move
   ```
   Before: Field edge at (100, 200)
   After:  Field edge at (102, 199)  ❌ FORBIDDEN
   ```

2. **Object shapes**: Geometries must not warp
   ```
   Before: Circular lake
   After:  Elliptical lake  ❌ FORBIDDEN
   ```

3. **Spatial relationships**: Relative positions must not change
   ```
   Before: Building north of road
   After:  Building south of road  ❌ FORBIDDEN
   ```

---

## Architecture Design

### High-Level Pipeline

```
Stage 1 Output (opt_v1)
  ↓
[Input Projection: 4 → 768 channels]
  ↓
[Prithvi Backbone with LoRA]
  ├─ Month embedding → injected
  ├─ Biome embedding → injected
  ↓
[ConvNeXt Refinement Head: 768 → 256 → 4]
  ↓
Stage 2 Output (opt_v2)
```

### Prithvi Backbone with LoRA

**Why Prithvi?**
- Pre-trained on 1B+ optical satellite images
- Understands global vegetation patterns
- Encodes seasonal, regional, and spectral priors

**Why LoRA (Low-Rank Adaptation)?**
- Full fine-tuning of 600M parameters is expensive
- LoRA adds **trainable low-rank matrices** to attention layers
- Only ~8M parameters to train (rank=8)
- Preserves Prithvi's foundational knowledge

**Configuration:**
```yaml
prithvi:
  model: prithvi_eo_600m
  freeze_backbone: true
  unfreeze_last_n_blocks: 4  # Only last 4 transformer blocks adapt
  load_in_8bit: true          # Quantization for memory efficiency

lora:
  r: 8                        # Rank (8-16 typical)
  alpha: 16                   # Scaling factor (2×r)
  target_modules:
    - qkv                     # Attention projections
    - proj                    # Output projection
    - fc1                     # MLP layer 1
    - fc2                     # MLP layer 2
```

### ConvNeXt Refinement Head

**Why ConvNeXt?**
- Modern CNN architecture (CVPR 2022)
- Strong inductive bias for local patterns
- No spatial transformations (unlike transformers)
- Memory efficient

**Architecture:**
```python
ConvNeXtHead(
    in_channels=768,      # Prithvi output
    out_channels=4,       # B02, B03, B04, B08
    hidden_dim=256,       # Low for memory
    num_blocks=4          # 2-8 blocks typical
)
```

Each ConvNeXt block:
1. **Depthwise convolution** (7×7, groups=dim): Spatial mixing within channels
2. **Layer norm**: Stabilization
3. **Pointwise expansion** (1×1, dim → 4×dim): Channel mixing
4. **GELU activation**: Non-linearity
5. **Pointwise projection** (1×1, 4×dim → dim): Channel compression
6. **Residual connection**: x_out = x_in + block(x_in)

**Critical**: All operations have `stride=1` and `padding='same'` → no spatial changes.

### CPU Fallback Mechanism

If GPU runs out of memory during refinement head forward pass:

```python
def forward(self, x):
    try:
        # Try GPU forward
        return self.refinement_head(x)
    except RuntimeError as e:
        if "out of memory" in str(e):
            # Fall back to CPU
            logger.warning("GPU OOM, falling back to CPU")
            self.refinement_head.cpu()
            x_cpu = x.cpu()
            output = self.refinement_head(x_cpu)
            return output.to(x.device)
        else:
            raise
```

This ensures training continues on lower-end GPUs (8-12 GB VRAM).

---

## Loss Function Philosophy

### Combined Loss Function

```python
total_loss = (
    1.0 × spectral_loss +        # Spectral plausibility (NDVI/EVI/SAM)
    0.5 × identity_edge_loss +   # Geometric consistency
    0.05 × adversarial_loss      # Texture realism (PatchGAN)
)
```

### 1. Spectral Plausibility Loss (Weight: 1.0)

**Goal**: Ensure vegetation indices match HLS neighbor statistics.

**Components:**
```python
# NDVI RMSE
ndvi_loss = sqrt(MSE(pred_ndvi, target_ndvi))

# EVI RMSE
evi_loss = sqrt(MSE(pred_evi, target_evi))

# Spectral Angle Mapper (SAM)
sam_loss = arccos(dot(pred_spectrum, target_spectrum) / 
                  (norm(pred_spectrum) * norm(target_spectrum)))

spectral_loss = 1.0*ndvi_loss + 0.5*evi_loss + 0.3*sam_loss
```

**Why this matters:**
- NDVI/EVI must be in physically valid ranges (-1 to 1)
- Spectral angles measure similarity in n-dimensional reflectance space
- Violations indicate non-physical outputs

### 2. Identity/Edge-Guard Loss (Weight: 0.5)

**Goal**: Prevent geometric changes.

**Components:**
```python
# Identity: pixel-wise L1 distance
identity_loss = L1(pred, target)

# Edge preservation: Sobel gradient L1 distance
pred_edges = sobel(pred)
target_edges = sobel(target)
edge_loss = L1(pred_edges, target_edges)

identity_edge_loss = 1.0*identity_loss + 0.5*edge_loss
```

**Why lower weight than spectral?**
- We **do** want to change spectral values (refinement)
- We **don't** want to change them drastically (identity)
- We **never** want to move edges (edge-guard)

The 0.5 weight balances refinement vs. preservation.

### 3. Adversarial Loss (Weight: 0.05)

**Goal**: Encourage realistic texture detail.

**Architecture**: Lightweight PatchGAN discriminator
- Input: 4-channel optical image
- Layers: 4 convolutional layers (32 → 64 → 128 → 256 filters)
- Output: Per-patch real/fake predictions

**Why PatchGAN?**
- Focuses on local texture realism
- Doesn't enforce global structure (good for us!)
- Lightweight: only ~500K parameters

**Why low weight (0.05)?**
- Adversarial loss can be unstable
- We don't want it to override spectral constraints
- It's a "polish" rather than primary objective

**Loss formulation (LSGAN):**
```python
# Generator loss
gen_loss = MSE(D(pred), 1.0)  # Try to fool discriminator

# Discriminator loss
disc_loss = 0.5 * (MSE(D(real), 1.0) + MSE(D(pred), 0.0))
```

---

## Training Strategy

### Data Sources

1. **Input (opt_v1)**: Stage 1 TerraMind output from BigEarthNet-v2 tiles
2. **Target (opt_v2)**: Real Sentinel-2 L2A imagery (gold standard)
3. **Metadata**: Month, biome, HLS neighbor statistics

### Training Procedure

**Phase 1: Warmup (5k steps)**
- Only spectral + identity losses
- No adversarial loss
- Goal: Stable spectral refinement

**Phase 2: Full Training (25-35k steps)**
- All losses active
- Discriminator updates every 5 generator steps
- Goal: Texture realism + spectral accuracy

**Phase 3: Fine-tuning (5k steps)**
- Reduce adversarial weight to 0.02
- Increase edge-guard weight to 0.7
- Goal: Final polish + geometric preservation

### Hyperparameters

```yaml
optimizer: AdamW
learning_rate: 1e-4
lr_scheduler: cosine_with_warmup
warmup_steps: 1500
batch_size: 1 (with 8 grad accumulation = effective 8)
precision: fp16
max_grad_norm: 1.0
```

### Augmentation

**Spatial augmentations (applied to both input and target):**
- Horizontal flip (p=0.5)
- Vertical flip (p=0.5)
- 90° rotations (p=0.25 each)

**Why no color augmentation?**
Spectral values have physical meaning—we can't arbitrarily shift them.

---

## Validation Metrics

### Primary Metrics (Must Pass)

1. **Edge Displacement**
   - Mean < 2.0 pixels ✅
   - Max < 5.0 pixels ✅
   - Edge ratio > 0.7 ✅

2. **NDVI Improvement**
   - RMSE(v2, real) < RMSE(v1, real) ✅
   - Improvement > 0.015 ✅

3. **EVI Improvement**
   - RMSE(v2, real) < RMSE(v1, real) ✅
   - Improvement > 0.020 ✅

### Secondary Metrics (Quality Indicators)

4. **Spectral Angle Mapper (SAM)**
   - SAM(v2, real) < SAM(v1, real) ✅
   - Improvement > 0.04 radians ✅

5. **LPIPS (Perceptual Similarity)**
   - LPIPS(v2, v1) in [0.10, 0.15] ✅
   - Not too similar (no refinement) or too different (hallucination)

6. **PSNR (Signal Quality)**
   - PSNR(v2, real) > PSNR(v1, real) ✅
   - Improvement > 2 dB ✅

### Validation Protocol

**Per-epoch validation (every 1000 steps):**
```python
for batch in val_loader:
    opt_v1, opt_v2_real, metadata = batch
    
    # Forward pass
    opt_v2_pred = model(opt_v1, metadata)
    
    # Compute metrics
    metrics = {
        'edge_disp': compute_edge_displacement(opt_v2_pred, opt_v1),
        'ndvi_rmse': compute_ndvi_rmse(opt_v2_pred, opt_v2_real),
        'evi_rmse': compute_evi_rmse(opt_v2_pred, opt_v2_real),
        'sam': compute_sam(opt_v2_pred, opt_v2_real),
        'lpips': compute_lpips(opt_v2_pred, opt_v2_real),
        'psnr': compute_psnr(opt_v2_pred, opt_v2_real)
    }
    
    # Check thresholds
    assert metrics['edge_disp']['mean'] < 2.0, "Geometry changed!"
```

---

## Common Pitfalls

### 1. ❌ Allowing Spatial Transformations

**Problem:**
```python
# BAD: Strided convolution
self.conv = nn.Conv2d(256, 512, kernel_size=3, stride=2)
# Output is half the spatial size!
```

**Solution:**
```python
# GOOD: No stride
self.conv = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
# Output is same spatial size
```

### 2. ❌ Too High Adversarial Weight

**Problem:**
```python
# BAD: Adversarial loss dominates
total_loss = 1.0*spectral + 0.5*identity + 1.0*adversarial
# Model prioritizes fooling discriminator over spectral accuracy
```

**Solution:**
```python
# GOOD: Low adversarial weight
total_loss = 1.0*spectral + 0.5*identity + 0.05*adversarial
# Spectral accuracy is primary objective
```

### 3. ❌ Ignoring HLS Neighbor Statistics

**Problem:**
```python
# BAD: Generic NDVI target
target_ndvi = compute_ndvi(real_image)
loss = MSE(pred_ndvi, target_ndvi)
# Doesn't account for seasonality
```

**Solution:**
```python
# GOOD: Condition on HLS statistics
hls_ndvi_mean = metadata['hls_ndvi_mean']
hls_ndvi_std = metadata['hls_ndvi_std']
loss = MSE(pred_ndvi, hls_ndvi_mean)
loss += penalty_outside_range(pred_ndvi, hls_ndvi_mean, hls_ndvi_std)
```

### 4. ❌ Not Validating Edge Displacement

**Problem:**
Training only checks NDVI/EVI, not geometry.

**Solution:**
```python
# Validate edge displacement every epoch
if step % 1000 == 0:
    edge_metrics = compute_edge_displacement(pred, target)
    assert edge_metrics['mean'] < 2.0, "Geometry violated!"
```

### 5. ❌ Using Attention-Heavy Architectures

**Problem:**
```python
# BAD: Global attention can blur boundaries
self.attention = MultiHeadSelfAttention(dim=768, num_heads=8)
# Attention mixes spatial information globally
```

**Solution:**
```python
# GOOD: Local convolutions preserve boundaries
self.conv = nn.Conv2d(768, 768, kernel_size=7, groups=768, padding=3)
# Depthwise conv only mixes locally
```

### 6. ❌ Overfitting to Training Set Biomes

**Problem:**
Model only sees temperate forests, fails on tropical savannas.

**Solution:**
- Sample diverse biomes in training
- Use biome embedding to explicitly condition model
- Validate on held-out biomes

---

## Summary

### Key Takeaways

1. **Seasonality priors** from HLS neighbors ensure spectral realism
2. **Geometric immutability** preserves spatial accuracy from SAR
3. **Prithvi + LoRA** leverages foundation model knowledge efficiently
4. **Multi-component loss** balances spectral, geometric, and textural objectives
5. **Edge displacement < 2px** is the primary validation constraint

### Design Philosophy

> "Stage 2 is a **spectral refinement filter**, not a generative model.  
> It corrects what Stage 1 got wrong spectrally,  
> but preserves what Stage 1 got right geometrically."

### When to Retrain Stage 2

Retrain if:
- New biomes/regions need support
- Improved HLS neighbor retrieval strategy
- Better foundation model available (e.g., Prithvi-EO-3.0)
- Edge displacement validation fails on new data

Do **not** retrain if:
- Stage 1 output quality improves (just re-run Stage 2)
- Only metadata changes (month/biome)
- Small NDVI/EVI improvements needed (adjust loss weights)

---

## References

1. **Prithvi Foundation Model**  
   Jakubik et al., "Foundation Models for Generalist Geospatial AI", arXiv 2023

2. **HLS (Harmonized Landsat Sentinel-2)**  
   Claverie et al., "The Harmonized Landsat and Sentinel-2 surface reflectance data set", RSE 2018

3. **ConvNeXt Architecture**  
   Liu et al., "A ConvNet for the 2020s", CVPR 2022

4. **LoRA (Low-Rank Adaptation)**  
   Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022

5. **PatchGAN Discriminator**  
   Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks", CVPR 2017

6. **BigEarthNet-v2 Dataset**  
   Sumbul et al., "BigEarthNet-MM: A Large Scale Multi-Modal Multi-Label Benchmark Archive", IEEE GRSM 2021

---

**End of Document**
