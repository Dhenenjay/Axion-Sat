# Corrected TerraMind Training Approach

## Key Insight from README

After reviewing the TerraMind README and examples, the issue is clear:

**We were trying to fine-tune the generation model, but TerraMind's fine-tuning examples use the ENCODER BACKBONE for downstream tasks, not the generation model.**

## TerraMind Architecture Components

### 1. Generation Model (`terramind_v1_large_generate`)
```python
# This is for ANY-TO-ANY GENERATION (inference only)
model = FULL_MODEL_REGISTRY.build(
    'terramind_v1_large_generate',
    modalities=['S2L2A'],
    output_modalities=['S1GRD'],
    timesteps=10
)
# Uses discrete sampling - CANNOT be trained with standard backprop
```

### 2. Encoder Backbone (`terramind_v1_large`)
```python
# This is for FINE-TUNING on downstream tasks
model = BACKBONE_REGISTRY.build(
    'terramind_v1_large',
    pretrained=True,
    modalities=['S2L2A', 'S1GRD']
)
# Outputs patch embeddings (B, 196, 768) - CAN be trained!
```

## Corrected Approach for Axion-Sat

### Option A: Standard Fine-Tuning (Recommended)
Use TerraMind encoder + segmentation head, following their Sen1Floods11 example:

```python
# 1. Build encoder backbone
from terratorch import BACKBONE_REGISTRY

encoder = BACKBONE_REGISTRY.build(
    'terramind_v1_large',
    pretrained=True,
    modalities=['S1GRD', 'S2L2A']  # Both SAR and optical inputs
)

# 2. Add segmentation head
from terratorch.models.heads import ClassificationHead

head = ClassificationHead(
    in_channels=768,  # TerraMind embedding dim
    num_classes=num_land_cover_classes
)

# 3. Train with standard supervised learning
outputs = encoder({'S1GRD': s1, 'S2L2A': s2})  # (B, 196, 768)
logits = head(outputs)  # (B, num_classes, H, W)
loss = criterion(logits, labels)
loss.backward()  # ✓ Gradients flow normally!
```

### Option B: Thinking-in-Modalities (TiM)
Use TerraMind's TiM approach where intermediate modalities are generated:

```python
encoder = BACKBONE_REGISTRY.build(
    'terramind_v1_large_tim',
    pretrained=True,
    modalities=['S1GRD'],  # Only SAR input
    tim_modalities=['S2L2A']  # Generate optical as intermediate step
)

# The model internally:
# 1. Takes S1GRD input
# 2. Generates S2L2A internally (using generation model)
# 3. Encoder processes both S1GRD + generated S2L2A
# 4. Returns embeddings for downstream task
```

### Option C: Two-Stage (SAR-to-Optical + Downstream Task)
Use generation for data augmentation, then train on downstream task:

```python
# Stage 1: Generate synthetic optical from SAR (inference only)
generator = FULL_MODEL_REGISTRY.build(
    'terramind_v1_large_generate',
    modalities=['S1GRD'],
    output_modalities=['S2L2A'],
    timesteps=10
)

with torch.no_grad():
    synthetic_s2 = generator({'S1GRD': s1})['S2L2A']

# Stage 2: Fine-tune encoder on downstream task
encoder = BACKBONE_REGISTRY.build('terramind_v1_large', ...)
embeddings = encoder({'S2L2A': synthetic_s2, 'S1GRD': s1})
# ... train with segmentation head
```

## Recommended Solution for Your Use Case

Based on your `benv2_catalog` dataset with paired S1/S2 data, I recommend **Option A with LoRA**:

### Architecture
```
Input: SAR (S1) + Optical (S2)
    ↓
TerraMind Encoder (with LoRA adapters)
    ↓
Patch Embeddings (B, 196, 768)
    ↓
Segmentation Head
    ↓
Land Cover Predictions
```

### Benefits
1. ✓ Uses both SAR and optical data
2. ✓ Trains with standard supervised learning (no gradient issues)
3. ✓ Follows TerraMind's intended fine-tuning approach
4. ✓ Can use LoRA for parameter-efficient training
5. ✓ Works with your existing GPU setup

### Implementation Changes Needed

1. **Change model initialization** (in `scripts/train_stage1.py`):
   ```python
   # OLD: Used generation model
   model = build_terramind_generator(
       input_modalities=("S1GRD",),
       output_modalities=("S2L2A",),
       ...
   )
   
   # NEW: Use encoder backbone
   from terratorch import BACKBONE_REGISTRY
   
   encoder = BACKBONE_REGISTRY.build(
       'terramind_v1_large',
       pretrained=True,
       modalities=['S1GRD', 'S2L2A']
   )
   ```

2. **Add segmentation head**:
   ```python
   from terratorch.models.heads import SegmentationHead
   
   head = SegmentationHead(
       in_channels=768,
       num_classes=num_classes,
       head_type='fcn'  # or 'uper', 'deeplab'
   )
   ```

3. **Modify forward pass**:
   ```python
   # Forward pass
   embeddings = encoder({'S1GRD': s1, 'S2L2A': s2})
   logits = head(embeddings)
   
   # Loss computation
   loss = criterion(logits, ground_truth_labels)
   loss.backward()  # ✓ Works now!
   ```

4. **LoRA application**:
   - LoRA can still be applied to the encoder
   - Will work correctly since encoder uses standard forward pass

## File Structure for Implementation

```
scripts/
├── train_stage1.py              # Needs modification
├── train_encoder_downstream.py  # NEW: Encoder-based training
└── ...

axs_lib/
├── models.py                    # Update to build encoder instead of generator
├── stage1_encoder.py            # NEW: Encoder-based wrapper
└── stage1_tm_s2o.py            # Keep for generation/inference
```

## Next Steps

1. **Immediate**: Create new training script using encoder backbone
2. **Test**: Run small experiment with 5-10 tiles to verify gradient flow
3. **Scale**: Once working, train on full dataset

Would you like me to:
1. Modify the existing training script to use the encoder approach?
2. Create a new training script from scratch following TerraMind examples?
3. First create a simple test script to verify the encoder + head works?
