"""
Simple test to verify Stage 1 → Stage 2 connection works.

This test confirms:
1. Stage 1 (TerraMind) produces 12-band S2L2A output from SAR
2. Stage 2 (Prithvi) accepts 6 HLS bands and produces 256-dim features
3. The pipeline is properly connected and ready for training

Note: Stage 2 output quality is NOT evaluated here since:
- Stage 2 produces features for Stage 3, not reconstructed images
- Stage 2 ConvNeXt head is randomly initialized (needs training)
- We only verify tensor shapes and data flow
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*80)
print("Stage 1 → Stage 2 Connection Test")
print("="*80)

# Load SAR data
sar_path = Path("data/tiles/benv2_test/S2B_MSIL2A_20180421T100029_N9999_R122_T33TWN_50_83.npz")
print(f"\nLoading SAR data from: {sar_path}")
data = np.load(sar_path)

# Extract SAR
s1_vv = data['s1_vv']
s1_vh = data['s1_vh']
sar = np.stack([s1_vv, s1_vh], axis=0)

# Load metadata for denormalization
import json
metadata_path = sar_path.with_suffix('.json')
with open(metadata_path) as f:
    metadata = json.load(f)

s1_norm = metadata['s1_normalization']
sar_db = sar * (s1_norm['max'] - s1_norm['min']) + s1_norm['min']

print(f"SAR shape: {sar_db.shape}")
print(f"SAR range (dB): [{sar_db.min():.2f}, {sar_db.max():.2f}]")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ============================================================================
# STAGE 1: TerraMind SAR → Optical
# ============================================================================
print("="*80)
print("STAGE 1: TerraMind")
print("="*80)

sar_tensor = torch.from_numpy(sar_db).float().unsqueeze(0).to(device)

# Pad to 224x224 (TerraMind expects this size)
pad_size = (224 - 120) // 2
sar_padded = torch.nn.functional.pad(sar_tensor, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

print(f"Input: {sar_padded.shape}")

try:
    from terratorch import FULL_MODEL_REGISTRY
    
    model_s1 = FULL_MODEL_REGISTRY.build(
        'terramind_v1_large_generate',
        pretrained=True,
        modalities=['S1GRD'],
        output_modalities=['S2L2A'],
        timesteps=10,
        standardize=True
    )
    model_s1 = model_s1.to(device)
    model_s1.eval()
    
    with torch.no_grad():
        outputs_s1 = model_s1({'S1GRD': sar_padded})
        s2_stage1_padded = outputs_s1['S2L2A']
    
    s2_stage1_padded = torch.clamp(s2_stage1_padded, 0, 10000)
    s2_stage1 = s2_stage1_padded[:, :, pad_size:pad_size+120, pad_size:pad_size+120]
    
    print(f"✓ Stage 1 output: {s2_stage1.shape}")
    print(f"  12-band S2L2A: [{s2_stage1.min():.2f}, {s2_stage1.max():.2f}] DN")
    
    # Extract 6 HLS bands for Stage 2 (B02, B03, B04, B08, B11, B12 = indices 1,2,3,7,10,11)
    s2_stage1_6band = s2_stage1[:, [1,2,3,7,10,11], :, :]
    print(f"  6 HLS bands: {s2_stage1_6band.shape}")
    
except Exception as e:
    print(f"✗ Stage 1 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STAGE 2: Prithvi Feature Extraction
# ============================================================================
print("\n" + "="*80)
print("STAGE 2: Prithvi Refinement")
print("="*80)

try:
    from axs_lib.stage2_prithvi_refine import build_prithvi_refiner
    
    local_checkpoint = Path("weights/hf/Prithvi-EO-2.0-600M/Prithvi_EO_V2_600M.pt")
    
    print("Building Prithvi refiner...")
    model_s2 = build_prithvi_refiner(
        prithvi_checkpoint=str(local_checkpoint) if local_checkpoint.exists() else None,
        num_input_channels=6,
        out_channels=256,
        hidden_dim=256,
        num_convnext_blocks=4,
        lora_r=8,
        lora_alpha=16,
        load_in_8bit=False,
        freeze_backbone=True,
        unfreeze_last_n_blocks=2,
        device=device,
        prithvi_model='prithvi_eo_v2_600'
    )
    model_s2.eval()
    
    # Normalize to [0, 1]
    s2_stage1_6band_norm = s2_stage1_6band / 10000.0
    
    print(f"\nInput: {s2_stage1_6band_norm.shape}")
    print(f"  Range: [{s2_stage1_6band_norm.min():.4f}, {s2_stage1_6band_norm.max():.4f}]")
    
    with torch.no_grad():
        s2_features = model_s2(s2_stage1_6band_norm)
    
    print(f"\n✓ Stage 2 output: {s2_features.shape}")
    print(f"  256-dim features: [{s2_features.min():.4f}, {s2_features.max():.4f}]")
    print(f"  Feature std: {s2_features.std():.4f}")
    
    # Check feature diversity (should not be constant)
    feature_variance = s2_features.var(dim=(0, 2, 3)).mean()
    print(f"  Mean channel variance: {feature_variance:.4f}")
    
    if feature_variance < 0.01:
        print("  ⚠ Warning: Low feature variance, features may be collapsed")
    else:
        print("  ✓ Features have good diversity")
    
    stage2_success = True
    
except Exception as e:
    print(f"✗ Stage 2 failed: {e}")
    import traceback
    traceback.print_exc()
    stage2_success = False

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("Pipeline Summary")
print("="*80)

if stage2_success:
    print("✓ Stage 1 → Stage 2 connection SUCCESSFUL")
    print()
    print("Pipeline flow:")
    print(f"  SAR (2, 120, 120) → Stage 1 → S2L2A (12, 120, 120)")
    print(f"  S2L2A subset (6 HLS bands) → Stage 2 → Features (256, 120, 120)")
    print()
    print("Next steps:")
    print("  1. Train Stage 2 with reconstruction loss (MAE style)")
    print("  2. Fine-tune with task-specific data")
    print("  3. Connect to Stage 3 for segmentation")
    print()
    print("Note: Stage 2 produces features, not reconstructed images.")
    print("      The ConvNeXt head is randomly initialized and needs training.")
else:
    print("✗ Pipeline connection FAILED")
    print("  Fix Stage 2 issues before proceeding")

print("="*80)
