"""
Test Stage 1 (TerraMind) + Stage 2 (Prithvi) together.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from axs_lib.stage2_prithvi_refine import build_prithvi_refiner
from axs_lib.io import load_checkpoint

print("="*80)
print("Stage 1 + Stage 2 Combined Test")
print("="*80)

# Load SAR data
sar_path = Path("data/tiles/benv2_test/S2B_MSIL2A_20180421T100029_N9999_R122_T33TWN_50_83.npz")
print(f"\nLoading SAR data from: {sar_path}")
data = np.load(sar_path)

# Extract and denormalize SAR
s1_vv = data.get('s1_vv', data.get('VV', None))
s1_vh = data.get('s1_vh', data.get('VH', None))
sar = np.stack([s1_vv, s1_vh], axis=0)

# Load metadata
metadata_path = sar_path.with_suffix('.json')
with open(metadata_path) as f:
    metadata = json.load(f)

s1_norm = metadata['s1_normalization']
sar_db = sar * (s1_norm['max'] - s1_norm['min']) + s1_norm['min']

print(f"SAR (dB): [{sar_db.min():.2f}, {sar_db.max():.2f}]")

# Also load ground truth S2 if available
if 's2' in data.keys():
    gt_s2 = data['s2']
    print(f"Ground truth S2: {gt_s2.shape}, range [{gt_s2.min():.0f}, {gt_s2.max():.0f}]")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# ============================================================================
# STAGE 1: TerraMind SAR → Optical
# ============================================================================
print("="*80)
print("STAGE 1: TerraMind Generation")
print("="*80)

sar_tensor = torch.from_numpy(sar_db).float().unsqueeze(0).to(device)

# Pad to 224x224
pad_size = (224 - 120) // 2
sar_padded = torch.nn.functional.pad(sar_tensor, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

print(f"Input SAR: {sar_padded.shape}")

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
    
    print("✓ TerraMind model loaded")
    
    with torch.no_grad():
        outputs_s1 = model_s1({'S1GRD': sar_padded})
        s2_stage1_padded = outputs_s1['S2L2A']
    
    # Clip and crop
    s2_stage1_padded = torch.clamp(s2_stage1_padded, 0, 10000)
    s2_stage1 = s2_stage1_padded[:, :, pad_size:pad_size+120, pad_size:pad_size+120]
    
    print(f"✓ Stage 1 output: {s2_stage1.shape}")
    print(f"  Range: [{s2_stage1.min():.2f}, {s2_stage1.max():.2f}] DN")
    print(f"  Mean: {s2_stage1.mean():.2f} DN")
    
except Exception as e:
    print(f"✗ Stage 1 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STAGE 2: Prithvi Refinement
# ============================================================================
print("\n" + "="*80)
print("STAGE 2: Prithvi Refinement")
print("="*80)

# Extract 6 HLS bands from 12 S2L2A bands
# S2L2A: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
# HLS: Blue(B02), Green(B03), Red(B04), NIR(B08), SWIR1(B11), SWIR2(B12)
# Indices: 1, 2, 3, 7, 10, 11
hls_indices = [1, 2, 3, 7, 10, 11]
s2_hls = s2_stage1[:, hls_indices, :, :]  # (1, 6, 120, 120)

# Convert to reflectance [0, 1]
s2_hls_reflectance = s2_hls / 10000.0

print(f"Extracted HLS bands: {s2_hls.shape}")
print(f"  Range (DN): [{s2_hls.min():.2f}, {s2_hls.max():.2f}]")
print(f"  Range (reflectance): [{s2_hls_reflectance.min():.4f}, {s2_hls_reflectance.max():.4f}]")

# Build Prithvi refiner
try:
    print("\nBuilding Prithvi refiner...")
    model_s2 = build_prithvi_refiner(
        num_input_channels=6,
        out_channels=256,
        hidden_dim=256,
        num_convnext_blocks=4,
        lora_r=4,
        lora_alpha=16,
        load_in_8bit=False,
        freeze_backbone=True,
        unfreeze_last_n_blocks=2,
        device=device
    )
    
    # Load checkpoint if exists
    checkpoint_path = Path("models/stage2_best_fp16/checkpoint.pt")
    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, device=device)
        model_s2.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✓ Checkpoint loaded")
    else:
        print("⚠ No checkpoint found, using pretrained backbone only")
    
    model_s2.eval()
    
    # Run Prithvi
    print("\nRunning Prithvi refinement...")
    with torch.no_grad():
        features_s2 = model_s2(s2_hls_reflectance)
    
    print(f"✓ Prithvi features: {features_s2.shape}")
    
    # Project back to 6 HLS channels
    proj = nn.Conv2d(features_s2.shape[1], 6, kernel_size=1).to(device)
    with torch.no_grad():
        s2_stage2 = proj(features_s2)
        s2_stage2 = torch.sigmoid(s2_stage2)  # Normalize to [0, 1]
    
    print(f"✓ Stage 2 output: {s2_stage2.shape}")
    print(f"  Range: [{s2_stage2.min():.4f}, {s2_stage2.max():.4f}]")
    
except Exception as e:
    print(f"✗ Stage 2 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "="*80)
print("Visualization")
print("="*80)

# Convert to numpy
sar_np = sar[0]  # VV
s1_12band_np = s2_stage1.cpu().numpy()[0]
s1_hls_np = s2_hls_reflectance.cpu().numpy()[0]
s2_hls_np = s2_stage2.cpu().numpy()[0]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# SAR VV
axes[0, 0].imshow(sar_np, cmap='gray', vmin=-30, vmax=5)
axes[0, 0].set_title(f'SAR VV (dB)\n[{sar_np.min():.1f}, {sar_np.max():.1f}]')
axes[0, 0].axis('off')

# Stage 1: Full 12 bands RGB (B04, B03, B02)
s1_rgb_12 = np.transpose(s1_12band_np[[3, 2, 1], :, :], (1, 2, 0)) / 10000.0
s1_rgb_12 = np.clip(s1_rgb_12, 0, 1)
axes[0, 1].imshow(s1_rgb_12)
axes[0, 1].set_title(f'Stage 1: TerraMind (12 bands)\n[{s1_rgb_12.min():.3f}, {s1_rgb_12.max():.3f}]')
axes[0, 1].axis('off')

# Stage 1: HLS RGB (indices 2, 1, 0 = Red, Green, Blue)
s1_rgb_hls = np.transpose(s1_hls_np[[2, 1, 0], :, :], (1, 2, 0))
s1_rgb_hls = np.clip(s1_rgb_hls, 0, 1)
axes[0, 2].imshow(s1_rgb_hls)
axes[0, 2].set_title(f'Stage 1: HLS 6 bands\n[{s1_rgb_hls.min():.3f}, {s1_rgb_hls.max():.3f}]')
axes[0, 2].axis('off')

# Stage 2: Prithvi RGB
s2_rgb = np.transpose(s2_hls_np[[2, 1, 0], :, :], (1, 2, 0))
s2_rgb = np.clip(s2_rgb, 0, 1)
axes[0, 3].imshow(s2_rgb)
axes[0, 3].set_title(f'Stage 2: Prithvi Refined\n[{s2_rgb.min():.3f}, {s2_rgb.max():.3f}]')
axes[0, 3].axis('off')

# Enhanced versions (gamma correction)
s1_enhanced = np.power(s1_rgb_hls, 0.5)
axes[1, 0].imshow(s1_enhanced)
axes[1, 0].set_title('Stage 1 Enhanced (γ=0.5)')
axes[1, 0].axis('off')

s2_enhanced = np.power(s2_rgb, 0.5)
axes[1, 1].imshow(s2_enhanced)
axes[1, 1].set_title('Stage 2 Enhanced (γ=0.5)')
axes[1, 1].axis('off')

# Difference map
diff = np.abs(s2_rgb - s1_rgb_hls)
axes[1, 2].imshow(diff)
axes[1, 2].set_title(f'|Stage 2 - Stage 1|\nMean diff: {diff.mean():.4f}')
axes[1, 2].axis('off')

# Histograms
axes[1, 3].hist(s1_rgb_hls.flatten(), bins=50, alpha=0.5, label='Stage 1', color='blue')
axes[1, 3].hist(s2_rgb.flatten(), bins=50, alpha=0.5, label='Stage 2', color='red')
axes[1, 3].set_xlabel('Reflectance')
axes[1, 3].set_ylabel('Frequency')
axes[1, 3].set_title('RGB Distribution')
axes[1, 3].legend()
axes[1, 3].grid(True, alpha=0.3)

plt.tight_layout()
output_path = Path("outputs/stage1_stage2_combined.png")
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_path}")

# Save data
np.savez(
    "outputs/stage1_stage2_combined.npz",
    sar=sar,
    s2_stage1_12band=s1_12band_np,
    s2_stage1_hls=s1_hls_np,
    s2_stage2=s2_hls_np
)
print(f"✓ Saved: outputs/stage1_stage2_combined.npz")

print("\n" + "="*80)
print("✓ Combined Stage 1+2 test complete!")
print("="*80)
