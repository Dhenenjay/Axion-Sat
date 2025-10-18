"""
Test Unified TerraMind Pipeline with Ground Truth

Tests the simplified architecture:
- Stage 1: SAR → Optical (TerraMind)
- Stage 3: SAR + Optical → Grounded (TerraMind backbone)
- Skips Stage 2 (Prithvi) entirely

Compares against ground truth Sentinel-2 imagery.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from axs_lib.unified_terramind_pipeline import build_unified_pipeline

print("="*80)
print("Unified TerraMind Pipeline Test (Stage 1 -> Stage 3)")
print("="*80)

# Load data
sar_path = Path("data/tiles/benv2_test/S2B_MSIL2A_20180421T100029_N9999_R122_T33TWN_50_83.npz")
print(f"\nLoading data from: {sar_path}")
data = np.load(sar_path)

# Extract SAR and denormalize to dB
s1_vv = data['s1_vv']
s1_vh = data['s1_vh']
sar = np.stack([s1_vv, s1_vh], axis=0)

# Load metadata
metadata_path = sar_path.with_suffix('.json')
with open(metadata_path) as f:
    metadata = json.load(f)

s1_norm = metadata['s1_normalization']
sar_db = sar * (s1_norm['max'] - s1_norm['min']) + s1_norm['min']

# Extract ground truth S2 (4 bands: B2, B3, B4, B8)
gt_s2 = np.stack([
    data['s2_b2'],
    data['s2_b3'],
    data['s2_b4'],
    data['s2_b8']
], axis=0)

print(f"SAR (dB): [{sar_db.min():.2f}, {sar_db.max():.2f}]")
print(f"Ground truth S2: {gt_s2.shape}, range [{gt_s2.min():.3f}, {gt_s2.max():.3f}]")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# ============================================================================
# Build Unified Pipeline
# ============================================================================
print("="*80)
print("Building Unified Pipeline")
print("="*80)

try:
    pipeline = build_unified_pipeline(
        use_stage2=False,  # Skip Prithvi - use Stage 1 output directly
        use_stage3=True,   # Enable SAR grounding
        stage1_timesteps=10,
        stage3_freeze_backbone=False,  # Allow fine-tuning
        device=device
    )
    
    print("\n✓ Pipeline ready")
    
except Exception as e:
    print(f"✗ Pipeline build failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Run Inference
# ============================================================================
print("\n" + "="*80)
print("Running Inference")
print("="*80)

sar_tensor = torch.from_numpy(sar_db).float().unsqueeze(0).to(device)
print(f"Input SAR: {sar_tensor.shape}")

try:
    outputs = pipeline(sar_tensor, return_intermediates=True)
    
    print("\n✓ Inference complete")
    print("\nOutputs:")
    for key, val in outputs.items():
        if val is not None:
            print(f"  {key}: {val.shape}, range [{val.min():.2f}, {val.max():.2f}] DN")
    
    # Extract 4-band subsets for comparison (B02, B03, B04, B08 = indices 1,2,3,7)
    stage1_4band = outputs['stage1'][:, [1,2,3,7], :, :].detach().cpu().numpy()[0] / 10000.0
    stage3_4band = outputs['final'][:, [1,2,3,7], :, :].detach().cpu().numpy()[0] / 10000.0
    
except Exception as e:
    print(f"✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "="*80)
print("Creating Visualization")
print("="*80)

sar_vv_np = sar_db[0]
gt_s2_np = gt_s2.astype(np.float32)

fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Row 1: SAR and Ground Truth
axes[0, 0].imshow(sar_vv_np, cmap='gray', vmin=-30, vmax=5)
axes[0, 0].set_title(f'SAR VV (dB)\n[{sar_vv_np.min():.1f}, {sar_vv_np.max():.1f}]')
axes[0, 0].axis('off')

gt_rgb = np.transpose(gt_s2_np[[2, 1, 0], :, :], (1, 2, 0))
gt_rgb = np.clip(gt_rgb, 0, 1)
axes[0, 1].imshow(gt_rgb)
axes[0, 1].set_title(f'Ground Truth S2 RGB\n[{gt_rgb.min():.3f}, {gt_rgb.max():.3f}]')
axes[0, 1].axis('off')

gt_nrg = np.transpose(gt_s2_np[[3, 2, 1], :, :], (1, 2, 0))
gt_nrg = np.clip(gt_nrg, 0, 1)
axes[0, 2].imshow(gt_nrg)
axes[0, 2].set_title('Ground Truth False Color (NIR-R-G)')
axes[0, 2].axis('off')

gt_enhanced = np.power(gt_rgb, 0.5)
axes[0, 3].imshow(gt_enhanced)
axes[0, 3].set_title('Ground Truth Enhanced (γ=0.5)')
axes[0, 3].axis('off')

# Row 2: Stage 1 outputs
s1_rgb = np.transpose(stage1_4band[[2, 1, 0], :, :], (1, 2, 0))
s1_rgb = np.clip(s1_rgb, 0, 1)
axes[1, 0].imshow(s1_rgb)
axes[1, 0].set_title(f'Stage 1: TerraMind RGB\n[{s1_rgb.min():.3f}, {s1_rgb.max():.3f}]')
axes[1, 0].axis('off')

s1_nrg = np.transpose(stage1_4band[[3, 2, 1], :, :], (1, 2, 0))
s1_nrg = np.clip(s1_nrg, 0, 1)
axes[1, 1].imshow(s1_nrg)
axes[1, 1].set_title('Stage 1 False Color (NIR-R-G)')
axes[1, 1].axis('off')

s1_enhanced = np.power(s1_rgb, 0.5)
axes[1, 2].imshow(s1_enhanced)
axes[1, 2].set_title('Stage 1 Enhanced (γ=0.5)')
axes[1, 2].axis('off')

# Band profiles
for i, band_name in enumerate(['B2(Blue)', 'B3(Green)', 'B4(Red)', 'B8(NIR)']):
    axes[1, 3].plot(gt_s2_np[i].flatten()[::10], label=f'GT {band_name}', alpha=0.7)
    axes[1, 3].plot(stage1_4band[i].flatten()[::10], label=f'S1 {band_name}', alpha=0.7, linestyle='--')
axes[1, 3].set_xlabel('Pixel index (sampled)')
axes[1, 3].set_ylabel('Reflectance')
axes[1, 3].set_title('Stage 1 Band Profiles')
axes[1, 3].legend(fontsize=7)
axes[1, 3].grid(True, alpha=0.3)

# Row 3: Stage 3 outputs
s3_rgb = np.transpose(stage3_4band[[2, 1, 0], :, :], (1, 2, 0))
s3_rgb = np.clip(s3_rgb, 0, 1)
axes[2, 0].imshow(s3_rgb)
axes[2, 0].set_title(f'Stage 3: SAR Grounded RGB\n[{s3_rgb.min():.3f}, {s3_rgb.max():.3f}]')
axes[2, 0].axis('off')

s3_nrg = np.transpose(stage3_4band[[3, 2, 1], :, :], (1, 2, 0))
s3_nrg = np.clip(s3_nrg, 0, 1)
axes[2, 1].imshow(s3_nrg)
axes[2, 1].set_title('Stage 3 False Color (NIR-R-G)')
axes[2, 1].axis('off')

s3_enhanced = np.power(s3_rgb, 0.5)
axes[2, 2].imshow(s3_enhanced)
axes[2, 2].set_title('Stage 3 Enhanced (γ=0.5)')
axes[2, 2].axis('off')

# Stage 3 band profiles
for i, band_name in enumerate(['B2(Blue)', 'B3(Green)', 'B4(Red)', 'B8(NIR)']):
    axes[2, 3].plot(gt_s2_np[i].flatten()[::10], label=f'GT {band_name}', alpha=0.7)
    axes[2, 3].plot(stage3_4band[i].flatten()[::10], label=f'S3 {band_name}', alpha=0.7, linestyle='--')
axes[2, 3].set_xlabel('Pixel index (sampled)')
axes[2, 3].set_ylabel('Reflectance')
axes[2, 3].set_title('Stage 3 Band Profiles')
axes[2, 3].legend(fontsize=7)
axes[2, 3].grid(True, alpha=0.3)

plt.tight_layout()
output_path = Path("outputs/unified_pipeline_test.png")
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_path}")

# ============================================================================
# Metrics
# ============================================================================
print("\n" + "="*80)
print("Performance Metrics")
print("="*80)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae_s1 = mean_absolute_error(gt_rgb.flatten(), s1_rgb.flatten())
rmse_s1 = np.sqrt(mean_squared_error(gt_rgb.flatten(), s1_rgb.flatten()))
r2_s1 = np.corrcoef(gt_rgb.flatten(), s1_rgb.flatten())[0, 1]**2

mae_s3 = mean_absolute_error(gt_rgb.flatten(), s3_rgb.flatten())
rmse_s3 = np.sqrt(mean_squared_error(gt_rgb.flatten(), s3_rgb.flatten()))
r2_s3 = np.corrcoef(gt_rgb.flatten(), s3_rgb.flatten())[0, 1]**2

print(f"\nStage 1 Performance:")
print(f"  MAE:  {mae_s1:.4f}")
print(f"  RMSE: {rmse_s1:.4f}")
print(f"  R²:   {r2_s1:.4f}")

print(f"\nStage 3 Performance:")
print(f"  MAE:  {mae_s3:.4f}  (Δ: {mae_s3-mae_s1:+.4f})")
print(f"  RMSE: {rmse_s3:.4f}  (Δ: {rmse_s3-rmse_s1:+.4f})")
print(f"  R²:   {r2_s3:.4f}  (Δ: {r2_s3-r2_s1:+.4f})")

print("\n" + "="*80)
print("✓ Test Complete!")
print("="*80)
print("\nSummary:")
print("- Unified pipeline using TerraMind for all stages")
print("- Stage 2 (Prithvi) skipped - no complexity, no issues")
print("- Stage 3 currently untrained, showing baseline performance")
print("- Next: Train Stage 3 with SAR grounding loss")
