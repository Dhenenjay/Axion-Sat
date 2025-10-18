"""
Test Stage 1 + Stage 2 with ground truth comparison.

This shows:
1. Ground truth Sentinel-2 optical
2. Stage 1: TerraMind SAR → Optical generation
3. Stage 2: Prithvi refinement
4. Comparisons and metrics
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

print("="*80)
print("Stage 1 + Stage 2 + Ground Truth Comparison")
print("="*80)

# Load data
sar_path = Path("data/tiles/benv2_test/S2B_MSIL2A_20180421T100029_N9999_R122_T33TWN_50_83.npz")
print(f"\nLoading data from: {sar_path}")
data = np.load(sar_path)

# Extract SAR and denormalize
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
print(f"Ground truth S2: {gt_s2.shape}, range [{gt_s2.min():.0f}, {gt_s2.max():.0f}] DN")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# ============================================================================
# STAGE 1: TerraMind
# ============================================================================
print("="*80)
print("STAGE 1: TerraMind Generation")
print("="*80)

sar_tensor = torch.from_numpy(sar_db).float().unsqueeze(0).to(device)
pad_size = (224 - 120) // 2
sar_padded = torch.nn.functional.pad(sar_tensor, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

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
    print(f"  Range: [{s2_stage1.min():.2f}, {s2_stage1.max():.2f}] DN")
    
    # Extract the same 4 bands as ground truth (B02, B03, B04, B08 = indices 1,2,3,7)
    s2_stage1_4band = s2_stage1[:, [1,2,3,7], :, :]
    
    # Extract 6 HLS bands for Stage 2 (B02, B03, B04, B08, B11, B12 = indices 1,2,3,7,10,11)
    s2_stage1_6band = s2_stage1[:, [1,2,3,7,10,11], :, :]
    
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

try:
    from axs_lib.stage2_prithvi_refine import build_prithvi_refiner
    
    # Build Stage 2 refiner (expects 6 HLS bands)
    print("Building Prithvi refiner...")
    local_checkpoint = Path("weights/hf/Prithvi-EO-2.0-600M/Prithvi_EO_V2_600M.pt")
    
    model_s2 = build_prithvi_refiner(
        prithvi_checkpoint=str(local_checkpoint) if local_checkpoint.exists() else None,
        num_input_channels=6,  # 6 HLS bands from Stage 1
        out_channels=256,      # Feature dimension for Stage 3
        hidden_dim=256,
        num_convnext_blocks=4,
        lora_r=8,
        lora_alpha=16,
        load_in_8bit=False,    # Disable 8-bit for now (H100 has plenty VRAM)
        freeze_backbone=True,
        unfreeze_last_n_blocks=2,
        device=device,
        prithvi_model='prithvi_eo_v2_600'  # Use local 600M model
    )
    model_s2.eval()
    
    # Normalize Stage 1 output to [0, 1] for Prithvi
    s2_stage1_6band_norm = s2_stage1_6band / 10000.0
    
    print(f"Stage 2 input: {s2_stage1_6band_norm.shape}")
    print(f"  Range: [{s2_stage1_6band_norm.min():.4f}, {s2_stage1_6band_norm.max():.4f}]")
    
    # Run Stage 2
    with torch.no_grad():
        s2_stage2_features = model_s2(s2_stage1_6band_norm)
    
    print(f"✓ Stage 2 output: {s2_stage2_features.shape}")
    print(f"  Range: [{s2_stage2_features.min():.4f}, {s2_stage2_features.max():.4f}]")
    
    # For visualization, project Stage 2 features back to 4-band RGB+NIR space
    # This is just for visualization - in practice, Stage 3 would use the features directly
    feature_to_bands = nn.Conv2d(256, 4, kernel_size=1).to(device)
    nn.init.xavier_uniform_(feature_to_bands.weight)
    with torch.no_grad():
        s2_stage2_4band = torch.sigmoid(feature_to_bands(s2_stage2_features))
    
    stage2_success = True
    
except Exception as e:
    print(f"⚠ Stage 2 failed: {e}")
    print("  Continuing with Stage 1 results only...")
    import traceback
    traceback.print_exc()
    stage2_success = False
    s2_stage2_4band = None

# ============================================================================
# Visualization with Ground Truth
# ============================================================================
print("\n" + "="*80)
print("Visualization")
print("="*80)

# Convert to numpy and reflectance
sar_vv_np = sar_db[0]
gt_s2_np = gt_s2.astype(np.float32)  # Already in reflectance [0, 1]
s1_4band_np = s2_stage1_4band.cpu().numpy()[0].astype(np.float32) / 10000.0

if stage2_success:
    s2_4band_np = s2_stage2_4band.cpu().numpy()[0].astype(np.float32)
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
else:
    s2_4band_np = None
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Row 1: SAR and Ground Truth
axes[0, 0].imshow(sar_vv_np, cmap='gray', vmin=-30, vmax=5)
axes[0, 0].set_title(f'SAR VV (dB)\n[{sar_vv_np.min():.1f}, {sar_vv_np.max():.1f}]')
axes[0, 0].axis('off')

# Ground truth RGB (B4, B3, B2 = indices 2, 1, 0)
gt_rgb = np.transpose(gt_s2_np[[2, 1, 0], :, :], (1, 2, 0))
gt_rgb = np.clip(gt_rgb, 0, 1)
axes[0, 1].imshow(gt_rgb)
axes[0, 1].set_title(f'Ground Truth S2 RGB\n[{gt_rgb.min():.3f}, {gt_rgb.max():.3f}]')
axes[0, 1].axis('off')

# Ground truth false color (NIR, R, G = indices 3, 2, 1)
gt_nrg = np.transpose(gt_s2_np[[3, 2, 1], :, :], (1, 2, 0))
gt_nrg = np.clip(gt_nrg, 0, 1)
axes[0, 2].imshow(gt_nrg)
axes[0, 2].set_title('Ground Truth False Color (NIR-R-G)')
axes[0, 2].axis('off')

# GT Enhanced
gt_enhanced = np.power(gt_rgb, 0.5)
axes[0, 3].imshow(gt_enhanced)
axes[0, 3].set_title('Ground Truth Enhanced (γ=0.5)')
axes[0, 3].axis('off')

# Row 2: Stage 1 outputs
s1_rgb = np.transpose(s1_4band_np[[2, 1, 0], :, :], (1, 2, 0))
s1_rgb = np.clip(s1_rgb, 0, 1)
axes[1, 0].imshow(s1_rgb)
axes[1, 0].set_title(f'Stage 1: TerraMind RGB\n[{s1_rgb.min():.3f}, {s1_rgb.max():.3f}]')
axes[1, 0].axis('off')

s1_nrg = np.transpose(s1_4band_np[[3, 2, 1], :, :], (1, 2, 0))
s1_nrg = np.clip(s1_nrg, 0, 1)
axes[1, 1].imshow(s1_nrg)
axes[1, 1].set_title('Stage 1 False Color (NIR-R-G)')
axes[1, 1].axis('off')

s1_enhanced = np.power(s1_rgb, 0.5)
axes[1, 2].imshow(s1_enhanced)
axes[1, 2].set_title('Stage 1 Enhanced (γ=0.5)')
axes[1, 2].axis('off')

# Stage 1 per-band comparison
for i, band_name in enumerate(['B2(Blue)', 'B3(Green)', 'B4(Red)', 'B8(NIR)']):
    if i < 3:
        axes[1, 3].plot(gt_s2_np[i].flatten()[::10], label=f'GT {band_name}', alpha=0.7)
        axes[1, 3].plot(s1_4band_np[i].flatten()[::10], label=f'S1 {band_name}', alpha=0.7, linestyle='--')
axes[1, 3].set_xlabel('Pixel index (sampled)')
axes[1, 3].set_ylabel('Reflectance')
axes[1, 3].set_title('Band Profiles')
axes[1, 3].legend(fontsize=8)
axes[1, 3].grid(True, alpha=0.3)

# Row 3: Stage 2 outputs (if available)
if stage2_success:
    s2_rgb = np.transpose(s2_4band_np[[2, 1, 0], :, :], (1, 2, 0))
    s2_rgb = np.clip(s2_rgb, 0, 1)
    axes[2, 0].imshow(s2_rgb)
    axes[2, 0].set_title(f'Stage 2: Prithvi Refined RGB\n[{s2_rgb.min():.3f}, {s2_rgb.max():.3f}]')
    axes[2, 0].axis('off')
    
    s2_nrg = np.transpose(s2_4band_np[[3, 2, 1], :, :], (1, 2, 0))
    s2_nrg = np.clip(s2_nrg, 0, 1)
    axes[2, 1].imshow(s2_nrg)
    axes[2, 1].set_title('Stage 2 False Color (NIR-R-G)')
    axes[2, 1].axis('off')
    
    s2_enhanced = np.power(s2_rgb, 0.5)
    axes[2, 2].imshow(s2_enhanced)
    axes[2, 2].set_title('Stage 2 Enhanced (γ=0.5)')
    axes[2, 2].axis('off')
    
    # Stage 1 vs Stage 2 difference
    diff_s1_s2 = np.abs(s1_rgb - s2_rgb)
    axes[2, 3].imshow(diff_s1_s2)
    axes[2, 3].set_title(f'|Stage1 - Stage2|\nMAE: {diff_s1_s2.mean():.4f}')
    axes[2, 3].axis('off')
    
    row_offset = 1
else:
    row_offset = 0

# Row 3/4: Comparisons and metrics
metrics_row = 2 + row_offset

diff_rgb = np.abs(gt_rgb - s1_rgb)
axes[metrics_row, 0].imshow(diff_rgb)
axes[metrics_row, 0].set_title(f'|GT - Stage1| RGB\nMAE: {diff_rgb.mean():.4f}')
axes[metrics_row, 0].axis('off')

# Scatter plot GT vs Stage 1 (and Stage 2 if available)
axes[metrics_row, 1].scatter(gt_rgb.flatten()[::100], s1_rgb.flatten()[::100], alpha=0.3, s=1, label='Stage 1')
if stage2_success:
    axes[metrics_row, 1].scatter(gt_rgb.flatten()[::100], s2_rgb.flatten()[::100], alpha=0.3, s=1, label='Stage 2', color='green')
axes[metrics_row, 1].plot([0, 1], [0, 1], 'r--', label='Perfect match')
axes[metrics_row, 1].set_xlabel('Ground Truth')
axes[metrics_row, 1].set_ylabel('Predicted')
axes[metrics_row, 1].set_title('RGB Correlation')
axes[metrics_row, 1].legend()
axes[metrics_row, 1].grid(True, alpha=0.3)

# Histograms
axes[metrics_row, 2].hist(gt_rgb.flatten(), bins=50, alpha=0.5, label='GT', color='blue')
axes[metrics_row, 2].hist(s1_rgb.flatten(), bins=50, alpha=0.5, label='Stage 1', color='orange')
if stage2_success:
    axes[metrics_row, 2].hist(s2_rgb.flatten(), bins=50, alpha=0.5, label='Stage 2', color='green')
axes[metrics_row, 2].set_xlabel('Reflectance')
axes[metrics_row, 2].set_ylabel('Frequency')
axes[metrics_row, 2].set_title('RGB Distribution')
axes[metrics_row, 2].legend()
axes[metrics_row, 2].grid(True, alpha=0.3)

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae_s1 = mean_absolute_error(gt_rgb.flatten(), s1_rgb.flatten())
rmse_s1 = np.sqrt(mean_squared_error(gt_rgb.flatten(), s1_rgb.flatten()))
r2_s1 = np.corrcoef(gt_rgb.flatten(), s1_rgb.flatten())[0, 1]**2

metrics_text = f"""
Stage 1 Performance:
  MAE:  {mae_s1:.4f}
  RMSE: {rmse_s1:.4f}
  R²:   {r2_s1:.4f}

  Per-band MAE:
    B2: {mean_absolute_error(gt_s2_np[0].flatten(), s1_4band_np[0].flatten()):.4f}
    B3: {mean_absolute_error(gt_s2_np[1].flatten(), s1_4band_np[1].flatten()):.4f}
    B4: {mean_absolute_error(gt_s2_np[2].flatten(), s1_4band_np[2].flatten()):.4f}
    B8: {mean_absolute_error(gt_s2_np[3].flatten(), s1_4band_np[3].flatten()):.4f}
"""

if stage2_success:
    mae_s2 = mean_absolute_error(gt_rgb.flatten(), s2_rgb.flatten())
    rmse_s2 = np.sqrt(mean_squared_error(gt_rgb.flatten(), s2_rgb.flatten()))
    r2_s2 = np.corrcoef(gt_rgb.flatten(), s2_rgb.flatten())[0, 1]**2
    
    metrics_text += f"""

Stage 2 Performance:
  MAE:  {mae_s2:.4f}  (Δ: {mae_s2-mae_s1:+.4f})
  RMSE: {rmse_s2:.4f}  (Δ: {rmse_s2-rmse_s1:+.4f})
  R²:   {r2_s2:.4f}  (Δ: {r2_s2-r2_s1:+.4f})

  Per-band MAE:
    B2: {mean_absolute_error(gt_s2_np[0].flatten(), s2_4band_np[0].flatten()):.4f}
    B3: {mean_absolute_error(gt_s2_np[1].flatten(), s2_4band_np[1].flatten()):.4f}
    B4: {mean_absolute_error(gt_s2_np[2].flatten(), s2_4band_np[2].flatten()):.4f}
    B8: {mean_absolute_error(gt_s2_np[3].flatten(), s2_4band_np[3].flatten()):.4f}
"""

axes[metrics_row, 3].text(0.05, 0.5, metrics_text, fontsize=9, family='monospace',
                          verticalalignment='center', transform=axes[metrics_row, 3].transAxes)
axes[metrics_row, 3].axis('off')

plt.tight_layout()
if stage2_success:
    output_path = Path("outputs/stage1_stage2_vs_groundtruth.png")
else:
    output_path = Path("outputs/stage1_vs_groundtruth.png")
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_path}")

print("\n" + "="*80)
print("✓ Analysis complete!")
print("="*80)
print(metrics_text)
