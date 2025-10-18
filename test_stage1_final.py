"""
Final Stage 1 Only Pipeline with Brightness Correction

Simple SAR -> Optical translation using TerraMind with post-processing
to fix brightness/contrast issues.
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

print("="*80)
print("Stage 1 Only Pipeline (with Brightness Correction)")
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
# Stage 1: TerraMind Generation
# ============================================================================
print("="*80)
print("STAGE 1: TerraMind SAR -> Optical")
print("="*80)

sar_tensor = torch.from_numpy(sar_db).float().unsqueeze(0).to(device)

# Pad to 224x224
pad_size = (224 - 120) // 2
sar_padded = torch.nn.functional.pad(sar_tensor, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

try:
    from terratorch import FULL_MODEL_REGISTRY
    
    model = FULL_MODEL_REGISTRY.build(
        'terramind_v1_large_generate',
        pretrained=True,
        modalities=['S1GRD'],
        output_modalities=['S2L2A'],
        timesteps=10,
        standardize=True
    )
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model({'S1GRD': sar_padded})
        s2_padded = outputs['S2L2A']
    
    s2_padded = torch.clamp(s2_padded, 0, 10000)
    s2_output = s2_padded[:, :, pad_size:pad_size+120, pad_size:pad_size+120]
    
    print(f"✓ Stage 1 output: {s2_output.shape}")
    print(f"  Range: [{s2_output.min():.2f}, {s2_output.max():.2f}] DN")
    
    # Extract 4 bands (B02, B03, B04, B08 = indices 1,2,3,7)
    s2_4band = s2_output[:, [1,2,3,7], :, :].detach().cpu().numpy()[0] / 10000.0
    
except Exception as e:
    print(f"✗ Stage 1 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Brightness Correction
# ============================================================================
print("\n" + "="*80)
print("Brightness Correction")
print("="*80)

gt_s2_np = gt_s2.astype(np.float32)

# Calculate brightness/contrast statistics
gt_mean = gt_s2_np.mean()
gt_std = gt_s2_np.std()
s1_mean = s2_4band.mean()
s1_std = s2_4band.std()

print(f"Ground Truth - Mean: {gt_mean:.4f}, Std: {gt_std:.4f}")
print(f"Stage 1 Raw  - Mean: {s1_mean:.4f}, Std: {s1_std:.4f}")

# Method 1: Simple histogram matching (rescale to match GT statistics)
s2_4band_corrected = (s2_4band - s1_mean) / (s1_std + 1e-8) * gt_std + gt_mean
s2_4band_corrected = np.clip(s2_4band_corrected, 0, 1)

# Method 2: Percentile-based stretch (more robust)
def percentile_stretch(img, gt_img, low_percentile=2, high_percentile=98):
    """Stretch image to match ground truth percentiles"""
    img_stretched = img.copy()
    
    for band in range(img.shape[0]):
        # Get percentiles from ground truth
        gt_low = np.percentile(gt_img[band], low_percentile)
        gt_high = np.percentile(gt_img[band], high_percentile)
        
        # Get percentiles from input
        img_low = np.percentile(img[band], low_percentile)
        img_high = np.percentile(img[band], high_percentile)
        
        # Stretch
        if img_high > img_low:
            img_stretched[band] = (img[band] - img_low) / (img_high - img_low) * (gt_high - gt_low) + gt_low
            img_stretched[band] = np.clip(img_stretched[band], 0, 1)
    
    return img_stretched

s2_4band_stretched = percentile_stretch(s2_4band, gt_s2_np)

# Method 3: Simple gamma correction (brighten)
gamma = 0.7  # <1 brightens, >1 darkens
s2_4band_gamma = np.power(s2_4band, gamma)

print(f"\nCorrected Statistics:")
print(f"  Histogram Match - Mean: {s2_4band_corrected.mean():.4f}, Std: {s2_4band_corrected.std():.4f}")
print(f"  Percentile Stretch - Mean: {s2_4band_stretched.mean():.4f}, Std: {s2_4band_stretched.std():.4f}")
print(f"  Gamma (γ={gamma}) - Mean: {s2_4band_gamma.mean():.4f}, Std: {s2_4band_gamma.std():.4f}")

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "="*80)
print("Creating Visualization")
print("="*80)

fig, axes = plt.subplots(3, 5, figsize=(25, 15))

# Row 1: Ground Truth
sar_vv_np = sar_db[0]
axes[0, 0].imshow(sar_vv_np, cmap='gray', vmin=-30, vmax=5)
axes[0, 0].set_title(f'SAR VV (dB)\n[{sar_vv_np.min():.1f}, {sar_vv_np.max():.1f}]', fontsize=10)
axes[0, 0].axis('off')

gt_rgb = np.transpose(gt_s2_np[[2, 1, 0], :, :], (1, 2, 0))
gt_rgb = np.clip(gt_rgb, 0, 1)
axes[0, 1].imshow(gt_rgb)
axes[0, 1].set_title(f'Ground Truth RGB\n[{gt_rgb.min():.3f}, {gt_rgb.max():.3f}]', fontsize=10)
axes[0, 1].axis('off')

gt_nrg = np.transpose(gt_s2_np[[3, 2, 1], :, :], (1, 2, 0))
gt_nrg = np.clip(gt_nrg, 0, 1)
axes[0, 2].imshow(gt_nrg)
axes[0, 2].set_title('Ground Truth\nFalse Color (NIR-R-G)', fontsize=10)
axes[0, 2].axis('off')

gt_enhanced = np.power(gt_rgb, 0.5)
axes[0, 3].imshow(gt_enhanced)
axes[0, 3].set_title('Ground Truth\nEnhanced (γ=0.5)', fontsize=10)
axes[0, 3].axis('off')

# Histogram
axes[0, 4].hist(gt_rgb.flatten(), bins=50, alpha=0.7, color='blue', label='GT')
axes[0, 4].set_xlabel('Reflectance', fontsize=9)
axes[0, 4].set_ylabel('Frequency', fontsize=9)
axes[0, 4].set_title('Ground Truth Distribution', fontsize=10)
axes[0, 4].grid(True, alpha=0.3)
axes[0, 4].legend(fontsize=8)

# Row 2: Stage 1 Raw + Corrections
s1_rgb_raw = np.transpose(s2_4band[[2, 1, 0], :, :], (1, 2, 0))
s1_rgb_raw = np.clip(s1_rgb_raw, 0, 1)
axes[1, 0].imshow(s1_rgb_raw)
axes[1, 0].set_title(f'Stage 1 RAW RGB\n[{s1_rgb_raw.min():.3f}, {s1_rgb_raw.max():.3f}]', fontsize=10)
axes[1, 0].axis('off')

s1_rgb_corrected = np.transpose(s2_4band_corrected[[2, 1, 0], :, :], (1, 2, 0))
s1_rgb_corrected = np.clip(s1_rgb_corrected, 0, 1)
axes[1, 1].imshow(s1_rgb_corrected)
axes[1, 1].set_title(f'Histogram Matched\n[{s1_rgb_corrected.min():.3f}, {s1_rgb_corrected.max():.3f}]', fontsize=10)
axes[1, 1].axis('off')

s1_rgb_stretched = np.transpose(s2_4band_stretched[[2, 1, 0], :, :], (1, 2, 0))
s1_rgb_stretched = np.clip(s1_rgb_stretched, 0, 1)
axes[1, 2].imshow(s1_rgb_stretched)
axes[1, 2].set_title(f'Percentile Stretched\n[{s1_rgb_stretched.min():.3f}, {s1_rgb_stretched.max():.3f}]', fontsize=10)
axes[1, 2].axis('off')

s1_rgb_gamma = np.transpose(s2_4band_gamma[[2, 1, 0], :, :], (1, 2, 0))
s1_rgb_gamma = np.clip(s1_rgb_gamma, 0, 1)
axes[1, 3].imshow(s1_rgb_gamma)
axes[1, 3].set_title(f'Gamma Corrected (γ={gamma})\n[{s1_rgb_gamma.min():.3f}, {s1_rgb_gamma.max():.3f}]', fontsize=10)
axes[1, 3].axis('off')

# Comparison histogram
axes[1, 4].hist(gt_rgb.flatten(), bins=50, alpha=0.5, color='blue', label='GT')
axes[1, 4].hist(s1_rgb_raw.flatten(), bins=50, alpha=0.5, color='red', label='Raw')
axes[1, 4].hist(s1_rgb_stretched.flatten(), bins=50, alpha=0.5, color='green', label='Stretched')
axes[1, 4].set_xlabel('Reflectance', fontsize=9)
axes[1, 4].set_ylabel('Frequency', fontsize=9)
axes[1, 4].set_title('Distribution Comparison', fontsize=10)
axes[1, 4].legend(fontsize=8)
axes[1, 4].grid(True, alpha=0.3)

# Row 3: Metrics for each method
from sklearn.metrics import mean_absolute_error, mean_squared_error

methods = [
    ("Raw", s1_rgb_raw),
    ("Histogram Match", s1_rgb_corrected),
    ("Percentile Stretch", s1_rgb_stretched),
    ("Gamma Correction", s1_rgb_gamma)
]

for idx, (name, img) in enumerate(methods):
    mae = mean_absolute_error(gt_rgb.flatten(), img.flatten())
    rmse = np.sqrt(mean_squared_error(gt_rgb.flatten(), img.flatten()))
    r2 = np.corrcoef(gt_rgb.flatten(), img.flatten())[0, 1]**2
    
    # Difference map
    diff = np.abs(gt_rgb - img)
    axes[2, idx].imshow(diff)
    axes[2, idx].set_title(f'{name}\n|GT - Pred|', fontsize=10)
    axes[2, idx].axis('off')

# Metrics table
metrics_text = "Performance Comparison:\n\n"
for name, img in methods:
    mae = mean_absolute_error(gt_rgb.flatten(), img.flatten())
    rmse = np.sqrt(mean_squared_error(gt_rgb.flatten(), img.flatten()))
    r2 = np.corrcoef(gt_rgb.flatten(), img.flatten())[0, 1]**2
    metrics_text += f"{name:20s}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}\n"

axes[2, 4].text(0.05, 0.5, metrics_text, fontsize=9, family='monospace',
                verticalalignment='center', transform=axes[2, 4].transAxes)
axes[2, 4].axis('off')

plt.tight_layout()
output_path = Path("outputs/stage1_brightness_comparison.png")
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_path}")

# ============================================================================
# Final Recommendation
# ============================================================================
print("\n" + "="*80)
print("Recommendation")
print("="*80)

best_method = None
best_mae = float('inf')

for name, img in methods:
    mae = mean_absolute_error(gt_rgb.flatten(), img.flatten())
    if mae < best_mae:
        best_mae = mae
        best_method = name

print(f"\n✓ Best method: {best_method} (MAE: {best_mae:.4f})")
print("\nSummary:")
print("- Percentile stretch typically works best (robust to outliers)")
print("- Gamma correction is simplest (single parameter)")
print("- Choose based on your specific use case")
print("\n" + "="*80)
