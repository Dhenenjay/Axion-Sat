"""
Gamma Correction Sweep - Find optimal brightness

Test different gamma values to find the best balance between
brightness and accuracy.
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
print("Gamma Correction Sweep")
print("="*80)

# Load data
sar_path = Path("data/tiles/benv2_test/S2B_MSIL2A_20180421T100029_N9999_R122_T33TWN_50_83.npz")
print(f"\nLoading data from: {sar_path}")
data = np.load(sar_path)

# Extract SAR
s1_vv = data['s1_vv']
s1_vh = data['s1_vh']
sar = np.stack([s1_vv, s1_vh], axis=0)

# Load metadata
metadata_path = sar_path.with_suffix('.json')
with open(metadata_path) as f:
    metadata = json.load(f)

s1_norm = metadata['s1_normalization']
sar_db = sar * (s1_norm['max'] - s1_norm['min']) + s1_norm['min']

# Extract ground truth
gt_s2 = np.stack([
    data['s2_b2'],
    data['s2_b3'],
    data['s2_b4'],
    data['s2_b8']
], axis=0).astype(np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# ============================================================================
# Stage 1: TerraMind Generation
# ============================================================================
print("="*80)
print("Running Stage 1")
print("="*80)

sar_tensor = torch.from_numpy(sar_db).float().unsqueeze(0).to(device)
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
    
    # Extract 4 bands
    s2_4band = s2_output[:, [1,2,3,7], :, :].detach().cpu().numpy()[0] / 10000.0
    
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# ============================================================================
# Test Multiple Gamma Values
# ============================================================================
print("\n" + "="*80)
print("Testing Gamma Values")
print("="*80)

gamma_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
results = []

from sklearn.metrics import mean_absolute_error, mean_squared_error

gt_rgb = np.transpose(gt_s2[[2, 1, 0], :, :], (1, 2, 0))
gt_rgb = np.clip(gt_rgb, 0, 1)

for gamma in gamma_values:
    corrected = np.power(s2_4band, gamma)
    corrected_rgb = np.transpose(corrected[[2, 1, 0], :, :], (1, 2, 0))
    corrected_rgb = np.clip(corrected_rgb, 0, 1)
    
    mae = mean_absolute_error(gt_rgb.flatten(), corrected_rgb.flatten())
    rmse = np.sqrt(mean_squared_error(gt_rgb.flatten(), corrected_rgb.flatten()))
    r2 = np.corrcoef(gt_rgb.flatten(), corrected_rgb.flatten())[0, 1]**2
    mean_brightness = corrected_rgb.mean()
    
    results.append({
        'gamma': gamma,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'brightness': mean_brightness,
        'image': corrected_rgb
    })
    
    print(f"γ={gamma:.1f}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, Mean={mean_brightness:.4f}")

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "="*80)
print("Creating Visualization")
print("="*80)

fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Row 1: Ground truth and raw
sar_vv_np = sar_db[0]
axes[0, 0].imshow(sar_vv_np, cmap='gray', vmin=-30, vmax=5)
axes[0, 0].set_title(f'SAR VV (dB)', fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(gt_rgb)
axes[0, 1].set_title(f'Ground Truth\nMean={gt_rgb.mean():.3f}', fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

s1_rgb_raw = np.transpose(s2_4band[[2, 1, 0], :, :], (1, 2, 0))
s1_rgb_raw = np.clip(s1_rgb_raw, 0, 1)
axes[0, 2].imshow(s1_rgb_raw)
axes[0, 2].set_title(f'Stage 1 RAW\nMean={s1_rgb_raw.mean():.3f}', fontsize=11, fontweight='bold')
axes[0, 2].axis('off')

# Histogram comparison
axes[0, 3].hist(gt_rgb.flatten(), bins=50, alpha=0.6, color='blue', label='GT')
axes[0, 3].hist(s1_rgb_raw.flatten(), bins=50, alpha=0.6, color='red', label='Raw')
axes[0, 3].set_xlabel('Reflectance', fontsize=10)
axes[0, 3].set_title('Distribution', fontsize=11, fontweight='bold')
axes[0, 3].legend()
axes[0, 3].grid(True, alpha=0.3)

# Row 2 & 3: Gamma corrected versions
for idx, result in enumerate(results):
    row = 1 + idx // 4
    col = idx % 4
    
    axes[row, col].imshow(result['image'])
    title = f"γ={result['gamma']:.1f}\n"
    title += f"MAE={result['mae']:.4f}\n"
    title += f"Mean={result['brightness']:.3f}"
    axes[row, col].set_title(title, fontsize=10)
    axes[row, col].axis('off')

plt.suptitle('Gamma Correction Sweep: Finding Optimal Brightness', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

output_path = Path("outputs/gamma_sweep.png")
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_path}")

# ============================================================================
# Find Best Gamma
# ============================================================================
print("\n" + "="*80)
print("Analysis")
print("="*80)

# Find gamma that matches GT brightness best
gt_brightness = gt_rgb.mean()
brightness_diffs = [abs(r['brightness'] - gt_brightness) for r in results]
best_brightness_idx = np.argmin(brightness_diffs)

# Find gamma with best MAE
maes = [r['mae'] for r in results]
best_mae_idx = np.argmin(maes)

# Find balanced gamma (good MAE + good brightness)
# Normalize both metrics to 0-1 and find minimum combined score
mae_normalized = np.array(maes) / max(maes)
brightness_diff_normalized = np.array(brightness_diffs) / max(brightness_diffs)
combined_scores = 0.5 * mae_normalized + 0.5 * brightness_diff_normalized
best_balanced_idx = np.argmin(combined_scores)

print(f"\nGround Truth Brightness: {gt_brightness:.4f}")
print(f"\nBest for Brightness Match: γ={results[best_brightness_idx]['gamma']:.1f}")
print(f"  Mean: {results[best_brightness_idx]['brightness']:.4f}")
print(f"  MAE: {results[best_brightness_idx]['mae']:.4f}")

print(f"\nBest for Accuracy (MAE): γ={results[best_mae_idx]['gamma']:.1f}")
print(f"  Mean: {results[best_mae_idx]['brightness']:.4f}")
print(f"  MAE: {results[best_mae_idx]['mae']:.4f}")

print(f"\n✓ RECOMMENDED (Balanced): γ={results[best_balanced_idx]['gamma']:.1f}")
print(f"  Mean: {results[best_balanced_idx]['brightness']:.4f}")
print(f"  MAE: {results[best_balanced_idx]['mae']:.4f}")
print(f"  R²: {results[best_balanced_idx]['r2']:.4f}")

print("\n" + "="*80)
print("Summary:")
print(f"- Use γ={results[best_balanced_idx]['gamma']:.1f} for best overall quality")
print(f"- Lower gamma (0.4-0.5) = brighter but less accurate")
print(f"- Higher gamma (0.8-1.0) = darker but more accurate")
print("="*80)
