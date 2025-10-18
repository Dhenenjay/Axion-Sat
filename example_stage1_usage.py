"""
Example: Using Stage 1 Inference Module

This demonstrates the various ways to use the production-ready
Stage 1 SAR-to-Optical inference module.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Import the inference module
from axs_lib.stage1_inference import infer_sar_to_optical

print("="*80)
print("Stage 1 Inference - Example Usage")
print("="*80)

# ============================================================================
# Example 1: Load from .npz file
# ============================================================================
print("\n" + "="*80)
print("Example 1: Load from .npz file")
print("="*80)

sar_path = Path("data/tiles/benv2_test/S2B_MSIL2A_20180421T100029_N9999_R122_T33TWN_50_83.npz")

# Simple one-liner
optical_12band = infer_sar_to_optical(
    sar_path,
    gamma=0.6,  # Optimal brightness
    return_all_bands=True  # Get all 12 S2L2A bands
)

print(f"\n✓ Output shape: {optical_12band.shape}")
print(f"  12 bands: B01-B12")
print(f"  Range: [{optical_12band.min():.1f}, {optical_12band.max():.1f}] DN")

# ============================================================================
# Example 2: Get 4-band RGB+NIR only
# ============================================================================
print("\n" + "="*80)
print("Example 2: Get 4-band RGB+NIR subset")
print("="*80)

optical_4band = infer_sar_to_optical(
    sar_path,
    gamma=0.6,
    return_all_bands=False  # Get 4 bands only
)

print(f"\n✓ Output shape: {optical_4band.shape}")
print(f"  4 bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)")
print(f"  Range: [{optical_4band.min():.4f}, {optical_4band.max():.4f}] reflectance")

# ============================================================================
# Example 3: From numpy array with custom gamma
# ============================================================================
print("\n" + "="*80)
print("Example 3: From numpy array with different gamma")
print("="*80)

# Load SAR as numpy
data = np.load(sar_path)
sar_array = np.stack([data['s1_vv'], data['s1_vh']], axis=0)

# Try different gamma values
gammas = [0.5, 0.6, 0.7]
results = {}

for g in gammas:
    optical = infer_sar_to_optical(
        sar_array,
        gamma=g,
        input_scale='normalized',  # Data is in [0, 1]
        return_all_bands=False
    )
    results[g] = optical
    print(f"  γ={g}: mean={optical.mean():.3f}")

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "="*80)
print("Creating Visualization")
print("="*80)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Load ground truth for comparison
gt_s2 = np.stack([
    data['s2_b2'],
    data['s2_b3'],
    data['s2_b4'],
    data['s2_b8']
], axis=0).astype(np.float32)

gt_rgb = np.transpose(gt_s2[[2, 1, 0], :, :], (1, 2, 0))
gt_rgb = np.clip(gt_rgb, 0, 1)

axes[0].imshow(gt_rgb)
axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
axes[0].axis('off')

for idx, (g, optical) in enumerate(results.items(), start=1):
    rgb = np.transpose(optical[[2, 1, 0], :, :], (1, 2, 0))
    rgb = np.clip(rgb, 0, 1)
    
    axes[idx].imshow(rgb)
    axes[idx].set_title(f'Stage 1 (γ={g})\nMean={rgb.mean():.3f}', 
                       fontsize=12, fontweight='bold')
    axes[idx].axis('off')

plt.suptitle('Stage 1 Inference: Different Gamma Values', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = Path("outputs/example_stage1_usage.png")
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_path}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("Summary")
print("="*80)

print("""
✓ Stage 1 Inference Module Ready!

Key Features:
- Flexible input: .npz, .npy, .tif files or numpy arrays
- Automatic scale detection (dB, linear, normalized)
- Gamma correction for optimal brightness
- Returns 4 or 12 bands as needed
- Model caching for efficiency

Recommended Settings:
- gamma=0.6 for best visual quality
- return_all_bands=True for full S2L2A output
- input_scale='auto' for automatic detection

Usage Examples:
  1. Simple:
     optical = infer_sar_to_optical('sar.tif')
  
  2. Custom gamma:
     optical = infer_sar_to_optical(sar_array, gamma=0.7)
  
  3. RGB+NIR only:
     optical = infer_sar_to_optical('sar.npz', return_all_bands=False)

Command Line:
  python -m axs_lib.stage1_inference sar.tif -o output.tif -g 0.6
""")

print("="*80)
