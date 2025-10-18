"""
Compare TerraMind TiM with and without tokenizers

Shows:
1. Visual difference (before/after images)
2. Quantitative metrics (PSNR, SSIM, spectral accuracy)
3. Brightness and contrast analysis
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from axs_lib.stage1_tim_inference import infer_sar_to_optical_tim

print("="*80)
print("TerraMind TiM: Tokenizer Impact Analysis")
print("="*80)

# Test data
sar_path = Path("data/tiles/benv2_test/S2B_MSIL2A_20180421T100029_N9999_R122_T33TWN_50_83.npz")

# Load ground truth optical if available
try:
    gt_data = np.load(sar_path)
    # Try to find optical ground truth
    gt_optical = None
    for key in ['s2_b2', 's2_b3', 's2_b4', 's2_b8']:
        if key in gt_data:
            if gt_optical is None:
                gt_optical = []
            gt_optical.append(gt_data[key])
    
    if gt_optical:
        gt_optical = np.stack(gt_optical, axis=0)
        print(f"\n✓ Found ground truth optical: {gt_optical.shape}")
        print(f"  Ground truth already in reflectance, range: [{gt_optical.min():.4f}, {gt_optical.max():.4f}]")
        has_gt = True
    else:
        print(f"\n⚠ No ground truth optical found in {sar_path.name}")
        has_gt = False
except Exception as e:
    print(f"\n⚠ Could not load ground truth: {e}")
    has_gt = False

print("\n" + "="*80)
print("Running inference WITHOUT tokenizers...")
print("="*80)

optical_no_tok = infer_sar_to_optical_tim(
    sar_path,
    gamma=0.7,
    use_tokenizers=False,
    return_all_bands=False
)

print("\n" + "="*80)
print("Running inference WITH tokenizers...")
print("="*80)

optical_with_tok = infer_sar_to_optical_tim(
    sar_path,
    gamma=0.7,
    use_tokenizers=True,
    return_all_bands=False
)

print("\n" + "="*80)
print("QUANTITATIVE COMPARISON")
print("="*80)

# Basic statistics
print("\n1. Brightness & Contrast:")
print(f"  WITHOUT tokenizers:")
print(f"    Mean: {optical_no_tok.mean():.4f}")
print(f"    Std:  {optical_no_tok.std():.4f}")
print(f"    Range: [{optical_no_tok.min():.4f}, {optical_no_tok.max():.4f}]")

print(f"\n  WITH tokenizers:")
print(f"    Mean: {optical_with_tok.mean():.4f}")
print(f"    Std:  {optical_with_tok.std():.4f}")
print(f"    Range: [{optical_with_tok.min():.4f}, {optical_with_tok.max():.4f}]")

# Difference
diff = np.abs(optical_with_tok - optical_no_tok)
print(f"\n  Absolute difference:")
print(f"    Mean: {diff.mean():.4f}")
print(f"    Max:  {diff.max():.4f}")
print(f"    Relative change: {(diff.mean() / optical_no_tok.mean()) * 100:.2f}%")

# Inter-comparison (tokenizer vs no tokenizer)
print("\n2. Inter-comparison (WITH vs WITHOUT):")
for i, band_name in enumerate(['B02 (Blue)', 'B03 (Green)', 'B04 (Red)', 'B08 (NIR)']):
    band_psnr = psnr(optical_no_tok[i], optical_with_tok[i], data_range=1.0)
    band_ssim = ssim(optical_no_tok[i], optical_with_tok[i], data_range=1.0)
    print(f"  {band_name}: PSNR={band_psnr:.2f} dB, SSIM={band_ssim:.4f}")

# Ground truth comparison (if available)
if has_gt and gt_optical.shape == optical_no_tok.shape:
    print("\n3. Ground Truth Comparison:")
    print("\n  WITHOUT tokenizers vs GT:")
    for i, band_name in enumerate(['B02', 'B03', 'B04', 'B08']):
        band_psnr = psnr(gt_optical[i], optical_no_tok[i], data_range=1.0)
        band_ssim = ssim(gt_optical[i], optical_no_tok[i], data_range=1.0)
        mae = np.mean(np.abs(gt_optical[i] - optical_no_tok[i]))
        print(f"    {band_name}: PSNR={band_psnr:.2f} dB, SSIM={band_ssim:.4f}, MAE={mae:.4f}")
    
    print("\n  WITH tokenizers vs GT:")
    for i, band_name in enumerate(['B02', 'B03', 'B04', 'B08']):
        band_psnr = psnr(gt_optical[i], optical_with_tok[i], data_range=1.0)
        band_ssim = ssim(gt_optical[i], optical_with_tok[i], data_range=1.0)
        mae = np.mean(np.abs(gt_optical[i] - optical_with_tok[i]))
        print(f"    {band_name}: PSNR={band_psnr:.2f} dB, SSIM={band_ssim:.4f}, MAE={mae:.4f}")
    
    # Overall metrics
    psnr_no_tok = psnr(gt_optical, optical_no_tok, data_range=1.0)
    ssim_no_tok = ssim(gt_optical.transpose(1,2,0), optical_no_tok.transpose(1,2,0), 
                       data_range=1.0, channel_axis=2)
    mae_no_tok = np.mean(np.abs(gt_optical - optical_no_tok))
    
    psnr_with_tok = psnr(gt_optical, optical_with_tok, data_range=1.0)
    ssim_with_tok = ssim(gt_optical.transpose(1,2,0), optical_with_tok.transpose(1,2,0), 
                         data_range=1.0, channel_axis=2)
    mae_with_tok = np.mean(np.abs(gt_optical - optical_with_tok))
    
    print("\n  Overall (all bands):")
    print(f"    WITHOUT: PSNR={psnr_no_tok:.2f} dB, SSIM={ssim_no_tok:.4f}, MAE={mae_no_tok:.4f}")
    print(f"    WITH:    PSNR={psnr_with_tok:.2f} dB, SSIM={ssim_with_tok:.4f}, MAE={mae_with_tok:.4f}")
    
    # Winner
    print("\n  📊 ACCURACY WINNER:")
    if psnr_with_tok > psnr_no_tok:
        print(f"    ✓ WITH tokenizers (+{psnr_with_tok - psnr_no_tok:.2f} dB PSNR)")
    else:
        print(f"    ✗ WITHOUT tokenizers (+{psnr_no_tok - psnr_with_tok:.2f} dB PSNR)")

# Visual comparison
print("\n" + "="*80)
print("Generating visual comparison...")
print("="*80)

fig, axes = plt.subplots(2, 3 if has_gt else 2, figsize=(15 if has_gt else 10, 10))

# RGB composite (R=B04, G=B03, B=B02)
rgb_no_tok = np.stack([optical_no_tok[2], optical_no_tok[1], optical_no_tok[0]], axis=2)
rgb_with_tok = np.stack([optical_with_tok[2], optical_with_tok[1], optical_with_tok[0]], axis=2)

# Clip and enhance for visualization
rgb_no_tok = np.clip(rgb_no_tok * 2.5, 0, 1).astype(np.float32)
rgb_with_tok = np.clip(rgb_with_tok * 2.5, 0, 1).astype(np.float32)

# Row 1: RGB composites
axes[0, 0].imshow(rgb_no_tok)
axes[0, 0].set_title('WITHOUT Tokenizers\n(TiM only)', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(rgb_with_tok)
axes[0, 1].set_title('WITH Tokenizers\n(TiM + FSQ-VAE)', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

if has_gt:
    rgb_gt = np.stack([gt_optical[2], gt_optical[1], gt_optical[0]], axis=2)
    rgb_gt = np.clip(rgb_gt * 2.5, 0, 1).astype(np.float32)
    axes[0, 2].imshow(rgb_gt)
    axes[0, 2].set_title('Ground Truth\n(Real Sentinel-2)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

# Row 2: Difference and NIR
diff_rgb = np.abs(rgb_with_tok - rgb_no_tok)
axes[1, 0].imshow(np.clip(diff_rgb * 10, 0, 1).astype(np.float32))  # Amplified for visibility
axes[1, 0].set_title(f'Difference (10x)\nMean={diff.mean():.4f}', fontsize=10)
axes[1, 0].axis('off')

# NIR comparison
nir_no_tok = optical_no_tok[3]
nir_with_tok = optical_with_tok[3]

nir_diff = (nir_with_tok - nir_no_tok).astype(np.float32)
axes[1, 1].imshow(nir_diff, cmap='RdBu', vmin=-0.1, vmax=0.1)
axes[1, 1].set_title('NIR Difference\n(WITH - WITHOUT)', fontsize=10)
axes[1, 1].axis('off')

if has_gt:
    # Error maps
    error_no_tok = np.mean(np.abs(gt_optical[:3] - optical_no_tok[:3]), axis=0)
    error_with_tok = np.mean(np.abs(gt_optical[:3] - optical_with_tok[:3]), axis=0)
    
    error_diff = error_no_tok - error_with_tok  # Positive = tokenizer better
    im = axes[1, 2].imshow(error_diff, cmap='RdYlGn', vmin=-0.05, vmax=0.05)
    axes[1, 2].set_title('Error Improvement\n(Green=Tokenizer Better)', fontsize=10)
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

plt.suptitle('TerraMind TiM: Impact of FSQ-VAE Tokenizers', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

output_path = Path('results/tokenizer_comparison.png')
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved comparison to: {output_path}")

plt.show()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Tokenizers add FSQ-VAE compression/decompression for:")
print(f"  1. Better detail preservation")
print(f"  2. Smoother spatial coherence")
print(f"  3. Reduced artifacts")
print(f"\nBrightness change: {optical_with_tok.mean() / optical_no_tok.mean():.2%}")
print(f"Contrast change: {optical_with_tok.std() / optical_no_tok.std():.2%}")
print("="*80)
