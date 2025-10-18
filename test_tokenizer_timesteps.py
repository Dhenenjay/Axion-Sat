"""
Test impact of tokenizer diffusion timesteps on quality

Tests timesteps: 5, 10, 20, 50 to find optimal balance
"""

from pathlib import Path
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

from axs_lib.stage1_tim_inference import infer_sar_to_optical_tim

print("="*80)
print("TerraMind Tokenizer: Timestep Optimization")
print("="*80)

sar_path = Path("data/tiles/benv2_test/S2B_MSIL2A_20180421T100029_N9999_R122_T33TWN_50_83.npz")

# Load ground truth
gt_data = np.load(sar_path)
gt_optical = np.stack([gt_data['s2_b2'], gt_data['s2_b3'], 
                       gt_data['s2_b4'], gt_data['s2_b8']], axis=0)

# Baseline: no tokenizer
print("\nBaseline (no tokenizer):")
baseline = infer_sar_to_optical_tim(sar_path, gamma=0.7, use_tokenizers=False, return_all_bands=False)
psnr_baseline = psnr(gt_optical, baseline, data_range=1.0)
ssim_baseline = ssim(gt_optical.transpose(1,2,0), baseline.transpose(1,2,0), 
                     data_range=1.0, channel_axis=2)
print(f"  PSNR: {psnr_baseline:.2f} dB")
print(f"  SSIM: {ssim_baseline:.4f}")
print(f"  Mean: {baseline.mean():.4f}")

# Test different timesteps
timesteps_list = [5, 10, 20, 50]
results = []

for ts in timesteps_list:
    print(f"\n{'='*80}")
    print(f"Testing timesteps={ts}...")
    print(f"{'='*80}")
    
    # Monkey-patch the timesteps
    import axs_lib.stage1_tim_inference as stage1_module
    
    # Run inference with modified timesteps
    from axs_lib.stage1_tim_inference import _get_model_and_tokenizers, _preprocess_sar
    import torch
    
    model, tokenizers, device = _get_model_and_tokenizers()
    sar_padded, (orig_h, orig_w) = _preprocess_sar(sar_path, 'auto', device)
    
    # Run TiM
    outputs = model({'S1GRD': sar_padded})
    s2_padded = torch.clamp(outputs['S2L2A'], 0, 10000)
    
    # Tokenizer refinement with custom timesteps
    tokenizer_s2 = tokenizers['s2l2a']
    s2_normalized = s2_padded / 10000.0
    quantized_s2, _, tokens_s2 = tokenizer_s2.encode(s2_normalized)
    s2_reconstructed = tokenizer_s2.decode_tokens(tokens_s2, timesteps=ts, verbose=False)
    s2_padded = torch.clamp(s2_reconstructed * 10000.0, 0, 10000)
    
    # Crop and convert
    if orig_h != 224 or orig_w != 224:
        pad_h = (224 - orig_h) // 2
        pad_w = (224 - orig_w) // 2
        s2_output = s2_padded[:, :, pad_h:pad_h+orig_h, pad_w:pad_w+orig_w]
    else:
        s2_output = s2_padded
    
    s2_numpy = s2_output.cpu().numpy()[0]
    
    # Gamma correction
    s2_reflectance = s2_numpy / 10000.0
    s2_corrected = np.power(s2_reflectance, 0.7)
    
    # Extract 4 bands
    optical = s2_corrected[[1, 2, 3, 7], :, :]
    
    # Metrics
    psnr_val = psnr(gt_optical, optical, data_range=1.0)
    ssim_val = ssim(gt_optical.transpose(1,2,0), optical.transpose(1,2,0), 
                    data_range=1.0, channel_axis=2)
    
    results.append({
        'timesteps': ts,
        'psnr': psnr_val,
        'ssim': ssim_val,
        'mean': optical.mean(),
        'optical': optical
    })
    
    print(f"  PSNR: {psnr_val:.2f} dB (Δ{psnr_val - psnr_baseline:+.2f})")
    print(f"  SSIM: {ssim_val:.4f} (Δ{ssim_val - ssim_baseline:+.4f})")
    print(f"  Mean: {optical.mean():.4f}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"{'Timesteps':<12} {'PSNR (dB)':<12} {'SSIM':<12} {'Δ vs Baseline'}")
print("-"*60)
print(f"{'Baseline':<12} {psnr_baseline:<12.2f} {ssim_baseline:<12.4f} {'---'}")
for r in results:
    delta = r['psnr'] - psnr_baseline
    marker = "✓" if delta > 0 else "✗"
    print(f"{r['timesteps']:<12} {r['psnr']:<12.2f} {r['ssim']:<12.4f} {marker} {delta:+.2f} dB")

# Find best
best = max(results, key=lambda x: x['psnr'])
print("\n🏆 BEST CONFIGURATION:")
print(f"  Timesteps: {best['timesteps']}")
print(f"  PSNR: {best['psnr']:.2f} dB ({best['psnr'] - psnr_baseline:+.2f} vs baseline)")
print(f"  SSIM: {best['ssim']:.4f}")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

images = [baseline] + [r['optical'] for r in results]
titles = ['Baseline\n(No Tokenizer)'] + [f'Timesteps={r["timesteps"]}\nPSNR={r["psnr"]:.2f}' for r in results]

for idx, (img, title) in enumerate(zip(images, titles)):
    row = idx // 3
    col = idx % 3
    
    rgb = np.stack([img[2], img[1], img[0]], axis=2)
    rgb = np.clip(rgb * 2.5, 0, 1)
    
    axes[row, col].imshow(rgb)
    axes[row, col].set_title(title, fontweight='bold' if img is best['optical'] else None)
    axes[row, col].axis('off')

# Hide last subplot
axes[1, 2].axis('off')

plt.suptitle('Tokenizer Diffusion Timesteps Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = Path('results/tokenizer_timesteps.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved comparison to: {output_path}")

plt.show()

print("="*80)
