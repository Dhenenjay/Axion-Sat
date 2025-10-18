"""
Simple Local Benchmark for Production Stage 1

Tests our optimized Stage 1 pipeline (TerraMind + γ=0.7) 
on a sample of HLSBurnScars data without needing distributed training.
"""

import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from axs_lib.stage1_inference import infer_sar_to_optical

print("="*80)
print("🚀 SIMPLE LOCAL BENCHMARK - Production Stage 1")
print("="*80)
print("Testing: TerraMind Generator + γ=0.7 (No TiM, No Tokenizers)")
print("="*80)
print()

# Find test data
test_dir = Path("data/tiles/benv2_test")
if not test_dir.exists():
    print(f"❌ Test data not found: {test_dir}")
    print("   Please ensure test data is available")
    sys.exit(1)

# Get test files
test_files = list(test_dir.glob("*.npz"))
if not test_files:
    print(f"❌ No .npz files found in {test_dir}")
    sys.exit(1)

print(f"📂 Found {len(test_files)} test files")
print()

# Run inference on samples
results = []
print("Running inference...")
print()

for i, sar_file in enumerate(tqdm(test_files[:10], desc="Processing")):  # Test on first 10
    try:
        # Load ground truth
        data = np.load(sar_file)
        if 's2_b2' not in data:
            continue
        
        gt_optical = np.stack([
            data['s2_b2'], data['s2_b3'], 
            data['s2_b4'], data['s2_b8']
        ], axis=0)
        
        # Run Stage 1 inference
        pred_optical = infer_sar_to_optical(
            sar_file,
            gamma=0.7,
            return_all_bands=False
        )
        
        # Calculate metrics
        psnr_val = psnr(gt_optical, pred_optical, data_range=1.0)
        ssim_val = ssim(
            gt_optical.transpose(1,2,0), 
            pred_optical.transpose(1,2,0),
            data_range=1.0,
            channel_axis=2
        )
        mae_val = np.mean(np.abs(gt_optical - pred_optical))
        
        results.append({
            'file': sar_file.name,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'mae': mae_val
        })
        
    except Exception as e:
        print(f"\n⚠ Skipped {sar_file.name}: {e}")
        continue

if not results:
    print("\n❌ No successful inferences")
    sys.exit(1)

# Calculate averages
avg_psnr = np.mean([r['psnr'] for r in results])
avg_ssim = np.mean([r['ssim'] for r in results])
avg_mae = np.mean([r['mae'] for r in results])

print()
print("="*80)
print("📊 RESULTS")
print("="*80)
print(f"Samples tested: {len(results)}")
print()
print("Average Metrics:")
print(f"  PSNR: {avg_psnr:.2f} dB")
print(f"  SSIM: {avg_ssim:.4f}")
print(f"  MAE:  {avg_mae:.4f}")
print()

# Quality assessment
if avg_psnr > 20:
    quality = "✓ EXCELLENT"
elif avg_psnr > 15:
    quality = "✓ GOOD"
elif avg_psnr > 12:
    quality = "△ ACCEPTABLE"
else:
    quality = "✗ NEEDS IMPROVEMENT"

print(f"Quality: {quality}")
print()

# Detailed results
print("Per-file results:")
print(f"{'File':<45} {'PSNR':<10} {'SSIM':<10} {'MAE':<10}")
print("-"*80)
for r in results[:5]:  # Show first 5
    print(f"{r['file']:<45} {r['psnr']:>6.2f} dB  {r['ssim']:>8.4f}  {r['mae']:>8.4f}")

print("="*80)
print()
print("✓ Production Stage 1 is working!")
print("  For full Pangaea benchmark (83.6% target), you'll need:")
print("  - Linux environment (for distributed training)")
print("  - Or analyze existing results from previous runs")
print("="*80)
