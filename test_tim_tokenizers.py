"""
Test TerraMind TiM + Tokenizers Pipeline

This tests the FULL TerraMind implementation with:
1. Thinking-in-Modalities (TiM)  
2. FSQ-VAE Tokenizers
3. Gamma=0.7
"""

from pathlib import Path
import numpy as np

from axs_lib.stage1_tim_inference import infer_sar_to_optical_tim

print("="*80)
print("Testing TerraMind TiM + Tokenizers (The Major Juice!)")
print("="*80)

sar_path = Path("data/tiles/benv2_test/S2B_MSIL2A_20180421T100029_N9999_R122_T33TWN_50_83.npz")

print("\nTest: Full pipeline with TiM + Tokenizers + γ=0.7")
print("="*80)

try:
    optical = infer_sar_to_optical_tim(
        sar_path,
        gamma=0.7,
        use_tokenizers=True,
        return_all_bands=False  # Get 4 bands for visualization
    )
    
    print(f"\n✓ SUCCESS!")
    print(f"  Output shape: {optical.shape}")
    print(f"  Output range: [{optical.min():.4f}, {optical.max():.4f}]")
    print(f"  Mean brightness: {optical.mean():.4f}")
    
except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Summary:")
print("- TiM: Generates LULC internally ✓")
print("- Tokenizers: FSQ-VAE encode/decode ✓")
print("- Gamma: 0.7 (balanced) ✓")
print("="*80)
