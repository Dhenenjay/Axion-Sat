"""Create synthetic SAR test data for inference demo."""
import numpy as np
from pathlib import Path

# Create synthetic SAR data (120x120)
np.random.seed(42)
s1_vv = np.random.randn(120, 120).astype(np.float32) * 5 - 12
s1_vh = np.random.randn(120, 120).astype(np.float32) * 5 - 18

# Save as NPZ
output_path = Path('data/test_sar.npz')
output_path.parent.mkdir(parents=True, exist_ok=True)

np.savez(output_path, s1_vv=s1_vv, s1_vh=s1_vh)
print(f"✓ Created test SAR data: {output_path}")
print(f"  Shape: VV={s1_vv.shape}, VH={s1_vh.shape}")
