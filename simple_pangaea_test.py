"""
Simple Pangaea Test Script

Quick test to see if we can beat DOFA (0.7715) with inference-time tricks.
Tests on a small dataset without needing full data downloads.
"""

import os
import sys

# Check if we're in the right directory
pangaea_dir = r"C:\Users\Dhenenjay\Axion-Sat\pangaea-bench"
os.chdir(pangaea_dir)
sys.path.insert(0, pangaea_dir)

print("=" * 80)
print("🛰️  AXION-SAT SIMPLE PANGAEA TEST")
print("=" * 80)
print()

# Test 1: Check environment
print("Step 1: Checking environment...")
try:
    import torch
    import numpy as np
    from pangaea.engine.tta_utils import TTAWrapper
    from pangaea.engine.multiscale_utils import MultiScaleWrapper
    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"✓ TTA and MultiScale modules loaded")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

print()

# Test 2: List available datasets
print("Step 2: Checking available datasets...")
import glob
config_path = os.path.join(pangaea_dir, "configs", "dataset", "*.yaml")
dataset_configs = glob.glob(config_path)
print(f"Found {len(dataset_configs)} dataset configs:")
for i, config in enumerate(dataset_configs[:10], 1):
    dataset_name = os.path.basename(config).replace(".yaml", "")
    print(f"  {i}. {dataset_name}")

print()

# Test 3: Check TerraMind encoder
print("Step 3: Checking TerraMind encoder config...")
terramind_config = os.path.join(pangaea_dir, "configs", "encoder", "terramind_large.yaml")
if os.path.exists(terramind_config):
    print(f"✓ TerraMind config found: {terramind_config}")
    with open(terramind_config, 'r') as f:
        print("Config contents:")
        print(f.read())
else:
    print("❌ TerraMind config not found")

print()
print("=" * 80)
print("Environment check complete!")
print()
print("Next steps:")
print("1. Download a small dataset (e.g., MADOS or HLSBurnScars)")
print("2. Run baseline: python run_axion_benchmark.py --dataset mados")
print("3. Run with TTA: python run_axion_benchmark.py --dataset mados --use_tta")
print("4. Run with both: python run_axion_benchmark.py --dataset mados --use_tta --use_multiscale")
print("=" * 80)
