"""
Test Stage 1 (TerraMind) in isolation to debug output issues.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*80)
print("Stage 1 Standalone Test - TerraMind SAR to Optical")
print("="*80)

# Load SAR data
sar_path = Path("data/tiles/benv2_test/S2B_MSIL2A_20180421T100029_N9999_R122_T33TWN_50_83.npz")
print(f"\nLoading SAR data from: {sar_path}")
data = np.load(sar_path)

# Extract SAR bands
s1_vv = data.get('s1_vv', data.get('VV', None))
s1_vh = data.get('s1_vh', data.get('VH', None))
sar = np.stack([s1_vv, s1_vh], axis=0)

print(f"\nOriginal SAR (normalized):")
print(f"  Shape: {sar.shape}")
print(f"  Range: [{sar.min():.4f}, {sar.max():.4f}]")
print(f"  Mean: {sar.mean():.4f}, std: {sar.std():.4f}")

# Load metadata for denormalization
metadata_path = sar_path.with_suffix('.json')
if metadata_path.exists():
    import json
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    s1_norm = metadata['s1_normalization']
    sar_min = s1_norm['min']
    sar_max = s1_norm['max']
    sar_mean = s1_norm['mean']
    sar_std = s1_norm['std']
    
    print(f"\nMetadata normalization stats (dB):")
    print(f"  Min: {sar_min:.2f} dB")
    print(f"  Max: {sar_max:.2f} dB")
    print(f"  Mean: {sar_mean:.2f} dB")
    print(f"  Std: {sar_std:.2f} dB")
    
    # Denormalize SAR back to dB scale
    # Formula: sar_db = sar_normalized * (max - min) + min
    sar_db = sar * (sar_max - sar_min) + sar_min
    
    print(f"\nDenormalized SAR (dB):")
    print(f"  Range: [{sar_db.min():.2f}, {sar_db.max():.2f}] dB")
    print(f"  Mean: {sar_db.mean():.2f} dB, std: {sar_db.std():.2f} dB")
    
    sar = sar_db
else:
    print(f"\n⚠ Warning: Metadata file not found at {metadata_path}")
    print("  Using normalized SAR [0, 1] - output may be dark")

sar_tensor = torch.from_numpy(sar).float().unsqueeze(0)

# Check if we have ground truth optical
if 's2' in data.keys():
    gt_s2 = data['s2']
    print(f"\nGround truth S2 available: {gt_s2.shape}")
    print(f"GT S2 range: [{gt_s2.min():.4f}, {gt_s2.max():.4f}]")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Pad to 224x224
original_size = sar_tensor.shape[2:]
pad_size = (224 - 120) // 2
sar_padded = torch.nn.functional.pad(sar_tensor, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
sar_padded = sar_padded.to(device)

print(f"\nPadded SAR shape: {sar_padded.shape}")

# Test 1: Try to list available models
print("\n" + "="*80)
print("Test 1: Check available TerraMind models")
print("="*80)

try:
    from terratorch import FULL_MODEL_REGISTRY
    print("FULL_MODEL_REGISTRY imported successfully")
    
    # Try to build the model
    print("\nAttempting to build terramind_v1_large_generate...")
    model = FULL_MODEL_REGISTRY.build(
        'terramind_v1_large_generate',
        pretrained=True,
        modalities=['S1GRD'],
        output_modalities=['S2L2A'],
        timesteps=10,
        standardize=True
    )
    print("✓ Model built successfully")
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    # Check model structure
    print(f"\nModel type: {type(model)}")
    print(f"Model on device: {next(model.parameters()).device}")
    
    # Test inference
    print("\n" + "="*80)
    print("Test 2: Running inference")
    print("="*80)
    
    with torch.no_grad():
        outputs = model({'S1GRD': sar_padded})
    
    print(f"Output type: {type(outputs)}")
    if isinstance(outputs, dict):
        print(f"Output keys: {outputs.keys()}")
        s2_output = outputs['S2L2A']
    else:
        s2_output = outputs
    
    print(f"S2 output shape: {s2_output.shape}")
    print(f"S2 output range (raw): [{s2_output.min():.4f}, {s2_output.max():.4f}]")
    print(f"S2 output mean: {s2_output.mean():.4f}, std: {s2_output.std():.4f}")
    
    # Clip negative values (shouldn't exist in reflectance)
    s2_output = torch.clamp(s2_output, 0, 10000)
    print(f"S2 output range (clipped): [{s2_output.min():.4f}, {s2_output.max():.4f}]")
    
    # Crop back
    s2_cropped = s2_output[:, :, pad_size:pad_size+120, pad_size:pad_size+120]
    print(f"Cropped S2 shape: {s2_cropped.shape}")
    
    # Convert to numpy
    s2_np = s2_cropped.cpu().numpy()[0]
    
    # Visualize
    print("\n" + "="*80)
    print("Test 3: Visualization")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # SAR VV
    axes[0, 0].imshow(sar[0], cmap='gray')
    axes[0, 0].set_title(f'SAR VV\nRange: [{sar[0].min():.2f}, {sar[0].max():.2f}]')
    axes[0, 0].axis('off')
    
    # SAR VH
    axes[0, 1].imshow(sar[1], cmap='gray')
    axes[0, 1].set_title(f'SAR VH\nRange: [{sar[1].min():.2f}, {sar[1].max():.2f}]')
    axes[0, 1].axis('off')
    
    # TerraMind RGB (using bands B04, B03, B02 - indices 3, 2, 1)
    if s2_np.shape[0] >= 4:
        rgb = np.transpose(s2_np[[3, 2, 1], :, :], (1, 2, 0))
        rgb = np.clip(rgb / 10000.0, 0, 1)  # Normalize from DN to reflectance
        axes[0, 2].imshow(rgb)
        axes[0, 2].set_title(f'TerraMind RGB\nRange: [{rgb.min():.4f}, {rgb.max():.4f}]')
    axes[0, 2].axis('off')
    
    # TerraMind RGB - enhanced (gamma correction)
    rgb_enhanced = np.power(rgb, 0.5)  # Gamma correction
    axes[1, 0].imshow(rgb_enhanced)
    axes[1, 0].set_title(f'TerraMind RGB (enhanced)\nGamma=0.5')
    axes[1, 0].axis('off')
    
    # TerraMind false color (NIR, Red, Green)
    if s2_np.shape[0] >= 8:
        false_color = np.transpose(s2_np[[7, 3, 2], :, :], (1, 2, 0))
        false_color = np.clip(false_color / 10000.0, 0, 1)
        axes[1, 1].imshow(false_color)
        axes[1, 1].set_title(f'False Color (NIR-R-G)\nRange: [{false_color.min():.4f}, {false_color.max():.4f}]')
    axes[1, 1].axis('off')
    
    # Histogram of RGB values
    axes[1, 2].hist(rgb.flatten(), bins=50, alpha=0.7, label='RGB')
    axes[1, 2].set_xlabel('Reflectance')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('RGB Histogram')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path("outputs/stage1_debug.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    
    # Save data
    np.savez(
        "outputs/stage1_debug.npz",
        sar=sar,
        s2_output=s2_np,
        rgb=rgb,
        rgb_enhanced=rgb_enhanced
    )
    print(f"✓ Saved data to: outputs/stage1_debug.npz")
    
    print("\n" + "="*80)
    print("✓ Stage 1 test complete!")
    print("="*80)
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
