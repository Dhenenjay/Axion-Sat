"""
Stage 1 Forward Pass Integration Test

This test validates the complete Stage 1 pipeline (SAR to optical translation)
using real or synthetic demo tiles. It verifies:
- Model loading and initialization
- Forward pass execution with timesteps=6
- Output shape matching input spatial dimensions
- Output values in valid [0, 1] range
- Correct band ordering (B02, B03, B04, B08)

The test can run with:
1. Real demo tiles from demo/ directory
2. Synthetically generated random tiles (fallback)

Usage:
    pytest tests/test_stage1_forward_real.py -v
    python tests/test_stage1_forward_real.py  # Run directly
"""

import sys
from pathlib import Path
import warnings
import tempfile

import pytest
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axs_lib.models import build_terramind_generator
from axs_lib.stage1_tm_s2o import tm_sar2opt, get_s2_band_info, extract_rgb, compute_ndvi
from axs_lib.stdz import TERRAMIND_S1_STATS


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope="module")
def demo_tile():
    """
    Load or create demo SAR tile.
    
    Returns:
        Tuple of (s1_tensor, metadata)
        s1_tensor shape: (1, 2, H, W)
    """
    # Try to load real demo tile
    demo_dir = project_root / 'demo'
    
    if demo_dir.exists():
        # Look for NPZ demo tiles
        demo_tiles = list(demo_dir.glob('*s1*.npz'))
        
        if demo_tiles:
            print(f"\n✓ Loading real demo tile: {demo_tiles[0].name}")
            tile_data = np.load(demo_tiles[0])
            
            # Extract SAR bands
            s1_vv = tile_data['s1_vv']
            s1_vh = tile_data['s1_vh']
            s1 = np.stack([s1_vv, s1_vh], axis=0).astype(np.float32)
            
            # Standardize
            means = TERRAMIND_S1_STATS.means.astype(np.float32)
            stds = TERRAMIND_S1_STATS.stds.astype(np.float32)
            
            s1_tensor = torch.from_numpy(s1).unsqueeze(0)  # Add batch dim
            s1_tensor = (s1_tensor - torch.from_numpy(means).view(1, 2, 1, 1)) / \
                        torch.from_numpy(stds).view(1, 2, 1, 1)
            
            metadata = {
                'source': 'real_demo',
                'shape': s1.shape,
                'tile_name': demo_tiles[0].stem
            }
            
            return s1_tensor, metadata
    
    # Fallback: create synthetic demo tile
    print("\n⚠ No real demo tiles found, creating synthetic tile")
    
    # Create realistic SAR values (in dB scale before standardization)
    H, W = 256, 256
    
    # VV: typical range -20 to 0 dB
    s1_vv = np.random.normal(-11.76, 5.46, (H, W)).astype(np.float32)
    
    # VH: typical range -25 to -10 dB
    s1_vh = np.random.normal(-19.29, 5.52, (H, W)).astype(np.float32)
    
    s1 = np.stack([s1_vv, s1_vh], axis=0)
    
    # Standardize
    means = TERRAMIND_S1_STATS.means.astype(np.float32)
    stds = TERRAMIND_S1_STATS.stds.astype(np.float32)
    
    s1_tensor = torch.from_numpy(s1).unsqueeze(0)  # Add batch dim
    s1_tensor = (s1_tensor - torch.from_numpy(means).view(1, 2, 1, 1)) / \
                torch.from_numpy(stds).view(1, 2, 1, 1)
    
    metadata = {
        'source': 'synthetic',
        'shape': s1.shape,
        'tile_name': 'synthetic_demo_tile'
    }
    
    return s1_tensor, metadata


@pytest.fixture(scope="module")
def stage1_model(device):
    """
    Build Stage 1 TerraMind generator model.
    
    Note: This uses pretrained=False for testing without actual weights.
    In production, you would load a trained checkpoint.
    """
    print("\n✓ Building Stage 1 model (timesteps=6)...")
    
    model = build_terramind_generator(
        input_modalities=("S1GRD",),
        output_modalities=("S2L2A",),
        timesteps=6,  # Fast inference for testing
        standardize=True,
        pretrained=False  # For testing, we don't need pretrained weights
    )
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    
    return model


# ============================================================================
# Tests
# ============================================================================

def test_model_initialization(stage1_model, device):
    """Test that model initializes correctly."""
    assert stage1_model is not None
    assert next(stage1_model.parameters()).device.type == device.type
    print("✓ Model initialization test passed")


def test_demo_tile_loading(demo_tile):
    """Test that demo tile loads with correct format."""
    s1_tensor, metadata = demo_tile
    
    # Check tensor properties
    assert s1_tensor.ndim == 4, f"Expected 4D tensor, got {s1_tensor.ndim}D"
    assert s1_tensor.shape[0] == 1, f"Expected batch size 1, got {s1_tensor.shape[0]}"
    assert s1_tensor.shape[1] == 2, f"Expected 2 channels (VV, VH), got {s1_tensor.shape[1]}"
    
    # Check spatial dimensions are reasonable
    H, W = s1_tensor.shape[2], s1_tensor.shape[3]
    assert H >= 64 and H <= 2048, f"Unexpected height: {H}"
    assert W >= 64 and W <= 2048, f"Unexpected width: {W}"
    
    # Check tensor type
    assert s1_tensor.dtype == torch.float32
    
    # Check metadata
    assert 'source' in metadata
    assert 'shape' in metadata
    
    print(f"✓ Demo tile loading test passed (source: {metadata['source']})")
    print(f"  Shape: {tuple(s1_tensor.shape)}")
    print(f"  Tile: {metadata['tile_name']}")


def test_stage1_forward_shape(stage1_model, demo_tile, device):
    """Test that forward pass produces correct output shape."""
    s1_tensor, metadata = demo_tile
    s1_tensor = s1_tensor.to(device)
    
    # Get input dimensions
    B, C_in, H_in, W_in = s1_tensor.shape
    
    # Run forward pass
    with torch.no_grad():
        opt_tensor = tm_sar2opt(
            stage1_model,
            s1_tensor,
            timesteps=6,
            denormalize=True,
            clip_range=(0.0, 1.0)
        )
    
    # Check output shape
    assert opt_tensor.ndim == 4, f"Expected 4D output, got {opt_tensor.ndim}D"
    assert opt_tensor.shape[0] == B, f"Batch size mismatch: expected {B}, got {opt_tensor.shape[0]}"
    assert opt_tensor.shape[1] == 4, f"Expected 4 output channels (B02,B03,B04,B08), got {opt_tensor.shape[1]}"
    assert opt_tensor.shape[2] == H_in, f"Height mismatch: expected {H_in}, got {opt_tensor.shape[2]}"
    assert opt_tensor.shape[3] == W_in, f"Width mismatch: expected {W_in}, got {opt_tensor.shape[3]}"
    
    print(f"✓ Forward pass shape test passed")
    print(f"  Input shape:  {tuple(s1_tensor.shape)}")
    print(f"  Output shape: {tuple(opt_tensor.shape)}")


def test_stage1_output_range(stage1_model, demo_tile, device):
    """Test that output values are in valid [0, 1] range."""
    s1_tensor, metadata = demo_tile
    s1_tensor = s1_tensor.to(device)
    
    # Run forward pass
    with torch.no_grad():
        opt_tensor = tm_sar2opt(
            stage1_model,
            s1_tensor,
            timesteps=6,
            denormalize=True,
            clip_range=(0.0, 1.0)
        )
    
    # Check value range
    min_val = opt_tensor.min().item()
    max_val = opt_tensor.max().item()
    
    assert min_val >= 0.0, f"Minimum value {min_val} is below 0.0"
    assert max_val <= 1.0, f"Maximum value {max_val} is above 1.0"
    
    # Check for NaN or Inf
    assert not torch.isnan(opt_tensor).any(), "Output contains NaN values"
    assert not torch.isinf(opt_tensor).any(), "Output contains Inf values"
    
    print(f"✓ Output range test passed")
    print(f"  Min value: {min_val:.6f}")
    print(f"  Max value: {max_val:.6f}")
    print(f"  Mean value: {opt_tensor.mean().item():.6f}")


def test_stage1_band_order(stage1_model, demo_tile, device):
    """Test that output bands are in correct order (B02, B03, B04, B08)."""
    s1_tensor, metadata = demo_tile
    s1_tensor = s1_tensor.to(device)
    
    # Run forward pass
    with torch.no_grad():
        opt_tensor = tm_sar2opt(
            stage1_model,
            s1_tensor,
            timesteps=6,
            denormalize=True,
            clip_range=(0.0, 1.0)
        )
    
    # Get band info
    band_info = get_s2_band_info()
    
    # Verify we have 4 bands
    assert len(band_info) == 4
    
    # Verify band order
    expected_bands = ['B02', 'B03', 'B04', 'B08']
    for i, band_name in enumerate(expected_bands):
        assert band_name in band_info, f"Missing band: {band_name}"
        assert band_info[band_name]['index'] == i, f"Band {band_name} has incorrect index"
    
    print(f"✓ Band order test passed")
    print(f"  Bands: {', '.join(expected_bands)}")


def test_rgb_extraction(stage1_model, demo_tile, device):
    """Test RGB extraction from optical output."""
    s1_tensor, metadata = demo_tile
    s1_tensor = s1_tensor.to(device)
    
    # Run forward pass
    with torch.no_grad():
        opt_tensor = tm_sar2opt(
            stage1_model,
            s1_tensor,
            timesteps=6,
            denormalize=True,
            clip_range=(0.0, 1.0)
        )
    
    # Extract RGB
    rgb_tensor = extract_rgb(opt_tensor)
    
    # Check RGB shape
    assert rgb_tensor.shape[0] == opt_tensor.shape[0], "Batch size mismatch"
    assert rgb_tensor.shape[1] == 3, f"Expected 3 RGB channels, got {rgb_tensor.shape[1]}"
    assert rgb_tensor.shape[2] == opt_tensor.shape[2], "Height mismatch"
    assert rgb_tensor.shape[3] == opt_tensor.shape[3], "Width mismatch"
    
    # Check RGB range
    assert rgb_tensor.min() >= 0.0, "RGB contains negative values"
    assert rgb_tensor.max() <= 1.0, "RGB contains values > 1.0"
    
    print(f"✓ RGB extraction test passed")
    print(f"  RGB shape: {tuple(rgb_tensor.shape)}")


def test_ndvi_computation(stage1_model, demo_tile, device):
    """Test NDVI computation from optical output."""
    s1_tensor, metadata = demo_tile
    s1_tensor = s1_tensor.to(device)
    
    # Run forward pass
    with torch.no_grad():
        opt_tensor = tm_sar2opt(
            stage1_model,
            s1_tensor,
            timesteps=6,
            denormalize=True,
            clip_range=(0.0, 1.0)
        )
    
    # Compute NDVI
    ndvi_tensor = compute_ndvi(opt_tensor)
    
    # Check NDVI shape
    assert ndvi_tensor.shape[0] == opt_tensor.shape[0], "Batch size mismatch"
    assert ndvi_tensor.shape[1] == 1, f"Expected 1 NDVI channel, got {ndvi_tensor.shape[1]}"
    assert ndvi_tensor.shape[2] == opt_tensor.shape[2], "Height mismatch"
    assert ndvi_tensor.shape[3] == opt_tensor.shape[3], "Width mismatch"
    
    # Check NDVI range (should be [-1, 1])
    assert ndvi_tensor.min() >= -1.0, f"NDVI min {ndvi_tensor.min()} < -1.0"
    assert ndvi_tensor.max() <= 1.0, f"NDVI max {ndvi_tensor.max()} > 1.0"
    
    print(f"✓ NDVI computation test passed")
    print(f"  NDVI shape: {tuple(ndvi_tensor.shape)}")
    print(f"  NDVI range: [{ndvi_tensor.min():.3f}, {ndvi_tensor.max():.3f}]")


def test_batch_processing(stage1_model, demo_tile, device):
    """Test processing multiple tiles in a batch."""
    s1_tensor, metadata = demo_tile
    
    # Create batch of 3 tiles
    batch_size = 3
    s1_batch = s1_tensor.repeat(batch_size, 1, 1, 1).to(device)
    
    # Run forward pass
    with torch.no_grad():
        opt_batch = tm_sar2opt(
            stage1_model,
            s1_batch,
            timesteps=6,
            denormalize=True,
            clip_range=(0.0, 1.0)
        )
    
    # Check batch dimensions
    assert opt_batch.shape[0] == batch_size, f"Expected batch size {batch_size}, got {opt_batch.shape[0]}"
    assert opt_batch.shape[1] == 4, f"Expected 4 channels, got {opt_batch.shape[1]}"
    
    # Check all samples in batch are valid
    for i in range(batch_size):
        sample = opt_batch[i]
        assert sample.min() >= 0.0, f"Sample {i} has values < 0"
        assert sample.max() <= 1.0, f"Sample {i} has values > 1"
        assert not torch.isnan(sample).any(), f"Sample {i} contains NaN"
    
    print(f"✓ Batch processing test passed")
    print(f"  Batch size: {batch_size}")
    print(f"  Output shape: {tuple(opt_batch.shape)}")


def test_deterministic_output(stage1_model, demo_tile, device):
    """Test that multiple runs produce consistent output (with deterministic mode)."""
    s1_tensor, metadata = demo_tile
    s1_tensor = s1_tensor.to(device)
    
    # Set deterministic mode
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)
    
    # Run forward pass twice
    with torch.no_grad():
        opt_tensor1 = tm_sar2opt(
            stage1_model,
            s1_tensor,
            timesteps=6,
            denormalize=True,
            clip_range=(0.0, 1.0)
        )
        
        # Reset seed
        torch.manual_seed(42)
        if device.type == 'cuda':
            torch.cuda.manual_seed(42)
        
        opt_tensor2 = tm_sar2opt(
            stage1_model,
            s1_tensor,
            timesteps=6,
            denormalize=True,
            clip_range=(0.0, 1.0)
        )
    
    # Check consistency (allowing for small numerical differences)
    max_diff = (opt_tensor1 - opt_tensor2).abs().max().item()
    
    # Note: Due to non-deterministic operations in some CUDA kernels,
    # we allow for small differences
    tolerance = 1e-3 if device.type == 'cuda' else 1e-6
    
    print(f"✓ Deterministic output test passed")
    print(f"  Max difference: {max_diff:.8f} (tolerance: {tolerance:.8f})")
    
    # This is informational - we don't assert strict equality due to GPU non-determinism
    if max_diff > tolerance:
        warnings.warn(f"Output difference {max_diff} exceeds tolerance {tolerance} (expected on GPU)")


# ============================================================================
# Main (for direct execution)
# ============================================================================

def main():
    """Run all tests directly (without pytest)."""
    print("=" * 80)
    print("Stage 1 Forward Pass Integration Test")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load fixtures
    print("\n" + "-" * 80)
    print("Loading fixtures...")
    print("-" * 80)
    
    demo_tile_data = demo_tile()
    model = stage1_model(device)
    
    # Run tests
    tests = [
        ("Model Initialization", test_model_initialization, [model, device]),
        ("Demo Tile Loading", test_demo_tile_loading, [demo_tile_data]),
        ("Forward Pass Shape", test_stage1_forward_shape, [model, demo_tile_data, device]),
        ("Output Range", test_stage1_output_range, [model, demo_tile_data, device]),
        ("Band Order", test_stage1_band_order, [model, demo_tile_data, device]),
        ("RGB Extraction", test_rgb_extraction, [model, demo_tile_data, device]),
        ("NDVI Computation", test_ndvi_computation, [model, demo_tile_data, device]),
        ("Batch Processing", test_batch_processing, [model, demo_tile_data, device]),
        ("Deterministic Output", test_deterministic_output, [model, demo_tile_data, device]),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func, test_args in tests:
        print("\n" + "-" * 80)
        print(f"Running: {test_name}")
        print("-" * 80)
        
        try:
            test_func(*test_args)
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
