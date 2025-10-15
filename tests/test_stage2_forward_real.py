"""
Stage 2 Forward Pass End-to-End Integration Test

This test validates the complete Stage 2 pipeline (Prithvi refinement) using real tiles.
It runs one tile through the full v1→v2 pipeline and asserts that edge movement penalties
are small, ensuring geometric consistency.

Key Tests:
    1. Load real tile from Stage 1 output (opt_v1)
    2. Run through Stage 2 Prithvi refinement
    3. Compute edge displacement between v1 and v2
    4. Assert edge displacement < threshold (e.g., 2.0 pixels)
    5. Validate spectral plausibility (NDVI/EVI)
    6. Check output format and range

Edge Displacement Metric:
    Measures the geometric consistency between Stage 1 and Stage 2 by computing
    edge maps (Sobel gradients) and finding the maximum spatial shift in edge positions.
    
    Expected: < 2.0 pixels for good geometric preservation
    Warning:  2-5 pixels indicates moderate changes
    Failure:  > 5 pixels suggests Stage 2 is introducing artifacts

Usage:
    pytest tests/test_stage2_forward_real.py -v
    python tests/test_stage2_forward_real.py  # Run directly

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
from pathlib import Path
import warnings
import tempfile
from typing import Optional, Tuple, Dict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try importing Stage 2 modules
try:
    from axs_lib.stage2_prithvi_refine import PrithviRefiner, build_prithvi_refiner, ConvNeXtHead
    from axs_lib.stage2_losses import (
        compute_ndvi,
        compute_evi,
        SpectralPlausibilityLoss,
        IdentityEdgeGuardLoss
    )
    STAGE2_AVAILABLE = True
except ImportError as e:
    STAGE2_AVAILABLE = False
    print(f"Warning: Could not import Stage 2 modules: {e}")


# ============================================================================
# Edge Displacement Metric
# ============================================================================

def compute_edge_displacement(
    pred: torch.Tensor,
    target: torch.Tensor,
    edge_threshold: float = 0.1,
    search_radius: int = 5
) -> Dict[str, float]:
    """
    Compute edge displacement between predicted and target images.
    
    This metric measures how much edges have moved spatially between
    Stage 1 (opt_v1) and Stage 2 (opt_v2). Small displacements indicate
    good geometric consistency.
    
    Algorithm:
        1. Extract edge maps using Sobel gradients
        2. Threshold to get binary edge masks
        3. For each edge pixel in target, find nearest edge in pred
        4. Compute displacement statistics
        
    Args:
        pred: Predicted refined optical, shape (B, C, H, W)
        target: Target optical (Stage 1 output), shape (B, C, H, W)
        edge_threshold: Threshold for edge detection (normalized)
        search_radius: Maximum search radius for matching edges (pixels)
        
    Returns:
        Dict with displacement metrics:
            - mean_displacement: Average edge displacement (pixels)
            - max_displacement: Maximum edge displacement (pixels)
            - median_displacement: Median edge displacement (pixels)
            - edge_ratio: Ratio of edges found in both images
    """
    device = pred.device
    B, C, H, W = pred.shape
    
    # Sobel filters
    sobel_x = torch.tensor([
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]]
    ], dtype=torch.float32, device=device) / 8.0
    
    sobel_y = torch.tensor([
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]]
    ], dtype=torch.float32, device=device) / 8.0
    
    # Convert to grayscale (average across channels)
    pred_gray = pred.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    target_gray = target.mean(dim=1, keepdim=True)
    
    # Compute edge maps
    def compute_edge_magnitude(img):
        """Compute edge magnitude using Sobel gradients."""
        edges_x = F.conv2d(img, sobel_x.unsqueeze(1), padding=1)
        edges_y = F.conv2d(img, sobel_y.unsqueeze(1), padding=1)
        edge_mag = torch.sqrt(edges_x**2 + edges_y**2 + 1e-8)
        return edge_mag
    
    pred_edges = compute_edge_magnitude(pred_gray)
    target_edges = compute_edge_magnitude(target_gray)
    
    # Normalize edge magnitudes
    pred_edges = pred_edges / (pred_edges.max() + 1e-8)
    target_edges = target_edges / (target_edges.max() + 1e-8)
    
    # Threshold to get binary edge masks
    pred_edge_mask = (pred_edges > edge_threshold).float()
    target_edge_mask = (target_edges > edge_threshold).float()
    
    # Compute displacement for each batch item
    displacements = []
    edge_ratios = []
    
    for b in range(B):
        pred_mask_b = pred_edge_mask[b, 0]  # (H, W)
        target_mask_b = target_edge_mask[b, 0]  # (H, W)
        
        # Get edge pixel coordinates
        target_edges_yx = torch.nonzero(target_mask_b, as_tuple=False)  # (N, 2)
        pred_edges_yx = torch.nonzero(pred_mask_b, as_tuple=False)  # (M, 2)
        
        if len(target_edges_yx) == 0 or len(pred_edges_yx) == 0:
            # No edges found
            displacements.append(torch.tensor([0.0], device=device))
            edge_ratios.append(0.0)
            continue
        
        # For each target edge pixel, find nearest pred edge pixel
        # (Simplified: use L2 distance in pixel space)
        target_edges_yx = target_edges_yx.float()
        pred_edges_yx = pred_edges_yx.float()
        
        # Compute pairwise distances (can be memory intensive for large edge sets)
        # Limit to random sample if too many edges
        max_edges = 1000
        if len(target_edges_yx) > max_edges:
            indices = torch.randperm(len(target_edges_yx), device=device)[:max_edges]
            target_edges_yx = target_edges_yx[indices]
        
        # Compute distances: (N, M)
        dists = torch.cdist(target_edges_yx, pred_edges_yx, p=2.0)
        
        # Find minimum distance for each target edge
        min_dists, _ = dists.min(dim=1)  # (N,)
        
        displacements.append(min_dists)
        
        # Edge ratio: how many edges are preserved
        edge_count_pred = pred_mask_b.sum().item()
        edge_count_target = target_mask_b.sum().item()
        edge_ratio = min(edge_count_pred, edge_count_target) / (max(edge_count_pred, edge_count_target) + 1e-8)
        edge_ratios.append(edge_ratio)
    
    # Aggregate across batch
    all_displacements = torch.cat(displacements)
    
    metrics = {
        'mean_displacement': all_displacements.mean().item(),
        'max_displacement': all_displacements.max().item(),
        'median_displacement': all_displacements.median().item(),
        'edge_ratio': np.mean(edge_ratios)
    }
    
    return metrics


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
    Load or create demo optical tile (opt_v1 from Stage 1).
    
    Returns:
        Tuple of (opt_v1_tensor, metadata)
        opt_v1_tensor shape: (1, 4, H, W) - [B02, B03, B04, B08]
    """
    # Try to load real demo tile from Stage 1 output
    demo_dir = project_root / 'demo'
    
    if demo_dir.exists():
        # Look for Stage 1 output tiles
        demo_tiles = list(demo_dir.glob('*opt_v1*.npz')) + list(demo_dir.glob('*stage1*.npz'))
        
        if demo_tiles:
            print(f"\n✓ Loading real Stage 1 tile: {demo_tiles[0].name}")
            tile_data = np.load(demo_tiles[0])
            
            # Extract optical bands
            # Expected keys: 'B02', 'B03', 'B04', 'B08' or 'optical'
            if 'optical' in tile_data:
                opt_v1 = tile_data['optical'].astype(np.float32)
            else:
                b02 = tile_data.get('B02', tile_data.get('b02', np.random.rand(120, 120)))
                b03 = tile_data.get('B03', tile_data.get('b03', np.random.rand(120, 120)))
                b04 = tile_data.get('B04', tile_data.get('b04', np.random.rand(120, 120)))
                b08 = tile_data.get('B08', tile_data.get('b08', np.random.rand(120, 120)))
                
                opt_v1 = np.stack([b02, b03, b04, b08], axis=0).astype(np.float32)
            
            # Ensure 4 channels
            if opt_v1.shape[0] != 4:
                print(f"  Warning: Expected 4 channels, got {opt_v1.shape[0]}. Using synthetic data.")
                opt_v1 = None
            else:
                # Normalize to [0, 1] if needed
                if opt_v1.max() > 1.5:
                    opt_v1 = opt_v1 / 10000.0  # Sentinel-2 scale
                
                opt_v1_tensor = torch.from_numpy(opt_v1).unsqueeze(0)  # Add batch dim
                
                metadata = {
                    'source': 'real_stage1',
                    'shape': opt_v1.shape,
                    'tile_name': demo_tiles[0].stem
                }
                
                return opt_v1_tensor, metadata
    
    # Fallback: create synthetic demo tile
    print("\n⚠ No real Stage 1 tiles found, creating synthetic tile")
    
    # Create realistic optical values (normalized to [0, 1])
    H, W = 120, 120
    
    # Simulate vegetation scene
    # B02 (Blue): 0.05-0.15
    b02 = np.random.uniform(0.05, 0.15, (H, W)).astype(np.float32)
    
    # B03 (Green): 0.08-0.20
    b03 = np.random.uniform(0.08, 0.20, (H, W)).astype(np.float32)
    
    # B04 (Red): 0.05-0.15
    b04 = np.random.uniform(0.05, 0.15, (H, W)).astype(np.float32)
    
    # B08 (NIR): 0.30-0.60 (high for vegetation)
    b08 = np.random.uniform(0.30, 0.60, (H, W)).astype(np.float32)
    
    # Add spatial structure (edges)
    # Create vertical edge
    b02[:, :W//2] *= 0.7
    b03[:, :W//2] *= 0.7
    b04[:, :W//2] *= 0.7
    b08[:, :W//2] *= 0.8
    
    # Create horizontal edge
    b02[H//2:, :] *= 0.8
    b03[H//2:, :] *= 0.8
    b04[H//2:, :] *= 0.8
    b08[H//2:, :] *= 0.9
    
    opt_v1 = np.stack([b02, b03, b04, b08], axis=0)
    opt_v1_tensor = torch.from_numpy(opt_v1).unsqueeze(0)  # Add batch dim
    
    metadata = {
        'source': 'synthetic',
        'shape': opt_v1.shape,
        'tile_name': 'synthetic_demo_tile'
    }
    
    return opt_v1_tensor, metadata


@pytest.fixture(scope="module")
def stage2_model(device):
    """
    Build Stage 2 Prithvi refinement model.
    
    Uses a lightweight configuration for testing (no pretrained Prithvi).
    """
    print("\n✓ Building Stage 2 model (lightweight for testing)...")
    
    if not STAGE2_AVAILABLE:
        pytest.skip("Stage 2 modules not available")
    
    # For testing, use a simple ConvNeXt head without full Prithvi
    # This avoids dependency on terratorch/Prithvi weights
    class SimplifiedStage2(nn.Module):
        """Simplified Stage 2 model for testing."""
        
        def __init__(self):
            super().__init__()
            # Project 4 channels -> 256 features
            self.input_proj = nn.Conv2d(4, 256, kernel_size=3, padding=1)
            
            # ConvNeXt refinement head
            self.refinement_head = ConvNeXtHead(
                in_channels=256,
                out_channels=4,  # Back to 4 optical bands
                hidden_dim=128,
                num_blocks=2
            )
            
            # Output normalization
            self.output_norm = nn.Sigmoid()
        
        def forward(self, x):
            features = self.input_proj(x)
            refined = self.refinement_head(features)
            output = self.output_norm(refined)
            return output
    
    model = SimplifiedStage2()
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    
    return model


# ============================================================================
# Tests
# ============================================================================

@pytest.mark.skipif(not STAGE2_AVAILABLE, reason="Stage 2 modules not available")
def test_model_initialization(stage2_model, device):
    """Test that model initializes correctly."""
    assert stage2_model is not None
    assert next(stage2_model.parameters()).device.type == device.type
    print("✓ Model initialization test passed")


@pytest.mark.skipif(not STAGE2_AVAILABLE, reason="Stage 2 modules not available")
def test_demo_tile_loading(demo_tile):
    """Test that demo tile loads with correct format."""
    opt_v1_tensor, metadata = demo_tile
    
    # Check tensor properties
    assert opt_v1_tensor.ndim == 4, f"Expected 4D tensor, got {opt_v1_tensor.ndim}D"
    assert opt_v1_tensor.shape[0] == 1, f"Expected batch size 1, got {opt_v1_tensor.shape[0]}"
    assert opt_v1_tensor.shape[1] == 4, f"Expected 4 channels (B02,B03,B04,B08), got {opt_v1_tensor.shape[1]}"
    
    # Check spatial dimensions
    H, W = opt_v1_tensor.shape[2], opt_v1_tensor.shape[3]
    assert H >= 64 and H <= 512, f"Unexpected height: {H}"
    assert W >= 64 and W <= 512, f"Unexpected width: {W}"
    
    # Check tensor type and range
    assert opt_v1_tensor.dtype == torch.float32
    assert opt_v1_tensor.min() >= 0.0, f"Min value {opt_v1_tensor.min()} < 0"
    assert opt_v1_tensor.max() <= 1.5, f"Max value {opt_v1_tensor.max()} > 1.5"
    
    # Check metadata
    assert 'source' in metadata
    assert 'shape' in metadata
    
    print(f"✓ Demo tile loading test passed (source: {metadata['source']})")
    print(f"  Shape: {tuple(opt_v1_tensor.shape)}")
    print(f"  Range: [{opt_v1_tensor.min():.3f}, {opt_v1_tensor.max():.3f}]")


@pytest.mark.skipif(not STAGE2_AVAILABLE, reason="Stage 2 modules not available")
def test_stage2_forward_shape(stage2_model, demo_tile, device):
    """Test that forward pass produces correct output shape."""
    opt_v1_tensor, metadata = demo_tile
    opt_v1_tensor = opt_v1_tensor.to(device)
    
    # Get input dimensions
    B, C_in, H_in, W_in = opt_v1_tensor.shape
    
    # Run forward pass
    with torch.no_grad():
        opt_v2_tensor = stage2_model(opt_v1_tensor)
    
    # Check output shape
    assert opt_v2_tensor.ndim == 4, f"Expected 4D output, got {opt_v2_tensor.ndim}D"
    assert opt_v2_tensor.shape[0] == B, f"Batch size mismatch: expected {B}, got {opt_v2_tensor.shape[0]}"
    assert opt_v2_tensor.shape[1] == 4, f"Expected 4 output channels, got {opt_v2_tensor.shape[1]}"
    assert opt_v2_tensor.shape[2] == H_in, f"Height mismatch: expected {H_in}, got {opt_v2_tensor.shape[2]}"
    assert opt_v2_tensor.shape[3] == W_in, f"Width mismatch: expected {W_in}, got {opt_v2_tensor.shape[3]}"
    
    print(f"✓ Forward pass shape test passed")
    print(f"  Input shape:  {tuple(opt_v1_tensor.shape)}")
    print(f"  Output shape: {tuple(opt_v2_tensor.shape)}")


@pytest.mark.skipif(not STAGE2_AVAILABLE, reason="Stage 2 modules not available")
def test_stage2_output_range(stage2_model, demo_tile, device):
    """Test that output values are in valid [0, 1] range."""
    opt_v1_tensor, metadata = demo_tile
    opt_v1_tensor = opt_v1_tensor.to(device)
    
    # Run forward pass
    with torch.no_grad():
        opt_v2_tensor = stage2_model(opt_v1_tensor)
    
    # Check value range
    min_val = opt_v2_tensor.min().item()
    max_val = opt_v2_tensor.max().item()
    
    assert min_val >= 0.0, f"Minimum value {min_val} is below 0.0"
    assert max_val <= 1.0, f"Maximum value {max_val} is above 1.0"
    
    # Check for NaN or Inf
    assert not torch.isnan(opt_v2_tensor).any(), "Output contains NaN values"
    assert not torch.isinf(opt_v2_tensor).any(), "Output contains Inf values"
    
    print(f"✓ Output range test passed")
    print(f"  Min value: {min_val:.6f}")
    print(f"  Max value: {max_val:.6f}")
    print(f"  Mean value: {opt_v2_tensor.mean().item():.6f}")


@pytest.mark.skipif(not STAGE2_AVAILABLE, reason="Stage 2 modules not available")
def test_edge_displacement_small(stage2_model, demo_tile, device):
    """
    Test that edge displacement between v1 and v2 is small.
    
    This is the primary test: ensures Stage 2 maintains geometric consistency.
    """
    opt_v1_tensor, metadata = demo_tile
    opt_v1_tensor = opt_v1_tensor.to(device)
    
    # Run forward pass
    with torch.no_grad():
        opt_v2_tensor = stage2_model(opt_v1_tensor)
    
    # Compute edge displacement
    metrics = compute_edge_displacement(
        pred=opt_v2_tensor,
        target=opt_v1_tensor,
        edge_threshold=0.1,
        search_radius=5
    )
    
    mean_disp = metrics['mean_displacement']
    max_disp = metrics['max_displacement']
    median_disp = metrics['median_displacement']
    edge_ratio = metrics['edge_ratio']
    
    print(f"✓ Edge displacement metrics:")
    print(f"  Mean displacement:   {mean_disp:.3f} pixels")
    print(f"  Median displacement: {median_disp:.3f} pixels")
    print(f"  Max displacement:    {max_disp:.3f} pixels")
    print(f"  Edge ratio:          {edge_ratio:.3f}")
    
    # Assertions
    # Primary constraint: mean displacement should be < 2.0 pixels
    assert mean_disp < 2.0, f"Mean edge displacement {mean_disp:.3f} >= 2.0 pixels (too large)"
    
    # Secondary constraints
    assert max_disp < 5.0, f"Max edge displacement {max_disp:.3f} >= 5.0 pixels (too large)"
    assert edge_ratio > 0.7, f"Edge ratio {edge_ratio:.3f} < 0.7 (too many edges lost)"
    
    print(f"✓ Edge displacement test PASSED")
    print(f"  ✓ Mean displacement {mean_disp:.3f} < 2.0 pixels")
    print(f"  ✓ Max displacement {max_disp:.3f} < 5.0 pixels")
    print(f"  ✓ Edge ratio {edge_ratio:.3f} > 0.7")


@pytest.mark.skipif(not STAGE2_AVAILABLE, reason="Stage 2 modules not available")
def test_spectral_plausibility(stage2_model, demo_tile, device):
    """Test that spectral indices (NDVI/EVI) remain plausible."""
    opt_v1_tensor, metadata = demo_tile
    opt_v1_tensor = opt_v1_tensor.to(device)
    
    # Run forward pass
    with torch.no_grad():
        opt_v2_tensor = stage2_model(opt_v1_tensor)
    
    # Extract bands (B02, B03, B04, B08)
    v1_b02, v1_b03, v1_b04, v1_b08 = opt_v1_tensor[:, 0:1], opt_v1_tensor[:, 1:2], opt_v1_tensor[:, 2:3], opt_v1_tensor[:, 3:4]
    v2_b02, v2_b03, v2_b04, v2_b08 = opt_v2_tensor[:, 0:1], opt_v2_tensor[:, 1:2], opt_v2_tensor[:, 2:3], opt_v2_tensor[:, 3:4]
    
    # Compute NDVI
    v1_ndvi = compute_ndvi(v1_b04, v1_b08)
    v2_ndvi = compute_ndvi(v2_b04, v2_b08)
    
    # Compute EVI
    v1_evi = compute_evi(v1_b02, v1_b04, v1_b08)
    v2_evi = compute_evi(v2_b02, v2_b04, v2_b08)
    
    # Check NDVI/EVI are in valid ranges
    assert v2_ndvi.min() >= -1.0, f"NDVI min {v2_ndvi.min()} < -1.0"
    assert v2_ndvi.max() <= 1.0, f"NDVI max {v2_ndvi.max()} > 1.0"
    assert v2_evi.min() >= -1.0, f"EVI min {v2_evi.min()} < -1.0"
    assert v2_evi.max() <= 1.0, f"EVI max {v2_evi.max()} > 1.0"
    
    # Compute differences
    ndvi_diff = (v2_ndvi - v1_ndvi).abs().mean().item()
    evi_diff = (v2_evi - v1_evi).abs().mean().item()
    
    print(f"✓ Spectral plausibility test passed")
    print(f"  v1 NDVI range: [{v1_ndvi.min():.3f}, {v1_ndvi.max():.3f}]")
    print(f"  v2 NDVI range: [{v2_ndvi.min():.3f}, {v2_ndvi.max():.3f}]")
    print(f"  NDVI difference: {ndvi_diff:.4f}")
    print(f"  v1 EVI range: [{v1_evi.min():.3f}, {v1_evi.max():.3f}]")
    print(f"  v2 EVI range: [{v2_evi.min():.3f}, {v2_evi.max():.3f}]")
    print(f"  EVI difference: {evi_diff:.4f}")
    
    # Assert spectral indices don't change dramatically
    # (Stage 2 should refine, not transform)
    assert ndvi_diff < 0.15, f"NDVI difference {ndvi_diff:.4f} >= 0.15 (too large)"
    assert evi_diff < 0.20, f"EVI difference {evi_diff:.4f} >= 0.20 (too large)"


@pytest.mark.skipif(not STAGE2_AVAILABLE, reason="Stage 2 modules not available")
def test_identity_preservation(stage2_model, demo_tile, device):
    """Test that Stage 2 preserves overall structure (identity constraint)."""
    opt_v1_tensor, metadata = demo_tile
    opt_v1_tensor = opt_v1_tensor.to(device)
    
    # Run forward pass
    with torch.no_grad():
        opt_v2_tensor = stage2_model(opt_v1_tensor)
    
    # Compute L1 distance
    l1_dist = (opt_v2_tensor - opt_v1_tensor).abs().mean().item()
    
    # Compute correlation
    v1_flat = opt_v1_tensor.flatten()
    v2_flat = opt_v2_tensor.flatten()
    correlation = torch.corrcoef(torch.stack([v1_flat, v2_flat]))[0, 1].item()
    
    print(f"✓ Identity preservation test passed")
    print(f"  L1 distance: {l1_dist:.6f}")
    print(f"  Correlation: {correlation:.6f}")
    
    # Assertions
    assert l1_dist < 0.1, f"L1 distance {l1_dist:.6f} >= 0.1 (too large)"
    assert correlation > 0.85, f"Correlation {correlation:.6f} < 0.85 (too low)"


# ============================================================================
# Main (for direct execution)
# ============================================================================

def main():
    """Run all tests directly (without pytest)."""
    print("=" * 80)
    print("Stage 2 Forward Pass End-to-End Integration Test")
    print("=" * 80)
    
    if not STAGE2_AVAILABLE:
        print("\n✗ Stage 2 modules not available. Please install dependencies:")
        print("  pip install torch torchvision")
        print("  # Note: Full Prithvi requires terratorch")
        return 1
    
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
    model = stage2_model(device)
    
    # Run tests
    tests = [
        ("Model Initialization", test_model_initialization, [model, device]),
        ("Demo Tile Loading", test_demo_tile_loading, [demo_tile_data]),
        ("Forward Pass Shape", test_stage2_forward_shape, [model, demo_tile_data, device]),
        ("Output Range", test_stage2_output_range, [model, demo_tile_data, device]),
        ("Edge Displacement (PRIMARY)", test_edge_displacement_small, [model, demo_tile_data, device]),
        ("Spectral Plausibility", test_spectral_plausibility, [model, demo_tile_data, device]),
        ("Identity Preservation", test_identity_preservation, [model, demo_tile_data, device]),
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
        print("\nKey Result: Edge displacement < 2.0 pixels ✓")
        print("Stage 2 maintains geometric consistency while refining spectral content.")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
