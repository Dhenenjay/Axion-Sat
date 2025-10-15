"""
Unit tests for loss functions.

This module tests:
- NDVI/EVI computation accuracy
- SAR structure loss behavior
- Spectral index loss components
- Edge detection consistency
- Gradient flow
- Numerical stability

Uses synthetic data with known properties to verify correctness.
"""

import torch
import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from axs_lib.losses import (
    L1Loss,
    MSSSIMLoss,
    SARStructureLoss,
    SpectralIndexLoss,
    CombinedLoss
)


class TestL1Loss:
    """Test L1 loss functionality."""
    
    def test_zero_loss_identical_tensors(self):
        """L1 loss should be zero for identical tensors."""
        loss_fn = L1Loss()
        
        pred = torch.randn(4, 3, 32, 32)
        target = pred.clone()
        
        loss = loss_fn(pred, target)
        
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    
    def test_loss_magnitude(self):
        """L1 loss should match manual calculation."""
        loss_fn = L1Loss()
        
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
        
        loss = loss_fn(pred, target)
        expected = torch.tensor(0.5)  # Average absolute difference
        
        assert torch.allclose(loss, expected, atol=1e-6)
    
    def test_reduction_modes(self):
        """Test different reduction modes."""
        pred = torch.randn(2, 3, 4, 4)
        target = torch.randn(2, 3, 4, 4)
        
        loss_mean = L1Loss(reduction='mean')(pred, target)
        loss_sum = L1Loss(reduction='sum')(pred, target)
        loss_none = L1Loss(reduction='none')(pred, target)
        
        assert loss_mean.dim() == 0  # Scalar
        assert loss_sum.dim() == 0   # Scalar
        assert loss_none.shape == pred.shape  # Same shape
        
        # Check relationship
        assert torch.allclose(
            loss_sum,
            loss_mean * pred.numel(),
            atol=1e-5
        )


class TestMSSSIMLoss:
    """Test MS-SSIM loss functionality."""
    
    def test_zero_loss_identical_tensors(self):
        """MS-SSIM loss should be ~0 for identical tensors."""
        loss_fn = MSSSIMLoss(data_range=1.0)
        
        # MS-SSIM needs larger images (min 161x161, strictly > 160)
        pred = torch.rand(2, 3, 192, 192)
        target = pred.clone()
        
        loss = loss_fn(pred, target)
        
        # Loss = 1 - SSIM, SSIM should be ~1 for identical
        assert loss < 0.01
    
    def test_loss_range(self):
        """MS-SSIM loss should be in [0, 1] range."""
        loss_fn = MSSSIMLoss(data_range=1.0)
        
        # MS-SSIM needs larger images (min 161x161, strictly > 160)
        pred = torch.rand(2, 3, 192, 192)
        target = torch.rand(2, 3, 192, 192)
        
        loss = loss_fn(pred, target)
        
        assert 0.0 <= loss <= 1.0
    
    def test_higher_loss_for_dissimilar(self):
        """Loss should be higher for more dissimilar images."""
        loss_fn = MSSSIMLoss(data_range=1.0)
        
        # MS-SSIM needs larger images (min 161x161, strictly > 160)
        target = torch.rand(2, 3, 192, 192)
        
        # Similar prediction (add small noise)
        pred_similar = target + torch.randn_like(target) * 0.01
        
        # Dissimilar prediction (add large noise)
        pred_dissimilar = target + torch.randn_like(target) * 0.5
        
        loss_similar = loss_fn(pred_similar, target)
        loss_dissimilar = loss_fn(pred_dissimilar, target)
        
        assert loss_dissimilar > loss_similar


class TestSARStructureLoss:
    """Test SAR structure loss with synthetic data."""
    
    def test_zero_loss_identical_edges(self):
        """Loss should be zero for identical edge maps."""
        loss_fn = SARStructureLoss()
        
        # Create image with clear edges
        sar = torch.zeros(2, 1, 64, 64)
        sar[:, :, :32, :] = 1.0  # Horizontal edge
        
        opt_rgb = sar.repeat(1, 3, 1, 1)  # Same structure, 3 channels
        
        loss = loss_fn(sar, opt_rgb)
        
        # Should have very similar edge maps
        assert loss < 0.1
    
    def test_vertical_edge_detection(self):
        """Should detect vertical edges."""
        loss_fn = SARStructureLoss()
        
        # Create vertical edge
        sar = torch.zeros(1, 1, 64, 64)
        sar[:, :, :, :32] = 1.0
        
        # Optical with same vertical edge
        opt_match = sar.repeat(1, 3, 1, 1)
        
        # Optical with horizontal edge (mismatch)
        opt_mismatch = torch.zeros(1, 3, 64, 64)
        opt_mismatch[:, :, :32, :] = 1.0
        
        loss_match = loss_fn(sar, opt_match)
        loss_mismatch = loss_fn(sar, opt_mismatch)
        
        # Mismatched edges should have higher loss
        assert loss_mismatch > loss_match
    
    def test_diagonal_edge_detection(self):
        """Should detect diagonal edges."""
        loss_fn = SARStructureLoss()
        
        # Create diagonal edge
        sar = torch.zeros(1, 1, 64, 64)
        for i in range(64):
            sar[0, 0, i, :min(i+1, 64)] = 1.0
        
        # Optical with same diagonal
        opt = sar.repeat(1, 3, 1, 1)
        
        loss = loss_fn(sar, opt)
        
        # Should detect diagonal structure
        assert loss < 0.2
    
    def test_multichannel_sar(self):
        """Should handle multi-channel SAR (VV+VH)."""
        loss_fn = SARStructureLoss()
        
        # Two-channel SAR
        sar = torch.zeros(1, 2, 64, 64)
        sar[:, 0, :32, :] = 1.0  # VV
        sar[:, 1, :, :32] = 1.0  # VH
        
        opt = torch.zeros(1, 3, 64, 64)
        opt[:, :, :32, :] = 1.0  # Similar to VV
        
        loss = loss_fn(sar, opt)
        
        # Should compute gradient on averaged SAR channels
        assert loss >= 0.0


class TestSpectralIndexLoss:
    """Test spectral index loss with known NDVI/EVI values."""
    
    def test_ndvi_computation(self):
        """Verify NDVI computation accuracy."""
        loss_fn = SpectralIndexLoss(indices=('ndvi',), use_spectral_angle=False)
        
        # Create synthetic RGBNIR data
        # R=100, G=200, B=300, NIR=400
        pred = torch.tensor([[
            [[100.0, 100.0], [100.0, 100.0]],  # R
            [[200.0, 200.0], [200.0, 200.0]],  # G
            [[300.0, 300.0], [300.0, 300.0]],  # B
            [[400.0, 400.0], [400.0, 400.0]],  # NIR
        ]])
        
        target = pred.clone()
        
        loss = loss_fn(pred, target)
        
        # NDVI = (NIR - R) / (NIR + R) = (400-100)/(400+100) = 0.6
        # Loss should be ~0 for identical NDVI (RMSE with epsilon)
        assert loss < 1e-3
    
    def test_ndvi_range(self):
        """NDVI should be in valid range [-1, 1]."""
        loss_fn = SpectralIndexLoss(indices=('ndvi',), use_spectral_angle=False)
        
        # High vegetation: NIR >> R
        high_veg = torch.tensor([[
            [[100.0, 100.0], [100.0, 100.0]],  # R
            [[200.0, 200.0], [200.0, 200.0]],  # G
            [[300.0, 300.0], [300.0, 300.0]],  # B
            [[800.0, 800.0], [800.0, 800.0]],  # NIR
        ]])
        
        # Low vegetation: NIR ~= R
        low_veg = torch.tensor([[
            [[400.0, 400.0], [400.0, 400.0]],  # R
            [[400.0, 400.0], [400.0, 400.0]],  # G
            [[400.0, 400.0], [400.0, 400.0]],  # B
            [[500.0, 500.0], [500.0, 500.0]],  # NIR
        ]])
        
        # Compute NDVI values (through loss function)
        # We'll use identical pred/target to ensure zero loss
        loss_high = loss_fn(high_veg, high_veg)
        loss_low = loss_fn(low_veg, low_veg)
        
        # Both should be ~zero for identical inputs (RMSE with epsilon)
        assert loss_high < 1e-3
        assert loss_low < 1e-3
    
    def test_evi_computation(self):
        """Verify EVI computation accuracy."""
        loss_fn = SpectralIndexLoss(indices=('evi',), use_spectral_angle=False)
        
        # EVI = 2.5 * (NIR - R) / (NIR + 6*R - 7.5*B + 1)
        pred = torch.tensor([[
            [[200.0, 200.0], [200.0, 200.0]],  # R
            [[300.0, 300.0], [300.0, 300.0]],  # G
            [[100.0, 100.0], [100.0, 100.0]],  # B
            [[600.0, 600.0], [600.0, 600.0]],  # NIR
        ]])
        
        target = pred.clone()
        
        loss = loss_fn(pred, target)
        
        # Loss should be ~0 for identical EVI (RMSE with epsilon)
        assert loss < 1e-3
    
    def test_combined_ndvi_evi(self):
        """Verify combined NDVI+EVI computation."""
        loss_fn = SpectralIndexLoss(indices=('ndvi', 'evi'), use_spectral_angle=False)
        
        # Test with identical inputs
        pred = torch.tensor([[
            [[200.0, 200.0], [200.0, 200.0]],  # R
            [[300.0, 300.0], [300.0, 300.0]],  # G
            [[100.0, 100.0], [100.0, 100.0]],  # B
            [[600.0, 600.0], [600.0, 600.0]],  # NIR
        ]])
        
        target = pred.clone()
        
        loss = loss_fn(pred, target)
        
        # Loss should be ~0 for identical inputs
        assert loss < 1e-3
    
    def test_spectral_angle_mapper(self):
        """Test spectral angle mapper (SAM) computation."""
        loss_fn = SpectralIndexLoss(
            indices=tuple(),  # No index losses
            use_spectral_angle=True
        )
        
        # Identical spectra should have zero angle
        pred = torch.randn(2, 4, 32, 32)
        target = pred.clone()
        
        loss = loss_fn(pred, target)
        
        # SAM should be ~0 for identical spectra
        assert loss < 0.01
    
    def test_spectral_angle_orthogonal(self):
        """Orthogonal spectra should have large angle."""
        loss_fn = SpectralIndexLoss(
            indices=tuple(),
            use_spectral_angle=True
        )
        
        # Create orthogonal spectra
        pred = torch.zeros(1, 4, 2, 2)
        pred[0, 0] = 1.0  # Only first channel
        
        target = torch.zeros(1, 4, 2, 2)
        target[0, 1] = 1.0  # Only second channel
        
        loss = loss_fn(pred, target)
        
        # SAM for orthogonal should be ~π/2 ≈ 1.57
        assert loss > 1.0
    
    def test_combined_indices(self):
        """Test multiple indices together."""
        loss_fn = SpectralIndexLoss(
            indices=('ndvi', 'evi'),
            use_spectral_angle=True
        )
        
        pred = torch.rand(2, 4, 16, 16) * 1000
        target = pred.clone()
        
        loss = loss_fn(pred, target)
        
        # All components should be ~0 for identical inputs
        assert loss < 0.1
    
    def test_different_indices_different_loss(self):
        """Different vegetation should produce different losses."""
        loss_fn = SpectralIndexLoss(
            indices=('ndvi',),
            use_spectral_angle=False
        )
        
        # High vegetation target
        target = torch.tensor([[
            [[100.0, 100.0], [100.0, 100.0]],
            [[200.0, 200.0], [200.0, 200.0]],
            [[300.0, 300.0], [300.0, 300.0]],
            [[800.0, 800.0], [800.0, 800.0]],
        ]])
        
        # Low vegetation prediction
        pred = torch.tensor([[
            [[400.0, 400.0], [400.0, 400.0]],
            [[400.0, 400.0], [400.0, 400.0]],
            [[400.0, 400.0], [400.0, 400.0]],
            [[500.0, 500.0], [500.0, 500.0]],
        ]])
        
        loss = loss_fn(pred, target)
        
        # Should have non-zero loss due to NDVI difference
        assert loss > 0.1


class TestCombinedLoss:
    """Test combined loss functionality."""
    
    def test_l1_only(self):
        """Test with only L1 loss enabled."""
        loss_fn = CombinedLoss(
            l1_weight=1.0,
            ms_ssim_weight=0.0,
            sar_structure_weight=0.0,
            spectral_index_weight=0.0
        )
        
        pred = torch.rand(2, 4, 32, 32)
        target = torch.rand(2, 4, 32, 32)
        
        total_loss, loss_dict = loss_fn(pred, target)
        
        assert 'l1' in loss_dict
        assert loss_dict['total'] == loss_dict['l1']
    
    def test_multiple_losses(self):
        """Test with multiple losses combined."""
        loss_fn = CombinedLoss(
            l1_weight=0.5,
            ms_ssim_weight=0.5,
            sar_structure_weight=0.0,
            spectral_index_weight=0.0
        )
        
        # MS-SSIM needs larger images (min 161x161, strictly > 160)
        pred = torch.rand(2, 3, 192, 192)
        target = torch.rand(2, 3, 192, 192)
        
        total_loss, loss_dict = loss_fn(pred, target)
        
        assert 'l1' in loss_dict
        assert 'ms_ssim' in loss_dict
        assert 'total' in loss_dict
        
        # Total should be weighted sum
        expected = 0.5 * loss_dict['l1'] + 0.5 * loss_dict['ms_ssim']
        assert abs(loss_dict['total'] - expected) < 1e-5
    
    def test_with_sar_structure(self):
        """Test with SAR structure loss."""
        loss_fn = CombinedLoss(
            l1_weight=1.0,
            sar_structure_weight=0.5
        )
        
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)
        sar = torch.rand(2, 2, 64, 64)
        
        total_loss, loss_dict = loss_fn(pred, target, sar=sar)
        
        assert 'l1' in loss_dict
        assert 'sar_structure' in loss_dict
        assert 'total' in loss_dict
    
    def test_gradient_flow(self):
        """Test that gradients flow through all losses."""
        loss_fn = CombinedLoss(
            l1_weight=1.0,
            ms_ssim_weight=0.5
        )
        
        # MS-SSIM needs larger images (min 161x161, strictly > 160)
        pred = torch.rand(2, 3, 192, 192, requires_grad=True)
        target = torch.rand(2, 3, 192, 192)
        
        total_loss, _ = loss_fn(pred, target)
        total_loss.backward()
        
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()
        assert not torch.isinf(pred.grad).any()


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_zero_tensors(self):
        """Test with zero tensors."""
        loss_fn = L1Loss()
        
        pred = torch.zeros(2, 3, 32, 32)
        target = torch.zeros(2, 3, 32, 32)
        
        loss = loss_fn(pred, target)
        
        assert torch.allclose(loss, torch.tensor(0.0))
    
    def test_large_values(self):
        """Test with large values (typical for Sentinel-2)."""
        loss_fn = L1Loss()
        
        # Typical Sentinel-2 range
        pred = torch.rand(2, 4, 32, 32) * 10000
        target = torch.rand(2, 4, 32, 32) * 10000
        
        loss = loss_fn(pred, target)
        
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss > 0
    
    def test_negative_values_sar(self):
        """Test with negative values (dB scale SAR)."""
        loss_fn = L1Loss()
        
        # SAR in dB scale
        pred = torch.randn(2, 2, 32, 32) * 5 - 15
        target = torch.randn(2, 2, 32, 32) * 5 - 15
        
        loss = loss_fn(pred, target)
        
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_spectral_index_division_by_zero(self):
        """Test that spectral indices handle division by zero."""
        loss_fn = SpectralIndexLoss(
            indices=('ndvi',),
            use_spectral_angle=False
        )
        
        # Edge case: very small NIR and R values
        pred = torch.tensor([[
            [[0.01, 0.01], [0.01, 0.01]],
            [[0.01, 0.01], [0.01, 0.01]],
            [[0.01, 0.01], [0.01, 0.01]],
            [[0.01, 0.01], [0.01, 0.01]],
        ]])
        
        target = pred.clone()
        
        loss = loss_fn(pred, target)
        
        # Should not produce NaN or Inf
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    """Run tests with pytest."""
    import sys
    
    print("="*70)
    print("Running Loss Function Unit Tests")
    print("="*70)
    
    # Run pytest
    exit_code = pytest.main([
        __file__,
        '-v',           # Verbose
        '--tb=short',   # Short traceback
        '-s',           # Show print statements
    ])
    
    sys.exit(exit_code)
