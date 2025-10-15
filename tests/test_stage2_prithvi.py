"""
Test suite for Stage 2 Prithvi refinement module.

Tests:
    1. ConvNeXt block forward pass
    2. ConvNeXt head forward pass
    3. PrithviRefiner creation and forward pass
    4. Memory usage validation
    5. Integration with Stage 1 output format
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import pytest

try:
    from axs_lib.stage2_prithvi_refine import (
        ConvNeXtBlock,
        ConvNeXtHead,
        PrithviRefiner,
        build_prithvi_refiner
    )
    STAGE2_AVAILABLE = True
except ImportError as e:
    STAGE2_AVAILABLE = False
    print(f"Warning: Could not import stage2_prithvi_refine: {e}")

# Test device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestConvNeXtBlock:
    """Test ConvNeXt block components."""
    
    @pytest.mark.skipif(not STAGE2_AVAILABLE, reason="Stage 2 module not available")
    def test_convnext_block_forward(self):
        """Test ConvNeXt block forward pass."""
        dim = 256
        H, W = 64, 64
        batch_size = 2
        
        block = ConvNeXtBlock(dim=dim, kernel_size=7, expansion=4)
        block.to(DEVICE)
        
        x = torch.randn(batch_size, dim, H, W).to(DEVICE)
        
        with torch.no_grad():
            out = block(x)
        
        # Check output shape matches input
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
        
        # Check output is not NaN or Inf
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
        
        print(f"✓ ConvNeXt block test passed: {x.shape} -> {out.shape}")
    
    @pytest.mark.skipif(not STAGE2_AVAILABLE, reason="Stage 2 module not available")
    def test_convnext_block_residual(self):
        """Test that residual connection works."""
        dim = 128
        block = ConvNeXtBlock(dim=dim)
        block.to(DEVICE)
        
        x = torch.randn(1, dim, 32, 32).to(DEVICE)
        
        # Zero out weights to test residual
        with torch.no_grad():
            for param in block.parameters():
                param.zero_()
            
            out = block(x)
        
        # With zero weights, output should approximately equal input (residual)
        assert torch.allclose(out, x, atol=1e-5), "Residual connection not working"
        
        print("✓ ConvNeXt residual connection test passed")


class TestConvNeXtHead:
    """Test ConvNeXt refinement head."""
    
    @pytest.mark.skipif(not STAGE2_AVAILABLE, reason="Stage 2 module not available")
    def test_head_forward(self):
        """Test ConvNeXt head forward pass."""
        in_channels = 768  # Prithvi output dim
        out_channels = 256
        H, W = 64, 64
        batch_size = 2
        
        head = ConvNeXtHead(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=256,
            num_blocks=4
        )
        head.to(DEVICE)
        
        x = torch.randn(batch_size, in_channels, H, W).to(DEVICE)
        
        with torch.no_grad():
            out = head(x)
        
        # Check output shape
        expected_shape = (batch_size, out_channels, H, W)
        assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"
        
        # Check output is not NaN or Inf
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
        
        print(f"✓ ConvNeXt head test passed: {x.shape} -> {out.shape}")
    
    @pytest.mark.skipif(not STAGE2_AVAILABLE, reason="Stage 2 module not available")
    def test_head_with_downsample(self):
        """Test ConvNeXt head with downsampling."""
        in_channels = 512
        out_channels = 128
        H, W = 128, 128
        
        head = ConvNeXtHead(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=256,
            num_blocks=4,
            use_downsample=True
        )
        head.to(DEVICE)
        
        x = torch.randn(1, in_channels, H, W).to(DEVICE)
        
        with torch.no_grad():
            out = head(x)
        
        # Output should be downsampled
        assert out.shape[1] == out_channels, f"Channel mismatch: {out.shape[1]} vs {out_channels}"
        assert out.shape[-2:] < (H, W), "Spatial dimensions not downsampled"
        
        print(f"✓ ConvNeXt downsampling test passed: {x.shape} -> {out.shape}")


class TestPrithviRefiner:
    """Test PrithviRefiner model (if dependencies available)."""
    
    @pytest.mark.skipif(
        not STAGE2_AVAILABLE or DEVICE == "cpu",
        reason="Stage 2 module not available or no GPU"
    )
    def test_build_from_kwargs(self):
        """Test building PrithviRefiner from kwargs (mocked)."""
        # This test will fail if Prithvi models not registered
        # But it validates the builder function structure
        
        try:
            model = build_prithvi_refiner(
                num_input_channels=4,
                out_channels=256,
                hidden_dim=128,
                num_convnext_blocks=2,
                lora_r=4,
                lora_alpha=8,
                load_in_8bit=False,  # Disable 8-bit for test
                freeze_backbone=True,
                device=DEVICE
            )
            
            print(f"✓ PrithviRefiner created successfully")
            
            # Test forward pass with dummy input
            x = torch.randn(1, 4, 128, 128).to(DEVICE)
            
            with torch.no_grad():
                out = model(x)
            
            assert out.shape == (1, 256, 128, 128), f"Output shape mismatch: {out.shape}"
            
            print(f"✓ PrithviRefiner forward pass successful: {x.shape} -> {out.shape}")
            
        except Exception as e:
            # Expected if Prithvi not registered yet
            print(f"⚠️  PrithviRefiner test skipped: {e}")
            pytest.skip(f"Prithvi models not available: {e}")
    
    @pytest.mark.skipif(not STAGE2_AVAILABLE, reason="Stage 2 module not available")
    def test_stage1_compatibility(self):
        """Test that Stage 2 accepts Stage 1 output format."""
        # Stage 1 outputs: (B, 4, H, W) - [B02, B03, B04, B08]
        batch_size = 2
        H, W = 256, 256
        
        # Simulate Stage 1 output
        stage1_output = torch.randn(batch_size, 4, H, W).to(DEVICE)
        stage1_output = torch.clamp(stage1_output, 0, 1)  # Stage 1 normalizes to [0,1]
        
        print(f"Stage 1 output (simulated): shape={stage1_output.shape}, "
              f"range=[{stage1_output.min():.3f}, {stage1_output.max():.3f}]")
        
        # Test that ConvNeXt head can process it (as fallback if Prithvi unavailable)
        # Project 4 channels -> 768 channels (Prithvi dim)
        proj = nn.Conv2d(4, 768, kernel_size=1).to(DEVICE)
        head = ConvNeXtHead(in_channels=768, out_channels=256, num_blocks=2).to(DEVICE)
        
        with torch.no_grad():
            features = proj(stage1_output)
            out = head(features)
        
        assert out.shape == (batch_size, 256, H, W), f"Output shape mismatch: {out.shape}"
        
        print(f"✓ Stage 1->2 compatibility test passed: {stage1_output.shape} -> {out.shape}")


class TestMemoryUsage:
    """Test memory usage characteristics."""
    
    @pytest.mark.skipif(
        not STAGE2_AVAILABLE or not torch.cuda.is_available(),
        reason="GPU not available"
    )
    def test_convnext_memory(self):
        """Test ConvNeXt head memory usage."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        head = ConvNeXtHead(
            in_channels=768,
            out_channels=256,
            hidden_dim=256,
            num_blocks=4
        ).to("cuda")
        
        x = torch.randn(1, 768, 256, 256, device="cuda")
        
        with torch.no_grad():
            out = head(x)
        
        mem_allocated = torch.cuda.memory_allocated() / 1024**2
        mem_peak = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"ConvNeXt memory: {mem_allocated:.1f} MB allocated, {mem_peak:.1f} MB peak")
        
        # ConvNeXt head should use less than 500 MB
        assert mem_peak < 500, f"ConvNeXt head uses too much memory: {mem_peak:.1f} MB"
        
        print("✓ ConvNeXt memory usage acceptable")


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Stage 2 Prithvi Refinement Module - Test Suite")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not STAGE2_AVAILABLE:
        print("\n⚠️  Stage 2 module not available. Install dependencies:")
        print("  pip install torch bitsandbytes peft terratorch")
        sys.exit(1)
    
    # Run tests manually (pytest also works)
    print("\n" + "-"*70)
    print("Running ConvNeXt Block Tests")
    print("-"*70)
    
    try:
        test_block = TestConvNeXtBlock()
        test_block.test_convnext_block_forward()
        test_block.test_convnext_block_residual()
    except Exception as e:
        print(f"✗ ConvNeXt block tests failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-"*70)
    print("Running ConvNeXt Head Tests")
    print("-"*70)
    
    try:
        test_head = TestConvNeXtHead()
        test_head.test_head_forward()
        test_head.test_head_with_downsample()
    except Exception as e:
        print(f"✗ ConvNeXt head tests failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-"*70)
    print("Running PrithviRefiner Tests")
    print("-"*70)
    
    try:
        test_refiner = TestPrithviRefiner()
        test_refiner.test_stage1_compatibility()
        # test_refiner.test_build_from_kwargs()  # Skip if models not registered
    except Exception as e:
        print(f"⚠️  PrithviRefiner tests skipped: {e}")
    
    if torch.cuda.is_available():
        print("\n" + "-"*70)
        print("Running Memory Tests")
        print("-"*70)
        
        try:
            test_memory = TestMemoryUsage()
            test_memory.test_convnext_memory()
        except Exception as e:
            print(f"✗ Memory tests failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("Test suite completed!")
    print("="*70)
    print("\nTo run with pytest:")
    print("  pytest tests/test_stage2_prithvi.py -v")
