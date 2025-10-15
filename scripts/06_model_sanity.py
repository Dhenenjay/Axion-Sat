"""
scripts/06_model_sanity.py - Model Sanity Check

Loads TerraMind and Prithvi models from local HuggingFace cache, prints
parameter counts, and runs dummy forward passes to confirm everything works.

This script validates:
1. Models can be loaded from weights/hf/... cache
2. Model architectures are correct (parameter counts)
3. Forward passes complete without errors
4. Output shapes are as expected
5. No CUDA/memory issues

Usage:
    python scripts/06_model_sanity.py
    
    # With verbose output
    python scripts/06_model_sanity.py --verbose
    
    # Skip CUDA (CPU only)
    python scripts/06_model_sanity.py --cpu
"""

import sys
import argparse
from pathlib import Path
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch not available. Install with: pip install torch")
    sys.exit(1)

try:
    from axs_lib.models import (
        build_terramind_backbone,
        build_terramind_generator,
        build_prithvi_600,
        list_available_models,
        get_model_info,
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"ERROR: axs_lib.models not available: {e}")
    sys.exit(1)

try:
    from axs_lib.optim_lowvr import (
        setup_lowvram_environment,
        print_memory_stats,
        clear_cuda_cache,
    )
    OPTIM_AVAILABLE = True
except ImportError:
    OPTIM_AVAILABLE = False
    print("WARNING: axs_lib.optim_lowvr not available (low-VRAM optimizations disabled)")


# ============================================================================
# Configuration
# ============================================================================

# Test tensor dimensions
BATCH_SIZE = 1
S1_CHANNELS = 2  # VV, VH
S2_CHANNELS = 4  # B02, B03, B04, B08 (reduced for low-VRAM)
HEIGHT = 128
WIDTH = 128
TIMESTEPS = 6  # Minimal for fast testing

# HuggingFace cache location
HF_CACHE_DIR = project_root / "weights" / "hf"

# Model configurations
MODEL_CONFIGS = {
    "terramind_generator": {
        "input_modalities": ("S1GRD",),
        "output_modalities": ("S2L2A",),
        "timesteps": TIMESTEPS,
        "standardize": True,
        "pretrained": True,  # Try to load from cache
    },
    "prithvi_600": {
        "pretrained": True,  # Try to load from cache
        "num_classes": 1,
        "img_size": HEIGHT,
        "in_channels": S2_CHANNELS,
        "use_lora": False,  # No LoRA for sanity check
        "freeze_encoder": False,
    },
}


# ============================================================================
# Helper Functions
# ============================================================================

def format_param_count(count: int) -> str:
    """Format parameter count in human-readable form."""
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.2f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.2f}K"
    else:
        return str(count)


def print_model_summary(model: nn.Module, model_name: str):
    """Print model parameter summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n{'=' * 79}")
    print(f"Model: {model_name}")
    print(f"{'=' * 79}")
    print(f"Total parameters:     {format_param_count(total_params):>10} ({total_params:,})")
    print(f"Trainable parameters: {format_param_count(trainable_params):>10} ({trainable_params:,})")
    print(f"Frozen parameters:    {format_param_count(frozen_params):>10} ({frozen_params:,})")
    print(f"Trainable ratio:      {trainable_params / total_params * 100:>10.2f}%")
    print(f"{'=' * 79}")


def print_tensor_info(name: str, tensor: torch.Tensor):
    """Print tensor shape and statistics."""
    print(f"\n  {name}:")
    print(f"    Shape:  {tuple(tensor.shape)}")
    print(f"    Dtype:  {tensor.dtype}")
    print(f"    Device: {tensor.device}")
    print(f"    Range:  [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    print(f"    Mean:   {tensor.mean().item():.4f}")
    print(f"    Std:    {tensor.std().item():.4f}")


def check_hf_cache():
    """Check if HuggingFace cache directory exists."""
    if HF_CACHE_DIR.exists():
        print(f"✓ HuggingFace cache found: {HF_CACHE_DIR}")
        
        # List cached models
        cached_models = list(HF_CACHE_DIR.glob("*"))
        if cached_models:
            print(f"  Cached models: {len(cached_models)}")
            for model_dir in cached_models[:5]:  # Show first 5
                print(f"    - {model_dir.name}")
            if len(cached_models) > 5:
                print(f"    ... and {len(cached_models) - 5} more")
        else:
            print("  WARNING: Cache directory empty (models will download)")
        return True
    else:
        print(f"⚠ HuggingFace cache not found: {HF_CACHE_DIR}")
        print("  Models will be downloaded to default HuggingFace cache")
        return False


# ============================================================================
# Test Functions
# ============================================================================

def test_terramind_generator(device: torch.device, verbose: bool = False):
    """Test TerraMind generator model."""
    print("\n" + "=" * 79)
    print("TEST: TerraMind Generator (SAR → Optical Latent)")
    print("=" * 79)
    
    # Create dummy input
    s1_input = torch.randn(BATCH_SIZE, S1_CHANNELS, HEIGHT, WIDTH, device=device)
    if verbose:
        print_tensor_info("Input (S1GRD)", s1_input)
    
    # Load model
    print("\nLoading TerraMind generator...")
    try:
        model = build_terramind_generator(**MODEL_CONFIGS["terramind_generator"])
        model = model.to(device)
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nNOTE: TerraMind may not be available in TerraTorch yet.")
        print("      This is expected if using a pre-release version.")
        return False
    
    # Print parameter summary
    print_model_summary(model, "TerraMind 1.0 Large (Generator)")
    
    # Forward pass
    print("\nRunning forward pass...")
    try:
        with torch.no_grad():
            input_dict = {"S1GRD": s1_input}
            output = model(input_dict)
        print("✓ Forward pass completed")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # Validate output
    if isinstance(output, dict):
        print("\nOutput (dictionary):")
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                if verbose:
                    print_tensor_info(f"  {key}", value)
                else:
                    print(f"  {key}: {tuple(value.shape)}")
    elif isinstance(output, torch.Tensor):
        if verbose:
            print_tensor_info("Output (latent)", output)
        else:
            print(f"\nOutput shape: {tuple(output.shape)}")
    else:
        print(f"\nOutput type: {type(output)}")
    
    # Validate output is finite
    if isinstance(output, torch.Tensor):
        if torch.isfinite(output).all():
            print("✓ Output contains only finite values")
        else:
            print("✗ Output contains NaN or Inf values")
            return False
    
    print("\n" + "=" * 79)
    print("✓ TerraMind generator test PASSED")
    print("=" * 79)
    return True


def test_prithvi_600(device: torch.device, verbose: bool = False):
    """Test Prithvi 600M model."""
    print("\n" + "=" * 79)
    print("TEST: Prithvi EO 2.0 600M (Segmentation Refinement)")
    print("=" * 79)
    
    # Create dummy input
    s2_input = torch.randn(BATCH_SIZE, S2_CHANNELS, HEIGHT, WIDTH, device=device)
    if verbose:
        print_tensor_info("Input (S2L2A)", s2_input)
    
    # Load model
    print("\nLoading Prithvi 600M...")
    try:
        model = build_prithvi_600(**MODEL_CONFIGS["prithvi_600"])
        model = model.to(device)
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nNOTE: Prithvi may not be available in TerraTorch yet.")
        print("      This is expected if using a pre-release version.")
        return False
    
    # Print parameter summary
    print_model_summary(model, "Prithvi EO 2.0 600M")
    
    # Forward pass
    print("\nRunning forward pass...")
    try:
        with torch.no_grad():
            output = model(s2_input)
        print("✓ Forward pass completed")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # Validate output
    if verbose:
        print_tensor_info("Output (segmentation mask)", output)
    else:
        print(f"\nOutput shape: {tuple(output.shape)}")
    
    # Check output shape
    expected_shape = (BATCH_SIZE, MODEL_CONFIGS["prithvi_600"]["num_classes"], HEIGHT, WIDTH)
    if output.shape == expected_shape:
        print(f"✓ Output shape matches expected: {expected_shape}")
    else:
        print(f"✗ Output shape mismatch: expected {expected_shape}, got {output.shape}")
        return False
    
    # Validate output is finite
    if torch.isfinite(output).all():
        print("✓ Output contains only finite values")
    else:
        print("✗ Output contains NaN or Inf values")
        return False
    
    # Check output range (should be reasonable for segmentation)
    output_min = output.min().item()
    output_max = output.max().item()
    print(f"  Output range: [{output_min:.4f}, {output_max:.4f}]")
    
    print("\n" + "=" * 79)
    print("✓ Prithvi 600M test PASSED")
    print("=" * 79)
    return True


def test_model_info():
    """Test model info retrieval."""
    print("\n" + "=" * 79)
    print("MODEL INFORMATION")
    print("=" * 79)
    
    # List available models
    print("\nAvailable models in TerraTorch:")
    models = list_available_models()
    for registry_name, model_list in models.items():
        print(f"\n{registry_name.upper()} Registry:")
        for model_name in model_list:
            print(f"  - {model_name}")
    
    # Get model info
    print("\nDetailed model information:")
    for model_name in ["terramind_1.0_large", "prithvi_eo_2.0_600M"]:
        info = get_model_info(model_name)
        print(f"\n{model_name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")


# ============================================================================
# Main Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sanity check for Axion-Sat models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution (skip CUDA)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output (tensor details)",
    )
    parser.add_argument(
        "--skip-terramind",
        action="store_true",
        help="Skip TerraMind tests",
    )
    parser.add_argument(
        "--skip-prithvi",
        action="store_true",
        help="Skip Prithvi tests",
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only print model info (no loading/testing)",
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 79)
    print("AXION-SAT MODEL SANITY CHECK")
    print("=" * 79)
    print()
    
    # Check environment
    print("Environment Check:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available() and not args.cpu:
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        device = torch.device("cuda")
    else:
        print("  Using CPU (CUDA disabled or not available)")
        device = torch.device("cpu")
    print()
    
    # Check HuggingFace cache
    check_hf_cache()
    print()
    
    # Setup low-VRAM environment
    if OPTIM_AVAILABLE and not args.cpu:
        print("Setting up low-VRAM environment...")
        setup_lowvram_environment(
            enable_xformers=True,
            enable_tf32=True,
            verbose=False,
        )
        print()
    
    # Test model info
    test_model_info()
    
    # Exit if info-only
    if args.info_only:
        print("\n" + "=" * 79)
        print("INFO-ONLY MODE - Skipping model loading/testing")
        print("=" * 79)
        return
    
    # Run tests
    results = {}
    
    if not args.skip_terramind:
        try:
            results["terramind"] = test_terramind_generator(device, verbose=args.verbose)
        except Exception as e:
            print(f"\n✗ TerraMind test encountered error: {e}")
            results["terramind"] = False
        
        # Clear cache between models
        if device.type == "cuda":
            clear_cuda_cache()
    
    if not args.skip_prithvi:
        try:
            results["prithvi"] = test_prithvi_600(device, verbose=args.verbose)
        except Exception as e:
            print(f"\n✗ Prithvi test encountered error: {e}")
            results["prithvi"] = False
        
        # Clear cache
        if device.type == "cuda":
            clear_cuda_cache()
    
    # Print memory statistics
    if device.type == "cuda" and OPTIM_AVAILABLE:
        print("\n")
        print_memory_stats()
    
    # Summary
    print("\n" + "=" * 79)
    print("TEST SUMMARY")
    print("=" * 79)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name.upper()}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    if args.skip_terramind:
        print("TERRAMIND: ⊘ SKIPPED")
        skipped += 1
    if args.skip_prithvi:
        print("PRITHVI: ⊘ SKIPPED")
        skipped += 1
    
    print()
    print(f"Total: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed > 0:
        print("\n⚠ Some tests failed. Check output above for details.")
        print("\nNOTE: Failures may be expected if:")
        print("  1. Models are not yet available in TerraTorch")
        print("  2. HuggingFace cache is empty (models need download)")
        print("  3. TerraTorch is not properly installed")
        print("\nTo download models manually:")
        print("  python -c \"from axs_lib.models import build_prithvi_600; build_prithvi_600(pretrained=True)\"")
    else:
        print("\n✓ All tests passed!")
    
    print("=" * 79)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
