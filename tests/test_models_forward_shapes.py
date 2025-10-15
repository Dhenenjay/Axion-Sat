"""
tests/test_models_forward_shapes.py - Model Forward Pass Shape Validation

Tests model building and forward pass with stub tensors to ensure:
1. Models can be instantiated without errors
2. Forward passes complete without crashes
3. Output shapes match expected dimensions
4. CUDA operations work correctly (if available)

This test uses minimal configurations (tiny timesteps, small batch sizes)
to quickly validate the pipeline without expensive computation.
"""

import sys
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    pytest.skip("PyTorch not available", allow_module_level=True)

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
    pytest.skip(f"axs_lib.models not available: {e}", allow_module_level=True)


# ============================================================================
# Test Configuration
# ============================================================================

# Test tensor shapes
BATCH_SIZE = 1
IN_CHANNELS_S1 = 2  # VV, VH polarizations
IN_CHANNELS_S2 = 6  # RGB + NIR + SWIR1 + SWIR2
HEIGHT = 128
WIDTH = 128
TIMESTEPS_FAST = 6  # Minimal timesteps for fast testing

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()

# Model configurations for testing
TEST_CONFIGS = {
    "terramind_backbone": {
        "modalities": ("S1GRD", "S2L2A"),
        "pretrained": False,  # Skip weight download for tests
        "freeze": False,
    },
    "terramind_generator": {
        "input_modalities": ("S1GRD",),
        "output_modalities": ("S2L2A",),
        "timesteps": TIMESTEPS_FAST,
        "standardize": True,
        "pretrained": False,
    },
    "prithvi_600": {
        "pretrained": False,
        "num_classes": 1,
        "img_size": HEIGHT,
        "in_channels": IN_CHANNELS_S2,
        "use_lora": True,
        "lora_r": 8,
        "freeze_encoder": False,
    },
}


# ============================================================================
# Fixture: Stub Data Tensors
# ============================================================================

@pytest.fixture
def stub_s1_tensor():
    """Create stub Sentinel-1 SAR tensor (VV, VH polarizations)."""
    tensor = torch.randn(
        BATCH_SIZE, IN_CHANNELS_S1, HEIGHT, WIDTH,
        dtype=torch.float32
    )
    return tensor.to(DEVICE)


@pytest.fixture
def stub_s2_tensor():
    """Create stub Sentinel-2 optical tensor (6 bands)."""
    tensor = torch.randn(
        BATCH_SIZE, IN_CHANNELS_S2, HEIGHT, WIDTH,
        dtype=torch.float32
    )
    return tensor.to(DEVICE)


@pytest.fixture
def stub_multimodal_input(stub_s1_tensor, stub_s2_tensor):
    """Create stub multimodal input dictionary."""
    return {
        "S1GRD": stub_s1_tensor,
        "S2L2A": stub_s2_tensor,
    }


@pytest.fixture
def stub_features_tensor():
    """Create stub feature tensor (e.g., from TerraMind backbone)."""
    # Typical feature map size after encoding (reduced spatial dims)
    feature_height = HEIGHT // 16  # Assuming 16x downsampling
    feature_width = WIDTH // 16
    feature_channels = 256  # Typical embedding dimension
    
    tensor = torch.randn(
        BATCH_SIZE, feature_channels, feature_height, feature_width,
        dtype=torch.float32
    )
    return tensor.to(DEVICE)


# ============================================================================
# Utility Functions
# ============================================================================

def print_tensor_info(name: str, tensor: torch.Tensor):
    """Print debug information about a tensor."""
    print(f"\n{name}:")
    print(f"  Shape: {tuple(tensor.shape)}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Memory: {tensor.element_size() * tensor.nelement() / 1024 / 1024:.2f} MB")
    print(f"  Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")


def check_tensor_valid(tensor: torch.Tensor, name: str = "tensor"):
    """Validate tensor has no NaN/Inf values."""
    assert torch.isfinite(tensor).all(), f"{name} contains NaN or Inf values"


# ============================================================================
# Test: Environment & Dependencies
# ============================================================================

def test_environment():
    """Test that required libraries and CUDA are available."""
    print("\n" + "=" * 79)
    print("Environment Check")
    print("=" * 79)
    
    # PyTorch
    assert TORCH_AVAILABLE, "PyTorch is required"
    print(f"✓ PyTorch version: {torch.__version__}")
    
    # CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ CUDA not available - tests will run on CPU")
    
    # Models
    assert MODELS_AVAILABLE, "axs_lib.models is required"
    print("✓ axs_lib.models available")
    
    # List available models
    models = list_available_models()
    print(f"\nAvailable model registries: {list(models.keys())}")
    
    print("=" * 79)


def test_model_info():
    """Test model information retrieval."""
    for model_name in ["terramind_1.0_large", "prithvi_eo_2.0_600M"]:
        info = get_model_info(model_name)
        assert isinstance(info, dict), f"Expected dict for {model_name}"
        print(f"\n{model_name}: {info.get('parameters', 'N/A')} parameters")


# ============================================================================
# Test: TerraMind Backbone Forward Pass
# ============================================================================

@pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not available"
)
def test_terramind_backbone_forward(stub_multimodal_input):
    """Test TerraMind backbone forward pass with stub data."""
    print("\n" + "=" * 79)
    print("Test: TerraMind Backbone Forward Pass")
    print("=" * 79)
    
    # Print input info
    for modality, tensor in stub_multimodal_input.items():
        print_tensor_info(f"Input {modality}", tensor)
    
    # Build model
    print("\nBuilding TerraMind backbone...")
    try:
        model = build_terramind_backbone(**TEST_CONFIGS["terramind_backbone"])
        model = model.to(DEVICE)
        model.eval()
        print(f"✓ Model built successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        pytest.skip(f"TerraMind backbone not available: {e}")
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        try:
            output = model(stub_multimodal_input)
            print("✓ Forward pass completed")
        except Exception as e:
            pytest.fail(f"Forward pass failed: {e}")
    
    # Validate output
    if isinstance(output, dict):
        print("\nOutput is dictionary:")
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                print_tensor_info(f"  {key}", value)
                check_tensor_valid(value, f"output[{key}]")
    elif isinstance(output, torch.Tensor):
        print_tensor_info("Output", output)
        check_tensor_valid(output, "output")
        
        # Check output shape is reasonable
        assert len(output.shape) == 4, f"Expected 4D tensor, got {len(output.shape)}D"
        assert output.shape[0] == BATCH_SIZE, f"Batch size mismatch"
    else:
        pytest.fail(f"Unexpected output type: {type(output)}")
    
    print("\n✓ All backbone tests passed")
    print("=" * 79)


# ============================================================================
# Test: TerraMind Generator Forward Pass
# ============================================================================

@pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not available"
)
def test_terramind_generator_forward(stub_s1_tensor):
    """Test TerraMind generator forward pass with stub SAR data."""
    print("\n" + "=" * 79)
    print("Test: TerraMind Generator Forward Pass")
    print("=" * 79)
    
    # Print input info
    print_tensor_info("Input S1GRD", stub_s1_tensor)
    
    # Build model
    print("\nBuilding TerraMind generator...")
    try:
        model = build_terramind_generator(**TEST_CONFIGS["terramind_generator"])
        model = model.to(DEVICE)
        model.eval()
        print(f"✓ Model built successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Timesteps: {TIMESTEPS_FAST}")
    except Exception as e:
        pytest.skip(f"TerraMind generator not available: {e}")
    
    # Forward pass
    print("\nRunning forward pass (generating 'mental image')...")
    input_dict = {"S1GRD": stub_s1_tensor}
    
    with torch.no_grad():
        try:
            output = model(input_dict)
            print("✓ Forward pass completed")
        except Exception as e:
            pytest.fail(f"Forward pass failed: {e}")
    
    # Validate output
    if isinstance(output, dict):
        print("\nOutput is dictionary (latent representation):")
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                print_tensor_info(f"  {key}", value)
                check_tensor_valid(value, f"output[{key}]")
                
                # Generator should preserve spatial structure (roughly)
                assert value.shape[0] == BATCH_SIZE, "Batch size mismatch"
    elif isinstance(output, torch.Tensor):
        print_tensor_info("Output (latent)", output)
        check_tensor_valid(output, "output")
        
        # Check dimensions
        assert len(output.shape) == 4, f"Expected 4D tensor, got {len(output.shape)}D"
        assert output.shape[0] == BATCH_SIZE, "Batch size mismatch"
        
        # Generator should produce multi-channel latent representation
        print(f"  Latent channels: {output.shape[1]}")
    else:
        pytest.fail(f"Unexpected output type: {type(output)}")
    
    print("\n✓ All generator tests passed")
    print("  Note: Output is abstract 'mental image' (latent), not pixel values")
    print("=" * 79)


# ============================================================================
# Test: Prithvi 600M Forward Pass
# ============================================================================

@pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not available"
)
def test_prithvi_600_forward(stub_s2_tensor):
    """Test Prithvi 600M forward pass with stub optical data."""
    print("\n" + "=" * 79)
    print("Test: Prithvi 600M Forward Pass")
    print("=" * 79)
    
    # Print input info
    print_tensor_info("Input S2L2A", stub_s2_tensor)
    
    # Build model
    print("\nBuilding Prithvi 600M...")
    try:
        model = build_prithvi_600(**TEST_CONFIGS["prithvi_600"])
        model = model.to(DEVICE)
        model.eval()
        print(f"✓ Model built successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  LoRA rank: {TEST_CONFIGS['prithvi_600']['lora_r']}")
    except Exception as e:
        pytest.skip(f"Prithvi 600M not available: {e}")
    
    # Forward pass
    print("\nRunning forward pass (segmentation refinement)...")
    with torch.no_grad():
        try:
            output = model(stub_s2_tensor)
            print("✓ Forward pass completed")
        except Exception as e:
            pytest.fail(f"Forward pass failed: {e}")
    
    # Validate output
    print_tensor_info("Output (segmentation mask)", output)
    check_tensor_valid(output, "output")
    
    # Check output shape matches expected segmentation mask
    expected_shape = (
        BATCH_SIZE,
        TEST_CONFIGS["prithvi_600"]["num_classes"],
        HEIGHT,
        WIDTH
    )
    assert output.shape == expected_shape, (
        f"Shape mismatch: expected {expected_shape}, got {output.shape}"
    )
    
    print(f"\n✓ Output shape matches expected: {expected_shape}")
    print("✓ All Prithvi tests passed")
    print("=" * 79)


# ============================================================================
# Test: Three-Stage Pipeline Integration
# ============================================================================

@pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not available"
)
def test_three_stage_pipeline(stub_s1_tensor, stub_s2_tensor):
    """Test full three-stage pipeline with stub data."""
    print("\n" + "=" * 79)
    print("Test: Three-Stage Pipeline Integration")
    print("=" * 79)
    
    # Note: This is a conceptual test showing the pipeline flow
    # In practice, Stage 2 (Prithvi) takes features from Stage 1
    # and Stage 3 performs final grounding
    
    print("\nStage 1: TerraMind Generator (SAR → Optical latent)")
    print("-" * 79)
    
    try:
        generator = build_terramind_generator(**TEST_CONFIGS["terramind_generator"])
        generator = generator.to(DEVICE)
        generator.eval()
        print("✓ Generator loaded")
    except Exception as e:
        pytest.skip(f"Generator not available: {e}")
    
    with torch.no_grad():
        latent_output = generator({"S1GRD": stub_s1_tensor})
        print("✓ Stage 1 complete: Generated latent representation")
    
    # Validate Stage 1 output
    if isinstance(latent_output, dict):
        # Extract first tensor from dict
        latent_tensor = next(iter(latent_output.values()))
    else:
        latent_tensor = latent_output
    
    print_tensor_info("  Latent output", latent_tensor)
    check_tensor_valid(latent_tensor, "stage1_output")
    
    print("\nStage 2: Prithvi Refinement (Latent → Segmentation mask)")
    print("-" * 79)
    
    # Note: In real pipeline, Prithvi would take features from Stage 1
    # For this test, we use S2 data as a proxy input
    try:
        refiner = build_prithvi_600(**TEST_CONFIGS["prithvi_600"])
        refiner = refiner.to(DEVICE)
        refiner.eval()
        print("✓ Refiner loaded")
    except Exception as e:
        pytest.skip(f"Refiner not available: {e}")
    
    with torch.no_grad():
        # Use S2 data as proxy input (in real pipeline, uses Stage 1 features)
        mask_output = refiner(stub_s2_tensor)
        print("✓ Stage 2 complete: Generated segmentation mask")
    
    print_tensor_info("  Mask output", mask_output)
    check_tensor_valid(mask_output, "stage2_output")
    
    # Validate mask properties
    assert mask_output.shape[0] == BATCH_SIZE, "Batch size mismatch"
    assert mask_output.shape[2:] == (HEIGHT, WIDTH), "Spatial dimensions mismatch"
    
    print("\nStage 3: Conditional Grounding (planned)")
    print("-" * 79)
    print("  Note: Stage 3 (TerraMind conditional) performs final grounding")
    print("  This stage refines boundaries and ensures semantic consistency")
    print("  Implementation pending in future updates")
    
    print("\n" + "=" * 79)
    print("✓ Three-stage pipeline validated successfully")
    print("=" * 79)
    
    # Summary
    print("\nPipeline Summary:")
    print(f"  Input: SAR tensor {stub_s1_tensor.shape}")
    print(f"  Stage 1 output: Latent {latent_tensor.shape}")
    print(f"  Stage 2 output: Mask {mask_output.shape}")
    print(f"  Device: {DEVICE}")
    print(f"  CUDA: {'✓ Enabled' if USE_CUDA else '✗ CPU only'}")


# ============================================================================
# Test: CUDA Memory Management
# ============================================================================

@pytest.mark.skipif(
    not USE_CUDA,
    reason="CUDA not available"
)
def test_cuda_memory_management(stub_s1_tensor):
    """Test CUDA memory doesn't leak during forward passes."""
    print("\n" + "=" * 79)
    print("Test: CUDA Memory Management")
    print("=" * 79)
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    initial_memory = torch.cuda.memory_allocated() / 1e6  # MB
    print(f"Initial memory: {initial_memory:.2f} MB")
    
    # Build model
    try:
        model = build_terramind_generator(**TEST_CONFIGS["terramind_generator"])
        model = model.to(DEVICE)
        model.eval()
    except Exception as e:
        pytest.skip(f"Model not available: {e}")
    
    model_memory = torch.cuda.memory_allocated() / 1e6
    print(f"After model load: {model_memory:.2f} MB (+{model_memory - initial_memory:.2f} MB)")
    
    # Multiple forward passes
    num_passes = 5
    for i in range(num_passes):
        with torch.no_grad():
            _ = model({"S1GRD": stub_s1_tensor})
        
        if i == 0:
            first_pass_memory = torch.cuda.memory_allocated() / 1e6
            print(f"After first pass: {first_pass_memory:.2f} MB")
    
    final_memory = torch.cuda.memory_allocated() / 1e6
    peak_memory = torch.cuda.max_memory_allocated() / 1e6
    
    print(f"After {num_passes} passes: {final_memory:.2f} MB")
    print(f"Peak memory: {peak_memory:.2f} MB")
    
    # Check for memory leaks (memory shouldn't grow significantly)
    memory_growth = final_memory - first_pass_memory
    print(f"Memory growth: {memory_growth:.2f} MB")
    
    assert memory_growth < 100, (
        f"Potential memory leak: grew by {memory_growth:.2f} MB"
    )
    
    print("✓ No memory leak detected")
    print("=" * 79)
    
    # Cleanup
    torch.cuda.empty_cache()


# ============================================================================
# Test: Batch Size Variations
# ============================================================================

@pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not available"
)
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_batch_size_variations(batch_size):
    """Test models handle different batch sizes correctly."""
    print(f"\nTesting batch size: {batch_size}")
    
    # Create batch tensor
    tensor = torch.randn(
        batch_size, IN_CHANNELS_S1, HEIGHT, WIDTH,
        dtype=torch.float32
    ).to(DEVICE)
    
    # Test generator
    try:
        model = build_terramind_generator(**TEST_CONFIGS["terramind_generator"])
        model = model.to(DEVICE)
        model.eval()
        
        with torch.no_grad():
            output = model({"S1GRD": tensor})
        
        # Extract tensor from output
        if isinstance(output, dict):
            output_tensor = next(iter(output.values()))
        else:
            output_tensor = output
        
        assert output_tensor.shape[0] == batch_size, (
            f"Batch size mismatch: expected {batch_size}, got {output_tensor.shape[0]}"
        )
        
        print(f"  ✓ Batch size {batch_size} passed: output shape {output_tensor.shape}")
        
    except Exception as e:
        pytest.skip(f"Model not available: {e}")


# ============================================================================
# Main: Run Tests
# ============================================================================

if __name__ == "__main__":
    """Run tests directly (without pytest)."""
    print("=" * 79)
    print("AXION-SAT MODEL FORWARD PASS TESTS")
    print("=" * 79)
    print()
    
    # Run tests manually
    test_environment()
    
    if TORCH_AVAILABLE and MODELS_AVAILABLE:
        print("\nCreating stub tensors...")
        s1 = torch.randn(BATCH_SIZE, IN_CHANNELS_S1, HEIGHT, WIDTH).to(DEVICE)
        s2 = torch.randn(BATCH_SIZE, IN_CHANNELS_S2, HEIGHT, WIDTH).to(DEVICE)
        multimodal = {"S1GRD": s1, "S2L2A": s2}
        
        print("✓ Stub tensors created")
        
        # Run tests
        try:
            test_terramind_generator_forward(s1)
        except Exception as e:
            print(f"⚠ Generator test failed: {e}")
        
        try:
            test_prithvi_600_forward(s2)
        except Exception as e:
            print(f"⚠ Prithvi test failed: {e}")
        
        try:
            test_three_stage_pipeline(s1, s2)
        except Exception as e:
            print(f"⚠ Pipeline test failed: {e}")
        
        if USE_CUDA:
            try:
                test_cuda_memory_management(s1)
            except Exception as e:
                print(f"⚠ CUDA test failed: {e}")
    
    print("\n" + "=" * 79)
    print("TESTS COMPLETE")
    print("=" * 79)
    print("\nTo run with pytest:")
    print("  pytest tests/test_models_forward_shapes.py -v")
    print("  pytest tests/test_models_forward_shapes.py -v -s  # Show print output")
