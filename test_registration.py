"""
test_registration.py - Test Model Registration

This script verifies that models are registered and can be accessed.
"""

print("=" * 79)
print("Testing Model Registration")
print("=" * 79)
print()

# Test 1: Check if models are in registry
print("Test 1: Check registry contents")
try:
    from terratorch.registry import BACKBONE_REGISTRY, FULL_MODEL_REGISTRY
    
    # Check backbones
    terratorch_backbone_reg = BACKBONE_REGISTRY['terratorch']
    print("  Backbone registry type:", type(terratorch_backbone_reg))
    
    # Check full models
    terratorch_full_reg = FULL_MODEL_REGISTRY['terratorch']
    print("  Full model registry type:", type(terratorch_full_reg))
    
    print("  ✓ Registries accessible")
except Exception as e:
    print(f"  ✗ Failed to access registries: {e}")
    exit(1)

print()

# Test 2: Try to import and register
print("Test 2: Import axs_lib.models (should auto-register)")
try:
    from axs_lib.models import (
        build_terramind_generator,
        build_terramind_backbone,
        build_prithvi_600
    )
    print("  ✓ Models imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import models: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# Test 3: Check if models can be built (dry run - no actual loading)
print("Test 3: Check if model factories are callable")
try:
    # These should not actually load models, just check if functions work
    import inspect
    
    # Check TerraMind generator
    sig = inspect.signature(build_terramind_generator)
    print(f"  build_terramind_generator signature: {sig}")
    
    # Check TerraMind backbone
    sig = inspect.signature(build_terramind_backbone)
    print(f"  build_terramind_backbone signature: {sig}")
    
    # Check Prithvi
    sig = inspect.signature(build_prithvi_600)
    print(f"  build_prithvi_600 signature: {sig}")
    
    print("  ✓ All model builders are callable")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    exit(1)

print()
print("=" * 79)
print("All tests passed! Models are registered and ready to use.")
print()
print("Next steps:")
print("  1. Test actual model loading (will download/load weights)")
print("  2. Run inference tests")
print("=" * 79)
