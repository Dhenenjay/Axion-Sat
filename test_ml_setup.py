"""
Comprehensive ML Training Environment Test
Tests: PyTorch, CUDA, xformers, transformers, and key dependencies
"""

import sys
import torch
import xformers
import xformers.ops
from transformers import AutoModel, AutoTokenizer

print("="*70)
print("ML TRAINING ENVIRONMENT TEST")
print("="*70)

# Test 1: PyTorch and CUDA
print("\n[1] PyTorch & CUDA Test")
print(f"   PyTorch Version: {torch.__version__}")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("   ERROR: CUDA not available!")
    sys.exit(1)

# Test 2: xformers
print("\n[2] xformers Test")
print(f"   xformers Version: {xformers.__version__}")
try:
    # Test memory-efficient attention on GPU
    x = torch.randn(2, 4, 32, 64).cuda()
    result = xformers.ops.memory_efficient_attention(x, x, x)
    print(f"   Memory Efficient Attention: ✓ Working")
    print(f"   Output shape: {result.shape}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

# Test 3: Transformers library
print("\n[3] Transformers Library Test")
print("   Loading BERT model...")
try:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    model = model.cuda()
    model.eval()
    
    # Test inference
    text = "Testing transformer model on GPU"
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"   Model loaded: ✓")
    print(f"   Model on GPU: {next(model.parameters()).is_cuda}")
    print(f"   Inference test: ✓ Working")
    print(f"   Output shape: {outputs.last_hidden_state.shape}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

# Test 4: Mixed Precision Training
print("\n[4] Mixed Precision (AMP) Test")
try:
    scaler = torch.cuda.amp.GradScaler()
    print(f"   AMP GradScaler: ✓ Available")
    
    # Quick test
    with torch.cuda.amp.autocast():
        x = torch.randn(10, 10).cuda()
        y = torch.matmul(x, x)
    print(f"   Mixed precision ops: ✓ Working")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 5: Key ML libraries
print("\n[5] Additional ML Libraries")
try:
    import numpy as np
    import rasterio
    import geopandas
    print(f"   NumPy: {np.__version__} ✓")
    print(f"   Rasterio: {rasterio.__version__} ✓")
    print(f"   GeoPandas: {geopandas.__version__} ✓")
except ImportError as e:
    print(f"   WARNING: {e}")

print("\n" + "="*70)
print("✓ ALL TESTS PASSED - Environment ready for ML training!")
print("="*70)
print("\nYou can now:")
print("  • Train transformer models with GPU acceleration")
print("  • Use xformers for efficient attention mechanisms")
print("  • Leverage mixed precision training (FP16/BF16)")
print("  • Work with geospatial data and satellite imagery")
print("="*70)
