"""
Download all TerraMind tokenizers locally from HuggingFace.

This pre-downloads all tokenizers so they don't need to be fetched during inference.
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*80)
print("Downloading TerraMind Tokenizers from HuggingFace")
print("="*80)

tokenizers = [
    'terramind_v1_tokenizer_s2l2a',   # Sentinel-2 L2A optical
    'terramind_v1_tokenizer_s1grd',   # Sentinel-1 GRD SAR
    'terramind_v1_tokenizer_s1rtc',   # Sentinel-1 RTC SAR
    'terramind_v1_tokenizer_dem',     # Digital Elevation Model
    'terramind_v1_tokenizer_lulc',    # Land Use Land Cover
    'terramind_v1_tokenizer_ndvi',    # Vegetation Index
]

try:
    from terratorch import FULL_MODEL_REGISTRY
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    for i, tokenizer_name in enumerate(tokenizers, 1):
        print(f"[{i}/6] Downloading {tokenizer_name}...")
        try:
            tokenizer = FULL_MODEL_REGISTRY.build(
                tokenizer_name,
                pretrained=True
            )
            tokenizer = tokenizer.to(device)
            tokenizer.eval()
            
            print(f"  ✓ {tokenizer_name} downloaded successfully")
            print(f"    Type: {type(tokenizer)}")
            
            # Test with dummy data
            if 's2l2a' in tokenizer_name:
                dummy = torch.randn(1, 12, 224, 224).to(device)
            elif 's1' in tokenizer_name:
                dummy = torch.randn(1, 2, 224, 224).to(device)
            elif 'dem' in tokenizer_name:
                dummy = torch.randn(1, 1, 224, 224).to(device)
            elif 'lulc' in tokenizer_name:
                dummy = torch.randint(0, 10, (1, 1, 224, 224)).float().to(device)
            elif 'ndvi' in tokenizer_name:
                dummy = torch.randn(1, 1, 224, 224).to(device)
            
            with torch.no_grad():
                _, _, tokens = tokenizer.encode(dummy)
            
            print(f"    Tokens shape: {tokens.shape}")
            print(f"    Codebook size: {tokens.max().item() + 1}")
            print()
            
        except Exception as e:
            print(f"  ✗ Failed to download {tokenizer_name}: {e}\n")
            continue
    
    print("="*80)
    print("✓ All tokenizers downloaded and cached!")
    print("="*80)
    print("\nTokenizers are now cached in:")
    print("  ~/.cache/huggingface/hub/ (Linux/Mac)")
    print("  C:\\Users\\<user>\\.cache\\huggingface\\hub\\ (Windows)")
    print("\nThey will be loaded from cache in future runs.")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
