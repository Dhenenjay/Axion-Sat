"""
TerraMind Thinking-in-Modalities (TiM) Wrapper

Proper implementation of TerraMind using:
1. BACKBONE_REGISTRY for feature extraction with TiM
2. FULL_MODEL_REGISTRY for modality generation
3. Tokenizers for encoding/decoding modalities

Reference: https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import warnings


class TerraMindTiMBackbone(nn.Module):
    """
    TerraMind Thinking-in-Modalities Backbone.
    
    Uses the TiM approach: generates intermediate modalities (LULC, NDVI) 
    from input modalities, then uses both for feature extraction.
    """
    
    def __init__(
        self,
        modalities: List[str] = ['S1GRD'],
        tim_modalities: List[str] = ['LULC'],
        pretrained: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()
        
        from terratorch import BACKBONE_REGISTRY
        
        print(f"Building TerraMind TiM backbone...")
        print(f"  Input modalities: {modalities}")
        print(f"  TiM modalities: {tim_modalities}")
        
        # Build TiM backbone - generates intermediate modalities internally
        self.backbone = BACKBONE_REGISTRY.build(
            'terramind_v1_large_tim',
            pretrained=pretrained,
            modalities=modalities,
            tim_modalities=tim_modalities
        )
        
        self.modalities = modalities
        self.tim_modalities = tim_modalities
        self.device = device
        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using TiM approach.
        
        Args:
            x: Input tensor (B, C, H, W) for first modality
        
        Returns:
            features: Patch embeddings (B, num_patches, embed_dim)
                     Default: (B, 196, 768) for 224x224 input
        """
        # TerraMind expects dict input
        if not isinstance(x, dict):
            x = {self.modalities[0]: x}
        
        # Forward through TiM backbone
        # This internally:
        # 1. Generates TiM modalities (LULC, NDVI)
        # 2. Uses both input + generated modalities for encoding
        features = self.backbone(x)
        
        return features


class TerraMindGenerator(nn.Module):
    """
    TerraMind modality generator using FULL_MODEL_REGISTRY.
    
    Can generate any output modality from any input modality.
    """
    
    def __init__(
        self,
        input_modalities: List[str] = ['S1GRD'],
        output_modalities: List[str] = ['S2L2A'],
        timesteps: int = 10,
        standardize: bool = True,
        pretrained: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()
        
        from terratorch import FULL_MODEL_REGISTRY
        
        print(f"Building TerraMind generator...")
        print(f"  Input: {input_modalities}")
        print(f"  Output: {output_modalities}")
        print(f"  Diffusion timesteps: {timesteps}")
        
        self.generator = FULL_MODEL_REGISTRY.build(
            'terramind_v1_large_generate',
            pretrained=pretrained,
            modalities=input_modalities,
            output_modalities=output_modalities,
            timesteps=timesteps,
            standardize=standardize
        )
        
        self.input_modalities = input_modalities
        self.output_modalities = output_modalities
        self.device = device
        
        # Move generator to device
        self.generator = self.generator.to(device)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate output modalities from input.
        
        Args:
            x: Input tensor (B, C, H, W) or dict of modalities
        
        Returns:
            outputs: Dict mapping modality names to tensors
        """
        # Convert to dict if needed
        if not isinstance(x, dict):
            x = {self.input_modalities[0]: x}
        
        # Generate modalities
        outputs = self.generator(x)
        
        return outputs


class TerraMindTokenizers:
    """
    Manager for TerraMind tokenizers (FSQ-VAE).
    
    Downloads and loads tokenizers from HuggingFace:
    - S1GRD, S1RTC: SAR tokenizers
    - S2L2A: Optical tokenizer
    - DEM: Elevation tokenizer
    - LULC: Land-use land-cover tokenizer
    - NDVI: Vegetation index tokenizer
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.tokenizers = {}
        
    def load_tokenizer(self, modality: str, pretrained: bool = True):
        """
        Load a specific tokenizer.
        
        Args:
            modality: One of 's1grd', 's1rtc', 's2l2a', 'dem', 'lulc', 'ndvi'
            pretrained: Load pretrained weights
        
        Returns:
            tokenizer: TerraMind tokenizer model
        """
        from terratorch import FULL_MODEL_REGISTRY
        
        modality_lower = modality.lower()
        tokenizer_name = f'terramind_v1_tokenizer_{modality_lower}'
        
        if modality_lower in self.tokenizers:
            return self.tokenizers[modality_lower]
        
        print(f"Loading tokenizer: {tokenizer_name}...")
        tokenizer = FULL_MODEL_REGISTRY.build(
            tokenizer_name,
            pretrained=pretrained
        )
        tokenizer = tokenizer.to(self.device)
        tokenizer.eval()
        
        self.tokenizers[modality_lower] = tokenizer
        return tokenizer
    
    def encode(self, modality: str, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode image to FSQ tokens.
        
        Returns:
            quantized: Quantized latent
            commitment_loss: Commitment loss
            tokens: FSQ tokens
        """
        tokenizer = self.load_tokenizer(modality)
        with torch.no_grad():
            quantized, commitment_loss, tokens = tokenizer.encode(x)
        return quantized, commitment_loss, tokens
    
    def decode(self, modality: str, tokens: torch.Tensor, timesteps: int = 10) -> torch.Tensor:
        """
        Decode FSQ tokens to image using diffusion.
        
        Args:
            modality: Output modality
            tokens: FSQ tokens
            timesteps: Number of diffusion steps
        
        Returns:
            reconstruction: Decoded image
        """
        tokenizer = self.load_tokenizer(modality)
        with torch.no_grad():
            reconstruction = tokenizer.decode_tokens(tokens, verbose=False, timesteps=timesteps)
        return reconstruction


class TerraMindSAR2Optical(nn.Module):
    """
    Complete SAR to Optical pipeline using TerraMind with tokenizers.
    
    Combines:
    1. TiM backbone for rich feature extraction
    2. Tokenizers for proper encoding/decoding
    3. Generator for any-to-any modality translation
    4. Proper standardization and band handling
    """
    
    def __init__(
        self,
        use_tokenizers: bool = True,
        use_tim: bool = True,
        tim_modalities: List[str] = ['LULC'],
        output_bands: str = 'all',  # 'all' (12 S2L2A) or 'hls' (6 bands)
        timesteps: int = 10,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.use_tokenizers = use_tokenizers
        self.use_tim = use_tim
        self.output_bands = output_bands
        self.timesteps = timesteps
        self.device = device
        
        if use_tokenizers:
            # Use tokenizer-based workflow
            print("Using tokenizer-based TerraMind workflow...")
            self.tokenizers = TerraMindTokenizers(device=device)
            
            # Load S1GRD and S2L2A tokenizers
            self.tokenizers.load_tokenizer('s1grd', pretrained=True)
            self.tokenizers.load_tokenizer('s2l2a', pretrained=True)
            
            # Use generator for SAR -> Optical
            self.generator = TerraMindGenerator(
                input_modalities=['S1GRD'],
                output_modalities=['S2L2A'],
                timesteps=timesteps,
                standardize=True,
                pretrained=True,
                device=device
            )
            
        elif use_tim:
            # Use TiM backbone with custom decoder
            print("Using TiM backbone with custom decoder...")
            self.backbone = TerraMindTiMBackbone(
                modalities=['S1GRD'],
                tim_modalities=tim_modalities,
                pretrained=True,
                device=device
            )
            
            # Add decoder head to convert embeddings to optical
            # Embeddings: (B, 196, 768) -> need to upsample to (B, 12, H, W)
            self.decoder = nn.Sequential(
                nn.Linear(768, 1536),
                nn.GELU(),
                nn.Linear(1536, 3072),
                nn.GELU(),
            )
            
            # Spatial upsampling
            self.spatial_upsample = nn.ConvTranspose2d(
                3072 // 196,  # channels per patch
                12,  # S2L2A bands
                kernel_size=16,
                stride=16
            )
            
        else:
            # Use generator directly (no tokenizers)
            print("Using generator directly...")
            self.generator = TerraMindGenerator(
                input_modalities=['S1GRD'],
                output_modalities=['S2L2A'],
                timesteps=timesteps,
                standardize=True,
                pretrained=True,
                device=device
            )
        
        self.to(device)
    
    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        """
        Convert SAR to optical using TerraMind.
        
        Args:
            sar: SAR tensor (B, 2, H, W) - VV, VH channels
        
        Returns:
            optical: Optical tensor (B, C, H, W)
                    C = 12 for all S2L2A bands
                    C = 6 for HLS bands (Blue, Green, Red, NIR, SWIR1, SWIR2)
        """
        if self.use_tokenizers:
            # Tokenizer-based workflow:
            # 1. Encode SAR to tokens (optional)
            # 2. Use generator for SAR -> Optical translation
            # 3. Decode optical tokens to image
            
            # Use generator directly (it handles tokenization internally)
            outputs = self.generator(sar)
            optical = outputs['S2L2A']
            
        elif self.use_tim:
            # Extract features with TiM
            features = self.backbone(sar)  # (B, 196, 768)
            
            # Decode to optical
            B, N, D = features.shape
            decoded = self.decoder(features)  # (B, 196, 3072)
            
            # Reshape for spatial upsampling
            H_patches = W_patches = int(N ** 0.5)  # 14x14 patches
            C_per_patch = decoded.shape[-1] // N
            decoded = decoded.view(B, H_patches, W_patches, C_per_patch)
            decoded = decoded.permute(0, 3, 1, 2)  # (B, C, 14, 14)
            
            # Upsample to original resolution
            optical = self.spatial_upsample(decoded)  # (B, 12, 224, 224)
            
        else:
            # Generate directly (no tokenizers, no TiM)
            outputs = self.generator(sar)
            optical = outputs['S2L2A']
        
        # Extract HLS bands if requested
        if self.output_bands == 'hls':
            # S2L2A order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
            # HLS bands: B02(1), B03(2), B04(3), B08(7), B11(10), B12(11)
            hls_indices = [1, 2, 3, 7, 10, 11]
            optical = optical[:, hls_indices, :, :]
        
        # Normalize to [0, 1] reflectance
        optical = torch.clamp(optical, 0, 10000) / 10000.0
        
        return optical


def build_terramind_stage1(
    use_tokenizers: bool = True,
    use_tim: bool = False,
    output_bands: str = 'all',
    device: str = 'cuda'
) -> nn.Module:
    """
    Build TerraMind Stage 1 model with proper configuration.
    
    Args:
        use_tokenizers: Use tokenizer-based workflow (recommended)
        use_tim: Use Thinking-in-Modalities backbone
        output_bands: 'all' (12 S2L2A) or 'hls' (6 HLS)
        device: Device to load model on
    
    Returns:
        model: TerraMind Stage 1 model
    """
    model = TerraMindSAR2Optical(
        use_tokenizers=use_tokenizers,
        use_tim=use_tim,
        tim_modalities=['LULC'],
        output_bands=output_bands,
        timesteps=10,
        device=device
    )
    
    model.eval()
    return model


def test_terramind_tim():
    """Test TerraMind TiM implementation."""
    print("Testing TerraMind TiM implementation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test 1: TiM Backbone
    print("\n1. Testing TiM Backbone...")
    backbone = TerraMindTiMBackbone(
        modalities=['S1GRD'],
        tim_modalities=['LULC'],
        device=device
    )
    
    sar = torch.randn(1, 2, 224, 224).to(device)
    features = backbone(sar)
    print(f"   Input: {sar.shape}")
    print(f"   Features: {features.shape}")
    assert features.shape == (1, 196, 768), "Incorrect feature shape"
    print("   ✓ TiM backbone works!")
    
    # Test 2: Generator
    print("\n2. Testing Generator...")
    generator = TerraMindGenerator(
        input_modalities=['S1GRD'],
        output_modalities=['S2L2A', 'LULC'],
        device=device
    )
    
    outputs = generator(sar)
    print(f"   Generated modalities: {outputs.keys()}")
    print(f"   S2L2A shape: {outputs['S2L2A'].shape}")
    print(f"   LULC shape: {outputs['LULC'].shape}")
    print("   ✓ Generator works!")
    
    # Test 3: SAR2Optical pipeline with tokenizers
    print("\n3. Testing SAR2Optical pipeline with tokenizers...")
    model = TerraMindSAR2Optical(use_tokenizers=True, use_tim=False, device=device)
    
    sar_small = torch.randn(1, 2, 120, 120).to(device)
    optical = model(sar_small)
    print(f"   Input: {sar_small.shape}")
    print(f"   Output: {optical.shape}")
    print(f"   Range: [{optical.min():.4f}, {optical.max():.4f}]")
    print("   ✓ SAR2Optical with tokenizers works!")
    
    # Test 4: Direct tokenizer usage
    print("\n4. Testing direct tokenizer usage...")
    tokenizers = TerraMindTokenizers(device=device)
    
    # Test S2L2A tokenizer encode/decode
    s2_test = torch.randn(1, 12, 224, 224).to(device)
    _, _, tokens = tokenizers.encode('s2l2a', s2_test)
    reconstruction = tokenizers.decode('s2l2a', tokens, timesteps=5)
    print(f"   Input: {s2_test.shape}")
    print(f"   Tokens: {tokens.shape}")
    print(f"   Reconstruction: {reconstruction.shape}")
    print("   ✓ Tokenizers work!")
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_terramind_tim()
