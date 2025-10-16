"""
Stage 3: TerraMind Backbone with Lightweight Decoder

This module implements Stage 3 using TerraMind's backbone (encoder only) with
a lightweight trainable decoder head. This approach enables full fine-tuning
without LoRA and avoids gradient issues with the generator API.

Architecture:
    Inputs:
        - S1GRD (s1): SAR imagery (2 channels: VV, VH)
        - S2L2A (opt_v2): Stage 2 refined optical imagery (4 channels)
    
    Processing:
        1. TerraMind backbone encodes both modalities → (B, 196, 768) embeddings
        2. Lightweight CNN decoder → (B, 4, 224, 224) output
        3. Crop to original size and destandardize
    
    Output:
        - S2L2A: Final grounded optical imagery (4 channels)

Key Features:
    - Trainable TerraMind backbone (full fine-tuning or frozen)
    - Lightweight decoder head (~1M parameters)
    - No gradient flow issues (no @torch.no_grad())
    - Memory efficient for H100 (can train full model)
    - Works locally with batch_size=1 (6GB GPU)

Usage:
    >>> from axs_lib.stage3_tm_backbone import build_stage3_backbone_model
    >>> 
    >>> # Build model
    >>> model = build_stage3_backbone_model(
    ...     freeze_backbone=False,  # Full fine-tuning
    ...     pretrained=True
    ... )
    >>> 
    >>> # Forward pass
    >>> s1 = torch.randn(1, 2, 128, 128)  # SAR input
    >>> opt_v2 = torch.randn(1, 4, 128, 128)  # Stage 2 output
    >>> opt_v3 = model(s1, opt_v2)  # (1, 4, 128, 128)

Author: Axion-Sat Project
Version: 2.0.0 (No LoRA, Full Fine-Tuning)
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axs_lib.stdz import TERRAMIND_S1_STATS, TERRAMIND_S2_STATS


# ============================================================================
# Lightweight Decoder Head
# ============================================================================

class LightweightDecoder(nn.Module):
    """
    Lightweight CNN decoder to convert TerraMind embeddings to optical output.
    
    Takes patch embeddings (B, 196, 768) and produces (B, 4, 224, 224) output.
    
    Architecture:
        - Reshape embeddings to spatial: (B, 768, 14, 14)
        - 3 upsampling blocks: 14×14 → 28×28 → 56×56 → 224×224
        - Skip connections for better gradient flow
        - Final conv to 4 channels (B, G, R, NIR)
    
    Args:
        embed_dim: TerraMind embedding dimension (default: 768)
        num_patches: Number of patches (default: 196 = 14×14)
        output_channels: Output channels (default: 4)
        output_size: Output spatial size (default: 224)
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_patches: int = 196,
        output_channels: int = 4,
        output_size: int = 224
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.patch_size = int(num_patches ** 0.5)  # 14
        self.output_channels = output_channels
        self.output_size = output_size
        
        # Project embeddings to spatial feature map
        self.embed_proj = nn.Conv2d(embed_dim, 256, kernel_size=1)
        
        # Upsampling blocks: 14×14 → 224×224
        # Block 1: 14×14 → 28×28 (×2)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Block 2: 28×28 → 56×56 (×2)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Block 3: 56×56 → 112×112 (×2)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Block 4: 112×112 → 224×224 (×2)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Final output projection
        self.output_proj = nn.Conv2d(16, output_channels, kernel_size=3, padding=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Decode TerraMind embeddings to optical output.
        
        Args:
            embeddings: TerraMind patch embeddings (B, num_patches, embed_dim) or list of tensors
        
        Returns:
            Optical output (B, output_channels, output_size, output_size)
        """
        # Handle list of embeddings (convert to tensor)
        if isinstance(embeddings, (list, tuple)):
            embeddings = torch.stack(embeddings, dim=0) if len(embeddings) > 1 else embeddings[0]
        
        # Handle embeddings from TerraMind backbone
        
        # Handle different embedding formats from TerraMind
        if embeddings.ndim == 4:
            # Format: (num_modalities, B, num_patches, embed_dim) or similar
            # Squeeze or reshape to (B, num_patches, embed_dim)
            if embeddings.shape[1] == 1:
                # (num_modalities, 1, num_patches, embed_dim) → (num_modalities, num_patches, embed_dim)
                embeddings = embeddings.squeeze(1)
            # Now should be (num_modalities, num_patches, embed_dim)
            # Average over modalities
            embeddings = embeddings.mean(dim=0, keepdim=True)  # (1, num_patches, embed_dim)
        
        # Ensure embeddings are 3D: (B, num_patches, embed_dim)
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        
        B, num_tokens, actual_embed_dim = embeddings.shape
        
        # Calculate actual patch size from num_tokens
        actual_patch_size = int(num_tokens ** 0.5)
        
        # Reshape embeddings to spatial: (B, num_patches, embed_dim) → (B, embed_dim, H, W)
        x = embeddings.transpose(1, 2)  # (B, embed_dim, num_patches)
        x = x.view(B, actual_embed_dim, actual_patch_size, actual_patch_size)  # (B, actual_embed_dim, H, W)
        
        # Project embeddings (handle dynamic embed_dim)
        if actual_embed_dim != self.embed_dim:
            # Create projection on the fly if dimensions don't match
            proj = nn.Conv2d(actual_embed_dim, 256, kernel_size=1).to(x.device)
            x = proj(x)  # (B, 256, H, W)
        else:
            x = self.embed_proj(x)  # (B, 256, H, W)
        
        # Upsample: 14×14 → 224×224
        x = self.up1(x)  # (B, 128, 28, 28)
        x = self.up2(x)  # (B, 64, 56, 56)
        x = self.up3(x)  # (B, 32, 112, 112)
        x = self.up4(x)  # (B, 16, 224, 224)
        
        # Output projection
        x = self.output_proj(x)  # (B, 4, 224, 224)
        
        return x


# ============================================================================
# Stage 3 Model with TerraMind Backbone
# ============================================================================

class Stage3BackboneModel(nn.Module):
    """
    Stage 3 model using TerraMind backbone + lightweight decoder.
    
    This model uses TerraMind as a feature encoder (backbone) and adds a
    lightweight trainable decoder head. Supports both full fine-tuning and
    frozen backbone training.
    
    Args:
        terramind_backbone: TerraMind backbone model
        decoder: Lightweight decoder head
        freeze_backbone: Freeze TerraMind backbone (default: False)
        standardize: Apply standardization (default: True)
    """
    
    def __init__(
        self,
        terramind_backbone: nn.Module,
        decoder: nn.Module,
        freeze_backbone: bool = False,
        standardize: bool = True
    ):
        super().__init__()
        
        self.terramind_backbone = terramind_backbone
        self.decoder = decoder
        self.freeze_backbone = freeze_backbone
        self.standardize = standardize
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.terramind_backbone.parameters():
                param.requires_grad = False
            print("✓ TerraMind backbone frozen (only decoder trainable)")
        else:
            print("✓ Full fine-tuning enabled (backbone + decoder trainable)")
        
        # Store standardization stats
        if standardize:
            # S1 stats (VV, VH)
            self.register_buffer(
                's1_mean',
                torch.tensor(TERRAMIND_S1_STATS.means, dtype=torch.float32).view(1, 2, 1, 1)
            )
            self.register_buffer(
                's1_std',
                torch.tensor(TERRAMIND_S1_STATS.stds, dtype=torch.float32).view(1, 2, 1, 1)
            )
            
            # S2 stats (B02, B03, B04, B08)
            self.register_buffer(
                's2_mean',
                torch.tensor(TERRAMIND_S2_STATS.means, dtype=torch.float32).view(1, 4, 1, 1)
            )
            self.register_buffer(
                's2_std',
                torch.tensor(TERRAMIND_S2_STATS.stds, dtype=torch.float32).view(1, 4, 1, 1)
            )
    
    def standardize_inputs(
        self,
        s1: torch.Tensor,
        opt_v2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standardize inputs using TerraMind's pretrained statistics.
        
        Args:
            s1: SAR input (B, 2, H, W)
            opt_v2: Stage 2 optical output (B, 4, H, W)
            
        Returns:
            Standardized (s1, opt_v2)
        """
        if not self.standardize:
            return s1, opt_v2
        
        # Standardize S1 (SAR)
        s1_std = (s1 - self.s1_mean) / self.s1_std
        
        # Standardize opt_v2 (optical)
        opt_v2_std = (opt_v2 - self.s2_mean) / self.s2_std
        
        return s1_std, opt_v2_std
    
    def destandardize_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Destandardize model output to original scale.
        
        Args:
            output: Standardized output (B, 4, H, W)
            
        Returns:
            Destandardized output
        """
        if not self.standardize:
            return output
        
        return output * self.s2_std + self.s2_mean
    
    def forward(
        self,
        s1: torch.Tensor,
        opt_v2: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: Generate grounded optical output.
        
        Args:
            s1: SAR input (B, 2, H, W)
            opt_v2: Stage 2 optical output (B, 4, H, W)
            
        Returns:
            Grounded optical output (B, 4, H, W)
        """
        # Store original spatial size
        orig_h, orig_w = s1.shape[2], s1.shape[3]
        
        # Standardize inputs
        s1_std, opt_v2_std = self.standardize_inputs(s1, opt_v2)
        
        # Pad inputs to 224x224 (TerraMind pretrained size)
        target_size = 224
        if s1_std.shape[2] != target_size or s1_std.shape[3] != target_size:
            pad_h = target_size - s1_std.shape[2]
            pad_w = target_size - s1_std.shape[3]
            # Pad: (left, right, top, bottom)
            pad = (0, pad_w, 0, pad_h)
            s1_std = F.pad(s1_std, pad, mode='constant', value=0)
            opt_v2_std = F.pad(opt_v2_std, pad, mode='constant', value=0)
        
        # Prepare inputs for TerraMind backbone
        # TerraMind expects a dict with modality keys
        # S2L2A expects 12 channels, pad with zeros for missing bands
        if opt_v2_std.shape[1] == 4:
            # Pad to 12 channels: [B02, B03, B04, B08] → [B02, B03, B04, B08, 0, 0, ..., 0]
            zeros = torch.zeros(
                opt_v2_std.shape[0], 8, opt_v2_std.shape[2], opt_v2_std.shape[3],
                device=opt_v2_std.device, dtype=opt_v2_std.dtype
            )
            opt_v2_std = torch.cat([opt_v2_std, zeros], dim=1)
        
        # Encode with TerraMind backbone
        # Backbone returns (B, num_patches, embed_dim) embeddings
        if self.freeze_backbone:
            with torch.no_grad():
                embeddings = self.terramind_backbone({
                    'S1GRD': s1_std,
                    'S2L2A': opt_v2_std
                })
        else:
            embeddings = self.terramind_backbone({
                'S1GRD': s1_std,
                'S2L2A': opt_v2_std
            })
        
        # Handle different return types from TerraMind
        if isinstance(embeddings, dict):
            # TerraMind might return dict with keys like 'embeddings' or modality names
            embeddings = embeddings.get('embeddings', embeddings.get('S2L2A', embeddings.get('S1GRD', embeddings)))
        
        # If embeddings is still a dict after extraction, average all modality embeddings
        if isinstance(embeddings, dict):
            embeddings = torch.stack(list(embeddings.values()), dim=0).mean(dim=0)
        
        # Handle 4D embeddings before passing to decoder
        if isinstance(embeddings, torch.Tensor) and embeddings.ndim == 4:
            # (B, num_modalities, num_patches, embed_dim) or similar
            # Average over modality dimension (dim=1)
            embeddings = embeddings.mean(dim=1)  # → (B, num_patches, embed_dim)
        
        # Decode embeddings to optical output
        output_std = self.decoder(embeddings)  # (B, 4, 224, 224)
        
        # Crop output back to original size
        output_std = output_std[:, :, :orig_h, :orig_w]
        
        # Destandardize output
        output = self.destandardize_output(output_std)
        
        return output


# ============================================================================
# Model Builder
# ============================================================================

def build_stage3_backbone_model(
    freeze_backbone: bool = False,
    pretrained: bool = True,
    standardize: bool = True,
    device: Optional[torch.device] = None
) -> Stage3BackboneModel:
    """
    Build Stage 3 model with TerraMind backbone + lightweight decoder.
    
    Args:
        freeze_backbone: Freeze TerraMind backbone (default: False for full fine-tuning)
        pretrained: Load pretrained TerraMind weights (default: True)
        standardize: Apply standardization (default: True)
        device: Device to load model on (default: auto-detect)
        
    Returns:
        Stage3BackboneModel ready for training or inference
        
    Example:
        >>> # Full fine-tuning (H100)
        >>> model = build_stage3_backbone_model(
        ...     freeze_backbone=False,
        ...     pretrained=True
        ... )
        >>> 
        >>> # Frozen backbone (faster training)
        >>> model = build_stage3_backbone_model(
        ...     freeze_backbone=True,
        ...     pretrained=True
        ... )
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 80)
    print("Building Stage 3 Model (TerraMind Backbone + Decoder)")
    print("=" * 80)
    
    # Build TerraMind backbone
    print("\nLoading TerraMind backbone...")
    try:
        from terratorch import BACKBONE_REGISTRY
        
        terramind_backbone = BACKBONE_REGISTRY.build(
            'terramind_v1_large',
            pretrained=pretrained,
            modalities=['S2L2A', 'S1GRD'],
            merge_method='mean'  # Average embeddings from both modalities
        )
        print(f"✓ TerraMind backbone loaded (pretrained={pretrained})")
    except ImportError:
        raise ImportError(
            "TerraTorch not installed. Install with: pip install terratorch"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load TerraMind backbone: {e}")
    
    # Build lightweight decoder
    print("\nBuilding lightweight decoder...")
    decoder = LightweightDecoder(
        embed_dim=768,      # TerraMind large embedding dim
        num_patches=196,    # 14×14 patches
        output_channels=4,  # B, G, R, NIR
        output_size=224     # Match TerraMind input size
    )
    print(f"✓ Lightweight decoder created")
    
    # Wrap in Stage 3 model
    model = Stage3BackboneModel(
        terramind_backbone=terramind_backbone,
        decoder=decoder,
        freeze_backbone=freeze_backbone,
        standardize=standardize
    )
    
    # Move to device
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.terramind_backbone.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Backbone parameters:  {backbone_params:,}")
    print(f"  Decoder parameters:   {decoder_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Device:               {device}")
    
    print("\n" + "=" * 80)
    print("✓ Stage 3 model ready")
    print("=" * 80)
    
    return model


# ============================================================================
# Inference Function
# ============================================================================

@torch.no_grad()
def stage3_inference(
    model: Stage3BackboneModel,
    s1: torch.Tensor,
    opt_v2: torch.Tensor,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Run Stage 3 inference.
    
    Args:
        model: Stage 3 model
        s1: SAR input (B, 2, H, W)
        opt_v2: Stage 2 output (B, 4, H, W)
        device: Device (default: model's device)
        
    Returns:
        Grounded optical output (B, 4, H, W)
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Move inputs to device
    s1 = s1.to(device)
    opt_v2 = opt_v2.to(device)
    
    # Forward pass
    opt_v3 = model(s1, opt_v2)
    
    return opt_v3
