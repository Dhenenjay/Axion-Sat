"""
Stage 3: TerraMind with SAR-Optical Cross-Attention (PANGAEA Optimized)

This is an enhanced version of Stage 3 that explicitly fuses SAR and optical
features using cross-attention, designed to maximize performance on benchmarks
like PANGAEA.

Key Improvements over baseline:
1. Separate SAR and optical embedding streams (no averaging)
2. Cross-attention decoder that conditions optical on SAR structure
3. Stronger SAR grounding loss (10× weight)
4. Multi-scale feature fusion

Architecture:
    Inputs:
        - S1GRD (s1): SAR imagery (2 channels: VV, VH)
        - S2L2A (opt_v2): Stage 2 refined optical (4 channels)
    
    Processing:
        1. TerraMind encodes S1 → SAR embeddings (B, 196, 1024)
        2. TerraMind encodes S2 → Optical embeddings (B, 196, 1024)
        3. Cross-Attention Decoder:
           - Query: Optical embeddings
           - Key/Value: SAR embeddings
           - Output: SAR-grounded optical features
        4. CNN decoder → (B, 4, 224, 224) output
    
    Output:
        - S2L2A: Final grounded optical imagery (4 channels)

Author: Axion-Sat Project
Version: 3.0.0 (Cross-Attention, PANGAEA-Optimized)
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axs_lib.stdz import TERRAMIND_S1_STATS, TERRAMIND_S2_STATS


# ============================================================================
# Cross-Attention Module for SAR-Optical Fusion
# ============================================================================

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention module to fuse SAR and optical features.
    
    Query: Optical embeddings (what to refine)
    Key/Value: SAR embeddings (structural guidance)
    
    Args:
        embed_dim: Embedding dimension (default: 1024)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Query projection (from optical)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        
        # Key/Value projections (from SAR)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cross-attention: optical attends to SAR.
        
        Args:
            optical: Optical embeddings (B, N, embed_dim) - QUERY
            sar: SAR embeddings (B, N, embed_dim) - KEY/VALUE
        
        Returns:
            Fused features (B, N, embed_dim)
        """
        B, N, C = optical.shape
        
        # Multi-head cross-attention
        # Query from optical
        q = self.q_proj(optical).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        
        # Key/Value from SAR
        k = self.k_proj(sar).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        v = self.v_proj(sar).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        
        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, N, N)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = attn_weights @ v  # (B, H, N, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)  # (B, N, C)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        # Residual connection + norm
        optical = self.norm1(optical + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(optical)
        optical = self.norm2(optical + ffn_output)
        
        return optical


# ============================================================================
# Enhanced Decoder with Cross-Attention
# ============================================================================

class CrossAttentionDecoder(nn.Module):
    """
    Enhanced decoder with SAR-optical cross-attention fusion.
    
    Takes separate SAR and optical embeddings, fuses them with cross-attention,
    then decodes to optical output.
    
    Args:
        embed_dim: Embedding dimension (default: 1024)
        num_patches: Number of patches (default: 196)
        output_channels: Output channels (default: 4)
        output_size: Output spatial size (default: 224)
        num_fusion_layers: Number of cross-attention layers (default: 2)
    """
    
    def __init__(
        self,
        embed_dim: int = 1024,
        num_patches: int = 196,
        output_channels: int = 4,
        output_size: int = 224,
        num_fusion_layers: int = 2
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.patch_size = int(num_patches ** 0.5)
        self.output_channels = output_channels
        self.output_size = output_size
        
        # Cross-attention fusion layers
        self.fusion_layers = nn.ModuleList([
            CrossAttentionFusion(embed_dim=embed_dim, num_heads=8, dropout=0.1)
            for _ in range(num_fusion_layers)
        ])
        
        # Project fused embeddings to spatial
        self.embed_proj = nn.Conv2d(embed_dim, 256, kernel_size=1)
        
        # Upsampling blocks
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Output projection
        self.output_proj = nn.Conv2d(16, output_channels, kernel_size=3, padding=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        optical_emb: torch.Tensor,
        sar_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode with SAR-optical cross-attention fusion.
        
        Args:
            optical_emb: Optical embeddings (B, num_patches, embed_dim)
            sar_emb: SAR embeddings (B, num_patches, embed_dim)
        
        Returns:
            Optical output (B, output_channels, output_size, output_size)
        """
        # Apply cross-attention fusion layers
        fused = optical_emb
        for fusion_layer in self.fusion_layers:
            fused = fusion_layer(fused, sar_emb)  # Optical attends to SAR
        
        # Reshape to spatial
        B = fused.shape[0]
        actual_patch_size = int(fused.shape[1] ** 0.5)
        actual_embed_dim = fused.shape[2]
        
        x = fused.transpose(1, 2)  # (B, embed_dim, num_patches)
        x = x.view(B, actual_embed_dim, actual_patch_size, actual_patch_size)
        
        # Handle dynamic embed_dim
        if actual_embed_dim != self.embed_dim:
            proj = nn.Conv2d(actual_embed_dim, 256, kernel_size=1).to(x.device)
            x = proj(x)
        else:
            x = self.embed_proj(x)
        
        # Upsample
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        
        # Output
        x = self.output_proj(x)
        
        return x


# ============================================================================
# Stage 3 Model with Cross-Attention
# ============================================================================

class Stage3CrossAttentionModel(nn.Module):
    """
    Stage 3 model with explicit SAR-optical cross-attention fusion.
    
    Args:
        terramind_backbone: TerraMind backbone model
        decoder: Cross-attention decoder
        freeze_backbone: Freeze TerraMind (default: False)
        standardize: Apply standardization (default: True)
    """
    
    def __init__(
        self,
        terramind_backbone: nn.Module,
        decoder: CrossAttentionDecoder,
        freeze_backbone: bool = False,
        standardize: bool = True
    ):
        super().__init__()
        
        self.terramind_backbone = terramind_backbone
        self.decoder = decoder
        self.freeze_backbone = freeze_backbone
        self.standardize = standardize
        
        if freeze_backbone:
            for param in self.terramind_backbone.parameters():
                param.requires_grad = False
            print("✓ TerraMind backbone frozen")
        else:
            print("✓ Full fine-tuning enabled (backbone + decoder)")
        
        # Standardization stats
        if standardize:
            self.register_buffer(
                's1_mean',
                torch.tensor(TERRAMIND_S1_STATS.means, dtype=torch.float32).view(1, 2, 1, 1)
            )
            self.register_buffer(
                's1_std',
                torch.tensor(TERRAMIND_S1_STATS.stds, dtype=torch.float32).view(1, 2, 1, 1)
            )
            self.register_buffer(
                's2_mean',
                torch.tensor(TERRAMIND_S2_STATS.means, dtype=torch.float32).view(1, 4, 1, 1)
            )
            self.register_buffer(
                's2_std',
                torch.tensor(TERRAMIND_S2_STATS.stds, dtype=torch.float32).view(1, 4, 1, 1)
            )
    
    def standardize_inputs(self, s1, opt_v2):
        if not self.standardize:
            return s1, opt_v2
        return (s1 - self.s1_mean) / self.s1_std, (opt_v2 - self.s2_mean) / self.s2_std
    
    def destandardize_output(self, output):
        if not self.standardize:
            return output
        return output * self.s2_std + self.s2_mean
    
    def forward(self, s1, opt_v2):
        orig_h, orig_w = s1.shape[2], s1.shape[3]
        
        # Standardize
        s1_std, opt_v2_std = self.standardize_inputs(s1, opt_v2)
        
        # Pad to 224×224
        target_size = 224
        if s1_std.shape[2] != target_size:
            pad_h = target_size - s1_std.shape[2]
            pad_w = target_size - s1_std.shape[3]
            pad = (0, pad_w, 0, pad_h)
            s1_std = F.pad(s1_std, pad, mode='constant', value=0)
            opt_v2_std = F.pad(opt_v2_std, pad, mode='constant', value=0)
        
        # Pad opt_v2 to 12 channels
        if opt_v2_std.shape[1] == 4:
            zeros = torch.zeros(
                opt_v2_std.shape[0], 8, opt_v2_std.shape[2], opt_v2_std.shape[3],
                device=opt_v2_std.device, dtype=opt_v2_std.dtype
            )
            opt_v2_std = torch.cat([opt_v2_std, zeros], dim=1)
        
        # Encode SAR and optical SEPARATELY (no averaging!)
        if self.freeze_backbone:
            with torch.no_grad():
                sar_emb = self.terramind_backbone({'S1GRD': s1_std})
                opt_emb = self.terramind_backbone({'S2L2A': opt_v2_std})
        else:
            sar_emb = self.terramind_backbone({'S1GRD': s1_std})
            opt_emb = self.terramind_backbone({'S2L2A': opt_v2_std})
        
        # Handle embedding format (might be list or 4D tensor)
        def process_emb(emb):
            if isinstance(emb, (list, tuple)):
                emb = emb[0] if len(emb) == 1 else torch.stack(emb, dim=0).mean(0)
            if emb.ndim == 4 and emb.shape[1] == 1:
                emb = emb.squeeze(1).mean(0, keepdim=True)
            elif emb.ndim == 4:
                emb = emb.mean(0, keepdim=True)
            if emb.dim() == 2:
                emb = emb.unsqueeze(0)
            return emb
        
        sar_emb = process_emb(sar_emb)
        opt_emb = process_emb(opt_emb)
        
        # Cross-attention fusion and decode
        output_std = self.decoder(opt_emb, sar_emb)
        
        # Crop and destandardize
        output_std = output_std[:, :, :orig_h, :orig_w]
        output = self.destandardize_output(output_std)
        
        return output


# ============================================================================
# Model Builder
# ============================================================================

def build_stage3_crossattn_model(
    freeze_backbone: bool = False,
    pretrained: bool = True,
    standardize: bool = True,
    num_fusion_layers: int = 2,
    device: Optional[torch.device] = None
) -> Stage3CrossAttentionModel:
    """
    Build Stage 3 with SAR-optical cross-attention (PANGAEA-optimized).
    
    Args:
        freeze_backbone: Freeze backbone (default: False)
        pretrained: Use pretrained weights (default: True)
        standardize: Apply standardization (default: True)
        num_fusion_layers: Number of cross-attention layers (default: 2)
        device: Device (default: auto-detect)
    
    Returns:
        Stage3CrossAttentionModel
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 80)
    print("Building Stage 3 Model (Cross-Attention, PANGAEA-Optimized)")
    print("=" * 80)
    
    # Build TerraMind backbone
    print("\nLoading TerraMind backbone...")
    try:
        from terratorch import BACKBONE_REGISTRY
        
        # Load backbone WITHOUT merge - we'll handle fusion explicitly
        terramind_backbone = BACKBONE_REGISTRY.build(
            'terramind_v1_large',
            pretrained=pretrained,
            modalities=['S2L2A', 'S1GRD'],
            merge_method=None  # ← NO MERGING! Keep streams separate
        )
        print(f"✓ TerraMind backbone loaded (separate modality streams)")
    except Exception as e:
        raise RuntimeError(f"Failed to load TerraMind: {e}")
    
    # Build cross-attention decoder
    print(f"\nBuilding cross-attention decoder ({num_fusion_layers} fusion layers)...")
    decoder = CrossAttentionDecoder(
        embed_dim=1024,
        num_patches=196,
        output_channels=4,
        output_size=224,
        num_fusion_layers=num_fusion_layers
    )
    print("✓ Cross-attention decoder created")
    
    # Wrap in model
    model = Stage3CrossAttentionModel(
        terramind_backbone=terramind_backbone,
        decoder=decoder,
        freeze_backbone=freeze_backbone,
        standardize=standardize
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Decoder parameters:   {decoder_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Device:               {device}")
    
    print("\n" + "=" * 80)
    print("✓ Stage 3 model ready (with SAR-optical cross-attention)")
    print("=" * 80)
    
    return model


# ============================================================================
# Inference
# ============================================================================

@torch.no_grad()
def stage3_inference(model, s1, opt_v2, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    return model(s1.to(device), opt_v2.to(device))
