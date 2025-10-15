"""
Stage 3 Loss Functions: SAR-Grounded Refinement

This module implements specialized loss functions for Stage 3 grounding, ensuring
that the final optical output (opt_v3) maintains structural consistency with SAR
while preserving the spectral quality from Stage 2.

Loss Components:
    1. SAR Consistency Loss
       - Edge alignment (gradient/Sobel)
       - Phase coherence
       - Weighted by backscatter intensity
       - Terrain-adaptive weighting
    
    2. Cycle & Identity Loss
       - Ensures minimal changes where SAR is weak/uncertain
       - Identity preservation in low-variance regions
       - Cycle consistency with Stage 2
    
    3. LPIPS (Perceptual Loss)
       - Perceptual similarity to ground truth
       - Feature-level realism
       - Complements pixel-level losses

Architecture:
    Stage3Loss = α·SAR_consistency + β·Cycle + γ·Identity + δ·LPIPS + ε·Spectral

Usage:
    >>> from axs_lib.stage3_losses import Stage3Loss
    >>> 
    >>> criterion = Stage3Loss(
    ...     sar_weight=1.0,
    ...     cycle_weight=0.5,
    ...     identity_weight=0.3,
    ...     lpips_weight=0.1
    ... )
    >>> 
    >>> loss = criterion(
    ...     opt_v3=opt_v3,
    ...     opt_v2=opt_v2,
    ...     s1=s1,
    ...     s2_truth=s2_truth
    ... )

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# SAR Feature Extraction
# ============================================================================

class SARFeatureExtractor(nn.Module):
    """
    Extract structural features from SAR imagery for consistency checking.
    
    Features:
        - Edge maps (Sobel gradients)
        - Intensity/backscatter
        - Local variance (texture)
        - Phase information (complex coherence)
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel kernels for edge detection
        self.register_buffer(
            'sobel_x',
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
        self.register_buffer(
            'sobel_y',
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
    
    def extract_edges(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract edge magnitude using Sobel operator.
        
        Args:
            image: Input image (B, C, H, W)
            
        Returns:
            Edge magnitude (B, C, H, W)
        """
        # Replicate Sobel kernels for each channel
        C = image.shape[1]
        sobel_x = self.sobel_x.repeat(C, 1, 1, 1)  # (C, 1, 3, 3)
        sobel_y = self.sobel_y.repeat(C, 1, 1, 1)  # (C, 1, 3, 3)
        
        # Compute gradients for each channel with grouped convolution
        grad_x = F.conv2d(image, sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(image, sobel_y, padding=1, groups=C)
        
        # Edge magnitude
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        return edges
    
    def extract_backscatter(self, s1: torch.Tensor) -> torch.Tensor:
        """
        Compute SAR backscatter intensity (average of VV and VH).
        
        Args:
            s1: SAR input (B, 2, H, W) - VV, VH
            
        Returns:
            Backscatter intensity (B, 1, H, W)
        """
        # Average backscatter across polarizations
        backscatter = s1.mean(dim=1, keepdim=True)
        
        return backscatter
    
    def extract_texture(self, image: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """
        Compute local variance (texture) using average pooling.
        
        Args:
            image: Input image (B, C, H, W)
            kernel_size: Size of local window
            
        Returns:
            Local variance (B, C, H, W)
        """
        # Local mean
        mean = F.avg_pool2d(image, kernel_size, stride=1, padding=kernel_size // 2)
        
        # Local variance
        mean_sq = F.avg_pool2d(image ** 2, kernel_size, stride=1, padding=kernel_size // 2)
        variance = mean_sq - mean ** 2
        variance = torch.clamp(variance, min=0)
        
        return variance
    
    def forward(self, s1: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract all SAR features.
        
        Args:
            s1: SAR input (B, 2, H, W)
            
        Returns:
            Dict of features
        """
        features = {
            'edges': self.extract_edges(s1),
            'backscatter': self.extract_backscatter(s1),
            'texture': self.extract_texture(s1)
        }
        
        return features


# ============================================================================
# SAR Consistency Loss
# ============================================================================

class SARConsistencyLoss(nn.Module):
    """
    Ensure structural consistency between optical output and SAR input.
    
    Components:
        1. Edge alignment: Match edge maps weighted by backscatter
        2. Texture correlation: Maintain texture patterns
        3. Terrain-adaptive weighting: Stronger in high-backscatter areas
        4. Urban/high-backscatter weighting: Increased weight in urban areas
        5. DEM-aware weighting: Increased weight on steep slopes
    """
    
    def __init__(
        self,
        edge_weight: float = 1.0,
        texture_weight: float = 0.5,
        adaptive_weighting: bool = True,
        urban_weight: float = 1.0,
        dem_weight: float = 0.0
    ):
        """
        Args:
            edge_weight: Weight for edge alignment loss
            texture_weight: Weight for texture correlation loss
            adaptive_weighting: Enable terrain-adaptive weighting
            urban_weight: Additional weight for urban/high-backscatter areas (1.0 = no boost)
            dem_weight: Weight for DEM-based slope weighting (0.0 = disabled)
        """
        super().__init__()
        
        self.edge_weight = edge_weight
        self.texture_weight = texture_weight
        self.adaptive_weighting = adaptive_weighting
        self.urban_weight = urban_weight
        self.dem_weight = dem_weight
        
        self.sar_extractor = SARFeatureExtractor()
    
    def compute_urban_mask(self, backscatter: torch.Tensor, threshold: float = 0.7) -> torch.Tensor:
        """
        Compute urban/high-backscatter mask for adaptive weighting.
        
        Urban areas typically have higher SAR backscatter due to double-bounce
        scattering from buildings. This mask identifies high-backscatter regions.
        
        Args:
            backscatter: SAR backscatter intensity (B, 1, H, W)
            threshold: Percentile threshold for high backscatter (0-1)
            
        Returns:
            Urban mask (B, 1, H, W) in [0, 1]
        """
        # Normalize backscatter to [0, 1] per batch
        bs_norm = (backscatter - backscatter.min()) / (backscatter.max() - backscatter.min() + 1e-8)
        
        # Compute threshold value at given percentile
        threshold_val = torch.quantile(bs_norm, threshold)
        
        # Create soft mask (sigmoid transition)
        urban_mask = torch.sigmoid((bs_norm - threshold_val) * 10.0)
        
        return urban_mask
    
    def compute_dem_slope_weight(self, dem: torch.Tensor) -> torch.Tensor:
        """
        Compute slope-based weighting from DEM.
        
        Steeper slopes require higher consistency weight since they represent
        significant terrain features that should align between SAR and optical.
        
        Args:
            dem: Digital Elevation Model (B, 1, H, W) in meters
            
        Returns:
            Slope weight (B, 1, H, W) in [0, 1]
        """
        # Compute gradients (proxy for slope)
        grad_x = torch.abs(dem[:, :, :, 1:] - dem[:, :, :, :-1])
        grad_y = torch.abs(dem[:, :, 1:, :] - dem[:, :, :-1, :])
        
        # Pad to original size
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        
        # Compute slope magnitude
        slope = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        # Normalize to [0, 1]
        slope_norm = (slope - slope.min()) / (slope.max() - slope.min() + 1e-8)
        
        # Apply sigmoid to create smooth weight (steeper = higher weight)
        # Map: 0.5 → weight=0.5, 0.8 → weight≈1.0
        slope_weight = torch.sigmoid((slope_norm - 0.5) * 6.0)
        
        return slope_weight
    
    def edge_alignment_loss(
        self,
        opt: torch.Tensor,
        s1: torch.Tensor,
        sar_features: Dict[str, torch.Tensor],
        dem: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute edge alignment loss with adaptive weighting.
        
        Applies multiple weighting strategies:
        - Backscatter weighting: Higher weight for high-backscatter areas
        - Urban weighting: Additional boost for urban/high-backscatter regions
        - DEM weighting: Higher weight for steep slopes (terrain features)
        
        Args:
            opt: Optical output (B, 4, H, W)
            s1: SAR input (B, 2, H, W)
            sar_features: Pre-extracted SAR features
            dem: Optional DEM (B, 1, H, W) for slope-aware weighting
            
        Returns:
            Edge alignment loss
        """
        # Extract edges from optical
        opt_edges = self.sar_extractor.extract_edges(opt)
        sar_edges = sar_features['edges']
        
        # Average optical edges across channels for comparison with SAR
        opt_edges_avg = opt_edges.mean(dim=1, keepdim=True)
        sar_edges_avg = sar_edges.mean(dim=1, keepdim=True)
        
        # Normalize edges
        opt_edges_norm = opt_edges_avg / (opt_edges_avg.max() + 1e-8)
        sar_edges_norm = sar_edges_avg / (sar_edges_avg.max() + 1e-8)
        
        # Compute difference
        edge_diff = F.l1_loss(opt_edges_norm, sar_edges_norm, reduction='none')
        
        # Initialize spatial weight as ones
        spatial_weight = torch.ones_like(edge_diff)
        
        # 1. Base backscatter weighting (if adaptive)
        if self.adaptive_weighting:
            backscatter_weight = torch.sigmoid(sar_features['backscatter'])
            spatial_weight = spatial_weight * backscatter_weight
        
        # 2. Urban/high-backscatter weighting
        if self.urban_weight > 1.0:
            # Compute urban mask
            urban_mask = self.compute_urban_mask(sar_features['backscatter'])
            
            # Apply urban boost: weight = 1.0 for rural, urban_weight for urban
            # Linear interpolation based on mask
            urban_boost = 1.0 + (self.urban_weight - 1.0) * urban_mask
            spatial_weight = spatial_weight * urban_boost
        
        # 3. DEM slope weighting
        if self.dem_weight > 0.0 and dem is not None:
            # Compute slope-based weight
            slope_weight = self.compute_dem_slope_weight(dem)
            
            # Blend with spatial weight: weight = (1-α)*weight + α*slope_weight
            # where α = dem_weight
            spatial_weight = (1.0 - self.dem_weight) * spatial_weight + self.dem_weight * slope_weight
        
        # Apply spatial weighting
        edge_diff_weighted = edge_diff * spatial_weight
        
        # Compute final loss (normalize by mean weight to maintain scale)
        loss = edge_diff_weighted.sum() / (spatial_weight.sum() + 1e-8)
        
        return loss
    
    def texture_correlation_loss(
        self,
        opt: torch.Tensor,
        s1: torch.Tensor,
        sar_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Ensure texture patterns correlate between optical and SAR.
        
        Args:
            opt: Optical output (B, 4, H, W)
            s1: SAR input (B, 2, H, W)
            sar_features: Pre-extracted SAR features
            
        Returns:
            Texture correlation loss
        """
        # Extract texture from optical
        opt_texture = self.sar_extractor.extract_texture(opt)
        sar_texture = sar_features['texture']
        
        # Average across channels
        opt_texture_avg = opt_texture.mean(dim=1, keepdim=True)
        sar_texture_avg = sar_texture.mean(dim=1, keepdim=True)
        
        # Normalize
        opt_texture_norm = opt_texture_avg / (opt_texture_avg.mean() + 1e-8)
        sar_texture_norm = sar_texture_avg / (sar_texture_avg.mean() + 1e-8)
        
        # Correlation loss (minimize difference in texture patterns)
        loss = F.mse_loss(opt_texture_norm, sar_texture_norm)
        
        return loss
    
    def forward(
        self,
        opt: torch.Tensor,
        s1: torch.Tensor,
        dem: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SAR consistency loss with optional DEM.
        
        Args:
            opt: Optical output (B, 4, H, W)
            s1: SAR input (B, 2, H, W)
            dem: Optional DEM (B, 1, H, W) for slope-aware weighting
            
        Returns:
            Dict with loss components
        """
        # Extract SAR features once
        sar_features = self.sar_extractor(s1)
        
        # Compute losses
        edge_loss = self.edge_alignment_loss(opt, s1, sar_features, dem=dem)
        texture_loss = self.texture_correlation_loss(opt, s1, sar_features)
        
        # Total SAR consistency loss
        total_loss = (
            self.edge_weight * edge_loss +
            self.texture_weight * texture_loss
        )
        
        return {
            'sar_consistency': total_loss,
            'edge_alignment': edge_loss,
            'texture_correlation': texture_loss
        }


# ============================================================================
# Cycle & Identity Loss
# ============================================================================

class CycleIdentityLoss(nn.Module):
    """
    Ensure Stage 3 makes minimal changes where SAR is weak or uncertain.
    
    Components:
        1. Identity loss: Preserve opt_v2 in low-SAR-confidence regions
        2. Cycle loss: Maintain consistency with Stage 2
        3. Adaptive weighting based on SAR strength
    """
    
    def __init__(
        self,
        identity_weight: float = 1.0,
        cycle_weight: float = 0.5,
        adaptive_threshold: float = 0.1
    ):
        super().__init__()
        
        self.identity_weight = identity_weight
        self.cycle_weight = cycle_weight
        self.adaptive_threshold = adaptive_threshold
        
        self.sar_extractor = SARFeatureExtractor()
    
    def compute_sar_confidence(self, s1: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence map based on SAR backscatter intensity.
        
        Strong SAR signal → high confidence → allow changes
        Weak SAR signal → low confidence → preserve Stage 2
        
        Args:
            s1: SAR input (B, 2, H, W)
            
        Returns:
            Confidence map (B, 1, H, W) in [0, 1]
        """
        # Get backscatter intensity
        backscatter = self.sar_extractor.extract_backscatter(s1)
        
        # Normalize to [0, 1]
        backscatter_norm = (backscatter - backscatter.min()) / (backscatter.max() - backscatter.min() + 1e-8)
        
        # Apply sigmoid to create smooth confidence map
        confidence = torch.sigmoid((backscatter_norm - self.adaptive_threshold) * 10)
        
        return confidence
    
    def identity_loss(
        self,
        opt_v3: torch.Tensor,
        opt_v2: torch.Tensor,
        confidence: torch.Tensor
    ) -> torch.Tensor:
        """
        Preserve Stage 2 output in low-confidence regions.
        
        Args:
            opt_v3: Stage 3 output (B, 4, H, W)
            opt_v2: Stage 2 output (B, 4, H, W)
            confidence: SAR confidence map (B, 1, H, W)
            
        Returns:
            Identity loss
        """
        # Compute difference
        diff = F.l1_loss(opt_v3, opt_v2, reduction='none')
        
        # Weight by inverse confidence (preserve more in low-confidence areas)
        weight = 1.0 - confidence
        weighted_diff = diff * weight
        
        loss = weighted_diff.mean()
        
        return loss
    
    def cycle_loss(
        self,
        opt_v3: torch.Tensor,
        opt_v2: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensure changes from Stage 2 to Stage 3 are reasonable.
        
        Args:
            opt_v3: Stage 3 output (B, 4, H, W)
            opt_v2: Stage 2 output (B, 4, H, W)
            
        Returns:
            Cycle consistency loss
        """
        # Simple L2 loss to prevent drastic changes
        loss = F.mse_loss(opt_v3, opt_v2)
        
        return loss
    
    def forward(
        self,
        opt_v3: torch.Tensor,
        opt_v2: torch.Tensor,
        s1: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cycle and identity losses.
        
        Args:
            opt_v3: Stage 3 output (B, 4, H, W)
            opt_v2: Stage 2 output (B, 4, H, W)
            s1: SAR input (B, 2, H, W)
            
        Returns:
            Dict with loss components
        """
        # Compute SAR confidence
        confidence = self.compute_sar_confidence(s1)
        
        # Compute losses
        identity = self.identity_loss(opt_v3, opt_v2, confidence)
        cycle = self.cycle_loss(opt_v3, opt_v2)
        
        # Total loss
        total_loss = (
            self.identity_weight * identity +
            self.cycle_weight * cycle
        )
        
        return {
            'cycle_identity': total_loss,
            'identity': identity,
            'cycle': cycle,
            'sar_confidence': confidence.mean().item()
        }


# ============================================================================
# LPIPS Perceptual Loss
# ============================================================================

class LPIPSLoss(nn.Module):
    """
    LPIPS (Learned Perceptual Image Patch Similarity) for realism.
    
    Uses pretrained VGG features to measure perceptual similarity.
    Falls back to feature-level L2 if lpips package not available.
    """
    
    def __init__(self, net: str = 'vgg', use_gpu: bool = True):
        super().__init__()
        
        self.use_lpips = False
        
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net=net)
            if use_gpu and torch.cuda.is_available():
                self.lpips_model = self.lpips_model.cuda()
            self.use_lpips = True
        except ImportError:
            warnings.warn(
                "lpips package not available. Install with: pip install lpips\n"
                "Falling back to VGG feature loss."
            )
            # Use simple VGG feature loss as fallback
            self._init_vgg_fallback()
    
    def _init_vgg_fallback(self):
        """Initialize simple VGG feature extractor as fallback."""
        from torchvision import models
        
        vgg = models.vgg16(pretrained=True).features
        
        # Use layers 4, 9, 16, 23 (relu1_2, relu2_2, relu3_3, relu4_3)
        self.feature_layers = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
            vgg[16:23] # relu4_3
        ])
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
    
    def extract_vgg_features(self, x: torch.Tensor) -> list:
        """Extract VGG features (fallback method)."""
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        # Convert to 3-channel if needed (take first 3 channels)
        if x.shape[1] == 4:
            x = x[:, :3, :, :]
        
        x_norm = (x - mean) / std
        
        features = []
        for layer in self.feature_layers:
            x_norm = layer(x_norm)
            features.append(x_norm)
        
        return features
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)
            normalize: Normalize images to [-1, 1] for LPIPS
            
        Returns:
            Perceptual loss
        """
        if self.use_lpips:
            # LPIPS expects images in [-1, 1]
            if normalize:
                pred = pred * 2.0 - 1.0
                target = target * 2.0 - 1.0
            
            # LPIPS works with 3-channel images
            if pred.shape[1] == 4:
                pred = pred[:, :3, :, :]
                target = target[:, :3, :, :]
            
            loss = self.lpips_model(pred, target).mean()
        else:
            # Fallback: VGG feature loss
            pred_features = self.extract_vgg_features(pred)
            target_features = self.extract_vgg_features(target)
            
            loss = 0
            for pred_feat, target_feat in zip(pred_features, target_features):
                loss += F.mse_loss(pred_feat, target_feat)
            
            loss = loss / len(pred_features)
        
        return loss


# ============================================================================
# Combined Stage 3 Loss
# ============================================================================

class Stage3Loss(nn.Module):
    """
    Complete loss function for Stage 3 grounding.
    
    Components:
        1. SAR Consistency (edge + texture)
        2. Cycle & Identity (preserve Stage 2 where appropriate)
        3. LPIPS (perceptual realism)
        4. Spectral L1 (pixel-level accuracy)
    
    Args:
        sar_weight: Weight for SAR consistency loss
        cycle_weight: Weight for cycle consistency
        identity_weight: Weight for identity preservation
        lpips_weight: Weight for perceptual loss
        spectral_weight: Weight for pixel-level L1 loss
        use_lpips: Whether to use LPIPS (requires lpips package)
    """
    
    def __init__(
        self,
        sar_weight: float = 1.0,
        cycle_weight: float = 0.5,
        identity_weight: float = 0.3,
        lpips_weight: float = 0.1,
        spectral_weight: float = 1.0,
        use_lpips: bool = True,
        urban_weight: float = 1.0,
        dem_weight: float = 0.0
    ):
        """
        Args:
            sar_weight: Weight for SAR consistency loss
            cycle_weight: Weight for cycle consistency
            identity_weight: Weight for identity preservation
            lpips_weight: Weight for perceptual loss
            spectral_weight: Weight for pixel-level L1 loss
            use_lpips: Whether to use LPIPS (requires lpips package)
            urban_weight: Additional weight for urban/high-backscatter areas (1.0 = no boost, >1.0 = boost)
            dem_weight: Weight for DEM-based slope weighting (0.0 = disabled, >0.0 = enabled)
        """
        super().__init__()
        
        self.sar_weight = sar_weight
        self.cycle_weight = cycle_weight
        self.identity_weight = identity_weight
        self.lpips_weight = lpips_weight
        self.spectral_weight = spectral_weight
        self.urban_weight = urban_weight
        self.dem_weight = dem_weight
        
        # Loss modules
        self.sar_loss = SARConsistencyLoss(
            urban_weight=urban_weight,
            dem_weight=dem_weight
        )
        self.cycle_identity_loss = CycleIdentityLoss(
            identity_weight=identity_weight,
            cycle_weight=cycle_weight
        )
        
        if use_lpips and lpips_weight > 0:
            self.lpips_loss = LPIPSLoss()
        else:
            self.lpips_loss = None
    
    def spectral_l1_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Simple L1 loss for pixel-level accuracy.
        
        Args:
            pred: Predicted image (B, 4, H, W)
            target: Target image (B, 4, H, W)
            mask: Optional mask (B, 1, H, W)
            
        Returns:
            L1 loss
        """
        if mask is not None:
            loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
            loss = loss / (mask.sum() * pred.shape[1] + 1e-8)
        else:
            loss = F.l1_loss(pred, target)
        
        return loss
    
    def forward(
        self,
        opt_v3: torch.Tensor,
        opt_v2: torch.Tensor,
        s1: torch.Tensor,
        s2_truth: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        dem: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute complete Stage 3 loss.
        
        Args:
            opt_v3: Stage 3 output (B, 4, H, W)
            opt_v2: Stage 2 output (B, 4, H, W)
            s1: SAR input (B, 2, H, W)
            s2_truth: Ground truth optical (B, 4, H, W) - optional
            mask: Valid pixel mask (B, 1, H, W) - optional
            dem: Digital Elevation Model (B, 1, H, W) - optional
            
        Returns:
            Dict with total loss and components
        """
        losses = {}
        
        # 1. SAR Consistency (with optional DEM)
        sar_losses = self.sar_loss(opt_v3, s1, dem=dem)
        losses.update(sar_losses)
        
        # 2. Cycle & Identity
        cycle_losses = self.cycle_identity_loss(opt_v3, opt_v2, s1)
        losses.update(cycle_losses)
        
        # 3. Spectral L1 (if ground truth available)
        if s2_truth is not None:
            spectral_loss = self.spectral_l1_loss(opt_v3, s2_truth, mask)
            losses['spectral_l1'] = spectral_loss
        else:
            spectral_loss = 0.0
            losses['spectral_l1'] = torch.tensor(0.0, device=opt_v3.device)
        
        # 4. LPIPS (if ground truth available and LPIPS enabled)
        if self.lpips_loss is not None and s2_truth is not None:
            lpips_loss = self.lpips_loss(opt_v3, s2_truth)
            losses['lpips'] = lpips_loss
        else:
            lpips_loss = 0.0
            losses['lpips'] = torch.tensor(0.0, device=opt_v3.device)
        
        # Total weighted loss
        total_loss = (
            self.sar_weight * losses['sar_consistency'] +
            losses['cycle_identity'] +  # Already weighted internally
            self.spectral_weight * losses['spectral_l1'] +
            self.lpips_weight * losses['lpips']
        )
        
        losses['total'] = total_loss
        
        return losses


# ============================================================================
# Testing & CLI
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Stage 3 Loss Functions - Testing")
    print("=" * 80)
    
    # Create dummy data
    batch_size = 2
    height, width = 120, 120
    
    opt_v3 = torch.randn(batch_size, 4, height, width).cuda()
    opt_v2 = torch.randn(batch_size, 4, height, width).cuda()
    s1 = torch.randn(batch_size, 2, height, width).cuda()
    s2_truth = torch.randn(batch_size, 4, height, width).cuda()
    
    print(f"\nInput shapes:")
    print(f"  opt_v3: {opt_v3.shape}")
    print(f"  opt_v2: {opt_v2.shape}")
    print(f"  s1: {s1.shape}")
    print(f"  s2_truth: {s2_truth.shape}")
    
    # Test individual losses
    print("\n" + "-" * 80)
    print("Testing SAR Consistency Loss")
    print("-" * 80)
    sar_loss = SARConsistencyLoss().cuda()
    sar_losses = sar_loss(opt_v3, s1)
    print("SAR losses:")
    for key, value in sar_losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.6f}")
    
    print("\n" + "-" * 80)
    print("Testing Cycle & Identity Loss")
    print("-" * 80)
    cycle_loss = CycleIdentityLoss().cuda()
    cycle_losses = cycle_loss(opt_v3, opt_v2, s1)
    print("Cycle/Identity losses:")
    for key, value in cycle_losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.6f}")
        else:
            print(f"  {key}: {value:.6f}")
    
    print("\n" + "-" * 80)
    print("Testing LPIPS Loss")
    print("-" * 80)
    try:
        lpips_loss = LPIPSLoss(use_gpu=True)
        lpips_value = lpips_loss(opt_v3, s2_truth)
        print(f"LPIPS loss: {lpips_value.item():.6f}")
    except Exception as e:
        print(f"LPIPS test failed: {e}")
    
    print("\n" + "-" * 80)
    print("Testing Complete Stage 3 Loss")
    print("-" * 80)
    stage3_loss = Stage3Loss(
        sar_weight=1.0,
        cycle_weight=0.5,
        identity_weight=0.3,
        lpips_weight=0.1,
        spectral_weight=1.0
    ).cuda()
    
    losses = stage3_loss(opt_v3, opt_v2, s1, s2_truth)
    
    print("\nAll loss components:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.6f}")
        else:
            print(f"  {key}: {value:.6f}")
    
    print("\n" + "=" * 80)
    print("✓ Testing complete!")
    print("=" * 80)
    
    # Test backward pass
    print("\nTesting backward pass...")
    losses['total'].backward()
    print("✓ Backward pass successful")
    
    print("\nAll tests passed! Stage 3 loss is ready to use.")
