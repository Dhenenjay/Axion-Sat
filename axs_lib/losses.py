"""
Loss functions for SAR-to-optical image translation.

This module provides various loss functions optimized for satellite imagery:
- L1 loss for pixel-wise reconstruction
- MS-SSIM for multi-scale structural similarity
- LPIPS for perceptual similarity (optional)
- SAR structure loss for edge/gradient consistency
- Spectral index loss for vegetation and land cover accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import warnings

try:
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False
    warnings.warn("torchmetrics not available, MS-SSIM will use fallback implementation")

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    warnings.warn("lpips package not available, LPIPS loss will be disabled")


class L1Loss(nn.Module):
    """
    Standard L1 (Mean Absolute Error) loss.
    
    Args:
        reduction: Specifies the reduction to apply ('mean', 'sum', 'none')
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 loss between prediction and target.
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
            
        Returns:
            L1 loss value
        """
        return F.l1_loss(pred, target, reduction=self.reduction)


class MSSSIMLoss(nn.Module):
    """
    Multi-Scale Structural Similarity Index loss.
    
    This measures perceptual similarity across multiple scales, which is
    particularly useful for preserving texture and structure in images.
    
    Args:
        data_range: The dynamic range of the images (default: 1.0 for normalized images)
        kernel_size: Size of the gaussian kernel (default: 11)
        k1: Algorithm parameter (default: 0.01)
        k2: Algorithm parameter (default: 0.03)
        use_torchmetrics: Whether to use torchmetrics implementation (default: True)
    """
    
    def __init__(
        self,
        data_range: float = 1.0,
        kernel_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03,
        use_torchmetrics: bool = True
    ):
        super().__init__()
        self.data_range = data_range
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        
        if use_torchmetrics and HAS_TORCHMETRICS:
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
                data_range=data_range,
                kernel_size=kernel_size,
                k1=k1,
                k2=k2,
                reduction='elementwise_mean'
            )
            self.use_torchmetrics = True
        else:
            self.use_torchmetrics = False
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MS-SSIM loss (1 - MS-SSIM) between prediction and target.
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
            
        Returns:
            MS-SSIM loss value (1 - MS-SSIM score)
        """
        if self.use_torchmetrics:
            # torchmetrics returns MS-SSIM score (higher is better)
            # Convert to loss (lower is better)
            ms_ssim_score = self.ms_ssim(pred, target)
            return 1.0 - ms_ssim_score
        else:
            # Fallback to simple SSIM if MS-SSIM not available
            return 1.0 - self._ssim(pred, target)
    
    def _ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Fallback single-scale SSIM implementation."""
        C1 = (self.k1 * self.data_range) ** 2
        C2 = (self.k2 * self.data_range) ** 2
        
        mu_pred = F.avg_pool2d(pred, self.kernel_size, stride=1, padding=self.kernel_size // 2)
        mu_target = F.avg_pool2d(target, self.kernel_size, stride=1, padding=self.kernel_size // 2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.avg_pool2d(pred ** 2, self.kernel_size, stride=1, padding=self.kernel_size // 2) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target ** 2, self.kernel_size, stride=1, padding=self.kernel_size // 2) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred * target, self.kernel_size, stride=1, padding=self.kernel_size // 2) - mu_pred_target
        
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        return ssim_map.mean()


class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity loss.
    
    Uses a pretrained network to compute perceptual similarity.
    Requires the 'lpips' package to be installed.
    
    Args:
        net: Network to use ('alex', 'vgg', 'squeeze')
        use_gpu: Whether to use GPU (automatically detected if None)
    """
    
    def __init__(self, net: str = 'alex', use_gpu: Optional[bool] = None):
        super().__init__()
        
        if not HAS_LPIPS:
            raise ImportError(
                "lpips package is required for LPIPS loss. "
                "Install it with: pip install lpips"
            )
        
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        
        self.loss_fn = lpips.LPIPS(net=net)
        if use_gpu:
            self.loss_fn = self.loss_fn.cuda()
        
        # Freeze LPIPS network
        for param in self.loss_fn.parameters():
            param.requires_grad = False
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS loss between prediction and target.
        
        Args:
            pred: Predicted images (B, C, H, W), values in [-1, 1] or [0, 1]
            target: Target images (B, C, H, W), values in [-1, 1] or [0, 1]
            
        Returns:
            LPIPS loss value
        """
        # LPIPS expects 3-channel RGB images
        if pred.shape[1] != 3:
            # If more than 3 channels, take first 3 (assumes RGB ordering)
            pred = pred[:, :3, :, :]
            target = target[:, :3, :, :]
        
        return self.loss_fn(pred, target).mean()


class SARStructureLoss(nn.Module):
    """
    SAR structure consistency loss using edge detection.
    
    This loss ensures that edges detected in SAR imagery are preserved
    in the generated optical imagery. Uses Canny-like gradient detection.
    
    Args:
        sar_threshold_low: Low threshold for SAR edge detection
        sar_threshold_high: High threshold for SAR edge detection
        optical_threshold_low: Low threshold for optical edge detection
        optical_threshold_high: High threshold for optical edge detection
        sobel_kernel_size: Size of Sobel kernel (default: 3)
    """
    
    def __init__(
        self,
        sar_threshold_low: float = 0.1,
        sar_threshold_high: float = 0.3,
        optical_threshold_low: float = 0.1,
        optical_threshold_high: float = 0.3,
        sobel_kernel_size: int = 3
    ):
        super().__init__()
        self.sar_threshold_low = sar_threshold_low
        self.sar_threshold_high = sar_threshold_high
        self.optical_threshold_low = optical_threshold_low
        self.optical_threshold_high = optical_threshold_high
        
        # Sobel kernels for gradient computation
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    
    def _compute_gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude using Sobel operator."""
        # If multi-channel, convert to single channel (average)
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        
        # Pad input
        x_padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
        
        # Compute gradients
        grad_x = F.conv2d(x_padded, self.sobel_x)
        grad_y = F.conv2d(x_padded, self.sobel_y)
        
        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        return grad_mag
    
    def _rgb_to_luminance(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to Y-channel (luminance) using ITU-R BT.601 standard."""
        # Assumes rgb is (B, C, H, W) with C >= 3 (RGB in first 3 channels)
        if rgb.shape[1] < 3:
            return rgb.mean(dim=1, keepdim=True)
        
        # Y = 0.299*R + 0.587*G + 0.114*B
        weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device).view(1, 3, 1, 1)
        y_channel = (rgb[:, :3, :, :] * weights).sum(dim=1, keepdim=True)
        
        return y_channel
    
    def _apply_threshold(self, grad_mag: torch.Tensor, low: float, high: float) -> torch.Tensor:
        """Apply hysteresis thresholding (simplified Canny)."""
        # Strong edges
        strong = (grad_mag > high).float()
        
        # Weak edges
        weak = ((grad_mag >= low) & (grad_mag <= high)).float()
        
        # Combine (simplified - no connectivity analysis)
        edges = strong + 0.5 * weak
        
        return edges
    
    def forward(self, sar: torch.Tensor, opt_rgb: torch.Tensor) -> torch.Tensor:
        """
        Compute SAR structure loss.
        
        Args:
            sar: SAR images (B, C, H, W) - uses all channels averaged
            opt_rgb: Optical RGB images (B, C, H, W) - converts to Y-channel
            
        Returns:
            Structure loss value
        """
        # Compute gradients
        sar_grad = self._compute_gradient_magnitude(sar)
        opt_y = self._rgb_to_luminance(opt_rgb)
        opt_grad = self._compute_gradient_magnitude(opt_y)
        
        # Apply edge detection
        sar_edges = self._apply_threshold(sar_grad, self.sar_threshold_low, self.sar_threshold_high)
        opt_edges = self._apply_threshold(opt_grad, self.optical_threshold_low, self.optical_threshold_high)
        
        # Compute L1 loss between edge maps
        structure_loss = F.l1_loss(opt_edges, sar_edges)
        
        return structure_loss


class SpectralIndexLoss(nn.Module):
    """
    Spectral index loss for vegetation and land cover accuracy.
    
    Computes RMSE of spectral indices (NDVI, EVI) and optionally spectral angle.
    
    Args:
        indices: List of indices to compute ('ndvi', 'evi')
        use_spectral_angle: Whether to include spectral angle mapper loss
        weights: Weights for each index (if None, uses equal weights)
    """
    
    def __init__(
        self,
        indices: Tuple[str, ...] = ('ndvi', 'evi'),
        use_spectral_angle: bool = True,
        weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.indices = indices
        self.use_spectral_angle = use_spectral_angle
        
        # Default weights
        if weights is None:
            weights = {idx: 1.0 for idx in indices}
            if use_spectral_angle:
                weights['spectral_angle'] = 1.0
        
        self.weights = weights
        
        # Validate indices
        valid_indices = {'ndvi', 'evi'}
        for idx in indices:
            if idx not in valid_indices:
                raise ValueError(f"Unknown index: {idx}. Valid indices: {valid_indices}")
    
    def _compute_ndvi(self, rgbnir: torch.Tensor) -> torch.Tensor:
        """
        Compute NDVI (Normalized Difference Vegetation Index).
        
        NDVI = (NIR - Red) / (NIR + Red)
        
        Assumes channel order: R, G, B, NIR
        """
        red = rgbnir[:, 0:1, :, :]
        nir = rgbnir[:, 3:4, :, :]
        
        ndvi = (nir - red) / (nir + red + 1e-8)
        
        return ndvi
    
    def _compute_evi(self, rgbnir: torch.Tensor) -> torch.Tensor:
        """
        Compute EVI (Enhanced Vegetation Index).
        
        EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        
        Assumes channel order: R, G, B, NIR
        """
        red = rgbnir[:, 0:1, :, :]
        blue = rgbnir[:, 2:3, :, :]
        nir = rgbnir[:, 3:4, :, :]
        
        evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + 1e-8)
        
        return evi
    
    def _compute_spectral_angle(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Spectral Angle Mapper (SAM) distance.
        
        SAM measures the angle between spectral signatures.
        """
        # Flatten spatial dimensions
        pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)  # (B, C, H*W)
        target_flat = target.reshape(target.shape[0], target.shape[1], -1)  # (B, C, H*W)
        
        # Normalize
        pred_norm = F.normalize(pred_flat, p=2, dim=1)
        target_norm = F.normalize(target_flat, p=2, dim=1)
        
        # Compute cosine similarity
        cos_sim = (pred_norm * target_norm).sum(dim=1)  # (B, H*W)
        
        # Clamp to avoid numerical issues with arccos
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        
        # Compute angle
        angle = torch.acos(cos_sim)
        
        return angle.mean()
    
    def forward(self, opt_rgbnir: torch.Tensor, ref_rgbnir: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral index loss.
        
        Args:
            opt_rgbnir: Predicted optical images (B, 4, H, W) with R, G, B, NIR
            ref_rgbnir: Reference optical images (B, 4, H, W) with R, G, B, NIR
            
        Returns:
            Spectral index loss value
        """
        if opt_rgbnir.shape[1] < 4 or ref_rgbnir.shape[1] < 4:
            raise ValueError(
                f"Spectral index loss requires at least 4 channels (R, G, B, NIR). "
                f"Got {opt_rgbnir.shape[1]} and {ref_rgbnir.shape[1]} channels."
            )
        
        total_loss = 0.0
        
        # Compute index losses
        for idx in self.indices:
            if idx == 'ndvi':
                pred_idx = self._compute_ndvi(opt_rgbnir)
                ref_idx = self._compute_ndvi(ref_rgbnir)
            elif idx == 'evi':
                pred_idx = self._compute_evi(opt_rgbnir)
                ref_idx = self._compute_evi(ref_rgbnir)
            
            # RMSE loss
            loss = torch.sqrt(F.mse_loss(pred_idx, ref_idx) + 1e-8)
            total_loss += self.weights.get(idx, 1.0) * loss
        
        # Spectral angle loss
        if self.use_spectral_angle:
            sam_loss = self._compute_spectral_angle(opt_rgbnir, ref_rgbnir)
            total_loss += self.weights.get('spectral_angle', 1.0) * sam_loss
        
        return total_loss


class ColorConsistencyLoss(nn.Module):
    """
    Color consistency loss for preserving spectral band relationships.
    
    This loss ensures that RGB ratios and NIR/Red relationships are maintained,
    which is critical for generating physically plausible optical imagery.
    Used in Stage 1 (GAC architecture) instead of SAR-structure loss.
    
    Computes L1 distance between band ratios:
    - Red/Green ratio
    - Red/Blue ratio
    - NIR/Red ratio (vegetation sensitivity)
    
    Args:
        eps: Small epsilon to avoid division by zero
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute color consistency loss.
        
        Args:
            pred: Predicted optical images (B, 4, H, W) with channels [B, G, R, NIR]
            target: Target optical images (B, 4, H, W) with channels [B, G, R, NIR]
            
        Returns:
            Color consistency loss value
        """
        if pred.shape[1] < 4 or target.shape[1] < 4:
            warnings.warn(
                f"ColorConsistencyLoss expects 4 channels (B, G, R, NIR), "
                f"got {pred.shape[1]} channels. Using first 3 channels only."
            )
            # Use only RGB if NIR not available
            pred_blue = pred[:, 0:1, :, :]
            pred_green = pred[:, 1:2, :, :]
            pred_red = pred[:, 2:3, :, :]
            
            target_blue = target[:, 0:1, :, :]
            target_green = target[:, 1:2, :, :]
            target_red = target[:, 2:3, :, :]
            
            # Red/Green ratio
            pred_rg = pred_red / (pred_green + self.eps)
            target_rg = target_red / (target_green + self.eps)
            
            # Red/Blue ratio
            pred_rb = pred_red / (pred_blue + self.eps)
            target_rb = target_red / (target_blue + self.eps)
            
            # Compute L1 loss on ratios
            loss = (F.l1_loss(pred_rg, target_rg) + F.l1_loss(pred_rb, target_rb)) / 2.0
            
            return loss
        
        # Extract bands (B, G, R, NIR)
        pred_blue = pred[:, 0:1, :, :]
        pred_green = pred[:, 1:2, :, :]
        pred_red = pred[:, 2:3, :, :]
        pred_nir = pred[:, 3:4, :, :]
        
        target_blue = target[:, 0:1, :, :]
        target_green = target[:, 1:2, :, :]
        target_red = target[:, 2:3, :, :]
        target_nir = target[:, 3:4, :, :]
        
        # Compute band ratios
        # Red/Green ratio
        pred_rg = pred_red / (pred_green + self.eps)
        target_rg = target_red / (target_green + self.eps)
        
        # Red/Blue ratio
        pred_rb = pred_red / (pred_blue + self.eps)
        target_rb = target_red / (target_blue + self.eps)
        
        # NIR/Red ratio (vegetation sensitivity)
        pred_nr = pred_nir / (pred_red + self.eps)
        target_nr = target_nir / (target_red + self.eps)
        
        # Compute L1 loss on ratios
        loss = (
            F.l1_loss(pred_rg, target_rg) +
            F.l1_loss(pred_rb, target_rb) +
            F.l1_loss(pred_nr, target_nr)
        ) / 3.0
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for SAR-to-optical translation.
    
    Combines multiple loss functions with configurable weights.
    
    GAC Architecture Notes:
    - Stage 1: Use l1 + ms_ssim + lpips + color_consistency (NO sar_structure)
    - Stage 3: Use sar_structure (in separate Stage3Loss)
    
    Args:
        l1_weight: Weight for L1 loss
        ms_ssim_weight: Weight for MS-SSIM loss
        lpips_weight: Weight for LPIPS loss (requires lpips package)
        color_consistency_weight: Weight for color consistency loss (GAC Stage 1)
        sar_structure_weight: Weight for SAR structure loss (default: 0.0, use Stage 3 only)
        spectral_index_weight: Weight for spectral index loss
        **kwargs: Additional arguments for individual loss functions
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        ms_ssim_weight: float = 0.0,
        lpips_weight: float = 0.0,
        color_consistency_weight: float = 0.0,
        sar_structure_weight: float = 0.0,
        spectral_index_weight: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.ms_ssim_weight = ms_ssim_weight
        self.lpips_weight = lpips_weight
        self.color_consistency_weight = color_consistency_weight
        self.sar_structure_weight = sar_structure_weight
        self.spectral_index_weight = spectral_index_weight
        
        # Initialize loss functions
        self.l1_loss = L1Loss() if l1_weight > 0 else None
        self.ms_ssim_loss = MSSSIMLoss(**kwargs.get('ms_ssim_kwargs', {})) if ms_ssim_weight > 0 else None
        self.lpips_loss = LPIPSLoss(**kwargs.get('lpips_kwargs', {})) if lpips_weight > 0 and HAS_LPIPS else None
        self.color_consistency_loss = ColorConsistencyLoss(**kwargs.get('color_consistency_kwargs', {})) if color_consistency_weight > 0 else None
        self.sar_structure_loss = SARStructureLoss(**kwargs.get('sar_structure_kwargs', {})) if sar_structure_weight > 0 else None
        self.spectral_index_loss = SpectralIndexLoss(**kwargs.get('spectral_index_kwargs', {})) if spectral_index_weight > 0 else None
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sar: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted optical images (B, C, H, W)
            target: Target optical images (B, C, H, W)
            sar: SAR images (B, C, H, W), required if sar_structure_weight > 0
            
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses
        """
        total_loss = 0.0
        loss_dict = {}
        
        # L1 loss
        if self.l1_loss is not None:
            l1 = self.l1_loss(pred, target)
            total_loss += self.l1_weight * l1
            loss_dict['l1'] = l1.item()
        
        # MS-SSIM loss
        if self.ms_ssim_loss is not None:
            ms_ssim = self.ms_ssim_loss(pred, target)
            total_loss += self.ms_ssim_weight * ms_ssim
            loss_dict['ms_ssim'] = ms_ssim.item()
        
        # LPIPS loss
        if self.lpips_loss is not None:
            lpips = self.lpips_loss(pred, target)
            total_loss += self.lpips_weight * lpips
            loss_dict['lpips'] = lpips.item()
        
        # Color consistency loss (GAC Stage 1)
        if self.color_consistency_loss is not None:
            color_consist = self.color_consistency_loss(pred, target)
            total_loss += self.color_consistency_weight * color_consist
            loss_dict['color_consistency'] = color_consist.item()
        
        # SAR structure loss (NOTE: Use 0.0 weight for GAC Stage 1, move to Stage 3)
        if self.sar_structure_loss is not None:
            if sar is None:
                raise ValueError("SAR input is required when sar_structure_weight > 0")
            sar_struct = self.sar_structure_loss(sar, pred)
            total_loss += self.sar_structure_weight * sar_struct
            loss_dict['sar_structure'] = sar_struct.item()
        
        # Spectral index loss
        if self.spectral_index_loss is not None:
            spectral = self.spectral_index_loss(pred, target)
            total_loss += self.spectral_index_weight * spectral
            loss_dict['spectral_index'] = spectral.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
