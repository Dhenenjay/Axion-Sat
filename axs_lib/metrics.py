"""
Evaluation metrics for SAR-to-optical image translation.

This module provides various metrics for assessing reconstruction quality:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- SAR-Edge Agreement Score (cosine similarity and F1 of edge maps)
- GAC-Score (Generative Accuracy Composite Score) - weighted combination

All metrics are designed to work with PyTorch tensors and support batch processing.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union
import warnings
import math

try:
    from torchmetrics.image import (
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure
    )
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False
    warnings.warn("torchmetrics not available, using fallback implementations")

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    warnings.warn("lpips package not available, LPIPS metric will be disabled")


class PSNRMetric:
    """
    Peak Signal-to-Noise Ratio metric.
    
    PSNR measures the ratio between the maximum possible power of a signal
    and the power of corrupting noise. Higher values indicate better quality.
    
    Args:
        data_range: The dynamic range of the images (default: 1.0)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, data_range: float = 1.0, reduction: str = 'mean'):
        self.data_range = data_range
        self.reduction = reduction
        
        if HAS_TORCHMETRICS:
            # torchmetrics PSNR uses 'elementwise_mean' instead of 'mean'
            tm_reduction = 'elementwise_mean' if reduction == 'mean' else reduction
            self.psnr = PeakSignalNoiseRatio(data_range=data_range, reduction=tm_reduction)
        else:
            self.psnr = None
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute PSNR between prediction and target.
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
            
        Returns:
            PSNR value in dB (higher is better)
        """
        if self.psnr is not None:
            return self.psnr(pred, target)
        else:
            return self._compute_psnr(pred, target)
    
    def _compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Fallback PSNR implementation."""
        mse = F.mse_loss(pred, target, reduction='none')
        
        if self.reduction == 'none':
            # Compute per-sample PSNR
            mse = mse.view(mse.shape[0], -1).mean(dim=1)
            psnr = 10 * torch.log10((self.data_range ** 2) / (mse + 1e-8))
        else:
            mse = mse.mean()
            psnr = 10 * torch.log10((self.data_range ** 2) / (mse + 1e-8))
            
            if self.reduction == 'sum':
                psnr = psnr * pred.shape[0]
        
        return psnr


class SSIMMetric:
    """
    Structural Similarity Index Measure.
    
    SSIM measures the perceptual difference between two images, considering
    luminance, contrast, and structure. Values range from -1 to 1, with
    1 indicating perfect similarity.
    
    Args:
        data_range: The dynamic range of the images (default: 1.0)
        kernel_size: Size of the gaussian kernel (default: 11)
        k1: Algorithm parameter (default: 0.01)
        k2: Algorithm parameter (default: 0.03)
    """
    
    def __init__(
        self,
        data_range: float = 1.0,
        kernel_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03
    ):
        self.data_range = data_range
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        
        if HAS_TORCHMETRICS:
            self.ssim = StructuralSimilarityIndexMeasure(
                data_range=data_range,
                kernel_size=kernel_size,
                k1=k1,
                k2=k2
            )
        else:
            self.ssim = None
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM between prediction and target.
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
            
        Returns:
            SSIM value (higher is better, range: -1 to 1)
        """
        if self.ssim is not None:
            return self.ssim(pred, target)
        else:
            return self._compute_ssim(pred, target)
    
    def _compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Fallback SSIM implementation."""
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


class LPIPSMetric:
    """
    Learned Perceptual Image Patch Similarity metric.
    
    LPIPS uses deep features from pretrained networks to measure perceptual
    similarity. Lower values indicate better similarity.
    
    Args:
        net: Network to use ('alex', 'vgg', 'squeeze')
        use_gpu: Whether to use GPU (automatically detected if None)
    """
    
    def __init__(self, net: str = 'alex', use_gpu: Optional[bool] = None):
        if not HAS_LPIPS:
            raise ImportError(
                "lpips package is required for LPIPS metric. "
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
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS distance between prediction and target.
        
        Args:
            pred: Predicted images (B, C, H, W), values in [-1, 1] or [0, 1]
            target: Target images (B, C, H, W), values in [-1, 1] or [0, 1]
            
        Returns:
            LPIPS distance (lower is better)
        """
        # LPIPS expects 3-channel RGB images
        if pred.shape[1] != 3:
            # If more than 3 channels, take first 3 (assumes RGB ordering)
            pred = pred[:, :3, :, :]
            target = target[:, :3, :, :]
        
        return self.loss_fn(pred, target).mean()


class SAREdgeAgreementScore:
    """
    SAR-Edge Agreement Score measuring structural consistency.
    
    This metric computes the agreement between edge maps extracted from
    SAR imagery and generated optical imagery. It provides both cosine
    similarity and F1 score of the edge maps.
    
    Args:
        sar_threshold_low: Low threshold for SAR edge detection
        sar_threshold_high: High threshold for SAR edge detection
        optical_threshold_low: Low threshold for optical edge detection
        optical_threshold_high: High threshold for optical edge detection
        beta: Beta parameter for F-beta score (default: 1.0 for F1)
    """
    
    def __init__(
        self,
        sar_threshold_low: float = 0.1,
        sar_threshold_high: float = 0.3,
        optical_threshold_low: float = 0.1,
        optical_threshold_high: float = 0.3,
        beta: float = 1.0
    ):
        self.sar_threshold_low = sar_threshold_low
        self.sar_threshold_high = sar_threshold_high
        self.optical_threshold_low = optical_threshold_low
        self.optical_threshold_high = optical_threshold_high
        self.beta = beta
        
        # Sobel kernels for gradient computation
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    def _compute_gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude using Sobel operator."""
        # If multi-channel, convert to single channel (average)
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        
        # Move sobel kernels to same device
        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)
        
        # Pad input
        x_padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
        
        # Compute gradients
        grad_x = F.conv2d(x_padded, sobel_x)
        grad_y = F.conv2d(x_padded, sobel_y)
        
        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        return grad_mag
    
    def _rgb_to_luminance(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to Y-channel (luminance)."""
        if rgb.shape[1] < 3:
            return rgb.mean(dim=1, keepdim=True)
        
        # Y = 0.299*R + 0.587*G + 0.114*B
        weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device).view(1, 3, 1, 1)
        y_channel = (rgb[:, :3, :, :] * weights).sum(dim=1, keepdim=True)
        
        return y_channel
    
    def _apply_threshold(self, grad_mag: torch.Tensor, low: float, high: float) -> torch.Tensor:
        """Apply hysteresis thresholding."""
        # Strong edges
        strong = (grad_mag > high).float()
        
        # Weak edges
        weak = ((grad_mag >= low) & (grad_mag <= high)).float()
        
        # Combine
        edges = strong + 0.5 * weak
        
        return edges
    
    def _cosine_similarity(self, sar_edges: torch.Tensor, opt_edges: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between edge maps."""
        # Flatten
        sar_flat = sar_edges.reshape(sar_edges.shape[0], -1)
        opt_flat = opt_edges.reshape(opt_edges.shape[0], -1)
        
        # Normalize
        sar_norm = F.normalize(sar_flat, p=2, dim=1)
        opt_norm = F.normalize(opt_flat, p=2, dim=1)
        
        # Cosine similarity
        cos_sim = (sar_norm * opt_norm).sum(dim=1).mean()
        
        return cos_sim
    
    def _f1_score(self, sar_edges: torch.Tensor, opt_edges: torch.Tensor) -> torch.Tensor:
        """Compute F-beta score between binary edge maps."""
        # Binarize edges (threshold at 0.5)
        sar_binary = (sar_edges > 0.5).float()
        opt_binary = (opt_edges > 0.5).float()
        
        # Compute TP, FP, FN
        tp = (sar_binary * opt_binary).sum()
        fp = ((1 - sar_binary) * opt_binary).sum()
        fn = (sar_binary * (1 - opt_binary)).sum()
        
        # Compute precision and recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        # F-beta score
        beta_sq = self.beta ** 2
        f_score = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall + 1e-8)
        
        return f_score
    
    def __call__(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute SAR-Edge Agreement Score.
        
        Args:
            sar: SAR images (B, C, H, W)
            optical: Optical images (B, C, H, W)
            return_components: If True, return individual components
            
        Returns:
            If return_components=False: Average of cosine similarity and F1 score
            If return_components=True: Tuple of (score, components_dict)
        """
        # Compute gradients
        sar_grad = self._compute_gradient_magnitude(sar)
        opt_y = self._rgb_to_luminance(optical)
        opt_grad = self._compute_gradient_magnitude(opt_y)
        
        # Apply edge detection
        sar_edges = self._apply_threshold(sar_grad, self.sar_threshold_low, self.sar_threshold_high)
        opt_edges = self._apply_threshold(opt_grad, self.optical_threshold_low, self.optical_threshold_high)
        
        # Compute metrics
        cos_sim = self._cosine_similarity(sar_edges, opt_edges)
        f1 = self._f1_score(sar_edges, opt_edges)
        
        # Average score
        score = (cos_sim + f1) / 2.0
        
        if return_components:
            components = {
                'cosine_similarity': cos_sim,
                'f1_score': f1,
                'sar_edges': sar_edges,
                'optical_edges': opt_edges
            }
            return score, components
        else:
            return score


class GACScore:
    """
    Generative Accuracy Composite (GAC) Score.
    
    A weighted combination of multiple metrics for holistic evaluation:
    - PSNR: Pixel-level reconstruction accuracy
    - SSIM: Structural similarity
    - LPIPS: Perceptual similarity (optional)
    - SAR-Edge: Structural consistency with SAR
    
    The score is normalized to [0, 1] range where higher is better.
    
    Args:
        weights: Dictionary of metric weights
        psnr_range: Expected PSNR range for normalization (min, max)
        include_lpips: Whether to include LPIPS in the score
        lpips_net: Network for LPIPS ('alex', 'vgg', 'squeeze')
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        psnr_range: Tuple[float, float] = (20.0, 40.0),
        include_lpips: bool = False,
        lpips_net: str = 'alex',
        device: Optional[torch.device] = None
    ):
        # Default weights
        if weights is None:
            if include_lpips:
                weights = {
                    'psnr': 0.25,
                    'ssim': 0.25,
                    'lpips': 0.25,
                    'sar_edge': 0.25
                }
            else:
                weights = {
                    'psnr': 0.33,
                    'ssim': 0.33,
                    'sar_edge': 0.34
                }
        
        self.weights = weights
        self.psnr_range = psnr_range
        self.include_lpips = include_lpips
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Validate weights sum to 1.0
        weight_sum = sum(weights.values())
        if not math.isclose(weight_sum, 1.0, abs_tol=1e-6):
            warnings.warn(
                f"Weights sum to {weight_sum}, not 1.0. "
                "Normalizing weights automatically."
            )
            self.weights = {k: v / weight_sum for k, v in weights.items()}
        
        # Initialize metrics and move to device
        self.psnr_metric = PSNRMetric()
        self.ssim_metric = SSIMMetric()
        self.sar_edge_metric = SAREdgeAgreementScore()
        
        # Move torchmetrics to device if they exist
        if hasattr(self.psnr_metric, 'psnr') and self.psnr_metric.psnr is not None:
            self.psnr_metric.psnr = self.psnr_metric.psnr.to(self.device)
        if hasattr(self.ssim_metric, 'ssim') and self.ssim_metric.ssim is not None:
            self.ssim_metric.ssim = self.ssim_metric.ssim.to(self.device)
        
        if include_lpips:
            if HAS_LPIPS:
                self.lpips_metric = LPIPSMetric(net=lpips_net)
            else:
                warnings.warn("LPIPS not available, excluding from GAC score")
                self.include_lpips = False
                # Re-normalize weights
                lpips_weight = self.weights.pop('lpips', 0.0)
                remaining = 1.0 - lpips_weight
                if remaining > 0:
                    self.weights = {k: v / remaining for k, v in self.weights.items()}
        
        # Accumulation for batch processing
        self.reset()
    
    def reset(self):
        """Reset accumulation buffers."""
        self.total_score = 0.0
        self.total_samples = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, sar: torch.Tensor):
        """Update with a batch of predictions.
        
        Args:
            pred: Predicted optical images (B, C, H, W)
            target: Target optical images (B, C, H, W)
            sar: SAR images (B, C, H, W)
        """
        # Ensure inputs are on the correct device
        pred = pred.to(self.device)
        target = target.to(self.device)
        sar = sar.to(self.device)
        
        # Compute GAC score for this batch
        gac_score = self(pred, target, sar, return_components=False)
        
        # Accumulate (weighted by batch size)
        batch_size = pred.shape[0]
        if isinstance(gac_score, torch.Tensor):
            gac_score = gac_score.item()
        
        self.total_score += gac_score * batch_size
        self.total_samples += batch_size
    
    def compute(self) -> Optional[float]:
        """Compute average GAC score across all batches.
        
        Returns:
            Average GAC score, or None if no samples were processed
        """
        if self.total_samples == 0:
            return None
        return self.total_score / self.total_samples
    
    def _normalize_psnr(self, psnr: torch.Tensor) -> torch.Tensor:
        """Normalize PSNR to [0, 1] range."""
        min_psnr, max_psnr = self.psnr_range
        normalized = (psnr - min_psnr) / (max_psnr - min_psnr)
        return torch.clamp(normalized, 0.0, 1.0)
    
    def _normalize_ssim(self, ssim: torch.Tensor) -> torch.Tensor:
        """Normalize SSIM to [0, 1] range."""
        # SSIM is already in [-1, 1], map to [0, 1]
        return (ssim + 1.0) / 2.0
    
    def _normalize_lpips(self, lpips: torch.Tensor) -> torch.Tensor:
        """Normalize LPIPS to [0, 1] range (invert since lower is better)."""
        # LPIPS typically ranges from 0 to 1, invert it
        return 1.0 - torch.clamp(lpips, 0.0, 1.0)
    
    def _normalize_sar_edge(self, sar_edge: torch.Tensor) -> torch.Tensor:
        """Normalize SAR-Edge score to [0, 1] range."""
        # Already in [0, 1] range from cosine and F1 average
        return torch.clamp(sar_edge, 0.0, 1.0)
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sar: torch.Tensor,
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Compute GAC Score.
        
        Args:
            pred: Predicted optical images (B, C, H, W)
            target: Target optical images (B, C, H, W)
            sar: SAR images (B, C, H, W)
            return_components: If True, return individual metric scores
            
        Returns:
            If return_components=False: GAC score (0 to 1, higher is better)
            If return_components=True: Tuple of (gac_score, components_dict)
        """
        components = {}
        gac_score = 0.0
        
        # Compute PSNR
        if 'psnr' in self.weights:
            psnr = self.psnr_metric(pred, target)
            psnr_norm = self._normalize_psnr(psnr)
            gac_score += self.weights['psnr'] * psnr_norm
            components['psnr'] = psnr.item()
            components['psnr_normalized'] = psnr_norm.item()
        
        # Compute SSIM
        if 'ssim' in self.weights:
            ssim = self.ssim_metric(pred, target)
            ssim_norm = self._normalize_ssim(ssim)
            gac_score += self.weights['ssim'] * ssim_norm
            components['ssim'] = ssim.item()
            components['ssim_normalized'] = ssim_norm.item()
        
        # Compute LPIPS
        if self.include_lpips and 'lpips' in self.weights:
            lpips_val = self.lpips_metric(pred, target)
            lpips_norm = self._normalize_lpips(lpips_val)
            gac_score += self.weights['lpips'] * lpips_norm
            components['lpips'] = lpips_val.item()
            components['lpips_normalized'] = lpips_norm.item()
        
        # Compute SAR-Edge Agreement
        if 'sar_edge' in self.weights:
            sar_edge = self.sar_edge_metric(sar, pred)
            sar_edge_norm = self._normalize_sar_edge(sar_edge)
            gac_score += self.weights['sar_edge'] * sar_edge_norm
            components['sar_edge'] = sar_edge.item()
            components['sar_edge_normalized'] = sar_edge_norm.item()
        
        components['gac_score'] = gac_score.item()
        
        if return_components:
            return gac_score, components
        else:
            return gac_score


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    sar: Optional[torch.Tensor] = None,
    include_lpips: bool = False,
    include_gac: bool = True
) -> Dict[str, float]:
    """
    Compute all available metrics at once.
    
    Args:
        pred: Predicted optical images (B, C, H, W)
        target: Target optical images (B, C, H, W)
        sar: SAR images (B, C, H, W), required for SAR-edge and GAC
        include_lpips: Whether to compute LPIPS
        include_gac: Whether to compute GAC score
        
    Returns:
        Dictionary of all computed metrics
    """
    results = {}
    
    # Basic metrics
    psnr_metric = PSNRMetric()
    ssim_metric = SSIMMetric()
    
    results['psnr'] = psnr_metric(pred, target).item()
    results['ssim'] = ssim_metric(pred, target).item()
    
    # LPIPS
    if include_lpips and HAS_LPIPS:
        lpips_metric = LPIPSMetric()
        results['lpips'] = lpips_metric(pred, target).item()
    
    # SAR-Edge Agreement
    if sar is not None:
        sar_edge_metric = SAREdgeAgreementScore()
        score, components = sar_edge_metric(sar, pred, return_components=True)
        results['sar_edge'] = score.item()
        results['sar_edge_cosine'] = components['cosine_similarity'].item()
        results['sar_edge_f1'] = components['f1_score'].item()
    
    # GAC Score
    if include_gac and sar is not None:
        gac_metric = GACScore(include_lpips=include_lpips)
        gac_score, gac_components = gac_metric(pred, target, sar, return_components=True)
        results['gac_score'] = gac_score.item()
        results.update({f'gac_{k}': v for k, v in gac_components.items()})
    
    return results


# Convenience wrapper for batch evaluation
class MetricsEvaluator:
    """
    Convenience class for evaluating multiple metrics with consistent configuration.
    
    Args:
        include_lpips: Whether to include LPIPS
        include_gac: Whether to include GAC score
        gac_weights: Custom weights for GAC score
        psnr_range: PSNR normalization range for GAC
    """
    
    def __init__(
        self,
        include_lpips: bool = False,
        include_gac: bool = True,
        gac_weights: Optional[Dict[str, float]] = None,
        psnr_range: Tuple[float, float] = (20.0, 40.0)
    ):
        self.include_lpips = include_lpips
        self.include_gac = include_gac
        
        # Initialize metrics
        self.psnr = PSNRMetric()
        self.ssim = SSIMMetric()
        self.sar_edge = SAREdgeAgreementScore()
        
        if include_lpips and HAS_LPIPS:
            self.lpips = LPIPSMetric()
        else:
            self.lpips = None
        
        if include_gac:
            self.gac = GACScore(
                weights=gac_weights,
                psnr_range=psnr_range,
                include_lpips=include_lpips
            )
        else:
            self.gac = None
    
    def evaluate(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sar: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Evaluate all configured metrics.
        
        Args:
            pred: Predicted optical images (B, C, H, W)
            target: Target optical images (B, C, H, W)
            sar: SAR images (B, C, H, W)
            
        Returns:
            Dictionary of metric values
        """
        return compute_all_metrics(
            pred, target, sar,
            include_lpips=self.include_lpips,
            include_gac=self.include_gac
        )
