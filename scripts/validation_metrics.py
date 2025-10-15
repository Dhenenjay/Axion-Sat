"""
Validation metrics for Stage 1 SAR-to-Optical translation.

Provides comprehensive metrics for evaluating synthetic optical generation:
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- PSNR (Peak Signal-to-Noise Ratio)
- MAE (Mean Absolute Error)
- SAR-Edge Agreement (structural consistency)

Also includes CSV logger for tracking validation metrics over training.
"""

import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False
    warnings.warn("torchmetrics not available, SSIM will use fallback implementation")

try:
    from axs_lib.losses import LPIPSLoss, SARStructureLoss
    HAS_AXS_LOSSES = True
except ImportError:
    HAS_AXS_LOSSES = False
    warnings.warn("axs_lib.losses not available, some metrics will be disabled")


class ValidationMetrics:
    """Compute validation metrics for SAR-to-optical translation."""
    
    def __init__(self, device: torch.device):
        self.device = device
        
        # SSIM metric
        if HAS_TORCHMETRICS:
            self.ssim_metric = StructuralSimilarityIndexMeasure(
                data_range=1.0
            ).to(device)
        else:
            self.ssim_metric = None
        
        # LPIPS metric (perceptual similarity)
        if HAS_AXS_LOSSES:
            try:
                self.lpips_metric = LPIPSLoss(net='alex', use_gpu=(device.type == 'cuda'))
            except:
                self.lpips_metric = None
                warnings.warn("LPIPS not available for validation metrics")
        else:
            self.lpips_metric = None
        
        # SAR structure metric
        if HAS_AXS_LOSSES:
            self.sar_structure_metric = SARStructureLoss()
        else:
            self.sar_structure_metric = None
    
    def compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute SSIM between prediction and target."""
        if self.ssim_metric is not None:
            # torchmetrics SSIM
            ssim_score = self.ssim_metric(pred, target)
            return ssim_score.item()
        else:
            # Fallback: simple SSIM implementation
            return self._compute_ssim_fallback(pred, target)
    
    def _compute_ssim_fallback(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Fallback SSIM implementation."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Convert to grayscale if multichannel
        if pred.shape[1] > 1:
            pred = pred.mean(dim=1, keepdim=True)
            target = target.mean(dim=1, keepdim=True)
        
        mu_pred = F.avg_pool2d(pred, 11, stride=1, padding=5)
        mu_target = F.avg_pool2d(target, 11, stride=1, padding=5)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.avg_pool2d(pred ** 2, 11, stride=1, padding=5) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target ** 2, 11, stride=1, padding=5) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred * target, 11, stride=1, padding=5) - mu_pred_target
        
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        return ssim_map.mean().item()
    
    def compute_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute LPIPS (perceptual similarity) between prediction and target."""
        if self.lpips_metric is None:
            return 0.0
        
        # LPIPS expects 3-channel RGB
        if pred.shape[1] == 4:
            # Take first 3 channels (B, G, R)
            pred = pred[:, :3, :, :]
            target = target[:, :3, :, :]
        elif pred.shape[1] == 1:
            # Replicate grayscale to 3 channels
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        lpips_score = self.lpips_metric(pred, target)
        return lpips_score.item()
    
    def compute_sar_edge_agreement(self, sar: torch.Tensor, optical: torch.Tensor) -> float:
        """Compute SAR-optical edge agreement."""
        if self.sar_structure_metric is None:
            return 0.0
        
        edge_loss = self.sar_structure_metric(sar, optical)
        # Convert loss to agreement score (lower loss = higher agreement)
        agreement = 1.0 - torch.clamp(edge_loss, 0.0, 1.0).item()
        return agreement
    
    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Peak Signal-to-Noise Ratio."""
        mse = F.mse_loss(pred, target)
        if mse < 1e-10:
            return 100.0
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()
    
    def compute_mae(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Mean Absolute Error."""
        mae = F.l1_loss(pred, target)
        return mae.item()
    
    def compute_all(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sar: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute all validation metrics.
        
        Args:
            pred: Predicted optical imagery (B, C, H, W)
            target: Target optical imagery (B, C, H, W)
            sar: SAR input (B, 2, H, W), optional for edge agreement
            
        Returns:
            Dictionary of metrics
        """
        # Normalize to [0, 1] for metrics
        pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        target_norm = (target - target.min()) / (target.max() - target.min() + 1e-8)
        
        metrics = {}
        
        # SSIM
        metrics['ssim'] = self.compute_ssim(pred_norm, target_norm)
        
        # LPIPS
        metrics['lpips'] = self.compute_lpips(pred_norm, target_norm)
        
        # PSNR
        metrics['psnr'] = self.compute_psnr(pred_norm, target_norm)
        
        # MAE
        metrics['mae'] = self.compute_mae(pred, target)
        
        # SAR-edge agreement
        if sar is not None:
            metrics['sar_edge_agreement'] = self.compute_sar_edge_agreement(sar, pred_norm)
        
        return metrics


class ValidationCSVLogger:
    """Log validation metrics to CSV file."""
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists
        self.file_exists = self.csv_path.exists()
        
        # Define headers
        self.headers = [
            'step', 'timestamp',
            'loss', 'l1', 'ms_ssim', 'lpips_loss', 'sar_structure',
            'ssim', 'ssim_std',
            'lpips', 'lpips_std',
            'psnr', 'psnr_std',
            'mae', 'mae_std',
            'sar_edge_agreement', 'sar_edge_agreement_std'
        ]
    
    def log(self, step: int, metrics: Dict[str, float]):
        """Log validation metrics to CSV."""
        # Open file in append mode
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            
            # Write header if file is new
            if not self.file_exists:
                writer.writeheader()
                self.file_exists = True
            
            # Prepare row
            row = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'loss': metrics.get('loss', 0.0),
                'l1': metrics.get('l1', 0.0),
                'ms_ssim': metrics.get('ms_ssim', 0.0),
                'lpips_loss': metrics.get('lpips', 0.0),
                'sar_structure': metrics.get('sar_structure', 0.0),
                'ssim': metrics.get('ssim', 0.0),
                'ssim_std': metrics.get('ssim_std', 0.0),
                'lpips': metrics.get('lpips', 0.0),
                'lpips_std': metrics.get('lpips_std', 0.0),
                'psnr': metrics.get('psnr', 0.0),
                'psnr_std': metrics.get('psnr_std', 0.0),
                'mae': metrics.get('mae', 0.0),
                'mae_std': metrics.get('mae_std', 0.0),
                'sar_edge_agreement': metrics.get('sar_edge_agreement', 0.0),
                'sar_edge_agreement_std': metrics.get('sar_edge_agreement_std', 0.0),
            }
            
            writer.writerow(row)
    
    def read(self) -> list:
        """Read all metrics from CSV file."""
        if not self.csv_path.exists():
            return []
        
        metrics_list = []
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for key in row:
                    if key not in ['timestamp']:
                        try:
                            row[key] = float(row[key])
                        except (ValueError, TypeError):
                            pass
                metrics_list.append(row)
        
        return metrics_list


def compute_validation_metrics_batch(
    pred_batch: torch.Tensor,
    target_batch: torch.Tensor,
    sar_batch: Optional[torch.Tensor],
    metrics_computer: ValidationMetrics
) -> Dict[str, list]:
    """
    Compute metrics for a batch of samples.
    
    Returns:
        Dictionary with lists of metrics for each sample
    """
    batch_metrics = {
        'ssim': [],
        'lpips': [],
        'psnr': [],
        'mae': [],
        'sar_edge_agreement': []
    }
    
    # Process each sample in batch
    for i in range(pred_batch.shape[0]):
        pred = pred_batch[i:i+1]
        target = target_batch[i:i+1]
        sar = sar_batch[i:i+1] if sar_batch is not None else None
        
        sample_metrics = metrics_computer.compute_all(pred, target, sar)
        
        for key in batch_metrics:
            if key in sample_metrics:
                batch_metrics[key].append(sample_metrics[key])
    
    return batch_metrics


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Validation Metrics Module")
    print("=" * 80)
    
    # Create dummy data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred = torch.randn(2, 4, 256, 256).to(device)
    target = torch.randn(2, 4, 256, 256).to(device)
    sar = torch.randn(2, 2, 256, 256).to(device)
    
    # Compute metrics
    metrics_computer = ValidationMetrics(device)
    metrics = metrics_computer.compute_all(pred, target, sar)
    
    print("\nComputed metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # CSV logging
    csv_logger = ValidationCSVLogger("test_validation.csv")
    csv_logger.log(step=100, metrics=metrics)
    
    print("\n✓ Metrics logged to test_validation.csv")
    print("\n" + "=" * 80)
