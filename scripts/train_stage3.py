"""
Stage 3 Training: TerraMind SAR Grounding with LoRA

This script trains Stage 3 of the Axion-Sat pipeline, which uses TerraMind's
conditional generator to ground Stage 2 optical outputs with SAR features.

Key Features:
    - LoRA fine-tuning on cross-attention layers (parameter-efficient)
    - Mixed precision (AMP fp16) for memory efficiency
    - Batch size = 1 with gradient accumulation = 8 (effective batch = 8)
    - Early stopping based on SAR-agreement plateau
    - Multi-component loss: SAR consistency + Cycle/Identity + LPIPS + Spectral

Training Configuration:
    Model: TerraMind conditional generator (S1 + opt_v2 → opt_v3)
    LoRA: Applied to cross-attention layers only
    Loss: SAR-consistency (edge + texture) + Cycle/Identity + LPIPS + L1
    Precision: Mixed precision (AMP fp16)
    Batch size: 1 per GPU
    Gradient accumulation: 8 steps
    Optimizer: AdamW with cosine scheduling
    Early stopping: SAR-agreement metric with patience=5

Memory Optimizations:
    - LoRA adapters (only cross-attention layers, ~1-2% trainable)
    - Gradient accumulation (effective batch size = 8)
    - AMP fp16 (reduces memory by ~50%)
    - No gradient checkpointing needed with LoRA

Usage:
    python scripts/train_stage3.py \\
        --data-dir tiles/ \\
        --stage2-dir checkpoints/stage2/ \\
        --output-dir checkpoints/stage3/ \\
        --batch-size 1 \\
        --grad-accum 8 \\
        --lr 1e-4 \\
        --epochs 50

    python scripts/train_stage3.py --help

Author: Axion-Sat Project
Version: 1.0.0
"""

import argparse
import os
import sys
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Tuple
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axs_lib.stage3_tm_backbone import build_stage3_backbone_model
from axs_lib.stage3_losses import Stage3Loss
from axs_lib.io import save_checkpoint, load_checkpoint
from axs_lib.reproducibility import set_seed
from axs_lib.dataloader_utils import create_dataloader, get_recommended_num_workers
from axs_lib.metrics import GACScore


# ============================================================================
# Note: LoRA implementation removed - using full fine-tuning with backbone
# ============================================================================

# LoRA classes removed - no longer needed
# Using TerraMind backbone + lightweight decoder instead

class _RemovedLoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for parameter-efficient fine-tuning.
    
    Adds trainable low-rank decomposition to frozen linear layers:
        h = W_0*x + (B*A)*x * scaling
    where W_0 is frozen, and A, B are trainable low-rank matrices.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of decomposition (lower = fewer parameters)
        alpha: Scaling factor (typically 2*rank)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices (low-rank decomposition)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA transformation.
        
        Args:
            x: Input tensor (..., in_features)
            
        Returns:
            LoRA output (..., out_features)
        """
        # Low-rank transformation: x @ A^T @ B^T
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        
        return lora_out * self.scaling


def apply_lora_to_cross_attention(
    module: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    verbose: bool = True
) -> int:
    """
    Apply LoRA adapters to cross-attention layers in TerraMind.
    
    Targets:
        - Cross-attention query, key, value projections
        - Cross-attention output projections
    
    Args:
        module: PyTorch module to modify
        rank: LoRA rank (default: 8)
        alpha: LoRA alpha scaling (default: 16)
        verbose: Print layer names
        
    Returns:
        Number of LoRA layers added
    """
    target_patterns = [
        'cross_attn',   # Cross-attention modules
        'cross_attention',
        'ca_',          # Cross-attention prefix
        'q_proj',       # Query projection
        'k_proj',       # Key projection
        'v_proj',       # Value projection
        'o_proj',       # Output projection
    ]
    
    # Get device of the module
    device = next(module.parameters()).device
    
    count = 0
    
    for name, submodule in module.named_modules():
        # Only target Linear layers
        if not isinstance(submodule, nn.Linear):
            continue
        
        # Check if name matches cross-attention patterns
        is_cross_attn = any(pattern in name.lower() for pattern in target_patterns)
        if not is_cross_attn:
            continue
        
        # Freeze original layer
        for param in submodule.parameters():
            param.requires_grad = False
        
        # Create LoRA adapter
        in_features = submodule.in_features
        out_features = submodule.out_features
        
        lora_layer = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha
        ).to(device)  # Move LoRA to same device as module
        
        # Register LoRA as a submodule (NOT just an attribute)
        # This ensures PyTorch tracks it in the module hierarchy and gradient graph
        submodule.add_module('lora_adapter', lora_layer)
        
        # Monkey-patch forward method
        original_forward = submodule.forward
        
        def make_lora_forward(orig_forward, lora):
            def lora_forward(x):
                # Base output (frozen) - always detach since base is frozen
                base_out = orig_forward(x)
                if base_out.requires_grad:
                    base_out = base_out.detach()
                # LoRA output (trainable) - ensure this creates a gradient graph
                # The LoRA parameters have requires_grad=True, so the matmul will create gradients
                lora_out = lora(x)
                # Sum: gradient only flows through LoRA, not base
                return base_out + lora_out
            return lora_forward
        
        submodule.forward = make_lora_forward(original_forward, lora_layer)
        
        count += 1
        if verbose:
            print(f"  ✓ LoRA added to: {name}")
    
    return count


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Extract all LoRA parameters from a model."""
    lora_params = []
    
    for module in model.modules():
        if hasattr(module, 'lora_adapter'):
            lora_params.extend(module.lora_adapter.parameters())
    
    return lora_params


# ============================================================================
# Dataset for Stage 3
# ============================================================================

class Stage3Dataset(Dataset):
    """
    Dataset for Stage 3 training.
    
    Loads:
        - s1: Sentinel-1 SAR (2 channels: VV, VH)
        - opt_v2: Stage 2 refined optical output (4 channels: B, G, R, NIR)
        - s2_truth: Ground truth Sentinel-2 optical (4 channels) - optional
    
    Expected NPZ tile structure:
        - s1_vv, s1_vh: SAR bands
        - s2_b2, s2_b3, s2_b4, s2_b8: Optical bands (target)
        - opt_v2_b2, opt_v2_b3, opt_v2_b4, opt_v2_b8: Stage 2 output (if pre-computed)
    """
    
    def __init__(
        self,
        data_dir: Path,
        split: str = 'train',
        stage2_dir: Optional[Path] = None,
        max_samples: Optional[int] = None,
        tile_size: int = 120,
        use_augmentation: bool = True
    ):
        """
        Args:
            data_dir: Directory containing NPZ tiles
            split: 'train', 'val', or 'test'
            stage2_dir: Directory containing Stage 2 opt_v2 outputs (if pre-computed)
            max_samples: Maximum number of samples (for debugging)
            tile_size: Expected tile size
            use_augmentation: Apply augmentation (training only)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.stage2_dir = Path(stage2_dir) if stage2_dir else None
        self.tile_size = tile_size
        self.use_augmentation = use_augmentation and (split == 'train')
        
        # Find all NPZ tiles
        self.tile_paths = self._find_tiles()
        
        if max_samples and len(self.tile_paths) > max_samples:
            self.tile_paths = self.tile_paths[:max_samples]
        
        print(f"Found {len(self.tile_paths)} tiles for {split} split")
    
    def _find_tiles(self) -> List[Path]:
        """Find all NPZ tiles for the split."""
        tiles = []
        
        for npz_file in self.data_dir.rglob('*.npz'):
            # Check split from metadata JSON if exists
            json_file = npz_file.with_suffix('.json')
            if json_file.exists():
                try:
                    with open(json_file) as f:
                        meta = json.load(f)
                        if meta.get('split') == self.split:
                            tiles.append(npz_file)
                except:
                    pass
            else:
                # Fallback: use filename pattern
                if self.split in str(npz_file):
                    tiles.append(npz_file)
        
        return sorted(tiles)
    
    def __len__(self) -> int:
        return len(self.tile_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single training sample.
        
        Returns:
            Dict with:
                - s1: SAR input (2, H, W)
                - opt_v2: Stage 2 output (4, H, W)
                - s2_truth: Ground truth optical (4, H, W)
                - valid_mask: Valid pixel mask (1, H, W)
        """
        tile_path = self.tile_paths[idx]
        
        # Load tile data
        data = np.load(tile_path)
        
        # Extract SAR (VV, VH)
        s1 = np.stack([
            data.get('s1_vv', np.zeros((self.tile_size, self.tile_size), dtype=np.float32)),
            data.get('s1_vh', np.zeros((self.tile_size, self.tile_size), dtype=np.float32))
        ], axis=0)  # (2, H, W)
        
        # Extract ground truth S2 (Blue, Green, Red, NIR)
        s2_truth = np.stack([
            data.get('s2_b2', np.zeros((self.tile_size, self.tile_size), dtype=np.float32)),
            data.get('s2_b3', np.zeros((self.tile_size, self.tile_size), dtype=np.float32)),
            data.get('s2_b4', np.zeros((self.tile_size, self.tile_size), dtype=np.float32)),
            data.get('s2_b8', np.zeros((self.tile_size, self.tile_size), dtype=np.float32))
        ], axis=0)  # (4, H, W)
        
        # Load Stage 2 output (opt_v2)
        if self.stage2_dir:
            # Load pre-computed opt_v2
            opt_v2_path = self.stage2_dir / tile_path.name
            if opt_v2_path.exists():
                opt_v2_data = np.load(opt_v2_path)
                opt_v2 = opt_v2_data.get('opt_v2', s2_truth.copy())
            else:
                # Fallback: use ground truth as proxy
                opt_v2 = s2_truth.copy()
        else:
            # For initial training without Stage 2: use noisy target as proxy
            opt_v2 = s2_truth + np.random.randn(*s2_truth.shape).astype(np.float32) * 0.05
            opt_v2 = np.clip(opt_v2, 0, 1)
        
        # Create valid mask (non-NaN pixels)
        valid_mask = ~(np.isnan(s1).any(axis=0, keepdims=True) | 
                       np.isnan(opt_v2).any(axis=0, keepdims=True) |
                       np.isnan(s2_truth).any(axis=0, keepdims=True))  # (1, H, W)
        
        # Replace NaN with zeros
        s1 = np.nan_to_num(s1, nan=0.0)
        opt_v2 = np.nan_to_num(opt_v2, nan=0.0)
        s2_truth = np.nan_to_num(s2_truth, nan=0.0)
        
        # Apply augmentation if training
        if self.use_augmentation:
            s1, opt_v2, s2_truth, valid_mask = self._augment(s1, opt_v2, s2_truth, valid_mask)
        
        # Pad to 128x128 for TerraMind (requires divisible by 16)
        # Current tiles are 120x120, pad to 128x128
        target_size = 128
        if s1.shape[1] != target_size or s1.shape[2] != target_size:
            pad_h = target_size - s1.shape[1]
            pad_w = target_size - s1.shape[2]
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            s1 = np.pad(s1, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
            opt_v2 = np.pad(opt_v2, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
            s2_truth = np.pad(s2_truth, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
            valid_mask = np.pad(valid_mask, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
        
        # Convert to tensors
        sample = {
            's1': torch.from_numpy(s1).float(),
            'opt_v2': torch.from_numpy(opt_v2).float(),
            's2_truth': torch.from_numpy(s2_truth).float(),
            'valid_mask': torch.from_numpy(valid_mask.astype(np.float32))
        }
        
        return sample
    
    def _augment(
        self,
        s1: np.ndarray,
        opt_v2: np.ndarray,
        s2_truth: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply data augmentation (flips and rotations).
        
        Args:
            s1: SAR (2, H, W)
            opt_v2: Stage 2 output (4, H, W)
            s2_truth: Ground truth (4, H, W)
            valid_mask: Valid pixels (1, H, W)
            
        Returns:
            Augmented (s1, opt_v2, s2_truth, valid_mask)
        """
        # Random horizontal flip
        if np.random.rand() < 0.5:
            s1 = np.flip(s1, axis=2).copy()
            opt_v2 = np.flip(opt_v2, axis=2).copy()
            s2_truth = np.flip(s2_truth, axis=2).copy()
            valid_mask = np.flip(valid_mask, axis=2).copy()
        
        # Random vertical flip
        if np.random.rand() < 0.5:
            s1 = np.flip(s1, axis=1).copy()
            opt_v2 = np.flip(opt_v2, axis=1).copy()
            s2_truth = np.flip(s2_truth, axis=1).copy()
            valid_mask = np.flip(valid_mask, axis=1).copy()
        
        # Random 90-degree rotations
        k = np.random.randint(0, 4)
        if k > 0:
            s1 = np.rot90(s1, k=k, axes=(1, 2)).copy()
            opt_v2 = np.rot90(opt_v2, k=k, axes=(1, 2)).copy()
            s2_truth = np.rot90(s2_truth, k=k, axes=(1, 2)).copy()
            valid_mask = np.rot90(valid_mask, k=k, axes=(1, 2)).copy()
        
        return s1, opt_v2, s2_truth, valid_mask


# ============================================================================
# SAR Agreement Metric (for early stopping)
# ============================================================================

def compute_sar_agreement(
    opt_v3: torch.Tensor,
    s1: torch.Tensor,
    reduction: str = 'mean'
) -> float:
    """
    Compute SAR-optical agreement metric based on edge correlation.
    
    This metric measures how well the optical output's edges align with
    SAR edges. Higher values indicate better agreement.
    
    Args:
        opt_v3: Optical output (B, 4, H, W)
        s1: SAR input (B, 2, H, W)
        reduction: 'mean' or 'none'
        
    Returns:
        SAR agreement score (higher is better)
    """
    # Convert to grayscale
    opt_gray = (0.2989 * opt_v3[:,0,:,:] + 
                0.5870 * opt_v3[:,1,:,:] + 
                0.1140 * opt_v3[:,2,:,:]).unsqueeze(1)
    s1_gray = s1.mean(dim=1, keepdim=True)
    
    # Sobel edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32, device=opt_v3.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32, device=opt_v3.device).view(1, 1, 3, 3)
    
    # Compute edges
    opt_edges_x = F.conv2d(opt_gray, sobel_x, padding=1)
    opt_edges_y = F.conv2d(opt_gray, sobel_y, padding=1)
    opt_edges = torch.sqrt(opt_edges_x ** 2 + opt_edges_y ** 2 + 1e-8)
    
    s1_edges_x = F.conv2d(s1_gray, sobel_x, padding=1)
    s1_edges_y = F.conv2d(s1_gray, sobel_y, padding=1)
    s1_edges = torch.sqrt(s1_edges_x ** 2 + s1_edges_y ** 2 + 1e-8)
    
    # Normalize edges
    opt_edges_norm = opt_edges / (opt_edges.max() + 1e-8)
    s1_edges_norm = s1_edges / (s1_edges.max() + 1e-8)
    
    # Compute correlation (higher is better)
    correlation = F.cosine_similarity(
        opt_edges_norm.flatten(1),
        s1_edges_norm.flatten(1),
        dim=1
    )
    
    if reduction == 'mean':
        return correlation.mean().item()
    else:
        return correlation.cpu().numpy()


class SARAgreementEarlyStopping:
    """
    Early stopping based on SAR-agreement metric plateau.
    
    Monitors SAR-optical edge agreement and stops training if no
    improvement is seen for a specified number of validations.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of validations to wait for improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        self.history = []
    
    def __call__(self, sar_agreement: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            sar_agreement: Current SAR agreement score (higher is better)
            epoch: Current epoch
            
        Returns:
            True if training should stop, False otherwise
        """
        self.history.append((epoch, sar_agreement))
        
        if self.best_score is None:
            self.best_score = sar_agreement
            self.best_epoch = epoch
            return False
        
        # Check if there's improvement (higher is better)
        improved = sar_agreement > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = sar_agreement
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"  ✓ SAR agreement improved to {sar_agreement:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  No improvement in SAR agreement ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n⚠ Early stopping triggered!")
                    print(f"  Best SAR agreement: {self.best_score:.6f} at epoch {self.best_epoch}")
                return True
        
        return False


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    grad_accum_steps: int,
    device: torch.device,
    epoch: int,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Stage 3 model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scaler: AMP gradient scaler
        grad_accum_steps: Gradient accumulation steps
        device: Device
        epoch: Current epoch
        verbose: Print progress
        
    Returns:
        Dict with loss statistics
    """
    model.train()
    
    total_loss = 0.0
    loss_components = {}
    num_batches = 0
    
    optimizer.zero_grad()
    
    if verbose:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    else:
        pbar = dataloader
    
    for i, batch in enumerate(pbar):
        # Move to device
        s1 = batch['s1'].to(device)
        opt_v2 = batch['opt_v2'].to(device)
        s2_truth = batch['s2_truth'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        
        # NOTE: We don't need to set requires_grad on inputs explicitly.
        # The LoRA parameters have requires_grad=True, so when inputs pass through
        # LoRA layers, PyTorch automatically creates a computation graph from
        # the LoRA parameters to the output. This is how LoRA fine-tuning works:
        # frozen base + trainable LoRA = gradients flow only through LoRA path.
        
        # Forward pass with AMP
        with autocast():
            # Generate opt_v3
            opt_v3 = model(s1, opt_v2)
            
            # Compute loss
            losses = criterion(
                opt_v3=opt_v3,
                opt_v2=opt_v2,
                s1=s1,
                s2_truth=s2_truth,
                mask=valid_mask
            )
            
            loss = losses['total'] / grad_accum_steps
        
        # Backward pass with AMP
        scaler.scale(loss).backward()
        
        # Update weights every grad_accum_steps
        if (i + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Accumulate loss statistics
        total_loss += losses['total'].item()
        for key, value in losses.items():
            if key != 'total':
                if key not in loss_components:
                    loss_components[key] = 0.0
                if isinstance(value, torch.Tensor):
                    loss_components[key] += value.item()
                else:
                    loss_components[key] += value
        
        num_batches += 1
        
        # Update progress bar
        if verbose:
            pbar.set_postfix({'loss': loss.item() * grad_accum_steps})
    
    # Average losses
    total_loss /= num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    return {
        'total_loss': total_loss,
        **loss_components
    }


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Args:
        model: Stage 3 model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device
        epoch: Current epoch
        verbose: Print progress
        
    Returns:
        Dict with loss and metric statistics
    """
    model.eval()
    
    total_loss = 0.0
    loss_components = {}
    sar_agreements = []
    num_batches = 0
    
    # Initialize GAC-Score metric
    gac_metric = GACScore(device=device)
    
    if verbose:
        pbar = tqdm(dataloader, desc=f"Validation {epoch}", leave=False)
    else:
        pbar = dataloader
    
    for batch in pbar:
        # Move to device
        s1 = batch['s1'].to(device)
        opt_v2 = batch['opt_v2'].to(device)
        s2_truth = batch['s2_truth'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        
        # Forward pass
        opt_v3 = model(s1, opt_v2)
        
        # Compute loss
        losses = criterion(
            opt_v3=opt_v3,
            opt_v2=opt_v2,
            s1=s1,
            s2_truth=s2_truth,
            mask=valid_mask
        )
        
        # Update GAC-Score (opt_v3 and s2_truth should be in [0, 1] range)
        gac_metric.update(opt_v3, s2_truth, s1)
        
        # Compute SAR agreement
        sar_agr = compute_sar_agreement(opt_v3, s1, reduction='mean')
        sar_agreements.append(sar_agr)
        
        # Accumulate loss statistics
        total_loss += losses['total'].item()
        for key, value in losses.items():
            if key != 'total':
                if key not in loss_components:
                    loss_components[key] = 0.0
                if isinstance(value, torch.Tensor):
                    loss_components[key] += value.item()
                else:
                    loss_components[key] += value
        
        num_batches += 1
        
        # Update progress bar
        if verbose:
            pbar.set_postfix({'loss': losses['total'].item(), 'sar_agr': sar_agr})
    
    # Average losses and metrics
    total_loss /= num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    avg_sar_agreement = np.mean(sar_agreements)
    
    # Compute GAC-Score
    gac_score = gac_metric.compute()
    
    return {
        'total_loss': total_loss,
        'sar_agreement': avg_sar_agreement,
        'gac_score': gac_score,
        **loss_components
    }


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Stage 3 TerraMind SAR Grounding with LoRA"
    )
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing NPZ tiles')
    parser.add_argument('--stage2-dir', type=str, default=None,
                       help='Directory containing Stage 2 opt_v2 outputs')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for checkpoints')
    
    # Model arguments
    parser.add_argument('--timesteps', type=int, default=10,
                       help='Number of diffusion timesteps (default: 10)')
    parser.add_argument('--lora-rank', type=int, default=8,
                       help='LoRA rank (default: 8)')
    parser.add_argument('--lora-alpha', type=int, default=16,
                       help='LoRA alpha (default: 16)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained TerraMind weights')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size per GPU (default: 1)')
    parser.add_argument('--grad-accum', type=int, default=8,
                       help='Gradient accumulation steps (default: 8)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay (default: 1e-5)')
    
    # Loss weights
    parser.add_argument('--sar-weight', type=float, default=1.0,
                       help='SAR consistency loss weight (default: 1.0)')
    parser.add_argument('--cycle-weight', type=float, default=0.5,
                       help='Cycle loss weight (default: 0.5)')
    parser.add_argument('--identity-weight', type=float, default=0.3,
                       help='Identity loss weight (default: 0.3)')
    parser.add_argument('--lpips-weight', type=float, default=0.1,
                       help='LPIPS weight (default: 0.1)')
    parser.add_argument('--spectral-weight', type=float, default=1.0,
                       help='Spectral L1 weight (default: 1.0)')
    parser.add_argument('--urban-weight', type=float, default=1.0,
                       help='Urban/high-backscatter weighting factor (1.0 = no boost, >1.0 = boost urban areas, default: 1.0)')
    parser.add_argument('--dem-weight', type=float, default=0.0,
                       help='DEM slope weighting factor (0.0 = disabled, >0.0 = enable slope-aware weighting, default: 0.0)')
    
    # Early stopping
    parser.add_argument('--early-stop-patience', type=int, default=5,
                       help='Early stopping patience (default: 5)')
    parser.add_argument('--early-stop-delta', type=float, default=1e-4,
                       help='Early stopping minimum delta (default: 1e-4)')
    
    # Other arguments
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of dataloader workers (default: auto)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples (for debugging)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("\n" + "=" * 80)
    print("Stage 3 Training: TerraMind SAR Grounding with LoRA")
    print("=" * 80)
    print(f"\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Build model
    print("=" * 80)
    print("Building Stage 3 Model")
    print("=" * 80)
    model = build_stage3_backbone_model(
        freeze_backbone=False,  # Full fine-tuning
        pretrained=args.pretrained,
        standardize=True,
        device=device
    )
    
    # Model already built with trainable parameters - no LoRA needed!
    
    # Build loss function
    print("\n" + "=" * 80)
    print("Building Loss Function")
    print("=" * 80)
    criterion = Stage3Loss(
        sar_weight=args.sar_weight,
        cycle_weight=args.cycle_weight,
        identity_weight=args.identity_weight,
        lpips_weight=args.lpips_weight,
        spectral_weight=args.spectral_weight,
        use_lpips=True,
        urban_weight=args.urban_weight,
        dem_weight=args.dem_weight
    ).to(device)
    print("✓ Stage 3 loss initialized")
    print(f"  SAR weight: {args.sar_weight}")
    print(f"  Cycle weight: {args.cycle_weight}")
    print(f"  Identity weight: {args.identity_weight}")
    print(f"  LPIPS weight: {args.lpips_weight}")
    print(f"  Spectral weight: {args.spectral_weight}")
    if args.urban_weight > 1.0:
        print(f"  Urban weight: {args.urban_weight} ✓ ENABLED")
    else:
        print(f"  Urban weight: {args.urban_weight} (disabled)")
    if args.dem_weight > 0.0:
        print(f"  DEM weight: {args.dem_weight} ✓ ENABLED")
    else:
        print(f"  DEM weight: {args.dem_weight} (disabled)")
    
    # Build optimizer (optimize all trainable parameters)
    print("\n" + "=" * 80)
    print("Building Optimizer")
    print("=" * 80)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print(f"✓ AdamW optimizer initialized")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Optimizing {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # AMP gradient scaler
    scaler = GradScaler()
    
    # Build datasets
    print("\n" + "=" * 80)
    print("Loading Datasets")
    print("=" * 80)
    
    train_dataset = Stage3Dataset(
        data_dir=Path(args.data_dir),
        split='train',
        stage2_dir=Path(args.stage2_dir) if args.stage2_dir else None,
        max_samples=args.max_samples,
        use_augmentation=True
    )
    
    val_dataset = Stage3Dataset(
        data_dir=Path(args.data_dir),
        split='val',
        stage2_dir=Path(args.stage2_dir) if args.stage2_dir else None,
        max_samples=args.max_samples // 4 if args.max_samples else None,
        use_augmentation=False
    )
    
    # Determine number of workers
    if args.num_workers is None:
        args.num_workers = get_recommended_num_workers()
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"\n✓ Dataloaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Effective train batch size: {args.batch_size * args.grad_accum}")
    
    # Early stopping
    early_stopping = SARAgreementEarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_delta,
        verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_sar_agreement = 0.0
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_sar_agreement = checkpoint.get('best_sar_agreement', 0.0)
        print(f"✓ Resumed from epoch {checkpoint['epoch']}")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_stats = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            grad_accum_steps=args.grad_accum,
            device=device,
            epoch=epoch + 1,
            verbose=True
        )
        
        # Validate
        val_stats = validate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch + 1,
            verbose=True
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print statistics
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train loss: {train_stats['total_loss']:.6f}")
        print(f"  Val loss: {val_stats['total_loss']:.6f}")
        print(f"  Val SAR agreement: {val_stats['sar_agreement']:.6f}")
        if 'gac_score' in val_stats:
            print(f"  Val GAC-Score: {val_stats['gac_score']:.4f}")
        print(f"  Learning rate: {scheduler.get_last_lr()[0]:.2e}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save checkpoint
        is_best = val_stats['sar_agreement'] > best_sar_agreement
        if is_best:
            best_sar_agreement = val_stats['sar_agreement']
        
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch + 1}.pt'
        save_checkpoint(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_stats': train_stats,
                'val_stats': val_stats,
                'best_sar_agreement': best_sar_agreement,
                'config': vars(args)
            },
            checkpoint_path,
            is_best=is_best,
            best_path=output_dir / 'best_model.pt'
        )
        
        # Early stopping check
        if early_stopping(val_stats['sar_agreement'], epoch + 1):
            print("\n⚠ Early stopping triggered!")
            break
    
    # Training complete
    total_time = time.time() - training_start_time
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Best SAR agreement: {best_sar_agreement:.6f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"Best model: {output_dir / 'best_model.pt'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
