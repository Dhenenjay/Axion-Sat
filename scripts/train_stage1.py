"""
Stage 1 Training: TerraMind SAR-to-Optical with LoRA

This script trains TerraMind's generative model for SAR-to-optical translation
using parameter-efficient LoRA adapters on the denoiser projections and cross-attention layers.

Training Configuration:
    - Model: TerraMind generator with timesteps=12
    - LoRA: Applied to denoiser projections and cross-attention
    - Loss: L1 + MS-SSIM + LPIPS (small) + SAR-structure
    - Precision: Mixed precision (AMP fp16)
    - Batch size: 1 per GPU
    - Gradient accumulation: 8 steps
    - Optimizer: AdamW with cosine scheduling

Memory Optimizations:
    - LoRA adapters (only ~1-5% of parameters trainable)
    - Gradient accumulation (effective batch size = 8)
    - AMP fp16 (reduces memory by ~50%)
    - Gradient checkpointing (optional, for very low VRAM)

Usage:
    python scripts/train_stage1.py --data-dir tiles/ --output-dir checkpoints/stage1/
    python scripts/train_stage1.py --help
"""

import argparse
import os
import sys
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axs_lib.models import build_terramind_generator
from axs_lib.stage1_tm_s2o import tm_sar2opt
from axs_lib.losses import CombinedLoss
from axs_lib.logging import TrainingLogger
from axs_lib.io import CheckpointManager
from axs_lib.stdz import TERRAMIND_S1_STATS, TERRAMIND_S2_STATS
from axs_lib.reproducibility import set_seed, get_seed_state, add_seed_to_checkpoint, create_reproducible_generator
from axs_lib.dataloader_utils import create_dataloader, get_recommended_num_workers, print_dataloader_config
from axs_lib.metrics import GACScore


# ============================================================================
# LoRA Adapter Implementation
# ============================================================================

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer.
    
    Adds trainable low-rank decomposition to frozen linear layers:
        h = W_0*x + (B*A)*x
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
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize A with Kaiming uniform, B with zeros
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
        # x @ A^T -> (..., rank)
        # (..., rank) @ B^T -> (..., out_features)
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        
        return lora_out * self.scaling


def apply_lora_to_linear(
    module: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    target_modules: Optional[List[str]] = None
) -> int:
    """
    Apply LoRA adapters to linear layers in a module.
    
    Args:
        module: PyTorch module to modify
        rank: LoRA rank
        alpha: LoRA alpha
        target_modules: List of module name patterns to target (None = all linear layers)
        
    Returns:
        Number of LoRA layers added
    """
    if target_modules is None:
        # Default: target projection and attention layers
        target_modules = [
            'proj',      # Projections in denoiser
            'q_proj',    # Query projection in attention
            'k_proj',    # Key projection in attention
            'v_proj',    # Value projection in attention
            'o_proj',    # Output projection in attention
            'c_attn',    # Combined QKV projection
            'cross_attn' # Cross-attention layers
        ]
    
    count = 0
    
    for name, submodule in module.named_modules():
        # Check if this is a linear layer we want to adapt
        if not isinstance(submodule, nn.Linear):
            continue
        
        # Check if name matches any target pattern
        matches = any(pattern in name.lower() for pattern in target_modules)
        if not matches:
            continue
        
        # Get parent module and attribute name
        parent_name, attr_name = name.rsplit('.', 1) if '.' in name else ('', name)
        parent = module if parent_name == '' else module.get_submodule(parent_name)
        
        # Create LoRA-enhanced linear layer
        in_features = submodule.in_features
        out_features = submodule.out_features
        
        # Freeze original layer
        for param in submodule.parameters():
            param.requires_grad = False
        
        # Add LoRA adapter as a new attribute
        lora_layer = LoRALayer(in_features, out_features, rank=rank, alpha=alpha)
        setattr(submodule, 'lora_adapter', lora_layer)
        
        # Monkey-patch forward method
        original_forward = submodule.forward
        
        def make_lora_forward(orig_forward, lora):
            def lora_forward(x):
                # Compute base output (frozen)
                with torch.no_grad():
                    base_out = orig_forward(x)
                # Add LoRA output (trainable)
                lora_out = lora(x)
                return base_out + lora_out
            return lora_forward
        
        submodule.forward = make_lora_forward(original_forward, lora_layer)
        
        count += 1
        print(f"  Added LoRA to: {name}")
    
    return count


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA parameters from a model."""
    lora_params = []
    
    for module in model.modules():
        if hasattr(module, 'lora_adapter'):
            lora_params.extend(module.lora_adapter.parameters())
    
    return lora_params


# ============================================================================
# Dataset
# ============================================================================

class TileDataset(Dataset):
    """
    Dataset for loading SAR/Optical tile pairs.
    
    Loads NPZ tiles containing 's1_vv', 's1_vh', 's2_b2', 's2_b3', 's2_b4', 's2_b8'.
    """
    
    def __init__(
        self,
        tile_dir: str,
        s1_mean: Optional[np.ndarray] = None,
        s1_std: Optional[np.ndarray] = None,
        s2_mean: Optional[np.ndarray] = None,
        s2_std: Optional[np.ndarray] = None,
        augment: bool = True,
        tile_size: Optional[int] = None,
        max_tiles: Optional[int] = None
    ):
        self.tile_dir = Path(tile_dir)
        self.augment = augment
        self.tile_size = tile_size
        
        # Find all NPZ tiles
        self.tile_paths = sorted(list(self.tile_dir.glob("*.npz")))
        
        if len(self.tile_paths) == 0:
            raise ValueError(f"No NPZ tiles found in {tile_dir}")
        
        # Limit number of tiles if specified
        if max_tiles is not None and max_tiles > 0:
            self.tile_paths = self.tile_paths[:max_tiles]
            print(f"Using {len(self.tile_paths)} tiles from {tile_dir} (limited by --max-tiles)")
        else:
            print(f"Found {len(self.tile_paths)} tiles in {tile_dir}")
        
        # Set normalization statistics
        if s1_mean is None:
            s1_mean = TERRAMIND_S1_STATS.means
        if s1_std is None:
            s1_std = TERRAMIND_S1_STATS.stds
        if s2_mean is None:
            s2_mean = TERRAMIND_S2_STATS.means
        if s2_std is None:
            s2_std = TERRAMIND_S2_STATS.stds
        
        self.s1_mean = torch.from_numpy(np.array(s1_mean, dtype=np.float32)).view(1, 2, 1, 1)
        self.s1_std = torch.from_numpy(np.array(s1_std, dtype=np.float32)).view(1, 2, 1, 1)
        self.s2_mean = torch.from_numpy(np.array(s2_mean, dtype=np.float32)).view(1, 4, 1, 1)
        self.s2_std = torch.from_numpy(np.array(s2_std, dtype=np.float32)).view(1, 4, 1, 1)
    
    def __len__(self) -> int:
        return len(self.tile_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load tile
        tile_data = np.load(self.tile_paths[idx])
        
        # Extract SAR (VV, VH)
        s1_vv = tile_data['s1_vv']
        s1_vh = tile_data['s1_vh']
        s1 = np.stack([s1_vv, s1_vh], axis=0).astype(np.float32)
        
        # Extract S2 (B02, B03, B04, B08)
        s2_b2 = tile_data['s2_b2']
        s2_b3 = tile_data['s2_b3']
        s2_b4 = tile_data['s2_b4']
        s2_b8 = tile_data['s2_b8']
        s2 = np.stack([s2_b2, s2_b3, s2_b4, s2_b8], axis=0).astype(np.float32)
        
        # Convert to tensors
        s1 = torch.from_numpy(s1)
        s2 = torch.from_numpy(s2)
        
        # Standardize
        s1 = (s1 - self.s1_mean.squeeze(0)) / self.s1_std.squeeze(0)
        s2 = (s2 - self.s2_mean.squeeze(0)) / self.s2_std.squeeze(0)
        
        # Resize if tile_size specified
        if self.tile_size is not None:
            current_size = s1.shape[-1]  # Assumes square tiles
            if current_size != self.tile_size:
                s1 = F.interpolate(
                    s1.unsqueeze(0),
                    size=(self.tile_size, self.tile_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                s2 = F.interpolate(
                    s2.unsqueeze(0),
                    size=(self.tile_size, self.tile_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
        
        # Augmentation
        if self.augment:
            s1, s2 = self._augment(s1, s2)
        
        return {
            's1': s1,
            's2': s2,
            'tile_id': self.tile_paths[idx].stem
        }
    
    def _augment(self, s1: torch.Tensor, s2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentations."""
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            s1 = torch.flip(s1, dims=[2])
            s2 = torch.flip(s2, dims=[2])
        
        # Random vertical flip
        if torch.rand(1) > 0.5:
            s1 = torch.flip(s1, dims=[1])
            s2 = torch.flip(s2, dims=[1])
        
        # Random 90-degree rotations
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            s1 = torch.rot90(s1, k, dims=[1, 2])
            s2 = torch.rot90(s2, k, dims=[1, 2])
        
        return s1, s2


# ============================================================================
# Early Stopping Utilities
# ============================================================================

class EarlyStopping:
    """Early stopping based on metric plateau.
    
    Monitors a metric (e.g., SAR-edge agreement, validation loss) and stops
    training if no improvement is seen for a specified number of validations.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of validations to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for metrics that should decrease (loss), 'max' for metrics that should increase (accuracy)
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_step = 0
        
        self.val_history = []
    
    def __call__(self, score: float, step: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric value
            step: Current training step
            
        Returns:
            True if training should stop, False otherwise
        """
        self.val_history.append((step, score))
        
        if self.best_score is None:
            self.best_score = score
            self.best_step = step
            return False
        
        # Check if there's improvement
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_step = step
            self.counter = 0
            if self.verbose:
                print(f"✓ Metric improved to {score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠️  No improvement for {self.counter}/{self.patience} validations (best: {self.best_score:.6f} at step {self.best_step})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n🛑 Early stopping triggered! No improvement for {self.patience} validations.")
                    print(f"   Best score: {self.best_score:.6f} at step {self.best_step}")
                return True
        
        return False


class TopKCheckpointManager:
    """Manages top-K best checkpoints based on a metric."""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        k: int = 3,
        mode: str = 'min',
        metric_name: str = 'sar_edge_agreement'
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            k: Number of best checkpoints to keep
            mode: 'min' or 'max' for metric comparison
            metric_name: Name of metric for naming files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.k = k
        self.mode = mode
        self.metric_name = metric_name
        
        # List of (score, step, filepath) tuples
        self.top_k = []
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        score: float,
        extra_state: Optional[Dict] = None
    ) -> Optional[Path]:
        """
        Save checkpoint if it's in top-K.
        
        Returns:
            Path to saved checkpoint if saved, None otherwise
        """
        # Check if this score qualifies for top-K
        should_save = False
        
        if len(self.top_k) < self.k:
            should_save = True
        else:
            # Compare with worst in top-K
            if self.mode == 'min':
                worst_score = max(self.top_k, key=lambda x: x[0])[0]
                should_save = score < worst_score
            else:  # mode == 'max'
                worst_score = min(self.top_k, key=lambda x: x[0])[0]
                should_save = score > worst_score
        
        if not should_save:
            return None
        
        # Save checkpoint
        filename = f"best_{self.metric_name}_{score:.6f}_step_{step}.pt"
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            self.metric_name: score
        }
        
        if extra_state is not None:
            checkpoint.update(extra_state)
        
        torch.save(checkpoint, filepath)
        
        # Update top-K list
        self.top_k.append((score, step, filepath))
        
        # Sort and keep only top-K
        if self.mode == 'min':
            self.top_k.sort(key=lambda x: x[0])  # Ascending for min
        else:
            self.top_k.sort(key=lambda x: x[0], reverse=True)  # Descending for max
        
        # Remove worst if we exceed K
        if len(self.top_k) > self.k:
            _, _, old_filepath = self.top_k.pop()
            if old_filepath.exists():
                old_filepath.unlink()
                print(f"  Removed old checkpoint: {old_filepath.name}")
        
        return filepath
    
    def get_best(self) -> Optional[Tuple[float, int, Path]]:
        """Get the best checkpoint (score, step, path)."""
        if not self.top_k:
            return None
        return self.top_k[0]


# ============================================================================
# GPU Memory Utilities
# ============================================================================

def log_gpu_memory(prefix: str = ""):
    """Log current GPU memory usage."""
    if not torch.cuda.is_available():
        return
    
    device_idx = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device_idx) / 1024**3  # GB
    reserved = torch.cuda.memory_reserved(device_idx) / 1024**3    # GB
    total = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3  # GB
    free = total - allocated
    
    print(f"{prefix}GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Free={free:.2f}GB, Total={total:.2f}GB")


def is_oom_error(e: Exception) -> bool:
    """Check if exception is an Out-Of-Memory error."""
    if not isinstance(e, RuntimeError):
        return False
    return "out of memory" in str(e).lower() or "cuda" in str(e).lower() and "memory" in str(e).lower()


# ============================================================================
# Training Loop
# ============================================================================

def train_step_based(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    grad_accum_steps: int = 8,
    timesteps: int = 12,
    total_steps: int = 10000,
    val_every: int = 500,
    save_every: int = 1000,
    start_step: int = 0,
    val_dataloader: Optional[DataLoader] = None,
    logger: Optional[TrainingLogger] = None,
    ckpt_manager: Optional[CheckpointManager] = None,
    output_dir: Optional[Path] = None,
    early_stopping: Optional[EarlyStopping] = None,
    topk_ckpt_manager: Optional[TopKCheckpointManager] = None,
    seed_info: Optional[Dict] = None
) -> Dict[str, float]:
    """Train using step-based training (not epoch-based).
    
    Includes automatic OOM handling: reduces timesteps by 2 on OOM and retries.
    Supports early stopping based on SAR-edge agreement plateau.
    """
    
    model.train()
    best_val_loss = float('inf')
    best_sar_edge = float('inf')  # Track best SAR-edge agreement (lower is better)
    
    optimizer.zero_grad()
    
    step = start_step
    data_iter = iter(dataloader)
    
    # Track current timesteps (can be reduced on OOM)
    current_timesteps = timesteps
    min_timesteps = 1  # Safety minimum
    
    print(f"\nTraining from step {start_step} to {total_steps}...")
    print(f"Validation every {val_every} steps, saving every {save_every} steps")
    print(f"Initial timesteps: {current_timesteps}")
    if early_stopping is not None:
        print(f"Early stopping: patience={early_stopping.patience}, min_delta={early_stopping.min_delta}")
    if topk_ckpt_manager is not None:
        print(f"Top-{topk_ckpt_manager.k} checkpoint management enabled")
    print("=" * 80)
    
    # Log initial GPU state
    log_gpu_memory("[Initial] ")
    
    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            # Restart data iterator
            data_iter = iter(dataloader)
            batch = next(data_iter)
        s1 = batch['s1'].to(device)
        s2_target = batch['s2'].to(device)
        
        # Log GPU memory before forward pass (every 100 steps)
        if step % 100 == 0:
            log_gpu_memory(f"[Step {step} - Before Forward] ")
        
        # Forward pass with AMP and OOM handling
        oom_retry = False
        try:
            with autocast():
                # Generate synthetic optical
                s2_pred = tm_sar2opt(
                    model,
                    s1,
                    timesteps=current_timesteps,
                    denormalize=False,  # Keep in standardized space for loss
                    clip_range=None
                )
                
                # Compute loss
                loss, loss_dict = criterion(s2_pred, s2_target, sar=s1)
                
                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
        except Exception as e:
            if is_oom_error(e):
                print(f"\n⚠️  OOM Error at step {step}! Current timesteps: {current_timesteps}")
                
                # Free CUDA cache
                torch.cuda.empty_cache()
                log_gpu_memory("[After OOM - Cache Cleared] ")
                
                # Reduce timesteps by 2 (with minimum bound)
                new_timesteps = max(current_timesteps - 2, min_timesteps)
                
                if new_timesteps < current_timesteps:
                    current_timesteps = new_timesteps
                    print(f"🔧 Reducing timesteps to {current_timesteps} and retrying...")
                    
                    # Retry with reduced timesteps
                    try:
                        optimizer.zero_grad()  # Clear any partial gradients
                        
                        with autocast():
                            # Retry generation with reduced timesteps
                            s2_pred = tm_sar2opt(
                                model,
                                s1,
                                timesteps=current_timesteps,
                                denormalize=False,
                                clip_range=None
                            )
                            
                            # Compute loss
                            loss, loss_dict = criterion(s2_pred, s2_target, sar=s1)
                            loss = loss / grad_accum_steps
                        
                        # Backward pass
                        scaler.scale(loss).backward()
                        oom_retry = True
                        print(f"✓ Retry successful with timesteps={current_timesteps}")
                        
                    except Exception as retry_e:
                        if is_oom_error(retry_e):
                            print(f"❌ Retry failed with OOM even at timesteps={current_timesteps}")
                            torch.cuda.empty_cache()
                            # Skip this batch and continue
                            step += 1
                            continue
                        else:
                            raise retry_e
                else:
                    print(f"❌ Already at minimum timesteps ({min_timesteps}), cannot reduce further")
                    torch.cuda.empty_cache()
                    # Skip this batch
                    step += 1
                    continue
            else:
                # Not an OOM error, re-raise
                raise e
        
        # Log GPU memory after backward pass (every 100 steps)
        if step % 100 == 0:
            log_gpu_memory(f"[Step {step} - After Backward] ")
        
        # Optimizer step every grad_accum_steps
        if (step + 1) % grad_accum_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Step scheduler
            scheduler.step()
            
            # Log step
            if logger is not None:
                logger.log_step(
                    step=step,
                    loss=loss.item() * grad_accum_steps,
                    lr=optimizer.param_groups[0]['lr'],
                    **loss_dict
                )
            
            # Progress
            if step % 10 == 0:
                print(f"  Step {step}/{total_steps}, Loss: {loss.item() * grad_accum_steps:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Validation
        if step % val_every == 0 and step > 0 and val_dataloader is not None:
            print(f"\n[Step {step}] Running validation...")
            log_gpu_memory("[Before Validation] ")
            
            val_metrics = validate(
                model=model,
                dataloader=val_dataloader,
                criterion=criterion,
                device=device,
                timesteps=current_timesteps,  # Use current (possibly reduced) timesteps
                check_s2_coverage=(step == val_every)  # Check coverage on first validation
            )
            
            log_gpu_memory("[After Validation] ")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            if val_metrics.get('num_batches_with_s2', 0) > 0:
                print(f"  ({val_metrics['num_batches_with_s2']}/{val_metrics['num_batches']} batches with S2 truth)")
            print(f"  L1: {val_metrics['l1']:.4f}, MS-SSIM: {val_metrics['ms_ssim']:.4f}")
            
            # GAC-Score (higher is better - composite metric)
            if 'gac_score' in val_metrics:
                print(f"  GAC-Score: {val_metrics['gac_score']:.4f}")
            
            # SAR-edge agreement is the SAR structure loss (lower is better)
            sar_edge_agreement = val_metrics.get('sar_structure', val_metrics['loss'])
            print(f"  SAR-Edge Agreement: {sar_edge_agreement:.4f}\n")
            
            if logger is not None:
                logger.log_validation(
                    step=step,
                    val_loss=val_metrics['loss'],
                    sar_edge_agreement=sar_edge_agreement,
                    **val_metrics
                )
            
            # Save best model (with seed info)
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                if output_dir is not None:
                    best_path = output_dir / 'best_model.pt'
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'step': step,
                        'val_loss': val_metrics['loss'],
                        'sar_edge_agreement': sar_edge_agreement
                    }
                    # Add GAC-Score if available
                    if 'gac_score' in val_metrics:
                        checkpoint['gac_score'] = val_metrics['gac_score']
                    # Add seed information
                    add_seed_to_checkpoint(checkpoint, seed_info)
                    torch.save(checkpoint, best_path)
                    print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
            
            # Save top-K checkpoints by SAR-edge agreement
            if topk_ckpt_manager is not None:
                if sar_edge_agreement < best_sar_edge:
                    best_sar_edge = sar_edge_agreement
                
                # Prepare extra state with seed info
                extra_state = {
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'val_loss': val_metrics['loss']
                }
                # Add GAC-Score if available
                if 'gac_score' in val_metrics:
                    extra_state['gac_score'] = val_metrics['gac_score']
                # Add seed information
                add_seed_to_checkpoint(extra_state, seed_info)
                
                saved_path = topk_ckpt_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    score=sar_edge_agreement,
                    extra_state=extra_state
                )
                if saved_path:
                    print(f"✓ Saved top-K checkpoint: {saved_path.name}")
            
            # Check early stopping
            if early_stopping is not None:
                if early_stopping(sar_edge_agreement, step):
                    print("\n" + "=" * 80)
                    print("Early stopping triggered - training complete!")
                    print(f"Best SAR-edge agreement: {early_stopping.best_score:.6f} at step {early_stopping.best_step}")
                    print("=" * 80 + "\n")
                    return {
                        'best_val_loss': best_val_loss,
                        'best_sar_edge_agreement': best_sar_edge,
                        'stopped_early': True,
                        'final_step': step
                    }
            
            model.train()
        
        # Save checkpoint
        if step % save_every == 0 and step > 0 and ckpt_manager is not None:
            print(f"\n[Step {step}] Saving checkpoint...")
            # Prepare checkpoint with seed info
            extra_state = {
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            # Add seed information
            add_seed_to_checkpoint(extra_state, seed_info)
            
            ckpt_path = ckpt_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                metrics={'step': step},
                extra_state=extra_state,
                filename=f"checkpoint_step_{step}.pt"
            )
            if logger is not None:
                logger.log_checkpoint(
                    step=step,
                    checkpoint_path=str(ckpt_path),
                    is_best=False
                )
            print(f"✓ Checkpoint saved: {ckpt_path}\n")
        
        step += 1
    
    return {
        'best_val_loss': best_val_loss,
        'best_sar_edge_agreement': best_sar_edge,
        'stopped_early': False,
        'final_step': step
    }


def check_validation_s2_coverage(dataloader: DataLoader) -> Tuple[int, int, float]:
    """
    Check how many validation tiles have valid S2 truth data.
    
    Args:
        dataloader: Validation dataloader
        
    Returns:
        Tuple of (total_tiles, tiles_with_s2, coverage_rate)
    """
    total_tiles = 0
    tiles_with_s2 = 0
    
    for batch in dataloader:
        s2 = batch['s2']
        batch_size = s2.shape[0]
        total_tiles += batch_size
        
        # Check if S2 data is not all zeros/NaNs (missing)
        for i in range(batch_size):
            s2_sample = s2[i]
            # Consider valid if not all zeros and not all NaNs
            is_valid = (torch.abs(s2_sample).max() > 1e-6) and not torch.isnan(s2_sample).any()
            if is_valid:
                tiles_with_s2 += 1
    
    coverage_rate = tiles_with_s2 / total_tiles if total_tiles > 0 else 0.0
    return total_tiles, tiles_with_s2, coverage_rate


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    timesteps: int = 12,
    check_s2_coverage: bool = False
) -> Dict[str, float]:
    """Validate the model.
    
    Includes automatic OOM handling: reduces timesteps by 2 on OOM and retries.
    Always keeps SAR-edge validation even if S2 truth is missing.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss criterion
        device: Device to run on
        timesteps: Number of diffusion timesteps
        check_s2_coverage: Whether to check S2 data coverage (first validation only)
    """
    
    model.eval()
    total_loss = 0.0
    component_losses = {
        'l1': 0.0,
        'ms_ssim': 0.0,
        'lpips': 0.0,
        'sar_structure': 0.0
    }
    num_batches = 0
    num_batches_with_s2 = 0
    num_batches_without_s2 = 0
    
    # Track current timesteps (can be reduced on OOM)
    current_timesteps = timesteps
    min_timesteps = 1
    
    # Initialize GAC-Score metric
    gac_metric = GACScore(device=device)
    
    for batch in dataloader:
        s1 = batch['s1'].to(device)
        s2_target = batch['s2'].to(device)
        
        # Check if S2 truth is available (not all zeros/NaNs)
        has_s2_truth = (torch.abs(s2_target).max() > 1e-6) and not torch.isnan(s2_target).any()
        
        # Generate synthetic optical with OOM handling
        try:
            s2_pred = tm_sar2opt(
                model,
                s1,
                timesteps=current_timesteps,
                denormalize=False,
                clip_range=None
            )
            
            # Compute loss (SAR-structure loss works even without S2 truth)
            loss, loss_dict = criterion(s2_pred, s2_target, sar=s1)
            
            # Update GAC-Score metric (only when S2 truth is available)
            if has_s2_truth:
                num_batches_with_s2 += 1
                # GAC-Score expects denormalized images in [0, 1] range
                # Denormalize predictions and targets
                s2_pred_denorm = (s2_pred * torch.tensor(TERRAMIND_S2_STATS['std'], device=device).view(1, -1, 1, 1) + 
                                 torch.tensor(TERRAMIND_S2_STATS['mean'], device=device).view(1, -1, 1, 1))
                s2_target_denorm = (s2_target * torch.tensor(TERRAMIND_S2_STATS['std'], device=device).view(1, -1, 1, 1) + 
                                   torch.tensor(TERRAMIND_S2_STATS['mean'], device=device).view(1, -1, 1, 1))
                # Clip to [0, 1]
                s2_pred_denorm = torch.clamp(s2_pred_denorm, 0, 1)
                s2_target_denorm = torch.clamp(s2_target_denorm, 0, 1)
                # Update metric
                gac_metric.update(s2_pred_denorm, s2_target_denorm, s1)
            else:
                num_batches_without_s2 += 1
            
            total_loss += loss.item()
            for key in component_losses:
                if key in loss_dict:
                    component_losses[key] += loss_dict[key]
            num_batches += 1
            
        except Exception as e:
            if is_oom_error(e):
                print(f"\n⚠️  OOM Error during validation! Current timesteps: {current_timesteps}")
                
                # Free CUDA cache
                torch.cuda.empty_cache()
                
                # Reduce timesteps by 2
                new_timesteps = max(current_timesteps - 2, min_timesteps)
                
                if new_timesteps < current_timesteps:
                    current_timesteps = new_timesteps
                    print(f"🔧 Reducing timesteps to {current_timesteps} and retrying validation batch...")
                    
                    # Retry with reduced timesteps
                    try:
                        s2_pred = tm_sar2opt(
                            model,
                            s1,
                            timesteps=current_timesteps,
                            denormalize=False,
                            clip_range=None
                        )
                        
                        loss, loss_dict = criterion(s2_pred, s2_target, sar=s1)
                        
                        # Update GAC-Score on successful retry
                        if has_s2_truth:
                            s2_pred_denorm = (s2_pred * torch.tensor(TERRAMIND_S2_STATS['std'], device=device).view(1, -1, 1, 1) + 
                                             torch.tensor(TERRAMIND_S2_STATS['mean'], device=device).view(1, -1, 1, 1))
                            s2_target_denorm = (s2_target * torch.tensor(TERRAMIND_S2_STATS['std'], device=device).view(1, -1, 1, 1) + 
                                               torch.tensor(TERRAMIND_S2_STATS['mean'], device=device).view(1, -1, 1, 1))
                            s2_pred_denorm = torch.clamp(s2_pred_denorm, 0, 1)
                            s2_target_denorm = torch.clamp(s2_target_denorm, 0, 1)
                            gac_metric.update(s2_pred_denorm, s2_target_denorm, s1)
                        
                        total_loss += loss.item()
                        for key in component_losses:
                            if key in loss_dict:
                                component_losses[key] += loss_dict[key]
                        num_batches += 1
                        print(f"✓ Validation batch retry successful with timesteps={current_timesteps}")
                        
                    except Exception as retry_e:
                        if is_oom_error(retry_e):
                            print(f"❌ Validation retry failed with OOM, skipping batch")
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise retry_e
                else:
                    print(f"❌ Already at minimum timesteps ({min_timesteps}), skipping batch")
                    torch.cuda.empty_cache()
                    continue
            else:
                # Not an OOM error, re-raise
                raise e
    
    # Average losses
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    for key in component_losses:
        component_losses[key] = component_losses[key] / num_batches if num_batches > 0 else 0.0
    
    # Compute GAC-Score (only if we have S2 batches)
    gac_score = gac_metric.compute() if num_batches_with_s2 > 0 else None
    
    result = {
        'loss': avg_loss,
        'num_batches': num_batches,
        'num_batches_with_s2': num_batches_with_s2,
        'num_batches_without_s2': num_batches_without_s2,
        **component_losses
    }
    
    # Add GAC-Score to result if available
    if gac_score is not None:
        result['gac_score'] = gac_score
    
    # Warn if too many validation batches are missing S2 truth
    if num_batches > 0:
        s2_missing_rate = num_batches_without_s2 / num_batches
        if s2_missing_rate > 0.5:
            print(f"\n⚠️  WARNING: {s2_missing_rate*100:.1f}% of validation batches are missing S2 truth!")
            print(f"   ({num_batches_without_s2}/{num_batches} batches without S2 data)")
            print(f"   Validation will rely primarily on SAR-edge agreement.")
            print(f"   Consider adding more tiles with optical coverage.\n")
    
    return result


# ============================================================================
# Main Training Script
# ============================================================================

def load_config_from_yaml(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_config_with_args(config: Dict, args: argparse.Namespace) -> argparse.Namespace:
    """Merge YAML config with command-line arguments.
    
    Command-line arguments take precedence over config file.
    """
    # Model configuration
    if 'model' in config:
        if not hasattr(args, 'timesteps') or args.timesteps == 12:  # Default value
            args.timesteps = config['model'].get('timesteps', 12)
        if not hasattr(args, 'standardize') or args.standardize is True:  # Default value
            args.standardize = config['model'].get('standardize', True)
        if 'lora' in config['model']:
            if not hasattr(args, 'lora_rank') or args.lora_rank == 8:
                args.lora_rank = config['model']['lora'].get('rank', 8)
            if not hasattr(args, 'lora_alpha') or args.lora_alpha == 16:
                args.lora_alpha = config['model']['lora'].get('alpha', 16)
    
    # Data configuration
    if 'data' in config:
        if not hasattr(args, 'data_dir') or args.data_dir is None:
            args.data_dir = config['data'].get('data_dir')
        if not hasattr(args, 'val_split') or args.val_split == 0.1:
            args.val_split = config['data'].get('val_split', 0.1)
        if not hasattr(args, 'tile_size') or args.tile_size is None:
            args.tile_size = config['data'].get('tile_size')
        if not hasattr(args, 'max_tiles') or args.max_tiles is None:
            args.max_tiles = config['data'].get('max_tiles')
        if not hasattr(args, 'num_workers') or args.num_workers == 4:
            args.num_workers = config['data'].get('num_workers', 4)
    
    # Training configuration
    if 'training' in config:
        if not hasattr(args, 'steps') or args.steps == 10000:
            args.steps = config['training'].get('steps', 10000)
        if not hasattr(args, 'batch_size') or args.batch_size == 1:
            args.batch_size = config['training'].get('batch_size', 1)
        if not hasattr(args, 'grad_accum_steps') or args.grad_accum_steps == 8:
            args.grad_accum_steps = config['training'].get('grad_accum_steps', 8)
        
        if 'optimizer' in config['training']:
            opt = config['training']['optimizer']
            if not hasattr(args, 'lr') or args.lr == 1e-4:
                args.lr = opt.get('lr', 1e-4)
            if not hasattr(args, 'weight_decay') or args.weight_decay == 0.01:
                args.weight_decay = opt.get('weight_decay', 0.01)
        
        if 'scheduler' in config['training']:
            sched = config['training']['scheduler']
            if not hasattr(args, 'warmup_steps') or args.warmup_steps == 100:
                args.warmup_steps = sched.get('warmup_steps', 100)
    
    # Loss configuration
    if 'loss' in config:
        if not hasattr(args, 'l1_weight') or args.l1_weight == 1.0:
            args.l1_weight = config['loss'].get('l1_weight', 1.0)
        if not hasattr(args, 'ms_ssim_weight') or args.ms_ssim_weight == 0.5:
            args.ms_ssim_weight = config['loss'].get('ms_ssim_weight', 0.5)
        if not hasattr(args, 'lpips_weight') or args.lpips_weight == 0.1:
            args.lpips_weight = config['loss'].get('lpips_weight', 0.1)
        if not hasattr(args, 'sar_structure_weight') or args.sar_structure_weight == 0.2:
            args.sar_structure_weight = config['loss'].get('sar_structure_weight', 0.2)
    
    # Validation configuration
    if 'validation' in config:
        if not hasattr(args, 'val_every') or args.val_every == 500:
            args.val_every = config['validation'].get('val_every', 500)
    
    # Checkpointing configuration
    if 'checkpointing' in config:
        if not hasattr(args, 'save_every') or args.save_every == 1000:
            args.save_every = config['checkpointing'].get('save_every', 1000)
    
    # Early stopping configuration
    if 'early_stopping' in config:
        es = config['early_stopping']
        if not hasattr(args, 'early_stopping'):
            args.early_stopping = es.get('enabled', False)
        if not hasattr(args, 'early_stopping_patience') or args.early_stopping_patience == 5:
            args.early_stopping_patience = es.get('patience', 5)
        if not hasattr(args, 'early_stopping_min_delta') or args.early_stopping_min_delta == 1e-4:
            args.early_stopping_min_delta = es.get('min_delta', 1e-4)
    
    # Top-K checkpoints
    if 'top_k_checkpoints' in config:
        if not hasattr(args, 'keep_top_k') or args.keep_top_k == 3:
            args.keep_top_k = config['top_k_checkpoints'].get('k', 3)
    
    # Output configuration
    if 'output' in config:
        if not hasattr(args, 'output_dir') or args.output_dir is None:
            args.output_dir = config['output'].get('output_dir')
    
    # Logging configuration
    if 'logging' in config:
        if not hasattr(args, 'log_file') or args.log_file is None:
            args.log_file = config['logging'].get('log_file')
    
    # System configuration
    if 'system' in config:
        if not hasattr(args, 'device') or args.device == 'cuda':
            args.device = config['system'].get('device', 'cuda')
        if not hasattr(args, 'seed') or args.seed == 42:
            args.seed = config['system'].get('seed', 42)
    
    return args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Stage 1: TerraMind SAR-to-Optical with LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Using config file
  python scripts/train_stage1.py --config configs/train/stage1.lowvr.yaml --data-dir tiles/ --output-dir runs/exp1/
  
  # Quick test run with 100 tiles
  python scripts/train_stage1.py --data-dir tiles/ --output-dir test/ --max-tiles 100 --steps 500
  
  # Full training with custom learning rate
  python scripts/train_stage1.py --data-dir tiles/ --output-dir runs/exp1/ --lr 2e-4 --steps 10000
  
  # Resume from checkpoint
  python scripts/train_stage1.py --data-dir tiles/ --output-dir runs/exp1/ --resume runs/exp1/checkpoints/checkpoint_step_5000.pt
  
  # Fast iteration with small tiles and frequent validation
  python scripts/train_stage1.py --data-dir tiles/ --output-dir debug/ --tile-size 128 --val-every 100 --max-tiles 50
        """
    )
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (optional, CLI args override config)')
    
    # Data
    parser.add_argument('--data-dir', type=str, required=False, default=None,
                        help='Directory containing NPZ tiles')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split fraction (default: 0.1)')
    parser.add_argument('--tile-size', type=int, default=None,
                        help='Tile size (crops/resizes loaded tiles to this size, None = use original size)')
    parser.add_argument('--max-tiles', type=int, default=None,
                        help='Maximum number of tiles to use (None = use all, useful for quick tests)')
    
    # Model
    parser.add_argument('--timesteps', type=int, default=12,
                        help='Number of diffusion timesteps (default: 12)')
    parser.add_argument('--standardize', action='store_true', default=True,
                        help='Apply TerraMind pretraining statistics for standardization (default: True)')
    parser.add_argument('--no-standardize', dest='standardize', action='store_false',
                        help='Disable standardization (use raw data values)')
    parser.add_argument('--lora-rank', type=int, default=8,
                        help='LoRA rank (default: 8)')
    parser.add_argument('--lora-alpha', type=int, default=16,
                        help='LoRA alpha (default: 16)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='[DEPRECATED: use --resume] Path to checkpoint to resume from')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    # Loss weights
    parser.add_argument('--l1-weight', type=float, default=1.0,
                        help='L1 loss weight (default: 1.0)')
    parser.add_argument('--ms-ssim-weight', type=float, default=0.5,
                        help='MS-SSIM loss weight (default: 0.5)')
    parser.add_argument('--lpips-weight', type=float, default=0.1,
                        help='LPIPS loss weight (default: 0.1)')
    parser.add_argument('--color-consistency-weight', type=float, default=0.5,
                        help='Color consistency loss weight (default: 0.5, GAC Stage 1)')
    parser.add_argument('--sar-structure-weight', type=float, default=0.0,
                        help='SAR structure loss weight (default: 0.0, GAC: use Stage 3 only)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: None, use --steps instead)')
    parser.add_argument('--steps', type=int, default=10000,
                        help='Number of training steps (default: 10000, ignored if --epochs is set)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size per GPU (default: 1)')
    parser.add_argument('--grad-accum-steps', type=int, default=8,
                        help='Gradient accumulation steps (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')
    parser.add_argument('--warmup-steps', type=int, default=100,
                        help='Warmup steps (default: 100)')
    
    # Output
    parser.add_argument('--output-dir', type=str, required=False, default=None,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--save-every', type=int, default=1000,
                        help='Save checkpoint every N steps (default: 1000)')
    parser.add_argument('--val-every', type=int, default=500,
                        help='Run validation every N steps (default: 500)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to log file (default: output_dir/training.jsonl)')
    
    # Early stopping
    parser.add_argument('--early-stopping', action='store_true',
                        help='Enable early stopping on SAR-edge agreement plateau')
    parser.add_argument('--early-stopping-patience', type=int, default=5,
                        help='Number of validations without improvement before stopping (default: 5)')
    parser.add_argument('--early-stopping-min-delta', type=float, default=1e-4,
                        help='Minimum change to qualify as improvement (default: 1e-4)')
    parser.add_argument('--keep-top-k', type=int, default=3,
                        help='Keep top-K checkpoints by SAR-edge agreement (default: 3)')
    
    # System
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    if args.config:
        try:
            config = load_config_from_yaml(args.config)
            args = merge_config_with_args(config, args)
            print(f"✓ Loaded configuration from {args.config}")
        except Exception as e:
            print(f"ERROR: Failed to load config from {args.config}: {e}")
            sys.exit(1)
    
    # Validate required arguments
    if args.data_dir is None:
        parser.error("--data-dir is required (either via CLI or config file)")
    if args.output_dir is None:
        parser.error("--output-dir is required (either via CLI or config file)")
    
    return args


def main():
    args = parse_args()
    
    # Handle deprecated --checkpoint flag
    if args.checkpoint is not None:
        warnings.warn("--checkpoint is deprecated, use --resume instead", DeprecationWarning)
        if args.resume is None:
            args.resume = args.checkpoint
    
    # Set random seed for reproducibility
    print(f"Setting random seed: {args.seed}")
    seed_info = set_seed(
        args.seed,
        deterministic=True,  # Enable deterministic operations
        benchmark=False,  # Disable benchmarking for reproducibility
        warn=True  # Show performance warnings
    )
    print(f"  Deterministic mode: {seed_info.get('deterministic', 'N/A')}")
    print(f"  CUDNN benchmark: {seed_info.get('cudnn_benchmark', 'N/A')}")
    if seed_info.get('cuda_available'):
        print(f"  CUDA devices: {seed_info.get('cuda_device_count', 0)}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize logger
    log_file = args.log_file if args.log_file else str(output_dir / 'training.jsonl')
    logger = TrainingLogger(log_file, append=args.resume is not None)
    
    print("\n" + "=" * 80)
    print("Stage 1 Training: TerraMind SAR-to-Optical with LoRA")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Standardization: {'Enabled' if args.standardize else 'Disabled (raw data)'}")
    if args.standardize:
        print(f"  Using TerraMind pretraining statistics")
        print(f"  S1 mean: {TERRAMIND_S1_STATS.means}")
        print(f"  S1 std:  {TERRAMIND_S1_STATS.stds}")
        print(f"  S2 mean: {TERRAMIND_S2_STATS.means}")
        print(f"  S2 std:  {TERRAMIND_S2_STATS.stds}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"Batch size: {args.batch_size}, Grad accum: {args.grad_accum_steps}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}")
    print(f"Learning rate: {args.lr}")
    if args.tile_size:
        print(f"Tile size: {args.tile_size}x{args.tile_size}")
    if args.max_tiles:
        print(f"Max tiles: {args.max_tiles} (quick test mode)")
    print()
    
    # Build model
    print(f"Building TerraMind generator (standardize={args.standardize})...")
    model = build_terramind_generator(
        input_modalities=("S1GRD",),
        output_modalities=("S2L2A",),
        timesteps=args.timesteps,
        standardize=args.standardize,  # Use CLI flag
        pretrained=True
    )
    model = model.to(device)
    print("✓ Model loaded")
    
    if not args.standardize:
        print("⚠️  WARNING: Standardization disabled!")
        print("   Model will receive raw data values")
        print("   This may reduce quality if pretrained weights expect standardized inputs")
    
    # Apply LoRA adapters
    print(f"\nApplying LoRA adapters (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    num_lora = apply_lora_to_linear(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target_modules=['proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'cross_attn']
    )
    print(f"✓ Added {num_lora} LoRA adapters")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Build loss (GAC Stage 1: L1 + MS-SSIM + LPIPS + Color Consistency)
    print("\nBuilding loss function (GAC Stage 1)...")
    criterion = CombinedLoss(
        l1_weight=args.l1_weight,
        ms_ssim_weight=args.ms_ssim_weight,
        lpips_weight=args.lpips_weight,
        color_consistency_weight=args.color_consistency_weight,  # GAC: replaces SAR-structure
        sar_structure_weight=args.sar_structure_weight  # GAC: should be 0.0 for Stage 1
    ).to(device)
    print("✓ Loss function built")
    print(f"  L1: {args.l1_weight}, MS-SSIM: {args.ms_ssim_weight}, LPIPS: {args.lpips_weight}")
    print(f"  Color Consistency: {args.color_consistency_weight} (GAC Stage 1)")
    if args.sar_structure_weight > 0:
        print(f"  ⚠️  SAR-structure: {args.sar_structure_weight} (GAC recommends 0.0 for Stage 1)")
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    full_dataset = TileDataset(
        args.data_dir,
        augment=True,
        tile_size=args.tile_size,
        max_tiles=args.max_tiles
    )
    
    # Split train/val (reproducibly)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    generator = create_reproducible_generator(args.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator  # Ensures reproducible split
    )
    
    print(f"✓ Train: {len(train_dataset)} tiles, Val: {len(val_dataset)} tiles")
    
    # Determine safe num_workers (0 on Windows for reproducibility and reliability)
    # See axs_lib/dataloader_utils.py for detailed explanation of Windows multiprocessing issues
    safe_num_workers = get_recommended_num_workers()
    if args.num_workers != safe_num_workers:
        print(f"\n⚠ Adjusting num_workers from {args.num_workers} to {safe_num_workers} for platform compatibility")
    
    # Create deterministic dataloaders with platform-aware configuration
    # On Windows: automatically sets num_workers=0 to avoid multiprocessing issues:
    #   - 'spawn' method causes serialization overhead and slow startup
    #   - CUDA initialization errors in spawned workers
    #   - Non-deterministic behavior even with proper seeding
    # On Linux/Mac: uses 'fork' method which works reliably with num_workers > 0
    print(f"\nCreating DataLoaders with reproducibility configuration...")
    print_dataloader_config(
        num_workers=safe_num_workers,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=True
    )
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,  # Will be adjusted automatically on Windows
        pin_memory=True,
        seed=args.seed,  # Ensures deterministic shuffling and worker initialization
        force_zero_workers_on_windows=True  # Recommended for reliability
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,  # Will be adjusted automatically on Windows
        pin_memory=True,
        seed=args.seed,  # Ensures deterministic worker initialization
        force_zero_workers_on_windows=True  # Recommended for reliability
    )
    
    # Setup optimizer (only LoRA parameters)
    lora_params = get_lora_parameters(model)
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Determine total training steps
    if args.epochs is not None:
        total_steps = args.epochs * len(train_loader)
        print(f"Training for {args.epochs} epochs = {total_steps} steps")
    else:
        total_steps = args.steps
        print(f"Training for {total_steps} steps")
    
    # Learning rate scheduler (linear warmup + cosine decay)
    def lr_lambda(current_step: int):
        if current_step < args.warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, args.warmup_steps))
        # Cosine decay
        progress = float(current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Gradient scaler for AMP
    scaler = GradScaler()
    
    # Checkpoint manager (regular checkpoints)
    ckpt_manager = CheckpointManager(output_dir / 'checkpoints')
    
    # Top-K checkpoint manager (best by SAR-edge agreement)
    topk_ckpt_manager = TopKCheckpointManager(
        checkpoint_dir=output_dir / 'checkpoints' / 'top_k',
        k=args.keep_top_k,
        mode='min',
        metric_name='sar_edge_agreement'
    )
    
    # Early stopping
    early_stopping_obj = None
    if args.early_stopping:
        early_stopping_obj = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            mode='min',
            verbose=True
        )
        print(f"\nEarly stopping enabled: patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta}")
    
    # Load checkpoint if resuming
    start_step = 0
    if args.resume:
        print(f"\nLoading checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_step = ckpt.get('step', ckpt.get('epoch', 0) * len(train_loader))
        print(f"✓ Resumed from step {start_step}")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    results = train_step_based(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        grad_accum_steps=args.grad_accum_steps,
        timesteps=args.timesteps,
        total_steps=total_steps,
        val_every=args.val_every,
        save_every=args.save_every,
        start_step=start_step,
        val_dataloader=val_loader,
        logger=logger,
        ckpt_manager=ckpt_manager,
        output_dir=output_dir,
        early_stopping=early_stopping_obj,
        topk_ckpt_manager=topk_ckpt_manager,
        seed_info=seed_info
    )
    
    best_val_loss = results.get('best_val_loss', float('inf'))
    best_sar_edge = results.get('best_sar_edge_agreement', float('inf'))
    stopped_early = results.get('stopped_early', False)
    final_step = results.get('final_step', total_steps)
    
    # Close logger
    logger.close()
    
    print("\n" + "=" * 80)
    print("Training complete!")
    if stopped_early:
        print(" (Early Stopping Triggered)")
    print("=" * 80)
    if best_val_loss < float('inf'):
        print(f"Best validation loss: {best_val_loss:.4f}")
    if best_sar_edge < float('inf'):
        print(f"Best SAR-edge agreement: {best_sar_edge:.4f}")
    print(f"Final step: {final_step}/{total_steps}")
    print(f"Checkpoints saved to: {output_dir}")
    if topk_ckpt_manager.top_k:
        print(f"Top-{args.keep_top_k} checkpoints saved to: {output_dir / 'checkpoints' / 'top_k'}")
    print(f"Logs saved to: {log_file}")


if __name__ == '__main__':
    main()
