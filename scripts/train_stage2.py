"""
Stage 2 Training Script: Prithvi Refinement with LoRA

This script trains the Stage 2 Prithvi refinement model to enhance TerraMind's
"mental images" (opt_v1) into refined optical features using:
- Prithvi-EO-2.0-600M backbone with 8-bit quantization
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- ConvNeXt refinement head
- Metadata conditioning (month, biome)
- S2/HLS neighbor targets for supervision

Memory-efficient training:
- Batch size = 1 with gradient accumulation = 8 (effective batch = 8)
- Mixed precision (FP16) with PyTorch AMP
- 8-bit quantization for Prithvi backbone
- LoRA fine-tuning (~1% trainable parameters)

Usage:
    python scripts/train_stage2.py --config configs/hardware.lowvr.yaml \\
        --data_dir data/tiles/benv2_catalog \\
        --stage1_checkpoint weights/stage1_best.pt \\
        --output_dir outputs/stage2

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
import os
from pathlib import Path
import argparse
import yaml
import json
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

try:
    from axs_lib.stage2_prithvi_refine import build_prithvi_refiner, PrithviRefiner
    from axs_lib.stage2_losses import Stage2Loss
    from axs_lib.io import save_checkpoint, load_checkpoint, list_checkpoints
    from axs_lib.lr_scheduler import CosineAnnealingWarmupLR
    from axs_lib.dataloader_utils import create_dataloader, get_recommended_num_workers, print_dataloader_config
    from axs_lib.reproducibility import set_seed
    from axs_lib.metrics import GACScore
    AXSLIB_AVAILABLE = True
except ImportError as e:
    AXSLIB_AVAILABLE = False
    warnings.warn(f"axs_lib modules not available: {e}")


# ============================================================================
# Dataset for Stage 2 Training
# ============================================================================

class Stage2Dataset(Dataset):
    """
    Dataset for Stage 2 Prithvi refinement training.
    
    Loads:
        - opt_v1: Stage 1 TerraMind synthetic optical output (4 channels)
        - metadata: month (1-12), biome_code (categorical)
        - target: Ground truth S2/HLS optical (4 channels)
    
    Expected tile structure (NPZ files):
        - s1_vv, s1_vh: Sentinel-1 SAR (for Stage 1 if needed)
        - s2_b2, s2_b3, s2_b4, s2_b8: Sentinel-2 optical (target)
        - month: Acquisition month (1-12)
        - biome_code: Biome classification (0-16)
    """
    
    def __init__(
        self,
        data_dir: Path,
        split: str = 'train',
        stage1_dir: Optional[Path] = None,
        max_samples: Optional[int] = None,
        tile_size: int = 120,
        use_augmentation: bool = True
    ):
        """
        Args:
            data_dir: Directory containing NPZ tiles
            split: 'train', 'val', or 'test'
            stage1_dir: Directory containing Stage 1 opt_v1 outputs (if pre-computed)
            max_samples: Maximum number of samples to load (for debugging)
            tile_size: Expected tile size
            use_augmentation: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.stage1_dir = Path(stage1_dir) if stage1_dir else None
        self.tile_size = tile_size
        self.use_augmentation = use_augmentation and (split == 'train')
        
        # Find all NPZ tiles
        self.tile_paths = self._find_tiles()
        
        if max_samples and len(self.tile_paths) > max_samples:
            self.tile_paths = self.tile_paths[:max_samples]
        
        print(f"Found {len(self.tile_paths)} tiles for {split} split")
    
    def _find_tiles(self) -> List[Path]:
        """Find all NPZ tiles for the split."""
        # Look for tiles with split in metadata or filename
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
                - opt_v1: Stage 1 synthetic optical (4, H, W)
                - month: Month token (1,)
                - biome: Biome token (1,)
                - target: Ground truth optical (4, H, W)
                - valid_mask: Mask for valid pixels (1, H, W)
        """
        tile_path = self.tile_paths[idx]
        
        # Load tile data
        data = np.load(tile_path)
        
        # Extract S2 bands as target (Blue, Green, Red, NIR)
        target = np.stack([
            data.get('s2_b2', np.zeros((self.tile_size, self.tile_size), dtype=np.float32)),
            data.get('s2_b3', np.zeros((self.tile_size, self.tile_size), dtype=np.float32)),
            data.get('s2_b4', np.zeros((self.tile_size, self.tile_size), dtype=np.float32)),
            data.get('s2_b8', np.zeros((self.tile_size, self.tile_size), dtype=np.float32))
        ], axis=0)  # (4, H, W)
        
        # Load pre-computed Stage 1 output (opt_v1)
        # Stage 2 REQUIRES precomputed Stage 1 outputs - no fallbacks!
        if not self.stage1_dir:
            raise RuntimeError(
                "Stage 2 training requires precomputed Stage 1 outputs. "
                "Please run Stage 1 precompute first with: "
                "python scripts/00_precompute_stage1.py"
            )
        
        opt_v1_path = self.stage1_dir / tile_path.name
        if not opt_v1_path.exists():
            raise FileNotFoundError(
                f"Stage 1 output not found: {opt_v1_path}\n"
                f"Ensure Stage 1 precompute covered all tiles in {self.split} split."
            )
        
        opt_v1_data = np.load(opt_v1_path)
        opt_v1 = opt_v1_data.get('opt_v1', None)
        if opt_v1 is None:
            raise ValueError(f"No 'opt_v1' key found in {opt_v1_path}")
        
        # Extract metadata
        month = data.get('month', 6)  # Default to June
        biome = data.get('biome_code', 0)  # Default to unknown
        
        # Create valid mask (non-NaN pixels in target)
        valid_mask = ~np.isnan(target).any(axis=0, keepdims=True)  # (1, H, W)
        
        # Replace NaN with zeros
        opt_v1 = np.nan_to_num(opt_v1, nan=0.0)
        target = np.nan_to_num(target, nan=0.0)
        
        # Apply augmentation if training
        if self.use_augmentation:
            opt_v1, target, valid_mask = self._augment(opt_v1, target, valid_mask)
        
        # Extract SAR data for GAC-Score computation (not used as model input)
        s1_vv = data.get('s1_vv', np.zeros((self.tile_size, self.tile_size), dtype=np.float32))
        s1_vh = data.get('s1_vh', np.zeros((self.tile_size, self.tile_size), dtype=np.float32))
        sar = np.stack([s1_vv, s1_vh], axis=0)  # (2, H, W)
        sar = np.nan_to_num(sar, nan=0.0)
        
        # Convert to tensors
        sample = {
            'opt_v1': torch.from_numpy(opt_v1).float(),
            'month': torch.tensor([month], dtype=torch.long),
            'biome': torch.tensor([biome], dtype=torch.long),
            'target': torch.from_numpy(target).float(),
            'valid_mask': torch.from_numpy(valid_mask.astype(np.float32)),
            'sar': torch.from_numpy(sar).float()  # For GAC-Score only
        }
        
        return sample
    
    def _augment(
        self,
        opt_v1: np.ndarray,
        target: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply data augmentation.
        
        Args:
            opt_v1: Stage 1 output (4, H, W)
            target: Ground truth (4, H, W)
            valid_mask: Valid pixel mask (1, H, W)
            
        Returns:
            Augmented (opt_v1, target, valid_mask)
        """
        # Random horizontal flip
        if np.random.rand() < 0.5:
            opt_v1 = np.flip(opt_v1, axis=2).copy()
            target = np.flip(target, axis=2).copy()
            valid_mask = np.flip(valid_mask, axis=2).copy()
        
        # Random vertical flip
        if np.random.rand() < 0.5:
            opt_v1 = np.flip(opt_v1, axis=1).copy()
            target = np.flip(target, axis=1).copy()
            valid_mask = np.flip(valid_mask, axis=1).copy()
        
        # Random 90-degree rotations
        k = np.random.randint(0, 4)
        if k > 0:
            opt_v1 = np.rot90(opt_v1, k=k, axes=(1, 2)).copy()
            target = np.rot90(target, k=k, axes=(1, 2)).copy()
            valid_mask = np.rot90(valid_mask, k=k, axes=(1, 2)).copy()
        
        return opt_v1, target, valid_mask


# ============================================================================
# Metadata Embedding Module
# ============================================================================

class MetadataEmbedding(nn.Module):
    """
    Embeds metadata tokens (month, biome) into channel-wise feature maps.
    
    Architecture:
        - Month embedding: Sinusoidal positional encoding (12 months)
        - Biome embedding: Learnable embedding table (16 biomes)
        - Projection to spatial feature maps via 1x1 conv
    """
    
    def __init__(
        self,
        num_months: int = 12,
        num_biomes: int = 16,
        embed_dim: int = 64,
        out_channels: int = 4
    ):
        """
        Args:
            num_months: Number of months (12)
            num_biomes: Number of biome categories (16)
            embed_dim: Embedding dimension
            out_channels: Number of output channels to concatenate
        """
        super().__init__()
        
        self.num_months = num_months
        self.num_biomes = num_biomes
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        
        # Month embedding (sinusoidal)
        # We'll compute this on the fly
        
        # Biome embedding (learnable)
        self.biome_embed = nn.Embedding(num_biomes, embed_dim)
        
        # Project embeddings to channel space
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, out_channels)
        )
    
    def forward(
        self,
        month: torch.Tensor,
        biome: torch.Tensor,
        spatial_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Embed metadata and broadcast to spatial dimensions.
        
        Args:
            month: Month indices (B,) or (B, 1)
            biome: Biome indices (B,) or (B, 1)
            spatial_shape: (H, W) for broadcasting
            
        Returns:
            Metadata feature maps (B, out_channels, H, W)
        """
        B = month.shape[0]
        H, W = spatial_shape
        
        # Flatten if needed
        if month.ndim > 1:
            month = month.squeeze(-1)
        if biome.ndim > 1:
            biome = biome.squeeze(-1)
        
        # Month sinusoidal encoding
        month_norm = month.float() / self.num_months  # Normalize to [0, 1]
        month_embed = self._sinusoidal_encoding(month_norm, self.embed_dim)  # (B, embed_dim)
        
        # Biome learnable embedding
        biome_embed = self.biome_embed(biome)  # (B, embed_dim)
        
        # Concatenate embeddings
        combined = torch.cat([month_embed, biome_embed], dim=1)  # (B, embed_dim*2)
        
        # Project to channel space
        features = self.proj(combined)  # (B, out_channels)
        
        # Broadcast to spatial dimensions
        features = features.unsqueeze(-1).unsqueeze(-1)  # (B, out_channels, 1, 1)
        features = features.expand(B, self.out_channels, H, W)  # (B, out_channels, H, W)
        
        return features
    
    def _sinusoidal_encoding(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Compute sinusoidal positional encoding.
        
        Args:
            x: Normalized values (B,) in [0, 1]
            dim: Embedding dimension
            
        Returns:
            Sinusoidal embeddings (B, dim)
        """
        B = x.shape[0]
        device = x.device
        
        # Create frequency bands
        freqs = torch.arange(0, dim // 2, device=device).float()
        freqs = 1.0 / (10000 ** (2 * freqs / dim))
        
        # Compute sine and cosine
        x = x.unsqueeze(-1)  # (B, 1)
        args = x * freqs.unsqueeze(0)  # (B, dim//2)
        
        sin_vals = torch.sin(2 * np.pi * args)
        cos_vals = torch.cos(2 * np.pi * args)
        
        # Interleave sine and cosine
        encoding = torch.stack([sin_vals, cos_vals], dim=2).flatten(1)  # (B, dim)
        
        return encoding


# ============================================================================
# Stage 2 Model Wrapper with Metadata
# ============================================================================

class Stage2ModelWithMetadata(nn.Module):
    """
    Wraps PrithviRefiner with metadata conditioning.
    
    Architecture:
        opt_v1 (4 ch) + metadata features (4 ch) → PrithviRefiner → refined (4 ch)
    """
    
    def __init__(
        self,
        prithvi_refiner: PrithviRefiner,
        num_months: int = 12,
        num_biomes: int = 16,
        embed_dim: int = 64
    ):
        super().__init__()
        
        self.prithvi_refiner = prithvi_refiner
        self.metadata_embed = MetadataEmbedding(
            num_months=num_months,
            num_biomes=num_biomes,
            embed_dim=embed_dim,
            out_channels=4
        )
        
        # Input projection: 8 channels (4 opt_v1 + 4 metadata) → 4 channels
        self.input_fusion = nn.Conv2d(8, 4, kernel_size=1)
    
    def forward(
        self,
        opt_v1: torch.Tensor,
        month: torch.Tensor,
        biome: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with metadata conditioning.
        
        Args:
            opt_v1: Stage 1 output (B, 4, H, W)
            month: Month indices (B,) or (B, 1)
            biome: Biome indices (B,) or (B, 1)
            
        Returns:
            Refined optical (B, 4, H, W)
        """
        B, C, H, W = opt_v1.shape
        
        # Embed metadata
        metadata_features = self.metadata_embed(month, biome, (H, W))  # (B, 4, H, W)
        
        # Concatenate opt_v1 with metadata features
        fused_input = torch.cat([opt_v1, metadata_features], dim=1)  # (B, 8, H, W)
        
        # Fuse to 4 channels
        fused_input = self.input_fusion(fused_input)  # (B, 4, H, W)
        
        # Pass through Prithvi refiner
        refined = self.prithvi_refiner(fused_input)  # (B, 4, H, W)
        
        return refined


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: Stage2Loss,
    optimizer: torch.optim.Optimizer,
    disc_optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    grad_accum_steps: int = 8,
    max_grad_norm: float = 1.0,
    disc_update_freq: int = 5,
    log_cpu_fallbacks: bool = True
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Stage2ModelWithMetadata
        train_loader: Training data loader
        criterion: Stage2Loss
        optimizer: Generator optimizer
        disc_optimizer: Discriminator optimizer (optional)
        scheduler: LR scheduler (optional, stepped per grad accumulation)
        scaler: GradScaler for AMP
        device: Device to train on
        epoch: Current epoch number
        grad_accum_steps: Gradient accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        disc_update_freq: Update discriminator every N batches
        log_cpu_fallbacks: Whether to track CPU fallback warnings
        
    Returns:
        Dict of average losses (includes 'cpu_fallback_count' if any occurred)
    """
    model.train()
    
    running_losses = {}
    num_batches = 0
    cpu_fallback_count = 0
    
    # Track warnings for CPU fallbacks
    import warnings
    original_showwarning = warnings.showwarning
    
    def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
        nonlocal cpu_fallback_count
        if log_cpu_fallbacks and "CPU fallback" in str(message):
            cpu_fallback_count += 1
        # Call original handler
        original_showwarning(message, category, filename, lineno, file, line)
    
    warnings.showwarning = custom_warning_handler
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        opt_v1 = batch['opt_v1'].to(device)
        month = batch['month'].to(device)
        biome = batch['biome'].to(device)
        target = batch['target'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        
        # ====================================================================
        # Generator Step
        # ====================================================================
        
        # Forward pass with AMP
        with autocast():
            pred = model(opt_v1, month, biome)
            
            # Mask out invalid pixels in loss
            pred_masked = pred * valid_mask
            target_masked = target * valid_mask
            
            # Compute generator loss
            losses = criterion(pred_masked, target_masked, mode='generator')
            gen_loss = losses['loss'] / grad_accum_steps
        
        # Backward with gradient scaling
        scaler.scale(gen_loss).backward()
        
        # Accumulate losses for logging
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            running_losses[k] = running_losses.get(k, 0) + v
        
        # Gradient accumulation: update every N steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Step scheduler (per accumulated step, not per batch)
            if scheduler is not None:
                scheduler.step()
        
        # ====================================================================
        # Discriminator Step (optional, every N batches)
        # ====================================================================
        
        if disc_optimizer and (batch_idx % disc_update_freq == 0):
            with autocast():
                pred_detached = model(opt_v1, month, biome).detach()
                disc_losses = criterion(pred_detached, target_masked, mode='discriminator')
                disc_loss = disc_losses['loss']
            
            disc_optimizer.zero_grad()
            scaler.scale(disc_loss).backward()
            scaler.step(disc_optimizer)
            scaler.update()
            
            running_losses['disc_loss'] = running_losses.get('disc_loss', 0) + disc_loss.item()
        
        num_batches += 1
        
        # Update progress bar
        if num_batches % 10 == 0:
            avg_loss = running_losses.get('loss', 0) / num_batches
            postfix = {'loss': f'{avg_loss:.4f}'}
            if scheduler is not None:
                postfix['lr'] = f"{scheduler.get_last_lr()[0]:.2e}"
            if cpu_fallback_count > 0:
                postfix['cpu_fb'] = cpu_fallback_count
            pbar.set_postfix(postfix)
    
    # Restore original warning handler
    warnings.showwarning = original_showwarning
    
    # Compute averages
    avg_losses = {k: v / num_batches for k, v in running_losses.items()}
    
    # Add CPU fallback count
    if cpu_fallback_count > 0:
        avg_losses['cpu_fallback_count'] = cpu_fallback_count
        print(f"\n⚠️  CPU fallback occurred {cpu_fallback_count} times this epoch")
        print("   Consider reducing memory usage to avoid performance penalty.")
    
    return avg_losses


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: Stage2Loss,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate model.
    
    Args:
        model: Stage2ModelWithMetadata
        val_loader: Validation data loader
        criterion: Stage2Loss
        device: Device
        
    Returns:
        Dict of average validation losses and GAC-Score
    """
    model.eval()
    
    running_losses = {}
    num_batches = 0
    
    # Initialize GAC-Score metric
    gac_metric = GACScore(device=device)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            opt_v1 = batch['opt_v1'].to(device)
            month = batch['month'].to(device)
            biome = batch['biome'].to(device)
            target = batch['target'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            sar = batch.get('sar', None)
            if sar is not None:
                sar = sar.to(device)
            
            # Forward pass
            with autocast():
                pred = model(opt_v1, month, biome)
                
                # Mask out invalid pixels
                pred_masked = pred * valid_mask
                target_masked = target * valid_mask
                
                # Compute loss
                losses = criterion(pred_masked, target_masked, mode='generator')
            
            # Update GAC-Score if SAR is available
            if sar is not None:
                # GAC-Score expects images in [0, 1] range, already normalized in dataset
                gac_metric.update(pred, target, sar)
            
            # Accumulate losses
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                running_losses[k] = running_losses.get(k, 0) + v
            
            num_batches += 1
    
    # Compute averages
    avg_losses = {k: v / num_batches for k, v in running_losses.items()}
    
    # Compute GAC-Score
    gac_score = gac_metric.compute()
    if gac_score is not None:
        avg_losses['gac_score'] = gac_score
    
    return avg_losses


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 2 Prithvi Refinement Training")
    parser.add_argument('--config', type=str, default='configs/hardware.lowvr.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing training tiles')
    parser.add_argument('--stage1_dir', type=str, default=None,
                        help='Directory containing pre-computed Stage 1 outputs')
    parser.add_argument('--output_dir', type=str, default='outputs/stage2',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples per split (for debugging)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--grad_accum', type=int, default=None,
                        help='Gradient accumulation steps (overrides config)')
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help='Number of warmup steps for LR scheduler (default: 10%% of total steps)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Override config with CLI args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.grad_accum:
        config['training']['grad_accum_steps'] = args.grad_accum
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Save config to output dir
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Set seed for reproducibility
    seed = config.get('seed', 42)
    print(f"\nSetting random seed: {seed}")
    set_seed(seed, deterministic=True, benchmark=False)
    
    # ========================================================================
    # Create Datasets
    # ========================================================================
    
    print("\nCreating datasets...")
    
    train_dataset = Stage2Dataset(
        data_dir=args.data_dir,
        split='train',
        stage1_dir=args.stage1_dir,
        max_samples=args.max_samples,
        tile_size=config['data'].get('tile_size', 120),
        use_augmentation=True
    )
    
    val_dataset = Stage2Dataset(
        data_dir=args.data_dir,
        split='val',
        stage1_dir=args.stage1_dir,
        max_samples=args.max_samples // 4 if args.max_samples else None,
        tile_size=config['data'].get('tile_size', 120),
        use_augmentation=False
    )
    
    # Create deterministic dataloaders with platform-aware configuration
    # On Windows: automatically sets num_workers=0 to avoid multiprocessing issues:
    #   - Windows uses 'spawn' method for multiprocessing (vs 'fork' on Linux/Mac)
    #   - 'spawn' causes: serialization overhead, CUDA initialization errors,
    #     non-deterministic behavior, and potential crashes
    #   - Single-threaded loading (num_workers=0) is more reliable and still fast
    # On Linux/Mac: uses 'fork' method which works well with num_workers > 0
    # See axs_lib/dataloader_utils.py for detailed explanation
    
    requested_workers = config['data'].get('num_workers', 2)
    safe_workers = get_recommended_num_workers()
    
    print(f"\nDataLoader Configuration:")
    print(f"  Requested workers: {requested_workers}")
    print(f"  Platform-safe workers: {safe_workers}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Seed: {seed}")
    
    if requested_workers != safe_workers:
        print(f"  ⚠ Adjusting num_workers for platform compatibility")
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=requested_workers,  # Will be adjusted automatically on Windows
        pin_memory=config['data'].get('pin_memory', True),
        seed=seed,  # Ensures deterministic shuffling and worker initialization
        force_zero_workers_on_windows=True  # Recommended for reliability
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=requested_workers,  # Will be adjusted automatically on Windows
        pin_memory=config['data'].get('pin_memory', True),
        seed=seed,  # Ensures deterministic worker initialization
        force_zero_workers_on_windows=True  # Recommended for reliability
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # ========================================================================
    # Build Model
    # ========================================================================
    
    print("\nBuilding model...")
    
    prithvi_refiner = build_prithvi_refiner(
        config=config,
        device=device
    )
    
    model = Stage2ModelWithMetadata(
        prithvi_refiner=prithvi_refiner,
        num_months=12,
        num_biomes=16,
        embed_dim=64
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # ========================================================================
    # Create Loss and Optimizers
    # ========================================================================
    
    print("\nSetting up training...")
    
    criterion = Stage2Loss(
        spectral_weight=1.0,
        identity_weight=0.5,
        adversarial_weight=0.05
    ).to(device)
    
    # Generator optimizer (model parameters)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['training']['learning_rate'],
        betas=config['training']['optimizer']['betas'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    # Discriminator optimizer (for adversarial loss)
    disc_optimizer = torch.optim.AdamW(
        criterion.adversarial_loss.discriminator.parameters(),
        lr=config['training']['learning_rate'] * 0.1,  # Lower LR for discriminator
        betas=config['training']['optimizer']['betas'],
        weight_decay=0.0
    )
    
    # Learning rate scheduler with warmup
    # Calculate total training steps
    steps_per_epoch = len(train_loader) // config['training']['grad_accum_steps']
    total_steps = steps_per_epoch * config['training']['epochs']
    
    # Determine warmup steps
    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    elif 'warmup_steps' in config['training'].get('lr_scheduler', {}):
        warmup_steps = config['training']['lr_scheduler']['warmup_steps']
    else:
        # Default: 10% of total steps
        warmup_steps = int(0.1 * total_steps)
    
    print(f"\nLearning rate schedule:")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps} ({100 * warmup_steps / total_steps:.1f}%)")
    print(f"  Base LR: {config['training']['learning_rate']:.2e}")
    print(f"  Min LR: {config['training']['lr_scheduler']['min_lr']:.2e}")
    
    scheduler = CosineAnnealingWarmupLR(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        eta_min=config['training']['lr_scheduler']['min_lr']
    )
    
    # Gradient scaler for AMP
    scaler = GradScaler()
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume and Path(args.resume).exists():
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume, device=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}")
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    
    print(f"\nStarting training for {config['training']['epochs']} epochs...\n")
    
    log_file = output_dir / 'training_log.jsonl'
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        print("=" * 70)
        
        # Train
        train_losses = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            disc_optimizer=disc_optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch + 1,
            grad_accum_steps=config['training']['grad_accum_steps'],
            max_grad_norm=config['training']['max_grad_norm']
        )
        
        # Log training losses
        print(f"\nTrain - Loss: {train_losses['loss']:.6f}")
        print(f"  Spectral: {train_losses.get('spectral_loss', 0):.6f}")
        print(f"  Identity: {train_losses.get('identity_loss', 0):.6f}")
        print(f"  Adversarial: {train_losses.get('adversarial_loss', 0):.6f}")
        
        # Validate
        if (epoch + 1) % config['training']['validation_interval'] == 0:
            val_losses = validate(model, val_loader, criterion, device)
            
            print(f"\nVal - Loss: {val_losses['loss']:.6f}")
            print(f"  Spectral: {val_losses.get('spectral_loss', 0):.6f}")
            print(f"  Identity: {val_losses.get('identity_loss', 0):.6f}")
            if 'gac_score' in val_losses:
                print(f"  GAC-Score: {val_losses['gac_score']:.4f}")
            
            # Save best model
            if val_losses['loss'] < best_val_loss:
                best_val_loss = val_losses['loss']
                save_checkpoint(
                    output_dir / 'best.pt',
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={'val_loss': val_losses['loss']},
                    config=config
                )
                print(f"✓ Saved best model (val_loss: {best_val_loss:.6f})")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            save_checkpoint(
                output_dir / f'checkpoint_epoch_{epoch+1:03d}.pt',
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={'train_loss': train_losses['loss']},
                config=config
            )
            print(f"✓ Saved checkpoint")
        
        # Log to file
        log_entry = {
            'epoch': epoch + 1,
            'train': train_losses,
            'val': val_losses if (epoch + 1) % config['training']['validation_interval'] == 0 else {},
            'lr': scheduler.get_last_lr()[0],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
