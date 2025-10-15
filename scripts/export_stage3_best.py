"""
Export Best Stage 3 Checkpoint

Selects the best Stage 3 checkpoint from training runs based on validation metrics:
- SAR edge agreement (primary metric)
- Composite score (perceptual quality + SAR alignment)
- Loss metrics

The script:
1. Scans checkpoint directory for all Stage 3 checkpoints
2. Loads validation metrics from training logs or validation runs
3. Computes composite scores
4. Selects best checkpoint based on criteria
5. Exports to a standardized location

Composite Score Formula:
    score = (0.6 * sar_agreement) - (0.2 * normalized_loss) + (0.2 * perceptual_score)

Usage:
    # Select by SAR agreement (default)
    python scripts/export_stage3_best.py \
        --checkpoint-dir checkpoints/stage3/ \
        --output best_stage3.pt

    # Select by composite score
    python scripts/export_stage3_best.py \
        --checkpoint-dir checkpoints/stage3/ \
        --output best_stage3.pt \
        --criterion composite

    # With validation data
    python scripts/export_stage3_best.py \
        --checkpoint-dir checkpoints/stage3/ \
        --validation-dir validation/stage3/ \
        --output best_stage3.pt \
        --recompute

    # Export top N checkpoints
    python scripts/export_stage3_best.py \
        --checkpoint-dir checkpoints/stage3/ \
        --output-dir exported_models/ \
        --top-n 3

Author: Axion-Sat Project
Version: 1.0.0
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
from dataclasses import dataclass, asdict

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CheckpointMetrics:
    """Metrics for a single checkpoint."""
    checkpoint_path: str
    epoch: int
    
    # Training metrics
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    
    # SAR alignment metrics
    sar_agreement: Optional[float] = None
    sar_edge_correlation: Optional[float] = None
    
    # Perceptual metrics
    lpips_mean: Optional[float] = None
    lpips_std: Optional[float] = None
    
    # Quality metrics
    psnr_mean: Optional[float] = None
    ssim_mean: Optional[float] = None
    
    # Composite scores
    composite_score: Optional[float] = None
    weighted_score: Optional[float] = None
    
    # Metadata
    timestamp: Optional[str] = None
    config: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def compute_composite_score(
        self,
        sar_weight: float = 0.6,
        loss_weight: float = 0.2,
        perceptual_weight: float = 0.2
    ) -> float:
        """
        Compute composite score from available metrics.
        
        Args:
            sar_weight: Weight for SAR agreement (default: 0.6)
            loss_weight: Weight for loss term (default: 0.2)
            perceptual_weight: Weight for perceptual quality (default: 0.2)
            
        Returns:
            Composite score (higher is better)
        """
        score = 0.0
        total_weight = 0.0
        
        # SAR agreement component (higher is better)
        if self.sar_agreement is not None:
            score += sar_weight * self.sar_agreement
            total_weight += sar_weight
        
        # Loss component (lower is better, so negate and normalize)
        if self.val_loss is not None:
            # Assume loss is in reasonable range [0, 1]
            # Negate so lower loss = higher score
            normalized_loss = max(0, 1 - self.val_loss)
            score += loss_weight * normalized_loss
            total_weight += loss_weight
        
        # Perceptual component (lower LPIPS is better)
        if self.lpips_mean is not None:
            # LPIPS typically in [0, 1], lower is better
            perceptual_score = max(0, 1 - self.lpips_mean)
            score += perceptual_weight * perceptual_score
            total_weight += perceptual_weight
        
        # Normalize by actual weights used
        if total_weight > 0:
            score = score / total_weight
        
        self.composite_score = score
        return score


# ============================================================================
# Checkpoint Discovery
# ============================================================================

def find_checkpoints(checkpoint_dir: Path, pattern: str = "*.pt") -> List[Path]:
    """
    Find all checkpoint files in directory.
    
    Args:
        checkpoint_dir: Directory to search
        pattern: File pattern to match
        
    Returns:
        List of checkpoint paths
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    # Find all checkpoints
    checkpoints = list(checkpoint_dir.rglob(pattern))
    
    # Filter out best/latest symlinks to avoid duplicates
    checkpoints = [
        p for p in checkpoints
        if 'best' not in p.stem.lower() and 'latest' not in p.stem.lower() and 'exported' not in p.stem.lower()
    ]
    
    return sorted(checkpoints)


# ============================================================================
# Metrics Loading
# ============================================================================

def load_checkpoint_basic_info(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Load basic info from checkpoint without loading full model.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dict with basic checkpoint info
    """
    try:
        # Load checkpoint to CPU
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'epoch': checkpoint.get('epoch', -1),
            'train_loss': checkpoint.get('train_loss'),
            'val_loss': checkpoint.get('val_loss'),
            'config': checkpoint.get('config', {}),
            'timestamp': checkpoint.get('timestamp'),
        }
        
        # Extract metrics if present
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            info.update({
                'sar_agreement': metrics.get('sar_agreement'),
                'sar_edge_correlation': metrics.get('sar_edge_correlation'),
                'lpips_mean': metrics.get('lpips_mean'),
                'lpips_std': metrics.get('lpips_std'),
                'psnr_mean': metrics.get('psnr_mean'),
                'ssim_mean': metrics.get('ssim_mean'),
            })
        
        return info
        
    except Exception as e:
        warnings.warn(f"Error loading checkpoint {checkpoint_path}: {e}")
        return {}


def load_validation_metrics(validation_dir: Path, checkpoint_name: str) -> Dict[str, Any]:
    """
    Load validation metrics for a checkpoint from validation results.
    
    Args:
        validation_dir: Directory containing validation results
        checkpoint_name: Name of checkpoint (e.g., "epoch_010")
        
    Returns:
        Dict with validation metrics
    """
    validation_dir = Path(validation_dir)
    
    if not validation_dir.exists():
        return {}
    
    # Try to find matching validation results
    # Look for JSON files with checkpoint name or epoch number
    epoch_num = None
    if 'epoch' in checkpoint_name:
        try:
            epoch_num = int(checkpoint_name.split('epoch')[-1].split('_')[0].split('.')[0])
        except:
            pass
    
    # Search for validation results
    possible_names = [
        f"{checkpoint_name}_metrics.json",
        f"{checkpoint_name}.json",
        f"epoch_{epoch_num:03d}_metrics.json" if epoch_num else None,
        f"validation_epoch_{epoch_num}.json" if epoch_num else None,
    ]
    
    for name in possible_names:
        if name is None:
            continue
        
        json_path = validation_dir / name
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                warnings.warn(f"Error loading validation metrics from {json_path}: {e}")
    
    return {}


def extract_metrics_from_checkpoint(
    checkpoint_path: Path,
    validation_dir: Optional[Path] = None
) -> CheckpointMetrics:
    """
    Extract all available metrics for a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        validation_dir: Optional directory with validation results
        
    Returns:
        CheckpointMetrics object
    """
    # Load basic info from checkpoint
    info = load_checkpoint_basic_info(checkpoint_path)
    
    # Load validation metrics if available
    if validation_dir:
        val_metrics = load_validation_metrics(validation_dir, checkpoint_path.stem)
        
        # Merge validation metrics
        if 'aggregate_stats' in val_metrics:
            stats = val_metrics['aggregate_stats']
            info.update({
                'sar_agreement': stats.get('sar_agreement_mean'),
                'lpips_mean': stats.get('lpips_mean'),
                'lpips_std': stats.get('lpips_std'),
                'psnr_mean': stats.get('psnr_mean'),
                'ssim_mean': stats.get('ssim_mean'),
            })
        
        # Direct metrics
        if 'sar_agreement' in val_metrics:
            info['sar_agreement'] = val_metrics['sar_agreement']
    
    # Create CheckpointMetrics object
    metrics = CheckpointMetrics(
        checkpoint_path=str(checkpoint_path),
        epoch=info.get('epoch', -1),
        train_loss=info.get('train_loss'),
        val_loss=info.get('val_loss'),
        sar_agreement=info.get('sar_agreement'),
        sar_edge_correlation=info.get('sar_edge_correlation'),
        lpips_mean=info.get('lpips_mean'),
        lpips_std=info.get('lpips_std'),
        psnr_mean=info.get('psnr_mean'),
        ssim_mean=info.get('ssim_mean'),
        timestamp=info.get('timestamp'),
        config=info.get('config'),
    )
    
    return metrics


# ============================================================================
# Checkpoint Selection
# ============================================================================

def select_best_checkpoint(
    checkpoints_metrics: List[CheckpointMetrics],
    criterion: str = 'sar_agreement',
    sar_weight: float = 0.6,
    loss_weight: float = 0.2,
    perceptual_weight: float = 0.2,
    verbose: bool = True
) -> Optional[CheckpointMetrics]:
    """
    Select best checkpoint based on criterion.
    
    Args:
        checkpoints_metrics: List of checkpoint metrics
        criterion: Selection criterion ('sar_agreement', 'composite', 'loss')
        sar_weight: Weight for SAR in composite score
        loss_weight: Weight for loss in composite score
        perceptual_weight: Weight for perceptual quality in composite score
        verbose: Print selection details
        
    Returns:
        Best checkpoint metrics or None if no valid checkpoints
    """
    if len(checkpoints_metrics) == 0:
        return None
    
    # Filter checkpoints with required metric
    valid_checkpoints = []
    
    if criterion == 'sar_agreement':
        valid_checkpoints = [c for c in checkpoints_metrics if c.sar_agreement is not None]
        if len(valid_checkpoints) == 0:
            warnings.warn("No checkpoints with SAR agreement metric. Falling back to loss.")
            criterion = 'loss'
    
    if criterion == 'composite':
        # Compute composite scores
        for ckpt in checkpoints_metrics:
            ckpt.compute_composite_score(sar_weight, loss_weight, perceptual_weight)
        
        valid_checkpoints = [c for c in checkpoints_metrics if c.composite_score is not None]
        
        if len(valid_checkpoints) == 0:
            warnings.warn("No checkpoints with composite score. Falling back to SAR agreement.")
            criterion = 'sar_agreement'
            valid_checkpoints = [c for c in checkpoints_metrics if c.sar_agreement is not None]
    
    if criterion == 'loss':
        valid_checkpoints = [c for c in checkpoints_metrics if c.val_loss is not None]
        if len(valid_checkpoints) == 0:
            warnings.warn("No checkpoints with validation loss. Using all checkpoints.")
            valid_checkpoints = checkpoints_metrics
    
    if len(valid_checkpoints) == 0:
        return None
    
    # Select best based on criterion
    if criterion == 'sar_agreement':
        best = max(valid_checkpoints, key=lambda c: c.sar_agreement)
        if verbose:
            print(f"\nSelected by SAR Agreement: {best.sar_agreement:.6f}")
    
    elif criterion == 'composite':
        best = max(valid_checkpoints, key=lambda c: c.composite_score)
        if verbose:
            print(f"\nSelected by Composite Score: {best.composite_score:.6f}")
            print(f"  SAR Agreement: {best.sar_agreement:.6f}" if best.sar_agreement else "")
            print(f"  Val Loss: {best.val_loss:.6f}" if best.val_loss else "")
            print(f"  LPIPS: {best.lpips_mean:.6f}" if best.lpips_mean else "")
    
    elif criterion == 'loss':
        best = min(valid_checkpoints, key=lambda c: c.val_loss if c.val_loss is not None else float('inf'))
        if verbose:
            print(f"\nSelected by Validation Loss: {best.val_loss:.6f}")
    
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    return best


def select_top_n_checkpoints(
    checkpoints_metrics: List[CheckpointMetrics],
    n: int,
    criterion: str = 'sar_agreement',
    **kwargs
) -> List[CheckpointMetrics]:
    """
    Select top N checkpoints.
    
    Args:
        checkpoints_metrics: List of checkpoint metrics
        n: Number of checkpoints to select
        criterion: Selection criterion
        **kwargs: Additional arguments for selection
        
    Returns:
        List of top N checkpoint metrics
    """
    if len(checkpoints_metrics) == 0:
        return []
    
    # Filter valid checkpoints
    if criterion == 'sar_agreement':
        valid = [c for c in checkpoints_metrics if c.sar_agreement is not None]
        sorted_checkpoints = sorted(valid, key=lambda c: c.sar_agreement, reverse=True)
    
    elif criterion == 'composite':
        # Compute composite scores
        sar_weight = kwargs.get('sar_weight', 0.6)
        loss_weight = kwargs.get('loss_weight', 0.2)
        perceptual_weight = kwargs.get('perceptual_weight', 0.2)
        
        for ckpt in checkpoints_metrics:
            ckpt.compute_composite_score(sar_weight, loss_weight, perceptual_weight)
        
        valid = [c for c in checkpoints_metrics if c.composite_score is not None]
        sorted_checkpoints = sorted(valid, key=lambda c: c.composite_score, reverse=True)
    
    elif criterion == 'loss':
        valid = [c for c in checkpoints_metrics if c.val_loss is not None]
        sorted_checkpoints = sorted(valid, key=lambda c: c.val_loss)
    
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    return sorted_checkpoints[:n]


# ============================================================================
# Export Functions
# ============================================================================

def export_checkpoint(
    source_path: Path,
    output_path: Path,
    metrics: CheckpointMetrics,
    include_metadata: bool = True
) -> Path:
    """
    Export checkpoint to output location with metadata.
    
    Args:
        source_path: Source checkpoint path
        output_path: Output path for exported checkpoint
        metrics: Checkpoint metrics
        include_metadata: Include metrics in exported checkpoint
        
    Returns:
        Path to exported checkpoint
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if include_metadata:
        # Load checkpoint
        checkpoint = torch.load(source_path, map_location='cpu')
        
        # Add/update metrics
        checkpoint['exported_metrics'] = metrics.to_dict()
        checkpoint['export_criterion'] = 'best'
        
        # Save with metadata
        torch.save(checkpoint, output_path)
    else:
        # Simple copy
        shutil.copy2(source_path, output_path)
    
    return output_path


def save_selection_report(
    output_path: Path,
    best_checkpoint: CheckpointMetrics,
    all_checkpoints: List[CheckpointMetrics],
    criterion: str
):
    """
    Save detailed report about checkpoint selection.
    
    Args:
        output_path: Path for report JSON
        best_checkpoint: Selected best checkpoint
        all_checkpoints: All evaluated checkpoints
        criterion: Selection criterion used
    """
    report = {
        'selection_criterion': criterion,
        'best_checkpoint': best_checkpoint.to_dict(),
        'total_checkpoints_evaluated': len(all_checkpoints),
        'all_checkpoints': [c.to_dict() for c in all_checkpoints],
    }
    
    # Add ranking
    if criterion == 'sar_agreement':
        ranked = sorted(
            [c for c in all_checkpoints if c.sar_agreement is not None],
            key=lambda c: c.sar_agreement,
            reverse=True
        )
        report['ranking'] = [
            {
                'rank': i + 1,
                'checkpoint': c.checkpoint_path,
                'epoch': c.epoch,
                'sar_agreement': c.sar_agreement
            }
            for i, c in enumerate(ranked[:10])  # Top 10
        ]
    
    elif criterion == 'composite':
        ranked = sorted(
            [c for c in all_checkpoints if c.composite_score is not None],
            key=lambda c: c.composite_score,
            reverse=True
        )
        report['ranking'] = [
            {
                'rank': i + 1,
                'checkpoint': c.checkpoint_path,
                'epoch': c.epoch,
                'composite_score': c.composite_score,
                'sar_agreement': c.sar_agreement,
                'val_loss': c.val_loss
            }
            for i, c in enumerate(ranked[:10])
        ]
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export best Stage 3 checkpoint based on validation metrics"
    )
    
    # Required arguments
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Directory containing Stage 3 checkpoints')
    
    # Output options
    parser.add_argument('--output', type=str,
                       help='Output path for best checkpoint (single file)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for multiple checkpoints')
    
    # Selection criteria
    parser.add_argument('--criterion', type=str, default='sar_agreement',
                       choices=['sar_agreement', 'composite', 'loss'],
                       help='Selection criterion (default: sar_agreement)')
    parser.add_argument('--top-n', type=int, default=1,
                       help='Number of top checkpoints to export (default: 1)')
    
    # Composite score weights
    parser.add_argument('--sar-weight', type=float, default=0.6,
                       help='SAR agreement weight in composite score (default: 0.6)')
    parser.add_argument('--loss-weight', type=float, default=0.2,
                       help='Loss weight in composite score (default: 0.2)')
    parser.add_argument('--perceptual-weight', type=float, default=0.2,
                       help='Perceptual quality weight in composite score (default: 0.2)')
    
    # Validation options
    parser.add_argument('--validation-dir', type=str,
                       help='Directory with validation results')
    parser.add_argument('--recompute', action='store_true',
                       help='Recompute metrics from validation data')
    
    # Output options
    parser.add_argument('--no-metadata', action='store_true',
                       help='Do not include metrics in exported checkpoint')
    parser.add_argument('--save-report', action='store_true', default=True,
                       help='Save selection report (default: True)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Validate arguments
    if not args.output and not args.output_dir:
        parser.error("Either --output or --output-dir must be specified")
    
    if args.output and args.top_n > 1:
        parser.error("Cannot use --output with --top-n > 1. Use --output-dir instead.")
    
    # Print header
    if verbose:
        print("\n" + "=" * 80)
        print("Export Best Stage 3 Checkpoint")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Checkpoint directory: {args.checkpoint_dir}")
        print(f"  Selection criterion: {args.criterion}")
        print(f"  Top N: {args.top_n}")
        if args.criterion == 'composite':
            print(f"  SAR weight: {args.sar_weight}")
            print(f"  Loss weight: {args.loss_weight}")
            print(f"  Perceptual weight: {args.perceptual_weight}")
        print()
    
    # Find checkpoints
    if verbose:
        print("Finding checkpoints...")
    
    checkpoint_paths = find_checkpoints(Path(args.checkpoint_dir))
    
    if len(checkpoint_paths) == 0:
        print(f"❌ No checkpoints found in {args.checkpoint_dir}")
        return 1
    
    if verbose:
        print(f"✓ Found {len(checkpoint_paths)} checkpoints")
    
    # Extract metrics
    if verbose:
        print("\nExtracting metrics from checkpoints...")
        pbar = tqdm(checkpoint_paths, desc="Loading checkpoints")
    else:
        pbar = checkpoint_paths
    
    validation_dir = Path(args.validation_dir) if args.validation_dir else None
    
    checkpoints_metrics = []
    for ckpt_path in pbar:
        metrics = extract_metrics_from_checkpoint(ckpt_path, validation_dir)
        checkpoints_metrics.append(metrics)
    
    if verbose and hasattr(pbar, 'close'):
        pbar.close()
    
    # Select best checkpoint(s)
    if verbose:
        print(f"\nSelecting best checkpoint(s) by {args.criterion}...")
    
    if args.top_n == 1:
        best = select_best_checkpoint(
            checkpoints_metrics,
            criterion=args.criterion,
            sar_weight=args.sar_weight,
            loss_weight=args.loss_weight,
            perceptual_weight=args.perceptual_weight,
            verbose=verbose
        )
        
        if best is None:
            print("❌ Could not select best checkpoint (no valid metrics)")
            return 1
        
        selected = [best]
    
    else:
        selected = select_top_n_checkpoints(
            checkpoints_metrics,
            n=args.top_n,
            criterion=args.criterion,
            sar_weight=args.sar_weight,
            loss_weight=args.loss_weight,
            perceptual_weight=args.perceptual_weight
        )
        
        if len(selected) == 0:
            print("❌ Could not select checkpoints (no valid metrics)")
            return 1
        
        if verbose:
            print(f"\nSelected top {len(selected)} checkpoint(s):")
            for i, ckpt in enumerate(selected, 1):
                print(f"  {i}. Epoch {ckpt.epoch}: {Path(ckpt.checkpoint_path).name}")
    
    # Export checkpoint(s)
    if verbose:
        print("\nExporting checkpoint(s)...")
    
    exported_paths = []
    
    if args.output:
        # Single output file
        output_path = export_checkpoint(
            source_path=Path(selected[0].checkpoint_path),
            output_path=Path(args.output),
            metrics=selected[0],
            include_metadata=not args.no_metadata
        )
        exported_paths.append(output_path)
        
        if verbose:
            print(f"✓ Exported to: {output_path}")
    
    else:
        # Multiple files to output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, ckpt in enumerate(selected, 1):
            source_path = Path(ckpt.checkpoint_path)
            
            # Create output filename
            if args.top_n == 1:
                output_name = f"best_stage3_{args.criterion}.pt"
            else:
                output_name = f"top{i:02d}_stage3_{args.criterion}_epoch{ckpt.epoch:03d}.pt"
            
            output_path = export_checkpoint(
                source_path=source_path,
                output_path=output_dir / output_name,
                metrics=ckpt,
                include_metadata=not args.no_metadata
            )
            exported_paths.append(output_path)
            
            if verbose:
                print(f"  {i}. {output_path.name}")
    
    # Save selection report
    if args.save_report:
        if args.output:
            report_path = Path(args.output).with_suffix('.json')
        else:
            report_path = Path(args.output_dir) / f"selection_report_{args.criterion}.json"
        
        save_selection_report(
            output_path=report_path,
            best_checkpoint=selected[0],
            all_checkpoints=checkpoints_metrics,
            criterion=args.criterion
        )
        
        if verbose:
            print(f"\n✓ Selection report saved to: {report_path}")
    
    # Print summary
    if verbose:
        print("\n" + "=" * 80)
        print("Export Complete!")
        print("=" * 80)
        print(f"\nBest Checkpoint: {Path(selected[0].checkpoint_path).name}")
        print(f"  Epoch: {selected[0].epoch}")
        
        if selected[0].sar_agreement is not None:
            print(f"  SAR Agreement: {selected[0].sar_agreement:.6f}")
        
        if selected[0].composite_score is not None:
            print(f"  Composite Score: {selected[0].composite_score:.6f}")
        
        if selected[0].val_loss is not None:
            print(f"  Validation Loss: {selected[0].val_loss:.6f}")
        
        if selected[0].lpips_mean is not None:
            print(f"  LPIPS: {selected[0].lpips_mean:.6f}")
        
        print(f"\nExported {len(exported_paths)} checkpoint(s)")
        print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
