"""
Export Best Stage 1 Checkpoint

This script analyzes validation metrics from training logs, computes a composite
quality score for each checkpoint, and exports the best checkpoint to a standard
location for deployment.

Composite Score Components:
- SSIM (higher is better): Structural similarity
- LPIPS (lower is better): Perceptual similarity  
- PSNR (higher is better): Signal quality
- SAR-Edge Agreement (higher is better): Structure preservation
- Loss (lower is better): Training objective

The script supports multiple selection strategies:
- 'composite': Weighted average of normalized metrics (default)
- 'ssim': Best SSIM score
- 'loss': Lowest validation loss
- 'balanced': Equal weight to all metrics

Usage:
    python scripts/export_stage1_best.py --run-dir runs/exp1/
    python scripts/export_stage1_best.py --run-dir runs/exp1/ --strategy composite --output weights/best_stage1.pt
    python scripts/export_stage1_best.py --csv runs/exp1/logs/stage1_val.csv --checkpoint-dir runs/exp1/checkpoints/
"""

import argparse
import sys
from pathlib import Path
import shutil
import csv
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Metric Analysis
# ============================================================================

def read_validation_csv(csv_path: Path) -> List[Dict]:
    """
    Read validation metrics from CSV file.
    
    Returns:
        List of metric dictionaries, one per validation step
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found: {csv_path}")
    
    metrics_list = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            metric_dict = {}
            for key, value in row.items():
                if key == 'timestamp':
                    metric_dict[key] = value
                else:
                    try:
                        metric_dict[key] = float(value)
                    except (ValueError, TypeError):
                        metric_dict[key] = value
            
            metrics_list.append(metric_dict)
    
    return metrics_list


def normalize_metrics(
    metrics_list: List[Dict],
    higher_is_better: List[str] = ['ssim', 'psnr', 'sar_edge_agreement'],
    lower_is_better: List[str] = ['loss', 'lpips', 'mae']
) -> List[Dict]:
    """
    Normalize metrics to [0, 1] range for comparison.
    
    For 'higher is better' metrics: normalized = (value - min) / (max - min)
    For 'lower is better' metrics: normalized = (max - value) / (max - min)
    
    Args:
        metrics_list: List of metric dictionaries
        higher_is_better: Metrics where higher values are better
        lower_is_better: Metrics where lower values are better
        
    Returns:
        List of metric dictionaries with normalized values
    """
    if not metrics_list:
        return []
    
    # Collect all metric names
    all_metrics = set()
    for metrics in metrics_list:
        all_metrics.update(metrics.keys())
    
    # Remove non-numeric fields
    all_metrics = all_metrics - {'timestamp', 'step'}
    
    # Compute min/max for each metric
    metric_ranges = {}
    for metric_name in all_metrics:
        values = [m.get(metric_name, np.nan) for m in metrics_list]
        values = [v for v in values if not np.isnan(v)]
        
        if values:
            metric_ranges[metric_name] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
    
    # Normalize
    normalized_list = []
    
    for metrics in metrics_list:
        normalized = metrics.copy()
        
        for metric_name in all_metrics:
            if metric_name not in metrics or metric_name not in metric_ranges:
                continue
            
            value = metrics[metric_name]
            if np.isnan(value):
                continue
            
            min_val = metric_ranges[metric_name]['min']
            max_val = metric_ranges[metric_name]['max']
            
            # Avoid division by zero
            if max_val - min_val < 1e-10:
                normalized[f'{metric_name}_norm'] = 1.0
                continue
            
            # Normalize based on direction
            if metric_name in higher_is_better:
                # Higher is better: (value - min) / (max - min)
                normalized[f'{metric_name}_norm'] = (value - min_val) / (max_val - min_val)
            elif metric_name in lower_is_better:
                # Lower is better: (max - value) / (max - min)
                normalized[f'{metric_name}_norm'] = (max_val - value) / (max_val - min_val)
            else:
                # Unknown direction, assume higher is better
                normalized[f'{metric_name}_norm'] = (value - min_val) / (max_val - min_val)
        
        normalized_list.append(normalized)
    
    return normalized_list


def compute_composite_score(
    metrics: Dict,
    strategy: str = 'composite',
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute composite quality score from normalized metrics.
    
    Args:
        metrics: Dictionary of metrics (must include normalized versions)
        strategy: Selection strategy ('composite', 'ssim', 'loss', 'balanced')
        weights: Custom weights for composite score (only for 'composite' strategy)
        
    Returns:
        Composite score (higher is better)
    """
    if strategy == 'ssim':
        # Use SSIM only
        return metrics.get('ssim_norm', 0.0)
    
    elif strategy == 'loss':
        # Use loss only (normalized so higher is better)
        return metrics.get('loss_norm', 0.0)
    
    elif strategy == 'balanced':
        # Equal weight to all available metrics
        metric_names = ['ssim_norm', 'lpips_norm', 'psnr_norm', 'sar_edge_agreement_norm', 'loss_norm']
        available_metrics = [metrics.get(m, 0.0) for m in metric_names if m in metrics]
        
        if not available_metrics:
            return 0.0
        
        return np.mean(available_metrics)
    
    elif strategy == 'composite':
        # Weighted composite (default)
        if weights is None:
            # Default weights: emphasize SSIM and structure, less weight on loss
            weights = {
                'ssim_norm': 0.30,
                'lpips_norm': 0.15,
                'psnr_norm': 0.20,
                'sar_edge_agreement_norm': 0.25,
                'loss_norm': 0.10
            }
        
        score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                score += weight * metrics[metric_name]
                total_weight += weight
        
        # Normalize by total weight (in case some metrics are missing)
        if total_weight > 0:
            score /= total_weight
        
        return score
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def select_best_checkpoint(
    metrics_list: List[Dict],
    strategy: str = 'composite',
    weights: Optional[Dict[str, float]] = None
) -> Tuple[Dict, int]:
    """
    Select the best checkpoint based on validation metrics.
    
    Args:
        metrics_list: List of metric dictionaries (normalized)
        strategy: Selection strategy
        weights: Custom weights for composite score
        
    Returns:
        Tuple of (best_metrics, best_index)
    """
    if not metrics_list:
        raise ValueError("No metrics available for selection")
    
    best_score = -float('inf')
    best_metrics = None
    best_index = -1
    
    for idx, metrics in enumerate(metrics_list):
        score = compute_composite_score(metrics, strategy, weights)
        
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_index = idx
    
    return best_metrics, best_index


# ============================================================================
# Checkpoint Export
# ============================================================================

def find_checkpoint_for_step(
    checkpoint_dir: Path,
    step: int,
    tolerance: int = 50
) -> Optional[Path]:
    """
    Find checkpoint file corresponding to a validation step.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        step: Target step number
        tolerance: Acceptable difference in step numbers
        
    Returns:
        Path to checkpoint file or None if not found
    """
    if not checkpoint_dir.exists():
        return None
    
    # Look for checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pt"))
    
    if not checkpoint_files:
        return None
    
    # Find closest checkpoint
    best_match = None
    best_diff = float('inf')
    
    for ckpt_path in checkpoint_files:
        # Extract step number from filename
        # Expected format: checkpoint_step_5000.pt or checkpoint_epoch_10.pt
        stem = ckpt_path.stem
        
        try:
            if 'step' in stem:
                parts = stem.split('step_')
                if len(parts) > 1:
                    ckpt_step = int(parts[1])
                    diff = abs(ckpt_step - step)
                    
                    if diff < best_diff and diff <= tolerance:
                        best_diff = diff
                        best_match = ckpt_path
        except (ValueError, IndexError):
            continue
    
    return best_match


def export_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    metrics: Dict,
    include_metadata: bool = True
):
    """
    Export checkpoint to standard location with metadata.
    
    Args:
        checkpoint_path: Source checkpoint file
        output_path: Destination path
        metrics: Validation metrics to include in export
        include_metadata: Whether to save metadata alongside checkpoint
    """
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy checkpoint
    print(f"Copying checkpoint...")
    print(f"  From: {checkpoint_path}")
    print(f"  To:   {output_path}")
    shutil.copy2(checkpoint_path, output_path)
    
    # Save metadata
    if include_metadata:
        import json
        
        metadata_path = output_path.with_suffix('.json')
        
        metadata = {
            'source_checkpoint': str(checkpoint_path),
            'export_timestamp': str(np.datetime64('now')),
            'validation_metrics': {
                'step': int(metrics.get('step', 0)),
                'loss': float(metrics.get('loss', 0.0)),
                'ssim': float(metrics.get('ssim', 0.0)),
                'lpips': float(metrics.get('lpips', 0.0)),
                'psnr': float(metrics.get('psnr', 0.0)),
                'sar_edge_agreement': float(metrics.get('sar_edge_agreement', 0.0)),
                'composite_score': float(metrics.get('composite_score', 0.0))
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Metadata: {metadata_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export best Stage 1 checkpoint based on validation metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export best checkpoint from run directory (auto-detect paths)
  python scripts/export_stage1_best.py --run-dir runs/exp1/
  
  # Specify CSV and checkpoint directory explicitly
  python scripts/export_stage1_best.py --csv runs/exp1/logs/stage1_val.csv --checkpoint-dir runs/exp1/checkpoints/
  
  # Use different selection strategy
  python scripts/export_stage1_best.py --run-dir runs/exp1/ --strategy ssim
  
  # Custom output location
  python scripts/export_stage1_best.py --run-dir runs/exp1/ --output weights/production/stage1.pt
  
  # Show analysis without exporting
  python scripts/export_stage1_best.py --run-dir runs/exp1/ --dry-run

Selection Strategies:
  composite: Weighted average of all metrics (default)
             Weights: SSIM=0.30, LPIPS=0.15, PSNR=0.20, SAR-Edge=0.25, Loss=0.10
  ssim:      Select checkpoint with best SSIM
  loss:      Select checkpoint with lowest validation loss  
  balanced:  Equal weight to all available metrics
        """
    )
    
    # Input
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Run directory (auto-detects CSV and checkpoint paths)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to validation CSV file (overrides run-dir)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory containing checkpoints (overrides run-dir)')
    
    # Selection
    parser.add_argument('--strategy', type=str, default='composite',
                        choices=['composite', 'ssim', 'loss', 'balanced'],
                        help='Checkpoint selection strategy (default: composite)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Show top K checkpoints (default: 5)')
    
    # Output
    parser.add_argument('--output', type=str, default='weights/best_stage1.pt',
                        help='Output path for best checkpoint (default: weights/best_stage1.pt)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show analysis without copying checkpoint')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("Export Best Stage 1 Checkpoint")
    print("=" * 80)
    print(f"Selection strategy: {args.strategy}")
    print()
    
    # Determine paths
    if args.run_dir:
        run_dir = Path(args.run_dir)
        
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            sys.exit(1)
        
        # Auto-detect CSV
        if args.csv is None:
            csv_candidates = [
                run_dir / 'logs' / 'stage1_val.csv',
                run_dir / 'stage1_val.csv',
                run_dir / 'validation.csv'
            ]
            
            csv_path = None
            for candidate in csv_candidates:
                if candidate.exists():
                    csv_path = candidate
                    break
            
            if csv_path is None:
                print(f"Error: Could not find validation CSV in {run_dir}")
                print("Expected locations:")
                for c in csv_candidates:
                    print(f"  - {c}")
                sys.exit(1)
        else:
            csv_path = Path(args.csv)
        
        # Auto-detect checkpoint directory
        if args.checkpoint_dir is None:
            checkpoint_candidates = [
                run_dir / 'checkpoints',
                run_dir / 'ckpts',
                run_dir
            ]
            
            checkpoint_dir = None
            for candidate in checkpoint_candidates:
                if candidate.exists() and list(candidate.glob("checkpoint_*.pt")):
                    checkpoint_dir = candidate
                    break
            
            if checkpoint_dir is None:
                print(f"Error: Could not find checkpoints in {run_dir}")
                sys.exit(1)
        else:
            checkpoint_dir = Path(args.checkpoint_dir)
    else:
        if args.csv is None or args.checkpoint_dir is None:
            print("Error: Must specify either --run-dir or both --csv and --checkpoint-dir")
            sys.exit(1)
        
        csv_path = Path(args.csv)
        checkpoint_dir = Path(args.checkpoint_dir)
    
    print(f"Validation CSV: {csv_path}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print()
    
    # Read validation metrics
    print("Reading validation metrics...")
    metrics_list = read_validation_csv(csv_path)
    print(f"✓ Found {len(metrics_list)} validation checkpoints")
    
    if len(metrics_list) == 0:
        print("Error: No validation metrics found in CSV")
        sys.exit(1)
    
    # Normalize metrics
    print("\nNormalizing metrics...")
    normalized_metrics = normalize_metrics(metrics_list)
    
    # Compute composite scores
    for metrics in normalized_metrics:
        metrics['composite_score'] = compute_composite_score(metrics, args.strategy)
    
    # Select best checkpoint
    print(f"\nSelecting best checkpoint (strategy: {args.strategy})...")
    best_metrics, best_index = select_best_checkpoint(normalized_metrics, args.strategy)
    
    # Display top-k checkpoints
    print("\n" + "=" * 80)
    print(f"Top {args.top_k} Checkpoints")
    print("=" * 80)
    
    # Sort by composite score
    sorted_metrics = sorted(normalized_metrics, key=lambda m: m['composite_score'], reverse=True)
    
    for rank, metrics in enumerate(sorted_metrics[:args.top_k], 1):
        step = int(metrics.get('step', 0))
        score = metrics['composite_score']
        
        print(f"\n#{rank} - Step {step} (Score: {score:.4f})")
        print(f"  Loss: {metrics.get('loss', 0.0):.4f}")
        
        if 'ssim' in metrics:
            print(f"  SSIM: {metrics.get('ssim', 0.0):.4f}")
        if 'lpips' in metrics:
            print(f"  LPIPS: {metrics.get('lpips', 0.0):.4f}")
        if 'psnr' in metrics:
            print(f"  PSNR: {metrics.get('psnr', 0.0):.2f} dB")
        if 'sar_edge_agreement' in metrics:
            print(f"  SAR-Edge: {metrics.get('sar_edge_agreement', 0.0):.4f}")
    
    # Display best checkpoint details
    print("\n" + "=" * 80)
    print("Selected Best Checkpoint")
    print("=" * 80)
    
    best_step = int(best_metrics.get('step', 0))
    print(f"Step: {best_step}")
    print(f"Composite Score: {best_metrics['composite_score']:.4f}")
    print()
    print("Metrics:")
    print(f"  Loss: {best_metrics.get('loss', 0.0):.4f}")
    if 'ssim' in best_metrics:
        print(f"  SSIM: {best_metrics.get('ssim', 0.0):.4f}")
    if 'lpips' in best_metrics:
        print(f"  LPIPS: {best_metrics.get('lpips', 0.0):.4f}")
    if 'psnr' in best_metrics:
        print(f"  PSNR: {best_metrics.get('psnr', 0.0):.2f} dB")
    if 'sar_edge_agreement' in best_metrics:
        print(f"  SAR-Edge: {best_metrics.get('sar_edge_agreement', 0.0):.4f}")
    
    # Find checkpoint file
    print("\nLocating checkpoint file...")
    checkpoint_path = find_checkpoint_for_step(checkpoint_dir, best_step)
    
    if checkpoint_path is None:
        print(f"✗ Could not find checkpoint for step {best_step}")
        print(f"  Searched in: {checkpoint_dir}")
        sys.exit(1)
    
    print(f"✓ Found: {checkpoint_path}")
    
    # Export
    if args.dry_run:
        print("\n" + "=" * 80)
        print("Dry run - checkpoint would be exported to:")
        print(f"  {args.output}")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("Exporting checkpoint...")
        print("=" * 80)
        
        output_path = Path(args.output)
        export_checkpoint(checkpoint_path, output_path, best_metrics)
        
        print("\n" + "=" * 80)
        print("✓ Export complete!")
        print("=" * 80)
        print(f"Best checkpoint saved to: {output_path}")
        print(f"Metadata saved to: {output_path.with_suffix('.json')}")


if __name__ == '__main__':
    main()
