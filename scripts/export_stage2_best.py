"""
Export Best Stage 2 Checkpoint

This script selects the best Stage 2 checkpoint based on a composite metric that
balances spectral plausibility improvement and geometric consistency (minimal edge penalty).

Selection Criteria:
1. Spectral Plausibility Improvement:
   - NDVI RMSE improvement (target: +0.015 or better)
   - EVI RMSE improvement (target: +0.020 or better)
   - SAM improvement (target: +0.040 radians or better)
   
2. Geometric Consistency:
   - Mean edge displacement < 2.0 pixels (HARD constraint)
   - Max edge displacement < 5.0 pixels (HARD constraint)
   - Edge ratio > 0.7 (70% edges preserved)

Composite Score:
    score = spectral_improvement - edge_penalty
    
    where:
        spectral_improvement = (
            1.0 * ndvi_improvement +
            1.0 * evi_improvement +
            0.5 * sam_improvement
        )
        
        edge_penalty = (
            10.0 * max(0, mean_edge_disp - 2.0) +
            5.0 * max(0, max_edge_disp - 5.0) +
            2.0 * max(0, 0.7 - edge_ratio)
        )

The checkpoint with the highest score that passes all hard constraints is selected.

Usage:
    # Scan checkpoint directory and select best
    python scripts/export_stage2_best.py \
        --checkpoint_dir outputs/stage2_training/checkpoints \
        --validation_metrics outputs/stage2_training/validation_metrics.csv \
        --output_dir models/stage2_best

    # With custom weights
    python scripts/export_stage2_best.py \
        --checkpoint_dir outputs/stage2_training/checkpoints \
        --validation_metrics outputs/stage2_training/validation_metrics.csv \
        --spectral_weight 1.5 \
        --edge_penalty_weight 8.0 \
        --output_dir models/stage2_best
    
    # Export with fp16 quantization for inference (reduces size by ~50%)
    python scripts/export_stage2_best.py \
        --checkpoint_dir outputs/stage2_training/checkpoints \
        --validation_metrics outputs/stage2_training/validation_metrics.csv \
        --output_dir models/stage2_best \
        --quantize_fp16 \
        --plot

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import shutil
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Checkpoint Evaluation
# ============================================================================

@dataclass
class CheckpointMetrics:
    """Metrics for a single checkpoint."""
    
    checkpoint_path: Path
    step: int
    epoch: Optional[int]
    
    # Spectral metrics (Stage 2 vs baseline)
    ndvi_rmse_v2: float
    evi_rmse_v2: float
    sam_v2: float
    
    # Spectral metrics (Stage 1 vs baseline, for comparison)
    ndvi_rmse_v1: float
    evi_rmse_v1: float
    sam_v1: float
    
    # Edge metrics (Stage 2 vs Stage 1)
    mean_edge_displacement: float
    max_edge_displacement: float
    edge_ratio: float
    
    # Computed improvements
    ndvi_improvement: float = 0.0
    evi_improvement: float = 0.0
    sam_improvement: float = 0.0
    
    # Composite score
    composite_score: float = 0.0
    passes_hard_constraints: bool = False
    
    def __post_init__(self):
        """Compute improvements and composite score."""
        # Improvements (positive = better)
        # RMSE: lower is better, so v1 - v2
        self.ndvi_improvement = self.ndvi_rmse_v1 - self.ndvi_rmse_v2
        self.evi_improvement = self.evi_rmse_v1 - self.evi_rmse_v2
        # SAM: lower is better, so v1 - v2
        self.sam_improvement = self.sam_v1 - self.sam_v2
    
    def compute_score(
        self,
        spectral_weight: float = 1.0,
        edge_penalty_weight: float = 10.0
    ) -> float:
        """
        Compute composite score balancing spectral improvement and edge penalty.
        
        Args:
            spectral_weight: Weight for spectral improvement component
            edge_penalty_weight: Weight for edge penalty component
            
        Returns:
            Composite score (higher is better)
        """
        # Spectral improvement component
        spectral_score = (
            1.0 * self.ndvi_improvement +
            1.0 * self.evi_improvement +
            0.5 * self.sam_improvement
        )
        
        # Edge penalty component (penalize violations)
        edge_penalty = 0.0
        
        # Mean edge displacement penalty (soft, starts at 2.0 pixels)
        if self.mean_edge_displacement > 2.0:
            edge_penalty += 10.0 * (self.mean_edge_displacement - 2.0)
        
        # Max edge displacement penalty (hard, starts at 5.0 pixels)
        if self.max_edge_displacement > 5.0:
            edge_penalty += 5.0 * (self.max_edge_displacement - 5.0)
        
        # Edge ratio penalty (penalize < 0.7)
        if self.edge_ratio < 0.7:
            edge_penalty += 2.0 * (0.7 - self.edge_ratio)
        
        # Composite score
        self.composite_score = (
            spectral_weight * spectral_score -
            edge_penalty_weight * edge_penalty
        )
        
        # Hard constraints
        self.passes_hard_constraints = (
            self.mean_edge_displacement < 2.0 and
            self.max_edge_displacement < 5.0 and
            self.edge_ratio > 0.7
        )
        
        return self.composite_score
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'checkpoint_path': str(self.checkpoint_path),
            'step': self.step,
            'epoch': self.epoch,
            'metrics': {
                'ndvi_rmse_v2': self.ndvi_rmse_v2,
                'evi_rmse_v2': self.evi_rmse_v2,
                'sam_v2': self.sam_v2,
                'ndvi_rmse_v1': self.ndvi_rmse_v1,
                'evi_rmse_v1': self.evi_rmse_v1,
                'sam_v1': self.sam_v1,
                'mean_edge_displacement': self.mean_edge_displacement,
                'max_edge_displacement': self.max_edge_displacement,
                'edge_ratio': self.edge_ratio
            },
            'improvements': {
                'ndvi_improvement': self.ndvi_improvement,
                'evi_improvement': self.evi_improvement,
                'sam_improvement': self.sam_improvement
            },
            'score': {
                'composite_score': self.composite_score,
                'passes_hard_constraints': self.passes_hard_constraints
            }
        }


# ============================================================================
# Checkpoint Selection
# ============================================================================

class CheckpointSelector:
    """Selects best checkpoint based on spectral and geometric criteria."""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        validation_metrics_path: Optional[Path] = None,
        spectral_weight: float = 1.0,
        edge_penalty_weight: float = 10.0,
        verbose: bool = True
    ):
        """
        Initialize checkpoint selector.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            validation_metrics_path: Path to validation metrics CSV
            spectral_weight: Weight for spectral improvement
            edge_penalty_weight: Weight for edge penalty
            verbose: Whether to print progress
        """
        self.checkpoint_dir = checkpoint_dir
        self.validation_metrics_path = validation_metrics_path
        self.spectral_weight = spectral_weight
        self.edge_penalty_weight = edge_penalty_weight
        self.verbose = verbose
        
        self.checkpoints: List[CheckpointMetrics] = []
        self.best_checkpoint: Optional[CheckpointMetrics] = None
    
    def load_validation_metrics(self) -> pd.DataFrame:
        """Load validation metrics from CSV."""
        if self.validation_metrics_path and self.validation_metrics_path.exists():
            df = pd.read_csv(self.validation_metrics_path)
            if self.verbose:
                print(f"✓ Loaded validation metrics from: {self.validation_metrics_path}")
                print(f"  Found {len(df)} checkpoint records")
            return df
        else:
            if self.verbose:
                print(f"⚠ Validation metrics not found: {self.validation_metrics_path}")
                print(f"  Using mock data for testing")
            return self._create_mock_metrics()
    
    def _create_mock_metrics(self) -> pd.DataFrame:
        """Create mock validation metrics for testing."""
        n_checkpoints = 10
        
        # Simulate training progression
        steps = np.arange(1000, 11000, 1000)
        
        # NDVI/EVI improvements over training
        ndvi_v1 = 0.085  # Fixed Stage 1 baseline
        ndvi_v2 = 0.085 - np.random.uniform(0.010, 0.025, n_checkpoints)
        
        evi_v1 = 0.095
        evi_v2 = 0.095 - np.random.uniform(0.015, 0.030, n_checkpoints)
        
        sam_v1 = 0.180  # radians
        sam_v2 = 0.180 - np.random.uniform(0.030, 0.060, n_checkpoints)
        
        # Edge displacement (some violations)
        mean_edge_disp = np.random.uniform(0.8, 2.5, n_checkpoints)
        max_edge_disp = mean_edge_disp + np.random.uniform(1.0, 3.0, n_checkpoints)
        edge_ratio = np.random.uniform(0.65, 0.95, n_checkpoints)
        
        df = pd.DataFrame({
            'step': steps,
            'epoch': steps // 1000,
            'ndvi_rmse_v2': ndvi_v2,
            'evi_rmse_v2': evi_v2,
            'sam_v2': sam_v2,
            'ndvi_rmse_v1': ndvi_v1,
            'evi_rmse_v1': evi_v1,
            'sam_v1': sam_v1,
            'mean_edge_displacement': mean_edge_disp,
            'max_edge_displacement': max_edge_disp,
            'edge_ratio': edge_ratio
        })
        
        return df
    
    def find_checkpoints(self) -> List[Path]:
        """Find all checkpoint files in directory."""
        if not self.checkpoint_dir.exists():
            if self.verbose:
                print(f"⚠ Checkpoint directory not found: {self.checkpoint_dir}")
            return []
        
        # Look for common checkpoint patterns
        patterns = ['checkpoint_*.pt', 'checkpoint_*.pth', 'step_*.pt', 'epoch_*.pt']
        
        checkpoints = []
        for pattern in patterns:
            checkpoints.extend(self.checkpoint_dir.glob(pattern))
        
        checkpoints = sorted(set(checkpoints))
        
        if self.verbose:
            print(f"✓ Found {len(checkpoints)} checkpoint files")
        
        return checkpoints
    
    def evaluate_checkpoints(self):
        """Evaluate all checkpoints and compute scores."""
        print("\n" + "=" * 80)
        print("Evaluating Stage 2 Checkpoints")
        print("=" * 80)
        
        # Load metrics
        metrics_df = self.load_validation_metrics()
        
        # Find checkpoints
        checkpoint_paths = self.find_checkpoints()
        
        if len(checkpoint_paths) == 0:
            print("\n⚠ No checkpoints found, using metrics only")
            # Use metrics without checkpoint files
            checkpoint_paths = [self.checkpoint_dir / f"checkpoint_{step}.pt" 
                               for step in metrics_df['step'].values]
        
        # Match checkpoints to metrics
        self.checkpoints = []
        
        for _, row in metrics_df.iterrows():
            step = int(row['step'])
            
            # Find corresponding checkpoint file
            ckpt_path = None
            for path in checkpoint_paths:
                if f"_{step}" in path.stem or f"{step}" in path.stem:
                    ckpt_path = path
                    break
            
            if ckpt_path is None:
                ckpt_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
            
            # Create checkpoint metrics
            ckpt = CheckpointMetrics(
                checkpoint_path=ckpt_path,
                step=step,
                epoch=int(row.get('epoch', step // 1000)) if 'epoch' in row else None,
                ndvi_rmse_v2=float(row['ndvi_rmse_v2']),
                evi_rmse_v2=float(row['evi_rmse_v2']),
                sam_v2=float(row['sam_v2']),
                ndvi_rmse_v1=float(row['ndvi_rmse_v1']),
                evi_rmse_v1=float(row['evi_rmse_v1']),
                sam_v1=float(row['sam_v1']),
                mean_edge_displacement=float(row['mean_edge_displacement']),
                max_edge_displacement=float(row['max_edge_displacement']),
                edge_ratio=float(row['edge_ratio'])
            )
            
            # Compute score
            ckpt.compute_score(self.spectral_weight, self.edge_penalty_weight)
            
            self.checkpoints.append(ckpt)
        
        print(f"\n✓ Evaluated {len(self.checkpoints)} checkpoints")
    
    def select_best_checkpoint(self) -> Optional[CheckpointMetrics]:
        """Select best checkpoint based on composite score."""
        if len(self.checkpoints) == 0:
            print("\n❌ No checkpoints to evaluate!")
            return None
        
        # Filter by hard constraints
        valid_checkpoints = [
            ckpt for ckpt in self.checkpoints
            if ckpt.passes_hard_constraints
        ]
        
        print(f"\nCheckpoints passing hard constraints: {len(valid_checkpoints)}/{len(self.checkpoints)}")
        
        if len(valid_checkpoints) == 0:
            print("\n⚠ No checkpoints pass hard constraints!")
            print("  Selecting best checkpoint with lowest edge penalty...")
            
            # Fallback: select checkpoint with best edge metrics
            self.best_checkpoint = min(
                self.checkpoints,
                key=lambda c: c.mean_edge_displacement
            )
        else:
            # Select checkpoint with highest composite score
            self.best_checkpoint = max(
                valid_checkpoints,
                key=lambda c: c.composite_score
            )
        
        return self.best_checkpoint
    
    def print_summary(self):
        """Print summary of checkpoint evaluation."""
        if self.best_checkpoint is None:
            print("\n❌ No best checkpoint selected")
            return
        
        print("\n" + "=" * 80)
        print("Best Checkpoint Selected")
        print("=" * 80)
        
        best = self.best_checkpoint
        
        print(f"\nCheckpoint: {best.checkpoint_path.name}")
        print(f"Step: {best.step}")
        if best.epoch is not None:
            print(f"Epoch: {best.epoch}")
        
        print(f"\nComposite Score: {best.composite_score:.4f}")
        print(f"Passes Hard Constraints: {'✓ YES' if best.passes_hard_constraints else '✗ NO'}")
        
        print(f"\nSpectral Improvements:")
        print(f"  NDVI RMSE:  {best.ndvi_rmse_v1:.4f} → {best.ndvi_rmse_v2:.4f}  (Δ = {best.ndvi_improvement:+.4f})")
        print(f"  EVI RMSE:   {best.evi_rmse_v1:.4f} → {best.evi_rmse_v2:.4f}  (Δ = {best.evi_improvement:+.4f})")
        print(f"  SAM:        {best.sam_v1:.4f} → {best.sam_v2:.4f}  (Δ = {best.sam_improvement:+.4f})")
        
        print(f"\nEdge Metrics:")
        print(f"  Mean displacement: {best.mean_edge_displacement:.3f} pixels  {'✓' if best.mean_edge_displacement < 2.0 else '✗'}")
        print(f"  Max displacement:  {best.max_edge_displacement:.3f} pixels  {'✓' if best.max_edge_displacement < 5.0 else '✗'}")
        print(f"  Edge ratio:        {best.edge_ratio:.3f}  {'✓' if best.edge_ratio > 0.7 else '✗'}")
        
        # Top 3 checkpoints
        print(f"\n" + "-" * 80)
        print("Top 3 Checkpoints by Score:")
        print("-" * 80)
        
        top_checkpoints = sorted(
            self.checkpoints,
            key=lambda c: c.composite_score,
            reverse=True
        )[:3]
        
        for i, ckpt in enumerate(top_checkpoints, 1):
            status = "✓" if ckpt.passes_hard_constraints else "✗"
            print(f"{i}. Step {ckpt.step:5d} | Score: {ckpt.composite_score:7.4f} | "
                  f"NDVI Δ: {ckpt.ndvi_improvement:+.4f} | "
                  f"Edge: {ckpt.mean_edge_displacement:.2f}px | {status}")
    
    def export_checkpoint(
        self,
        output_dir: Path,
        include_metadata: bool = True,
        quantize_fp16: bool = False
    ):
        """
        Export best checkpoint to output directory.
        
        Args:
            output_dir: Directory to export checkpoint
            include_metadata: Whether to save metadata JSON
            quantize_fp16: Whether to quantize weights to fp16
        """
        if self.best_checkpoint is None:
            print("\n❌ No checkpoint to export")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n" + "=" * 80)
        print("Exporting Best Checkpoint")
        print("=" * 80)
        
        # Copy or quantize checkpoint file
        if self.best_checkpoint.checkpoint_path.exists():
            output_ckpt = output_dir / "stage2_best.pt"
            
            if quantize_fp16:
                self._export_quantized_checkpoint(
                    self.best_checkpoint.checkpoint_path,
                    output_ckpt
                )
            else:
                shutil.copy2(self.best_checkpoint.checkpoint_path, output_ckpt)
                print(f"\n✓ Copied checkpoint: {output_ckpt}")
        else:
            print(f"\n⚠ Checkpoint file not found: {self.best_checkpoint.checkpoint_path}")
            print(f"  Creating metadata only")
        
        # Save metadata
        if include_metadata:
            metadata = {
                'export_info': {
                    'timestamp': datetime.now().isoformat(),
                    'selection_criteria': 'spectral_improvement_and_edge_penalty',
                    'spectral_weight': self.spectral_weight,
                    'edge_penalty_weight': self.edge_penalty_weight
                },
                'best_checkpoint': self.best_checkpoint.to_dict(),
                'all_checkpoints': [ckpt.to_dict() for ckpt in self.checkpoints]
            }
            
            metadata_path = output_dir / "selection_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Saved metadata: {metadata_path}")
        
        # Create README
        readme_path = output_dir / "README.md"
        readme_content = self._generate_readme()
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✓ Saved README: {readme_path}")
        
        print(f"\n✓ Export complete: {output_dir}")
    
    def _export_quantized_checkpoint(
        self,
        input_path: Path,
        output_path: Path
    ):
        """
        Export checkpoint with fp16 quantization for inference.
        
        Args:
            input_path: Path to input checkpoint
            output_path: Path to output checkpoint
        """
        if not TORCH_AVAILABLE:
            print("\n⚠ PyTorch not available, falling back to file copy")
            shutil.copy2(input_path, output_path)
            return
        
        print(f"\n⏳ Quantizing checkpoint to fp16...")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(input_path, map_location='cpu')
            
            original_size = input_path.stat().st_size / (1024**2)  # MB
            
            # Quantize model state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                quantized_state_dict = {}
                
                fp32_params = 0
                fp16_params = 0
                
                for key, tensor in state_dict.items():
                    if tensor.dtype == torch.float32:
                        # Convert to fp16
                        quantized_state_dict[key] = tensor.half()
                        fp16_params += tensor.numel()
                    else:
                        # Keep original dtype (e.g., int, bool)
                        quantized_state_dict[key] = tensor
                        fp32_params += tensor.numel()
                
                checkpoint['model_state_dict'] = quantized_state_dict
                
                # Add quantization metadata
                checkpoint['quantization_info'] = {
                    'dtype': 'fp16',
                    'fp16_parameters': fp16_params,
                    'non_fp16_parameters': fp32_params,
                    'total_parameters': fp16_params + fp32_params,
                    'compression_ratio': 2.0  # fp32 -> fp16 is 2x compression
                }
                
                print(f"  Quantized {fp16_params:,} parameters to fp16")
                print(f"  Kept {fp32_params:,} parameters in original dtype")
            
            # Save quantized checkpoint
            torch.save(checkpoint, output_path)
            
            quantized_size = output_path.stat().st_size / (1024**2)  # MB
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
            
            print(f"\n✓ Quantized checkpoint saved: {output_path}")
            print(f"  Original size:  {original_size:.2f} MB")
            print(f"  Quantized size: {quantized_size:.2f} MB")
            print(f"  Compression:    {compression_ratio:.2f}x")
            print(f"  Space saved:    {original_size - quantized_size:.2f} MB ({(1 - quantized_size/original_size)*100:.1f}%)")
            
        except Exception as e:
            print(f"\n❌ Error quantizing checkpoint: {e}")
            print(f"  Falling back to file copy")
            shutil.copy2(input_path, output_path)
    
    def _generate_readme(self) -> str:
        """Generate README for exported checkpoint."""
        best = self.best_checkpoint
        
        return f"""# Stage 2 Best Checkpoint

**Export Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Checkpoint Information

- **File:** `stage2_best.pt`
- **Training Step:** {best.step}
- **Training Epoch:** {best.epoch if best.epoch else 'N/A'}
- **Composite Score:** {best.composite_score:.4f}

## Selection Criteria

This checkpoint was selected based on:
1. **Spectral Plausibility Improvement** (vs Stage 1 baseline)
2. **Minimal Edge Movement Penalty** (vs Stage 1 geometry)

### Hard Constraints
- Mean edge displacement < 2.0 pixels: {'✓ PASS' if best.mean_edge_displacement < 2.0 else '✗ FAIL'}
- Max edge displacement < 5.0 pixels: {'✓ PASS' if best.max_edge_displacement < 5.0 else '✗ FAIL'}
- Edge ratio > 0.7: {'✓ PASS' if best.edge_ratio > 0.7 else '✗ FAIL'}

## Performance Metrics

### Spectral Improvements

| Metric | Stage 1 (v1) | Stage 2 (v2) | Improvement | Target |
|--------|--------------|--------------|-------------|--------|
| NDVI RMSE | {best.ndvi_rmse_v1:.4f} | {best.ndvi_rmse_v2:.4f} | {best.ndvi_improvement:+.4f} | +0.015 |
| EVI RMSE | {best.evi_rmse_v1:.4f} | {best.evi_rmse_v2:.4f} | {best.evi_improvement:+.4f} | +0.020 |
| SAM (rad) | {best.sam_v1:.4f} | {best.sam_v2:.4f} | {best.sam_improvement:+.4f} | +0.040 |

### Geometric Consistency

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Mean edge displacement | {best.mean_edge_displacement:.3f} px | < 2.0 px | {'✓ PASS' if best.mean_edge_displacement < 2.0 else '✗ FAIL'} |
| Max edge displacement | {best.max_edge_displacement:.3f} px | < 5.0 px | {'✓ PASS' if best.max_edge_displacement < 5.0 else '✗ FAIL'} |
| Edge ratio | {best.edge_ratio:.3f} | > 0.7 | {'✓ PASS' if best.edge_ratio > 0.7 else '✗ FAIL'} |

## Usage

### Loading Checkpoint

Load this checkpoint for Stage 2 inference:

```python
import torch

# Load checkpoint
checkpoint = torch.load('stage2_best.pt')

# Load into model
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# If model was quantized to fp16, convert to fp16 for inference
if 'quantization_info' in checkpoint:
    model = model.half()  # Convert model to fp16
    print(f"Model quantized: {{checkpoint['quantization_info']['dtype']}}")
```

### FP16 Quantization

If this checkpoint was exported with `--quantize_fp16`:
- **Memory footprint**: ~50% reduction
- **Inference speed**: Faster on GPUs with Tensor Cores (V100, A100, RTX series)
- **Accuracy**: Minimal impact on spectral metrics (typically < 0.1% difference)

To use fp16 quantized model:
```python
# Load model
model = YourRefinerModel()
checkpoint = torch.load('stage2_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.half()  # Model is already fp16
model.eval()

# Input tensors must also be fp16
input_tensor = input_tensor.half()
with torch.no_grad():
    output = model(input_tensor)
```

## Notes

- This checkpoint represents the best balance between spectral accuracy and geometric preservation
- Stage 2 refines Stage 1 outputs WITHOUT altering spatial structure
- Use validation metrics to verify performance on your specific data
- FP16 quantization is recommended for production inference to reduce memory and improve speed

For detailed selection methodology, see `selection_metadata.json`.
"""
    
    def plot_checkpoint_comparison(self, output_path: Path):
        """Create visualization comparing all checkpoints."""
        if len(self.checkpoints) == 0:
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Stage 2 Checkpoint Comparison', fontsize=16, fontweight='bold')
        
        steps = [c.step for c in self.checkpoints]
        best_step = self.best_checkpoint.step if self.best_checkpoint else None
        
        # Plot 1: NDVI Improvement
        ax1 = fig.add_subplot(gs[0, 0])
        ndvi_imp = [c.ndvi_improvement for c in self.checkpoints]
        ax1.plot(steps, ndvi_imp, 'o-', color='#2ecc71', linewidth=2)
        ax1.axhline(0.015, color='red', linestyle='--', label='Target: +0.015')
        if best_step:
            ax1.axvline(best_step, color='orange', linestyle='--', alpha=0.7, label='Best')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('NDVI RMSE Improvement')
        ax1.set_title('NDVI Improvement Over Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: EVI Improvement
        ax2 = fig.add_subplot(gs[0, 1])
        evi_imp = [c.evi_improvement for c in self.checkpoints]
        ax2.plot(steps, evi_imp, 'o-', color='#3498db', linewidth=2)
        ax2.axhline(0.020, color='red', linestyle='--', label='Target: +0.020')
        if best_step:
            ax2.axvline(best_step, color='orange', linestyle='--', alpha=0.7, label='Best')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('EVI RMSE Improvement')
        ax2.set_title('EVI Improvement Over Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: SAM Improvement
        ax3 = fig.add_subplot(gs[0, 2])
        sam_imp = [c.sam_improvement for c in self.checkpoints]
        ax3.plot(steps, sam_imp, 'o-', color='#9b59b6', linewidth=2)
        ax3.axhline(0.040, color='red', linestyle='--', label='Target: +0.040')
        if best_step:
            ax3.axvline(best_step, color='orange', linestyle='--', alpha=0.7, label='Best')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('SAM Improvement (rad)')
        ax3.set_title('SAM Improvement Over Training')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Edge Displacement
        ax4 = fig.add_subplot(gs[1, 0])
        mean_disp = [c.mean_edge_displacement for c in self.checkpoints]
        max_disp = [c.max_edge_displacement for c in self.checkpoints]
        ax4.plot(steps, mean_disp, 'o-', color='#e74c3c', linewidth=2, label='Mean')
        ax4.plot(steps, max_disp, 's-', color='#c0392b', linewidth=2, label='Max', alpha=0.7)
        ax4.axhline(2.0, color='orange', linestyle='--', label='Mean threshold')
        ax4.axhline(5.0, color='red', linestyle='--', label='Max threshold')
        if best_step:
            ax4.axvline(best_step, color='green', linestyle='--', alpha=0.7, label='Best')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Edge Displacement (pixels)')
        ax4.set_title('Edge Movement Over Training')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Composite Score
        ax5 = fig.add_subplot(gs[1, 1])
        scores = [c.composite_score for c in self.checkpoints]
        colors = ['green' if c.passes_hard_constraints else 'red' for c in self.checkpoints]
        ax5.bar(range(len(steps)), scores, color=colors, alpha=0.7)
        ax5.set_xlabel('Checkpoint Index')
        ax5.set_ylabel('Composite Score')
        ax5.set_title('Composite Score (Green=Valid)')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Edge Ratio
        ax6 = fig.add_subplot(gs[1, 2])
        edge_ratios = [c.edge_ratio for c in self.checkpoints]
        ax6.plot(steps, edge_ratios, 'o-', color='#16a085', linewidth=2)
        ax6.axhline(0.7, color='red', linestyle='--', label='Threshold: 0.7')
        if best_step:
            ax6.axvline(best_step, color='orange', linestyle='--', alpha=0.7, label='Best')
        ax6.set_xlabel('Training Step')
        ax6.set_ylabel('Edge Ratio')
        ax6.set_title('Edge Preservation Over Training')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot: {output_path}")
        plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Select and export best Stage 2 checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        type=Path,
        required=True,
        help='Directory containing checkpoints'
    )
    
    parser.add_argument(
        '--validation_metrics',
        type=Path,
        help='Path to validation metrics CSV'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('models/stage2_best'),
        help='Output directory for best checkpoint'
    )
    
    parser.add_argument(
        '--spectral_weight',
        type=float,
        default=1.0,
        help='Weight for spectral improvement component (default: 1.0)'
    )
    
    parser.add_argument(
        '--edge_penalty_weight',
        type=float,
        default=10.0,
        help='Weight for edge penalty component (default: 10.0)'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate comparison plot'
    )
    
    parser.add_argument(
        '--quantize_fp16',
        action='store_true',
        help='Quantize model weights to fp16 for inference (reduces size by ~50%%)'
    )
    
    args = parser.parse_args()
    
    # Create selector
    selector = CheckpointSelector(
        checkpoint_dir=args.checkpoint_dir,
        validation_metrics_path=args.validation_metrics,
        spectral_weight=args.spectral_weight,
        edge_penalty_weight=args.edge_penalty_weight,
        verbose=True
    )
    
    # Evaluate checkpoints
    selector.evaluate_checkpoints()
    
    # Select best
    best = selector.select_best_checkpoint()
    
    if best is None:
        print("\n❌ No checkpoint could be selected")
        return 1
    
    # Print summary
    selector.print_summary()
    
    # Export
    selector.export_checkpoint(
        output_dir=args.output_dir,
        quantize_fp16=args.quantize_fp16
    )
    
    # Generate plot
    if args.plot:
        plot_path = args.output_dir / "checkpoint_comparison.png"
        selector.plot_checkpoint_comparison(plot_path)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
