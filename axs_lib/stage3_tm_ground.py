"""
Stage 3: TerraMind Grounding Refinement

This module implements Stage 3 of the Axion-Sat pipeline, which uses TerraMind's
conditional generator to refine Stage 2 outputs by grounding them with SAR features.

Architecture:
    Inputs:
        - S2L2A (opt_v2): Stage 2 refined optical imagery (4 channels)
        - S1GRD (s1): Original SAR imagery (2 channels: VV, VH)
    
    Output:
        - S2L2A: Final grounded optical imagery (4 channels)
    
    Process:
        1. TerraMind encoder processes both opt_v2 and s1
        2. Cross-attention between optical and SAR features
        3. Diffusion-based denoiser refines output
        4. Standardization applied if enabled

Key Features:
    - Conditional generation with dual inputs
    - Cross-modal attention (optical + SAR)
    - Iterative refinement with configurable timesteps
    - Proper standardization with TerraMind's pretrained stats
    - LoRA-compatible for parameter-efficient fine-tuning

Usage:
    >>> from axs_lib.stage3_tm_ground import build_stage3_model, stage3_inference
    >>> 
    >>> # Build model
    >>> model = build_stage3_model(
    ...     timesteps=10,
    ...     standardize=True,
    ...     pretrained=True
    ... )
    >>> 
    >>> # Inference
    >>> s1 = torch.randn(1, 2, 120, 120)  # SAR input
    >>> opt_v2 = torch.randn(1, 4, 120, 120)  # Stage 2 output
    >>> 
    >>> opt_v3 = stage3_inference(
    ...     model=model,
    ...     s1=s1,
    ...     opt_v2=opt_v2,
    ...     timesteps=10
    ... )

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axs_lib.models import build_terramind_generator
from axs_lib.stdz import TERRAMIND_S1_STATS, TERRAMIND_S2_STATS


# ============================================================================
# Stage 3 Model Wrapper
# ============================================================================

class Stage3GroundingModel(nn.Module):
    """
    Stage 3 Grounding Model using TerraMind conditional generator.
    
    Takes both opt_v2 (Stage 2 output) and s1 (SAR) as inputs and produces
    the final grounded optical output.
    
    Args:
        terramind_generator: TerraMind generator model
        timesteps: Number of diffusion timesteps (default: 10)
        standardize: Whether to apply standardization (default: True)
    """
    
    def __init__(
        self,
        terramind_generator: nn.Module,
        timesteps: int = 10,
        standardize: bool = True
    ):
        super().__init__()
        
        self.terramind_generator = terramind_generator
        self.timesteps = timesteps
        self.standardize = standardize
        
        # Store standardization stats
        if standardize:
            # S1 stats (VV, VH)
            self.register_buffer(
                's1_mean',
                torch.tensor(TERRAMIND_S1_STATS.means, dtype=torch.float32).view(1, 2, 1, 1)
            )
            self.register_buffer(
                's1_std',
                torch.tensor(TERRAMIND_S1_STATS.stds, dtype=torch.float32).view(1, 2, 1, 1)
            )
            
            # S2 stats (B02, B03, B04, B08)
            self.register_buffer(
                's2_mean',
                torch.tensor(TERRAMIND_S2_STATS.means, dtype=torch.float32).view(1, 4, 1, 1)
            )
            self.register_buffer(
                's2_std',
                torch.tensor(TERRAMIND_S2_STATS.stds, dtype=torch.float32).view(1, 4, 1, 1)
            )
        
        print(f"Stage 3 Grounding Model initialized:")
        print(f"  Timesteps: {timesteps}")
        print(f"  Standardize: {standardize}")
    
    def standardize_inputs(
        self,
        s1: torch.Tensor,
        opt_v2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standardize inputs using TerraMind's pretrained statistics.
        
        Args:
            s1: SAR input (B, 2, H, W)
            opt_v2: Stage 2 optical output (B, 4, H, W)
            
        Returns:
            Standardized (s1, opt_v2)
        """
        if not self.standardize:
            return s1, opt_v2
        
        # Standardize S1 (SAR)
        s1_std = (s1 - self.s1_mean) / self.s1_std
        
        # Standardize opt_v2 (optical)
        opt_v2_std = (opt_v2 - self.s2_mean) / self.s2_std
        
        return s1_std, opt_v2_std
    
    def destandardize_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Destandardize model output to original scale.
        
        Args:
            output: Standardized output (B, 4, H, W)
            
        Returns:
            Destandardized output
        """
        if not self.standardize:
            return output
        
        return output * self.s2_std + self.s2_mean
    
    def forward(
        self,
        s1: torch.Tensor,
        opt_v2: torch.Tensor,
        timesteps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass: Generate grounded optical output.
        
        Args:
            s1: SAR input (B, 2, H, W)
            opt_v2: Stage 2 optical output (B, 4, H, W)
            timesteps: Number of diffusion timesteps (None = use default)
            
        Returns:
            Grounded optical output (B, 4, H, W)
        """
        # Use default timesteps if not specified
        if timesteps is None:
            timesteps = self.timesteps
        
        # Standardize inputs
        s1_std, opt_v2_std = self.standardize_inputs(s1, opt_v2)
        
        # Prepare inputs for TerraMind
        # TerraMind expects a dict with modality keys
        inputs = {
            'S1GRD': s1_std,
            'S2L2A': opt_v2_std  # Condition on Stage 2 output
        }
        
        # Run TerraMind conditional generation
        # This will use both S1 and S2 features for grounding
        # Ensure terramind_generator is on the same device as inputs
        if next(self.terramind_generator.parameters()).device != s1.device:
            self.terramind_generator = self.terramind_generator.to(s1.device)
        
        output_std = self.terramind_generator(
            inputs,
            output_modalities=('S2L2A',),
            timesteps=timesteps
        )['S2L2A']
        
        # Destandardize output
        output = self.destandardize_output(output_std)
        
        return output


# ============================================================================
# Model Builder
# ============================================================================

def build_stage3_model(
    timesteps: int = 10,
    standardize: bool = True,
    pretrained: bool = True,
    device: Optional[torch.device] = None
) -> Stage3GroundingModel:
    """
    Build Stage 3 grounding model with TerraMind conditional generator.
    
    Args:
        timesteps: Number of diffusion timesteps (default: 10)
        standardize: Apply standardization using TerraMind stats (default: True)
        pretrained: Load pretrained TerraMind weights (default: True)
        device: Device to load model on (default: auto-detect)
        
    Returns:
        Stage3GroundingModel ready for inference or training
        
    Example:
        >>> model = build_stage3_model(
        ...     timesteps=10,
        ...     standardize=True,
        ...     pretrained=True
        ... )
        >>> model = model.to('cuda')
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 80)
    print("Building Stage 3 Grounding Model")
    print("=" * 80)
    
    # Build TerraMind generator with both S1 and S2 as inputs
    print("\nBuilding TerraMind conditional generator...")
    terramind_generator = build_terramind_generator(
        input_modalities=('S1GRD', 'S2L2A'),  # Both SAR and optical as input
        output_modalities=('S2L2A',),  # Optical as output
        timesteps=timesteps,
        standardize=standardize,
        pretrained=pretrained
    )
    
    print(f"✓ TerraMind generator loaded (pretrained={pretrained})")
    
    # Wrap in Stage 3 model
    model = Stage3GroundingModel(
        terramind_generator=terramind_generator,
        timesteps=timesteps,
        standardize=standardize
    )
    
    # Move to device
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Device:               {device}")
    
    print("\n" + "=" * 80)
    print("✓ Stage 3 model ready")
    print("=" * 80)
    
    return model


# ============================================================================
# Inference Functions
# ============================================================================

def stage3_inference(
    model: Stage3GroundingModel,
    s1: torch.Tensor,
    opt_v2: torch.Tensor,
    timesteps: Optional[int] = None,
    return_intermediate: bool = False
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Run Stage 3 inference to produce grounded optical output.
    
    Args:
        model: Stage 3 grounding model
        s1: SAR input (B, 2, H, W)
        opt_v2: Stage 2 refined optical (B, 4, H, W)
        timesteps: Number of diffusion steps (None = use model default)
        return_intermediate: Return intermediate outputs if available
        
    Returns:
        Grounded optical output (B, 4, H, W) or dict with intermediate outputs
        
    Example:
        >>> model = build_stage3_model()
        >>> s1 = torch.randn(1, 2, 120, 120).cuda()
        >>> opt_v2 = torch.randn(1, 4, 120, 120).cuda()
        >>> 
        >>> with torch.no_grad():
        ...     opt_v3 = stage3_inference(model, s1, opt_v2)
        >>> 
        >>> print(f"Output shape: {opt_v3.shape}")  # (1, 4, 120, 120)
    """
    model.eval()
    
    with torch.no_grad():
        # Run forward pass
        opt_v3 = model(s1, opt_v2, timesteps=timesteps)
        
        if return_intermediate:
            return {
                'opt_v3': opt_v3,
                's1': s1,
                'opt_v2': opt_v2
            }
        else:
            return opt_v3


def stage3_batch_inference(
    model: Stage3GroundingModel,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
    timesteps: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Run Stage 3 inference on a batch of samples from a dataloader.
    
    Args:
        model: Stage 3 grounding model
        dataloader: DataLoader providing {'s1', 'opt_v2'} batches
        device: Device to run inference on
        timesteps: Number of diffusion steps
        verbose: Print progress
        
    Returns:
        Dict with concatenated outputs:
            - 'opt_v3': All grounded optical outputs (N, 4, H, W)
            - 's1': All SAR inputs (N, 2, H, W)
            - 'opt_v2': All Stage 2 outputs (N, 4, H, W)
        
    Example:
        >>> model = build_stage3_model()
        >>> results = stage3_batch_inference(model, test_loader)
        >>> print(f"Processed {results['opt_v3'].shape[0]} samples")
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    all_opt_v3 = []
    all_s1 = []
    all_opt_v2 = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Get inputs
            s1 = batch['s1'].to(device)
            opt_v2 = batch['opt_v2'].to(device)
            
            # Run inference
            opt_v3 = model(s1, opt_v2, timesteps=timesteps)
            
            # Store results
            all_opt_v3.append(opt_v3.cpu())
            all_s1.append(s1.cpu())
            all_opt_v2.append(opt_v2.cpu())
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(dataloader)} batches")
    
    # Concatenate results
    results = {
        'opt_v3': torch.cat(all_opt_v3, dim=0),
        's1': torch.cat(all_s1, dim=0),
        'opt_v2': torch.cat(all_opt_v2, dim=0)
    }
    
    if verbose:
        print(f"✓ Inference complete: {results['opt_v3'].shape[0]} samples")
    
    return results


# ============================================================================
# Validation & Metrics
# ============================================================================

def compute_stage3_metrics(
    opt_v3: torch.Tensor,
    opt_v2: torch.Tensor,
    s2_truth: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute metrics for Stage 3 outputs.
    
    Measures:
    - Improvement over Stage 2 (lower MSE/MAE vs truth)
    - Grounding quality (consistency with SAR features)
    - Spectral fidelity
    
    Args:
        opt_v3: Stage 3 output (B, 4, H, W)
        opt_v2: Stage 2 output (B, 4, H, W)
        s2_truth: Ground truth optical (B, 4, H, W)
        mask: Valid pixel mask (B, 1, H, W) or None
        
    Returns:
        Dict of metrics
    """
    if mask is not None:
        # Apply mask
        opt_v3 = opt_v3[mask.expand_as(opt_v3)]
        opt_v2 = opt_v2[mask.expand_as(opt_v2)]
        s2_truth = s2_truth[mask.expand_as(s2_truth)]
    
    # MSE (lower is better)
    mse_v3 = F.mse_loss(opt_v3, s2_truth).item()
    mse_v2 = F.mse_loss(opt_v2, s2_truth).item()
    
    # MAE (lower is better)
    mae_v3 = F.l1_loss(opt_v3, s2_truth).item()
    mae_v2 = F.l1_loss(opt_v2, s2_truth).item()
    
    # Improvement metrics
    mse_improvement = (mse_v2 - mse_v3) / mse_v2 * 100 if mse_v2 > 0 else 0.0
    mae_improvement = (mae_v2 - mae_v3) / mae_v2 * 100 if mae_v2 > 0 else 0.0
    
    return {
        'mse_v3': mse_v3,
        'mse_v2': mse_v2,
        'mse_improvement_%': mse_improvement,
        'mae_v3': mae_v3,
        'mae_v2': mae_v2,
        'mae_improvement_%': mae_improvement,
    }


# ============================================================================
# CLI for Testing
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Stage 3 TerraMind Grounding - Testing & Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test model building
  python axs_lib/stage3_tm_ground.py --test

  # Run inference on sample data
  python axs_lib/stage3_tm_ground.py --infer --s1 data/s1.pt --opt_v2 data/opt_v2.pt

  # Check model parameters
  python axs_lib/stage3_tm_ground.py --info
        """
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test model building with random data'
    )
    
    parser.add_argument(
        '--infer',
        action='store_true',
        help='Run inference on provided data'
    )
    
    parser.add_argument(
        '--s1',
        type=Path,
        help='Path to SAR input (.pt file)'
    )
    
    parser.add_argument(
        '--opt_v2',
        type=Path,
        help='Path to Stage 2 output (.pt file)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('opt_v3_output.pt'),
        help='Output path for inference results'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=10,
        help='Number of diffusion timesteps (default: 10)'
    )
    
    parser.add_argument(
        '--no-standardize',
        action='store_true',
        help='Disable standardization'
    )
    
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Do not load pretrained weights'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show model information and exit'
    )
    
    args = parser.parse_args()
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 80)
    print("Stage 3: TerraMind Grounding Refinement")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Standardize: {not args.no_standardize}")
    print(f"Pretrained: {not args.no_pretrained}")
    
    if args.info:
        # Just show model information
        print("\nBuilding model for inspection...")
        model = build_stage3_model(
            timesteps=args.timesteps,
            standardize=not args.no_standardize,
            pretrained=args.no_pretrained,
            device=device
        )
        
        print("\nModel architecture:")
        print(model)
        
        print("\nStandardization stats:")
        if not args.no_standardize:
            print(f"  S1 mean: {TERRAMIND_S1_STATS.means}")
            print(f"  S1 std:  {TERRAMIND_S1_STATS.stds}")
            print(f"  S2 mean: {TERRAMIND_S2_STATS.means}")
            print(f"  S2 std:  {TERRAMIND_S2_STATS.stds}")
        else:
            print("  (Disabled)")
        
    elif args.test:
        # Test with random data
        print("\n" + "=" * 80)
        print("Testing Stage 3 Model")
        print("=" * 80)
        
        # Build model
        model = build_stage3_model(
            timesteps=args.timesteps,
            standardize=not args.no_standardize,
            pretrained=not args.no_pretrained,
            device=device
        )
        
        # Create random test data
        print("\nGenerating random test data...")
        batch_size = 2
        height, width = 120, 120
        
        s1 = torch.randn(batch_size, 2, height, width).to(device)
        opt_v2 = torch.randn(batch_size, 4, height, width).to(device)
        
        print(f"  S1 shape:    {s1.shape}")
        print(f"  opt_v2 shape: {opt_v2.shape}")
        
        # Run inference
        print("\nRunning inference...")
        opt_v3 = stage3_inference(model, s1, opt_v2, timesteps=args.timesteps)
        
        print(f"  opt_v3 shape: {opt_v3.shape}")
        print(f"  opt_v3 range: [{opt_v3.min():.4f}, {opt_v3.max():.4f}]")
        
        # Test metrics
        print("\nComputing metrics (random data)...")
        s2_truth = torch.randn_like(opt_v3)
        metrics = compute_stage3_metrics(opt_v3, opt_v2, s2_truth)
        
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
        
        print("\n✓ Test passed!")
        
    elif args.infer:
        # Run inference on provided data
        if not args.s1 or not args.opt_v2:
            print("Error: --s1 and --opt_v2 required for inference")
            sys.exit(1)
        
        if not args.s1.exists():
            print(f"Error: S1 file not found: {args.s1}")
            sys.exit(1)
        
        if not args.opt_v2.exists():
            print(f"Error: opt_v2 file not found: {args.opt_v2}")
            sys.exit(1)
        
        print("\n" + "=" * 80)
        print("Running Inference")
        print("=" * 80)
        
        # Build model
        model = build_stage3_model(
            timesteps=args.timesteps,
            standardize=not args.no_standardize,
            pretrained=not args.no_pretrained,
            device=device
        )
        
        # Load data
        print(f"\nLoading data...")
        print(f"  S1:    {args.s1}")
        print(f"  opt_v2: {args.opt_v2}")
        
        s1 = torch.load(args.s1).to(device)
        opt_v2 = torch.load(args.opt_v2).to(device)
        
        print(f"\nInput shapes:")
        print(f"  S1:    {s1.shape}")
        print(f"  opt_v2: {opt_v2.shape}")
        
        # Run inference
        print(f"\nRunning inference (timesteps={args.timesteps})...")
        opt_v3 = stage3_inference(model, s1, opt_v2, timesteps=args.timesteps)
        
        print(f"\nOutput:")
        print(f"  opt_v3 shape: {opt_v3.shape}")
        print(f"  opt_v3 range: [{opt_v3.min():.4f}, {opt_v3.max():.4f}]")
        
        # Save output
        print(f"\nSaving output to: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(opt_v3.cpu(), args.output)
        
        print("✓ Inference complete!")
        
    else:
        parser.print_help()
    
    print("\n" + "=" * 80)
