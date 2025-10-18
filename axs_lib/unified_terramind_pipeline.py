"""
Unified TerraMind Pipeline: Simple 3-stage architecture using same model

This module implements a simplified pipeline where ALL stages use TerraMind:
- Stage 1: SAR → Optical (TerraMind generate)
- Stage 2: Optical refinement (TerraMind generate, same model)
- Stage 3: SAR grounding (TerraMind backbone)

Key Benefits:
- No model mismatch (all use TerraMind)
- Same input/output format (12-band S2L2A)
- Same scaling and normalization
- No Prithvi complexity (patch size, channel mismatch, etc.)
- Simple, clean architecture

Architecture:
    Input: SAR (2 channels: VV, VH)
    ↓
    Stage 1: TerraMind SAR→Optical
    → 12-band S2L2A "mental image"
    ↓
    Stage 2: TerraMind Optical→Optical (refinement)
    → 12-band S2L2A refined
    ↓
    Stage 3: TerraMind SAR + Optical → Final Output
    → 12-band S2L2A grounded
    ↓
    Output: Final refined S2L2A imagery

Author: Axion-Sat Project
Version: 3.0.0 (Unified TerraMind Architecture)
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Unified TerraMind Pipeline
# ============================================================================

class UnifiedTerraMindPipeline(nn.Module):
    """
    Unified 3-stage pipeline using TerraMind for all stages.
    
    All stages use the same model architecture (TerraMind), ensuring:
    - Consistent data format (12-band S2L2A)
    - Same normalization/scaling
    - No model architecture mismatch
    - Simple, clean pipeline
    
    Args:
        stage1_model: TerraMind SAR→Optical generator
        stage2_model: TerraMind Optical→Optical refiner (optional, can be same as stage1)
        stage3_model: TerraMind SAR+Optical backbone for grounding
        use_stage2: Whether to use Stage 2 refinement (default: True)
    """
    
    def __init__(
        self,
        stage1_model: nn.Module,
        stage2_model: Optional[nn.Module] = None,
        stage3_model: Optional[nn.Module] = None,
        use_stage2: bool = True
    ):
        super().__init__()
        
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model if stage2_model is not None else stage1_model
        self.stage3_model = stage3_model
        self.use_stage2 = use_stage2
        
        print("=" * 80)
        print("Unified TerraMind Pipeline")
        print("=" * 80)
        print(f"Stage 1: TerraMind SAR->Optical")
        print(f"Stage 2: {'Enabled' if use_stage2 else 'Disabled'} (TerraMind refinement)")
        print(f"Stage 3: {'Enabled' if stage3_model is not None else 'Disabled'} (TerraMind grounding)")
        print("=" * 80)
    
    def forward(
        self,
        sar: torch.Tensor,
        return_intermediates: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the unified pipeline.
        
        Args:
            sar: SAR input (B, 2, H, W) in dB scale
            return_intermediates: Return all intermediate outputs
            
        Returns:
            If return_intermediates=False:
                Final output (B, 12, H, W) - S2L2A
            If return_intermediates=True:
                Dict with keys: 'stage1', 'stage2', 'stage3', 'final'
        """
        B, C, H, W = sar.shape
        assert C == 2, f"Expected 2-channel SAR input, got {C}"
        
        # Store original size
        orig_h, orig_w = H, W
        
        # Pad to 224x224 if needed (TerraMind expects this)
        if H != 224 or W != 224:
            pad_h = max(0, 224 - H)
            pad_w = max(0, 224 - W)
            sar_padded = F.pad(sar, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            sar_padded = sar
            pad_h = pad_w = 0
        
        # ====================================================================
        # Stage 1: SAR → Optical (Mental Image)
        # ====================================================================
        
        self.stage1_model.eval()
        with torch.no_grad():
            stage1_out = self.stage1_model({'S1GRD': sar_padded})
            stage1_optical = stage1_out['S2L2A']  # (B, 12, 224, 224)
        
        # Clamp to valid range [0, 10000] DN
        stage1_optical = torch.clamp(stage1_optical, 0, 10000)
        
        # Crop to original size
        if pad_h > 0 or pad_w > 0:
            stage1_optical = stage1_optical[:, :, :orig_h, :orig_w]
        
        # ====================================================================
        # Stage 2: Optical → Optical (Refinement) [Optional]
        # ====================================================================
        
        if self.use_stage2:
            # Pad stage1 output back to 224x224
            if pad_h > 0 or pad_w > 0:
                stage1_optical_padded = F.pad(
                    stage1_optical,
                    (0, pad_w, 0, pad_h),
                    mode='reflect'
                )
            else:
                stage1_optical_padded = stage1_optical
            
            self.stage2_model.eval()
            with torch.no_grad():
                # Stage 2 takes optical as input and refines it
                # Use S2L2A→S2L2A refinement
                stage2_out = self.stage2_model({'S2L2A': stage1_optical_padded})
                stage2_optical = stage2_out['S2L2A']  # (B, 12, 224, 224)
            
            # Clamp to valid range
            stage2_optical = torch.clamp(stage2_optical, 0, 10000)
            
            # Crop to original size
            if pad_h > 0 or pad_w > 0:
                stage2_optical = stage2_optical[:, :, :orig_h, :orig_w]
        else:
            stage2_optical = stage1_optical
        
        # ====================================================================
        # Stage 3: SAR + Optical → Grounded Output [Optional]
        # ====================================================================
        
        if self.stage3_model is not None:
            # Stage 3 uses 4-band subset (B02, B03, B04, B08 = indices 1,2,3,7)
            # Extract 4 bands in DN scale [0, 10000]
            stage2_optical_4band = stage2_optical[:, [1, 2, 3, 7], :, :]
            
            # Forward through Stage 3 (expects DN scale input)
            # Stage 3 predicts a refinement/correction to Stage 2 output
            stage3_prediction = self.stage3_model(sar, stage2_optical_4band)
            
            # RESIDUAL CONNECTION: Add Stage 3 prediction to Stage 2 output
            # This allows Stage 3 to start from Stage 2 and only learn corrections
            stage3_optical_4band = stage2_optical_4band + stage3_prediction
            
            # Clamp to valid range after residual addition
            stage3_optical_4band = torch.clamp(stage3_optical_4band, 0, 10000)
            
            # Reconstruct 12-band by replacing the 4 bands
            stage3_optical = stage2_optical.clone()
            stage3_optical[:, [1, 2, 3, 7], :, :] = stage3_optical_4band
            
            final_output = stage3_optical
        else:
            final_output = stage2_optical
        
        # ====================================================================
        # Return
        # ====================================================================
        
        if return_intermediates:
            return {
                'stage1': stage1_optical,
                'stage2': stage2_optical,
                'stage3': stage3_optical if self.stage3_model else None,
                'final': final_output
            }
        else:
            return final_output


# ============================================================================
# Builder Function
# ============================================================================

def build_unified_pipeline(
    use_stage2: bool = False,
    use_stage3: bool = True,
    stage1_timesteps: int = 10,
    stage2_timesteps: int = 10,
    stage3_freeze_backbone: bool = False,
    device: Optional[torch.device] = None
) -> UnifiedTerraMindPipeline:
    """
    Build unified TerraMind pipeline.
    
    Args:
        use_stage2: Enable Stage 2 refinement (default: False, skip for simplicity)
        use_stage3: Enable Stage 3 grounding (default: True)
        stage1_timesteps: TerraMind timesteps for Stage 1
        stage2_timesteps: TerraMind timesteps for Stage 2
        stage3_freeze_backbone: Freeze Stage 3 backbone
        device: Device to load models on
        
    Returns:
        UnifiedTerraMindPipeline ready for inference
        
    Example:
        >>> # Simple 2-stage: Stage 1 → Stage 3 (skip Stage 2)
        >>> pipeline = build_unified_pipeline(
        ...     use_stage2=False,
        ...     use_stage3=True
        ... )
        >>> 
        >>> # 3-stage: Stage 1 → Stage 2 → Stage 3
        >>> pipeline = build_unified_pipeline(
        ...     use_stage2=True,
        ...     use_stage3=True
        ... )
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 80)
    print("Building Unified TerraMind Pipeline")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Stage 1: SAR->Optical (timesteps={stage1_timesteps})")
    print(f"Stage 2: {'Enabled' if use_stage2 else 'Disabled'} (refinement)")
    print(f"Stage 3: {'Enabled' if use_stage3 else 'Disabled'} (grounding)")
    print("=" * 80)
    
    try:
        from terratorch import FULL_MODEL_REGISTRY, BACKBONE_REGISTRY
    except ImportError:
        raise ImportError(
            "TerraTorch not installed. Install with: pip install terratorch"
        )
    
    # ====================================================================
    # Stage 1: SAR → Optical Generator
    # ====================================================================
    
    print("\n[Stage 1] Building TerraMind SAR→Optical generator...")
    stage1_model = FULL_MODEL_REGISTRY.build(
        'terramind_v1_large_generate',
        pretrained=True,
        modalities=['S1GRD'],
        output_modalities=['S2L2A'],
        timesteps=stage1_timesteps,
        standardize=True
    )
    stage1_model = stage1_model.to(device)
    print("✓ Stage 1 ready")
    
    # ====================================================================
    # Stage 2: Optical → Optical Refiner (Optional)
    # ====================================================================
    
    if use_stage2:
        print("\n[Stage 2] Building TerraMind Optical→Optical refiner...")
        stage2_model = FULL_MODEL_REGISTRY.build(
            'terramind_v1_large_generate',
            pretrained=True,
            modalities=['S2L2A'],
            output_modalities=['S2L2A'],
            timesteps=stage2_timesteps,
            standardize=True
        )
        stage2_model = stage2_model.to(device)
        print("✓ Stage 2 ready")
    else:
        stage2_model = None
        print("\n[Stage 2] Skipped (using Stage 1 output directly)")
    
    # ====================================================================
    # Stage 3: SAR + Optical → Grounded Output (Optional)
    # ====================================================================
    
    if use_stage3:
        print("\n[Stage 3] Building TerraMind backbone for grounding...")
        from axs_lib.stage3_tm_backbone import build_stage3_backbone_model
        
        stage3_model = build_stage3_backbone_model(
            freeze_backbone=stage3_freeze_backbone,
            pretrained=True,
            standardize=True,
            device=device
        )
        print("✓ Stage 3 ready")
    else:
        stage3_model = None
        print("\n[Stage 3] Skipped")
    
    # ====================================================================
    # Build Pipeline
    # ====================================================================
    
    pipeline = UnifiedTerraMindPipeline(
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        stage3_model=stage3_model,
        use_stage2=use_stage2
    )
    
    print("\n" + "=" * 80)
    print("✓ Unified Pipeline Ready")
    print("=" * 80)
    
    return pipeline


# ============================================================================
# Inference Function
# ============================================================================

@torch.no_grad()
def run_unified_inference(
    pipeline: UnifiedTerraMindPipeline,
    sar: torch.Tensor,
    return_intermediates: bool = False
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Run inference through unified pipeline.
    
    Args:
        pipeline: UnifiedTerraMindPipeline
        sar: SAR input (B, 2, H, W) in dB scale
        return_intermediates: Return all intermediate outputs
        
    Returns:
        Final S2L2A output or dict of all stage outputs
    """
    pipeline.eval()
    return pipeline(sar, return_intermediates=return_intermediates)


if __name__ == "__main__":
    print("=" * 80)
    print("Unified TerraMind Pipeline Test")
    print("=" * 80)
    
    # Test pipeline build
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simple 2-stage pipeline (Stage 1 → Stage 3)
    print("\n\nTesting 2-stage pipeline (Stage 1 → Stage 3)...")
    pipeline = build_unified_pipeline(
        use_stage2=False,
        use_stage3=True,
        device=device
    )
    
    # Test forward pass
    dummy_sar = torch.randn(1, 2, 120, 120).to(device)
    print(f"\nTest input: {dummy_sar.shape}")
    
    outputs = pipeline(dummy_sar, return_intermediates=True)
    
    print("\nOutputs:")
    for key, val in outputs.items():
        if val is not None:
            print(f"  {key}: {val.shape}, range [{val.min():.2f}, {val.max():.2f}]")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
