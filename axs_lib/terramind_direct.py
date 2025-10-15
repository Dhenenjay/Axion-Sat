"""
Direct TerraMind Model Loader
==============================

This module provides direct loading of TerraMind from local checkpoints,
bypassing TerraTorch's registry system when it's unavailable or incompatible.

Author: Axion-Sat Project
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
import warnings


def load_terramind_direct(
    checkpoint_path: str = "weights/hf/TerraMind-1.0-large/TerraMind_v1_large.pt",
    input_modalities: Tuple[str, ...] = ("S1GRD",),
    output_modalities: Tuple[str, ...] = ("S2L2A",),
    device: str = "cuda",
    **kwargs
) -> nn.Module:
    """
    Load TerraMind model directly from checkpoint file.
    
    This function loads the TerraMind weights directly without relying on
    TerraTorch's registry system, which may be unavailable or incompatible.
    
    Args:
        checkpoint_path: Path to TerraMind_v1_large.pt file
        input_modalities: Input modalities (default: SAR)
        output_modalities: Output modalities (default: Optical)
        device: Device to load model on
        **kwargs: Additional model configuration
        
    Returns:
        TerraMind model wrapped in a simple interface
        
    Example:
        >>> model = load_terramind_direct()
        >>> output = model({'S1GRD': sar_tensor}, num_inference_steps=3)
        >>> opt_tensor = output['S2L2A']
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"TerraMind checkpoint not found: {checkpoint_path}\\n"
            f"Please ensure the model weights are downloaded to weights/hf/TerraMind-1.0-large/"
        )
    
    print(f"Loading TerraMind from {checkpoint_path}...")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create simple wrapper model
        class TerraMindWrapper(nn.Module):
            """Simple wrapper for TerraMind checkpoint."""
            
            def __init__(self, state_dict, input_mods, output_mods):
                super().__init__()
                self.input_modalities = input_mods
                self.output_modalities = output_mods
                
                # Store state dict
                self._state_dict = state_dict
                
                # Initialize parameters from checkpoint
                for name, param in state_dict.items():
                    if isinstance(param, torch.Tensor):
                        self.register_buffer(name, param)
                
                print(f"  Loaded {len(state_dict)} parameters")
            
            def forward(self, inputs, num_inference_steps=12, **kwargs):
                """
                Forward pass through TerraMind.
                
                Args:
                    inputs: Dict with modality keys or single tensor
                    num_inference_steps: Number of diffusion steps
                    
                Returns:
                    Dict with output modality keys
                """
                # Handle single tensor input
                if isinstance(inputs, torch.Tensor):
                    inputs = {self.input_modalities[0]: inputs}
                
                # Get SAR input
                if 'S1GRD' in inputs:
                    sar = inputs['S1GRD']
                else:
                    raise ValueError(f"Expected S1GRD input, got: {list(inputs.keys())}")
                
                # For now, return identity mapping wrapped in expected format
                # This will be replaced with actual TerraMind inference
                # once we understand the checkpoint structure better
                
                warnings.warn(
                    "Using simplified TerraMind inference. "
                    "Full diffusion not yet implemented for direct loading."
                )
                
                # Simple pass-through for now
                # Real implementation would do proper diffusion inference
                output = {
                    'S2L2A': sar.repeat(1, 2, 1, 1)  # Duplicate channels as placeholder
                }
                
                return output
        
        model = TerraMindWrapper(checkpoint, input_modalities, output_modalities)
        model = model.to(device)
        model.eval()
        
        print(f"  Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load TerraMind checkpoint: {e}")


# Alias for compatibility
build_terramind_generator_direct = load_terramind_direct
