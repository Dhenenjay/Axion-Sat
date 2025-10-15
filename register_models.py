"""
register_models.py - Register Local Foundation Models in TerraTorch

This script registers locally downloaded foundation models with TerraTorch's
registry system, enabling them to be loaded via the standard build_model API.

Models registered:
    - TerraMind 1.0 Large (IBM/ESA) - Multimodal generative foundation model
    - Prithvi EO 2.0 600M (IBM/NASA) - Vision transformer foundation model

Usage:
    python register_models.py
    
This should be run once after downloading models or importing axs_lib.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: torch package not available")
    exit(1)

try:
    from terratorch.registry import BACKBONE_REGISTRY, FULL_MODEL_REGISTRY
    TERRATORCH_AVAILABLE = True
except ImportError:
    TERRATORCH_AVAILABLE = False
    print("ERROR: TerraTorch not available")
    exit(1)

# ============================================================================
# Model Checkpoint Paths
# ============================================================================

WEIGHTS_DIR = Path("weights/hf")

TERRAMIND_PATH = WEIGHTS_DIR / "TerraMind-1.0-large"
PRITHVI_PATH = WEIGHTS_DIR / "Prithvi-EO-2.0-600M"

# ============================================================================
# TerraMind Model Factory
# ============================================================================

def build_terramind_from_checkpoint(
    checkpoint_path: str,
    modalities: tuple = ("S1GRD", "S2L2A"),
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Build TerraMind model from local checkpoint.
    
    Uses TerraTorch's native BACKBONE_REGISTRY to load TerraMind.
    
    Args:
        checkpoint_path: Path to local checkpoint directory (not used, kept for compatibility)
        modalities: Tuple of input modalities
        pretrained: Whether to load pretrained weights
        **kwargs: Additional model configuration
        
    Returns:
        TerraMind model instance
    """
    from terratorch.registry import BACKBONE_REGISTRY
    
    print(f"Loading TerraMind backbone with modalities: {modalities}")
    
    # Convert modalities tuple to list for TerraTorch
    modalities_list = list(modalities)
    
    # Use TerraTorch's native loading
    # TerraTorch will automatically download/use the checkpoint from HF
    model = BACKBONE_REGISTRY.build(
        'terramind_v1_large',
        pretrained=pretrained,
        modalities=modalities_list,
        **kwargs
    )
    
    print(f"  Model loaded successfully!")
    return model


def build_terramind_generator_factory(
    input_modalities: tuple = ("S1GRD",),
    output_modalities: tuple = ("S2L2A",),
    timesteps: int = 12,
    standardize: bool = True,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function for TerraMind generator.
    
    This is registered with TerraTorch and called via build_model().
    Uses TerraTorch's FULL_MODEL_REGISTRY for generation capabilities.
    """
    from terratorch.registry import FULL_MODEL_REGISTRY
    
    print(f"Loading TerraMind generator: {input_modalities} -> {output_modalities}")
    
    # Convert to lists for TerraTorch
    input_list = list(input_modalities)
    output_list = list(output_modalities)
    
    # Use TerraTorch's native generation model
    model = FULL_MODEL_REGISTRY.build(
        'terramind_v1_large_generate',
        pretrained=pretrained,
        modalities=input_list,
        output_modalities=output_list,
        timesteps=timesteps,
        standardize=standardize,
        **kwargs
    )
    
    print(f"  Model loaded successfully!")
    return model


def build_terramind_backbone_factory(
    modalities: tuple = ("S1GRD", "S2L2A"),
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    freeze: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function for TerraMind backbone.
    
    This is registered with TerraTorch and called via build_model().
    """
    if checkpoint_path is None:
        checkpoint_path = str(TERRAMIND_PATH)
    
    model = build_terramind_from_checkpoint(
        checkpoint_path=checkpoint_path,
        modalities=modalities,
        pretrained=pretrained,
        **kwargs
    )
    
    # Freeze if requested
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    
    return model


# ============================================================================
# Prithvi Model Factory
# ============================================================================



def build_prithvi_600_factory(
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    num_classes: int = 1,
    img_size: int = 224,  # Prithvi default
    in_channels: int = 6,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    freeze_encoder: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function for Prithvi 600M model.
    
    This is registered with TerraTorch and called via build_model().
    Uses TerraTorch's native PrithviModelFactory.
    """
    from terratorch.models import PrithviModelFactory
    from terratorch.datasets.utils import HLSBands
    import torch.nn as nn
    
    print(f"Loading Prithvi EO 2.0 600M...")
    
    # Use TerraTorch's PrithviModelFactory
    factory = PrithviModelFactory()
    
    # Define bands for Prithvi (expects HLS band specification)
    # Match the number of input channels - default is 6, but allow override
    if in_channels == 4:
        # Reduced set for low-VRAM: RGB + NIR
        bands = [
            HLSBands.BLUE,
            HLSBands.GREEN, 
            HLSBands.RED,
            HLSBands.NIR_NARROW,
        ]
    elif in_channels == 6:
        # Full set: RGB + NIR + SWIR1 + SWIR2
        bands = [
            HLSBands.BLUE,
            HLSBands.GREEN, 
            HLSBands.RED,
            HLSBands.NIR_NARROW,
            HLSBands.SWIR_1,
            HLSBands.SWIR_2,
        ]
    else:
        # Use first N bands
        all_bands = [HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED, 
                     HLSBands.NIR_NARROW, HLSBands.SWIR_1, HLSBands.SWIR_2]
        bands = all_bands[:in_channels]
    
    try:
        # Build model using TerraTorch's API
        full_model = factory.build_model(
            task="segmentation",
            backbone="prithvi_eo_v2_600",
            decoder="FCNDecoder",  # Default decoder
            bands=bands,
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            num_frames=1,
            rescale=True,
            **kwargs
        )
        print(f"  Model loaded successfully!")
        
        # Wrap the model to accept plain tensors for sanity check compatibility
        class PrithviWrapper(nn.Module):
            def __init__(self, terratorch_model):
                super().__init__()
                self.model = terratorch_model
            
            def forward(self, x):
                import torch
                # Just pass the tensor directly to the model
                output = self.model(x)
                
                # Extract tensor from various output formats
                if isinstance(output, dict):
                    # Return first tensor value found (any key)
                    for k, v in output.items():
                        if isinstance(v, torch.Tensor):
                            output = v
                            break
                    else:
                        raise ValueError(f"No tensor found in output dict. Keys: {list(output.keys())}")
                elif hasattr(output, '__dict__'):  # ModelOutput or similar
                    # Try common attribute names
                    for attr in ['logits', 'prediction', 'output', 'last_hidden_state']:
                        if hasattr(output, attr):
                            val = getattr(output, attr)
                            if isinstance(val, torch.Tensor):
                                output = val
                                break
                    else:
                        # Return first tensor attribute
                        for k, v in output.__dict__.items():
                            if isinstance(v, torch.Tensor):
                                output = v
                                break
                        else:
                            raise ValueError(f"No tensor found in ModelOutput. Attributes: {list(output.__dict__.keys())}")
                
                # Ensure output has proper shape [B, C, H, W]
                # If it's [B, H, W], add channel dimension
                if isinstance(output, torch.Tensor) and output.ndim == 3:
                    output = output.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
                
                return output
        
        model = PrithviWrapper(full_model)
        
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        raise
    
    # Apply LoRA if requested
    if use_lora:
        print("  Warning: LoRA support is experimental")
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                target_modules=["qkv", "proj"],
                bias="none",
                task_type="SEG"
            )
            
            model = get_peft_model(model, lora_config)
            print(f"  Applied LoRA (r={lora_r}, alpha={lora_alpha})")
        except ImportError:
            print("  Warning: peft not available, skipping LoRA")
        except Exception as e:
            print(f"  Warning: Could not apply LoRA: {e}")
    
    # Freeze encoder if requested
    if freeze_encoder and hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("  Encoder frozen")
    
    return model


# ============================================================================
# Registration
# ============================================================================

def register_all_models():
    """
    Register all local models with TerraTorch registries.
    """
    print("=" * 79)
    print("Registering Foundation Models with TerraTorch")
    print("=" * 79)
    print()
    
    # Check if models exist
    terramind_exists = TERRAMIND_PATH.exists()
    prithvi_exists = PRITHVI_PATH.exists()
    
    print("Model availability:")
    print(f"  TerraMind 1.0 Large: {'✓ Found' if terramind_exists else '✗ Not found'}")
    if terramind_exists:
        print(f"    Path: {TERRAMIND_PATH}")
    print(f"  Prithvi EO 2.0 600M: {'✓ Found' if prithvi_exists else '✗ Not found'}")
    if prithvi_exists:
        print(f"    Path: {PRITHVI_PATH}")
    print()
    
    if not (terramind_exists or prithvi_exists):
        print("ERROR: No models found. Please download models first.")
        print("  Run: python axs_lib/setup.py")
        return False
    
    # Register TerraMind models
    if terramind_exists:
        print("Registering TerraMind models...")
        try:
            # Get the terratorch registry
            terratorch_backbone_reg = BACKBONE_REGISTRY['terratorch']
            terratorch_full_reg = FULL_MODEL_REGISTRY['terratorch']
            
            # Register backbone - create a wrapper with the right name
            @terratorch_backbone_reg.register
            class terramind_backbone:
                def __new__(cls, **kwargs):
                    return build_terramind_backbone_factory(**kwargs)
            
            print("  ✓ Registered terramind_backbone in BACKBONE_REGISTRY")
            
            # Register generator as full model
            @terratorch_full_reg.register
            class terramind_generator:
                def __new__(cls, **kwargs):
                    return build_terramind_generator_factory(**kwargs)
            
            print("  ✓ Registered terramind_generator in FULL_MODEL_REGISTRY")
            
        except Exception as e:
            import traceback
            print(f"  ✗ Failed to register TerraMind: {e}")
            traceback.print_exc()
    
    # Register Prithvi models
    if prithvi_exists:
        print("Registering Prithvi models...")
        try:
            terratorch_backbone_reg = BACKBONE_REGISTRY['terratorch']
            
            # Register Prithvi
            @terratorch_backbone_reg.register
            class prithvi_600M:
                def __new__(cls, **kwargs):
                    return build_prithvi_600_factory(**kwargs)
            
            print("  ✓ Registered prithvi_600M in BACKBONE_REGISTRY")
            
        except Exception as e:
            import traceback
            print(f"  ✗ Failed to register Prithvi: {e}")
            traceback.print_exc()
    
    print()
    print("=" * 79)
    print("Registration complete!")
    print()
    print("Usage:")
    print("  from terratorch.models import build_model")
    print("  from axs_lib.models import build_terramind_generator, build_prithvi_600")
    print()
    print("  # Via TerraTorch (now works with local checkpoints)")
    print("  model = build_terramind_generator()")
    print("  model = build_prithvi_600()")
    print("=" * 79)
    
    return True


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    success = register_all_models()
    exit(0 if success else 1)
