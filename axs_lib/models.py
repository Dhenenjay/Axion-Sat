"""
axs_lib/models.py - Model Loading with TerraTorch Registries

This module provides high-level loaders for foundation models used in the
Axion-Sat pipeline, leveraging TerraTorch's model registry system.

Models:
    - TerraMind 1.0 Large: Multimodal generative foundation model (IBM/ESA)
    - Prithvi EO 2.0 600M: Vision transformer foundation model (IBM/NASA)

TerraTorch Integration:
    TerraTorch provides a unified registry for geospatial foundation models,
    handling model instantiation, weight loading, and configuration management.

Philosophy:
    TerraMind generations are "mental images" - high-quality latent 
    representations that capture cross-modal correlations. These remain 
    abstract until grounded through conditional models that map them to 
    concrete spatial predictions.
"""

from typing import Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Models cannot be loaded.")

try:
    from terratorch.registry import BACKBONE_REGISTRY, FULL_MODEL_REGISTRY
    TERRATORCH_AVAILABLE = True
    
    # Auto-register local models on import
    try:
        import sys
        from pathlib import Path
        # Add project root to path to import register_models
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from register_models import register_all_models
        # Register models silently (only print on errors)
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            register_all_models()
    except Exception as e:
        import traceback
        print(f"Warning: Could not auto-register models: {e}")
        traceback.print_exc()
        print("  Run 'python register_models.py' manually if needed.")
        
except ImportError:
    TERRATORCH_AVAILABLE = False
    print("Warning: TerraTorch not available. Using fallback implementations.")


# ============================================================================
# TerraMind Model Loaders
# ============================================================================

def build_terramind_backbone(
    modalities: Tuple[str, ...] = ("S1GRD", "S2L2A"),
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    freeze: bool = False,
    **kwargs
) -> nn.Module:
    """
    Build TerraMind backbone for feature extraction.
    
    The TerraMind backbone is a dual-scale transformer-based encoder that
    processes multiple Earth Observation modalities simultaneously. It produces
    rich semantic embeddings that capture cross-modal correlations.
    
    Architecture:
        - Modality-specific patch embeddings
        - Dual-scale transformer encoder (pixel-level + token-level)
        - Pre-trained on 500B tokens from TerraMesh dataset
        - Supports any-to-any modality combinations
    
    Args:
        modalities: Tuple of input modalities to process. Common options:
            - "S1GRD": Sentinel-1 SAR (Ground Range Detected)
            - "S2L2A": Sentinel-2 Level-2A (Multispectral)
            - "S2L1C": Sentinel-2 Level-1C (Top-of-Atmosphere)
            - "DEM": Digital Elevation Model
            - "coordinates": Geographic coordinates
        pretrained: Load pre-trained weights
        checkpoint_path: Path to custom checkpoint (overrides default)
        freeze: Freeze backbone weights (for feature extraction only)
        **kwargs: Additional arguments passed to TerraTorch model builder
        
    Returns:
        TerraMind backbone module
        
    Example:
        >>> # Sentinel-1 + Sentinel-2 backbone
        >>> backbone = build_terramind_backbone(
        ...     modalities=("S1GRD", "S2L2A"),
        ...     pretrained=True
        ... )
        >>> 
        >>> # Extract features
        >>> features = backbone({
        ...     "S1GRD": s1_tensor,
        ...     "S2L2A": s2_tensor
        ... })
        
    Note:
        TerraMind backbone outputs are feature embeddings, not final predictions.
        Use `build_terramind_generator` for generative tasks or downstream
        task-specific heads for classification/segmentation.
    """
    if not TERRATORCH_AVAILABLE:
        raise ImportError(
            "TerraTorch is required to load TerraMind models. "
            "Install with: pip install terratorch"
        )
    
    # Build via TerraTorch registry
    try:
        terratorch_backbone_reg = BACKBONE_REGISTRY['terratorch']
        # The registered factory returns a model instance directly
        # Pass only the parameters the factory expects
        model = terratorch_backbone_reg.build(
            'terramind_backbone',
            modalities=modalities,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            freeze=freeze,
            **kwargs
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to build TerraMind backbone: {e}\n"
            "Ensure TerraTorch is properly installed and model is registered."
        )
    
    # Freeze if requested
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    
    return model


def build_terramind_generator(
    input_modalities: Tuple[str, ...] = ("S1GRD",),
    output_modalities: Tuple[str, ...] = ("S2L2A",),
    timesteps: int = 12,
    standardize: bool = True,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    Build TerraMind generator for cross-modal synthesis.
    
    The TerraMind generator performs any-to-any modality translation using
    masked token reconstruction. It learns to generate high-quality latent
    representations that capture complex cross-modal correlations.
    
    IMPORTANT CONCEPTUAL NOTE:
        TerraMind generations are "mental images" - they exist in a learned
        latent space that captures the essence of the target modality without
        being directly interpretable as pixel values. These representations
        remain abstract until they are:
        
        1. Decoded through modality-specific FSQ-VAEs (for image modalities)
        2. Grounded via conditional models (Stage 3) that map latents to 
           concrete spatial predictions
        
        Think of TerraMind outputs as "thoughts" about what the target modality
        should look like, rather than the final visualization itself. This
        abstraction allows the model to reason about cross-modal relationships
        at a semantic level before committing to specific pixel values.
    
    Architecture:
        - Encoder: Processes input modalities into latent representations
        - Decoder: Generates output modality representations
        - Diffusion-based generation (configurable timesteps)
        - Optional standardization for stable training
    
    Args:
        input_modalities: Tuple of input modalities (e.g., SAR)
        output_modalities: Tuple of output modalities to generate (e.g., optical)
        timesteps: Number of diffusion timesteps (lower = faster, less quality)
            - Training: 50 (default)
            - Low VRAM: 12 (acceptable quality/speed trade-off)
            - Inference: 6-8 (fast generation)
        standardize: Apply input/output standardization
        pretrained: Load pre-trained weights
        checkpoint_path: Path to custom checkpoint
        **kwargs: Additional arguments for model configuration
        
    Returns:
        TerraMind generator module
        
    Example:
        >>> # SAR to optical synthesis
        >>> generator = build_terramind_generator(
        ...     input_modalities=("S1GRD",),
        ...     output_modalities=("S2L2A",),
        ...     timesteps=12,
        ...     standardize=True
        ... )
        >>> 
        >>> # Generate "mental image" (latent representation)
        >>> with torch.no_grad():
        ...     latent = generator({"S1GRD": sar_tensor})
        >>> 
        >>> # Latent must be grounded/decoded for visualization
        >>> # (via Stage 3 conditional model or FSQ-VAE decoder)
        
    Note:
        The generator output is NOT ready for visualization. It requires:
        1. Decoding via FSQ-VAE (for pixel-level reconstruction)
        2. Grounding via conditional model (for task-specific predictions)
        
        See `build_prithvi_600` for the refinement stage that grounds these
        abstract representations into concrete segmentation masks.
    """
    if not TERRATORCH_AVAILABLE:
        raise ImportError(
            "TerraTorch is required to load TerraMind models. "
            "Install with: pip install terratorch"
        )
    
    # Build via TerraTorch registry
    try:
        terratorch_full_reg = FULL_MODEL_REGISTRY['terratorch']
        # The registered factory returns a model instance directly
        # Pass only the parameters the factory expects
        model = terratorch_full_reg.build(
            'terramind_generator',
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            timesteps=timesteps,
            standardize=standardize,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            **kwargs
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to build TerraMind generator: {e}\n"
            "Ensure TerraTorch is properly installed and model is registered."
        )
    
    return model


# ============================================================================
# Prithvi Model Loaders
# ============================================================================

def build_prithvi_600(
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    num_classes: int = 1,
    img_size: int = 384,
    in_channels: int = 6,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    freeze_encoder: bool = False,
    **kwargs
) -> nn.Module:
    """
    Build Prithvi EO 2.0 600M foundation model for segmentation refinement.
    
    Prithvi is a vision transformer (ViT) based foundation model pre-trained
    using masked autoencoder (MAE) approach. It serves as the refinement stage
    (Stage 2) that converts TerraMind's abstract "mental images" into concrete
    per-pixel segmentation masks.
    
    Role in Pipeline:
        Prithvi acts as the bridge between TerraMind's abstract latent 
        representations and concrete spatial predictions. It:
        
        1. Receives feature maps from TerraMind (Stage 1)
        2. Applies temporal fusion and multi-scale processing
        3. Produces dense per-pixel segmentation masks
        4. Outputs refined spatial predictions for Stage 3 grounding
    
    Architecture:
        - Base: ViT with 3D patch embeddings (spatial + temporal)
        - Encoder: Pre-trained on geospatial imagery
        - Decoder: U-Net style for dense prediction
        - Modifications: Geolocation-aware, temporal fusion support
    
    LoRA Fine-tuning:
        For low-VRAM systems, LoRA (Low-Rank Adaptation) is recommended:
        - Reduces trainable parameters by 90%+
        - Maintains most of pre-trained model quality
        - Enables fine-tuning on consumer GPUs (8-12 GB VRAM)
    
    Args:
        pretrained: Load pre-trained weights from IBM/NASA
        checkpoint_path: Path to custom checkpoint
        num_classes: Number of output classes (1 for binary segmentation)
        img_size: Input image size (must match preprocessing)
        in_channels: Number of input channels (e.g., 6 for RGB+NIR+SWIR1+SWIR2)
        use_lora: Enable LoRA for parameter-efficient fine-tuning
        lora_r: LoRA rank (lower = less memory, may reduce capacity)
            - 4: Minimal (low VRAM)
            - 8: Balanced (recommended)
            - 16: High quality (more VRAM)
        lora_alpha: LoRA scaling factor (typically 2x rank)
        freeze_encoder: Freeze encoder weights (only train decoder)
        **kwargs: Additional model configuration
        
    Returns:
        Prithvi model module
        
    Example:
        >>> # Standard configuration
        >>> model = build_prithvi_600(
        ...     pretrained=True,
        ...     num_classes=1,
        ...     img_size=384,
        ...     in_channels=6
        ... )
        >>> 
        >>> # Low-VRAM configuration with LoRA
        >>> model = build_prithvi_600(
        ...     pretrained=True,
        ...     use_lora=True,
        ...     lora_r=8,
        ...     freeze_encoder=True
        ... )
        >>> 
        >>> # Generate segmentation mask
        >>> with torch.no_grad():
        ...     mask = model(features)  # From TerraMind Stage 1
        
    Note:
        Prithvi outputs are dense segmentation masks (H×W×num_classes), not
        final predictions. Stage 3 (TerraMind conditional) performs semantic
        grounding and boundary refinement before final output.
    """
    if not TERRATORCH_AVAILABLE:
        raise ImportError(
            "TerraTorch is required to load Prithvi models. "
            "Install with: pip install terratorch"
        )
    
    # Build via TerraTorch registry
    try:
        terratorch_backbone_reg = BACKBONE_REGISTRY['terratorch']
        # The registered factory returns a model instance directly
        # Pass only the parameters the factory expects
        model = terratorch_backbone_reg.build(
            'prithvi_600M',
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            img_size=img_size,
            in_channels=in_channels,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            freeze_encoder=freeze_encoder,
            **kwargs
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to build Prithvi model: {e}\n"
            "Ensure TerraTorch is properly installed and model is registered."
        )
    
    # Freeze encoder if requested
    if freeze_encoder and hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    return model


# ============================================================================
# Utility Functions
# ============================================================================

def list_available_models() -> Dict[str, List[str]]:
    """
    List all available models in TerraTorch registries.
    
    Returns:
        Dictionary mapping registry names to model lists
        
    Example:
        >>> models = list_available_models()
        >>> print(models["backbone"])
        ['terramind_1.0_large', 'prithvi_eo_2.0_600M', ...]
    """
    if not TERRATORCH_AVAILABLE:
        return {
            "backbone": ["terramind_backbone", "prithvi_600M"],
            "full_model": ["terramind_generator"],
            "note": "TerraTorch not available - showing expected models"
        }
    
    available = {}
    
    if hasattr(BACKBONE_REGISTRY, "list_models"):
        available["backbone"] = BACKBONE_REGISTRY.list_models()
    elif hasattr(BACKBONE_REGISTRY, "_registry"):
        available["backbone"] = list(BACKBONE_REGISTRY._registry.keys())
    else:
        available["backbone"] = ["terramind_backbone", "prithvi_600M"]
    
    if hasattr(FULL_MODEL_REGISTRY, "list_models"):
        available["full_model"] = FULL_MODEL_REGISTRY.list_models()
    elif hasattr(FULL_MODEL_REGISTRY, "_registry"):
        available["full_model"] = list(FULL_MODEL_REGISTRY._registry.keys())
    else:
        available["full_model"] = ["terramind_generator"]
    
    return available


def get_model_info(model_name: str) -> Dict[str, any]:
    """
    Get information about a specific model.
    
    Args:
        model_name: Name of model in TerraTorch registry
        
    Returns:
        Dictionary with model metadata
        
    Example:
        >>> info = get_model_info("terramind_1.0_large")
        >>> print(info["parameters"])
        1000000000  # 1B parameters
    """
    # Placeholder implementation
    # In practice, this would query TerraTorch metadata
    model_metadata = {
        "terramind_1.0_large": {
            "parameters": 1_000_000_000,
            "modalities": ["S1GRD", "S2L2A", "S2L1C", "DEM", "coordinates"],
            "input_size": [384, 384],
            "pretrained": True,
            "license": "Apache-2.0",
            "developers": ["IBM", "ESA", "Forschungszentrum Jülich"],
        },
        "prithvi_eo_2.0_600M": {
            "parameters": 600_000_000,
            "input_size": [384, 384],
            "in_channels": 6,
            "pretrained": True,
            "license": "Apache-2.0",
            "developers": ["IBM", "NASA"],
        }
    }
    
    return model_metadata.get(model_name, {"error": "Model not found"})


# ============================================================================
# Model Testing & Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 79)
    print("Model Loader Tests")
    print("=" * 79)
    print()
    
    # Check dependencies
    print("Dependency Check:")
    print(f"  PyTorch: {'✓ Available' if TORCH_AVAILABLE else '✗ Not available'}")
    print(f"  TerraTorch: {'✓ Available' if TERRATORCH_AVAILABLE else '✗ Not available'}")
    print()
    
    # List available models
    print("Available Models:")
    models = list_available_models()
    for registry, model_list in models.items():
        print(f"  {registry}:")
        for model in model_list:
            print(f"    - {model}")
    print()
    
    # Model info
    print("Model Information:")
    for model_name in ["terramind_1.0_large", "prithvi_eo_2.0_600M"]:
        info = get_model_info(model_name)
        print(f"  {model_name}:")
        for key, value in info.items():
            print(f"    {key}: {value}")
        print()
    
    print("=" * 79)
    print("Model loaders are ready!")
    print()
    print("Usage examples:")
    print("  from axs_lib.models import build_terramind_backbone, build_prithvi_600")
    print("  backbone = build_terramind_backbone(modalities=('S1GRD', 'S2L2A'))")
    print("  refiner = build_prithvi_600(use_lora=True, lora_r=8)")
    print("=" * 79)
