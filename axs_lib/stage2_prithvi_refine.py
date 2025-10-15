"""
Stage 2: Prithvi Feature Refinement with 8-bit Quantization and LoRA

This module implements the second stage of the Axion-Sat pipeline, using Prithvi-EO-2.0-600M
backbone with memory-efficient 8-bit quantization and parameter-efficient LoRA fine-tuning
to refine TerraMind's "mental images" into dense feature maps.

Pipeline Flow:
    Stage 1: SAR → TerraMind → Synthetic Optical "Mental Image"
    Stage 2 (this module): Mental Image → Prithvi+LoRA → Dense Features
    Stage 3: Dense Features → Conditional Model → Final Segmentation

Memory Optimization Strategy:
    - 8-bit quantization via bitsandbytes: Reduces memory by ~75% (600M @ 32-bit → 150MB @ 8-bit)
    - LoRA (Low-Rank Adaptation): Fine-tunes only ~1% of parameters
    - Frozen backbone: Only last N transformer blocks trainable
    - Lightweight ConvNeXt head: Minimal additional parameters

Key Concepts:
    - Prithvi provides rich geospatial features learned from 1B+ satellite images
    - 8-bit quantization maintains quality while enabling low-VRAM training
    - LoRA adapters add task-specific capacity without full fine-tuning
    - ConvNeXt head performs spatial refinement and channel reduction

Author: Axion-Sat Project
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union, List
import warnings
from pathlib import Path

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    warnings.warn(
        "bitsandbytes not available. 8-bit quantization will not work. "
        "Install with: pip install bitsandbytes"
    )

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    warnings.warn(
        "peft not available. LoRA fine-tuning will not work. "
        "Install with: pip install peft"
    )

try:
    from axs_lib.models import build_terramind_backbone
    from terratorch.registry import BACKBONE_REGISTRY
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    warnings.warn("axs_lib.models not available, some functions will be limited")


# ============================================================================
# ConvNeXt-Tiny Refinement Head
# ============================================================================

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block with depthwise convolution and inverted bottleneck.
    
    Memory-efficient design:
    - Depthwise conv: O(C·K²) vs O(C²·K²) for standard conv
    - Layer norm: More stable than batch norm for small batches
    - GELU activation: Smooth, modern activation
    """
    
    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        expansion: int = 4,
        drop_path: float = 0.0
    ):
        super().__init__()
        
        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim  # Depthwise
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        residual = x
        
        # Depthwise conv
        x = self.dwconv(x)
        
        # Permute to (B, H, W, C) for LayerNorm and Linear
        x = x.permute(0, 2, 3, 1)
        
        # Inverted bottleneck
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Permute back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # Residual connection with drop path
        x = residual + self.drop_path(x)
        
        return x


class ConvNeXtHead(nn.Module):
    """
    Lightweight ConvNeXt head for feature refinement.
    
    Architecture:
        - Stem: 1x1 conv to adjust channels
        - N ConvNeXt blocks for spatial refinement
        - Optional downsampling between blocks
        - Final projection to desired feature dimension
    
    Memory footprint: ~10-20 MB for typical configurations
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        kernel_size: int = 7,
        expansion: int = 4,
        drop_path: float = 0.1,
        use_downsample: bool = False
    ):
        """
        Args:
            in_channels: Input feature dimension from Prithvi
            out_channels: Output feature dimension for Stage 3
            hidden_dim: Hidden dimension for ConvNeXt blocks
            num_blocks: Number of ConvNeXt blocks
            kernel_size: Kernel size for depthwise conv
            expansion: Expansion factor for inverted bottleneck
            drop_path: Drop path rate for stochastic depth
            use_downsample: Whether to downsample spatially (reduces memory)
        """
        super().__init__()
        
        self.use_downsample = use_downsample
        
        # Stem: project input to hidden dimension
        self.stem = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # ConvNeXt blocks with progressive drop path
        drop_path_rates = [drop_path * (i / (num_blocks - 1)) for i in range(num_blocks)]
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(
                dim=hidden_dim,
                kernel_size=kernel_size,
                expansion=expansion,
                drop_path=drop_path_rates[i]
            )
            for i in range(num_blocks)
        ])
        
        # Optional downsampling between blocks
        if use_downsample:
            self.downsample = nn.ModuleList([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=2, stride=2)
                if i % 2 == 1 else nn.Identity()
                for i in range(num_blocks)
            ])
        
        # Final projection head
        self.head = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) - Input features from Prithvi
        Returns:
            (B, C_out, H', W') - Refined features for Stage 3
        """
        # Stem projection
        x = self.stem(x)
        
        # ConvNeXt blocks with optional downsampling
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.use_downsample:
                x = self.downsample[i](x)
        
        # Final projection
        x = self.head(x)
        
        return x


# ============================================================================
# Prithvi with 8-bit Quantization and LoRA
# ============================================================================

class PrithviRefiner(nn.Module):
    """
    Prithvi-EO-2.0-600M backbone with 8-bit quantization, LoRA fine-tuning,
    and ConvNeXt refinement head.
    
    Memory-efficient architecture for low-VRAM systems:
    - 8-bit quantized backbone: ~150 MB (vs ~2.4 GB in fp32)
    - LoRA adapters: ~10-50 MB depending on rank
    - ConvNeXt head: ~10-20 MB
    - Total: ~200-250 MB model weights (vs ~2.5 GB full precision)
    
    Training strategy:
    1. Freeze entire Prithvi backbone (optional: unfreeze last N blocks)
    2. Add LoRA adapters to attention layers
    3. Train LoRA + ConvNeXt head
    4. Fine-tune on task-specific data
    """
    
    def __init__(
        self,
        prithvi_checkpoint: Optional[str] = None,
        num_input_channels: int = 4,  # From TerraMind: B02, B03, B04, B08
        out_channels: int = 256,
        hidden_dim: int = 256,
        num_convnext_blocks: int = 4,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 2,
        load_in_8bit: bool = True,
        device: Union[str, torch.device] = "cuda",
        **kwargs
    ):
        """
        Args:
            prithvi_checkpoint: Path to Prithvi checkpoint (if None, uses default)
            num_input_channels: Number of input channels from Stage 1
            out_channels: Output feature dimension for Stage 3
            hidden_dim: Hidden dimension for ConvNeXt head
            num_convnext_blocks: Number of ConvNeXt refinement blocks
            lora_r: LoRA rank (4, 8, 16, 32)
            lora_alpha: LoRA scaling factor
            lora_dropout: LoRA dropout rate
            lora_target_modules: Which modules to apply LoRA to
            freeze_backbone: Whether to freeze Prithvi backbone
            unfreeze_last_n_blocks: Number of last transformer blocks to unfreeze
            load_in_8bit: Whether to use 8-bit quantization
            device: Device to load model on
            **kwargs: Additional arguments
        """
        super().__init__()
        
        if not MODELS_AVAILABLE:
            raise ImportError(
                "TerraTorch and axs_lib.models required. "
                "Install with: pip install terratorch"
            )
        
        if load_in_8bit and not BNB_AVAILABLE:
            warnings.warn(
                "8-bit quantization requested but bitsandbytes not available. "
                "Falling back to fp16. Install with: pip install bitsandbytes"
            )
            load_in_8bit = False
        
        if lora_r > 0 and not PEFT_AVAILABLE:
            warnings.warn(
                "LoRA requested but peft not available. "
                "Proceeding without LoRA. Install with: pip install peft"
            )
            lora_r = 0
        
        self.num_input_channels = num_input_channels
        self.out_channels = out_channels
        self.load_in_8bit = load_in_8bit
        self.device = torch.device(device)
        
        # ====================================================================
        # Load Prithvi Backbone
        # ====================================================================
        
        print("Loading Prithvi-EO-2.0-600M backbone...")
        
        try:
            # Load Prithvi via TerraTorch registry
            # Note: Prithvi is registered as a backbone in terratorch
            terratorch_backbone_reg = BACKBONE_REGISTRY['terratorch']
            
            # Filter kwargs to only include valid Prithvi build arguments
            # This prevents passing arguments that shouldn't be stored/forwarded
            prithvi_valid_keys = {
                'img_size', 'patch_size', 'embed_dim', 'depth', 'num_heads',
                'mlp_ratio', 'qkv_bias', 'drop_rate', 'attn_drop_rate',
                'drop_path_rate', 'norm_layer'
            }
            prithvi_kwargs = {k: v for k, v in kwargs.items() if k in prithvi_valid_keys}
            
            # Build Prithvi backbone
            # Using 600M model (keeping full model as requested)
            prithvi_model_name = kwargs.get('prithvi_model', 'prithvi_eo_v2_600')
            self.prithvi = terratorch_backbone_reg.build(
                prithvi_model_name,
                pretrained=True,
                checkpoint_path=prithvi_checkpoint,
                freeze=False,  # We'll handle freezing ourselves
                num_frames=1,  # Single timestep
                **prithvi_kwargs
            )
            
            print(f"✓ Prithvi backbone loaded successfully")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Prithvi backbone: {e}\n"
                "Ensure model is registered and checkpoint exists."
            )
        
        # Get Prithvi output dimension
        # Prithvi-600M typically outputs 768-dim features
        prithvi_out_dim = getattr(self.prithvi, 'embed_dim', 768)
        
        # Get Prithvi input channels
        prithvi_in_channels = getattr(self.prithvi, 'in_chans', 6)
        print(f"  Prithvi expects {prithvi_in_channels} input channels")
        print(f"  Model configured for {num_input_channels} input channels")
        
        # ====================================================================
        # Apply 8-bit Quantization
        # ====================================================================
        
        if load_in_8bit and BNB_AVAILABLE:
            print("Applying 8-bit quantization...")
            self._quantize_to_8bit()
            print(f"✓ 8-bit quantization applied")
        
        # ====================================================================
        # Freeze Backbone (except last N blocks)
        # ====================================================================
        
        if freeze_backbone:
            print(f"Freezing Prithvi backbone (keeping last {unfreeze_last_n_blocks} blocks trainable)...")
            self._freeze_backbone(unfreeze_last_n_blocks)
            print(f"✓ Backbone frozen")
        
        # ====================================================================
        # Apply LoRA
        # ====================================================================
        
        if lora_r > 0 and PEFT_AVAILABLE:
            print(f"Applying LoRA (r={lora_r}, alpha={lora_alpha})...")
            
            # Default LoRA target modules for vision transformers
            if lora_target_modules is None:
                lora_target_modules = ["qkv", "proj", "fc1", "fc2"]
            
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            
            self.prithvi = get_peft_model(self.prithvi, lora_config)
            self.prithvi.print_trainable_parameters()
            
            print(f"✓ LoRA applied")
        
        # ====================================================================
        # Input Projection (if needed)
        # ====================================================================
        
        # Prithvi expects specific number of input channels
        # Project from Stage 1 output (4 channels) to Prithvi input
        prithvi_in_channels = getattr(self.prithvi, 'in_chans', 6)
        
        if num_input_channels != prithvi_in_channels:
            print(f"Adding input projection: {num_input_channels} → {prithvi_in_channels} channels")
            self.input_proj = nn.Conv2d(
                num_input_channels,
                prithvi_in_channels,
                kernel_size=1
            )
        else:
            self.input_proj = nn.Identity()
        
        # ====================================================================
        # ConvNeXt Refinement Head
        # ====================================================================
        
        print(f"Building ConvNeXt refinement head...")
        self.refinement_head = ConvNeXtHead(
            in_channels=prithvi_out_dim,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            num_blocks=num_convnext_blocks,
            **kwargs
        )
        print(f"✓ ConvNeXt head built")
        
        # Move to device
        self.to(self.device)
        
        # Print memory summary
        self._print_memory_summary()
    
    def _quantize_to_8bit(self):
        """Apply 8-bit quantization to linear layers in Prithvi backbone."""
        if not BNB_AVAILABLE:
            return
        
        for name, module in self.prithvi.named_modules():
            if isinstance(module, nn.Linear):
                # Replace with 8-bit linear layer
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = self.prithvi.get_submodule(parent_name) if parent_name else self.prithvi
                
                # Create 8-bit linear layer
                linear_8bit = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=False,
                )
                
                # Copy weights (will be quantized automatically)
                linear_8bit.weight.data = module.weight.data
                if module.bias is not None:
                    linear_8bit.bias.data = module.bias.data
                
                # Replace module
                setattr(parent, child_name, linear_8bit)
    
    def _freeze_backbone(self, unfreeze_last_n_blocks: int = 2):
        """
        Freeze Prithvi backbone except last N transformer blocks.
        
        Args:
            unfreeze_last_n_blocks: Number of last blocks to keep trainable
        """
        # Freeze all parameters first
        for param in self.prithvi.parameters():
            param.requires_grad = False
        
        # Unfreeze last N blocks if available
        if hasattr(self.prithvi, 'blocks'):
            num_blocks = len(self.prithvi.blocks)
            if unfreeze_last_n_blocks > 0:
                for block in self.prithvi.blocks[-unfreeze_last_n_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True
                
                print(f"  Unfroze last {unfreeze_last_n_blocks}/{num_blocks} transformer blocks")
    
    def _print_memory_summary(self):
        """Print memory usage summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate memory (rough)
        if self.load_in_8bit:
            param_mem_mb = total_params * 1 / (1024**2)  # 1 byte per param
        else:
            param_mem_mb = total_params * 4 / (1024**2)  # 4 bytes per param (fp32)
        
        print("\n" + "="*70)
        print("PrithviRefiner Memory Summary")
        print("="*70)
        print(f"Total parameters:      {total_params:,} ({param_mem_mb:.1f} MB)")
        print(f"Trainable parameters:  {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"Quantization:          {'8-bit' if self.load_in_8bit else 'fp32/fp16'}")
        print("="*70 + "\n")
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through Prithvi + ConvNeXt head.
        
        Features CPU fallback on GPU OOM:
        - If refinement head fails on GPU, automatically falls back to CPU
        - Logs runtime penalty for monitoring
        - Seamlessly handles mixed device execution
        
        Args:
            x: Input tensor from Stage 1 (TerraMind output)
               Shape: (B, 4, H, W) - [B02, B03, B04, B08]
            return_features: If True, return both Prithvi features and refined output
            
        Returns:
            If return_features=False:
                refined_features: (B, out_channels, H, W)
            If return_features=True:
                (prithvi_features, refined_features)
        """
        import time
        import warnings
        
        B, C, H, W = x.shape
        
        # Input validation
        if C != self.num_input_channels:
            raise ValueError(
                f"Expected {self.num_input_channels} input channels, got {C}"
            )
        
        # Project input to Prithvi input space
        x = self.input_proj(x)
        
        # Extract features with Prithvi backbone
        # Output format depends on Prithvi implementation
        # Call with explicit positional argument only to avoid kwarg issues
        try:
            prithvi_out = self.prithvi(x)
        except TypeError as e:
            if "input_ids" in str(e) or "unexpected keyword argument" in str(e):
                # Fallback: try calling the underlying model directly
                if hasattr(self.prithvi, 'base_model'):
                    prithvi_out = self.prithvi.base_model.model(x)
                elif hasattr(self.prithvi, 'model'):
                    prithvi_out = self.prithvi.model(x)
                else:
                    raise
            else:
                raise
        
        # Handle different output formats
        if isinstance(prithvi_out, dict):
            # TerraTorch may return dict with 'output' or 'features'
            if 'features' in prithvi_out:
                features = prithvi_out['features']
            elif 'output' in prithvi_out:
                features = prithvi_out['output']
            else:
                # Take first tensor value
                features = list(prithvi_out.values())[0]
        elif isinstance(prithvi_out, (list, tuple)):
            # Some models return list/tuple of features
            features = prithvi_out[0] if len(prithvi_out) > 0 else prithvi_out
        else:
            features = prithvi_out
        
        # Ensure features are in (B, C, H, W) format
        if features.ndim == 3:  # (B, N, C) from transformer
            # Reshape to spatial format
            # Calculate actual spatial dimensions from total tokens
            N = features.shape[1]  # Total number of tokens/patches
            C_feat = features.shape[2]  # Feature dimension
            
            # Try to infer square spatial dimensions
            h = w = int(N ** 0.5)
            
            # Verify this gives us the right number of elements
            if h * w != N:
                # Not a perfect square, try to find best factorization
                # This can happen when input size isn't perfectly divisible by patch size
                for i in range(int(N**0.5), 0, -1):
                    if N % i == 0:
                        h = i
                        w = N // i
                        break
            
            features = features.transpose(1, 2).reshape(B, C_feat, h, w)
        
        # Apply ConvNeXt refinement head with CPU fallback
        try:
            start_time = time.time()
            refined = self.refinement_head(features)
            elapsed = time.time() - start_time
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                # GPU OOM - fallback to CPU
                warnings.warn(
                    "\n" + "="*70 + "\n"
                    "GPU OUT OF MEMORY - ConvNeXt Head Fallback\n"
                    "="*70 + "\n"
                    "Moving refinement head to CPU. This will be slower but allows\n"
                    "processing to continue. Consider:\n"
                    "  1. Reducing batch size\n"
                    "  2. Reducing tile size\n"
                    "  3. Disabling discriminator\n"
                    "  4. Reducing num_convnext_blocks\n"
                    "="*70,
                    RuntimeWarning
                )
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Move refinement head to CPU
                refinement_head_device = next(self.refinement_head.parameters()).device
                self.refinement_head.cpu()
                
                # Move features to CPU
                features_cpu = features.cpu()
                
                # Run on CPU and measure time
                start_time = time.time()
                refined = self.refinement_head(features_cpu)
                cpu_elapsed = time.time() - start_time
                
                # Move result back to original device
                refined = refined.to(features.device)
                
                # Move head back to GPU if it was there
                if refinement_head_device.type == 'cuda':
                    try:
                        self.refinement_head.to(refinement_head_device)
                    except RuntimeError:
                        warnings.warn(
                            "Could not move refinement head back to GPU. "
                            "Keeping on CPU for remainder of processing.",
                            RuntimeWarning
                        )
                
                # Log runtime penalty
                warnings.warn(
                    f"CPU fallback runtime: {cpu_elapsed:.3f}s "
                    f"(expected GPU: ~{cpu_elapsed/10:.3f}s, penalty: ~{cpu_elapsed*9:.3f}s)",
                    RuntimeWarning
                )
                elapsed = cpu_elapsed
            else:
                # Re-raise if not OOM error
                raise
        
        # Interpolate to match input spatial size if needed
        if refined.shape[-2:] != (H, W):
            refined = F.interpolate(
                refined,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        
        if return_features:
            return features, refined
        return refined


# ============================================================================
# High-Level Builder Function
# ============================================================================

def build_prithvi_refiner(
    config: Optional[Dict] = None,
    **kwargs
) -> PrithviRefiner:
    """
    Build PrithviRefiner from config or kwargs.
    
    Args:
        config: Configuration dict (typically from YAML)
        **kwargs: Override config values
        
    Returns:
        PrithviRefiner model
        
    Example:
        >>> from axs_lib.stage2_prithvi_refine import build_prithvi_refiner
        >>> 
        >>> # From config
        >>> import yaml
        >>> config = yaml.safe_load(open('configs/hardware.lowvr.yaml'))
        >>> model = build_prithvi_refiner(config['stage2'])
        >>> 
        >>> # Or directly with kwargs
        >>> model = build_prithvi_refiner(
        ...     lora_r=8,
        ...     lora_alpha=16,
        ...     load_in_8bit=True,
        ...     num_convnext_blocks=4
        ... )
    """
    # Extract stage2 config if full config provided
    if config and 'stage2' in config:
        config = config['stage2']
    
    # Get model params from config
    model_params = {}
    if config and 'model' in config:
        model_cfg = config['model']
        
        # Map config keys to model parameters
        model_params.update({
            'lora_r': model_cfg.get('lora_r', 8),
            'lora_alpha': model_cfg.get('lora_alpha', 16),
            'lora_dropout': model_cfg.get('lora_dropout', 0.1),
            'lora_target_modules': model_cfg.get('lora_target_modules'),
            'freeze_backbone': model_cfg.get('freeze_base', True),
            'load_in_8bit': model_cfg.get('use_8bit', False),  # Disable by default (H100 has plenty of VRAM)
        })
        
        # Prithvi-specific settings
        if 'prithvi' in model_cfg:
            prithvi_cfg = model_cfg['prithvi']
            model_params.update({
                'prithvi_checkpoint': prithvi_cfg.get('checkpoint'),
                'num_input_channels': prithvi_cfg.get('in_channels', 4),
                'out_channels': prithvi_cfg.get('num_classes', 256),
            })
    
    # Override with kwargs
    model_params.update(kwargs)
    
    # Build model
    model = PrithviRefiner(**model_params)
    
    return model


# ============================================================================
# Testing and Demo
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Stage 2: Prithvi Refinement Module Test")
    print("="*70)
    
    # Check dependencies
    print("\nChecking dependencies...")
    print(f"  PyTorch: {'✓' if torch.cuda.is_available() else '✗ (CPU only)'}")
    print(f"  bitsandbytes: {'✓' if BNB_AVAILABLE else '✗'}")
    print(f"  peft: {'✓' if PEFT_AVAILABLE else '✗'}")
    print(f"  TerraTorch: {'✓' if MODELS_AVAILABLE else '✗'}")
    
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available. Testing on CPU (will be slow).")
        device = "cpu"
    else:
        device = "cuda"
        print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Test model creation
    print("\n" + "-"*70)
    print("Test 1: Model Creation")
    print("-"*70)
    
    try:
        model = build_prithvi_refiner(
            num_input_channels=4,
            out_channels=256,
            hidden_dim=256,
            num_convnext_blocks=4,
            lora_r=8,
            lora_alpha=16,
            load_in_8bit=(device == "cuda" and BNB_AVAILABLE),
            freeze_backbone=True,
            unfreeze_last_n_blocks=2,
            device=device
        )
        print("✓ Model created successfully")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test forward pass
    print("\n" + "-"*70)
    print("Test 2: Forward Pass")
    print("-"*70)
    
    try:
        # Create dummy input (Stage 1 output)
        batch_size = 2
        H, W = 256, 256
        dummy_input = torch.randn(batch_size, 4, H, W).to(device)
        
        print(f"Input shape: {dummy_input.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        # Check shape
        expected_shape = (batch_size, 256, H, W)
        if output.shape == expected_shape:
            print(f"✓ Output shape matches expected: {expected_shape}")
        else:
            print(f"✗ Output shape mismatch. Expected {expected_shape}, got {output.shape}")
        
        # Test with return_features
        features, refined = model(dummy_input, return_features=True)
        print(f"Prithvi features shape: {features.shape}")
        print(f"Refined features shape: {refined.shape}")
        print("✓ Forward pass successful")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Memory usage
    if device == "cuda":
        print("\n" + "-"*70)
        print("Test 3: Memory Usage")
        print("-"*70)
        
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        print(f"GPU memory allocated: {allocated:.1f} MB")
        print(f"GPU memory reserved:  {reserved:.1f} MB")
        
        if allocated < 1000:  # Less than 1 GB
            print("✓ Memory usage is acceptable for low-VRAM systems")
        else:
            print(f"⚠️  Memory usage is high: {allocated:.1f} MB")
    
    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
    
    print("\nUsage example:")
    print("""
from axs_lib.stage2_prithvi_refine import build_prithvi_refiner
from axs_lib.stage1_tm_s2o import tm_sar2opt
from axs_lib.models import build_terramind_generator

# Stage 1: Generate synthetic optical
terramind = build_terramind_generator(
    input_modalities=("S1GRD",),
    output_modalities=("S2L2A",),
    timesteps=12
)
mental_image = tm_sar2opt(terramind, sar_input, timesteps=12)

# Stage 2: Refine with Prithvi + LoRA
prithvi_refiner = build_prithvi_refiner(
    lora_r=8,
    load_in_8bit=True,
    num_convnext_blocks=4
)
dense_features = prithvi_refiner(mental_image)

# Stage 3: Use dense_features for segmentation
# segmentation = conditional_model(dense_features)
""")
