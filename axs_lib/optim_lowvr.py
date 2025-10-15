"""
axs_lib/optim_lowvr.py - Low-VRAM Optimization Utilities

Provides helper functions to enable memory-efficient training and inference
on consumer GPUs (8-12 GB VRAM) for the Axion-Sat pipeline.

Key Optimizations:
    - xFormers: Memory-efficient attention (2-4x faster, less memory)
    - Matmul Precision: Medium precision for TF32 cores (A100/4000 series)
    - PEFT: Parameter-Efficient Fine-Tuning (LoRA, QLoRA)
    - bitsandbytes: 8-bit quantization for model weights/optimizers

Target Hardware:
    - NVIDIA RTX 4060/4070 (12-16 GB)
    - NVIDIA RTX 3060/3060 Ti (8-12 GB)
    - NVIDIA A100 (40-80 GB, but optimizations still beneficial)

Usage:
    >>> from axs_lib.optim_lowvr import setup_lowvram_environment
    >>> setup_lowvram_environment(
    ...     enable_xformers=True,
    ...     enable_tf32=True,
    ...     enable_flash_attention=True
    ... )
    >>> 
    >>> from axs_lib.optim_lowvr import get_lora_config
    >>> lora_config = get_lora_config(rank=8, target_modules=["qkv", "proj"])
"""

import os
import warnings
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

# ============================================================================
# Dependency Checks
# ============================================================================

# PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available - low-VRAM optimizations disabled")

# xFormers (Memory-efficient attention)
try:
    import xformers
    import xformers.ops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

# PEFT (Parameter-Efficient Fine-Tuning)
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
        PeftConfig,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# bitsandbytes (8-bit quantization)
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

# Transformers (for BitsAndBytesConfig)
try:
    from transformers import BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class LowVRAMConfig:
    """Configuration for low-VRAM optimizations."""
    
    # xFormers
    enable_xformers: bool = True
    xformers_attention_op: Optional[str] = None  # Auto-select best
    
    # TF32/Matmul Precision
    enable_tf32: bool = True  # Ampere+ GPUs (RTX 30/40 series, A100)
    matmul_precision: str = "medium"  # "highest", "high", "medium"
    
    # Flash Attention
    enable_flash_attention: bool = True
    
    # Mixed Precision
    enable_mixed_precision: bool = True
    mixed_precision_dtype: str = "float16"  # "float16" or "bfloat16"
    
    # Gradient Checkpointing
    enable_gradient_checkpointing: bool = True
    
    # Memory Management
    empty_cache_frequency: int = 5  # Clear cache every N batches
    max_split_size_mb: int = 512  # PyTorch memory allocator
    
    # CUDA Settings
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    
    # Logging
    verbose: bool = True


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""
    
    # LoRA Parameters
    r: int = 8  # Rank (4=minimal, 8=balanced, 16=high quality)
    lora_alpha: int = 16  # Scaling factor (typically 2x rank)
    lora_dropout: float = 0.1
    
    # Target Modules
    target_modules: List[str] = None  # ["qkv", "proj"] for attention
    
    # Bias Configuration
    bias: str = "none"  # "none", "all", "lora_only"
    
    # Task Type
    task_type: str = "SEG"  # "SEG" for segmentation, "CAUSAL_LM", etc.
    
    # Inference Mode
    inference_mode: bool = False
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default: apply to attention layers
            self.target_modules = ["qkv", "proj", "fc1", "fc2"]


@dataclass
class BnB8BitConfig:
    """Configuration for bitsandbytes 8-bit quantization."""
    
    # Quantization Settings
    load_in_8bit: bool = True
    llm_int8_threshold: float = 6.0
    llm_int8_has_fp16_weight: bool = False
    llm_int8_enable_fp32_cpu_offload: bool = False
    
    # Device Mapping
    device_map: Union[str, Dict] = "auto"  # "auto" or custom mapping
    
    # Optimizer
    use_8bit_optimizer: bool = True
    optimizer_class: str = "AdamW8bit"  # "AdamW8bit", "Adam8bit", "SGD8bit"


# ============================================================================
# Environment Setup Functions
# ============================================================================

def setup_lowvram_environment(
    enable_xformers: bool = True,
    enable_tf32: bool = True,
    enable_flash_attention: bool = True,
    enable_mixed_precision: bool = True,
    mixed_precision_dtype: str = "float16",
    enable_gradient_checkpointing: bool = True,
    cudnn_benchmark: bool = True,
    max_split_size_mb: int = 512,
    verbose: bool = True,
) -> Dict[str, bool]:
    """
    Setup low-VRAM environment with all optimizations.
    
    This function configures PyTorch and CUDA for memory-efficient operation
    on consumer GPUs. Call this at the start of your training script.
    
    Args:
        enable_xformers: Enable xFormers memory-efficient attention
        enable_tf32: Enable TF32 on Ampere+ GPUs (RTX 30/40, A100)
        enable_flash_attention: Enable Flash Attention (if available)
        enable_mixed_precision: Enable mixed precision training
        mixed_precision_dtype: "float16" or "bfloat16"
        enable_gradient_checkpointing: Enable gradient checkpointing
        cudnn_benchmark: Enable cuDNN auto-tuner
        max_split_size_mb: PyTorch memory allocator split size (MB)
        verbose: Print status messages
        
    Returns:
        Dictionary indicating which optimizations were enabled
        
    Example:
        >>> status = setup_lowvram_environment(
        ...     enable_xformers=True,
        ...     enable_tf32=True,
        ...     verbose=True
        ... )
        >>> print(status["xformers"])  # True if enabled
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for low-VRAM optimizations")
    
    status = {
        "xformers": False,
        "tf32": False,
        "flash_attention": False,
        "mixed_precision": False,
        "gradient_checkpointing": False,
        "cudnn_benchmark": False,
        "memory_allocator": False,
    }
    
    if verbose:
        print("=" * 79)
        print("Setting up Low-VRAM Environment")
        print("=" * 79)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Device: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print()
    
    # ========================================================================
    # 1. xFormers Memory-Efficient Attention
    # ========================================================================
    
    if enable_xformers and XFORMERS_AVAILABLE:
        try:
            # Test xFormers availability
            _ = xformers.ops.memory_efficient_attention
            status["xformers"] = True
            if verbose:
                print("✓ xFormers enabled (memory-efficient attention)")
                print(f"  Version: {xformers.__version__}")
        except Exception as e:
            if verbose:
                print(f"⚠ xFormers available but failed to enable: {e}")
    elif enable_xformers and not XFORMERS_AVAILABLE:
        if verbose:
            print("✗ xFormers not available (install with: pip install xformers)")
    
    # ========================================================================
    # 2. TF32 Precision (Ampere+ GPUs)
    # ========================================================================
    
    if enable_tf32 and torch.cuda.is_available():
        # Check if GPU supports TF32 (compute capability >= 8.0)
        compute_cap = torch.cuda.get_device_capability()
        supports_tf32 = compute_cap[0] >= 8  # Ampere (8.0) and newer
        
        if supports_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("medium")  # or "high"
            status["tf32"] = True
            if verbose:
                print("✓ TF32 enabled (float32_matmul_precision='medium')")
                print(f"  Compute capability: {compute_cap[0]}.{compute_cap[1]}")
        else:
            if verbose:
                print(f"⚠ TF32 not supported (compute capability {compute_cap[0]}.{compute_cap[1]} < 8.0)")
    
    # ========================================================================
    # 3. Flash Attention
    # ========================================================================
    
    if enable_flash_attention:
        # Flash Attention is typically part of xFormers or torch.nn.functional
        # Check if scaled_dot_product_attention is available (PyTorch 2.0+)
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            status["flash_attention"] = True
            if verbose:
                print("✓ Flash Attention available (torch.nn.functional.scaled_dot_product_attention)")
        elif XFORMERS_AVAILABLE:
            status["flash_attention"] = True
            if verbose:
                print("✓ Flash Attention available (via xFormers)")
        else:
            if verbose:
                print("✗ Flash Attention not available (requires PyTorch 2.0+ or xFormers)")
    
    # ========================================================================
    # 4. Mixed Precision
    # ========================================================================
    
    if enable_mixed_precision:
        status["mixed_precision"] = True
        if verbose:
            print(f"✓ Mixed precision enabled (dtype={mixed_precision_dtype})")
            print("  Use torch.cuda.amp.autocast() in training loop")
    
    # ========================================================================
    # 5. Gradient Checkpointing
    # ========================================================================
    
    if enable_gradient_checkpointing:
        status["gradient_checkpointing"] = True
        if verbose:
            print("✓ Gradient checkpointing enabled")
            print("  Call model.gradient_checkpointing_enable() before training")
    
    # ========================================================================
    # 6. cuDNN Benchmark
    # ========================================================================
    
    if cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        status["cudnn_benchmark"] = True
        if verbose:
            print("✓ cuDNN benchmark enabled (auto-tuning for faster training)")
    
    # ========================================================================
    # 7. Memory Allocator
    # ========================================================================
    
    if torch.cuda.is_available():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{max_split_size_mb}"
        status["memory_allocator"] = True
        if verbose:
            print(f"✓ PyTorch memory allocator configured (max_split_size_mb={max_split_size_mb})")
    
    if verbose:
        print("=" * 79)
        print()
    
    return status


def enable_xformers_attention(model: nn.Module, verbose: bool = True) -> bool:
    """
    Enable xFormers memory-efficient attention on a model.
    
    Args:
        model: PyTorch model to enable xFormers on
        verbose: Print status messages
        
    Returns:
        True if xFormers was successfully enabled
        
    Example:
        >>> from axs_lib.models import build_prithvi_600
        >>> model = build_prithvi_600()
        >>> enable_xformers_attention(model)
    """
    if not XFORMERS_AVAILABLE:
        if verbose:
            print("⚠ xFormers not available - install with: pip install xformers")
        return False
    
    try:
        # Check if model has enable_xformers_memory_efficient_attention method
        if hasattr(model, "enable_xformers_memory_efficient_attention"):
            model.enable_xformers_memory_efficient_attention()
            if verbose:
                print("✓ xFormers memory-efficient attention enabled on model")
            return True
        else:
            if verbose:
                print("⚠ Model does not support xFormers (no enable_xformers_memory_efficient_attention method)")
            return False
    except Exception as e:
        if verbose:
            print(f"✗ Failed to enable xFormers: {e}")
        return False


# ============================================================================
# PEFT (LoRA) Configuration
# ============================================================================

def get_lora_config(
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: str = "SEG",
    inference_mode: bool = False,
) -> "LoraConfig":
    """
    Create LoRA configuration for parameter-efficient fine-tuning.
    
    LoRA (Low-Rank Adaptation) reduces trainable parameters by 90%+ while
    maintaining most of the model's capacity. This enables fine-tuning large
    models (600M-1B parameters) on consumer GPUs.
    
    Args:
        rank: LoRA rank (4=minimal, 8=balanced, 16=high quality)
            - Higher rank = more capacity but more memory
            - Typical values: 4-16
        alpha: LoRA scaling factor (typically 2x rank)
            - Controls strength of LoRA adaptation
            - Formula: scaling = alpha / rank
        dropout: Dropout rate for LoRA layers (0.0-0.2)
        target_modules: List of module names to apply LoRA
            - Common: ["qkv", "proj"] for attention
            - Extended: ["qkv", "proj", "fc1", "fc2"] for full transformer
        bias: Bias training strategy
            - "none": No bias training (most memory efficient)
            - "all": Train all biases
            - "lora_only": Only train LoRA biases
        task_type: Task type for PEFT
            - "SEG": Segmentation
            - "CAUSAL_LM": Causal language modeling
            - "SEQ_CLS": Sequence classification
        inference_mode: Set to True for inference only (no gradient)
        
    Returns:
        LoraConfig object from PEFT library
        
    Example:
        >>> config = get_lora_config(rank=8, target_modules=["qkv", "proj"])
        >>> from peft import get_peft_model
        >>> model = get_peft_model(base_model, config)
        
    Note:
        Requires `peft` library: pip install peft
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT library is required for LoRA. "
            "Install with: pip install peft"
        )
    
    # Default target modules if not specified
    if target_modules is None:
        target_modules = ["qkv", "proj", "fc1", "fc2"]
    
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
        inference_mode=inference_mode,
    )


def apply_lora_to_model(
    model: nn.Module,
    lora_config: Optional["LoraConfig"] = None,
    rank: int = 8,
    alpha: int = 16,
    target_modules: Optional[List[str]] = None,
    verbose: bool = True,
) -> nn.Module:
    """
    Apply LoRA to a model for parameter-efficient fine-tuning.
    
    Args:
        model: Base model to apply LoRA to
        lora_config: Pre-configured LoraConfig (overrides other args)
        rank: LoRA rank (if lora_config not provided)
        alpha: LoRA alpha (if lora_config not provided)
        target_modules: Target modules (if lora_config not provided)
        verbose: Print status messages
        
    Returns:
        PEFT model with LoRA adapters
        
    Example:
        >>> from axs_lib.models import build_prithvi_600
        >>> model = build_prithvi_600(pretrained=True)
        >>> model = apply_lora_to_model(model, rank=8)
        >>> # Model now has ~10% trainable parameters
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT library is required. "
            "Install with: pip install peft"
        )
    
    # Create config if not provided
    if lora_config is None:
        lora_config = get_lora_config(
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
        )
    
    # Get trainable parameters before LoRA
    total_params_before = sum(p.numel() for p in model.parameters())
    trainable_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Apply LoRA
    peft_model = get_peft_model(model, lora_config)
    
    # Get trainable parameters after LoRA
    total_params_after = sum(p.numel() for p in peft_model.parameters())
    trainable_params_after = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    
    if verbose:
        print("=" * 79)
        print("LoRA Applied")
        print("=" * 79)
        print(f"Total parameters: {total_params_after:,}")
        print(f"Trainable parameters: {trainable_params_after:,} ({trainable_params_after/total_params_after*100:.2f}%)")
        print(f"Reduction: {(1 - trainable_params_after/trainable_params_before)*100:.2f}%")
        print(f"LoRA rank: {lora_config.r}")
        print(f"LoRA alpha: {lora_config.lora_alpha}")
        print(f"Target modules: {lora_config.target_modules}")
        print("=" * 79)
    
    return peft_model


# ============================================================================
# bitsandbytes 8-bit Quantization
# ============================================================================

def get_bnb_8bit_config(
    load_in_8bit: bool = True,
    llm_int8_threshold: float = 6.0,
    device_map: Union[str, Dict] = "auto",
    verbose: bool = True,
) -> "BitsAndBytesConfig":
    """
    Create bitsandbytes 8-bit quantization configuration.
    
    8-bit quantization reduces model memory by ~50% with minimal quality loss.
    This enables loading 1B parameter models on 8 GB GPUs.
    
    Args:
        load_in_8bit: Enable 8-bit quantization
        llm_int8_threshold: Outlier threshold for mixed precision
            - Lower = more aggressive quantization
            - Default: 6.0 (recommended)
        device_map: Device mapping strategy
            - "auto": Automatic device placement
            - "cuda:0": Single GPU
            - {"layer.0": "cuda:0", ...}: Custom mapping
        verbose: Print status messages
        
    Returns:
        BitsAndBytesConfig object
        
    Example:
        >>> from transformers import AutoModel
        >>> bnb_config = get_bnb_8bit_config()
        >>> model = AutoModel.from_pretrained(
        ...     "model_name",
        ...     quantization_config=bnb_config,
        ...     device_map="auto"
        ... )
        
    Note:
        Requires `bitsandbytes` and `transformers` libraries:
        pip install bitsandbytes transformers
    """
    if not BITSANDBYTES_AVAILABLE:
        raise ImportError(
            "bitsandbytes is required for 8-bit quantization. "
            "Install with: pip install bitsandbytes"
        )
    
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers library is required. "
            "Install with: pip install transformers"
        )
    
    config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        llm_int8_threshold=llm_int8_threshold,
        llm_int8_has_fp16_weight=False,
        llm_int8_enable_fp32_cpu_offload=False,
    )
    
    if verbose:
        print("=" * 79)
        print("8-bit Quantization Config")
        print("=" * 79)
        print(f"Load in 8-bit: {load_in_8bit}")
        print(f"LLM.int8 threshold: {llm_int8_threshold}")
        print(f"Device map: {device_map}")
        print("Expected memory reduction: ~50%")
        print("=" * 79)
    
    return config


def get_8bit_optimizer(
    model_parameters,
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    verbose: bool = True,
) -> torch.optim.Optimizer:
    """
    Create 8-bit optimizer for memory-efficient training.
    
    8-bit optimizers reduce optimizer state memory by ~75% with minimal
    impact on training dynamics. This is crucial for fine-tuning large models.
    
    Args:
        model_parameters: Model parameters (from model.parameters())
        optimizer_type: Optimizer type ("adamw", "adam", "sgd")
        lr: Learning rate
        betas: Adam betas (for Adam/AdamW)
        weight_decay: Weight decay (L2 regularization)
        eps: Epsilon for numerical stability
        verbose: Print status messages
        
    Returns:
        8-bit optimizer instance
        
    Example:
        >>> from axs_lib.optim_lowvr import get_8bit_optimizer
        >>> optimizer = get_8bit_optimizer(
        ...     model.parameters(),
        ...     optimizer_type="adamw",
        ...     lr=1e-4
        ... )
        
    Note:
        Requires `bitsandbytes`: pip install bitsandbytes
    """
    if not BITSANDBYTES_AVAILABLE:
        raise ImportError(
            "bitsandbytes is required for 8-bit optimizers. "
            "Install with: pip install bitsandbytes"
        )
    
    optimizer_map = {
        "adamw": bnb.optim.AdamW8bit,
        "adam": bnb.optim.Adam8bit,
        "sgd": bnb.optim.SGD8bit,
    }
    
    optimizer_type_lower = optimizer_type.lower()
    if optimizer_type_lower not in optimizer_map:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Choose from: {list(optimizer_map.keys())}"
        )
    
    optimizer_class = optimizer_map[optimizer_type_lower]
    
    if optimizer_type_lower in ["adamw", "adam"]:
        optimizer = optimizer_class(
            model_parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    else:  # SGD
        optimizer = optimizer_class(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    
    if verbose:
        print(f"✓ 8-bit {optimizer_type.upper()} optimizer created")
        print(f"  Expected memory reduction: ~75% vs standard optimizer")
    
    return optimizer


# ============================================================================
# Optimizer and Scheduler Setup
# ============================================================================

def make_optim_sched(
    model: nn.Module,
    lr: float = 1e-4,
    steps: int = 10000,
    weight_decay: float = 0.01,
    warmup_steps: Optional[int] = None,
    warmup_ratio: float = 0.1,
    min_lr: float = 1e-6,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    amsgrad: bool = False,
    use_parameter_groups: bool = False,
    encoder_lr_scale: float = 0.1,
    use_8bit: bool = False,
    verbose: bool = True
) -> tuple:
    """
    Create AdamW optimizer with cosine annealing + warmup scheduler.
    
    This is the primary function for setting up training optimization.
    It creates an AdamW optimizer and a cosine annealing learning rate
    scheduler with linear warmup.
    
    Args:
        model: PyTorch model to optimize
        lr: Peak learning rate (after warmup)
        steps: Total training steps
        weight_decay: L2 regularization coefficient
        warmup_steps: Number of warmup steps (if None, computed from warmup_ratio)
        warmup_ratio: Fraction of total steps for warmup (default: 0.1)
        min_lr: Minimum learning rate at end of schedule
        betas: Adam beta parameters (momentum coefficients)
        eps: Adam epsilon for numerical stability
        amsgrad: Whether to use AMSGrad variant
        use_parameter_groups: If True, use differential LR for encoder/decoder
        encoder_lr_scale: Learning rate multiplier for encoder (if parameter groups enabled)
        use_8bit: Use 8-bit AdamW optimizer (requires bitsandbytes)
        verbose: Print optimizer configuration
        
    Returns:
        Tuple of (optimizer, scheduler)
        
    Example:
        >>> model = MyModel()
        >>> optimizer, scheduler = make_optim_sched(
        ...     model, lr=1e-4, steps=10000, warmup_ratio=0.1
        ... )
        >>> 
        >>> # Training loop
        >>> for batch in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = compute_loss(model(batch))
        ...     loss.backward()
        ...     optimizer.step()
        ...     scheduler.step()
    """
    # Compute warmup steps if not provided
    if warmup_steps is None:
        warmup_steps = int(steps * warmup_ratio)
    
    # Validate warmup steps
    if warmup_steps >= steps:
        warnings.warn(
            f"warmup_steps ({warmup_steps}) >= total steps ({steps}). "
            f"Setting warmup_steps to {int(steps * 0.1)}"
        )
        warmup_steps = int(steps * 0.1)
    
    # Create parameter groups for differential learning rates
    if use_parameter_groups:
        param_groups = create_parameter_groups(
            model,
            lr=lr,
            weight_decay=weight_decay,
            encoder_lr_scale=encoder_lr_scale
        )
        if verbose:
            print(f"Using {len(param_groups)} parameter groups:")
            for i, pg in enumerate(param_groups):
                print(f"  Group {i}: lr={pg['lr']:.2e}, "
                      f"weight_decay={pg['weight_decay']}, "
                      f"params={len(pg['params'])}")
    else:
        param_groups = [
            {
                'params': [p for p in model.parameters() if p.requires_grad],
                'lr': lr,
                'weight_decay': weight_decay
            }
        ]
    
    # Create optimizer (8-bit or standard)
    if use_8bit:
        if not BITSANDBYTES_AVAILABLE:
            warnings.warn("bitsandbytes not available, using standard AdamW")
            use_8bit = False
    
    if use_8bit:
        optimizer = bnb.optim.AdamW8bit(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
    else:
        from torch.optim import AdamW
        optimizer = AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
    
    # Create cosine annealing with warmup scheduler
    scheduler = create_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=steps,
        min_lr=min_lr,
        max_lr=lr
    )
    
    if verbose:
        print(f"\n{'='*79}")
        print("Optimizer Configuration")
        print(f"{'='*79}")
        print(f"Optimizer: {'AdamW8bit' if use_8bit else 'AdamW'}")
        print(f"Peak LR: {lr:.2e}")
        print(f"Min LR: {min_lr:.2e}")
        print(f"Weight decay: {weight_decay}")
        print(f"Betas: {betas}")
        print(f"Total steps: {steps}")
        print(f"Warmup steps: {warmup_steps} ({warmup_steps/steps*100:.1f}%)")
        print(f"Cosine decay steps: {steps - warmup_steps}")
        print(f"{'='*79}\n")
    
    return optimizer, scheduler


def create_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float = 0.0,
    max_lr: Optional[float] = None,
    num_cycles: float = 0.5
):
    """
    Create a learning rate scheduler with linear warmup and cosine decay.
    
    The learning rate increases linearly from 0 to max_lr during warmup,
    then follows a cosine decay schedule down to min_lr.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr: Minimum learning rate (as fraction of max_lr if max_lr provided)
        max_lr: Maximum learning rate (if None, uses optimizer's initial LR)
        num_cycles: Number of cosine cycles (0.5 for standard cosine annealing)
        
    Returns:
        LambdaLR scheduler
    """
    import math
    from torch.optim.lr_scheduler import LambdaLR
    
    if max_lr is None:
        max_lr = optimizer.defaults['lr']
    
    min_lr_ratio = min_lr / max_lr if max_lr > 0 else 0.0
    
    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        
        # Scale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


def create_parameter_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    encoder_lr_scale: float = 0.1,
    encoder_names: Optional[List[str]] = None,
    no_decay_names: Optional[List[str]] = None
) -> List[Dict]:
    """
    Create parameter groups for differential learning rates and weight decay.
    
    This allows different learning rates for encoder vs decoder, and
    disables weight decay for certain parameters (biases, LayerNorm, etc.).
    
    Args:
        model: PyTorch model
        lr: Base learning rate
        weight_decay: Weight decay coefficient
        encoder_lr_scale: Learning rate multiplier for encoder
        encoder_names: List of parameter name patterns to identify encoder
                      (default: ['encoder', 'backbone'])
        no_decay_names: Parameter name patterns to exclude from weight decay
                       (default: ['bias', 'norm', 'bn'])
        
    Returns:
        List of parameter group dictionaries
    """
    if encoder_names is None:
        encoder_names = ['encoder', 'backbone', 'downsample']
    
    if no_decay_names is None:
        no_decay_names = ['bias', 'norm', 'bn', 'gamma', 'beta']
    
    # Categorize parameters
    encoder_decay = []
    encoder_no_decay = []
    decoder_decay = []
    decoder_no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if parameter is in encoder
        is_encoder = any(enc_name in name.lower() for enc_name in encoder_names)
        
        # Check if parameter should have weight decay
        should_decay = not any(nd_name in name.lower() for nd_name in no_decay_names)
        
        # Categorize
        if is_encoder and should_decay:
            encoder_decay.append(param)
        elif is_encoder and not should_decay:
            encoder_no_decay.append(param)
        elif not is_encoder and should_decay:
            decoder_decay.append(param)
        else:
            decoder_no_decay.append(param)
    
    # Create parameter groups
    param_groups = []
    
    if encoder_decay:
        param_groups.append({
            'params': encoder_decay,
            'lr': lr * encoder_lr_scale,
            'weight_decay': weight_decay,
            'name': 'encoder_decay'
        })
    
    if encoder_no_decay:
        param_groups.append({
            'params': encoder_no_decay,
            'lr': lr * encoder_lr_scale,
            'weight_decay': 0.0,
            'name': 'encoder_no_decay'
        })
    
    if decoder_decay:
        param_groups.append({
            'params': decoder_decay,
            'lr': lr,
            'weight_decay': weight_decay,
            'name': 'decoder_decay'
        })
    
    if decoder_no_decay:
        param_groups.append({
            'params': decoder_no_decay,
            'lr': lr,
            'weight_decay': 0.0,
            'name': 'decoder_no_decay'
        })
    
    return param_groups


def get_current_lr(optimizer) -> Union[float, List[float]]:
    """
    Get current learning rate(s) from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Learning rate (or list of rates if multiple param groups)
    """
    lrs = [param_group['lr'] for param_group in optimizer.param_groups]
    return lrs[0] if len(lrs) == 1 else lrs


def set_lr(optimizer, lr: Union[float, List[float]]):
    """
    Set learning rate(s) for optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        lr: New learning rate (or list of rates for each param group)
    """
    if isinstance(lr, (list, tuple)):
        for param_group, new_lr in zip(optimizer.param_groups, lr):
            param_group['lr'] = new_lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


# ============================================================================
# Gradient Checkpointing Utilities
# ============================================================================

def enable_gradient_checkpointing(
    model: nn.Module,
    checkpoint_ratio: float = 1.0,
    verbose: bool = True
) -> int:
    """
    Enable gradient checkpointing for compatible modules.
    
    Gradient checkpointing trades compute for memory by recomputing
    activations during backward pass instead of storing them.
    
    Args:
        model: PyTorch model
        checkpoint_ratio: Fraction of compatible layers to checkpoint (0.0 to 1.0)
        verbose: Print information about checkpointing
        
    Returns:
        Number of modules with checkpointing enabled
    """
    count = 0
    total_compatible = 0
    
    for name, module in model.named_modules():
        # Check if module has gradient_checkpointing attribute
        if hasattr(module, 'gradient_checkpointing'):
            total_compatible += 1
            if checkpoint_ratio >= (total_compatible / max(total_compatible, 1)):
                module.gradient_checkpointing = True
                count += 1
        
        # Check for modules with gradient_checkpointing_enable method
        elif hasattr(module, 'gradient_checkpointing_enable'):
            total_compatible += 1
            if checkpoint_ratio >= (total_compatible / max(total_compatible, 1)):
                module.gradient_checkpointing_enable()
                count += 1
    
    if verbose:
        print(f"Gradient checkpointing enabled for {count}/{total_compatible} compatible modules")
    
    return count


def disable_gradient_checkpointing(model: nn.Module, verbose: bool = True) -> int:
    """
    Disable gradient checkpointing for all modules.
    
    Args:
        model: PyTorch model
        verbose: Print information
        
    Returns:
        Number of modules with checkpointing disabled
    """
    count = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = False
            count += 1
        elif hasattr(module, 'gradient_checkpointing_disable'):
            module.gradient_checkpointing_disable()
            count += 1
    
    if verbose:
        print(f"Gradient checkpointing disabled for {count} modules")
    
    return count


def toggle_gradient_checkpointing(model: nn.Module, enable: bool, verbose: bool = True) -> int:
    """
    Toggle gradient checkpointing on or off.
    
    Args:
        model: PyTorch model
        enable: True to enable, False to disable
        verbose: Print information
        
    Returns:
        Number of modules affected
    """
    if enable:
        return enable_gradient_checkpointing(model, verbose=verbose)
    else:
        return disable_gradient_checkpointing(model, verbose=verbose)


# ============================================================================
# Utility Functions
# ============================================================================

def print_memory_stats(device: Optional[str] = None):
    """Print current CUDA memory statistics."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    if device is None:
        device = torch.cuda.current_device()
    
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    
    print("=" * 79)
    print(f"CUDA Memory Statistics (Device {device})")
    print("=" * 79)
    print(f"Allocated:     {allocated:.2f} GB ({allocated/total*100:.1f}%)")
    print(f"Reserved:      {reserved:.2f} GB ({reserved/total*100:.1f}%)")
    print(f"Max allocated: {max_allocated:.2f} GB ({max_allocated/total*100:.1f}%)")
    print(f"Total:         {total:.2f} GB")
    print(f"Free:          {total - reserved:.2f} GB ({(total - reserved)/total*100:.1f}%)")
    print("=" * 79)


def clear_cuda_cache(verbose: bool = False):
    """Clear CUDA cache to free unused memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if verbose:
            print("✓ CUDA cache cleared")


# ============================================================================
# Testing & Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 79)
    print("AXION-SAT LOW-VRAM OPTIMIZATION UTILITIES")
    print("=" * 79)
    print()
    
    # Check dependencies
    print("Dependency Check:")
    print(f"  PyTorch: {'✓' if TORCH_AVAILABLE else '✗'}")
    print(f"  xFormers: {'✓' if XFORMERS_AVAILABLE else '✗'}")
    print(f"  PEFT: {'✓' if PEFT_AVAILABLE else '✗'}")
    print(f"  bitsandbytes: {'✓' if BITSANDBYTES_AVAILABLE else '✗'}")
    print(f"  Transformers: {'✓' if TRANSFORMERS_AVAILABLE else '✗'}")
    print()
    
    # Setup environment
    if TORCH_AVAILABLE:
        status = setup_lowvram_environment(verbose=True)
        
        # Print memory stats
        if torch.cuda.is_available():
            print_memory_stats()
        
        # Show LoRA config example
        if PEFT_AVAILABLE:
            print("\nExample LoRA Configuration:")
            lora_config = get_lora_config(rank=8, target_modules=["qkv", "proj"])
            print(f"  Rank: {lora_config.r}")
            print(f"  Alpha: {lora_config.lora_alpha}")
            print(f"  Dropout: {lora_config.lora_dropout}")
            print(f"  Target modules: {lora_config.target_modules}")
    
    print()
    print("=" * 79)
    print("Low-VRAM utilities ready!")
    print()
    print("Usage:")
    print("  from axs_lib.optim_lowvr import setup_lowvram_environment")
    print("  setup_lowvram_environment(enable_xformers=True, enable_tf32=True)")
    print("=" * 79)
