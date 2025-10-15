"""
OOM (Out of Memory) Guard for Training.

This module provides automatic recovery from CUDA OOM errors by:
1. Catching OOM exceptions during training/inference
2. Reducing batch size, tile size, or other memory-intensive parameters
3. Retrying the operation with reduced settings
4. Tracking and logging OOM events for optimization

The guard reads fallback strategies from hardware.lowvr.yaml and applies
them progressively until the operation succeeds or all strategies are exhausted.

Usage:
    >>> from tools.oom_guard import OOMGuard, oom_safe_step
    >>> 
    >>> guard = OOMGuard(config_path="hardware.lowvr.yaml")
    >>> 
    >>> # Wrap training step
    >>> @oom_safe_step(guard)
    >>> def training_step(batch, model, optimizer):
    ...     outputs = model(batch)
    ...     loss = criterion(outputs, targets)
    ...     loss.backward()
    ...     optimizer.step()
    ...     return loss.item()
"""

import torch
import yaml
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from functools import wraps
import traceback


@dataclass
class OOMStrategy:
    """Configuration for OOM recovery strategy."""
    
    name: str
    description: str
    
    # Parameter reduction ratios
    batch_size_scale: float = 1.0  # Multiply batch size by this
    tile_size_scale: float = 1.0  # Multiply tile size by this
    num_timesteps_scale: float = 1.0  # For diffusion models
    
    # Memory management
    enable_gradient_checkpointing: bool = False
    empty_cache: bool = True
    
    # Mixed precision
    use_fp16: bool = False
    use_bf16: bool = False
    
    # Model modifications
    reduce_attention_heads: bool = False
    reduce_hidden_dims: bool = False
    
    # Other settings
    max_retries: int = 1
    cooldown_seconds: float = 2.0


@dataclass
class OOMEvent:
    """Record of an OOM event."""
    
    timestamp: float
    strategy_index: int
    strategy_name: str
    error_message: str
    traceback_str: str
    success: bool = False
    
    # Context
    batch_size: Optional[int] = None
    tile_size: Optional[int] = None
    num_timesteps: Optional[int] = None
    
    # Memory stats
    allocated_gb: Optional[float] = None
    reserved_gb: Optional[float] = None
    max_allocated_gb: Optional[float] = None


class OOMGuard:
    """
    Guard against CUDA Out of Memory errors with automatic recovery.
    
    The guard progressively applies fallback strategies when OOM occurs:
    1. Clear CUDA cache
    2. Reduce batch size by 50%
    3. Reduce tile size
    4. Enable gradient checkpointing
    5. Reduce to single sample + checkpointing
    
    Args:
        config_path: Path to hardware.lowvr.yaml configuration file
        strategies: List of OOMStrategy objects (overrides config)
        verbose: Print detailed information
        log_file: Path to log OOM events (optional)
        max_total_retries: Maximum total retry attempts across all strategies
    """
    
    def __init__(
        self,
        config_path: Optional[str] = "hardware.lowvr.yaml",
        strategies: Optional[List[OOMStrategy]] = None,
        verbose: bool = True,
        log_file: Optional[str] = None,
        max_total_retries: int = 5
    ):
        self.verbose = verbose
        self.log_file = log_file
        self.max_total_retries = max_total_retries
        
        # Load strategies from config or use provided ones
        if strategies is not None:
            self.strategies = strategies
        else:
            self.strategies = self._load_strategies_from_config(config_path)
        
        # Event tracking
        self.events: List[OOMEvent] = []
        self.total_oom_count = 0
        self.successful_recoveries = 0
        
        # Current state
        self.current_strategy_index = 0
        self.retry_count = 0
        
        if self.verbose:
            print(f"OOM Guard initialized with {len(self.strategies)} strategies")
    
    def _load_strategies_from_config(self, config_path: str) -> List[OOMStrategy]:
        """Load OOM recovery strategies from YAML config."""
        # Default strategies if config not found
        default_strategies = [
            OOMStrategy(
                name="cache_clear",
                description="Clear CUDA cache",
                empty_cache=True,
                max_retries=1
            ),
            OOMStrategy(
                name="reduce_batch_50",
                description="Reduce batch size by 50%",
                batch_size_scale=0.5,
                empty_cache=True,
                max_retries=2
            ),
            OOMStrategy(
                name="reduce_batch_and_tile",
                description="Reduce batch size and tile size",
                batch_size_scale=0.5,
                tile_size_scale=0.75,
                empty_cache=True,
                enable_gradient_checkpointing=True,
                max_retries=2
            ),
            OOMStrategy(
                name="aggressive_reduction",
                description="Aggressive reduction: batch=1, small tiles",
                batch_size_scale=0.125,  # Down to 1 if started at 8
                tile_size_scale=0.5,
                empty_cache=True,
                enable_gradient_checkpointing=True,
                use_fp16=True,
                max_retries=2
            ),
            OOMStrategy(
                name="minimal_mode",
                description="Minimal mode: single sample, checkpointing, FP16",
                batch_size_scale=0.0625,  # Effectively batch_size=1
                tile_size_scale=0.5,
                num_timesteps_scale=0.75,
                empty_cache=True,
                enable_gradient_checkpointing=True,
                use_fp16=True,
                max_retries=1
            )
        ]
        
        # Try to load from config
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                if config and 'oom_strategies' in config:
                    strategies = []
                    for strat_config in config['oom_strategies']:
                        strategies.append(OOMStrategy(**strat_config))
                    
                    if self.verbose:
                        print(f"Loaded {len(strategies)} OOM strategies from {config_path}")
                    return strategies
                
            except Exception as e:
                warnings.warn(f"Failed to load OOM strategies from {config_path}: {e}")
        
        if self.verbose:
            print(f"Using {len(default_strategies)} default OOM strategies")
        
        return default_strategies
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current CUDA memory statistics."""
        if not torch.cuda.is_available():
            return {}
        
        device = torch.cuda.current_device()
        return {
            'allocated_gb': torch.cuda.memory_allocated(device) / 1e9,
            'reserved_gb': torch.cuda.memory_reserved(device) / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated(device) / 1e9,
        }
    
    def _apply_strategy(
        self,
        strategy: OOMStrategy,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply an OOM recovery strategy.
        
        Args:
            strategy: OOMStrategy to apply
            context: Current context (batch_size, tile_size, etc.)
            
        Returns:
            Updated context with modified parameters
        """
        if context is None:
            context = {}
        
        modified_context = context.copy()
        
        # Clear CUDA cache
        if strategy.empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.verbose:
                print("  → Cleared CUDA cache")
        
        # Reduce batch size
        if strategy.batch_size_scale < 1.0 and 'batch_size' in context:
            old_bs = context['batch_size']
            new_bs = max(1, int(old_bs * strategy.batch_size_scale))
            modified_context['batch_size'] = new_bs
            if self.verbose:
                print(f"  → Reduced batch size: {old_bs} → {new_bs}")
        
        # Reduce tile size
        if strategy.tile_size_scale < 1.0 and 'tile_size' in context:
            old_ts = context['tile_size']
            new_ts = max(64, int(old_ts * strategy.tile_size_scale))
            modified_context['tile_size'] = new_ts
            if self.verbose:
                print(f"  → Reduced tile size: {old_ts} → {new_ts}")
        
        # Reduce timesteps (for diffusion models)
        if strategy.num_timesteps_scale < 1.0 and 'num_timesteps' in context:
            old_ts = context['num_timesteps']
            new_ts = max(10, int(old_ts * strategy.num_timesteps_scale))
            modified_context['num_timesteps'] = new_ts
            if self.verbose:
                print(f"  → Reduced timesteps: {old_ts} → {new_ts}")
        
        # Enable gradient checkpointing
        if strategy.enable_gradient_checkpointing:
            modified_context['gradient_checkpointing'] = True
            if self.verbose:
                print("  → Enabled gradient checkpointing")
        
        # Mixed precision settings
        if strategy.use_fp16:
            modified_context['use_fp16'] = True
            modified_context['use_bf16'] = False
            if self.verbose:
                print("  → Enabled FP16 mixed precision")
        elif strategy.use_bf16:
            modified_context['use_bf16'] = True
            modified_context['use_fp16'] = False
            if self.verbose:
                print("  → Enabled BF16 mixed precision")
        
        # Cooldown
        if strategy.cooldown_seconds > 0:
            time.sleep(strategy.cooldown_seconds)
        
        return modified_context
    
    def _log_event(self, event: OOMEvent):
        """Log an OOM event."""
        self.events.append(event)
        
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    log_entry = {
                        'timestamp': event.timestamp,
                        'strategy': event.strategy_name,
                        'success': event.success,
                        'batch_size': event.batch_size,
                        'tile_size': event.tile_size,
                        'memory_gb': event.allocated_gb
                    }
                    f.write(yaml.dump([log_entry]) + '\n')
            except Exception as e:
                warnings.warn(f"Failed to log OOM event: {e}")
    
    def safe_execute(
        self,
        func: Callable,
        *args,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute a function with OOM protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            context: Dictionary with execution context (batch_size, tile_size, etc.)
            **kwargs: Keyword arguments for func
            
        Returns:
            Tuple of (result, final_context)
            
        Raises:
            RuntimeError: If all recovery strategies fail
        """
        if context is None:
            context = {}
        
        current_context = context.copy()
        strategy_index = 0
        total_retries = 0
        
        while total_retries < self.max_total_retries:
            try:
                # Execute function
                result = func(*args, **kwargs, **current_context)
                
                # Success!
                if strategy_index > 0:
                    self.successful_recoveries += 1
                    if self.verbose:
                        print(f"✓ Successfully recovered using strategy: "
                              f"{self.strategies[strategy_index-1].name}")
                
                return result, current_context
                
            except RuntimeError as e:
                error_str = str(e)
                
                # Check if it's a CUDA OOM error
                if "out of memory" not in error_str.lower():
                    # Not an OOM error, re-raise
                    raise
                
                # It's an OOM error
                self.total_oom_count += 1
                total_retries += 1
                
                if self.verbose:
                    print(f"\n⚠ CUDA OOM detected (attempt {total_retries}/{self.max_total_retries})")
                    print(f"  Error: {error_str[:100]}...")
                
                # Get memory stats
                mem_stats = self._get_memory_stats()
                
                # Log event
                event = OOMEvent(
                    timestamp=time.time(),
                    strategy_index=strategy_index,
                    strategy_name=self.strategies[strategy_index].name if strategy_index < len(self.strategies) else "none",
                    error_message=error_str,
                    traceback_str=traceback.format_exc(),
                    success=False,
                    batch_size=current_context.get('batch_size'),
                    tile_size=current_context.get('tile_size'),
                    num_timesteps=current_context.get('num_timesteps'),
                    **mem_stats
                )
                self._log_event(event)
                
                # Check if we have more strategies to try
                if strategy_index >= len(self.strategies):
                    raise RuntimeError(
                        f"CUDA OOM: All {len(self.strategies)} recovery strategies exhausted. "
                        f"Consider reducing batch size, tile size, or using a GPU with more VRAM."
                    )
                
                # Apply next strategy
                strategy = self.strategies[strategy_index]
                if self.verbose:
                    print(f"\n→ Applying strategy {strategy_index + 1}/{len(self.strategies)}: "
                          f"{strategy.name}")
                    print(f"  {strategy.description}")
                
                current_context = self._apply_strategy(strategy, current_context)
                
                # Move to next strategy
                strategy_index += 1
                
                # Wait before retry
                if strategy.cooldown_seconds > 0:
                    if self.verbose:
                        print(f"  Waiting {strategy.cooldown_seconds}s before retry...")
                    time.sleep(strategy.cooldown_seconds)
        
        # All retries exhausted
        raise RuntimeError(
            f"CUDA OOM: Maximum retry limit ({self.max_total_retries}) reached. "
            f"Unable to recover from out of memory error."
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about OOM events."""
        return {
            'total_oom_count': self.total_oom_count,
            'successful_recoveries': self.successful_recoveries,
            'total_events': len(self.events),
            'strategies_available': len(self.strategies),
            'success_rate': self.successful_recoveries / max(1, self.total_oom_count)
        }
    
    def print_stats(self):
        """Print OOM statistics."""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("OOM Guard Statistics")
        print("="*60)
        print(f"Total OOM events: {stats['total_oom_count']}")
        print(f"Successful recoveries: {stats['successful_recoveries']}")
        print(f"Success rate: {stats['success_rate']*100:.1f}%")
        print(f"Strategies available: {stats['strategies_available']}")
        print("="*60 + "\n")


# ============================================================================
# Decorator for OOM-safe functions
# ============================================================================

def oom_safe_step(guard: OOMGuard, context: Optional[Dict[str, Any]] = None):
    """
    Decorator to make a training step OOM-safe.
    
    Args:
        guard: OOMGuard instance
        context: Execution context (batch_size, tile_size, etc.)
        
    Example:
        >>> guard = OOMGuard()
        >>> 
        >>> @oom_safe_step(guard, context={'batch_size': 8, 'tile_size': 256})
        >>> def training_step(batch, model, optimizer):
        ...     outputs = model(batch)
        ...     loss = criterion(outputs, targets)
        ...     loss.backward()
        ...     optimizer.step()
        ...     return loss.item()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result, final_context = guard.safe_execute(
                func, *args, context=context, **kwargs
            )
            return result
        return wrapper
    return decorator


# ============================================================================
# Context Manager for OOM protection
# ============================================================================

class OOMProtection:
    """
    Context manager for OOM-protected code blocks.
    
    Example:
        >>> guard = OOMGuard()
        >>> context = {'batch_size': 8, 'tile_size': 256}
        >>> 
        >>> with OOMProtection(guard, context) as ctx:
        ...     outputs = model(batch)
        ...     loss = criterion(outputs, targets)
        ...     loss.backward()
    """
    
    def __init__(self, guard: OOMGuard, context: Optional[Dict[str, Any]] = None):
        self.guard = guard
        self.context = context or {}
        self.initial_context = self.context.copy()
    
    def __enter__(self):
        return self.context
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Check if it's an OOM error
        if exc_type is RuntimeError and exc_val is not None:
            error_str = str(exc_val)
            if "out of memory" in error_str.lower():
                # Attempt recovery
                try:
                    # Apply first available strategy
                    if len(self.guard.strategies) > 0:
                        strategy = self.guard.strategies[0]
                        self.context = self.guard._apply_strategy(strategy, self.context)
                        
                        if self.guard.verbose:
                            print(f"OOM Protection: Applied strategy '{strategy.name}'")
                            print("Please retry the operation with updated context")
                        
                        return False  # Don't suppress exception
                except Exception as e:
                    warnings.warn(f"OOM Protection recovery failed: {e}")
        
        return False  # Don't suppress other exceptions


# ============================================================================
# Utility Functions
# ============================================================================

def create_default_config(output_path: str = "hardware.lowvr.yaml"):
    """
    Create a default hardware.lowvr.yaml configuration file.
    
    Args:
        output_path: Path to save the configuration file
    """
    default_config = {
        'hardware': {
            'gpu_vram_gb': 12,
            'target_utilization': 0.85,
            'emergency_threshold': 0.95
        },
        'oom_strategies': [
            {
                'name': 'cache_clear',
                'description': 'Clear CUDA cache',
                'empty_cache': True,
                'max_retries': 1,
                'cooldown_seconds': 1.0
            },
            {
                'name': 'reduce_batch_50',
                'description': 'Reduce batch size by 50%',
                'batch_size_scale': 0.5,
                'empty_cache': True,
                'max_retries': 2,
                'cooldown_seconds': 2.0
            },
            {
                'name': 'reduce_batch_and_tile',
                'description': 'Reduce batch size and tile size',
                'batch_size_scale': 0.5,
                'tile_size_scale': 0.75,
                'empty_cache': True,
                'enable_gradient_checkpointing': True,
                'max_retries': 2,
                'cooldown_seconds': 2.0
            },
            {
                'name': 'aggressive_reduction',
                'description': 'Aggressive reduction with FP16',
                'batch_size_scale': 0.125,
                'tile_size_scale': 0.5,
                'empty_cache': True,
                'enable_gradient_checkpointing': True,
                'use_fp16': True,
                'max_retries': 2,
                'cooldown_seconds': 3.0
            },
            {
                'name': 'minimal_mode',
                'description': 'Minimal mode: single sample',
                'batch_size_scale': 0.0625,
                'tile_size_scale': 0.5,
                'num_timesteps_scale': 0.75,
                'empty_cache': True,
                'enable_gradient_checkpointing': True,
                'use_fp16': True,
                'max_retries': 1,
                'cooldown_seconds': 3.0
            }
        ]
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created default configuration: {output_path}")


def estimate_memory_usage(
    batch_size: int,
    tile_size: int,
    num_channels: int = 4,
    model_params: int = 600_000_000,
    dtype_bytes: int = 4
) -> float:
    """
    Estimate memory usage for a training step.
    
    Args:
        batch_size: Batch size
        tile_size: Tile size (assumes square tiles)
        num_channels: Number of input channels
        model_params: Number of model parameters
        dtype_bytes: Bytes per parameter (4 for float32, 2 for float16)
        
    Returns:
        Estimated memory usage in GB
    """
    # Input tensor
    input_mem = batch_size * num_channels * tile_size * tile_size * dtype_bytes
    
    # Model parameters
    model_mem = model_params * dtype_bytes
    
    # Gradients (same size as model)
    grad_mem = model_params * dtype_bytes
    
    # Optimizer state (Adam: 2x model size for momentum)
    optimizer_mem = model_params * dtype_bytes * 2
    
    # Activations (rough estimate: 4x input size)
    activation_mem = input_mem * 4
    
    # Total
    total_bytes = input_mem + model_mem + grad_mem + optimizer_mem + activation_mem
    
    return total_bytes / 1e9


# ============================================================================
# Main / Testing
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("OOM Guard Utility")
    print("="*60)
    
    # Create default config if it doesn't exist
    config_path = "hardware.lowvr.yaml"
    if not os.path.exists(config_path):
        print(f"\nCreating default configuration: {config_path}")
        create_default_config(config_path)
    
    # Initialize guard
    print(f"\nInitializing OOM Guard...")
    guard = OOMGuard(config_path=config_path, verbose=True)
    
    print(f"\nLoaded {len(guard.strategies)} strategies:")
    for i, strategy in enumerate(guard.strategies):
        print(f"  {i+1}. {strategy.name}: {strategy.description}")
    
    # Estimate memory usage
    print("\n" + "="*60)
    print("Memory Usage Estimation")
    print("="*60)
    
    configs = [
        (8, 256, "Full batch"),
        (4, 256, "Half batch"),
        (2, 256, "Quarter batch"),
        (1, 256, "Single sample"),
        (1, 128, "Single sample, small tile")
    ]
    
    for batch_size, tile_size, desc in configs:
        mem_gb = estimate_memory_usage(batch_size, tile_size)
        print(f"{desc:30s} (bs={batch_size}, ts={tile_size}): {mem_gb:6.2f} GB")
    
    print("\n" + "="*60)
    print("OOM Guard ready for use!")
    print("="*60)
    print("\nUsage example:")
    print("  from tools.oom_guard import OOMGuard")
    print("  guard = OOMGuard('hardware.lowvr.yaml')")
    print("  result, context = guard.safe_execute(training_step, batch, model)")
