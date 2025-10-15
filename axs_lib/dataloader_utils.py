"""
DataLoader Utilities: Deterministic and Platform-Aware Configuration

Provides utilities for creating deterministic PyTorch DataLoaders with
platform-specific handling for multiprocessing issues (especially Windows).

Key Features:
    - Automatic detection of Windows platform
    - Sets num_workers=0 on Windows to avoid multiprocessing issues
    - Configures worker_init_fn for reproducibility
    - Provides generator for deterministic shuffling
    - Comprehensive documentation of limitations

Windows Multiprocessing Issues:
    Windows uses 'spawn' method for multiprocessing (vs 'fork' on Linux/Mac).
    This causes several issues with PyTorch DataLoaders:
    
    1. **Serialization Overhead**: Each worker process must serialize/deserialize
       the dataset object, which is slow and memory-intensive.
    
    2. **CUDA Initialization**: CUDA cannot be initialized in spawned workers,
       causing errors if the dataset or transforms use GPU operations.
    
    3. **Module Imports**: Spawned processes must re-import all modules,
       causing slow startup and potential import errors.
    
    4. **Shared Memory**: Windows doesn't support shared memory well, making
       efficient data sharing between workers difficult.
    
    5. **Reproducibility**: Even with proper seeding, spawned workers can
       have subtle non-deterministic behavior due to OS scheduling.
    
    For these reasons, we recommend num_workers=0 on Windows for:
    - Reliability (no multiprocessing crashes)
    - Reproducibility (deterministic behavior)
    - Simplicity (easier debugging)
    
    Trade-off: Slightly slower data loading, but often negligible with:
    - Small datasets (< 10k samples)
    - Fast dataloaders (already cached/preprocessed data)
    - GPU-bound training (data loading not the bottleneck)

Usage:
    >>> from axs_lib.dataloader_utils import create_dataloader
    >>> from torch.utils.data import Dataset
    >>> 
    >>> # Automatically handles Windows and reproducibility
    >>> loader = create_dataloader(
    ...     dataset,
    ...     batch_size=32,
    ...     shuffle=True,
    ...     num_workers=4,  # Will be set to 0 on Windows
    ...     seed=42  # For reproducible shuffling
    ... )

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
import platform
import warnings
from typing import Optional, Callable
import torch
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# Platform Detection
# ============================================================================

def is_windows() -> bool:
    """
    Check if running on Windows.
    
    Returns:
        True if Windows, False otherwise
    """
    return platform.system() == 'Windows' or sys.platform == 'win32'


def get_safe_num_workers(
    num_workers: int,
    force_zero_on_windows: bool = True,
    warn: bool = True
) -> int:
    """
    Get safe number of workers for DataLoader based on platform.
    
    On Windows, using num_workers > 0 can cause:
    - Multiprocessing crashes
    - Non-deterministic behavior
    - CUDA initialization errors
    - Slow spawning of worker processes
    
    Args:
        num_workers: Requested number of workers
        force_zero_on_windows: If True, force num_workers=0 on Windows
        warn: Print warning when modifying num_workers
        
    Returns:
        Safe number of workers for current platform
        
    Example:
        >>> num_workers = get_safe_num_workers(4)
        >>> # Returns 0 on Windows, 4 on Linux/Mac
    """
    if not is_windows() or not force_zero_on_windows:
        return num_workers
    
    if num_workers > 0 and warn:
        warnings.warn(
            f"Setting num_workers=0 (requested {num_workers}) on Windows platform.\n"
            f"Reason: Windows uses 'spawn' for multiprocessing, which causes:\n"
            f"  - Serialization overhead and slow startup\n"
            f"  - CUDA initialization issues\n"
            f"  - Potential non-deterministic behavior\n"
            f"  - Multiprocessing crashes in some environments\n"
            f"\n"
            f"For most use cases, single-threaded loading is sufficient and more reliable.\n"
            f"Set force_zero_on_windows=False to override this behavior.",
            UserWarning,
            stacklevel=2
        )
    
    return 0 if num_workers > 0 else num_workers


# ============================================================================
# Reproducibility Utilities
# ============================================================================

def create_worker_init_fn(base_seed: int) -> Callable:
    """
    Create worker initialization function for reproducible DataLoader workers.
    
    Each worker gets a unique but deterministic seed based on:
    - Base seed
    - Worker ID
    - PyTorch's internal seed
    
    Args:
        base_seed: Base random seed
        
    Returns:
        Worker initialization function for DataLoader
        
    Example:
        >>> worker_init_fn = create_worker_init_fn(42)
        >>> loader = DataLoader(..., worker_init_fn=worker_init_fn)
    """
    def worker_init_fn(worker_id: int):
        """Initialize worker with deterministic seed."""
        import random
        import numpy as np
        
        # Compute worker seed from base seed + worker ID + torch seed
        worker_seed = (base_seed + worker_id + torch.initial_seed()) % (2**32)
        
        # Set seeds
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    return worker_init_fn


def create_generator(seed: int) -> torch.Generator:
    """
    Create PyTorch Generator with fixed seed for reproducible operations.
    
    Use for:
    - Reproducible shuffling in DataLoader
    - Reproducible data splitting
    - Reproducible sampling
    
    Args:
        seed: Random seed
        
    Returns:
        torch.Generator with fixed seed
        
    Example:
        >>> generator = create_generator(42)
        >>> loader = DataLoader(..., shuffle=True, generator=generator)
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# ============================================================================
# DataLoader Creation
# ============================================================================

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    seed: Optional[int] = None,
    force_zero_workers_on_windows: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader with automatic platform-aware and reproducibility configuration.
    
    Features:
    - Automatically sets num_workers=0 on Windows (overridable)
    - Configures worker_init_fn for reproducibility if seed provided
    - Uses generator for reproducible shuffling if seed provided
    - Passes through additional DataLoader kwargs
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data (use with seed for reproducibility)
        num_workers: Number of worker processes (set to 0 on Windows automatically)
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        seed: Random seed for reproducibility (None = non-deterministic)
        force_zero_workers_on_windows: Force num_workers=0 on Windows
        **kwargs: Additional arguments passed to DataLoader
        
    Returns:
        Configured DataLoader
        
    Example:
        >>> # Simple usage
        >>> loader = create_dataloader(dataset, batch_size=32)
        >>> 
        >>> # With reproducibility
        >>> loader = create_dataloader(
        ...     dataset,
        ...     batch_size=32,
        ...     shuffle=True,
        ...     num_workers=4,  # Will be 0 on Windows
        ...     seed=42  # Deterministic shuffling
        ... )
        >>> 
        >>> # Override Windows behavior (not recommended)
        >>> loader = create_dataloader(
        ...     dataset,
        ...     batch_size=32,
        ...     num_workers=4,
        ...     force_zero_workers_on_windows=False
        ... )
    
    Note:
        On Windows, using num_workers > 0 can cause issues. We automatically
        set it to 0 unless force_zero_workers_on_windows=False.
        
        For reproducibility, provide a seed. This will:
        - Configure worker_init_fn for deterministic worker seeding
        - Use generator for deterministic shuffling
    """
    # Get safe number of workers
    safe_num_workers = get_safe_num_workers(
        num_workers,
        force_zero_on_windows=force_zero_workers_on_windows,
        warn=True
    )
    
    # Prepare DataLoader kwargs
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': safe_num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
    }
    
    # Add reproducibility configuration if seed provided
    if seed is not None:
        # Worker initialization for multi-worker reproducibility
        if safe_num_workers > 0:
            dataloader_kwargs['worker_init_fn'] = create_worker_init_fn(seed)
        
        # Generator for reproducible shuffling
        if shuffle:
            dataloader_kwargs['generator'] = create_generator(seed)
    
    # Merge with additional kwargs
    dataloader_kwargs.update(kwargs)
    
    return DataLoader(dataset, **dataloader_kwargs)


# ============================================================================
# Information Utilities
# ============================================================================

def print_dataloader_config(
    num_workers: int,
    batch_size: int,
    seed: Optional[int] = None,
    shuffle: bool = False
):
    """
    Print DataLoader configuration information.
    
    Useful for debugging and ensuring correct configuration.
    
    Args:
        num_workers: Number of workers
        batch_size: Batch size
        seed: Random seed (None if non-deterministic)
        shuffle: Whether shuffling is enabled
    """
    print("=" * 80)
    print("DataLoader Configuration")
    print("=" * 80)
    print(f"  Platform:     {platform.system()} ({sys.platform})")
    print(f"  Num Workers:  {num_workers}")
    print(f"  Batch Size:   {batch_size}")
    print(f"  Shuffle:      {shuffle}")
    print(f"  Seed:         {seed if seed is not None else 'None (non-deterministic)'}")
    
    if is_windows() and num_workers > 0:
        print("\n⚠ WARNING: Using num_workers > 0 on Windows!")
        print("  This may cause:")
        print("    - Multiprocessing crashes")
        print("    - Non-deterministic behavior")
        print("    - CUDA initialization errors")
        print("  Recommendation: Set num_workers=0")
    
    if seed is None and shuffle:
        print("\n⚠ WARNING: Shuffling without seed (non-deterministic)")
        print("  Recommendation: Provide seed for reproducibility")
    
    print("=" * 80)


def get_recommended_num_workers() -> int:
    """
    Get recommended number of workers for current platform.
    
    Returns:
        Recommended num_workers based on platform and CPU count
        
    Example:
        >>> num_workers = get_recommended_num_workers()
        >>> loader = create_dataloader(dataset, num_workers=num_workers)
    """
    if is_windows():
        # On Windows, always recommend 0 for reliability
        return 0
    else:
        # On Linux/Mac, use CPU count but cap at 8
        import os
        cpu_count = os.cpu_count() or 1
        return min(cpu_count, 8)


# ============================================================================
# CLI for Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DataLoader Utilities - Platform Information")
    print("=" * 80)
    print()
    
    print(f"Platform:              {platform.system()}")
    print(f"Platform Detail:       {sys.platform}")
    print(f"Is Windows:            {is_windows()}")
    print(f"Recommended Workers:   {get_recommended_num_workers()}")
    print()
    
    # Test safe num_workers
    test_values = [0, 1, 2, 4, 8]
    print("Safe num_workers (with Windows handling):")
    for val in test_values:
        safe_val = get_safe_num_workers(val, force_zero_on_windows=True, warn=False)
        print(f"  Requested: {val:2d}  →  Safe: {safe_val:2d}")
    print()
    
    # Example configuration
    print("Example DataLoader Configuration:")
    print_dataloader_config(
        num_workers=get_recommended_num_workers(),
        batch_size=32,
        seed=42,
        shuffle=True
    )
