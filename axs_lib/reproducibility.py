"""
axs_lib/reproducibility.py - Reproducibility Utilities

Ensures deterministic training by setting all random seeds and configuring
backends for reproducible behavior. Seeds are saved in checkpoint metadata
to enable exact reproduction of results.

Key Features:
    - Sets seeds for torch, numpy, random, and CUDA
    - Configures CUDA determinism (with performance warnings)
    - Saves seed metadata in checkpoints
    - Validates reproducibility setup

Usage:
    >>> from axs_lib.reproducibility import set_seed, get_seed_state
    >>> 
    >>> # Set all seeds for reproducibility
    >>> set_seed(42)
    >>> 
    >>> # Get current seed state for checkpoint
    >>> seed_state = get_seed_state()
    >>> torch.save({'model': model.state_dict(), 'seed_state': seed_state}, 'ckpt.pt')
    >>> 
    >>> # Restore seed state from checkpoint
    >>> ckpt = torch.load('ckpt.pt')
    >>> restore_seed_state(ckpt['seed_state'])

Author: Axion-Sat Project
"""

import os
import random
import warnings
from typing import Dict, Optional, Any
import numpy as np


# Check for torch availability
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, AttributeError) as e:
    TORCH_AVAILABLE = False
    if 'torch' not in str(e):
        # Not a missing torch error, might be import issue
        warnings.warn(f"PyTorch import issue: {e}. Reproducibility limited to numpy and random.")
    else:
        warnings.warn("PyTorch not available. Reproducibility limited to numpy and random.")


# ============================================================================
# Seed Setting Functions
# ============================================================================

def set_seed(
    seed: int,
    deterministic: bool = True,
    benchmark: bool = False,
    warn: bool = True
) -> Dict[str, Any]:
    """
    Set all random seeds for reproducibility.
    
    Sets seeds for:
        - Python's random module
        - NumPy
        - PyTorch (if available)
        - CUDA (if available)
    
    Args:
        seed: Random seed value (0-2^32-1)
        deterministic: Enable deterministic CUDA operations (slower but reproducible)
        benchmark: Enable CUDNN benchmarking (faster but non-deterministic)
        warn: Print warnings about performance implications
        
    Returns:
        Dictionary with seed configuration metadata
        
    Example:
        >>> import torch
        >>> from axs_lib.reproducibility import set_seed
        >>> 
        >>> # For reproducible training
        >>> seed_state = set_seed(42, deterministic=True, benchmark=False)
        >>> 
        >>> # For faster training (non-reproducible)
        >>> seed_state = set_seed(42, deterministic=False, benchmark=True)
        
    Note:
        - deterministic=True may slow training by 10-30%
        - benchmark=True speeds up training but breaks reproducibility
        - For exact reproducibility: deterministic=True, benchmark=False
    """
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed must be in range [0, 2^32-1], got {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    seed_info = {
        'seed': seed,
        'deterministic': deterministic,
        'benchmark': benchmark,
        'torch_available': TORCH_AVAILABLE,
        'cuda_available': False,
        'cuda_device_count': 0
    }
    
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        
        # CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            seed_info['cuda_available'] = True
            seed_info['cuda_device_count'] = torch.cuda.device_count()
            
            # CUDNN settings
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                if warn:
                    warnings.warn(
                        "CUDNN deterministic mode enabled. This may reduce performance by 10-30%. "
                        "Set deterministic=False for faster training (non-reproducible).",
                        UserWarning
                    )
            else:
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = benchmark
                
                if warn and not benchmark:
                    warnings.warn(
                        "CUDNN deterministic mode disabled. Results may not be exactly reproducible. "
                        "Set deterministic=True for reproducibility.",
                        UserWarning
                    )
            
            seed_info['cudnn_deterministic'] = torch.backends.cudnn.deterministic
            seed_info['cudnn_benchmark'] = torch.backends.cudnn.benchmark
            
            # Set device for deterministic behavior
            if deterministic and hasattr(torch, 'use_deterministic_algorithms'):
                try:
                    # Use warn_only=True to allow non-deterministic operations like cumsum
                    # while still enforcing determinism where possible
                    torch.use_deterministic_algorithms(True, warn_only=True)
                    seed_info['use_deterministic_algorithms'] = True
                    seed_info['use_deterministic_algorithms_warn_only'] = True
                except RuntimeError as e:
                    if warn:
                        warnings.warn(
                            f"Could not enable deterministic algorithms: {e}. "
                            "Some operations may still be non-deterministic.",
                            UserWarning
                        )
                    seed_info['use_deterministic_algorithms'] = False
                    seed_info['use_deterministic_algorithms_warn_only'] = False
            else:
                seed_info['use_deterministic_algorithms'] = False
                seed_info['use_deterministic_algorithms_warn_only'] = False
    
    # Environment variable for additional reproducibility
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        seed_info['cublas_workspace_config'] = ':4096:8'
    
    return seed_info


def get_seed_state() -> Dict[str, Any]:
    """
    Get current random state for all RNGs.
    
    Captures the current state of:
        - Python's random module
        - NumPy's random state
        - PyTorch's RNG state
        - CUDA RNG states (all devices)
    
    Returns:
        Dictionary with all RNG states
        
    Example:
        >>> state = get_seed_state()
        >>> # ... training ...
        >>> restore_seed_state(state)  # Restore to exact same state
    """
    state = {
        'random_state': random.getstate(),
        'numpy_state': np.random.get_state(),
        'torch_available': TORCH_AVAILABLE
    }
    
    if TORCH_AVAILABLE:
        state['torch_rng_state'] = torch.get_rng_state()
        
        if torch.cuda.is_available():
            state['cuda_available'] = True
            state['cuda_rng_state'] = torch.cuda.get_rng_state()
            
            # All GPU states
            if torch.cuda.device_count() > 1:
                state['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        else:
            state['cuda_available'] = False
    
    return state


def restore_seed_state(state: Dict[str, Any]):
    """
    Restore random state from saved state dictionary.
    
    Args:
        state: Dictionary from get_seed_state()
        
    Example:
        >>> # Save state before training
        >>> state = get_seed_state()
        >>> 
        >>> # ... training ...
        >>> 
        >>> # Restore to exact same state
        >>> restore_seed_state(state)
    """
    # Python random
    if 'random_state' in state:
        random.setstate(state['random_state'])
    
    # NumPy
    if 'numpy_state' in state:
        np.random.set_state(state['numpy_state'])
    
    # PyTorch
    if TORCH_AVAILABLE and state.get('torch_available', False):
        if 'torch_rng_state' in state:
            torch.set_rng_state(state['torch_rng_state'])
        
        # CUDA
        if state.get('cuda_available', False) and torch.cuda.is_available():
            if 'cuda_rng_state' in state:
                torch.cuda.set_rng_state(state['cuda_rng_state'])
            
            if 'cuda_rng_state_all' in state and torch.cuda.device_count() > 1:
                torch.cuda.set_rng_state_all(state['cuda_rng_state_all'])


def seed_worker(worker_id: int):
    """
    Seed function for DataLoader workers to ensure reproducibility.
    
    Use with DataLoader's worker_init_fn parameter:
        DataLoader(dataset, worker_init_fn=seed_worker, ...)
    
    Args:
        worker_id: Worker ID (provided by DataLoader)
        
    Example:
        >>> from torch.utils.data import DataLoader
        >>> from axs_lib.reproducibility import seed_worker
        >>> 
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     num_workers=4,
        ...     worker_init_fn=seed_worker  # Ensures reproducible worker seeds
        ... )
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================================
# Generator for Reproducible Data Splitting
# ============================================================================

def create_reproducible_generator(seed: int):
    """
    Create a PyTorch Generator for reproducible random operations.
    
    Use for reproducible data splitting, sampling, etc.
    
    Args:
        seed: Random seed
        
    Returns:
        torch.Generator with fixed seed
        
    Example:
        >>> from torch.utils.data import random_split
        >>> from axs_lib.reproducibility import create_reproducible_generator
        >>> 
        >>> generator = create_reproducible_generator(42)
        >>> train_set, val_set = random_split(
        ...     dataset, [0.8, 0.2], generator=generator
        ... )
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for Generator")
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# ============================================================================
# Checkpoint Utilities
# ============================================================================

def add_seed_to_checkpoint(
    checkpoint: Dict[str, Any],
    seed_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Add seed information to checkpoint dictionary.
    
    Args:
        checkpoint: Checkpoint dictionary (will be modified in-place)
        seed_info: Seed configuration (from set_seed), or None to capture current state
        
    Returns:
        Updated checkpoint dictionary
        
    Example:
        >>> seed_info = set_seed(42)
        >>> checkpoint = {
        ...     'model_state_dict': model.state_dict(),
        ...     'optimizer_state_dict': optimizer.state_dict()
        ... }
        >>> add_seed_to_checkpoint(checkpoint, seed_info)
        >>> torch.save(checkpoint, 'model.pt')
    """
    if seed_info is None:
        seed_info = {}
    
    # Add seed configuration
    checkpoint['seed_info'] = seed_info
    
    # Add current RNG states
    checkpoint['rng_state'] = get_seed_state()
    
    return checkpoint


def validate_checkpoint_seed(
    checkpoint: Dict[str, Any],
    expected_seed: Optional[int] = None,
    warn_on_mismatch: bool = True
) -> bool:
    """
    Validate seed information in checkpoint.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        expected_seed: Expected seed value (None to skip validation)
        warn_on_mismatch: Print warning if seeds don't match
        
    Returns:
        True if seed is valid/matches, False otherwise
        
    Example:
        >>> checkpoint = torch.load('model.pt')
        >>> if validate_checkpoint_seed(checkpoint, expected_seed=42):
        ...     print("Seed validated!")
        ... else:
        ...     print("Warning: Seed mismatch!")
    """
    if 'seed_info' not in checkpoint:
        if warn_on_mismatch:
            warnings.warn("Checkpoint does not contain seed information!", UserWarning)
        return False
    
    seed_info = checkpoint['seed_info']
    
    if expected_seed is not None:
        checkpoint_seed = seed_info.get('seed')
        if checkpoint_seed != expected_seed:
            if warn_on_mismatch:
                warnings.warn(
                    f"Seed mismatch! Expected {expected_seed}, checkpoint has {checkpoint_seed}",
                    UserWarning
                )
            return False
    
    return True


# ============================================================================
# Reproducibility Verification
# ============================================================================

def verify_reproducibility(
    num_iterations: int = 10,
    verbose: bool = True
) -> bool:
    """
    Verify that reproducibility is working correctly.
    
    Runs a simple test to check if operations produce identical results
    with the same seed.
    
    Args:
        num_iterations: Number of test iterations
        verbose: Print detailed information
        
    Returns:
        True if reproducibility is verified, False otherwise
        
    Example:
        >>> from axs_lib.reproducibility import set_seed, verify_reproducibility
        >>> 
        >>> set_seed(42, deterministic=True)
        >>> if verify_reproducibility():
        ...     print("Reproducibility verified!")
    """
    if not TORCH_AVAILABLE:
        if verbose:
            print("⚠️  PyTorch not available, skipping verification")
        return False
    
    if verbose:
        print("Testing reproducibility...")
        print(f"  Running {num_iterations} iterations...")
    
    results = []
    
    for i in range(num_iterations):
        # Set seed
        set_seed(42, deterministic=True, benchmark=False, warn=False)
        
        # Generate random data
        torch_rand = torch.randn(100).sum().item()
        numpy_rand = np.random.randn(100).sum()
        python_rand = random.random()
        
        results.append((torch_rand, numpy_rand, python_rand))
    
    # Check if all iterations produced identical results
    first_result = results[0]
    all_match = all(r == first_result for r in results)
    
    if verbose:
        if all_match:
            print("  ✓ All iterations produced identical results")
            print("  ✓ Reproducibility verified!")
        else:
            print("  ✗ Results varied across iterations")
            print("  ✗ Reproducibility NOT verified!")
            print(f"  First result: {first_result}")
            print(f"  Other results: {[r for r in results if r != first_result][:3]}")
    
    return all_match


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test and verify reproducibility setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify reproducibility
  python axs_lib/reproducibility.py --verify
  
  # Test with specific seed
  python axs_lib/reproducibility.py --seed 42 --verify
  
  # Show seed state
  python axs_lib/reproducibility.py --show-state
        """
    )
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify reproducibility')
    parser.add_argument('--show-state', action='store_true',
                        help='Show current RNG state')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Enable deterministic mode')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of verification iterations (default: 10)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Reproducibility Utilities")
    print("=" * 80)
    print()
    
    if args.verify:
        print(f"Setting seed: {args.seed}")
        seed_info = set_seed(args.seed, deterministic=args.deterministic, warn=False)
        print()
        
        print("Seed Configuration:")
        for key, value in seed_info.items():
            print(f"  {key}: {value}")
        print()
        
        success = verify_reproducibility(num_iterations=args.iterations, verbose=True)
        print()
        
        if success:
            print("=" * 80)
            print("✓ Reproducibility verified!")
            print("=" * 80)
            exit(0)
        else:
            print("=" * 80)
            print("✗ Reproducibility verification failed!")
            print("=" * 80)
            exit(1)
    
    elif args.show_state:
        print("Current RNG State:")
        state = get_seed_state()
        for key, value in state.items():
            if 'state' not in key.lower():  # Skip large state objects
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: <state object>")
        print()
    
    else:
        parser.print_help()
