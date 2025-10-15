"""
Stage 1 Profiling: Measure Training Throughput

This script profiles TerraMind Stage 1 training to measure throughput
(samples/second) for different configurations:
    - Tile sizes: 256x256, 384x384
    - Timesteps: 8, 12, 16

Profiles both forward and backward passes to get realistic training throughput.

Usage:
    python scripts/profile_stage1.py
    python scripts/profile_stage1.py --device cuda:0
    python scripts/profile_stage1.py --batch-size 4 --warmup 5 --iterations 20
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axs_lib.models import build_terramind_generator
from axs_lib.losses import CombinedLoss


# ============================================================================
# Profiling Utilities
# ============================================================================

class ProfileResult:
    """Container for profiling results."""
    
    def __init__(
        self,
        tile_size: int,
        timesteps: int,
        batch_size: int,
        device: str
    ):
        self.tile_size = tile_size
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.device = device
        
        # Timing measurements
        self.forward_times: List[float] = []
        self.backward_times: List[float] = []
        self.total_times: List[float] = []
        
        # Memory measurements
        self.peak_memory_mb: float = 0.0
        self.allocated_memory_mb: float = 0.0
        
    def add_measurement(
        self,
        forward_time: float,
        backward_time: float,
        peak_memory: float,
        allocated_memory: float
    ):
        """Add a timing measurement."""
        self.forward_times.append(forward_time)
        self.backward_times.append(backward_time)
        self.total_times.append(forward_time + backward_time)
        self.peak_memory_mb = max(self.peak_memory_mb, peak_memory / 1024**2)
        self.allocated_memory_mb = max(self.allocated_memory_mb, allocated_memory / 1024**2)
    
    def get_statistics(self) -> Dict:
        """Compute profiling statistics."""
        if not self.total_times:
            return {}
        
        forward_mean = np.mean(self.forward_times)
        backward_mean = np.mean(self.backward_times)
        total_mean = np.mean(self.total_times)
        
        # Throughput (samples/second)
        throughput = self.batch_size / total_mean
        
        return {
            'config': {
                'tile_size': self.tile_size,
                'timesteps': self.timesteps,
                'batch_size': self.batch_size,
                'device': self.device
            },
            'timing': {
                'forward_ms': forward_mean * 1000,
                'backward_ms': backward_mean * 1000,
                'total_ms': total_mean * 1000,
                'forward_std_ms': np.std(self.forward_times) * 1000,
                'backward_std_ms': np.std(self.backward_times) * 1000,
                'total_std_ms': np.std(self.total_times) * 1000,
            },
            'throughput': {
                'samples_per_second': throughput,
                'seconds_per_sample': total_mean / self.batch_size,
                'ms_per_sample': (total_mean / self.batch_size) * 1000,
            },
            'memory': {
                'peak_memory_mb': self.peak_memory_mb,
                'allocated_memory_mb': self.allocated_memory_mb,
                'peak_memory_gb': self.peak_memory_mb / 1024,
            }
        }


def create_dummy_batch(
    batch_size: int,
    tile_size: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create dummy input/target batch for profiling.
    
    Args:
        batch_size: Number of samples
        tile_size: Spatial size (height/width)
        device: Device to create tensors on
        
    Returns:
        (s1_input, s2_target) tensors
    """
    # SAR input: 2 channels (VV, VH)
    s1_input = torch.randn(
        batch_size, 2, tile_size, tile_size,
        device=device,
        dtype=torch.float32
    )
    
    # Optical target: 4 channels (B02, B03, B04, B08)
    s2_target = torch.randn(
        batch_size, 4, tile_size, tile_size,
        device=device,
        dtype=torch.float32
    )
    
    return s1_input, s2_target


def profile_configuration(
    tile_size: int,
    timesteps: int,
    batch_size: int,
    device: str,
    warmup_iterations: int = 3,
    profile_iterations: int = 10,
    use_amp: bool = True,
    verbose: bool = True
) -> ProfileResult:
    """
    Profile a specific configuration.
    
    Args:
        tile_size: Spatial size (256 or 384)
        timesteps: Number of diffusion timesteps (8, 12, or 16)
        batch_size: Batch size
        device: Device to run on
        warmup_iterations: Number of warmup iterations (not measured)
        profile_iterations: Number of iterations to measure
        use_amp: Use automatic mixed precision
        verbose: Print progress
        
    Returns:
        ProfileResult with timing and memory statistics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Profiling: tile_size={tile_size}, timesteps={timesteps}, batch_size={batch_size}")
        print(f"{'='*70}")
    
    # Create model
    if verbose:
        print("Building TerraMind generator...")
    
    model = build_terramind_generator(
        input_modalities=("S1GRD",),
        output_modalities=("S2L2A",),
        timesteps=timesteps,
        standardize=True,
        pretrained=False  # Don't load weights for profiling
    )
    model = model.to(device)
    model.train()
    
    # Create loss function
    loss_fn = CombinedLoss(
        l1_weight=1.0,
        ms_ssim_weight=0.5,
        lpips_weight=0.1,
        sar_struct_weight=0.0  # Disable for profiling
    )
    loss_fn = loss_fn.to(device)
    
    # Create optimizer (needed for backward pass)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create gradient scaler for AMP
    scaler = GradScaler() if use_amp else None
    
    # Create result container
    result = ProfileResult(tile_size, timesteps, batch_size, device)
    
    # Warmup iterations
    if verbose:
        print(f"Warming up ({warmup_iterations} iterations)...")
    
    for i in range(warmup_iterations):
        s1_input, s2_target = create_dummy_batch(batch_size, tile_size, device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                output = model(s1_input)
                loss = loss_fn(output, s2_target, s1_input)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(s1_input)
            loss = loss_fn(output, s2_target, s1_input)
            loss.backward()
            optimizer.step()
        
        # Synchronize GPU
        if device.startswith('cuda'):
            torch.cuda.synchronize()
    
    if verbose:
        print(f"Profiling ({profile_iterations} iterations)...")
    
    # Profile iterations
    for i in range(profile_iterations):
        s1_input, s2_target = create_dummy_batch(batch_size, tile_size, device)
        
        optimizer.zero_grad()
        
        # Reset memory stats
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        
        # Time forward pass
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        forward_start = time.perf_counter()
        
        if use_amp:
            with autocast():
                output = model(s1_input)
                loss = loss_fn(output, s2_target, s1_input)
        else:
            output = model(s1_input)
            loss = loss_fn(output, s2_target, s1_input)
        
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        forward_end = time.perf_counter()
        forward_time = forward_end - forward_start
        
        # Time backward pass
        backward_start = time.perf_counter()
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        backward_end = time.perf_counter()
        backward_time = backward_end - backward_start
        
        # Measure memory
        if device.startswith('cuda'):
            peak_memory = torch.cuda.max_memory_allocated()
            allocated_memory = torch.cuda.memory_allocated()
        else:
            peak_memory = 0
            allocated_memory = 0
        
        # Record measurement
        result.add_measurement(forward_time, backward_time, peak_memory, allocated_memory)
        
        if verbose:
            total_time = forward_time + backward_time
            throughput = batch_size / total_time
            print(f"  Iter {i+1}/{profile_iterations}: "
                  f"forward={forward_time*1000:.1f}ms, "
                  f"backward={backward_time*1000:.1f}ms, "
                  f"total={total_time*1000:.1f}ms, "
                  f"throughput={throughput:.2f} samples/s")
    
    # Clean up
    del model, loss_fn, optimizer
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    
    return result


# ============================================================================
# Main Profiling Script
# ============================================================================

def print_results_table(results: List[ProfileResult]):
    """Print results in a formatted table."""
    print("\n" + "="*100)
    print("PROFILING RESULTS SUMMARY")
    print("="*100)
    
    # Table header
    print(f"\n{'Tile Size':<12} {'Timesteps':<12} {'Batch':<8} "
          f"{'Forward (ms)':<15} {'Backward (ms)':<15} {'Total (ms)':<15} "
          f"{'Throughput':<15} {'Memory (GB)':<12}")
    print("-"*100)
    
    # Table rows
    for result in results:
        stats = result.get_statistics()
        config = stats['config']
        timing = stats['timing']
        throughput = stats['throughput']
        memory = stats['memory']
        
        print(f"{config['tile_size']:<12} "
              f"{config['timesteps']:<12} "
              f"{config['batch_size']:<8} "
              f"{timing['forward_ms']:<15.1f} "
              f"{timing['backward_ms']:<15.1f} "
              f"{timing['total_ms']:<15.1f} "
              f"{throughput['samples_per_second']:<15.2f} "
              f"{memory['peak_memory_gb']:<12.2f}")
    
    print("-"*100)


def print_detailed_results(results: List[ProfileResult]):
    """Print detailed results for each configuration."""
    print("\n" + "="*100)
    print("DETAILED PROFILING RESULTS")
    print("="*100)
    
    for result in results:
        stats = result.get_statistics()
        config = stats['config']
        timing = stats['timing']
        throughput = stats['throughput']
        memory = stats['memory']
        
        print(f"\n{'='*70}")
        print(f"Configuration: tile_size={config['tile_size']}, "
              f"timesteps={config['timesteps']}, batch_size={config['batch_size']}")
        print(f"{'='*70}")
        
        print("\nTiming:")
        print(f"  Forward pass:  {timing['forward_ms']:>8.1f} ± {timing['forward_std_ms']:>6.1f} ms")
        print(f"  Backward pass: {timing['backward_ms']:>8.1f} ± {timing['backward_std_ms']:>6.1f} ms")
        print(f"  Total:         {timing['total_ms']:>8.1f} ± {timing['total_std_ms']:>6.1f} ms")
        
        print("\nThroughput:")
        print(f"  Samples/second:     {throughput['samples_per_second']:>8.2f}")
        print(f"  Seconds/sample:     {throughput['seconds_per_sample']:>8.3f}")
        print(f"  Milliseconds/sample: {throughput['ms_per_sample']:>8.1f}")
        
        print("\nMemory:")
        print(f"  Peak memory:      {memory['peak_memory_mb']:>8.1f} MB ({memory['peak_memory_gb']:.2f} GB)")
        print(f"  Allocated memory: {memory['allocated_memory_mb']:>8.1f} MB")


def save_results_json(results: List[ProfileResult], output_path: str):
    """Save results to JSON file."""
    output_data = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': results[0].device if results else 'unknown',
            'num_configurations': len(results)
        },
        'results': [r.get_statistics() for r in results]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile TerraMind Stage 1 training throughput"
    )
    
    # Configuration
    parser.add_argument(
        '--tile-sizes',
        type=int,
        nargs='+',
        default=[256, 384],
        help='Tile sizes to profile (default: 256 384)'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        nargs='+',
        default=[8, 12, 16],
        help='Timestep values to profile (default: 8 12 16)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size (default: 1)'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to profile on (default: cuda if available, else cpu)'
    )
    
    # Profiling parameters
    parser.add_argument(
        '--warmup',
        type=int,
        default=3,
        help='Number of warmup iterations (default: 3)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Number of profiling iterations (default: 10)'
    )
    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='Disable automatic mixed precision'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save JSON results (default: profile_results_<timestamp>.json)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Validate device
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("ERROR: CUDA not available. Using CPU instead.")
        args.device = 'cpu'
    
    # Print configuration
    if not args.quiet:
        print("="*70)
        print("TERRAMIND STAGE 1 PROFILING")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Tile sizes:  {args.tile_sizes}")
        print(f"  Timesteps:   {args.timesteps}")
        print(f"  Batch size:  {args.batch_size}")
        print(f"  Device:      {args.device}")
        print(f"  Warmup:      {args.warmup} iterations")
        print(f"  Profile:     {args.iterations} iterations")
        print(f"  AMP:         {not args.no_amp}")
        
        if args.device.startswith('cuda'):
            print(f"\nGPU Info:")
            print(f"  Name:        {torch.cuda.get_device_name(args.device)}")
            print(f"  Memory:      {torch.cuda.get_device_properties(args.device).total_memory / 1024**3:.1f} GB")
    
    # Run profiling for all configurations
    results = []
    total_configs = len(args.tile_sizes) * len(args.timesteps)
    current_config = 0
    
    for tile_size in args.tile_sizes:
        for timesteps in args.timesteps:
            current_config += 1
            
            if not args.quiet:
                print(f"\n[{current_config}/{total_configs}] Profiling configuration...")
            
            try:
                result = profile_configuration(
                    tile_size=tile_size,
                    timesteps=timesteps,
                    batch_size=args.batch_size,
                    device=args.device,
                    warmup_iterations=args.warmup,
                    profile_iterations=args.iterations,
                    use_amp=not args.no_amp,
                    verbose=not args.quiet
                )
                results.append(result)
                
            except Exception as e:
                print(f"\nERROR profiling tile_size={tile_size}, timesteps={timesteps}:")
                print(f"  {e}")
                import traceback
                traceback.print_exc()
    
    # Print results
    if results:
        print_results_table(results)
        
        if not args.quiet:
            print_detailed_results(results)
        
        # Save to JSON
        if args.output is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            args.output = f"profile_results_{timestamp}.json"
        
        save_results_json(results, args.output)
        
        # Print recommendations
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        
        # Find fastest configuration
        best_result = max(results, key=lambda r: r.get_statistics()['throughput']['samples_per_second'])
        best_stats = best_result.get_statistics()
        
        print(f"\nFastest configuration:")
        print(f"  Tile size:  {best_stats['config']['tile_size']}")
        print(f"  Timesteps:  {best_stats['config']['timesteps']}")
        print(f"  Throughput: {best_stats['throughput']['samples_per_second']:.2f} samples/s")
        print(f"  Memory:     {best_stats['memory']['peak_memory_gb']:.2f} GB")
        
        # Find lowest memory configuration
        lowest_memory = min(results, key=lambda r: r.get_statistics()['memory']['peak_memory_gb'])
        mem_stats = lowest_memory.get_statistics()
        
        print(f"\nLowest memory configuration:")
        print(f"  Tile size:  {mem_stats['config']['tile_size']}")
        print(f"  Timesteps:  {mem_stats['config']['timesteps']}")
        print(f"  Throughput: {mem_stats['throughput']['samples_per_second']:.2f} samples/s")
        print(f"  Memory:     {mem_stats['memory']['peak_memory_gb']:.2f} GB")
        
    else:
        print("\nNo successful profiling results.")


if __name__ == '__main__':
    main()
