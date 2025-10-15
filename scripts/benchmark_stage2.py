"""
Stage 2 Memory and Throughput Benchmark

Benchmarks Stage 2 Prithvi refinement model performance across different tile sizes.
Measures:
- GPU memory usage (allocated, reserved, peak)
- Inference throughput (samples/sec)
- Forward pass time
- Backward pass time (if training)

Results logged to: logs/stage2_bench.csv

Usage:
    # Benchmark with default settings
    python scripts/benchmark_stage2.py
    
    # Custom tile sizes
    python scripts/benchmark_stage2.py --tile_sizes 256 384 512
    
    # With training mode (forward + backward)
    python scripts/benchmark_stage2.py --mode train
    
    # Larger batch sizes
    python scripts/benchmark_stage2.py --batch_sizes 1 2 4

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
from pathlib import Path
import argparse
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


# ============================================================================
# Mock Stage 2 Model for Benchmarking
# ============================================================================

class MockPrithviRefiner(nn.Module):
    """
    Mock Prithvi refiner model that simulates Stage 2 architecture.
    
    Approximates the computational and memory characteristics of:
    - Prithvi backbone (frozen, 8-bit quantized)
    - LoRA adapters
    - ConvNeXt refinement head
    """
    
    def __init__(self, in_channels: int = 4, out_channels: int = 4):
        super().__init__()
        
        # Simulate Prithvi encoder (frozen backbone with LoRA)
        # Prithvi has ~600M params, we simulate with smaller but similar structure
        self.encoder = nn.Sequential(
            # Initial projection
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Encoder blocks (simulate transformer-like computation)
            self._make_encoder_block(64, 128),
            self._make_encoder_block(128, 256),
            self._make_encoder_block(256, 512),
        )
        
        # Simulate ConvNeXt refinement head
        self.refinement_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, out_channels, kernel_size=1),
        )
    
    def _make_encoder_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create encoder block simulating transformer-like computation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Simulate attention-like operations
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.encoder(x)
        refined = self.refinement_head(features)
        output = self.decoder(refined)
        return output


# ============================================================================
# Memory Tracking
# ============================================================================

def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get current GPU memory usage.
    
    Returns:
        Dict with memory stats in MB
    """
    if not torch.cuda.is_available():
        return {
            'allocated_mb': 0.0,
            'reserved_mb': 0.0,
            'peak_allocated_mb': 0.0,
        }
    
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    
    return {
        'allocated_mb': allocated,
        'reserved_mb': reserved,
        'peak_allocated_mb': peak_allocated,
    }


def reset_peak_memory_stats():
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_inference(
    model: nn.Module,
    tile_size: int,
    batch_size: int,
    device: torch.device,
    num_warmup: int = 10,
    num_iterations: int = 50
) -> Dict[str, float]:
    """
    Benchmark inference performance.
    
    Args:
        model: Model to benchmark
        tile_size: Input tile size (H, W)
        batch_size: Batch size
        device: Device to run on
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        
    Returns:
        Dict with benchmark results
    """
    model.eval()
    
    # Create dummy input
    input_shape = (batch_size, 4, tile_size, tile_size)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            x = torch.randn(input_shape, device=device)
            _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Reset memory stats
    reset_peak_memory_stats()
    
    # Benchmark
    times = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            x = torch.randn(input_shape, device=device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            output = model(x)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
    
    # Get memory stats
    memory_info = get_gpu_memory_info()
    
    # Compute statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = batch_size / avg_time  # samples/sec
    
    return {
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'throughput_samples_per_sec': throughput,
        'memory_allocated_mb': memory_info['allocated_mb'],
        'memory_reserved_mb': memory_info['reserved_mb'],
        'memory_peak_mb': memory_info['peak_allocated_mb'],
    }


def benchmark_training(
    model: nn.Module,
    tile_size: int,
    batch_size: int,
    device: torch.device,
    num_warmup: int = 5,
    num_iterations: int = 20
) -> Dict[str, float]:
    """
    Benchmark training performance (forward + backward).
    
    Args:
        model: Model to benchmark
        tile_size: Input tile size (H, W)
        batch_size: Batch size
        device: Device to run on
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        
    Returns:
        Dict with benchmark results
    """
    model.train()
    
    # Create dummy input and target
    input_shape = (batch_size, 4, tile_size, tile_size)
    
    # Simple optimizer for gradient computation
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Warmup
    for _ in range(num_warmup):
        x = torch.randn(input_shape, device=device)
        target = torch.randn(input_shape, device=device)
        
        output = model(x)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Reset memory stats
    reset_peak_memory_stats()
    
    # Benchmark
    forward_times = []
    backward_times = []
    total_times = []
    
    for _ in range(num_iterations):
        x = torch.randn(input_shape, device=device)
        target = torch.randn(input_shape, device=device)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Forward pass
        start_time = time.perf_counter()
        output = model(x)
        loss = criterion(output, target)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        forward_time = time.perf_counter() - start_time
        
        # Backward pass
        start_time = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        backward_time = time.perf_counter() - start_time
        
        optimizer.step()
        
        forward_times.append(forward_time)
        backward_times.append(backward_time)
        total_times.append(forward_time + backward_time)
    
    # Get memory stats
    memory_info = get_gpu_memory_info()
    
    # Compute statistics
    avg_forward = np.mean(forward_times)
    avg_backward = np.mean(backward_times)
    avg_total = np.mean(total_times)
    throughput = batch_size / avg_total  # samples/sec
    
    return {
        'forward_time_ms': avg_forward * 1000,
        'backward_time_ms': avg_backward * 1000,
        'total_time_ms': avg_total * 1000,
        'throughput_samples_per_sec': throughput,
        'memory_allocated_mb': memory_info['allocated_mb'],
        'memory_reserved_mb': memory_info['reserved_mb'],
        'memory_peak_mb': memory_info['peak_allocated_mb'],
    }


# ============================================================================
# Main Benchmark Runner
# ============================================================================

def run_benchmarks(
    tile_sizes: List[int],
    batch_sizes: List[int],
    mode: str,
    device: str,
    output_file: Path
) -> List[Dict]:
    """
    Run comprehensive benchmarks.
    
    Args:
        tile_sizes: List of tile sizes to benchmark
        batch_sizes: List of batch sizes to benchmark
        mode: 'inference' or 'train'
        device: Device to run on ('cuda' or 'cpu')
        output_file: CSV file to save results
        
    Returns:
        List of benchmark results
    """
    print("\n" + "=" * 80)
    print(f"Stage 2 Memory & Throughput Benchmark")
    print("=" * 80)
    
    device_obj = torch.device(device)
    print(f"\nDevice: {device}")
    
    if device == 'cuda' and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"Mode: {mode}")
    print(f"Tile sizes: {tile_sizes}")
    print(f"Batch sizes: {batch_sizes}")
    
    results = []
    
    # Progress bar
    total_configs = len(tile_sizes) * len(batch_sizes)
    pbar = tqdm(total=total_configs, desc="Benchmarking")
    
    for tile_size in tile_sizes:
        for batch_size in batch_sizes:
            config_name = f"tile_{tile_size}_batch_{batch_size}"
            
            try:
                # Create fresh model for each config
                model = MockPrithviRefiner(in_channels=4, out_channels=4)
                model = model.to(device_obj)
                
                # Get model size
                param_count = sum(p.numel() for p in model.parameters())
                param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
                
                # Run benchmark
                if mode == 'inference':
                    bench_results = benchmark_inference(
                        model, tile_size, batch_size, device_obj
                    )
                else:
                    bench_results = benchmark_training(
                        model, tile_size, batch_size, device_obj
                    )
                
                # Compile results
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'mode': mode,
                    'tile_size': tile_size,
                    'batch_size': batch_size,
                    'device': device,
                    'model_params': param_count,
                    'model_size_mb': param_size_mb,
                    **bench_results
                }
                
                results.append(result)
                
                # Update progress
                pbar.set_postfix({
                    'tile': tile_size,
                    'batch': batch_size,
                    'mem': f"{bench_results.get('memory_peak_mb', 0):.0f}MB"
                })
                pbar.update(1)
                
                # Cleanup
                del model
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠ OOM: tile_size={tile_size}, batch_size={batch_size}")
                    result = {
                        'timestamp': datetime.now().isoformat(),
                        'mode': mode,
                        'tile_size': tile_size,
                        'batch_size': batch_size,
                        'device': device,
                        'error': 'OOM',
                    }
                    results.append(result)
                    
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                else:
                    raise
            
            pbar.update(1)
    
    pbar.close()
    
    # Save results
    save_results_to_csv(results, output_file)
    
    return results


def save_results_to_csv(results: List[Dict], output_file: Path):
    """Save benchmark results to CSV."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine if file exists and has data
    file_exists = output_file.exists()
    
    if not results:
        print(f"\n⚠ No results to save")
        return
    
    # Get all possible fieldnames
    all_fields = set()
    for result in results:
        all_fields.update(result.keys())
    
    fieldnames = sorted(all_fields)
    
    # Write to CSV
    mode = 'a' if file_exists else 'w'
    with open(output_file, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for result in results:
            writer.writerow(result)
    
    print(f"\n✓ Results saved to: {output_file}")


def print_summary(results: List[Dict]):
    """Print benchmark summary."""
    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    
    if not results:
        print("\nNo results to summarize")
        return
    
    # Filter out OOM results
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        print("\nAll configurations resulted in OOM!")
        return
    
    print(f"\nSuccessful configurations: {len(valid_results)}/{len(results)}")
    
    # Group by tile size
    for tile_size in sorted(set(r['tile_size'] for r in valid_results)):
        tile_results = [r for r in valid_results if r['tile_size'] == tile_size]
        
        print(f"\nTile Size: {tile_size}x{tile_size}")
        print("-" * 80)
        
        for result in sorted(tile_results, key=lambda x: x['batch_size']):
            batch = result['batch_size']
            throughput = result.get('throughput_samples_per_sec', 0)
            memory = result.get('memory_peak_mb', 0)
            
            if 'total_time_ms' in result:
                time_str = f"{result['total_time_ms']:.1f}ms"
            else:
                time_str = f"{result.get('avg_time_ms', 0):.1f}ms"
            
            print(f"  Batch {batch}: {throughput:.2f} samples/sec | "
                  f"{time_str} | Mem: {memory:.0f} MB")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Stage 2 memory and throughput',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--tile_sizes',
        type=int,
        nargs='+',
        default=[256, 384],
        help='Tile sizes to benchmark (default: 256 384)'
    )
    
    parser.add_argument(
        '--batch_sizes',
        type=int,
        nargs='+',
        default=[1, 2, 4],
        help='Batch sizes to benchmark (default: 1 2 4)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['inference', 'train'],
        default='inference',
        help='Benchmark mode (default: inference)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('logs/stage2_bench.csv'),
        help='Output CSV file (default: logs/stage2_bench.csv)'
    )
    
    args = parser.parse_args()
    
    # Run benchmarks
    results = run_benchmarks(
        tile_sizes=args.tile_sizes,
        batch_sizes=args.batch_sizes,
        mode=args.mode,
        device=args.device,
        output_file=args.output
    )
    
    # Print summary
    print_summary(results)
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
