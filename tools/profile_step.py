"""
CUDA profiling tool for analyzing per-operation timing in forward/backward passes.

This tool helps identify performance bottlenecks by profiling:
- Individual CUDA operations (kernels)
- Forward pass timing
- Backward pass timing
- Memory allocation/deallocation
- Data transfers (CPU ↔ GPU)

Usage:
    python tools/profile_step.py --model <model_name> --input-shape 1 3 256 256
    python tools/profile_step.py --model resnet50 --batch-size 16 --profile-memory
    python tools/profile_step.py --help
"""

import argparse
import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity


# ============================================================================
# Profiling Functions
# ============================================================================

def profile_single_step(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = 'cuda',
    warmup_steps: int = 3,
    profile_steps: int = 1,
    profile_memory: bool = False,
    with_backward: bool = True
) -> Tuple[str, Dict]:
    """
    Profile a single forward/backward step with detailed per-op timing.
    
    Args:
        model: PyTorch model to profile
        input_tensor: Input tensor for the model
        device: Device to run on ('cuda' or 'cpu')
        warmup_steps: Number of warmup iterations before profiling
        profile_steps: Number of steps to profile
        profile_memory: Whether to profile memory usage
        with_backward: Whether to include backward pass
        
    Returns:
        Tuple of (profile_output_string, summary_dict)
    """
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warmup
    print(f"Running {warmup_steps} warmup steps...")
    with torch.no_grad():
        for _ in range(warmup_steps):
            output = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
    
    # Setup profiler
    activities = [ProfilerActivity.CPU]
    if device == 'cuda' and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    
    profile_kwargs = {
        'activities': activities,
        'record_shapes': True,
        'profile_memory': profile_memory,
        'with_stack': False,  # Set to True for detailed stack traces (slower)
        'with_flops': True,
    }
    
    print(f"\nProfiling {profile_steps} step(s) with backward={with_backward}...")
    print("=" * 80)
    
    with profile(**profile_kwargs) as prof:
        for step in range(profile_steps):
            with record_function(f"step_{step}"):
                # Forward pass
                with record_function("forward"):
                    output = model(input_tensor)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                
                # Backward pass
                if with_backward:
                    with record_function("backward"):
                        # Create dummy loss
                        if isinstance(output, tuple):
                            loss = output[0].sum()
                        else:
                            loss = output.sum()
                        
                        loss.backward()
                        
                        if device == 'cuda':
                            torch.cuda.synchronize()
                        
                        # Clear gradients for next iteration
                        model.zero_grad()
    
    # Get profile output
    return prof, prof.key_averages()


def print_profile_summary(
    key_averages,
    sort_by: str = 'cuda_time_total',
    top_k: int = 20,
    row_limit: int = 100
):
    """
    Print formatted profile summary.
    
    Args:
        key_averages: Profiler key averages
        sort_by: Metric to sort by ('cuda_time_total', 'cpu_time_total', 'count', etc.)
        top_k: Number of top operations to highlight
        row_limit: Maximum number of rows to display
    """
    print("\n" + "=" * 80)
    print(f"TOP {top_k} OPERATIONS (sorted by {sort_by})")
    print("=" * 80)
    
    # Print table with CUDA time
    print(key_averages.table(
        sort_by=sort_by,
        row_limit=row_limit,
        max_src_column_width=60
    ))


def analyze_profile_breakdown(key_averages) -> Dict[str, float]:
    """
    Analyze profile data and compute breakdown by category.
    
    Returns:
        Dictionary with timing breakdown
    """
    breakdown = {
        'forward_time_ms': 0.0,
        'backward_time_ms': 0.0,
        'total_cuda_time_ms': 0.0,
        'total_cpu_time_ms': 0.0,
        'memory_ops_time_ms': 0.0,
        'compute_ops_time_ms': 0.0,
    }
    
    for evt in key_averages:
        cuda_time_ms = evt.cuda_time_total / 1000.0  # Convert µs to ms
        cpu_time_ms = evt.cpu_time_total / 1000.0
        
        # Categorize by operation name
        op_name = evt.key.lower()
        
        if 'forward' in op_name:
            breakdown['forward_time_ms'] += cuda_time_ms
        elif 'backward' in op_name:
            breakdown['backward_time_ms'] += cuda_time_ms
        
        if 'memcpy' in op_name or 'memset' in op_name:
            breakdown['memory_ops_time_ms'] += cuda_time_ms
        else:
            breakdown['compute_ops_time_ms'] += cuda_time_ms
        
        breakdown['total_cuda_time_ms'] += cuda_time_ms
        breakdown['total_cpu_time_ms'] += cpu_time_ms
    
    return breakdown


def print_breakdown_summary(breakdown: Dict[str, float]):
    """Print summary breakdown of profiling results."""
    print("\n" + "=" * 80)
    print("TIMING BREAKDOWN SUMMARY")
    print("=" * 80)
    
    total_time = breakdown['total_cuda_time_ms']
    
    if total_time > 0:
        print(f"\nTotal CUDA Time: {total_time:.2f} ms")
        print(f"Total CPU Time:  {breakdown['total_cpu_time_ms']:.2f} ms")
        print()
        
        if breakdown['forward_time_ms'] > 0:
            fwd_pct = 100 * breakdown['forward_time_ms'] / total_time
            print(f"Forward Pass:    {breakdown['forward_time_ms']:8.2f} ms ({fwd_pct:5.1f}%)")
        
        if breakdown['backward_time_ms'] > 0:
            bwd_pct = 100 * breakdown['backward_time_ms'] / total_time
            print(f"Backward Pass:   {breakdown['backward_time_ms']:8.2f} ms ({bwd_pct:5.1f}%)")
        
        print()
        compute_pct = 100 * breakdown['compute_ops_time_ms'] / total_time
        memory_pct = 100 * breakdown['memory_ops_time_ms'] / total_time
        print(f"Compute Ops:     {breakdown['compute_ops_time_ms']:8.2f} ms ({compute_pct:5.1f}%)")
        print(f"Memory Ops:      {breakdown['memory_ops_time_ms']:8.2f} ms ({memory_pct:5.1f}%)")
    else:
        print("No CUDA time recorded (running on CPU or no CUDA operations)")


def save_profile_chrome_trace(prof, output_path: str):
    """
    Save profile in Chrome trace format for visualization.
    
    Can be viewed at chrome://tracing
    """
    prof.export_chrome_trace(output_path)
    print(f"\nChrome trace saved to: {output_path}")
    print("View at: chrome://tracing")


def print_memory_summary(key_averages):
    """Print memory usage summary."""
    print("\n" + "=" * 80)
    print("MEMORY OPERATIONS")
    print("=" * 80)
    
    # Filter memory-related operations
    memory_ops = [
        evt for evt in key_averages 
        if 'memory' in evt.key.lower() or 
           'alloc' in evt.key.lower() or 
           'memcpy' in evt.key.lower() or
           'memset' in evt.key.lower()
    ]
    
    if memory_ops:
        print(f"\n{'Operation':<50} {'Count':>8} {'CUDA Time (ms)':>15}")
        print("-" * 80)
        
        for evt in sorted(memory_ops, key=lambda x: x.cuda_time_total, reverse=True)[:20]:
            cuda_time_ms = evt.cuda_time_total / 1000.0
            print(f"{evt.key:<50} {evt.count:>8} {cuda_time_ms:>15.3f}")
    else:
        print("No memory operations recorded")


# ============================================================================
# Example Models
# ============================================================================

def create_example_model(model_name: str, **kwargs) -> nn.Module:
    """Create an example model for profiling."""
    
    if model_name == 'simple_conv':
        # Simple convolutional model
        class SimpleConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.relu = nn.ReLU(inplace=True)
                self.fc = nn.Linear(256 * 32 * 32, 10)
            
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.pool(x)
                x = self.relu(self.conv2(x))
                x = self.pool(x)
                x = self.relu(self.conv3(x))
                x = x.flatten(1)
                x = self.fc(x)
                return x
        
        return SimpleConv()
    
    elif model_name == 'resnet_block':
        # Single ResNet block
        class ResNetBlock(nn.Module):
            def __init__(self, channels=64):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(channels)
                self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(channels)
                self.relu = nn.ReLU(inplace=True)
            
            def forward(self, x):
                identity = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += identity
                out = self.relu(out)
                return out
        
        return ResNetBlock()
    
    elif model_name == 'transformer_block':
        # Transformer encoder block
        class TransformerBlock(nn.Module):
            def __init__(self, d_model=512, nhead=8, dim_feedforward=2048):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
                self.linear1 = nn.Linear(d_model, dim_feedforward)
                self.linear2 = nn.Linear(dim_feedforward, d_model)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(0.1)
                self.activation = nn.ReLU()
            
            def forward(self, x):
                # Self attention
                attn_out, _ = self.self_attn(x, x, x)
                x = self.norm1(x + self.dropout(attn_out))
                
                # Feedforward
                ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
                x = self.norm2(x + self.dropout(ff_out))
                
                return x
        
        return TransformerBlock()
    
    else:
        # Try to load from torchvision
        try:
            import torchvision.models as models
            
            if hasattr(models, model_name):
                model_fn = getattr(models, model_name)
                return model_fn(**kwargs)
            else:
                raise ValueError(f"Unknown model: {model_name}")
        except ImportError:
            raise ValueError(f"Unknown model: {model_name} (torchvision not available)")


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile CUDA operations for forward/backward passes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile simple conv model with default input
  python tools/profile_step.py --model simple_conv
  
  # Profile ResNet block with custom batch size
  python tools/profile_step.py --model resnet_block --batch-size 32
  
  # Profile with memory tracking and save trace
  python tools/profile_step.py --model simple_conv --profile-memory --save-trace profile.json
  
  # Profile only forward pass
  python tools/profile_step.py --model simple_conv --no-backward
  
  # Profile torchvision model (requires torchvision)
  python tools/profile_step.py --model resnet18 --input-shape 8 3 224 224
  
Available built-in models:
  - simple_conv: Simple 3-layer CNN
  - resnet_block: Single ResNet block
  - transformer_block: Single transformer encoder block
  
  Any torchvision model name (e.g., resnet18, resnet50, vgg16, etc.)
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='simple_conv',
        help='Model to profile (default: simple_conv)'
    )
    
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs='+',
        default=None,
        help='Input tensor shape (e.g., 1 3 256 256 for batch=1, channels=3, H=256, W=256)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size (default: 1)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on (default: cuda)'
    )
    
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=3,
        help='Number of warmup steps (default: 3)'
    )
    
    parser.add_argument(
        '--profile-steps',
        type=int,
        default=1,
        help='Number of steps to profile (default: 1)'
    )
    
    parser.add_argument(
        '--profile-memory',
        action='store_true',
        help='Profile memory usage'
    )
    
    parser.add_argument(
        '--no-backward',
        action='store_true',
        help='Profile forward pass only (no backward)'
    )
    
    parser.add_argument(
        '--sort-by',
        type=str,
        default='cuda_time_total',
        choices=['cuda_time_total', 'cpu_time_total', 'count', 'cpu_memory_usage', 'cuda_memory_usage'],
        help='Metric to sort results by (default: cuda_time_total)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        help='Number of top operations to show (default: 20)'
    )
    
    parser.add_argument(
        '--save-trace',
        type=str,
        default=None,
        help='Save Chrome trace to file (view at chrome://tracing)'
    )
    
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Save summary statistics to JSON file'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU")
        args.device = 'cpu'
    
    print("=" * 80)
    print("CUDA OPERATION PROFILER")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    if args.device == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print()
    
    # Create model
    try:
        print(f"Creating model: {args.model}")
        model = create_example_model(args.model)
        print(f"✓ Model created successfully")
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        sys.exit(1)
    
    # Determine input shape
    if args.input_shape:
        input_shape = tuple(args.input_shape)
    else:
        # Default shapes for different model types
        if args.model == 'transformer_block':
            input_shape = (args.batch_size, 100, 512)  # (batch, seq_len, d_model)
        elif args.model == 'resnet_block':
            input_shape = (args.batch_size, 64, 128, 128)  # (batch, channels, H, W)
        else:
            input_shape = (args.batch_size, 3, 256, 256)  # Default image input
    
    print(f"Input shape: {input_shape}")
    print()
    
    # Create input tensor
    input_tensor = torch.randn(*input_shape)
    
    # Profile
    try:
        prof, key_averages = profile_single_step(
            model=model,
            input_tensor=input_tensor,
            device=args.device,
            warmup_steps=args.warmup_steps,
            profile_steps=args.profile_steps,
            profile_memory=args.profile_memory,
            with_backward=not args.no_backward
        )
        
        # Print results
        print_profile_summary(
            key_averages,
            sort_by=args.sort_by,
            top_k=args.top_k
        )
        
        # Breakdown analysis
        breakdown = analyze_profile_breakdown(key_averages)
        print_breakdown_summary(breakdown)
        
        # Memory summary
        if args.profile_memory:
            print_memory_summary(key_averages)
        
        # Save trace if requested
        if args.save_trace:
            save_profile_chrome_trace(prof, args.save_trace)
        
        # Save JSON summary if requested
        if args.output_json:
            summary = {
                'model': args.model,
                'input_shape': list(input_shape),
                'device': args.device,
                'breakdown': breakdown,
            }
            
            with open(args.output_json, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nSummary saved to: {args.output_json}")
        
        print("\n" + "=" * 80)
        print("PROFILING COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
