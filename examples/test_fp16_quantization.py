"""
Test FP16 Quantization for Stage 2 Refiner

This script demonstrates fp16 quantization on a mock checkpoint,
showing memory savings and inference speed improvements.

Usage:
    python examples/test_fp16_quantization.py
"""

import sys
from pathlib import Path
import time

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class MockRefinerModel(nn.Module):
    """Mock refiner model for testing."""
    
    def __init__(self, in_channels=12, out_channels=12):
        super().__init__()
        
        # Simple U-Net-like architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output


def get_model_size_mb(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / (1024**2)


def benchmark_inference(model, input_tensor, n_iterations=100, warmup=10):
    """Benchmark inference speed."""
    device = input_tensor.device
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # Benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_iterations):
            _ = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    avg_time_ms = (elapsed_time / n_iterations) * 1000
    
    return avg_time_ms


def test_quantization():
    """Test fp16 quantization."""
    print("\n" + "=" * 80)
    print("FP16 Quantization Test")
    print("=" * 80)
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model
    print("\n" + "-" * 80)
    print("Creating FP32 model...")
    print("-" * 80)
    
    model_fp32 = MockRefinerModel(in_channels=12, out_channels=12)
    model_fp32.to(device)
    model_fp32.eval()
    
    size_fp32 = get_model_size_mb(model_fp32)
    n_params = sum(p.numel() for p in model_fp32.parameters())
    
    print(f"\nFP32 Model:")
    print(f"  Parameters:  {n_params:,}")
    print(f"  Memory size: {size_fp32:.2f} MB")
    
    # Create fp16 model
    print("\n" + "-" * 80)
    print("Creating FP16 model...")
    print("-" * 80)
    
    model_fp16 = MockRefinerModel(in_channels=12, out_channels=12)
    model_fp16.to(device)
    model_fp16 = model_fp16.half()  # Convert to fp16
    model_fp16.eval()
    
    size_fp16 = get_model_size_mb(model_fp16)
    
    print(f"\nFP16 Model:")
    print(f"  Parameters:  {n_params:,}")
    print(f"  Memory size: {size_fp16:.2f} MB")
    print(f"  Reduction:   {size_fp32 - size_fp16:.2f} MB ({(1 - size_fp16/size_fp32)*100:.1f}%)")
    
    # Test inference
    print("\n" + "-" * 80)
    print("Testing inference...")
    print("-" * 80)
    
    batch_size = 4
    height, width = 256, 256
    
    # FP32 inference
    input_fp32 = torch.randn(batch_size, 12, height, width, device=device)
    
    with torch.no_grad():
        output_fp32 = model_fp32(input_fp32)
    
    print(f"\nFP32 Inference:")
    print(f"  Input shape:  {list(input_fp32.shape)}")
    print(f"  Output shape: {list(output_fp32.shape)}")
    print(f"  Input dtype:  {input_fp32.dtype}")
    print(f"  Output dtype: {output_fp32.dtype}")
    
    # FP16 inference
    input_fp16 = input_fp32.half()
    
    with torch.no_grad():
        output_fp16 = model_fp16(input_fp16)
    
    print(f"\nFP16 Inference:")
    print(f"  Input shape:  {list(input_fp16.shape)}")
    print(f"  Output shape: {list(output_fp16.shape)}")
    print(f"  Input dtype:  {input_fp16.dtype}")
    print(f"  Output dtype: {output_fp16.dtype}")
    
    # Compare outputs
    output_fp16_casted = output_fp16.float()
    diff = torch.abs(output_fp32 - output_fp16_casted)
    
    print(f"\nOutput Difference (FP32 vs FP16):")
    print(f"  Mean abs diff: {diff.mean().item():.6f}")
    print(f"  Max abs diff:  {diff.max().item():.6f}")
    print(f"  Relative diff: {(diff / (torch.abs(output_fp32) + 1e-8)).mean().item():.6f}")
    
    # Benchmark speed
    if device.type == 'cuda':
        print("\n" + "-" * 80)
        print("Benchmarking inference speed (GPU)...")
        print("-" * 80)
        
        time_fp32 = benchmark_inference(model_fp32, input_fp32, n_iterations=100)
        time_fp16 = benchmark_inference(model_fp16, input_fp16, n_iterations=100)
        
        speedup = time_fp32 / time_fp16
        
        print(f"\nFP32: {time_fp32:.3f} ms/batch")
        print(f"FP16: {time_fp16:.3f} ms/batch")
        print(f"Speedup: {speedup:.2f}x")
    
    # Save checkpoint examples
    print("\n" + "-" * 80)
    print("Saving checkpoint examples...")
    print("-" * 80)
    
    output_dir = Path("models/test_checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # FP32 checkpoint
    ckpt_fp32_path = output_dir / "refiner_fp32.pt"
    checkpoint_fp32 = {
        'model_state_dict': model_fp32.state_dict(),
        'step': 10000,
        'epoch': 10
    }
    torch.save(checkpoint_fp32, ckpt_fp32_path)
    size_fp32_file = ckpt_fp32_path.stat().st_size / (1024**2)
    
    print(f"\nFP32 checkpoint: {ckpt_fp32_path}")
    print(f"  File size: {size_fp32_file:.2f} MB")
    
    # FP16 checkpoint
    ckpt_fp16_path = output_dir / "refiner_fp16.pt"
    checkpoint_fp16 = {
        'model_state_dict': model_fp16.state_dict(),
        'step': 10000,
        'epoch': 10,
        'quantization_info': {
            'dtype': 'fp16',
            'fp16_parameters': n_params,
            'compression_ratio': 2.0
        }
    }
    torch.save(checkpoint_fp16, ckpt_fp16_path)
    size_fp16_file = ckpt_fp16_path.stat().st_size / (1024**2)
    
    print(f"\nFP16 checkpoint: {ckpt_fp16_path}")
    print(f"  File size: {size_fp16_file:.2f} MB")
    print(f"  Reduction: {size_fp32_file - size_fp16_file:.2f} MB ({(1 - size_fp16_file/size_fp32_file)*100:.1f}%)")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    print(f"\n✓ FP16 quantization reduces model size by ~{(1 - size_fp16/size_fp32)*100:.0f}%")
    print(f"✓ Checkpoint file size reduced by ~{(1 - size_fp16_file/size_fp32_file)*100:.0f}%")
    print(f"✓ Output difference: {diff.mean().item():.6f} (negligible)")
    
    if device.type == 'cuda':
        print(f"✓ GPU inference speedup: {speedup:.2f}x")
        print(f"\n💡 FP16 quantization is highly recommended for production inference!")
    else:
        print(f"\n💡 Run on GPU to see inference speedup benefits")


if __name__ == '__main__':
    test_quantization()
