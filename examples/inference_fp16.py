"""
Inference Example with FP16 Quantized Checkpoint

This example demonstrates how to load and use an fp16-quantized Stage 2 refiner
checkpoint for efficient inference.

Usage:
    python examples/inference_fp16.py --checkpoint models/stage2_best/stage2_best.pt
"""

import sys
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class MockRefinerModel(nn.Module):
    """Mock refiner model (replace with actual Stage 2 refiner)."""
    
    def __init__(self, in_channels=12, out_channels=12):
        super().__init__()
        
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


def load_checkpoint_auto(checkpoint_path, device='cuda'):
    """
    Load checkpoint with automatic FP16 detection and configuration.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        model: Loaded model (fp16 or fp32)
        quantization_info: Quantization metadata (if available)
        dtype: Model dtype (torch.float16 or torch.float32)
    """
    print(f"\n{'='*80}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"{'='*80}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\nCheckpoint info:")
    print(f"  Training step: {checkpoint.get('step', 'N/A')}")
    print(f"  Training epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # Check for quantization
    quantization_info = checkpoint.get('quantization_info')
    is_quantized = quantization_info is not None
    
    if is_quantized:
        print(f"\n✓ FP16 quantized checkpoint detected")
        print(f"  Quantized parameters: {quantization_info['fp16_parameters']:,}")
        print(f"  Other parameters: {quantization_info['non_fp16_parameters']:,}")
        print(f"  Compression ratio: {quantization_info['compression_ratio']:.1f}x")
        dtype = torch.float16
    else:
        print(f"\n⚠ FP32 checkpoint (not quantized)")
        dtype = torch.float32
    
    # Create model
    print(f"\nLoading model...")
    model = MockRefinerModel(in_channels=12, out_channels=12)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Convert to appropriate dtype
    if is_quantized:
        model = model.half()
        print(f"  Converted model to fp16")
    
    # Move to device
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Warning: fp16 on CPU is very slow
    if is_quantized and device_obj.type == 'cpu':
        print(f"\n⚠ WARNING: FP16 on CPU is extremely slow!")
        print(f"  Converting model back to FP32 for CPU inference...")
        model = model.float()  # Convert back to fp32 for CPU
        dtype = torch.float32
    
    model = model.to(device_obj)
    model.eval()
    
    print(f"  Model moved to: {device_obj}")
    
    # Calculate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024**2)
    
    print(f"  Model size: {size_mb:.2f} MB")
    print(f"  Model dtype: {dtype}")
    
    return model, quantization_info, dtype


def run_inference(model, batch_size=4, height=256, width=256, device='cuda'):
    """
    Run inference with the model.
    
    Args:
        model: Loaded model
        batch_size: Batch size for inference
        height: Input height
        width: Input width
        device: Device for inference
    """
    print(f"\n{'='*80}")
    print(f"Running inference")
    print(f"{'='*80}")
    
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Determine input dtype from model
    model_dtype = next(model.parameters()).dtype
    
    print(f"\nInput configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Spatial size: {height}x{width}")
    print(f"  Channels: 12 (Stage 1 output)")
    print(f"  Device: {device_obj}")
    print(f"  Dtype: {model_dtype}")
    
    # Create dummy input (Stage 1 output)
    input_tensor = torch.randn(batch_size, 12, height, width)
    input_tensor = input_tensor.to(dtype=model_dtype, device=device_obj)
    
    print(f"\nRunning forward pass...")
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"\n✓ Inference complete")
    print(f"  Input shape: {list(input_tensor.shape)}")
    print(f"  Output shape: {list(output.shape)}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Run inference with FP16 quantized checkpoint'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('models/stage2_best_fp16/stage2_best.pt'),
        help='Path to checkpoint file'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for inference'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=256,
        help='Spatial size (height and width)'
    )
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not args.checkpoint.exists():
        print(f"\n❌ Checkpoint not found: {args.checkpoint}")
        print(f"\nTo create an fp16 quantized checkpoint:")
        print(f"  python scripts/export_stage2_best.py \\")
        print(f"      --checkpoint_dir outputs/stage2_training/checkpoints \\")
        print(f"      --validation_metrics outputs/stage2_training/validation_metrics.csv \\")
        print(f"      --output_dir models/stage2_best \\")
        print(f"      --quantize_fp16")
        return 1
    
    # Load checkpoint
    model, quantization_info, dtype = load_checkpoint_auto(
        args.checkpoint,
        device=args.device
    )
    
    # Run inference
    output = run_inference(
        model,
        batch_size=args.batch_size,
        height=args.size,
        width=args.size,
        device=args.device
    )
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")
    
    if quantization_info:
        print(f"\n✓ Using FP16 quantized model")
        print(f"  - 50% memory reduction vs FP32")
        print(f"  - Faster inference on modern GPUs")
        print(f"  - Negligible accuracy impact")
    else:
        print(f"\n⚠ Using FP32 model")
        print(f"  Consider exporting with --quantize_fp16 for production")
    
    print(f"\n💡 Inference pipeline ready for production!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
