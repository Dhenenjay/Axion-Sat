"""
Debug GAC Pipeline - Inspect each stage's output to find issues.

This script runs the pipeline and prints detailed diagnostics at each stage.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_tensor_stats(name, tensor):
    """Print detailed statistics about a tensor."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Range: [{tensor.min():.6f}, {tensor.max():.6f}]")
    print(f"  Mean: {tensor.mean():.6f}")
    print(f"  Std: {tensor.std():.6f}")
    print(f"  Non-zero: {np.count_nonzero(tensor)} / {tensor.size} ({100*np.count_nonzero(tensor)/tensor.size:.1f}%)")
    
    # Check for constant values
    unique = np.unique(tensor)
    if len(unique) < 10:
        print(f"  Unique values ({len(unique)}): {unique[:10]}")
    else:
        print(f"  Unique values: {len(unique)} (showing 10): {unique[:10]}")


def visualize_stage(name, data, output_dir):
    """Visualize a stage's output."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    # Convert to float32 if needed
    if data.dtype == np.float16:
        data = data.astype(np.float32)
    
    # If 4D (B, C, H, W), take first batch
    if data.ndim == 4:
        data = data[0]
    
    # If 3D (C, H, W), create RGB composite
    if data.ndim == 3 and data.shape[0] >= 3:
        rgb = np.transpose(data[[2, 1, 0], :, :], (1, 2, 0))  # R, G, B
        rgb = np.clip(rgb, 0, 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # RGB composite
        axes[0].imshow(rgb)
        axes[0].set_title(f'{name} - RGB Composite')
        axes[0].axis('off')
        
        # First channel
        im = axes[1].imshow(data[0], cmap='viridis')
        axes[1].set_title(f'{name} - Channel 0')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{name.lower().replace(" ", "_")}.png', dpi=150)
        plt.close()


def main():
    print("=" * 80)
    print("GAC Pipeline Debugger")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load test tile
    tile_path = Path("data/tiles/benv2_catalog/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57.npz")
    print(f"\nLoading tile: {tile_path.name}")
    
    data = np.load(tile_path)
    
    # Load ground truth
    s2_gt = np.stack([
        data['s2_b2'],
        data['s2_b3'],
        data['s2_b4'],
        data['s2_b8']
    ], axis=0)
    
    print_tensor_stats("Ground Truth S2", s2_gt)
    
    # Load SAR
    s1 = np.stack([data['s1_vv'], data['s1_vh']], axis=0)
    print_tensor_stats("Input SAR", s1)
    
    # Convert to tensor
    s1_tensor = torch.from_numpy(s1).float().unsqueeze(0).to(device)
    
    # Create output directory
    output_dir = Path("outputs/pipeline_debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize ground truth
    visualize_stage("Ground Truth", s2_gt, output_dir)
    
    print("\n" + "=" * 80)
    print("STAGE 1: TerraMind")
    print("=" * 80)
    
    try:
        from axs_lib.models import build_terramind_generator
        from axs_lib.stage1_tm_s2o import tm_sar2opt
        
        # Pad SAR to 128x128
        s1_padded = torch.nn.functional.pad(s1_tensor, (4, 4, 4, 4), mode='reflect')
        print(f"Padded SAR shape: {s1_padded.shape}")
        
        # Build generator
        print("\nBuilding TerraMind generator...")
        generator = build_terramind_generator(
            input_modalities=("S1GRD",),
            output_modalities=("S2L2A",),
            timesteps=12,
            standardize=True,
            pretrained=True
        )
        generator = generator.to(device)
        generator.eval()
        print("[OK] Generator loaded")
        
        # Standardize
        s1_mean = torch.tensor([-11.76, -18.01]).view(1, 2, 1, 1).to(device)
        s1_std = torch.tensor([4.62, 5.18]).view(1, 2, 1, 1).to(device)
        s1_std_padded = (s1_padded - s1_mean) / s1_std
        
        print_tensor_stats("Standardized SAR", s1_std_padded)
        
        # Generate
        print("\nGenerating optical with TerraMind...")
        with torch.no_grad():
            opt_v1_padded = tm_sar2opt(
                generator, 
                s1_std_padded, 
                timesteps=12,
                device=device
            )
        
        # Crop back
        opt_v1 = opt_v1_padded[:, :, 4:124, 4:124]
        
        print_tensor_stats("Stage 1 Output", opt_v1)
        visualize_stage("Stage 1", opt_v1, output_dir)
        
        # Check if all same value
        if opt_v1.std() < 0.001:
            print("\n[WARNING] Stage 1 output is nearly constant!")
            print("   This will corrupt downstream stages.")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 1 failed: {e}")
        import traceback
        traceback.print_exc()
        opt_v1 = torch.randn(1, 4, 120, 120).to(device)
        print("Using random fallback")
    
    print("\n" + "=" * 80)
    print("STAGE 2: Prithvi Refinement")
    print("=" * 80)
    
    try:
        from axs_lib.stage2_prithvi_refine import build_prithvi_refiner
        from axs_lib.io import load_checkpoint
        
        # Build model
        print("\nBuilding Prithvi refiner...")
        stage2_model = build_prithvi_refiner(
            config={'model': {'lora_rank': 4, 'lora_alpha': 16}},
            device=device
        )
        
        # Load checkpoint
        ckpt_path = Path("weights/stage2_best.pt")
        if ckpt_path.exists():
            print(f"Loading checkpoint: {ckpt_path}")
            checkpoint = load_checkpoint(ckpt_path, device=device)
            stage2_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("[OK] Checkpoint loaded")
        else:
            print("[WARNING] No checkpoint found, using untrained model")
        
        stage2_model.eval()
        
        # Run inference
        print("\nRunning Stage 2 inference...")
        with torch.no_grad():
            features = stage2_model(opt_v1)
        
        print_tensor_stats("Stage 2 Features", features)
        
        # Project to 4 channels
        proj = torch.nn.Conv2d(features.shape[1], 4, kernel_size=1).to(device)
        with torch.no_grad():
            opt_v2 = proj(features)
            opt_v2 = torch.sigmoid(opt_v2)
        
        print_tensor_stats("Stage 2 Output", opt_v2)
        visualize_stage("Stage 2", opt_v2, output_dir)
        
        # Check output
        if opt_v2.std() < 0.01:
            print("\n[WARNING] Stage 2 output has very low variance!")
            print("   Model may be producing uniform colors.")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 2 failed: {e}")
        import traceback
        traceback.print_exc()
        opt_v2 = opt_v1.clone()
    
    print("\n" + "=" * 80)
    print("STAGE 3: SAR Grounding")
    print("=" * 80)
    
    try:
        from axs_lib.stage3_tm_backbone import build_stage3_backbone_model
        from axs_lib.io import load_checkpoint
        
        # Build model
        print("\nBuilding Stage 3 model...")
        stage3_model = build_stage3_backbone_model(
            freeze_backbone=True,
            pretrained=True,
            device=device
        )
        
        # Load checkpoint
        ckpt_path = Path("weights/stage3_best.pt")
        if ckpt_path.exists():
            print(f"Loading checkpoint: {ckpt_path}")
            checkpoint = load_checkpoint(ckpt_path, device=device)
            stage3_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("[OK] Checkpoint loaded")
        else:
            print("[WARNING] No checkpoint found, using untrained model")
        
        stage3_model.eval()
        
        # Run inference
        print("\nRunning Stage 3 inference...")
        with torch.no_grad():
            opt_v3 = stage3_model(s1_tensor, opt_v2)
        
        print_tensor_stats("Stage 3 Output (raw)", opt_v3)
        
        # Check shape
        if opt_v3.shape[0] != 1:
            print(f"\n[WARNING] Stage 3 output has batch size {opt_v3.shape[0]}, expected 1")
            print("   Taking first element...")
            opt_v3 = opt_v3[0:1]
        
        # Clip to [0, 1]
        opt_v3 = torch.clamp(opt_v3, 0, 1)
        
        print_tensor_stats("Stage 3 Output (clipped)", opt_v3)
        visualize_stage("Stage 3", opt_v3, output_dir)
        
        # Check sparsity
        non_zero_ratio = (opt_v3 > 0.01).sum().item() / opt_v3.numel()
        print(f"\nNon-background pixels: {non_zero_ratio*100:.1f}%")
        
        if non_zero_ratio < 0.5:
            print("\n[WARNING] Stage 3 output is very sparse!")
            print("   Model is only predicting a small fraction of pixels.")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 3 failed: {e}")
        import traceback
        traceback.print_exc()
        opt_v3 = opt_v2.clone()
    
    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    
    opt_v3_np = opt_v3.detach().cpu().numpy()[0]
    
    # Compute simple metrics
    mae = np.mean(np.abs(s2_gt - opt_v3_np))
    mse = np.mean((s2_gt - opt_v3_np) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    print(f"\nFinal Output vs Ground Truth:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  PSNR: {psnr:.2f} dB")
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    gt_rgb = np.transpose(s2_gt[[2, 1, 0], :, :], (1, 2, 0))
    gt_rgb = np.clip(gt_rgb, 0, 1)
    axes[0].imshow(gt_rgb)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    pred_rgb = np.transpose(opt_v3_np[[2, 1, 0], :, :], (1, 2, 0))
    pred_rgb = np.clip(pred_rgb, 0, 1)
    axes[1].imshow(pred_rgb)
    axes[1].set_title(f'Stage 3 Output (PSNR: {psnr:.1f} dB)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_comparison.png', dpi=150)
    plt.close()
    
    print(f"\n[OK] Debug visualizations saved to: {output_dir}")
    print("\n" + "=" * 80)
    print("Debug Complete!")
    print("=" * 80)
    print("\nCheck the visualizations to see what each stage is producing.")


if __name__ == '__main__':
    main()
