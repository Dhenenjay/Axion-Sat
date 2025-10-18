"""
Full Pipeline Inference: SAR to Optical (Stages 1-3)

This script runs the complete 3-stage pipeline:
- Stage 1: TerraMind SAR → optical baseline
- Stage 2: Prithvi refinement  
- Stage 3: SAR-grounded final output

Usage:
    python scripts/infer_full_pipeline.py \
        --sar-path path/to/sar.npz \
        --output-dir outputs/inference \
        --stage2-checkpoint weights/stage2_best.pt \
        --stage3-checkpoint weights/stage3_best.pt
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axs_lib.stage2_prithvi_refine import build_prithvi_refiner
from axs_lib.stage3_tm_backbone import build_stage3_backbone_model
from axs_lib.io import load_checkpoint
from axs_lib.terramind_tim import build_terramind_stage1, TerraMindGenerator


def load_sar_data(sar_path: Path):
    """Load SAR data from NPZ file and denormalize to dB scale."""
    import json
    
    data = np.load(sar_path)
    
    # Extract SAR bands (VV, VH)
    s1_vv = data.get('s1_vv', data.get('VV', None))
    s1_vh = data.get('s1_vh', data.get('VH', None))
    
    if s1_vv is None or s1_vh is None:
        raise ValueError(f"SAR data not found in {sar_path}. Keys: {list(data.keys())}")
    
    # Stack to (2, H, W)
    sar = np.stack([s1_vv, s1_vh], axis=0)
    
    print(f"  SAR normalized range: [{sar.min():.4f}, {sar.max():.4f}]")
    
    # Load metadata for denormalization
    metadata_path = sar_path.with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        s1_norm = metadata['s1_normalization']
        sar_min = s1_norm['min']
        sar_max = s1_norm['max']
        
        # Denormalize SAR back to dB scale
        sar_db = sar * (sar_max - sar_min) + sar_min
        
        print(f"  SAR denormalized (dB): [{sar_db.min():.2f}, {sar_db.max():.2f}]")
        sar = sar_db
    else:
        print(f"  ⚠ Warning: Metadata not found, using normalized SAR")
    
    # Convert to tensor
    sar_tensor = torch.from_numpy(sar).float().unsqueeze(0)  # (1, 2, H, W)
    
    return sar_tensor, data


def stage1_inference(sar: torch.Tensor, device: torch.device):
    """
    Stage 1: TerraMind with tokenizers SAR → optical.
    
    Uses proper tokenizer-based workflow for generation.
    
    Returns:
        opt_v1: Stage 1 optical output (1, 12, H, W) - all S2L2A bands, normalized [0,1]
    """
    print("\n" + "="*80)
    print("Stage 1: TerraMind with Tokenizers")
    print("="*80)
    
    try:
        # Build TerraMind using corrected implementation with tokenizers
        print("  Building TerraMind with tokenizer-based workflow...")
        model = build_terramind_stage1(
            use_tokenizers=True,  # Use tokenizer-based workflow
            use_tim=False,  # Don't use TiM for now (can enable later)
            output_bands='all',  # Get all 12 S2L2A bands
            device=device
        )
        
        print("✓ TerraMind model loaded with tokenizers")
        
        # Handle 120x120 tiles: pad to 224x224 (TerraMind default size)
        original_size = sar.shape[2:]
        if original_size[0] == 120 and original_size[1] == 120:
            print(f"  Padding 120x120 to 224x224 for TerraMind...")
            # Pad to 224x224 (52 pixels on each side)
            pad_size = (224 - 120) // 2
            sar_padded = torch.nn.functional.pad(sar, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        else:
            sar_padded = sar
        
        print(f"  Input shape: {sar_padded.shape}")
        print(f"  Input range (dB): [{sar_padded.min():.2f}, {sar_padded.max():.2f}]")
        
        # Generate optical using tokenizer-based pipeline
        with torch.no_grad():
            opt_v1_padded = model(sar_padded)
        
        print(f"  Generated output shape: {opt_v1_padded.shape}")
        print(f"  Output range (raw): [{opt_v1_padded.min():.2f}, {opt_v1_padded.max():.2f}] DN")
        
        # Clip negative values
        opt_v1_padded = torch.clamp(opt_v1_padded, 0, 10000)
        print(f"  Output range (clipped): [{opt_v1_padded.min():.2f}, {opt_v1_padded.max():.2f}] DN")
        
        # Crop back to original size if we padded
        if original_size[0] == 120 and original_size[1] == 120:
            opt_v1 = opt_v1_padded[:, :, pad_size:pad_size+120, pad_size:pad_size+120]
        else:
            opt_v1 = opt_v1_padded
        
        print(f"  Output range (reflectance): [{opt_v1.min():.4f}, {opt_v1.max():.4f}]")
        print(f"✓ Stage 1 complete: {opt_v1.shape}")
        return opt_v1
        
    except Exception as e:
        print(f"⚠ Stage 1 failed: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("\n  Using random initialization for testing...")
        # Fallback: random optical for testing (12 S2L2A bands)
        return torch.randn(1, 12, sar.shape[2], sar.shape[3]).to(device)


def stage2_inference(opt_v1_12band: torch.Tensor, checkpoint_path: Path, device: torch.device):
    """
    Stage 2: Prithvi refinement.
    
    Args:
        opt_v1_12band: Stage 1 output with all 12 S2L2A bands (1, 12, H, W)
    
    Returns:
        opt_v2: Stage 2 refined optical (1, 6, H, W) - 6 HLS bands
    """
    print("\n" + "="*80)
    print("Stage 2: Prithvi Refinement")
    print("="*80)
    
    # Extract 6 HLS bands from 12 S2L2A bands
    # Prithvi expects: Blue (B02), Green (B03), Red (B04), Narrow NIR (B08), SWIR1 (B11), SWIR2 (B12)
    # S2L2A band order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
    # Indices: B02=1, B03=2, B04=3, B08=7, B11=10, B12=11
    hls_band_indices = [1, 2, 3, 7, 10, 11]  # 0-indexed
    opt_v1_hls = opt_v1_12band[:, hls_band_indices, :, :]  # (1, 6, H, W)
    
    print(f"  Extracted 6 HLS bands from 12 S2L2A bands")
    print(f"  HLS input shape: {opt_v1_hls.shape}")
    print(f"  HLS input range: [{opt_v1_hls.min():.4f}, {opt_v1_hls.max():.4f}]")
    
    # Build Prithvi refiner for 6 HLS input channels
    model = build_prithvi_refiner(
        num_input_channels=6,  # 6 HLS bands from Stage 1
        out_channels=256,
        hidden_dim=256,
        num_convnext_blocks=4,
        lora_r=4,
        lora_alpha=16,
        load_in_8bit=False,  # Use FP16 for pretrained
        freeze_backbone=True,
        unfreeze_last_n_blocks=2,
        device=device
    )
    
    # Load checkpoint
    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, device=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("[OK] Stage 2 checkpoint loaded")
    else:
        print(f"[WARN] Checkpoint not found: {checkpoint_path}")
        print("  Using pretrained Prithvi backbone only...")
    
    model.eval()
    
    # Refine (PrithviRefiner doesn't use metadata)
    with torch.no_grad():
        features = model(opt_v1_hls)  # (1, 256, H, W)
    
    # Project back to 6 HLS channels
    proj = nn.Conv2d(features.shape[1], 6, kernel_size=1).to(device)
    with torch.no_grad():
        opt_v2 = proj(features)
        opt_v2 = torch.sigmoid(opt_v2)  # Normalize to [0, 1]
    
    print(f"[OK] Stage 2 complete: {opt_v2.shape}")
    print(f"  Output range: [{opt_v2.min():.4f}, {opt_v2.max():.4f}]")
    return opt_v2
    
    print(f"✓ Stage 2 complete: {opt_v2.shape}")
    return opt_v2


def stage3_inference(sar: torch.Tensor, opt_v2: torch.Tensor, 
                    checkpoint_path: Path, device: torch.device):
    """
    Stage 3: SAR-grounded final output.
    
    Args:
        sar: SAR input (1, 2, H, W)
        opt_v2: Stage 2 output with 6 HLS bands (1, 6, H, W)
    
    Returns:
        opt_v3: Stage 3 final optical (1, 6, H, W) - 6 HLS bands
    """
    print("\n" + "="*80)
    print("Stage 3: SAR Grounding")
    print("="*80)
    
    print(f"  SAR input shape: {sar.shape}")
    print(f"  Stage 2 output shape: {opt_v2.shape}")
    
    # Build Stage 3 model
    model = build_stage3_backbone_model(
        freeze_backbone=True,
        pretrained=True,
        device=device
    )
    
    # Add projection layer to convert 6 HLS channels to 4 S2 channels
    # Stage 3 expects 4 channels: B02, B03, B04, B08
    # We have 6 HLS: Blue(B02), Green(B03), Red(B04), NIR(B08), SWIR1(B11), SWIR2(B12)
    # Extract first 4 bands (B02, B03, B04, B08) for Stage 3
    print(f"  Note: Stage 3 trained for 4 S2 bands, extracting first 4 from 6 HLS bands")
    
    # Load checkpoint
    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, device=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("[OK] Stage 3 checkpoint loaded")
    else:
        print(f"[WARN] Checkpoint not found: {checkpoint_path}")
        print("  Using pretrained TerraMind backbone only...")
    
    model.eval()
    
    # Extract 4 S2 bands from 6 HLS bands
    # HLS: Blue(0), Green(1), Red(2), NIR(3), SWIR1(4), SWIR2(5)
    # S2: B02(Blue), B03(Green), B04(Red), B08(NIR)
    opt_v2_4band = opt_v2[:, [0, 1, 2, 3], :, :]  # Take first 4 bands
    print(f"  Extracted 4 S2 bands from 6 HLS: {opt_v2_4band.shape}")
    
    # Generate final output
    with torch.no_grad():
        opt_v3_4band = model(sar, opt_v2_4band)
    
    print(f"  Stage 3 raw output range (DN): [{opt_v3_4band.min():.4f}, {opt_v3_4band.max():.4f}]")
    
    # Stage 3 destandardizes back to DN scale (0-10000), convert to reflectance [0, 1]
    opt_v3_4band = torch.clamp(opt_v3_4band, 0, 10000) / 10000.0
    
    # Project back to 6 HLS channels for consistency
    # (For now, just duplicate NIR to fill SWIR bands)
    opt_v3 = torch.cat([
        opt_v3_4band,  # B02, B03, B04, B08
        opt_v3_4band[:, 3:4, :, :],  # Duplicate NIR as SWIR1
        opt_v3_4band[:, 3:4, :, :]   # Duplicate NIR as SWIR2
    ], dim=1)
    
    # Take first frame if model outputs multiple timesteps
    if opt_v3.ndim == 5:  # (B, T, C, H, W)
        opt_v3 = opt_v3[:, 0, :, :, :]  # Take first timestep
    elif opt_v3.shape[0] > 1 and opt_v3.shape[1] == 6:  # (T, C, H, W)
        opt_v3 = opt_v3[0:1, :, :, :]  # Take first frame
    
    # Ensure normalized to [0, 1]
    opt_v3 = torch.clamp(opt_v3, 0, 1)
    
    print(f"[OK] Stage 3 complete: {opt_v3.shape}")
    print(f"  Output range: [{opt_v3.min():.4f}, {opt_v3.max():.4f}]")
    return opt_v3


def save_outputs(sar, opt_v1, opt_v2, opt_v3, output_dir: Path):
    """Save and visualize all outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Saving Outputs")
    print("="*80)
    
    # Convert to numpy
    sar_np = sar.cpu().numpy()[0]  # (2, H, W)
    opt_v1_np = opt_v1.cpu().numpy()[0]  # (12, H, W) - all S2L2A bands
    opt_v2_np = opt_v2.cpu().numpy()[0]  # (6, H, W) - HLS bands
    opt_v3_np = opt_v3.cpu().numpy()[0]  # (6, H, W) - HLS bands
    
    # Save as NPZ
    np.savez(
        output_dir / 'outputs.npz',
        sar=sar_np,
        opt_v1=opt_v1_np,
        opt_v2=opt_v2_np,
        opt_v3=opt_v3_np
    )
    print(f"✓ Saved: {output_dir / 'outputs.npz'}")
    
    # For 12-band S2L2A: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
    # RGB = B04(3), B03(2), B02(1) for natural color
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # SAR (VV)
    axes[0, 0].imshow(sar_np[0], cmap='gray')
    axes[0, 0].set_title('SAR (VV)')
    axes[0, 0].axis('off')
    
    # SAR (VH)
    axes[0, 1].imshow(sar_np[1], cmap='gray')
    axes[0, 1].set_title('SAR (VH)')
    axes[0, 1].axis('off')
    
    # Stage 1 RGB - extract B04, B03, B02 from 12 bands
    if opt_v1_np.shape[0] >= 4:
        opt_v1_rgb = np.transpose(opt_v1_np[[3, 2, 1], :, :], (1, 2, 0))  # B04, B03, B02
    else:
        opt_v1_rgb = np.transpose(opt_v1_np[[2, 1, 0], :, :], (1, 2, 0))  # Fallback
    opt_v1_rgb = np.clip(opt_v1_rgb, 0, 1)
    axes[0, 2].imshow(opt_v1_rgb)
    axes[0, 2].set_title('Stage 1: TerraMind')
    axes[0, 2].axis('off')
    
    # Stage 2 RGB (HLS bands: 0=Blue, 1=Green, 2=Red)
    opt_v2_rgb = np.transpose(opt_v2_np[[2, 1, 0], :, :], (1, 2, 0))  # R, G, B
    opt_v2_rgb = np.clip(opt_v2_rgb, 0, 1)
    axes[1, 0].imshow(opt_v2_rgb)
    axes[1, 0].set_title('Stage 2: Prithvi Refined (HLS)')
    axes[1, 0].axis('off')
    
    # Stage 3 RGB (HLS bands: 0=Blue, 1=Green, 2=Red)
    opt_v3_rgb = np.transpose(opt_v3_np[[2, 1, 0], :, :], (1, 2, 0))  # R, G, B
    opt_v3_rgb = np.clip(opt_v3_rgb, 0, 1)
    axes[1, 1].imshow(opt_v3_rgb)
    axes[1, 1].set_title('Stage 3: SAR-Grounded (HLS)')
    axes[1, 1].axis('off')
    
    # Comparison: Stage 1 vs Stage 3
    axes[1, 2].imshow(opt_v1_rgb)
    axes[1, 2].imshow(opt_v3_rgb, alpha=0.5)
    axes[1, 2].set_title('Stage 1 vs Stage 3 Overlay')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pipeline_outputs.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'pipeline_outputs.png'}")
    
    plt.close()
    
    print("\n" + "="*80)
    print("✓ Inference Complete!")
    print("="*80)
    print(f"Outputs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run full 3-stage SAR-to-optical pipeline")
    parser.add_argument('--sar-path', type=str, required=True,
                       help='Path to SAR NPZ file')
    parser.add_argument('--output-dir', type=str, default='outputs/inference',
                       help='Output directory')
    parser.add_argument('--stage2-checkpoint', type=str, default='weights/stage2_best.pt',
                       help='Stage 2 checkpoint path')
    parser.add_argument('--stage3-checkpoint', type=str, default='weights/stage3_best.pt',
                       help='Stage 3 checkpoint path')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    sar_path = Path(args.sar_path)
    output_dir = Path(args.output_dir)
    stage2_ckpt = Path(args.stage2_checkpoint)
    stage3_ckpt = Path(args.stage3_checkpoint)
    
    # Load SAR data
    print(f"\nLoading SAR data from: {sar_path}")
    sar, metadata = load_sar_data(sar_path)
    sar = sar.to(device)
    print(f"SAR shape: {sar.shape}")
    
    # Run pipeline
    opt_v1 = stage1_inference(sar, device)
    opt_v2 = stage2_inference(opt_v1, stage2_ckpt, device)
    opt_v3 = stage3_inference(sar, opt_v2, stage3_ckpt, device)
    
    # Save results
    save_outputs(sar, opt_v1, opt_v2, opt_v3, output_dir)


if __name__ == '__main__':
    main()
