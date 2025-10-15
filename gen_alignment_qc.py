import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def show_alignment(npz_path, save_to=None):
    """Show S1/S2 alignment for a tile."""
    data = np.load(npz_path)
    
    # Convert to float32 for matplotlib compatibility
    data = {k: v.astype(np.float32) for k, v in data.items()}
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f'Tile: {Path(npz_path).stem}', fontsize=14, fontweight='bold')
    
    # S2 RGB (using B4=Red, B3=Green, B2=Blue)
    s2_rgb = np.stack([data['s2_b4'], data['s2_b3'], data['s2_b2']], axis=-1)
    s2_rgb = np.clip(s2_rgb, 0, 1)
    axes[0, 0].imshow(s2_rgb)
    axes[0, 0].set_title('S2 RGB (B4-B3-B2)', fontsize=12)
    axes[0, 0].axis('off')
    
    # S2 NIR
    axes[0, 1].imshow(data['s2_b8'], cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('S2 NIR (B8)', fontsize=12)
    axes[0, 1].axis('off')
    
    # S2 False color (NIR-Red-Green)
    s2_false = np.stack([data['s2_b8'], data['s2_b4'], data['s2_b3']], axis=-1)
    s2_false = np.clip(s2_false, 0, 1)
    axes[0, 2].imshow(s2_false)
    axes[0, 2].set_title('S2 False Color (B8-B4-B3)', fontsize=12)
    axes[0, 2].axis('off')
    
    # S1 VV
    axes[1, 0].imshow(data['s1_vv'], cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('S1 VV Polarization', fontsize=12)
    axes[1, 0].axis('off')
    
    # S1 VH
    axes[1, 1].imshow(data['s1_vh'], cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('S1 VH Polarization', fontsize=12)
    axes[1, 1].axis('off')
    
    # S1 Composite
    s1_composite = np.stack([data['s1_vv'], data['s1_vh'], data['s1_vv']*0.5+data['s1_vh']*0.5], axis=-1)
    s1_composite = np.clip(s1_composite, 0, 1)
    axes[1, 2].imshow(s1_composite)
    axes[1, 2].set_title('S1 Composite (VV-VH-Combined)', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_to:
        plt.savefig(save_to, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {save_to}')
    else:
        plt.show()

# Main script
os.makedirs(r'.\outputs\qc\align', exist_ok=True)

# Get all NPZ files (not filtered by split since they're all in one directory)
all_tiles = glob.glob(r'.\data\tiles\benv2_catalog\*.npz')

if not all_tiles:
    print('ERROR: No NPZ files found in data/tiles/benv2_catalog/')
else:
    print(f'Found {len(all_tiles):,} tiles total')
    
    # Select 3 random tiles
    import random
    random.seed(42)
    selected = random.sample(all_tiles, min(3, len(all_tiles)))
    
    for i, p in enumerate(selected):
        show_alignment(p, save_to=rf'.\outputs\qc\align\benv2_{i}.png')
    
    print(f'\nSaved 3 alignment panels to outputs\\qc\\align')
