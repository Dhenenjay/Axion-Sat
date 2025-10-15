from axs_lib.data import load_tile, load_tiles_batch, list_tiles
from pathlib import Path

# Load single tile
print("Testing single tile load...")
data, meta = load_tile('output/demo_tiles/demo_r00000_c00000.npz')
print(f"✓ Loaded tile: {meta.tile_id}")
print(f"  Keys: {list(data.keys())}")
print(f"  Shape: {data['s2_b4'].shape}")
print(f"  Dtype: {data['s2_b4'].dtype}")
print(f"  CRS: {meta.crs}")
print()

# List all tiles
print("Listing tiles...")
tiles = list_tiles('output/demo_tiles/')
print(f"✓ Found {len(tiles)} tiles")
print()

# Load batch
print("Testing batch load...")
batch = load_tiles_batch(tiles[:4], keys=['s2_b4', 's1_vv'], stack=True)
print(f"✓ Loaded batch")
print(f"  s2_b4 shape: {batch['s2_b4'].shape}")
print(f"  s1_vv shape: {batch['s1_vv'].shape}")
print()

print("All tests passed! ✓")
