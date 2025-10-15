#!/usr/bin/env python3
"""
Reorganize BigEarthNet v2 directory structure to add parent level.

The ingestion script expects:
  root/level1/parent/child/*.tif

But extraction created:
  root/level1/child/*.tif

This script moves child directories under their appropriate parent directories.
"""

import re
import shutil
from pathlib import Path
from collections import defaultdict


def extract_parent_from_s2(child_name: str) -> str:
    """Extract parent directory name from S2 child name.
    
    Child: S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_27_55
    Parent: S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP
    """
    match = re.match(r'(S2[AB]_MSIL2A_\d{8}T\d{6}_N\d{4}_R\d{3}_T\w+)_\d+_\d+', child_name)
    if match:
        return match.group(1)
    return None


def extract_parent_from_s1(child_name: str) -> str:
    """Extract parent directory name from S1 child name.
    
    Child: S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39
    Parent: S1A_IW_GRDH_1SDV_20170613T165043
    """
    match = re.match(r'(S1[AB]_IW_GRDH_\d[A-Z]{3}_\d{8}T\d{6})_\w+_\d+_\d+', child_name)
    if match:
        return match.group(1)
    return None


def reorganize_dataset(root: Path, sensor: str):
    """Reorganize dataset to add parent level."""
    print(f"\n[*] Reorganizing {sensor} dataset at {root}")
    
    if not root.exists():
        print(f"   [SKIP] Directory does not exist")
        return
    
    # Find all level1 directories
    level1_dirs = [d for d in root.iterdir() if d.is_dir()]
    
    if not level1_dirs:
        print(f"   [SKIP] No subdirectories found")
        return
    
    extract_parent = extract_parent_from_s2 if sensor == 'S2' else extract_parent_from_s1
    
    total_moved = 0
    
    for level1 in level1_dirs:
        print(f"\n   Processing {level1.name}...")
        
        # Find all potential child directories
        children = [d for d in level1.iterdir() if d.is_dir()]
        
        # Group children by parent
        parent_groups = defaultdict(list)
        for child in children:
            parent_name = extract_parent(child.name)
            if parent_name:
                parent_groups[parent_name].append(child)
            else:
                print(f"      [WARN] Could not extract parent from: {child.name}")
        
        print(f"      Found {len(children)} children -> {len(parent_groups)} parent groups")
        
        # Create parent directories and move children
        for parent_name, child_list in parent_groups.items():
            parent_dir = level1 / parent_name
            
            # Create parent directory if it doesn't exist
            if not parent_dir.exists():
                parent_dir.mkdir(parents=True)
                print(f"      Created parent: {parent_name}")
            
            # Move children into parent
            for child in child_list:
                try:
                    new_path = parent_dir / child.name
                    if not new_path.exists():
                        shutil.move(str(child), str(new_path))
                        total_moved += 1
                    else:
                        print(f"      [WARN] Target already exists: {new_path}")
                except Exception as e:
                    print(f"      [ERR] Failed to move {child.name}: {e}")
    
    print(f"\n   [OK] Moved {total_moved} child directories under parents")


def main():
    """Main reorganization routine."""
    print("="*60)
    print("BigEarthNet v2 Directory Structure Reorganizer")
    print("="*60)
    
    # Get data root from user
    data_root = Path(input("\nEnter path to data/raw directory (e.g., data/raw or /path/to/data/raw): ").strip())
    
    if not data_root.exists():
        print(f"[ERR] Directory not found: {data_root}")
        return
    
    s2_root = data_root / "BigEarthNet-S2"
    s1_root = data_root / "BigEarthNet-S1"
    
    # Reorganize S2
    reorganize_dataset(s2_root, 'S2')
    
    # Reorganize S1
    reorganize_dataset(s1_root, 'S1')
    
    print("\n" + "="*60)
    print("[COMPLETE] Reorganization finished!")
    print("="*60)
    print("\nYou can now run the ingestion script:")
    print("  python scripts/ingest_benv2_streaming.py \\")
    print(f"    --s2_root {s2_root} \\")
    print(f"    --s1_root {s1_root} \\")
    print("    --out_dir data/processed/benv2_tiles \\")
    print("    --target_gb 50")
    print()


if __name__ == '__main__':
    main()
