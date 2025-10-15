#!/usr/bin/env python3
"""
BigEarthNet v2 Streaming Ingestion Pipeline

Pairs S1 and S2 patches, processes them into aligned tiles, and deletes originals
to save disk space during conversion.

Features:
- Streaming deletion after successful tile creation
- Space monitoring and auto-pause when disk is low
- Train/val/test split with country stratification
- Automatic S1/S2 alignment and resampling
- Robust error handling and atomic file writes
- Progress tracking and JSON logging
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import warnings

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)


class DiskSpaceMonitor:
    """Monitor disk space and pause processing if too low."""
    
    def __init__(self, min_free_gb: float = 30.0):
        self.min_free_gb = min_free_gb
        
    def get_free_gb(self, path: Path) -> float:
        """Get free disk space in GB for the drive containing path."""
        # If path doesn't exist, check parent or use root
        check_path = path
        while not check_path.exists() and check_path.parent != check_path:
            check_path = check_path.parent
        
        if sys.platform == 'win32':
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                str(check_path), None, None, ctypes.pointer(free_bytes)
            )
            return free_bytes.value / (1024**3)
        else:
            stat = os.statvfs(str(check_path))
            return (stat.f_bavail * stat.f_frsize) / (1024**3)
    
    def check_and_wait(self, path: Path) -> float:
        """Check free space and wait if below threshold. Returns current free GB."""
        while True:
            free_gb = self.get_free_gb(path)
            if free_gb >= self.min_free_gb:
                return free_gb
            
            print(f"\n[!] LOW DISK SPACE: {free_gb:.2f} GB free (need {self.min_free_gb} GB)")
            print(f"    Waiting 30s for space to free up... Press Ctrl+C to abort")
            time.sleep(30)


class BENv2PatchFinder:
    """Find and pair BigEarthNet v2 S1 and S2 patches."""
    
    def __init__(self, s2_root: Path, s1_root: Path):
        self.s2_root = Path(s2_root)
        self.s1_root = Path(s1_root)
        
    def find_all_patches(self) -> Dict[str, Dict[str, Path]]:
        """
        Find all S1 and S2 patches and pair them by tile ID and row/col.
        
        Returns:
            Dict mapping patch_name to {'s2': Path, 's1': Path, 'country': str}
        """
        print("[*] Scanning for BigEarthNet patches...")
        
        # Find S2 patch subdirectories (individual patches with row/col)
        s2_patches = {}  # key: "TILE_ROW_COL", value: {'path': Path, 'parent': str}
        s2_count = 0
        for s2_parent in self._find_s2_patch_parents(self.s2_root):
            for s2_child in s2_parent.iterdir():
                if not s2_child.is_dir():
                    continue
                
                # Extract tile_row_col from child name
                # Format: S2A_MSIL2A_..._TXXXXX_ROW_COL
                tile_rowcol = self._extract_tile_rowcol_s2(s2_child.name)
                if tile_rowcol:
                    tile_id, row, col = tile_rowcol
                    key = f"{tile_id}_{row}_{col}"
                    country = self._tile_to_country(tile_id)
                    s2_patches[key] = {
                        'path': s2_child,
                        'parent': s2_parent.name,
                        'country': country
                    }
                    s2_count += 1
        
        print(f"   Found {s2_count} S2 patches")
        
        # Find S1 patch subdirectories and pair with S2
        s1_patches = {}  # key: "TILE_ROW_COL", value: {'path': Path, 'parent': str}
        s1_count = 0
        for s1_parent in self._find_s1_patch_parents(self.s1_root):
            for s1_child in s1_parent.iterdir():
                if not s1_child.is_dir():
                    continue
                
                # Extract tile_row_col from child name
                # Format: S1A_IW_GRDH_..._XXXXX_ROW_COL
                tile_rowcol = self._extract_tile_rowcol_s1(s1_child.name)
                if tile_rowcol:
                    tile_id, row, col = tile_rowcol
                    key = f"{tile_id}_{row}_{col}"
                    s1_patches[key] = {
                        'path': s1_child,
                        'parent': s1_parent.name
                    }
                    s1_count += 1
        
        print(f"   Found {s1_count} S1 patches")
        
        # Pair S1 and S2 by matching tile_row_col keys
        paired = {}
        for key, s2_info in s2_patches.items():
            if key in s1_patches:
                # Use full S2 child name as the patch ID
                patch_id = s2_info['path'].name
                paired[patch_id] = {
                    's2': s2_info['path'],
                    's1': s1_patches[key]['path'],
                    'country': s2_info['country']
                }
        
        print(f"    [OK] {len(paired)} complete S1+S2 pairs ready")
        
        return paired
    
    def _find_s2_patch_parents(self, root: Path) -> List[Path]:
        """Find S2 parent patch directories."""
        patches = []
        
        if not root.exists():
            return patches
        
        # S2 structure: root/BigEarthNet-S2/PATCH_PARENT/PATCH_CHILD/*.tif
        for level1 in root.iterdir():
            if not level1.is_dir():
                continue
            
            for level2 in level1.iterdir():
                if not level2.is_dir():
                    continue
                
                # Check if this looks like a S2 patch parent
                if re.match(r'S2[AB]_MSIL2A_\d{8}T\d{6}_N\d{4}_R\d{3}_T\w+', level2.name):
                    patches.append(level2)
        
        return patches
    
    def _find_s1_patch_parents(self, root: Path) -> List[Path]:
        """Find S1 parent patch directories."""
        patches = []
        
        if not root.exists():
            return patches
        
        # S1 structure: root/BigEarthNet-S1/PATCH_PARENT/PATCH_CHILD/*.tif
        for level1 in root.iterdir():
            if not level1.is_dir():
                continue
            
            for level2 in level1.iterdir():
                if not level2.is_dir():
                    continue
                
                # Check if this looks like a S1 patch parent
                if re.match(r'S1[AB]_IW_GRDH_\d[A-Z]{3}_\d{8}T\d{6}', level2.name):
                    patches.append(level2)
        
        return patches
    
    def _extract_tile_rowcol_s2(self, dirname: str) -> Optional[Tuple[str, str, str]]:
        """Extract tile ID and row/col from S2 patch child directory name."""
        # S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57
        match = re.match(r'S2[AB]_MSIL2A_\d{8}T\d{6}_N\d{4}_R\d{3}_T(\w+)_(\d+)_(\d+)', dirname)
        if match:
            return match.group(1), match.group(2), match.group(3)  # tile_id, row, col
        return None
    
    def _extract_tile_rowcol_s1(self, dirname: str) -> Optional[Tuple[str, str, str]]:
        """Extract tile ID and row/col from S1 patch child directory name."""
        # S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39
        match = re.match(r'S1[AB]_IW_GRDH_\d[A-Z]{3}_\d{8}T\d{6}_(\w+)_(\d+)_(\d+)', dirname)
        if match:
            return match.group(1), match.group(2), match.group(3)  # tile_id, row, col
        return None
    
    def _tile_to_country(self, tile_id: str) -> str:
        """Extract country from tile ID (e.g., '33UUP' -> 'DE')."""
        # Extract UTM zone number from tile ID
        zone_match = re.match(r'(\d{2})', tile_id)
        if zone_match:
            zone = int(zone_match.group(1))
            return self._zone_to_country(zone)
        return 'EU'
    
    def _zone_to_country(self, zone: int) -> str:
        """Rough mapping of UTM zones to European countries."""
        zone_map = {
            32: 'DE',  # Germany/Austria
            33: 'DE',  # Germany/Poland/Austria
            34: 'PL',  # Poland
            35: 'PL',  # Poland/Lithuania
            29: 'ES',  # Spain/Portugal
            30: 'ES',  # Spain
            31: 'FR',  # France
        }
        return zone_map.get(zone, 'EU')


class BENv2PatchProcessor:
    """Process paired S1/S2 patches into aligned tiles."""
    
    def __init__(self, float16: bool = True):
        self.float16 = float16
        self.dtype = np.float16 if float16 else np.float32
        
    def process_pair(
        self, 
        s2_dir: Path, 
        s1_dir: Path,
        patch_id: str
    ) -> Tuple[Optional[np.ndarray], Optional[Dict], Optional[int], Optional[int]]:
        """
        Process a paired S1/S2 patch.
        
        Returns:
            (data_dict, metadata, height, width) or (None, None, None, None) on error
        """
        try:
            # Read S2 bands
            s2_data, s2_meta = self._read_s2_patch(s2_dir)
            if s2_data is None:
                return None, None, None, None
            
            # Read S1 bands
            s1_data, s1_meta = self._read_s1_patch(s1_dir)
            if s1_data is None:
                return None, None, None, None
            
            # Align S1 to S2 grid
            s1_aligned = self._align_s1_to_s2(s1_data, s1_meta, s2_data, s2_meta)
            if s1_aligned is None:
                return None, None, None, None
            
            # Normalize to [0, 1]
            s2_normalized, s2_stats = self._normalize_data(s2_data)
            s1_normalized, s1_stats = self._normalize_data(s1_aligned)
            
            # Build output dict
            height, width = s2_normalized.shape[1], s2_normalized.shape[2]
            data_dict = {
                's2_b2': s2_normalized[0].astype(self.dtype),
                's2_b3': s2_normalized[1].astype(self.dtype),
                's2_b4': s2_normalized[2].astype(self.dtype),
                's2_b8': s2_normalized[3].astype(self.dtype),
                's1_vv': s1_normalized[0].astype(self.dtype),
                's1_vh': s1_normalized[1].astype(self.dtype),
            }
            
            # Build metadata
            metadata = {
                'id': patch_id,
                'height': int(height),
                'width': int(width),
                'date_s2': s2_meta.get('date', 'unknown'),
                'date_s1': s1_meta.get('date', 'unknown'),
                's2_normalization': s2_stats,
                's1_normalization': s1_stats,
            }
            
            return data_dict, metadata, height, width
            
        except Exception as e:
            import traceback
            print(f"   [ERR] Error processing {patch_id}: {e}")
            print(f"   [ERR] Traceback: {traceback.format_exc()}")
            return None, None, None, None
    
    def _read_s2_patch(self, s2_dir: Path) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Read S2 B02, B03, B04, B08 bands."""
        try:
            band_map = {
                'B02': 0,  # Blue
                'B03': 1,  # Green
                'B04': 2,  # Red
                'B08': 3,  # NIR
            }
            
            bands = [None] * 4
            date = None
            
            # BigEarthNet structure: s2_dir is the patch child directory containing TIF files directly
            # Extract date from directory name
            date_match = re.search(r'(\d{8})', s2_dir.name)
            if date_match:
                date = date_match.group(1)
            
            # Look for band files with pattern: *_B*.tif directly in this directory
            for band_name, band_idx in band_map.items():
                # Pattern: <patch_name>_<band>.tif
                band_pattern = f"*_{band_name}.tif"
                band_files = list(s2_dir.glob(band_pattern))
                
                if band_files:
                    with rasterio.open(band_files[0]) as src:
                        bands[band_idx] = src.read(1).astype(np.float32)
            
            # Check if all bands found
            if any(b is None for b in bands):
                missing = [f"B{i+2:02d}" if i < 3 else 'B08' for i, b in enumerate(bands) if b is None]
                print(f"   [WARN] Missing S2 bands: {missing} in {s2_dir.name}")
                return None, None
            
            data = np.stack(bands, axis=0)
            metadata = {'date': date or 'unknown'}
            
            return data, metadata
            
        except Exception as e:
            import traceback
            print(f"   [ERR] Error reading S2 patch {s2_dir}: {e}")
            print(f"   [ERR] {traceback.format_exc()}")
            return None, None
    
    def _read_s1_patch(self, s1_dir: Path) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Read S1 VV and VH bands."""
        try:
            vv_data = None
            vh_data = None
            date = None
            
            # BigEarthNet structure: s1_dir is the patch child directory containing TIF files directly
            # Extract date from directory name
            date_match = re.search(r'(\d{8})', s1_dir.name)
            if date_match:
                date = date_match.group(1)
            
            # Look for VV and VH files with pattern: *_VV.tif / *_VH.tif directly in this directory
            vv_files = list(s1_dir.glob('*_VV.tif'))
            if vv_files:
                with rasterio.open(vv_files[0]) as src:
                    vv_data = src.read(1).astype(np.float32)
            
            vh_files = list(s1_dir.glob('*_VH.tif'))
            if vh_files:
                with rasterio.open(vh_files[0]) as src:
                    vh_data = src.read(1).astype(np.float32)
            
            if vv_data is None or vh_data is None:
                missing = []
                if vv_data is None:
                    missing.append('VV')
                if vh_data is None:
                    missing.append('VH')
                print(f"   [WARN] Missing S1 bands: {missing} in {s1_dir.name}")
                return None, None
            
            data = np.stack([vv_data, vh_data], axis=0)
            metadata = {'date': date or 'unknown'}
            
            return data, metadata
            
        except Exception as e:
            import traceback
            print(f"   [ERR] Error reading S1 patch {s1_dir}: {e}")
            print(f"   [ERR] {traceback.format_exc()}")
            return None, None
    
    def _align_s1_to_s2(
        self, 
        s1_data: np.ndarray, 
        s1_meta: Dict, 
        s2_data: np.ndarray, 
        s2_meta: Dict
    ) -> Optional[np.ndarray]:
        """Align S1 data to S2 grid using resampling."""
        target_shape = s2_data.shape[1:]  # (H, W)
        
        if s1_data.shape[1:] == target_shape:
            return s1_data
        
        # Resample each S1 band to match S2 dimensions
        s1_aligned = np.zeros((s1_data.shape[0], *target_shape), dtype=np.float32)
        
        from scipy.ndimage import zoom
        zoom_factors = (target_shape[0] / s1_data.shape[1], target_shape[1] / s1_data.shape[2])
        
        for i in range(s1_data.shape[0]):
            s1_aligned[i] = zoom(s1_data[i], zoom_factors, order=1)
        
        return s1_aligned
    
    def _normalize_data(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Normalize data to [0, 1] and return normalization stats."""
        min_val = float(np.nanmin(data))
        max_val = float(np.nanmax(data))
        
        if max_val == min_val:
            normalized = np.zeros_like(data)
        else:
            normalized = (data - min_val) / (max_val - min_val)
        
        # Clip to [0, 1] in case of any outliers
        normalized = np.clip(normalized, 0, 1)
        
        stats = {
            'min': min_val,
            'max': max_val,
            'mean': float(np.nanmean(data)),
            'std': float(np.nanstd(data))
        }
        
        return normalized, stats


class BENv2StreamingConverter:
    """Main streaming converter with deletion and space management."""
    
    def __init__(
        self,
        s2_root: Path,
        s1_root: Path,
        out_dir: Path,
        target_gb: float = 50.0,
        min_free_gb: float = 30.0,
        float16: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        countries_keep: Optional[List[str]] = None,
        countries_drop: Optional[List[str]] = None,
        max_per_country: Optional[int] = None,
    ):
        self.s2_root = Path(s2_root)
        self.s1_root = Path(s1_root)
        self.out_dir = Path(out_dir)
        self.target_gb = target_gb
        self.float16 = float16
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        
        self.disk_monitor = DiskSpaceMonitor(min_free_gb)
        self.patch_finder = BENv2PatchFinder(s2_root, s1_root)
        self.processor = BENv2PatchProcessor(float16)
        
        self.countries_keep = set(countries_keep) if countries_keep else None
        self.countries_drop = set(countries_drop) if countries_drop else set()
        self.max_per_country = max_per_country
        
        self.log_file = Path('logs/benv2_ingest.jsonl')
        self.log_file.parent.mkdir(exist_ok=True)
        
        self.stats = {
            'processed': 0,
            'success': 0,
            'failed': 0,
            'deleted_s1_mb': 0.0,
            'deleted_s2_mb': 0.0,
            'total_bytes': 0,
        }
    
    def run(self):
        """Main conversion loop."""
        print("\n" + "="*60)
        print("BigEarthNet v2 Streaming Converter")
        print("="*60)
        print(f"[DIR] S2 root: {self.s2_root}")
        print(f"[DIR] S1 root: {self.s1_root}")
        print(f"[DIR] Output: {self.out_dir}")
        print(f"[CFG] Target size: {self.target_gb} GB")
        print(f"[CFG] Min free space: {self.disk_monitor.min_free_gb} GB")
        print(f"[CFG] Data type: {'float16' if self.float16 else 'float32'}")
        print("="*60 + "\n")
        
        # Create output directory
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all patches
        patches = self.patch_finder.find_all_patches()
        
        # Filter by country
        patches = self._filter_by_country(patches)
        
        # Assign splits
        patches_with_splits = self._assign_splits(patches)
        
        print(f"\n[SPLIT] Distribution:")
        split_counts = {'train': 0, 'val': 0, 'test': 0}
        for _, info in patches_with_splits.items():
            split_counts[info['split']] += 1
        for split, count in split_counts.items():
            print(f"   {split}: {count} patches")
        
        # Process patches
        print(f"\n[START] Beginning conversion...\n")
        
        for patch_id, info in patches_with_splits.items():
            # Check disk space
            free_gb = self.disk_monitor.check_and_wait(self.out_dir)
            
            # Check target size
            current_gb = self.stats['total_bytes'] / (1024**3)
            if current_gb >= self.target_gb:
                print(f"\n[OK] Target size reached: {current_gb:.2f} GB / {self.target_gb} GB")
                break
            
            # Process patch
            self._process_and_delete_patch(patch_id, info, free_gb)
            
            # Progress update
            if self.stats['processed'] % 10 == 0:
                self._print_progress(free_gb)
        
        # Final summary
        self._print_summary()
    
    def _filter_by_country(self, patches: Dict) -> Dict:
        """Filter patches by country criteria."""
        filtered = {}
        country_counts = {}
        
        for patch_id, info in patches.items():
            country = info['country']
            
            # Apply keep/drop filters
            if self.countries_keep and country not in self.countries_keep:
                continue
            if country in self.countries_drop:
                continue
            
            # Apply max per country
            if self.max_per_country:
                count = country_counts.get(country, 0)
                if count >= self.max_per_country:
                    continue
                country_counts[country] = count + 1
            
            filtered[patch_id] = info
        
        print(f"\n[FILTER] Country filtering:")
        for country, count in sorted(country_counts.items()):
            print(f"   {country}: {count} patches")
        
        return filtered
    
    def _assign_splits(self, patches: Dict) -> Dict:
        """Assign train/val/test splits, stratified by country if possible."""
        # Group by country
        by_country = {}
        for patch_id, info in patches.items():
            country = info['country']
            if country not in by_country:
                by_country[country] = []
            by_country[country].append((patch_id, info))
        
        # Assign splits within each country
        patches_with_splits = {}
        
        for country, patch_list in by_country.items():
            n = len(patch_list)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            
            # Shuffle patches
            np.random.shuffle(patch_list)
            
            for i, (patch_id, info) in enumerate(patch_list):
                if i < n_train:
                    split = 'train'
                elif i < n_train + n_val:
                    split = 'val'
                else:
                    split = 'test'
                
                info['split'] = split
                patches_with_splits[patch_id] = info
        
        return patches_with_splits
    
    def _process_and_delete_patch(self, patch_id: str, info: Dict, free_gb: float):
        """Process a single patch and delete originals on success."""
        self.stats['processed'] += 1
        
        try:
            # Process patch
            data_dict, metadata, height, width = self.processor.process_pair(
                info['s2'], info['s1'], patch_id
            )
            
            if data_dict is None:
                self.stats['failed'] += 1
                self._log_result(patch_id, info, False, 0, 0, 0, free_gb, "processing_failed")
                return
            
            # Add metadata fields
            metadata['country'] = info['country']
            metadata['split'] = info['split']
            
            # Write output
            out_path = self.out_dir / f"{patch_id}.npz"
            json_path = self.out_dir / f"{patch_id}.json"
            
            success, npz_bytes = self._atomic_write(out_path, data_dict)
            if not success:
                self.stats['failed'] += 1
                self._log_result(patch_id, info, False, 0, 0, 0, free_gb, "write_failed")
                return
            
            # Write JSON metadata
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Verify written file
            if not self._verify_tile(out_path, data_dict):
                self.stats['failed'] += 1
                self._log_result(patch_id, info, False, npz_bytes, 0, 0, free_gb, "verification_failed")
                out_path.unlink(missing_ok=True)
                json_path.unlink(missing_ok=True)
                return
            
            # Delete originals
            s1_mb = self._delete_directory(info['s1'])
            s2_mb = self._delete_directory(info['s2'])
            
            self.stats['success'] += 1
            self.stats['total_bytes'] += npz_bytes
            self.stats['deleted_s1_mb'] += s1_mb
            self.stats['deleted_s2_mb'] += s2_mb
            
            self._log_result(patch_id, info, True, npz_bytes, s1_mb, s2_mb, free_gb, "ok")
            
            print(f"[OK] {patch_id[:40]:<40} | {info['split']:<5} | {npz_bytes/1024/1024:.1f}MB | freed {s1_mb+s2_mb:.1f}MB")
            
        except Exception as e:
            self.stats['failed'] += 1
            self._log_result(patch_id, info, False, 0, 0, 0, free_gb, f"error: {e}")
            print(f"[ERR] {patch_id[:40]:<40} | ERROR: {e}")
    
    def _atomic_write(self, path: Path, data_dict: Dict) -> Tuple[bool, int]:
        """Atomically write NPZ file."""
        tmp_path = path.with_suffix('.tmp.npz')
        
        try:
            np.savez_compressed(tmp_path, **data_dict)
            tmp_path.replace(path)
            return True, path.stat().st_size
        except Exception as e:
            print(f"   Write error: {e}")
            tmp_path.unlink(missing_ok=True)
            return False, 0
    
    def _verify_tile(self, path: Path, expected: Dict) -> bool:
        """Verify tile can be loaded and has correct structure."""
        try:
            data = np.load(path)
            
            # Check all keys present
            for key in expected.keys():
                if key not in data:
                    return False
                
                # Check shape matches
                if data[key].shape != expected[key].shape:
                    return False
                
                # Check for NaNs
                if np.any(np.isnan(data[key])):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _delete_directory(self, path: Path) -> float:
        """Delete directory and return size in MB."""
        try:
            # Calculate size before deletion
            total_bytes = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            
            # Delete
            shutil.rmtree(path)
            
            return total_bytes / (1024**2)
            
        except Exception as e:
            print(f"   Warning: Could not delete {path}: {e}")
            return 0.0
    
    def _log_result(self, patch_id: str, info: Dict, success: bool, 
                    npz_bytes: int, s1_mb: float, s2_mb: float, free_gb: float, status: str):
        """Log result to JSON lines file."""
        log_entry = {
            'id': patch_id,
            'split': info['split'],
            'country': info['country'],
            'ok': success,
            'bytes_npz': npz_bytes,
            'deleted_s1_mb': s1_mb,
            'deleted_s2_mb': s2_mb,
            'free_gb': round(free_gb, 2),
            'status': status,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _print_progress(self, free_gb: float):
        """Print progress summary."""
        current_gb = self.stats['total_bytes'] / (1024**3)
        deleted_gb = (self.stats['deleted_s1_mb'] + self.stats['deleted_s2_mb']) / 1024
        
        print(f"\n[PROGRESS] {self.stats['processed']} processed | "
              f"{self.stats['success']} success | {self.stats['failed']} failed")
        print(f"   Created: {current_gb:.2f} / {self.target_gb:.2f} GB")
        print(f"   Deleted: {deleted_gb:.2f} GB | Free: {free_gb:.2f} GB\n")
    
    def _print_summary(self):
        """Print final conversion summary."""
        current_gb = self.stats['total_bytes'] / (1024**3)
        deleted_gb = (self.stats['deleted_s1_mb'] + self.stats['deleted_s2_mb']) / 1024
        
        print("="*60)
        print("[COMPLETE] Conversion Finished!")
        print("="*60)
        print(f"Processed: {self.stats['processed']} patches")
        print(f"Success: {self.stats['success']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Output size: {current_gb:.2f} GB")
        print(f"Space freed: {deleted_gb:.2f} GB")
        print(f"Log file: {self.log_file}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='BigEarthNet v2 Streaming Converter',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--s2_root', type=str, required=True,
                        help='Path to S2 data root')
    parser.add_argument('--s1_root', type=str, required=True,
                        help='Path to S1 data root')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for tiles')
    parser.add_argument('--target_gb', type=float, default=50.0,
                        help='Target output size in GB (default: 50)')
    parser.add_argument('--min_free_gb', type=float, default=30.0,
                        help='Minimum free disk space in GB (default: 30)')
    parser.add_argument('--float16', type=lambda x: str(x).lower() == 'true', 
                        default=True, help='Use float16 (default: true)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train split ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Val split ratio (default: 0.1)')
    parser.add_argument('--countries', type=str, default=None,
                        help='Country filter: keep=DE,FR,IT or drop=ES,PT')
    parser.add_argument('--max_per_country', type=int, default=None,
                        help='Maximum patches per country')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splits (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Parse country filters
    countries_keep = None
    countries_drop = None
    if args.countries:
        if args.countries.startswith('keep='):
            countries_keep = args.countries[5:].split(',')
        elif args.countries.startswith('drop='):
            countries_drop = args.countries[5:].split(',')
    
    # Create converter
    converter = BENv2StreamingConverter(
        s2_root=args.s2_root,
        s1_root=args.s1_root,
        out_dir=args.out_dir,
        target_gb=args.target_gb,
        min_free_gb=args.min_free_gb,
        float16=args.float16,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        countries_keep=countries_keep,
        countries_drop=countries_drop,
        max_per_country=args.max_per_country,
    )
    
    # Run conversion
    converter.run()


if __name__ == '__main__':
    main()
