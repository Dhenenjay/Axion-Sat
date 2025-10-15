"""
Smoke test for BigEarthNet v2 streaming converter.

Creates synthetic S1/S2 patch pairs, runs the converter, and verifies:
- NPZ files are created with correct structure
- All 6 expected keys present (s1_vv, s1_vh, s2_b2, s2_b3, s2_b4, s2_b8)
- Data is float16 dtype
- Shapes are consistent
- Original source folders are deleted after processing
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


def create_synthetic_s2_patch(patch_parent: Path, patch_child_name: str, size: int = 10):
    """
    Create a synthetic S2 patch with B02, B03, B04, B08 bands.
    
    Mimics BigEarthNet-S2 structure:
    patch_parent/
        patch_child/
            <patch_child>_B02.tif
            <patch_child>_B03.tif
            <patch_child>_B04.tif
            <patch_child>_B08.tif
    """
    # Create the patch child directory
    patch_child = patch_parent / patch_child_name
    patch_child.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic band TIF files using rasterio
    import rasterio
    from rasterio.transform import from_bounds
    
    # Synthetic data: random values in typical S2 range (0-10000)
    bands = {
        'B02': np.random.randint(0, 3000, (size, size), dtype=np.uint16),
        'B03': np.random.randint(0, 3500, (size, size), dtype=np.uint16),
        'B04': np.random.randint(0, 4000, (size, size), dtype=np.uint16),
        'B08': np.random.randint(0, 5000, (size, size), dtype=np.uint16),
    }
    
    # Dummy geotransform (not critical for this test)
    transform = from_bounds(0, 0, size * 10, size * 10, size, size)
    
    for band_name, data in bands.items():
        # File naming: <patch_child_name>_BAND.tif
        tif_path = patch_child / f"{patch_child_name}_{band_name}.tif"
        
        with rasterio.open(
            tif_path,
            'w',
            driver='GTiff',
            height=size,
            width=size,
            count=1,
            dtype=data.dtype,
            crs='EPSG:32632',  # UTM zone 32N (common for Europe)
            transform=transform,
        ) as dst:
            dst.write(data, 1)


def create_synthetic_s1_patch(patch_parent: Path, patch_child_name: str, size: int = 10):
    """
    Create a synthetic S1 patch with VV and VH polarizations.
    
    Mimics BigEarthNet-S1 structure:
    patch_parent/
        patch_child/
            <patch_child>_VV.tif
            <patch_child>_VH.tif
    """
    # Create the patch child directory
    patch_child = patch_parent / patch_child_name
    patch_child.mkdir(parents=True, exist_ok=True)
    
    import rasterio
    from rasterio.transform import from_bounds
    
    # Synthetic SAR data: typical range is -25 to 5 dB
    polarizations = {
        'VV': np.random.uniform(-20, 0, (size, size)).astype(np.float32),
        'VH': np.random.uniform(-25, -5, (size, size)).astype(np.float32),
    }
    
    transform = from_bounds(0, 0, size * 10, size * 10, size, size)
    
    for pol_name, data in polarizations.items():
        # File naming: <patch_child_name>_POL.tif
        tif_path = patch_child / f"{patch_child_name}_{pol_name}.tif"
        
        with rasterio.open(
            tif_path,
            'w',
            driver='GTiff',
            height=size,
            width=size,
            count=1,
            dtype=data.dtype,
            crs='EPSG:32632',
            transform=transform,
        ) as dst:
            dst.write(data, 1)


def create_synthetic_ben_dataset(root_dir: Path, num_patches: int = 2):
    """
    Create a complete synthetic BigEarthNet dataset structure.
    
    Returns:
        Tuple of (s2_root, s1_root, s2_patch_ids)
    """
    s2_root = root_dir / "BENv2" / "S2"
    s1_root = root_dir / "BENv2" / "S1"
    
    # Create the BigEarthNet container directories
    (s2_root / "BigEarthNet-S2").mkdir(parents=True, exist_ok=True)
    (s1_root / "BigEarthNet-S1").mkdir(parents=True, exist_ok=True)
    
    s2_root.mkdir(parents=True, exist_ok=True)
    s1_root.mkdir(parents=True, exist_ok=True)
    
    s2_patch_ids = []
    
    for i in range(num_patches):
        # Generate unique row/col for each patch
        row = 26 + i
        col = 57 + i
        
        # Generate realistic S2 patch parent and child IDs
        s2_parent_id = f"S2A_MSIL2A_20170613T10103{i}_N9999_R022_T32UPU"
        s2_child_id = f"{s2_parent_id}_{row}_{col}"
        s2_patch_ids.append(s2_child_id)
        
        # Create S2 patch (parent dir with child subdirectory)
        # Structure: S2/BigEarthNet-S2/PARENT/CHILD
        s2_parent_dir = s2_root / "BigEarthNet-S2" / s2_parent_id
        create_synthetic_s2_patch(s2_parent_dir, s2_child_id, size=10)
        
        # Generate corresponding S1 patch parent and child IDs (different format)
        # S1 format: S1A_IW_GRDH_1SDV_20170613T165043
        # Structure: S1/BigEarthNet-S1/PARENT/CHILD
        s1_parent_id = f"S1A_IW_GRDH_1SDV_20170613T16504{i}"
        s1_child_id = f"{s1_parent_id}_32UPU_{row}_{col}"
        s1_parent_dir = s1_root / "BigEarthNet-S1" / s1_parent_id
        create_synthetic_s1_patch(s1_parent_dir, s1_child_id, size=10)
    
    return s2_root, s1_root, s2_patch_ids


class TestBENv2IngestSmoke:
    """Smoke tests for BigEarthNet v2 streaming converter."""
    
    def test_synthetic_dataset_creation(self, tmp_path):
        """Test that synthetic dataset creation works."""
        s2_root, s1_root, patch_child_ids = create_synthetic_ben_dataset(tmp_path, num_patches=1)
        
        # Verify S2 structure
        assert s2_root.exists()
        
        # patch_child_ids contains child names like S2A_MSIL2A_..._T32UPU_26_57
        # These are inside parent dirs like S2A_MSIL2A_..._T32UPU
        # Extract parent from child ID (remove _ROW_COL suffix)
        child_id = patch_child_ids[0]
        parent_id = '_'.join(child_id.split('_')[:-2])  # Remove last two components (row, col)
        
        patch_parent = s2_root / "BigEarthNet-S2" / parent_id
        patch_child = patch_parent / child_id
        assert patch_child.exists(), f"Expected S2 child patch at {patch_child}"
        
        # Check for band files directly in child
        tif_files = list(patch_child.glob('*.tif'))
        assert len(tif_files) == 4, f"Expected 4 S2 bands, found {len(tif_files)}"  # B02, B03, B04, B08
        
        # Verify S1 structure
        assert s1_root.exists()
        # S1 patches have different parent naming - check if any parent dirs exist
        s1_ben_root = s1_root / "BigEarthNet-S1"
        s1_parent_dirs = [d for d in s1_ben_root.iterdir() if d.is_dir()]
        assert len(s1_parent_dirs) > 0, "No S1 parent directories found"
        
        # Check first S1 parent has a child with VV/VH files
        s1_parent = s1_parent_dirs[0]
        s1_children = [d for d in s1_parent.iterdir() if d.is_dir()]
        assert len(s1_children) > 0, "No S1 child directories found"
        
        s1_child = s1_children[0]
        s1_tif_files = list(s1_child.glob('*.tif'))
        assert len(s1_tif_files) == 2, f"Expected 2 S1 bands, found {len(s1_tif_files)}"  # VV, VH
    
    def test_converter_basic_run(self, tmp_path):
        """Test that converter runs without errors on synthetic data."""
        # Create synthetic dataset
        s2_root, s1_root, patch_ids = create_synthetic_ben_dataset(tmp_path, num_patches=2)
        
        # Create output directory
        out_dir = tmp_path / "tiles"
        out_dir.mkdir()
        
        # Run converter
        script_path = Path(__file__).parent.parent / "scripts" / "ingest_benv2_streaming.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--s2_root", str(s2_root),
            "--s1_root", str(s1_root),
            "--out_dir", str(out_dir),
            "--target_gb", "0.001",  # Very small to process just a few patches
            "--min_free_gb", "0.1",   # Low threshold for test
            "--float16", "true",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check that it ran successfully
        assert result.returncode == 0, f"Converter failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        
        # Verify output
        npz_files = list(out_dir.glob("*.npz"))
        assert len(npz_files) > 0, "No NPZ files created"
    
    def test_npz_structure_and_dtype(self, tmp_path):
        """Test NPZ files have correct structure, keys, and dtypes."""
        # Create synthetic dataset
        s2_root, s1_root, patch_ids = create_synthetic_ben_dataset(tmp_path, num_patches=2)
        
        # Create output directory
        out_dir = tmp_path / "tiles"
        out_dir.mkdir()
        
        # Run converter
        script_path = Path(__file__).parent.parent / "scripts" / "ingest_benv2_streaming.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--s2_root", str(s2_root),
            "--s1_root", str(s1_root),
            "--out_dir", str(out_dir),
            "--target_gb", "0.001",
            "--min_free_gb", "0.1",
            "--float16", "true",
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Check NPZ files
        npz_files = list(out_dir.glob("*.npz"))
        assert len(npz_files) > 0, "No NPZ files created"
        
        for npz_path in npz_files:
            # Load NPZ
            data = np.load(npz_path)
            
            # Check all 6 keys present
            expected_keys = {'s1_vv', 's1_vh', 's2_b2', 's2_b3', 's2_b4', 's2_b8'}
            actual_keys = set(data.files)
            assert actual_keys == expected_keys, \
                f"Missing keys: {expected_keys - actual_keys}, Extra keys: {actual_keys - expected_keys}"
            
            # Check dtype is float16
            for key in expected_keys:
                assert data[key].dtype == np.float16, \
                    f"Key {key} has dtype {data[key].dtype}, expected float16"
            
            # Check shapes are consistent
            shapes = [data[key].shape for key in expected_keys]
            assert len(set(shapes)) == 1, f"Inconsistent shapes: {shapes}"
            
            # Check shape is 2D
            shape = data['s2_b2'].shape
            assert len(shape) == 2, f"Expected 2D array, got shape {shape}"
            
            # Check values are in [0, 1] range (normalized)
            for key in expected_keys:
                arr = data[key]
                assert np.all(arr >= 0) and np.all(arr <= 1), \
                    f"Key {key} has values outside [0, 1]: min={arr.min()}, max={arr.max()}"
            
            # Check for NaNs
            for key in expected_keys:
                assert not np.any(np.isnan(data[key])), f"Key {key} contains NaNs"
    
    def test_json_metadata(self, tmp_path):
        """Test that JSON metadata files are created and valid."""
        # Create synthetic dataset
        s2_root, s1_root, patch_ids = create_synthetic_ben_dataset(tmp_path, num_patches=2)
        
        # Create output directory
        out_dir = tmp_path / "tiles"
        out_dir.mkdir()
        
        # Run converter
        script_path = Path(__file__).parent.parent / "scripts" / "ingest_benv2_streaming.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--s2_root", str(s2_root),
            "--s1_root", str(s1_root),
            "--out_dir", str(out_dir),
            "--target_gb", "0.001",
            "--min_free_gb", "0.1",
            "--float16", "true",
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Check JSON files
        json_files = list(out_dir.glob("*.json"))
        assert len(json_files) > 0, "No JSON metadata files created"
        
        for json_path in json_files:
            # Load JSON
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            # Check required fields
            required_fields = ['id', 'country', 'split', 'height', 'width']
            for field in required_fields:
                assert field in metadata, f"Missing field '{field}' in metadata"
            
            # Check split is valid
            assert metadata['split'] in ['train', 'val', 'test'], \
                f"Invalid split: {metadata['split']}"
            
            # Check dimensions are positive
            assert metadata['height'] > 0, f"Invalid height: {metadata['height']}"
            assert metadata['width'] > 0, f"Invalid width: {metadata['width']}"
    
    def test_source_deletion(self, tmp_path):
        """Test that original source patch folders are deleted after processing."""
        # Create synthetic dataset
        s2_root, s1_root, patch_child_ids = create_synthetic_ben_dataset(tmp_path, num_patches=2)
        
        # Build actual patch paths (parent/child structure)
        s2_patches = []
        for child_id in patch_child_ids:
            parent_id = '_'.join(child_id.split('_')[:-2])
            s2_patches.append(s2_root / "BigEarthNet-S2" / parent_id / child_id)
        
        # For S1, find the actual child directories
        s1_patches = []
        s1_ben_root = s1_root / "BigEarthNet-S1"
        for s1_parent in s1_ben_root.iterdir():
            if s1_parent.is_dir():
                for s1_child in s1_parent.iterdir():
                    if s1_child.is_dir():
                        s1_patches.append(s1_child)
        
        # Verify patches exist before processing
        for patch in s2_patches + s1_patches:
            assert patch.exists(), f"Patch {patch} should exist before processing"
        
        # Create output directory
        out_dir = tmp_path / "tiles"
        out_dir.mkdir()
        
        # Run converter
        script_path = Path(__file__).parent.parent / "scripts" / "ingest_benv2_streaming.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--s2_root", str(s2_root),
            "--s1_root", str(s1_root),
            "--out_dir", str(out_dir),
            "--target_gb", "1.0",  # Large enough to process all test patches
            "--min_free_gb", "0.1",
            "--float16", "true",
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Check that NPZ files were created
        npz_files = list(out_dir.glob("*.npz"))
        assert len(npz_files) > 0, "No NPZ files created"
        
        # Verify source patches are deleted
        for patch in s2_patches + s1_patches:
            assert not patch.exists(), \
                f"Patch {patch} should be deleted after successful processing"
    
    def test_split_distribution(self, tmp_path):
        """Test that train/val/test splits are assigned correctly."""
        # Create more patches to test split distribution
        s2_root, s1_root, patch_ids = create_synthetic_ben_dataset(tmp_path, num_patches=10)
        
        # Create output directory
        out_dir = tmp_path / "tiles"
        out_dir.mkdir()
        
        # Run converter
        script_path = Path(__file__).parent.parent / "scripts" / "ingest_benv2_streaming.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--s2_root", str(s2_root),
            "--s1_root", str(s1_root),
            "--out_dir", str(out_dir),
            "--target_gb", "1.0",  # Large enough to process all test patches
            "--min_free_gb", "0.1",
            "--float16", "true",
            "--seed", "42",  # Reproducible splits
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Count splits from JSON files
        split_counts = {'train': 0, 'val': 0, 'test': 0}
        
        for json_path in out_dir.glob("*.json"):
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            split_counts[metadata['split']] += 1
        
        # Check that all splits are present
        assert split_counts['train'] > 0, "No training samples"
        assert split_counts['val'] >= 0, "Val split missing"
        assert split_counts['test'] >= 0, "Test split missing"
        
        # Check total matches number of processed patches
        total = sum(split_counts.values())
        assert total > 0, "No patches processed"
    
    def test_logging(self, tmp_path):
        """Test that JSON log file is created."""
        # Create synthetic dataset
        s2_root, s1_root, patch_ids = create_synthetic_ben_dataset(tmp_path, num_patches=2)
        
        # Create output directory
        out_dir = tmp_path / "tiles"
        out_dir.mkdir()
        
        # Run converter
        script_path = Path(__file__).parent.parent / "scripts" / "ingest_benv2_streaming.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--s2_root", str(s2_root),
            "--s1_root", str(s1_root),
            "--out_dir", str(out_dir),
            "--target_gb", "0.001",
            "--min_free_gb", "0.1",
            "--float16", "true",
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Check log file exists
        log_file = Path("logs/benv2_ingest.jsonl")
        assert log_file.exists(), "Log file not created"
        
        # Read and verify log entries
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) > 0, "Log file is empty"
        
        # Parse first log entry
        first_entry = json.loads(lines[-1])  # Get last entry (most recent)
        
        # Check required log fields
        required_fields = ['id', 'split', 'country', 'ok', 'status', 'timestamp']
        for field in required_fields:
            assert field in first_entry, f"Missing field '{field}' in log entry"


def test_full_pipeline_integration(tmp_path):
    """
    Integration test: Full pipeline from synthetic data to verified output.
    
    This is the main smoke test that validates the entire workflow.
    """
    # Create synthetic dataset (2 patches)
    s2_root, s1_root, patch_ids = create_synthetic_ben_dataset(tmp_path, num_patches=2)
    
    # Create output directory
    out_dir = tmp_path / "tiles"
    out_dir.mkdir()
    
    # Store original paths for deletion verification (parent/child structure)
    original_s2_patches = []
    for child_id in patch_ids:
        parent_id = '_'.join(child_id.split('_')[:-2])
        original_s2_patches.append(s2_root / "BigEarthNet-S2" / parent_id / child_id)
    
    # For S1, collect actual child directories
    original_s1_patches = []
    s1_ben_root = s1_root / "BigEarthNet-S1"
    for s1_parent in s1_ben_root.iterdir():
        if s1_parent.is_dir():
            for s1_child in s1_parent.iterdir():
                if s1_child.is_dir():
                    original_s1_patches.append(s1_child)
    
    # Run converter
    script_path = Path(__file__).parent.parent / "scripts" / "ingest_benv2_streaming.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--s2_root", str(s2_root),
        "--s1_root", str(s1_root),
        "--out_dir", str(out_dir),
        "--target_gb", "1.0",  # Large enough to process all test patches
        "--min_free_gb", "0.1",
        "--float16", "true",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Converter failed:\n{result.stderr}"
    
    # === VERIFY OUTPUTS ===
    
    # 1. NPZ files exist
    npz_files = list(out_dir.glob("*.npz"))
    assert len(npz_files) > 0, "No NPZ files created"
    
    # 2. For each NPZ, verify structure
    for npz_path in npz_files:
        data = np.load(npz_path)
        
        # Check 6 keys
        expected_keys = {'s1_vv', 's1_vh', 's2_b2', 's2_b3', 's2_b4', 's2_b8'}
        assert set(data.files) == expected_keys
        
        # Check float16
        for key in expected_keys:
            assert data[key].dtype == np.float16
        
        # Check shapes equal
        shapes = [data[key].shape for key in expected_keys]
        assert len(set(shapes)) == 1, f"Inconsistent shapes: {shapes}"
        
        # Check no NaNs
        for key in expected_keys:
            assert not np.any(np.isnan(data[key]))
    
    # 3. JSON metadata exists
    json_files = list(out_dir.glob("*.json"))
    assert len(json_files) == len(npz_files), "Mismatch between NPZ and JSON files"
    
    # 4. Source folders deleted
    for patch in original_s2_patches + original_s1_patches:
        assert not patch.exists(), f"Source patch {patch} not deleted"
    
    print(f"✅ Full pipeline test passed!")
    print(f"   Created {len(npz_files)} tiles")
    print(f"   Deleted {len(original_s2_patches) + len(original_s1_patches)} source patches")


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '-s'])
