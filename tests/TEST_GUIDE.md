# Test Guide for BigEarthNet v2 Converter

## 📋 Test Overview

**File:** `tests/test_benv2_ingest_smoke.py`

Comprehensive smoke tests for the BigEarthNet v2 streaming converter that:
1. Creates synthetic S1/S2 patch pairs
2. Runs the converter
3. Verifies output structure and correctness
4. Confirms source deletion

## 🚀 Running Tests

### Run all tests
```powershell
pytest tests/test_benv2_ingest_smoke.py -v
```

### Run specific test
```powershell
pytest tests/test_benv2_ingest_smoke.py::test_full_pipeline_integration -v
```

### Run with output visible
```powershell
pytest tests/test_benv2_ingest_smoke.py -v -s
```

### Run test class
```powershell
pytest tests/test_benv2_ingest_smoke.py::TestBENv2IngestSmoke -v
```

## 🧪 Test Suite

### Main Integration Test
**`test_full_pipeline_integration`** ⭐
- Creates 2 synthetic patches
- Runs full converter pipeline
- Verifies all outputs
- Confirms source deletion
- **This is the primary smoke test**

### Individual Component Tests

1. **`test_synthetic_dataset_creation`**
   - Verifies synthetic data generator works
   - Checks S1/S2 patch structure

2. **`test_converter_basic_run`**
   - Runs converter without errors
   - Checks basic output creation

3. **`test_npz_structure_and_dtype`**
   - Verifies 6 keys present: `s1_vv`, `s1_vh`, `s2_b2`, `s2_b3`, `s2_b4`, `s2_b8`
   - Confirms `float16` dtype
   - Checks shape consistency
   - Validates normalization [0, 1]
   - Tests for NaNs

4. **`test_json_metadata`**
   - Checks JSON files created
   - Validates metadata structure
   - Verifies required fields

5. **`test_source_deletion`**
   - Confirms original patches exist before
   - Verifies patches deleted after success
   - **Tests the streaming deletion feature**

6. **`test_split_distribution`**
   - Tests train/val/test split assignment
   - Verifies split ratios

7. **`test_logging`**
   - Checks JSONL log file creation
   - Validates log entry structure

## ✅ Expected Output

```
tests/test_benv2_ingest_smoke.py::test_full_pipeline_integration PASSED
✅ Full pipeline test passed!
   Created 2 tiles
   Deleted 4 source patches

tests/test_benv2_ingest_smoke.py::TestBENv2IngestSmoke::test_synthetic_dataset_creation PASSED
tests/test_benv2_ingest_smoke.py::TestBENv2IngestSmoke::test_converter_basic_run PASSED
tests/test_benv2_ingest_smoke.py::TestBENv2IngestSmoke::test_npz_structure_and_dtype PASSED
tests/test_benv2_ingest_smoke.py::TestBENv2IngestSmoke::test_json_metadata PASSED
tests/test_benv2_ingest_smoke.py::TestBENv2IngestSmoke::test_source_deletion PASSED
tests/test_benv2_ingest_smoke.py::TestBENv2IngestSmoke::test_split_distribution PASSED
tests/test_benv2_ingest_smoke.py::TestBENv2IngestSmoke::test_logging PASSED

======================== 8 passed in 15.2s ========================
```

## 📊 What Gets Tested

### Input Generation
- ✅ Synthetic S2 patches with 4 bands (B02, B03, B04, B08)
- ✅ Synthetic S1 patches with 2 polarizations (VV, VH)
- ✅ Realistic BigEarthNet directory structure
- ✅ Valid GeoTIFF files with proper metadata

### Converter Behavior
- ✅ Successfully finds and pairs patches
- ✅ Processes S1/S2 data correctly
- ✅ Aligns S1 to S2 grid
- ✅ Normalizes to [0, 1] range
- ✅ Saves float16 NPZ files
- ✅ Creates JSON metadata
- ✅ Deletes source patches after success
- ✅ Handles errors gracefully

### Output Validation
- ✅ NPZ contains exactly 6 keys
- ✅ All arrays are float16
- ✅ Shapes are consistent
- ✅ No NaN values
- ✅ Values in [0, 1] range
- ✅ JSON metadata complete
- ✅ Train/val/test splits assigned

## 🔧 Dependencies

The tests require:
```powershell
pip install pytest numpy rasterio scipy
```

## 🐛 Troubleshooting

### Test fails with "No NPZ files created"
- Check converter script path is correct
- Verify synthetic data generation works
- Run with `-s` flag to see converter output

### Import errors
```powershell
# Make sure you're in the project root
cd C:\Users\Dhenenjay\Axion-Sat

# Activate virtual environment
.venv\Scripts\activate

# Run tests
pytest tests/test_benv2_ingest_smoke.py -v
```

### Slow tests
- Tests create temp files and run subprocess
- Expect ~15-30 seconds for full suite
- Individual tests: 2-5 seconds each

### Windows path issues
- Tests handle Windows paths correctly
- Uses `pathlib.Path` for cross-platform compatibility

## 📝 Test Data

Each test uses `tmp_path` fixture (automatic cleanup):
- Synthetic S2 patches: 10×10 pixels
- Synthetic S1 patches: 10×10 pixels
- Total size per patch: ~50 KB
- Test completes in seconds

## 🎯 Quick Verification

Run just the main integration test:
```powershell
pytest tests/test_benv2_ingest_smoke.py::test_full_pipeline_integration -v -s
```

This single test validates:
- ✅ NPZ creation with 6 keys
- ✅ float16 dtype
- ✅ Shape consistency
- ✅ No NaNs
- ✅ Source deletion

## ✨ Adding More Tests

To add new tests, follow this pattern:

```python
def test_my_feature(tmp_path):
    """Test description."""
    # Create synthetic data
    s2_root, s1_root, patch_ids = create_synthetic_ben_dataset(tmp_path, num_patches=2)
    
    # Setup
    out_dir = tmp_path / "tiles"
    out_dir.mkdir()
    
    # Run converter
    script_path = Path(__file__).parent.parent / "scripts" / "ingest_benv2_streaming.py"
    cmd = [sys.executable, str(script_path), ...]
    subprocess.run(cmd, check=True)
    
    # Assert something
    assert condition, "Error message"
```

## 🚀 Ready to Test!

Run the tests to verify the converter works:

```powershell
.venv\Scripts\activate
pytest tests/test_benv2_ingest_smoke.py -v
```

Good luck! 🎉
