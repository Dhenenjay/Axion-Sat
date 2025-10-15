# Windows Path Safety Guide

This document outlines our approach to Windows-safe file I/O operations in the Axion-Sat project.

## Summary

All file I/O operations in this project use `pathlib.Path` for cross-platform compatibility. This ensures:
- ✅ Windows backslash handling (automatic)
- ✅ Forward slash compatibility  
- ✅ No hardcoded path separators
- ✅ Cross-platform path operations

## Quick Reference

### ✅ DO: Use pathlib.Path

```python
from pathlib import Path

# Good: Path operations
data_dir = Path("data/tiles")
file_path = data_dir / "sample.npz"

# Good: Path construction
output_dir = Path("outputs") / "checkpoints" / "stage1"
output_dir.mkdir(parents=True, exist_ok=True)

# Good: File listing
npz_files = list(data_dir.glob("*.npz"))
for file in data_dir.rglob("*.npz"):
    process(file)

# Good: Path normalization
absolute_path = file_path.resolve()
posix_path = file_path.as_posix()  # Forward slashes
```

### ❌ DON'T: Use string concatenation

```python
# Bad: String concatenation
path = "data" + "/" + "tiles"  # Use Path / operator instead

# Bad: Hardcoded backslashes
path = "C:\\Users\\data\\file.txt"  # Use forward slashes or Path

# Bad: os.path (prefer pathlib)
import os
path = os.path.join("data", "tiles")  # Use Path / operator
```

## Utilities

### `axs_lib.path_utils`

Comprehensive path utilities for Windows-safe operations:

```python
from axs_lib.path_utils import (
    ensure_path,      # Convert to Path
    safe_mkdir,       # Create directories
    list_files,       # List files with patterns
    sanitize_filename # Clean invalid characters
)

# Example usage
path = ensure_path("data/tiles/sample.npz")
safe_mkdir("outputs/checkpoints")
files = list_files("data", pattern="*.npz", recursive=True)
```

### `tools/verify_windows_paths.py`

Automated verification tool:

```bash
# Check all Python files
python tools/verify_windows_paths.py

# Check specific directory
python tools/verify_windows_paths.py --dir axs_lib

# Quiet mode (summary only)
python tools/verify_windows_paths.py --quiet
```

## Windows-Specific Issues

### 1. Path Separators
- **Windows**: Uses backslash `\` 
- **Unix**: Uses forward slash `/`
- **Solution**: `pathlib.Path` handles both automatically

```python
# These all work on Windows and Unix
Path("data/tiles/sample.npz")
Path("data") / "tiles" / "sample.npz"
Path(r"C:\Users\data")  # Raw string for backslashes
```

### 2. Reserved Names
Windows reserves these names (case-insensitive):
- `CON`, `PRN`, `AUX`, `NUL`
- `COM1`-`COM9`, `LPT1`-`LPT9`

```python
from axs_lib.path_utils import is_safe_filename, sanitize_filename

# Check filename safety
is_safe_filename("CON.txt")  # False on Windows

# Sanitize automatically
sanitize_filename("CON.txt")  # Returns "CON_.txt"
```

### 3. Invalid Characters
Windows doesn't allow: `< > : " | ? *`

```python
# Automatic sanitization
sanitize_filename("data<test>.npz")  # Returns "data_test_.npz"
```

### 4. Path Length Limits
- Windows: 260 characters (MAX_PATH)
- Can be extended with `\\?\` prefix
- Use short, descriptive names

### 5. Case Sensitivity
- Windows: Case-insensitive (DATA.txt == data.txt)
- Unix: Case-sensitive (DATA.txt ≠ data.txt)
- Always use consistent casing

## Best Practices

### 1. Always Import Path

```python
from pathlib import Path

# At the start of every file that handles paths
```

### 2. Use Forward Slashes in Literals

```python
# Good: Forward slashes work everywhere
data_dir = Path("data/tiles")

# Acceptable: Raw strings for Windows paths
windows_path = Path(r"C:\Users\data")

# Bad: Regular strings with backslashes
path = "data\\tiles"  # Requires escaping
```

### 3. Use Path Operators

```python
# Good: / operator
output_path = base_dir / "checkpoints" / "model.pt"

# Bad: String concatenation
output_path = base_dir + "/checkpoints/" + "model.pt"
```

### 4. Convert Early

```python
def process_data(data_path: Union[str, Path]):
    # Convert to Path immediately
    data_path = Path(data_path)
    
    # Now all Path methods work
    if not data_path.exists():
        raise FileNotFoundError(f"Not found: {data_path}")
```

### 5. Use Utilities for Common Operations

```python
from axs_lib.path_utils import safe_mkdir, list_files

# Instead of os.makedirs
safe_mkdir("outputs/checkpoints", parents=True, exist_ok=True)

# Instead of os.listdir + filtering
npz_files = list_files("data", pattern="*.npz", recursive=True)
```

## Verification

### Manual Check

```bash
# Verify all Python files
python tools/verify_windows_paths.py

# Expected output
# Files checked: X
# Files with issues: 0
# ✓ PASSED: All files use Windows-safe path handling
```

### CI Integration

Add to your CI/CD pipeline:

```yaml
# .github/workflows/ci.yml
- name: Verify Windows path safety
  run: python tools/verify_windows_paths.py
```

## Common Patterns

### Loading Data

```python
from pathlib import Path

def load_tile(tile_path: Union[str, Path]) -> np.ndarray:
    """Load tile from NPZ file."""
    tile_path = Path(tile_path)  # Ensure Path
    
    if not tile_path.exists():
        raise FileNotFoundError(f"Tile not found: {tile_path}")
    
    data = np.load(tile_path)
    return data
```

### Saving Results

```python
def save_checkpoint(model, output_dir: Union[str, Path], name: str):
    """Save model checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = output_dir / f"{name}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    
    print(f"Saved checkpoint: {checkpoint_path}")
```

### Iterating Files

```python
from pathlib import Path

def process_all_tiles(data_dir: Union[str, Path]):
    """Process all NPZ tiles in directory."""
    data_dir = Path(data_dir)
    
    # Recursive search
    for tile_path in data_dir.rglob("*.npz"):
        print(f"Processing: {tile_path.name}")
        process_tile(tile_path)
```

### Configuration Paths

```python
# Good: POSIX paths in config files
config = {
    'data_dir': 'data/tiles/benv2_catalog',  # Works on all platforms
    'output_dir': 'outputs/stage2',
    'checkpoint': 'weights/stage1_best.pt'
}

# Convert to Path when loading
data_dir = Path(config['data_dir'])
```

## Testing on Windows

### Virtual Paths
```python
# Use forward slashes in tests - works everywhere
test_path = Path("tests/fixtures/sample.npz")
```

### Temp Directories
```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)  # Cross-platform temp dir
    output_file = tmp_path / "output.npz"
    # Use tmp_path...
```

### Mock Paths
```python
from unittest.mock import patch
from pathlib import Path

@patch('pathlib.Path.exists')
def test_file_loading(mock_exists):
    mock_exists.return_value = True
    # Test with Path objects
```

## Migration Guide

### From os.path to pathlib

```python
# Before
import os
path = os.path.join("data", "tiles", "sample.npz")
exists = os.path.exists(path)
dirname = os.path.dirname(path)

# After
from pathlib import Path
path = Path("data") / "tiles" / "sample.npz"
exists = path.exists()
dirname = path.parent
```

### From string concatenation

```python
# Before
base = "data"
path = base + "/" + "tiles" + "/" + filename

# After
from pathlib import Path
base = Path("data")
path = base / "tiles" / filename
```

## Related Files

- `axs_lib/path_utils.py` - Path utilities module
- `tools/verify_windows_paths.py` - Verification tool
- `tools/color_checker.py` - Example usage
- `tools/panel_v2_vs_v1.py` - Example usage

## References

- [pathlib Documentation](https://docs.python.org/3/library/pathlib.html)
- [Windows Path Limitations](https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file)
- [Cross-Platform Python](https://docs.python.org/3/library/os.path.html)

---

*Last updated: 2025-10-14*
*Verification status: ✓ PASSED (0 errors)*
