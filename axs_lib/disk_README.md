# Disk Utility Module (`axs_lib/disk.py`)

Cross-platform disk space monitoring and management utilities.

## ✅ Functions

### `get_free_gb(path) -> float`
Get available free disk space in GB.

```python
from axs_lib.disk import get_free_gb

free = get_free_gb('.')
print(f"Free space: {free:.2f} GB")
# Output: Free space: 39.44 GB
```

### `wait_for_space(path, min_free_gb, poll_s=30, timeout_s=None) -> float`
Wait until sufficient disk space is available.

```python
from axs_lib.disk import wait_for_space

# Wait until at least 50 GB is free (checks every 30 seconds)
free = wait_for_space('/data/output', min_free_gb=50.0)
print(f"Ready to proceed with {free:.2f} GB free")

# With timeout (raises TimeoutError if exceeded)
try:
    free = wait_for_space('/data', min_free_gb=100, poll_s=60, timeout_s=1800)
except TimeoutError:
    print("Timeout waiting for disk space")
```

**Output when waiting:**
```
⚠️  INSUFFICIENT DISK SPACE
   Path: C:\Users\Dhenenjay\Axion-Sat
   Current: 39.44 GB free
   Required: 50.00 GB
   Need: 10.56 GB more

   Waiting for space to free up...
   Checking every 30s (Press Ctrl+C to abort)
   [2m] Still 10.56 GB short (39.44 / 50.00 GB)...
```

### `get_drive_info(path) -> dict`
Get detailed drive/filesystem information.

```python
from axs_lib.disk import get_drive_info

info = get_drive_info('.')
print(f"Total:  {info['total_gb']:.2f} GB")
print(f"Used:   {info['used_gb']:.2f} GB ({info['percent_used']:.1f}%)")
print(f"Free:   {info['free_gb']:.2f} GB")

# Returns:
# {
#     'total_gb': 475.66,
#     'used_gb': 436.22,
#     'free_gb': 39.44,
#     'percent_used': 91.7,
#     'path': 'C:\\Users\\Dhenenjay\\Axion-Sat'
# }
```

### `has_space(path, required_gb) -> bool`
Quick check if sufficient space is available.

```python
from axs_lib.disk import has_space

if has_space('/output', 50.0):
    print("OK to proceed")
else:
    print("Need more space")
```

### `format_bytes(bytes_value) -> str`
Format bytes into human-readable string.

```python
from axs_lib.disk import format_bytes

print(format_bytes(1536))          # "1.5 KB"
print(format_bytes(1073741824))    # "1.0 GB"
```

## 🚀 Command Line Usage

### Check disk space
```powershell
python axs_lib/disk.py .
```

**Output:**
```
Disk space for: C:\Users\Dhenenjay\Axion-Sat
  Total:  475.66 GB
  Used:   436.22 GB (91.7%)
  Free:   39.45 GB
```

### Wait for specific amount of space
```powershell
python axs_lib/disk.py . --wait 50
```

**Output:**
```
Waiting for 50.00 GB free space on .

⚠️  INSUFFICIENT DISK SPACE
   Path: C:\Users\Dhenenjay\Axion-Sat
   Current: 39.44 GB free
   Required: 50.00 GB
   Need: 10.56 GB more

   Waiting for space to free up...
   Checking every 30s (Press Ctrl+C to abort)
```

### Custom polling interval
```powershell
python axs_lib/disk.py . --wait 50 --poll 60
```

## 📊 Use Cases

### 1. Pre-flight check before data processing
```python
from axs_lib.disk import has_space

if not has_space('/output', required_gb=50.0):
    print("⚠️  Insufficient disk space!")
    print("Please free up space before continuing.")
    exit(1)

# Proceed with processing...
```

### 2. Auto-pause during long-running jobs
```python
from axs_lib.disk import wait_for_space

for batch in data_batches:
    # Wait if disk is getting full
    wait_for_space('/output', min_free_gb=30.0, poll_s=30)
    
    # Process batch
    process_batch(batch)
```

### 3. Monitor disk usage during streaming conversion
```python
from axs_lib.disk import get_free_gb

for item in large_dataset:
    free_gb = get_free_gb('/output')
    
    if free_gb < 30:
        print(f"⚠️  Low disk space: {free_gb:.2f} GB")
        # Trigger cleanup or pause
    
    process_item(item)
```

### 4. Integration with BigEarthNet converter
```python
from axs_lib.disk import wait_for_space, get_free_gb

# Before processing each patch
free_gb = wait_for_space(out_dir, min_free_gb=30.0)

# Process and save tile
save_tile(data, out_path)

# Check space after
new_free = get_free_gb(out_dir)
print(f"Freed: {free_gb - new_free:.2f} GB")
```

## 🔧 Platform Support

- ✅ **Windows**: Uses `GetDiskFreeSpaceExW` (respects disk quotas)
- ✅ **Linux/Unix**: Uses `os.statvfs`
- ✅ **macOS**: Uses `os.statvfs`

## ⚠️ Important Notes

1. **Path handling**: If path doesn't exist, uses parent directory
2. **Free space**: Returns space available to current user (respects quotas)
3. **Polling**: `wait_for_space` blocks the current thread
4. **Interruption**: Can be interrupted with Ctrl+C (raises `KeyboardInterrupt`)
5. **Timeout**: Optional timeout raises `TimeoutError`

## 🧪 Testing

```python
# Test all functions
from axs_lib.disk import get_free_gb, get_drive_info, has_space

# Basic checks
assert get_free_gb('.') > 0
assert get_drive_info('.')['free_gb'] > 0
assert isinstance(has_space('.', 1.0), bool)

print("✅ All tests passed!")
```

## 📝 Examples in Real Scripts

### Example 1: Simple guard
```python
from axs_lib.disk import has_space
import sys

if not has_space('data/output', 50.0):
    print("ERROR: Need at least 50 GB free")
    sys.exit(1)
```

### Example 2: Progress monitoring
```python
from axs_lib.disk import get_free_gb

initial = get_free_gb('.')
# ... do processing ...
final = get_free_gb('.')

print(f"Disk space used: {initial - final:.2f} GB")
```

### Example 3: Conditional cleanup
```python
from axs_lib.disk import get_free_gb

def cleanup_if_needed(path, threshold_gb=30.0):
    if get_free_gb(path) < threshold_gb:
        print("Running cleanup...")
        # Remove temporary files
        cleanup_temp_files(path)
```

## ✅ Ready to Use!

The module is fully functional and tested. Import it in your scripts:

```python
from axs_lib.disk import get_free_gb, wait_for_space
```
