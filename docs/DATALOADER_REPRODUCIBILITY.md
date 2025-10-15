# DataLoader Reproducibility & Windows Multiprocessing

This document explains our approach to deterministic DataLoader configuration and platform-specific handling of Windows multiprocessing issues.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Windows Multiprocessing Issues](#windows-multiprocessing-issues)
3. [Our Solution](#our-solution)
4. [Usage Examples](#usage-examples)
5. [Performance Considerations](#performance-considerations)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

**TL;DR**: On Windows, we automatically set `num_workers=0` for DataLoaders to ensure:
- ✅ Reliability (no multiprocessing crashes)
- ✅ Reproducibility (deterministic behavior)
- ✅ Simplicity (easier debugging)

```python
from axs_lib.dataloader_utils import create_dataloader

# This automatically handles Windows and reproducibility
loader = create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Will be set to 0 on Windows
    seed=42  # For reproducible shuffling
)
```

---

## Windows Multiprocessing Issues

### Why `num_workers > 0` is Problematic on Windows

Windows uses the **'spawn' method** for multiprocessing, unlike Linux/Mac which use **'fork'**. This fundamental difference causes several issues:

### 1. **Serialization Overhead**
- **Problem**: Each worker process must serialize/deserialize the entire dataset object
- **Impact**: Slow startup (can take 10-60 seconds per worker)
- **Why**: 'spawn' creates fresh Python processes from scratch, requiring full object serialization

```python
# On Linux (fork): Workers share memory with parent → fast startup
# On Windows (spawn): Workers get copies via pickle → slow startup
```

### 2. **CUDA Initialization Errors**
- **Problem**: CUDA cannot be initialized in spawned worker processes
- **Impact**: Crashes if dataset/transforms use GPU operations
- **Why**: CUDA context cannot be serialized and passed to spawned workers

```python
# This will crash on Windows with num_workers > 0:
class MyDataset(Dataset):
    def __init__(self):
        self.device = torch.device('cuda')  # ❌ CUDA in __init__
    
    def __getitem__(self, idx):
        return torch.randn(3, 224, 224).to(self.device)  # ❌ GPU operation
```

### 3. **Module Import Issues**
- **Problem**: Spawned processes must re-import all modules
- **Impact**: Slow startup and potential import errors
- **Why**: Fresh Python interpreter with clean import state

### 4. **Shared Memory Limitations**
- **Problem**: Windows doesn't support Unix shared memory well
- **Impact**: Inefficient data sharing between workers
- **Why**: No `/dev/shm` equivalent on Windows

### 5. **Non-Deterministic Behavior**
- **Problem**: Even with proper seeding, subtle non-determinism can occur
- **Impact**: Results differ across runs despite same seed
- **Why**: OS scheduling and process initialization order variations

### 6. **Pickle Compatibility**
- **Problem**: Complex objects (lambdas, local functions, CUDA tensors) can't be pickled
- **Impact**: `PicklingError` when trying to use workers
- **Example**:
  ```python
  # This will fail with num_workers > 0 on Windows:
  transforms = transforms.Compose([
      transforms.Lambda(lambda x: x * 2)  # ❌ Lambda not picklable
  ])
  ```

---

## Our Solution

### `axs_lib/dataloader_utils.py`

We provide a unified DataLoader creation utility that:

1. **Automatically detects Windows** and sets `num_workers=0`
2. **Configures reproducibility** with proper seeding
3. **Provides clear warnings** when adjusting settings
4. **Documents the rationale** for these decisions

### Key Functions

#### `create_dataloader()`
Main function to create platform-aware DataLoaders:

```python
loader = create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Adjusted automatically
    seed=42,  # For reproducibility
    force_zero_workers_on_windows=True  # Default: True
)
```

#### `get_recommended_num_workers()`
Returns safe worker count for current platform:

```python
num_workers = get_recommended_num_workers()
# Returns 0 on Windows
# Returns min(cpu_count, 8) on Linux/Mac
```

#### `print_dataloader_config()`
Debug utility to verify configuration:

```python
print_dataloader_config(
    num_workers=0,
    batch_size=32,
    seed=42,
    shuffle=True
)
```

---

## Usage Examples

### Basic Usage

```python
from axs_lib.dataloader_utils import create_dataloader

# Simple case - handles everything automatically
train_loader = create_dataloader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Will be 0 on Windows
    seed=42
)
```

### Training Script Integration

```python
from axs_lib.dataloader_utils import (
    create_dataloader,
    get_recommended_num_workers,
    print_dataloader_config
)
from axs_lib.reproducibility import set_seed

# Set global seed
set_seed(42, deterministic=True)

# Get platform-appropriate workers
safe_workers = get_recommended_num_workers()

# Print configuration for debugging
print_dataloader_config(
    num_workers=safe_workers,
    batch_size=32,
    seed=42,
    shuffle=True
)

# Create loaders
train_loader = create_dataloader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=safe_workers,
    seed=42
)

val_loader = create_dataloader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=safe_workers,
    seed=42
)
```

### Override Windows Handling (Not Recommended)

```python
# Force multi-worker loading on Windows (use with caution!)
loader = create_dataloader(
    dataset,
    batch_size=32,
    num_workers=4,
    force_zero_workers_on_windows=False  # ⚠️ May cause issues
)
```

---

## Performance Considerations

### When is `num_workers=0` Acceptable?

Single-threaded data loading is sufficient when:

1. **Small datasets** (< 10,000 samples)
   - Data loading overhead is negligible

2. **Fast dataloaders** (pre-processed/cached data)
   - Already optimized, workers won't help much

3. **GPU-bound training** (data loading not the bottleneck)
   - GPU compute takes >> data loading time

4. **Complex transforms** (CPU-intensive preprocessing)
   - Python GIL limits multi-threading benefits anyway

### Performance Comparison

| Configuration | Linux (fork) | Windows (spawn) |
|---------------|--------------|-----------------|
| `num_workers=0` | Baseline | Baseline |
| `num_workers=4` | ~2-3x faster | Often slower! |
| Startup time | < 1 sec | 10-60 sec |
| Reliability | ✅ Excellent | ⚠️ Problematic |
| Reproducibility | ✅ Perfect | ⚠️ Issues |

### Optimization Tips

If single-threaded loading is too slow:

1. **Pre-process data offline**
   ```bash
   python scripts/preprocess_tiles.py --data_dir data/ --output_dir data_preprocessed/
   ```

2. **Use disk caching**
   ```python
   from joblib import Memory
   memory = Memory("./cache", verbose=0)
   
   @memory.cache
   def load_tile(path):
       return np.load(path)
   ```

3. **Reduce image resolution**
   ```python
   transforms = transforms.Resize((224, 224))  # Instead of (512, 512)
   ```

4. **Use faster data formats**
   ```python
   # Faster: HDF5, LMDB
   # Slower: Individual PNG files
   ```

---

## Reproducibility Configuration

### Complete Reproducibility Setup

```python
from axs_lib.reproducibility import set_seed
from axs_lib.dataloader_utils import create_dataloader

# 1. Set global seeds
seed_info = set_seed(42, deterministic=True, benchmark=False)

# 2. Create reproducible data splits
from torch.utils.data import random_split

generator = torch.Generator().manual_seed(42)
train_set, val_set = random_split(
    full_dataset,
    [0.8, 0.2],
    generator=generator
)

# 3. Create deterministic dataloaders
train_loader = create_dataloader(
    train_set,
    batch_size=32,
    shuffle=True,
    num_workers=0,  # Explicit for clarity
    seed=42  # Ensures deterministic shuffling
)

val_loader = create_dataloader(
    val_set,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    seed=42
)

# 4. Verify reproducibility
from axs_lib.reproducibility import verify_reproducibility
if verify_reproducibility(verbose=True):
    print("✓ Reproducibility verified!")
```

---

## Troubleshooting

### Issue: "DataLoader worker exited unexpectedly"

**Cause**: Windows multiprocessing issue

**Solution**:
```python
# Set num_workers=0
loader = create_dataloader(dataset, num_workers=0)
```

### Issue: "RuntimeError: CUDA error in DataLoader worker"

**Cause**: CUDA operations in spawned workers

**Solution**:
```python
# Option 1: Use num_workers=0
loader = create_dataloader(dataset, num_workers=0)

# Option 2: Move CUDA ops to main process
class Dataset:
    def __getitem__(self, idx):
        # Load on CPU, move to GPU in training loop
        return torch.randn(3, 224, 224)  # CPU tensor
```

### Issue: "PicklingError: Can't pickle lambda"

**Cause**: Unpicklable objects in dataset

**Solution**:
```python
# Bad: Lambda function
transform = transforms.Lambda(lambda x: x * 2)

# Good: Named function
def scale_by_two(x):
    return x * 2

transform = transforms.Lambda(scale_by_two)
```

### Issue: "Results not reproducible on Windows"

**Cause**: Multiple workers with non-deterministic behavior

**Solution**:
```python
# Use num_workers=0 for perfect reproducibility
loader = create_dataloader(
    dataset,
    num_workers=0,  # Ensures deterministic behavior
    seed=42
)
```

### Issue: "Slow data loading"

**Diagnosis**:
```python
import time

# Time a single batch
start = time.time()
batch = next(iter(loader))
print(f"Batch loading time: {time.time() - start:.2f}s")
```

**Solutions**:
1. Pre-process data offline
2. Use faster data formats (HDF5, LMDB)
3. Reduce image size
4. Cache preprocessed samples

---

## Platform Detection

The utilities automatically detect your platform:

```python
from axs_lib.dataloader_utils import is_windows

if is_windows():
    print("Running on Windows - using safe configuration")
else:
    print("Running on Linux/Mac - multi-worker loading available")
```

### Testing Platform Detection

```bash
# Run platform detection test
python -m axs_lib.dataloader_utils
```

Expected output on Windows:
```
================================================================================
DataLoader Utilities - Platform Information
================================================================================

Platform:              Windows
Platform Detail:       win32
Is Windows:            True
Recommended Workers:   0

Safe num_workers (with Windows handling):
  Requested:  0  →  Safe:  0
  Requested:  1  →  Safe:  0
  Requested:  2  →  Safe:  0
  Requested:  4  →  Safe:  0
  Requested:  8  →  Safe:  0
```

---

## Integration with Existing Code

### Before (Manual Configuration)

```python
from torch.utils.data import DataLoader

# Problematic on Windows
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Will cause issues on Windows
    pin_memory=True
)
```

### After (Platform-Aware)

```python
from axs_lib.dataloader_utils import create_dataloader

# Automatically handles Windows
train_loader = create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Adjusted automatically on Windows
    seed=42  # Reproducible
)
```

---

## References

### PyTorch Documentation
- [Multiprocessing Best Practices](https://pytorch.org/docs/stable/notes/multiprocessing.html)
- [DataLoader Documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- [Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)

### Related Files
- `axs_lib/dataloader_utils.py` - Platform-aware DataLoader utilities
- `axs_lib/reproducibility.py` - Seed management and reproducibility
- `scripts/train_stage1.py` - Example usage in Stage 1 training
- `scripts/train_stage2.py` - Example usage in Stage 2 training

---

## Summary

**On Windows**:
- ✅ Use `num_workers=0` by default (automatic)
- ✅ Ensures reliability and reproducibility
- ✅ Slight performance trade-off, but worth it for stability

**On Linux/Mac**:
- ✅ Use `num_workers > 0` safely
- ✅ Faster data loading with multi-processing
- ✅ Still deterministic with proper seeding

**Always**:
- ✅ Use `create_dataloader()` for automatic platform handling
- ✅ Set `seed` parameter for reproducibility
- ✅ Test on your target platform before deployment

---

*For questions or issues, see the troubleshooting section or contact the Axion-Sat development team.*
