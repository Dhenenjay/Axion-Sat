# DataLoader Utilities

Quick reference for `axs_lib.dataloader_utils` module.

## Quick Start

```python
from axs_lib.dataloader_utils import create_dataloader

# Automatically handles Windows and reproducibility
loader = create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Will be 0 on Windows
    seed=42
)
```

## Why This Exists

**Problem**: Windows uses 'spawn' for multiprocessing (not 'fork'), causing:
- ❌ Multiprocessing crashes
- ❌ CUDA initialization errors
- ❌ Non-deterministic behavior
- ❌ Slow worker startup (10-60 seconds)

**Solution**: Automatically set `num_workers=0` on Windows

## Key Functions

### `create_dataloader()`
Platform-aware DataLoader creation with automatic reproducibility configuration.

### `get_recommended_num_workers()`
Returns safe worker count: 0 on Windows, min(cpu_count, 8) on Linux/Mac.

### `is_windows()`
Detect if running on Windows platform.

### `print_dataloader_config()`
Print configuration for debugging.

## Usage in Training Scripts

Both `scripts/train_stage1.py` and `scripts/train_stage2.py` use these utilities:

```python
from axs_lib.dataloader_utils import (
    create_dataloader,
    get_recommended_num_workers,
    print_dataloader_config
)

# Get safe worker count
safe_workers = get_recommended_num_workers()

# Create deterministic loader
train_loader = create_dataloader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=safe_workers,
    seed=42
)
```

## Testing

```bash
# Test platform detection
python -m axs_lib.dataloader_utils
```

## Documentation

See `docs/DATALOADER_REPRODUCIBILITY.md` for:
- Detailed explanation of Windows multiprocessing issues
- Performance considerations
- Troubleshooting guide
- Complete examples

## Related Modules

- `axs_lib/reproducibility.py` - Global seed management
- `axs_lib/dataloader_utils.py` - This module
- `docs/DATALOADER_REPRODUCIBILITY.md` - Full documentation
