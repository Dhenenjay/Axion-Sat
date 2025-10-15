# Reproducibility in Axion-Sat

## Overview

Axion-Sat implements comprehensive reproducibility features to ensure that training runs can be exactly reproduced. This is critical for scientific research, debugging, and comparing experiments.

## Key Features

✅ **All random sources seeded**: torch, numpy, random, CUDA  
✅ **Deterministic CUDA operations**: CUDNN determinism enabled  
✅ **Reproducible data splitting**: Fixed generator for train/val splits  
✅ **Reproducible data loading**: Worker seeds set deterministically  
✅ **Seed saved in checkpoints**: Every checkpoint includes full seed state  
✅ **Verification tools**: Built-in reproducibility testing

## Quick Start

### Basic Usage

```python
from axs_lib.reproducibility import set_seed

# Set all seeds for reproducible training
set_seed(42, deterministic=True, benchmark=False)

# Your training code here...
```

### Verify Reproducibility

```bash
# Test that reproducibility is working
python axs_lib/reproducibility.py --verify --seed 42
```

Output:
```
Testing reproducibility...
  Running 10 iterations...
  ✓ All iterations produced identical results
  ✓ Reproducibility verified!
```

## Implementation Details

### What Gets Seeded

1. **Python's `random` module**
   ```python
   random.seed(42)
   ```

2. **NumPy's random state**
   ```python
   np.random.seed(42)
   ```

3. **PyTorch CPU RNG**
   ```python
   torch.manual_seed(42)
   ```

4. **PyTorch CUDA RNG** (all devices)
   ```python
   torch.cuda.manual_seed_all(42)
   ```

5. **CUDNN determinism**
   ```python
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```

6. **PyTorch deterministic algorithms**
   ```python
   torch.use_deterministic_algorithms(True)
   ```

7. **Environment variables**
   ```python
   os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
   ```

### Training Script Integration

The `train_stage1.py` script automatically:

1. **Sets seed at startup**
   ```python
   seed_info = set_seed(args.seed, deterministic=True, benchmark=False)
   ```

2. **Creates reproducible data split**
   ```python
   generator = create_reproducible_generator(args.seed)
   train_set, val_set = random_split(dataset, [0.9, 0.1], generator=generator)
   ```

3. **Seeds DataLoader workers**
   ```python
   DataLoader(dataset, worker_init_fn=seed_worker, ...)
   ```

4. **Saves seed in checkpoints**
   ```python
   checkpoint = {'model': model.state_dict(), ...}
   add_seed_to_checkpoint(checkpoint, seed_info)
   torch.save(checkpoint, 'model.pt')
   ```

## Checkpoint Metadata

Every checkpoint saved by the training script includes:

### Seed Info
```python
{
    'seed': 42,
    'deterministic': True,
    'benchmark': False,
    'cuda_available': True,
    'cuda_device_count': 1,
    'cudnn_deterministic': True,
    'cudnn_benchmark': False,
    'use_deterministic_algorithms': True
}
```

### RNG State
```python
{
    'random_state': <Python random state>,
    'numpy_state': <NumPy random state>,
    'torch_rng_state': <PyTorch CPU RNG state>,
    'cuda_rng_state': <CUDA RNG state>
}
```

## Loading Checkpoints

### Validate Seed

```python
from axs_lib.reproducibility import validate_checkpoint_seed

checkpoint = torch.load('model.pt')

# Check seed matches
if validate_checkpoint_seed(checkpoint, expected_seed=42):
    print("✓ Seed validated!")
else:
    print("⚠️ Warning: Seed mismatch!")
```

### Restore RNG State

```python
from axs_lib.reproducibility import restore_seed_state

checkpoint = torch.load('model.pt')

# Restore to exact RNG state
if 'rng_state' in checkpoint:
    restore_seed_state(checkpoint['rng_state'])
    print("✓ RNG state restored")
```

## Performance Considerations

### Determinism vs Speed

Enabling determinism **slows training by 10-30%** due to:
- Disabling CUDNN auto-tuning (benchmark mode)
- Using deterministic implementations (slower than optimized non-deterministic ones)
- Additional synchronization requirements

### Configuration Options

**Full reproducibility** (slower):
```python
set_seed(42, deterministic=True, benchmark=False)
```

**Faster training** (non-reproducible):
```python
set_seed(42, deterministic=False, benchmark=True)
```

**Partial reproducibility** (balanced):
```python
set_seed(42, deterministic=False, benchmark=False)
# Seeds set, but some CUDA operations may be non-deterministic
```

## Reproducibility Guarantees

### ✅ Guaranteed Reproducible

With `deterministic=True`:
- Model initialization
- Data shuffling and splitting
- Forward/backward passes
- Gradient updates
- Parameter updates
- Augmentation transforms

### ⚠️ May Vary

Even with determinism:
- **Hardware differences**: Different GPU architectures may produce slightly different floating-point results
- **CUDA versions**: Different CUDA versions may have different implementations
- **PyTorch versions**: Different PyTorch versions may have different defaults
- **System libraries**: BLAS/LAPACK implementations may differ

### ❌ Not Reproducible

- **Multi-process DataLoader** without `worker_init_fn` (fixed in our implementation)
- **Asynchronous operations** without proper synchronization
- **Non-deterministic CUDA kernels** when `deterministic=False`

## Best Practices

### 1. Always Specify Seed

```bash
# Good: Explicit seed
python scripts/train_stage1.py --data-dir tiles/ --output-dir runs/exp1/ --seed 42

# Bad: Relies on default (42)
python scripts/train_stage1.py --data-dir tiles/ --output-dir runs/exp1/
```

### 2. Document Seed in Experiments

```
experiments/
├── exp1_seed42/          # Clear seed in directory name
│   ├── checkpoints/
│   └── logs/
├── exp2_seed123/
│   ├── checkpoints/
│   └── logs/
```

### 3. Validate Before Comparing

When comparing two training runs:
```python
ckpt1 = torch.load('run1/model.pt')
ckpt2 = torch.load('run2/model.pt')

# Ensure same seed was used
assert ckpt1['seed_info']['seed'] == ckpt2['seed_info']['seed']
```

### 4. Test Reproducibility

Before long training runs:
```bash
# Quick test with 100 steps
python scripts/train_stage1.py --data-dir tiles/ --output-dir test1/ --seed 42 --steps 100 --max-tiles 10
python scripts/train_stage1.py --data-dir tiles/ --output-dir test2/ --seed 42 --steps 100 --max-tiles 10

# Compare checkpoints
python tools/compare_checkpoints.py test1/checkpoints/checkpoint_step_100.pt test2/checkpoints/checkpoint_step_100.pt
```

### 5. Save Environment Info

```bash
# Save environment for reproduction
pip freeze > requirements_frozen.txt
python --version > python_version.txt
nvidia-smi > gpu_info.txt
```

## Verification

### Automated Testing

```python
from axs_lib.reproducibility import set_seed, verify_reproducibility

# Set seed
set_seed(42, deterministic=True)

# Verify it works
assert verify_reproducibility(num_iterations=10)
```

### Manual Testing

```bash
# Train same config twice
python scripts/train_stage1.py --seed 42 --steps 1000 --output-dir run1/
python scripts/train_stage1.py --seed 42 --steps 1000 --output-dir run2/

# Compare final checkpoints
diff <(python -c "import torch; print(torch.load('run1/checkpoints/checkpoint_step_1000.pt')['model_state_dict'])") \
     <(python -c "import torch; print(torch.load('run2/checkpoints/checkpoint_step_1000.pt')['model_state_dict'])")
```

Should show no differences!

## Troubleshooting

### Results Still Vary

**Problem**: Same seed produces different results

**Solutions**:
1. Check deterministic mode is enabled:
   ```python
   import torch
   print(f"CUDNN deterministic: {torch.backends.cudnn.deterministic}")
   print(f"CUDNN benchmark: {torch.backends.cudnn.benchmark}")
   ```

2. Verify CUDA version matches:
   ```bash
   nvcc --version  # Should be same across machines
   ```

3. Check PyTorch version:
   ```python
   import torch
   print(torch.__version__)  # Should be identical
   ```

### Training Very Slow

**Problem**: Training 30% slower with determinism

**Solution**: This is expected! Options:
1. **Accept slowdown** for reproducibility
2. **Disable determinism** for faster training:
   ```python
   set_seed(42, deterministic=False, benchmark=True)
   ```
3. **Use faster hardware** to offset slowdown

### Checkpoint Missing Seed

**Problem**: Old checkpoint doesn't have seed info

**Solution**:
```python
checkpoint = torch.load('old_model.pt')

if 'seed_info' not in checkpoint:
    print("⚠️ Old checkpoint, seed unknown")
    print("⚠️ Cannot guarantee reproducibility")
    # Assume default seed
    set_seed(42)
```

### Different GPUs, Different Results

**Problem**: RTX 3090 vs RTX 4090 produce slightly different results

**Explanation**: This is **expected and unavoidable**. Different GPU architectures have:
- Different floating-point precision
- Different rounding behavior
- Different optimized kernels

**Solution**: For true reproduction, use:
- Same GPU model
- Same CUDA version
- Same driver version

## API Reference

### Core Functions

```python
# Set all seeds
set_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> Dict

# Get current RNG state
get_seed_state() -> Dict

# Restore RNG state
restore_seed_state(state: Dict)

# Seed DataLoader workers
seed_worker(worker_id: int)

# Create reproducible generator
create_reproducible_generator(seed: int) -> torch.Generator
```

### Checkpoint Functions

```python
# Add seed to checkpoint
add_seed_to_checkpoint(checkpoint: Dict, seed_info: Dict) -> Dict

# Validate checkpoint seed
validate_checkpoint_seed(checkpoint: Dict, expected_seed: int) -> bool
```

### Verification

```python
# Verify reproducibility works
verify_reproducibility(num_iterations: int = 10) -> bool
```

## Examples

### Example 1: Reproducible Training

```python
from axs_lib.reproducibility import set_seed, add_seed_to_checkpoint

# Set seed
seed = 42
seed_info = set_seed(seed, deterministic=True)

# Train model
model = train_model(...)

# Save with seed
checkpoint = {
    'model_state_dict': model.state_dict(),
    'epoch': 10,
    'loss': 0.123
}
add_seed_to_checkpoint(checkpoint, seed_info)
torch.save(checkpoint, 'model.pt')
```

### Example 2: Resume Training Reproducibly

```python
from axs_lib.reproducibility import set_seed, restore_seed_state

# Load checkpoint
checkpoint = torch.load('model.pt')

# Restore seed
if 'seed_info' in checkpoint:
    seed = checkpoint['seed_info']['seed']
    set_seed(seed, deterministic=True)
    
    # Restore exact RNG state
    if 'rng_state' in checkpoint:
        restore_seed_state(checkpoint['rng_state'])

# Continue training from exact state
model.load_state_dict(checkpoint['model_state_dict'])
train_model(model, ...)
```

### Example 3: Compare Experiments

```python
from axs_lib.reproducibility import validate_checkpoint_seed

experiments = ['exp1/model.pt', 'exp2/model.pt', 'exp3/model.pt']

# Check all use same seed
seeds = []
for exp in experiments:
    ckpt = torch.load(exp)
    if 'seed_info' in ckpt:
        seeds.append(ckpt['seed_info']['seed'])
    else:
        print(f"⚠️ {exp}: No seed info!")

if len(set(seeds)) == 1:
    print(f"✓ All experiments use seed {seeds[0]}")
else:
    print(f"⚠️ Different seeds used: {set(seeds)}")
```

## See Also

- `axs_lib/reproducibility.py` - Implementation
- `scripts/train_stage1.py` - Training script with reproducibility
- `docs/early_stopping.md` - Training optimization
- `docs/metadata_handling.md` - Metadata for conditioning

## References

- PyTorch Reproducibility Guide: https://pytorch.org/docs/stable/notes/randomness.html
- CUDA Reproducibility: https://docs.nvidia.com/cuda/cublas/index.html#reproducibility
- Papers With Code Reproducibility Checklist: https://paperswithcode.com/rc2020
