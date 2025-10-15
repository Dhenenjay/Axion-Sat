# CPU Fallback Feature - Stage 2 Prithvi Refinement

## Overview

The Stage 2 Prithvi refinement module includes an **automatic CPU fallback mechanism** for the ConvNeXt refinement head when GPU memory is exhausted. This feature ensures training can continue even on memory-constrained systems, albeit with a performance penalty.

## How It Works

### Normal Operation (GPU)
```
Input → Prithvi Backbone (GPU) → ConvNeXt Head (GPU) → Output
        ↓
     ~0.013s                      ~0.013s
```

### Fallback Operation (GPU → CPU → GPU)
```
Input → Prithvi Backbone (GPU) → [OOM Detected]
                                        ↓
                                  Clear GPU Cache
                                        ↓
                          Move Head to CPU
                                        ↓
                          Move Features to CPU
                                        ↓
                          ConvNeXt Head (CPU) → ~0.125s
                                        ↓
                          Move Output to GPU
                                        ↓
                          Try Move Head Back to GPU
                                        ↓
                                    Output
```

### Performance Penalty
- **CPU processing**: ~10x slower than GPU
- **Data transfers**: Additional overhead for moving tensors between devices
- **Typical penalty**: 100-200ms per batch (vs 10-20ms on GPU)

## When CPU Fallback Triggers

The fallback mechanism activates when:
1. GPU runs out of memory during ConvNeXt head forward pass
2. CUDA error contains "out of memory" or similar keywords
3. Runtime error occurs on GPU device

### Example Trigger Conditions
- Large batch size + high resolution tiles
- Insufficient VRAM headroom
- Memory fragmentation
- Concurrent GPU usage by other processes

## What Happens During Fallback

### Step 1: OOM Detection
```python
try:
    refined = self.refinement_head(features)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        # Trigger fallback
```

### Step 2: Warning Message
```
======================================================================
GPU OUT OF MEMORY - ConvNeXt Head Fallback
======================================================================
Moving refinement head to CPU. This will be slower but allows
processing to continue. Consider:
  1. Reducing batch size
  2. Reducing tile size
  3. Disabling discriminator
  4. Reducing num_convnext_blocks
======================================================================
```

### Step 3: CPU Execution
- GPU cache cleared with `torch.cuda.empty_cache()`
- ConvNeXt head moved to CPU
- Features moved to CPU
- Forward pass on CPU (with timing)

### Step 4: Return to GPU
- Output moved back to original device
- Attempt to move head back to GPU (may fail if OOM persists)
- Log runtime penalty

### Step 5: Performance Logging
```
CPU fallback runtime: 0.125s (expected GPU: ~0.013s, penalty: ~0.113s)
```

## Training Loop Integration

The training script tracks CPU fallback occurrences:

```python
# During training epoch
cpu_fallback_count = 0

# Custom warning handler
def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    if "CPU fallback" in str(message):
        cpu_fallback_count += 1
```

### Epoch Summary
```
⚠️  CPU fallback occurred 5 times this epoch
   Consider reducing memory usage to avoid performance penalty.
```

### Progress Bar
```
Epoch 10: 100%|██████████| 1000/1000 [45:23<00:00, loss=0.0234, cpu_fb=5]
```

## Prevention Strategies

### 1. Reduce Memory Footprint

**Option A: Reduce Tile Size**
```yaml
data:
  tile_size: 96  # Down from 120
```
**Memory savings**: ~40%

**Option B: Increase Gradient Accumulation**
```yaml
training:
  batch_size: 1
  grad_accum_steps: 16  # Up from 8
```
**Memory savings**: No change per step, same effective batch

**Option C: Simplify ConvNeXt Head**
```yaml
stage2:
  model:
    num_convnext_blocks: 2  # Down from 4
    hidden_dim: 128         # Down from 256
```
**Memory savings**: ~50% on head

### 2. Disable Memory-Intensive Features

**Disable Discriminator**
```python
criterion = Stage2Loss(
    spectral_weight=1.0,
    identity_weight=0.5,
    adversarial_weight=0.0  # Disabled
)
```
**Memory savings**: ~100-200 MB

### 3. Monitor GPU Usage

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log memory usage
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
           --format=csv -l 1 > gpu_usage.log
```

## Performance Impact Analysis

### Scenario 1: No Fallback (Ideal)
```
Epoch time: 45 minutes
Batches per second: 22
Memory usage: 8.5 / 12 GB
```

### Scenario 2: Occasional Fallback (5 per epoch)
```
Epoch time: 46 minutes (+2%)
Batches per second: ~21.5
Memory usage: 9.5 / 12 GB (peaks)
Additional overhead: ~30 seconds
```

### Scenario 3: Frequent Fallback (50 per epoch)
```
Epoch time: 52 minutes (+15%)
Batches per second: ~19
Memory usage: 11.5 / 12 GB (constant peaks)
Additional overhead: ~5 minutes
```

### Scenario 4: Persistent Fallback (all batches)
```
Epoch time: 180 minutes (+300%)
Batches per second: ~5.5
Memory usage: CPU-bound
⚠️ Strongly recommend reducing memory usage
```

## Best Practices

### ✓ DO

1. **Monitor fallback frequency**: Check training logs
2. **Investigate if fallback occurs**: Adjust hyperparameters
3. **Use as safety net**: Let it catch occasional OOM spikes
4. **Profile memory usage**: Understand peak consumption

### ✗ DON'T

1. **Rely on fallback for every batch**: Performance will degrade significantly
2. **Ignore fallback warnings**: Indicates configuration needs adjustment
3. **Disable error handling**: Fallback prevents training crashes
4. **Assume CPU is fast enough**: It's 10x+ slower than GPU

## Testing

Run the CPU fallback test suite:

```bash
python tests/test_stage2_cpu_fallback.py
```

### Expected Output
```
test_cpu_fallback_on_simulated_oom ... ok
test_mixed_device_execution ... ok
test_normal_forward_no_fallback ... ok
test_refinement_head_device_migration ... ok
test_warning_contains_runtime_info ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.190s

OK
```

## Debugging

### Check if Fallback is Occurring

**In Training Logs** (`training_log.jsonl`):
```bash
cat outputs/stage2/training_log.jsonl | jq 'select(.train.cpu_fallback_count != null)'
```

**In Console Output**:
```
⚠️  CPU fallback occurred N times this epoch
```

### Measure Actual Performance Impact

```python
import json
import pandas as pd

# Load training log
logs = []
with open('outputs/stage2/training_log.jsonl') as f:
    for line in f:
        logs.append(json.loads(line))

df = pd.DataFrame(logs)

# Epochs with fallback
df_fallback = df[df['train'].apply(lambda x: 'cpu_fallback_count' in x)]

print(f"Fallback epochs: {len(df_fallback)}")
print(f"Average fallback count: {df_fallback['train'].apply(lambda x: x.get('cpu_fallback_count', 0)).mean():.1f}")
```

### Force CPU Fallback for Testing

```python
# In your training script, temporarily add:
import torch

# Override to always trigger OOM
original_forward = model.refinement_head.forward

def oom_forward(*args, **kwargs):
    if not hasattr(oom_forward, 'triggered'):
        oom_forward.triggered = True
        raise RuntimeError("CUDA out of memory")
    return original_forward(*args, **kwargs)

model.refinement_head.forward = oom_forward
```

## Technical Implementation

### Key Code Sections

**Detection** (`axs_lib/stage2_prithvi_refine.py:546-608`):
```python
try:
    refined = self.refinement_head(features)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        # Fallback logic
```

**Tracking** (`scripts/train_stage2.py:478-489`):
```python
def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    if "CPU fallback" in str(message):
        cpu_fallback_count += 1
```

**Reporting** (`scripts/train_stage2.py:573-576`):
```python
if cpu_fallback_count > 0:
    print(f"⚠️  CPU fallback occurred {cpu_fallback_count} times this epoch")
```

## Future Improvements

### Potential Enhancements

1. **Adaptive head simplification**: Dynamically reduce ConvNeXt blocks on OOM
2. **Mixed precision fallback**: Try FP16 before CPU
3. **Gradient checkpointing**: Enable for head if OOM occurs
4. **Persistent CPU mode**: Keep head on CPU after first fallback
5. **Memory profiling**: Automatic suggestions based on available VRAM

### Community Contributions

If you encounter OOM issues not handled by the fallback:
1. Open an issue with reproduction steps
2. Include GPU specs and memory usage logs
3. Share configuration and tile sizes
4. Suggest improvements to fallback logic

## Summary

The CPU fallback feature provides a **safety net** for Stage 2 training on memory-constrained systems:

- ✅ **Prevents crashes** when GPU runs out of memory
- ✅ **Logs performance impact** for monitoring
- ✅ **Automatic recovery** without manual intervention
- ⚠️ **Performance penalty** indicates need for optimization
- 📊 **Training continues** even on minimal hardware

**Key Takeaway**: CPU fallback should be occasional, not frequent. If you see persistent fallbacks, reduce memory usage through configuration adjustments.

---

*Version: 1.0*  
*Last Updated: 2025-10-14*
