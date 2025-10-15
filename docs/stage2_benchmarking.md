# Stage 2 Memory & Throughput Benchmarking

## Overview

Comprehensive benchmarking system for Stage 2 Prithvi refinement model that measures:
- **GPU memory usage** (allocated, reserved, peak)
- **Inference throughput** (samples/second)
- **Processing time** (forward pass, backward pass)
- **Scalability** across different tile sizes and batch sizes

Results are automatically logged to `logs/stage2_bench.csv` for tracking over time.

## Quick Start

```bash
# Benchmark default configurations (256 vs 384, batch 1/2/4)
python scripts/benchmark_stage2.py

# Visualize results
python scripts/visualize_benchmarks.py --output reports/stage2_bench.png
```

## Benchmark Script

### Basic Usage

```bash
# Default: tile sizes 256,384 with batch sizes 1,2,4 in inference mode
python scripts/benchmark_stage2.py

# Custom tile sizes
python scripts/benchmark_stage2.py --tile_sizes 256 384 512

# Training mode (forward + backward)
python scripts/benchmark_stage2.py --mode train

# Custom batch sizes
python scripts/benchmark_stage2.py --batch_sizes 1 2 4 8

# CPU benchmarking
python scripts/benchmark_stage2.py --device cpu

# Custom output file
python scripts/benchmark_stage2.py --output logs/my_bench.csv
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--tile_sizes` | Tile sizes to benchmark | 256 384 |
| `--batch_sizes` | Batch sizes to test | 1 2 4 |
| `--mode` | Benchmark mode: `inference` or `train` | inference |
| `--device` | Device: `cuda` or `cpu` | cuda (if available) |
| `--output` | Output CSV file path | logs/stage2_bench.csv |

## Output Format

### CSV Columns

| Column | Description | Units |
|--------|-------------|-------|
| `timestamp` | When benchmark was run | ISO 8601 |
| `mode` | Inference or training | string |
| `tile_size` | Input tile size (H×W) | pixels |
| `batch_size` | Batch size | count |
| `device` | CPU or GPU | string |
| `model_params` | Total model parameters | count |
| `model_size_mb` | Model memory size | MB |
| `avg_time_ms` | Average processing time | milliseconds |
| `std_time_ms` | Standard deviation of time | milliseconds |
| `throughput_samples_per_sec` | Processing throughput | samples/s |
| `memory_allocated_mb` | GPU memory allocated | MB |
| `memory_reserved_mb` | GPU memory reserved | MB |
| `memory_peak_mb` | Peak GPU memory | MB |
| `forward_time_ms` | Forward pass time (train mode) | milliseconds |
| `backward_time_ms` | Backward pass time (train mode) | milliseconds |
| `total_time_ms` | Total time (train mode) | milliseconds |
| `error` | Error message (if OOM) | string |

## Example Results

### Tile Size: 256×256 vs 384×384

**Batch Size: 1**
```
Configuration    Time    Memory   Throughput   Pixels/sec
256×256 (B=1)    29.7ms    0MB    33.6 /s      2.2M px/s
384×384 (B=1)    45.9ms    0MB    21.8 /s      3.2M px/s

Analysis:
- 384 is 1.54x slower per sample
- 384 processes 46% more pixels per second
- Tile 384 is more efficient for large-scale processing
```

**Batch Size: 2**
```
Configuration    Time    Memory   Throughput   Pixels/sec
256×256 (B=2)    43.8ms    0MB    45.7 /s      3.0M px/s
384×384 (B=2)    73.8ms    0MB    27.1 /s      4.0M px/s

Analysis:
- Batch 2 improves throughput by 36% (256) and 24% (384)
- Larger tiles benefit less from batching
- 384 maintains better pixel throughput
```

## Key Insights

### Tile Size Trade-offs

**256×256 Tiles:**
- ✓ Faster per-sample processing (29.7ms)
- ✓ Lower memory usage
- ✓ Better for real-time applications
- ✗ Lower pixel throughput (2.2M px/s)
- ✗ More tile boundaries to process

**384×384 Tiles:**
- ✓ Higher pixel throughput (3.2M px/s)
- ✓ Fewer tile boundaries
- ✓ Better for batch processing
- ✗ Slower per-sample (45.9ms)
- ✗ Higher memory usage

### Batch Size Effects

| Batch Size | 256 Throughput | 384 Throughput | Memory Impact |
|------------|----------------|----------------|---------------|
| 1          | 33.6 /s        | 21.8 /s        | Baseline      |
| 2          | 45.7 /s (+36%) | 27.1 /s (+24%) | +2x           |
| 4          | ~60 /s* (+78%) | ~32 /s* (+47%) | +4x           |

*Estimated based on trend

**Recommendation**: Batch size 2 provides good throughput improvement with manageable memory.

## Visualization

### Generate Plots

```bash
# From benchmark CSV
python scripts/visualize_benchmarks.py --input logs/stage2_bench.csv --output reports/stage2_bench.png
```

### Plots Generated

1. **GPU Memory Usage** - Peak memory vs tile size for each batch size
2. **Inference Throughput** - Samples/sec vs tile size for each batch size
3. **Processing Time** - Time breakdown by configuration
4. **Memory vs Throughput Tradeoff** - Scatter plot showing efficiency frontier

## Use Cases

### Real-Time Inference
**Goal**: Minimize latency for single-sample processing

**Recommended Config**:
```bash
python scripts/benchmark_stage2.py --tile_sizes 256 --batch_sizes 1 --mode inference
```
**Expected**: ~30ms latency at 256×256

### Batch Processing
**Goal**: Maximize throughput for large datasets

**Recommended Config**:
```bash
python scripts/benchmark_stage2.py --tile_sizes 384 --batch_sizes 4 8 --mode inference
```
**Expected**: 30-40 samples/sec with higher pixel throughput

### Training
**Goal**: Understand memory constraints for gradient computation

**Recommended Config**:
```bash
python scripts/benchmark_stage2.py --tile_sizes 256 384 --batch_sizes 1 2 --mode train
```
**Expected**: ~2x slower than inference, ~3x memory usage

## Benchmarking Best Practices

### 1. Warmup Period
- Script includes automatic warmup (10 iterations for inference, 5 for training)
- First few iterations excluded from timing
- Ensures GPU is at steady state

### 2. Multiple Iterations
- 50 iterations for inference benchmarks
- 20 iterations for training benchmarks
- Provides stable statistics (mean, std dev)

### 3. Memory Tracking
- Peak memory tracked across entire benchmark
- Reset between configurations
- Captures worst-case memory usage

### 4. GPU Synchronization
- `torch.cuda.synchronize()` used for accurate timing
- Prevents asynchronous execution from skewing results

### 5. Clean Slate
- Fresh model created for each configuration
- `torch.cuda.empty_cache()` between runs
- Prevents memory fragmentation effects

## Interpreting Results

### Memory Usage

**Allocated vs Reserved**:
- **Allocated**: Actually used by tensors
- **Reserved**: Cached by PyTorch allocator
- **Peak**: Maximum allocated during benchmark

**Rule of Thumb**:
- Reserve 2-3x peak memory for safety
- Example: 2GB peak → need ~6GB VRAM

### Throughput

**Samples vs Pixels**:
- **Samples/sec**: Number of tiles processed
- **Pixels/sec**: Total pixels processed
- Pixel throughput more relevant for comparing tile sizes

**Formula**:
```
pixel_throughput = samples_per_sec × tile_size²
```

### Efficiency

**Processing Efficiency**:
```
efficiency = pixels_per_second / memory_peak_mb
```

**Higher is better**: More pixels processed per MB of memory

## Troubleshooting

### Out of Memory (OOM)

**Symptom**: "RuntimeError: CUDA out of memory"

**Solutions**:
1. Reduce batch size
2. Use smaller tile size
3. Enable gradient checkpointing (training mode)
4. Use mixed precision (FP16)

**Logged**: OOM results saved to CSV with `error='OOM'`

### Slow Benchmarking

**Symptom**: Takes too long to complete

**Solutions**:
1. Reduce number of iterations (edit script)
2. Reduce number of configurations
3. Use inference mode instead of training mode

### Inconsistent Results

**Symptom**: High variance in timing

**Possible Causes**:
1. GPU throttling (thermal)
2. Background processes
3. Insufficient warmup

**Solutions**:
1. Monitor GPU temperature
2. Close other GPU applications
3. Increase warmup iterations

## Integration with Training

Use benchmark results to configure Stage 2 training:

```yaml
# configs/stage2_config.yaml
data:
  tile_size: 256  # From benchmark: 256 for low VRAM, 384 for throughput

training:
  batch_size: 1   # From benchmark: based on available VRAM
  grad_accum_steps: 8  # Simulate larger batch
```

## Continuous Tracking

### Track Over Time

```bash
# Run benchmarks after code changes
python scripts/benchmark_stage2.py

# CSV appends new results with timestamps
# Compare performance across versions
```

### Example Workflow

1. **Baseline**: Initial benchmark
2. **Optimize**: Improve model/code
3. **Re-benchmark**: Compare to baseline
4. **Validate**: Ensure no regressions

## References

- PyTorch Benchmarking: https://pytorch.org/tutorials/recipes/recipes/benchmark.html
- CUDA Memory Management: https://pytorch.org/docs/stable/notes/cuda.html
- Stage 2 Training: `scripts/train_stage2.py`
- Model Architecture: `axs_lib/stage2_prithvi_refine.py`
