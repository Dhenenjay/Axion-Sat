# Stage 2 Best Checkpoint

**Export Date:** 2025-10-14 09:57:34

## Checkpoint Information

- **File:** `stage2_best.pt`
- **Training Step:** 1000
- **Training Epoch:** 1
- **Composite Score:** 0.0728

## Selection Criteria

This checkpoint was selected based on:
1. **Spectral Plausibility Improvement** (vs Stage 1 baseline)
2. **Minimal Edge Movement Penalty** (vs Stage 1 geometry)

### Hard Constraints
- Mean edge displacement < 2.0 pixels: ✓ PASS
- Max edge displacement < 5.0 pixels: ✓ PASS
- Edge ratio > 0.7: ✓ PASS

## Performance Metrics

### Spectral Improvements

| Metric | Stage 1 (v1) | Stage 2 (v2) | Improvement | Target |
|--------|--------------|--------------|-------------|--------|
| NDVI RMSE | 0.0850 | 0.0634 | +0.0216 | +0.015 |
| EVI RMSE | 0.0950 | 0.0718 | +0.0232 | +0.020 |
| SAM (rad) | 0.1800 | 0.1242 | +0.0558 | +0.040 |

### Geometric Consistency

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Mean edge displacement | 1.841 px | < 2.0 px | ✓ PASS |
| Max edge displacement | 2.910 px | < 5.0 px | ✓ PASS |
| Edge ratio | 0.835 | > 0.7 | ✓ PASS |

## Usage

### Loading Checkpoint

Load this checkpoint for Stage 2 inference:

```python
import torch

# Load checkpoint
checkpoint = torch.load('stage2_best.pt')

# Load into model
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# If model was quantized to fp16, convert to fp16 for inference
if 'quantization_info' in checkpoint:
    model = model.half()  # Convert model to fp16
    print(f"Model quantized: {checkpoint['quantization_info']['dtype']}")
```

### FP16 Quantization

If this checkpoint was exported with `--quantize_fp16`:
- **Memory footprint**: ~50% reduction
- **Inference speed**: Faster on GPUs with Tensor Cores (V100, A100, RTX series)
- **Accuracy**: Minimal impact on spectral metrics (typically < 0.1% difference)

To use fp16 quantized model:
```python
# Load model
model = YourRefinerModel()
checkpoint = torch.load('stage2_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.half()  # Model is already fp16
model.eval()

# Input tensors must also be fp16
input_tensor = input_tensor.half()
with torch.no_grad():
    output = model(input_tensor)
```

## Notes

- This checkpoint represents the best balance between spectral accuracy and geometric preservation
- Stage 2 refines Stage 1 outputs WITHOUT altering spatial structure
- Use validation metrics to verify performance on your specific data
- FP16 quantization is recommended for production inference to reduce memory and improve speed

For detailed selection methodology, see `selection_metadata.json`.
