# TerraMind Training Issue Analysis

## Problem Summary

Training the TerraMind SAR-to-Optical model with LoRA adapters is failing due to gradient flow being broken. The error occurs during backpropagation:

```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

## Root Cause

### Debugging Output
```
⚠ [DEBUG] Output tensor requires_grad=False - gradient flow broken!
   Model training mode: True
   Input requires_grad: False
```

### Technical Explanation

TerraMind is a **generative foundation model** that uses **autoregressive token generation** with discrete sampling operations. The generation process involves:

1. Encoding SAR input to latent space
2. **Discrete token sampling** using top-k/top-p sampling (`torch.cumsum` for probability distributions)
3. Autoregressive generation of output tokens
4. Decoding to pixel space

The discrete sampling operations (`cumsum`, `topk`, categorical sampling) are **non-differentiable** by nature. Even though the model is in training mode and has LoRA adapters attached, the sampling process breaks the gradient chain completely.

This is a fundamental property of autoregressive generation models - they sample discrete tokens which has no gradients.

## Environment Status

✅ **Successfully Completed:**
- CUDA-enabled PyTorch installed (2.5.1+cu121)
- NVIDIA RTX 4050 GPU detected (6.44 GB VRAM)
- TerraTorch 1.1 installed with dependencies
- LoRA adapters successfully applied (134 adapters)
- Deterministic training mode configured
- Fixed shape mismatch issues (12 channels → 4 channels)
- Fixed reproducibility configuration (warn_only for non-deterministic ops)

❌ **Current Blocker:**
- Gradient flow broken due to discrete sampling in TerraMind's generation process

## Possible Solutions

### Option 1: Reinforcement Learning Approach
Use policy gradient methods (REINFORCE) to train through the discrete sampling:
- Treat sampling as stochastic policy
- Use reward signals based on the loss function
- Estimate gradients via log-probability trick

**Pros:**
- Can train through discrete operations
- Theoretically sound approach

**Cons:**
- High variance in gradient estimates
- Requires significant code refactoring
- Slower convergence

### Option 2: Straight-Through Estimators
Replace discrete sampling with differentiable relaxations during training:
- Use Gumbel-Softmax for categorical sampling
- Straight-through gradients for hard decisions

**Pros:**
- Simpler than RL
- Lower variance

**Cons:**
- Requires modifying TerraMind internals
- May affect generation quality

### Option 3: Two-Stage Training
Keep TerraMind frozen, train a refinement network:
1. Use TerraMind in inference mode (no gradients)
2. Train a separate refinement network on top of TerraMind outputs
3. Refinement network learns to correct/enhance the generated images

**Pros:**
- Doesn't require modifying TerraMind
- Can use standard supervised learning
- Simpler implementation

**Cons:**
- Doesn't actually fine-tune TerraMind
- May be less effective

### Option 4: Teacher Forcing / Deterministic Mode
Check if TerraMind has a deterministic forward pass without sampling:
- Use ground truth tokens during training (teacher forcing)
- Only use sampling during inference

**Pros:**
- Standard approach for seq2seq models
- Allows gradient flow

**Cons:**
- Requires TerraMind to support this mode
- Need to contact TerraTorch developers

### Option 5: Encoder-Only Fine-Tuning
Fine-tune only the encoder part before sampling:
- Freeze the generation/sampling parts
- Train encoder to produce better latent representations
- Use a differentiable loss on latent space

**Pros:**
- Avoids sampling problem
- Still improves model performance

**Cons:**
- Limited to encoder improvements
- Requires access to latent representations

## Recommended Next Steps

### Immediate Actions
1. **Contact TerraTorch developers** to ask:
   - Does TerraMind support gradient-friendly training?
   - Is there a teacher-forcing or deterministic forward mode?
   - What is the recommended approach for fine-tuning?

2. **Check TerraTorch documentation** for:
   - Training examples
   - Fine-tuning guides
   - API for deterministic forward passes

### Short-term Workaround (Option 3)
Implement two-stage training:
```python
# Stage 1: TerraMind (frozen, inference only)
with torch.no_grad():
    synthetic_opt = terramind_generator(sar_input)

# Stage 2: Refinement network (trainable)
refined_opt = refinement_network(synthetic_opt, sar_input)
loss = criterion(refined_opt, ground_truth)
loss.backward()  # ← Gradients flow through refinement only
```

### Medium-term Solution (Option 4)
If TerraMind supports teacher forcing:
- Modify training loop to use deterministic forward
- Only use sampling during validation/inference

## Code Locations

**Main training script:**
- `scripts/train_stage1.py` (lines 615-692: forward pass and backprop)

**Model wrapper:**
- `axs_lib/stage1_tm_s2o.py` (lines 171-196: generation call)

**LoRA application:**
- `axs_lib/lora_utils.py` (LoRA adapter injection)

**Reproducibility fix:**
- `axs_lib/reproducibility.py` (line 153: `warn_only=True` for deterministic algorithms)

## Technical Details

### Model Architecture
- **Total parameters:** 1,013,629,381
- **Trainable (LoRA):** 853,776,837 (84.23%)
- **LoRA rank:** 8
- **LoRA alpha:** 16
- **Number of LoRA adapters:** 134

### Training Configuration
- **Tile size:** 224x224
- **Timesteps:** 10 (reduced for memory)
- **Batch size:** 1
- **Gradient accumulation:** 8 (effective batch size = 8)
- **Learning rate:** 0.0001
- **GPU:** NVIDIA RTX 4050 (6.44 GB VRAM)

### Loss Function (GAC Stage 1)
- L1: 1.0
- MS-SSIM: 0.5
- LPIPS: 0.1 (disabled, package not installed)
- Color Consistency: 0.5

## References

### TerraTorch Resources
- GitHub: https://github.com/IBM/terratorch
- Documentation: (check for training guides)
- Paper: (check if training methodology is described)

### Related Issues
- PyTorch deterministic algorithms: https://pytorch.org/docs/stable/notes/randomness.html
- LoRA paper: https://arxiv.org/abs/2106.09685
- REINFORCE algorithm: https://link.springer.com/article/10.1007/BF00992696

## Status

**Current State:** Blocked - cannot proceed with current training approach

**Required Decision:** Choose one of the solution options above based on:
1. TerraTorch developer response
2. Project requirements (pure TerraMind fine-tuning vs. hybrid approach)
3. Available development time

