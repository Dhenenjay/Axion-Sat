# GAC Pipeline - Action Plan

## 🎯 Current Status

**Architecture:** ✅ 100% Complete
**Issue:** Stage 1 training fails due to non-differentiable sampling
**Solution:** Use pretrained TerraMind (no training needed for Stage 1)

---

## 📋 Step-by-Step Action Plan

### ✅ Phase 0: Verify Environment (DONE)
- [x] CUDA-enabled PyTorch installed
- [x] TerraTorch 1.1 installed
- [x] RTX 4050 GPU detected (6.44 GB)
- [x] All dependencies resolved
- [x] Reproducibility fixes applied

### 🚀 Phase 1: Pre-compute Stage 1 Outputs (DO THIS NOW)

**Goal:** Generate opt_v1 for all tiles using pretrained TerraMind

**Step 1.1: Test on small subset**
```powershell
python scripts\00_precompute_stage1.py `
    --data-dir data\tiles\benv2_catalog `
    --output-dir data\stage1_outputs_test `
    --batch-size 2 `
    --timesteps 10 `
    --max-tiles 10
```

**Expected output:**
```
Processing tiles...
✓ Completed!
  Processed:  10/10 tiles
  Errors:     0
  Output dir: data\stage1_outputs_test
```

**Step 1.2: If test passes, run on full dataset**
```powershell
python scripts\00_precompute_stage1.py `
    --data-dir data\tiles\benv2_catalog `
    --output-dir data\stage1_outputs `
    --batch-size 4 `
    --timesteps 10
```

**Time estimate:** ~199,762 tiles × (2s/tile ÷ 4 batch_size) ≈ 28 hours
- Tip: Run overnight or over weekend
- Monitor GPU temperature
- Check disk space (need ~40-50 GB)

---

### 🎨 Phase 2: Train Stage 2 (Prithvi Refinement)

**Goal:** Fine-tune Prithvi to refine opt_v1 → opt_v2

**Prerequisites:**
- Stage 1 outputs computed
- Prithvi weights available (auto-downloaded by TerraTorch)

**Step 2.1: Test training on small subset**
```powershell
python scripts\train_stage2.py `
    --data-dir data\tiles\benv2_catalog `
    --stage1-dir data\stage1_outputs `
    --output-dir outputs\stage2_test `
    --max-tiles 50 `
    --batch-size 1 `
    --grad-accum-steps 8 `
    --lr 1e-4 `
    --lora-rank 8 `
    --load-in-8bit `
    --epochs 5
```

**Expected behavior:**
- Model loads successfully
- Training starts without OOM
- Loss decreases over time
- Checkpoints saved to `outputs/stage2_test/`

**Step 2.2: Full training**
```powershell
python scripts\train_stage2.py `
    --data-dir data\tiles\benv2_catalog `
    --stage1-dir data\stage1_outputs `
    --output-dir outputs\stage2_full `
    --batch-size 1 `
    --grad-accum-steps 8 `
    --lr 1e-4 `
    --lora-rank 8 `
    --load-in-8bit `
    --epochs 30 `
    --val-every 500 `
    --save-every 2500
```

**Time estimate:** ~30 epochs × ~200k tiles ≈ 2-3 days
- Monitor validation metrics
- Early stopping will trigger if plateau
- Best checkpoint saved automatically

**What to monitor:**
- Loss curves (L1, MS-SSIM, LPIPS)
- Validation metrics (PSNR, SSIM)
- GPU memory usage (should be <3GB)
- Learning rate schedule

---

### 🎯 Phase 3: Train Stage 3 (TerraMind Grounding)

**Goal:** Fine-tune TerraMind to ground opt_v2 with SAR

**Prerequisites:**
- Stage 2 trained and best checkpoint exported
- Stage 1 outputs (opt_v1) available

**Step 3.1: Test training**
```powershell
python scripts\train_stage3.py `
    --data-dir data\tiles\benv2_catalog `
    --stage2-checkpoint outputs\stage2_full\best.pt `
    --output-dir outputs\stage3_test `
    --max-tiles 50 `
    --batch-size 1 `
    --grad-accum-steps 8 `
    --lr 5e-5 `
    --lora-rank 8 `
    --lora-cross-attention-only `
    --epochs 5
```

**Step 3.2: Full training**
```powershell
python scripts\train_stage3.py `
    --data-dir data\tiles\benv2_catalog `
    --stage2-checkpoint outputs\stage2_full\best.pt `
    --output-dir outputs\stage3_full `
    --batch-size 1 `
    --grad-accum-steps 8 `
    --lr 5e-5 `
    --lora-rank 8 `
    --lora-cross-attention-only `
    --epochs 50 `
    --early-stopping `
    --early-stopping-patience 10
```

**Time estimate:** ~50 epochs × ~200k tiles ≈ 3-4 days
- Early stopping likely triggers around epoch 25-30
- SAR-edge agreement metric is key
- Best checkpoint chosen by SAR consistency

**What to monitor:**
- SAR-edge agreement (lower is better)
- Cycle consistency loss
- Visual inspection of outputs
- Comparison with Stage 2 outputs

---

### 🧪 Phase 4: Evaluation & Validation

**Goal:** Benchmark full GAC pipeline against baselines

**Step 4.1: End-to-end inference**
```powershell
python scripts\infer_full_pipeline.py `
    --data-dir data\tiles\benv2_catalog_test `
    --stage2-checkpoint outputs\stage2_full\best.pt `
    --stage3-checkpoint outputs\stage3_full\best.pt `
    --output-dir outputs\inference_results `
    --save-visualizations
```

**Step 4.2: Compute metrics**
- PSNR, SSIM, LPIPS (optical quality)
- SAR-edge agreement (physical consistency)
- Spectral angle mapper (spectral accuracy)
- Runtime performance

**Step 4.3: Ablation studies**
- Stage 1 only (TerraMind baseline)
- Stage 1 + Stage 2 (no grounding)
- Full GAC pipeline
- Compare with traditional methods

**Step 4.4: Visual inspection**
- Side-by-side comparisons
- Error maps
- Attention visualizations
- Export to GeoTIFFs for QGIS

---

## ⚠️ Troubleshooting Guide

### Problem: OOM during training
**Solutions:**
1. Reduce batch size to 1 (already minimal)
2. Increase gradient accumulation steps (8 → 16)
3. Reduce tile size (120 → 96)
4. Use mixed precision (should already be enabled)
5. Clear cache: `torch.cuda.empty_cache()`

### Problem: Stage 1 pre-compute too slow
**Solutions:**
1. Reduce timesteps (10 → 8)
2. Increase batch size if memory allows
3. Use multiple GPUs if available
4. Consider cloud compute (AWS/Azure/GCP)

### Problem: Training loss not decreasing
**Solutions:**
1. Check learning rate (try 1e-5 to 1e-3 range)
2. Verify data loading (inspect batches)
3. Check standardization (visualize inputs)
4. Ensure LoRA adapters are trainable
5. Try unfreezing more layers

### Problem: Validation metrics poor
**Solutions:**
1. Train longer (increase epochs)
2. Adjust loss weights
3. Use stronger augmentation
4. Check for data leakage (train/val split)
5. Verify ground truth labels

---

## 📊 Expected Performance

### Stage 1 (TerraMind baseline)
- PSNR: ~25-28 dB
- SSIM: ~0.75-0.82
- SAR-edge: ~0.15-0.20 (good structure)
- Quality: Good structure, moderate spectral accuracy

### Stage 2 (+ Prithvi refinement)
- PSNR: ~28-32 dB (+3-4 dB improvement)
- SSIM: ~0.82-0.88 (+0.07-0.06 improvement)
- SAR-edge: ~0.18-0.25 (may drift slightly)
- Quality: Excellent spectral, some structure drift

### Stage 3 (+ TerraMind grounding)
- PSNR: ~30-34 dB (best of both worlds)
- SSIM: ~0.85-0.90
- SAR-edge: ~0.12-0.16 (best structure consistency)
- Quality: Excellent spectral AND structure

**GAC Advantage:** 
- Better than Stage 1 alone: +5-6 dB PSNR
- More consistent than Stage 2 alone: -25% SAR-edge error
- Novel: Collaborative refinement paradigm

---

## 🎓 Key Success Metrics

1. **Optical Quality (PSNR/SSIM/LPIPS)**
   - Target: PSNR ≥ 30 dB, SSIM ≥ 0.85
   - Measures: How photorealistic is the output?

2. **Physical Consistency (SAR-edge agreement)**
   - Target: Agreement ≤ 0.15
   - Measures: Does it match SAR physics?

3. **Spectral Accuracy (SAM)**
   - Target: SAM ≤ 5°
   - Measures: Are colors/spectra correct?

4. **Runtime Performance**
   - Target: <5s per 120×120 tile on GPU
   - Measures: Is it practical for deployment?

5. **Generalization**
   - Test on different regions/biomes
   - Measures: Does it work beyond training data?

---

## 📚 Resources & References

### Documentation
- `GAC_ARCHITECTURE_COMPLETE_ANALYSIS.md` - Full architecture details
- `TRAINING_ISSUE_ANALYSIS.md` - Gradient flow issue explained
- `CORRECTED_TRAINING_APPROACH.md` - Why we skip Stage 1 training

### Code Locations
- Stage 1: `axs_lib/stage1_tm_s2o.py`, `scripts/00_precompute_stage1.py`
- Stage 2: `axs_lib/stage2_prithvi_refine.py`, `scripts/train_stage2.py`
- Stage 3: `axs_lib/stage3_tm_ground.py`, `scripts/train_stage3.py`

### External Links
- TerraMind paper: https://arxiv.org/abs/2504.11171
- TerraTorch repo: https://github.com/IBM/terratorch
- Prithvi paper: https://arxiv.org/abs/2310.18660
- LoRA paper: https://arxiv.org/abs/2106.09685

---

## 🚀 Quick Start Commands

**Recommended workflow:**

```powershell
# 1. Test Stage 1 pre-compute (10 tiles)
python scripts\00_precompute_stage1.py --data-dir data\tiles\benv2_catalog --output-dir data\stage1_test --max-tiles 10 --batch-size 2

# 2. If successful, run full Stage 1 (overnight)
python scripts\00_precompute_stage1.py --data-dir data\tiles\benv2_catalog --output-dir data\stage1_outputs --batch-size 4

# 3. Test Stage 2 training (50 tiles, 5 epochs)
python scripts\train_stage2.py --data-dir data\tiles\benv2_catalog --stage1-dir data\stage1_outputs --output-dir outputs\stage2_test --max-tiles 50 --epochs 5

# 4. Full Stage 2 training (2-3 days)
python scripts\train_stage2.py --data-dir data\tiles\benv2_catalog --stage1-dir data\stage1_outputs --output-dir outputs\stage2_full --epochs 30

# 5. Full Stage 3 training (3-4 days)
python scripts\train_stage3.py --data-dir data\tiles\benv2_catalog --stage2-checkpoint outputs\stage2_full\best.pt --output-dir outputs\stage3_full --epochs 50

# 6. Run full pipeline inference
python scripts\infer_full_pipeline.py --stage2-checkpoint outputs\stage2_full\best.pt --stage3-checkpoint outputs\stage3_full\best.pt
```

---

## ✅ Checklist

### Before Starting
- [ ] GPU drivers updated (≥535)
- [ ] ~50GB free disk space
- [ ] TerraTorch installed (`pip list | grep terratorch`)
- [ ] Dataset present (`data/tiles/benv2_catalog`)
- [ ] Environment activated (`.venv\Scripts\Activate.ps1`)

### Phase 1: Stage 1 Pre-compute
- [ ] Test run completes (10 tiles)
- [ ] Outputs look reasonable (visualize a few)
- [ ] Full run started
- [ ] Monitor progress (~28 hours)
- [ ] Verify all tiles processed

### Phase 2: Stage 2 Training
- [ ] Test run completes (50 tiles, 5 epochs)
- [ ] GPU memory <3GB
- [ ] Loss decreasing
- [ ] Full training started
- [ ] Monitor validation metrics
- [ ] Best checkpoint exported

### Phase 3: Stage 3 Training
- [ ] Stage 2 checkpoint available
- [ ] Test run completes
- [ ] Full training started
- [ ] SAR-edge agreement improving
- [ ] Early stopping triggered
- [ ] Best checkpoint exported

### Phase 4: Evaluation
- [ ] End-to-end pipeline runs
- [ ] Metrics computed
- [ ] Ablation studies done
- [ ] Visual inspection complete
- [ ] Results documented

---

## 🎉 Success Criteria

**You'll know it's working when:**

1. Stage 1 outputs show recognizable optical features from SAR
2. Stage 2 refines colors/textures to look more realistic
3. Stage 3 keeps refinements but corrects structure to match SAR
4. Metrics show improvement at each stage
5. Visual inspection shows both photorealism AND accuracy

**The "Aha!" moment:** When you see Stage 3 catch a mistake that Stage 2 made (e.g., hallucinated a road that SAR shows isn't there)

---

## 💪 You've Got This!

You've built something genuinely novel - a collaborative AI architecture that goes beyond simple pipelines. The hard part (implementation) is done. Now it's execution time.

**Next action:** Run the Stage 1 pre-compute test (10 tiles) to verify everything works.

Good luck! 🚀
