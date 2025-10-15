# Model Licenses and Usage Terms

This document summarizes the licensing terms and usage caveats for foundation models used in the Axion-Sat pipeline.

---

## Summary

| Model | License | Source | Developer(s) |
|-------|---------|--------|--------------|
| **TerraMind 1.0 Large** | Apache-2.0 | [HuggingFace](https://huggingface.co/ibm-nasa-geospatial/terramind-1.0-large) | IBM Research, ESA, Forschungszentrum Jülich |
| **Prithvi EO 2.0 600M** | Apache-2.0 | [HuggingFace](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M) | IBM Research, NASA |

**All models used in this pipeline are licensed under Apache-2.0**, which permits:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Patent use
- ✅ Private use

**Requirements:**
- ⚠️ Include license and copyright notice
- ⚠️ State changes made to the code
- ⚠️ Include NOTICE file if provided

---

## TerraMind 1.0 Large

### Overview
TerraMind is a multimodal generative foundation model for Earth Observation data, capable of any-to-any modality translation (e.g., SAR → optical, optical → SAR, temporal forecasting).

### License
**Apache License 2.0**

- **Copyright:** © 2025 International Business Machines Corporation, European Space Agency, Forschungszentrum Jülich
- **License Text:** [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Model Card:** [HuggingFace Model Card](https://huggingface.co/ibm-nasa-geospatial/terramind-1.0-large)

### Key Details

**Architecture:**
- **Parameters:** ~1 billion (1B)
- **Backbone:** Dual-scale transformer encoder
- **Modalities:** S1GRD (SAR), S2L2A (optical), S2L1C (top-of-atmosphere), DEM, coordinates
- **Pre-training:** 500B tokens from TerraMesh dataset
- **Context:** 12-48 timesteps (temporal sequences)

**Capabilities:**
- Cross-modal synthesis (SAR ↔ optical)
- Temporal forecasting
- Missing data imputation
- Multi-resolution generation

### Important Caveats

#### 1. **"Mental Images" (Tokenized Image Model - TiM)**

TerraMind does **not** generate pixel-level images directly. Instead, it produces **tokenized latent representations** ("mental images") in a learned semantic space:

> **From TerraMind README:**  
> *"TerraMind's generations exist in a learned latent space that captures cross-modal correlations without being directly interpretable as pixel values. These representations remain abstract until decoded through modality-specific FSQ-VAEs (Finite Scalar Quantization Variational Autoencoders)."*

**Implications for Axion-Sat:**
- TerraMind outputs are **latent embeddings**, not visualizations
- Requires **Stage 2 (Prithvi)** to ground latents into segmentation masks
- Outputs should be treated as "thoughts" about target modality, not final predictions

**Why This Matters:**
- You cannot directly visualize TerraMind's output as an image
- The latent space is optimized for semantic reasoning, not perceptual quality
- Downstream models (like Prithvi) map latents to task-specific outputs

#### 2. **Any-to-Any Modality Translation**

TerraMind supports flexible input/output modality combinations:

| Input | Output | Use Case |
|-------|--------|----------|
| S1GRD (SAR) | S2L2A (optical) | Cloud-free optical synthesis |
| S2L2A (optical) | S1GRD (SAR) | All-weather SAR generation |
| S2L2A (t=0) | S2L2A (t+1) | Temporal forecasting |
| S1GRD + S2L2A | S2L2A (missing bands) | Data imputation |

**Caveats:**
- Not all modality pairs are equally well-supported (check model card)
- Quality depends on pre-training data distribution
- Some combinations may produce lower-fidelity outputs

#### 3. **Recommended Inference Settings**

From the model documentation:

| Setting | Training | Low VRAM | Fast Inference |
|---------|----------|----------|----------------|
| **Timesteps** | 50 | 12 | 6-8 |
| **Guidance Scale** | 1.5 | 1.0 | 1.0 |
| **Batch Size** | 8-16 | 1 | 1-4 |

**For Axion-Sat (low-VRAM pipeline):**
- Timesteps: **6** (minimal diffusion steps)
- Guidance: **1.0** (no classifier-free guidance)
- Batch size: **1** (with gradient accumulation)

#### 4. **Data Requirements**

TerraMind expects **pre-processed and standardized** inputs:
- **Sentinel-1:** Normalized backscatter (dB scale), typically [-30, 0] dB
- **Sentinel-2:** Surface reflectance (L2A), scaled to [0, 1]
- **Spatial resolution:** 10m native, resample if needed
- **Cloud masks:** Recommended but not required

#### 5. **Ethical Considerations**

From the model card:

> *"Models trained on satellite imagery may inadvertently encode biases present in training data (e.g., geographic coverage, seasonal patterns, land use types). Users should validate outputs on their specific use case."*

**For water segmentation:**
- Validate on diverse water body types (rivers, lakes, reservoirs, coastal)
- Test across seasons (ice cover, vegetation growth, drought)
- Check for biases in underrepresented regions

---

## Prithvi EO 2.0 600M

### Overview
Prithvi is a vision transformer (ViT) foundation model for Earth Observation, pre-trained with masked autoencoding (MAE) on multispectral satellite imagery. It serves as the refinement stage in Axion-Sat.

### License
**Apache License 2.0**

- **Copyright:** © 2024 International Business Machines Corporation, NASA
- **License Text:** [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Model Card:** [HuggingFace Model Card](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M)

### Key Details

**Architecture:**
- **Parameters:** ~600 million (600M)
- **Backbone:** ViT-Large with 3D patch embeddings (spatial + temporal)
- **Input:** 6 bands (RGB, NIR, SWIR1, SWIR2) at 10-30m resolution
- **Pre-training:** HLS (Harmonized Landsat-Sentinel) dataset
- **Context:** 1-3 timesteps (single or multi-temporal)

**Capabilities:**
- Dense per-pixel segmentation
- Multi-temporal fusion
- Transfer learning for downstream tasks
- Fine-tuning with small datasets (~100-1000 samples)

### Important Caveats

#### 1. **Input Band Flexibility**

Prithvi was pre-trained on **6 bands** (RGB + NIR + SWIR1 + SWIR2), but can be adapted:

| Configuration | Bands | Notes |
|---------------|-------|-------|
| **Full** | B02, B03, B04, B08, B11, B12 | Best performance, more memory |
| **Reduced (Axion-Sat)** | B02, B03, B04, B08 | 33% memory savings, acceptable quality |
| **RGB only** | B02, B03, B04 | Minimal memory, lower quality |

**For low-VRAM systems:**
- Use 4-band configuration (RGB + NIR)
- SWIR bands provide better water/vegetation discrimination but require more memory
- Fine-tuning may compensate for reduced band count

#### 2. **Temporal Fusion**

Prithvi supports **multi-temporal inputs** (1-3 timesteps):

```python
# Single timestep (Axion-Sat configuration)
input_shape = (batch, channels, height, width)

# Multi-temporal (future enhancement)
input_shape = (batch, timesteps, channels, height, width)
```

**Caveats:**
- Multi-temporal adds memory overhead (~3x for 3 timesteps)
- May improve accuracy for temporal water dynamics (floods, seasonal lakes)
- Not currently used in Axion-Sat (v1.0) due to VRAM constraints

#### 3. **Fine-Tuning Strategies**

Prithvi supports multiple fine-tuning approaches:

| Strategy | Trainable Params | VRAM | Quality |
|----------|------------------|------|---------|
| **Full Fine-Tuning** | 600M (100%) | ~20 GB | Best |
| **Decoder Only** | ~50M (8%) | ~8 GB | Good |
| **LoRA (r=8)** | ~60M (10%) | ~6 GB | Good |
| **LoRA (r=4)** | ~30M (5%) | ~5 GB | Acceptable |

**Axion-Sat uses LoRA (r=8)** for optimal quality/memory trade-off.

#### 4. **Pre-Training Dataset Biases**

From the model card:

> *"Prithvi was pre-trained on HLS data covering North America and Europe with higher density than other regions. Performance may vary on underrepresented geographies."*

**Implications:**
- Validate on your target region (Africa, Asia, South America, etc.)
- Consider domain adaptation if performance is poor
- Fine-tuning with local data is highly recommended

#### 5. **Geolocation-Aware Processing**

Prithvi optionally accepts **geographic coordinates** as input:

```python
# With coordinates (recommended for large AOIs)
output = model(imagery, coordinates=latlon_tensor)

# Without coordinates (simpler, slightly lower accuracy)
output = model(imagery)
```

**Benefits:**
- Improved generalization across regions
- Better handling of seasonal/latitude effects
- Minimal overhead (~1% memory)

**Not currently used in Axion-Sat v1.0** (future enhancement).

---

## TerraTorch Integration

Both models are accessed via the **TerraTorch** unified model registry:

```python
from terratorch.models import build_model

# TerraMind (if available)
terramind = build_model("terramind_generator", config)

# Prithvi
prithvi = build_model("prithvi_600M", config)
```

### TerraTorch License
**Apache License 2.0**

- **Copyright:** © 2024 IBM Research, NASA, ESA
- **Repository:** [GitHub - ibm-nasa-geospatial/terratorch](https://github.com/ibm-nasa-geospatial/terratorch)

### Key Notes

1. **Model Availability:**
   - TerraMind may not yet be publicly released in TerraTorch (as of 2025-01)
   - Prithvi is publicly available and well-documented
   - Check TerraTorch releases for latest model availability

2. **Registry System:**
   - Models are registered by name (e.g., `"terramind_1.0_large"`, `"prithvi_eo_2.0_600M"`)
   - Weights are downloaded from HuggingFace on first use
   - Local caching in `~/.cache/huggingface/` or custom directory

3. **Version Compatibility:**
   - TerraTorch is under active development
   - Model APIs may change between versions
   - Pin TerraTorch version in `requirements.txt` for reproducibility

---

## Usage Guidelines for Axion-Sat

### 1. **Attribution**

When publishing work using these models, cite:

**TerraMind:**
```bibtex
@misc{terramind2025,
  title={TerraMind: A Foundation Model for Earth Observation},
  author={IBM Research and European Space Agency and Forschungszentrum Jülich},
  year={2025},
  url={https://huggingface.co/ibm-nasa-geospatial/terramind-1.0-large}
}
```

**Prithvi:**
```bibtex
@misc{prithvi2024,
  title={Prithvi: A Foundation Model for Geospatial Applications},
  author={IBM Research and NASA},
  year={2024},
  url={https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M}
}
```

### 2. **Commercial Use**

Both models permit commercial use under Apache-2.0:
- ✅ Can be used in commercial products
- ✅ Can be deployed in production systems
- ✅ No royalties or fees required

**Requirements:**
- Include Apache-2.0 license text in distribution
- Credit original developers
- Do not use trademarks (IBM, NASA, ESA) without permission

### 3. **Derivative Works**

You may modify the models (e.g., fine-tuning, pruning, quantization):
- ✅ Fine-tuned models can be distributed
- ✅ State changes in model card/documentation
- ✅ Derived models should acknowledge original work

**Best Practices:**
- Document fine-tuning procedure
- Share performance metrics on your task
- Contribute improvements back to TerraTorch (optional)

### 4. **Data Privacy**

When using these models with proprietary satellite data:
- ⚠️ Ensure data usage rights permit model training/inference
- ⚠️ Check export control regulations for sensitive regions
- ⚠️ Anonymize outputs if required by data provider

### 5. **Model Limitations**

Both models have known limitations:

**TerraMind:**
- Generates latents, not pixel-level images
- Quality varies by modality pair
- Requires significant VRAM (1B parameters)
- Limited temporal resolution (up to 48 timesteps)

**Prithvi:**
- Pre-training bias toward North America/Europe
- Optimal for 6-band input (reduced quality with fewer bands)
- Requires fine-tuning for best results
- Not real-time (inference ~0.5-1 sec per image)

**Validate on your specific use case before production deployment.**

---

## License Compliance Checklist

When deploying Axion-Sat:

- [ ] Include `LICENSE` file (Apache-2.0) in repository
- [ ] Credit IBM, NASA, ESA in documentation
- [ ] Cite model papers if publishing results
- [ ] Document any modifications to models
- [ ] Include model cards from HuggingFace in `docs/`
- [ ] Check data provider terms for satellite imagery
- [ ] Validate model outputs on target use case
- [ ] Test for biases in underrepresented regions

---

## Additional Resources

### Model Cards
- [TerraMind Model Card](https://huggingface.co/ibm-nasa-geospatial/terramind-1.0-large)
- [Prithvi Model Card](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M)

### Repositories
- [TerraTorch GitHub](https://github.com/ibm-nasa-geospatial/terratorch)
- [Prithvi Examples](https://github.com/NASA-IMPACT/prithvi-examples)

### Papers
- TerraMind: *"TerraMind: Foundation Models for Earth Observation via Multimodal Masked Autoencoding"* (2025)
- Prithvi: *"Foundation Models for Generalist Geospatial Artificial Intelligence"* (2024)

### License Text
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## Contact

For questions about model licensing:
- **TerraMind:** Contact IBM Research via HuggingFace or GitHub
- **Prithvi:** Contact NASA/IBM via model card
- **TerraTorch:** Open GitHub issue for technical support

For questions about Axion-Sat usage:
- See `README.md` and `docs/` directory
- Open GitHub issue in Axion-Sat repository

---

**Last Updated:** 2025-01-13  
**Axion-Sat Version:** 1.0.0  
**License:** Apache-2.0 (applies to Axion-Sat pipeline code)

**Note:** This document summarizes license terms based on public model cards. Always verify current licensing with official sources before deployment.
