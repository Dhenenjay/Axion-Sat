# Axion-Sat

## Project Mission

Axion-Sat is an advanced machine learning platform designed to revolutionize satellite imagery analysis by leveraging state-of-the-art transformer architectures and geospatial AI techniques to extract actionable intelligence from Earth observation data. By combining cutting-edge deep learning models with comprehensive geospatial data processing pipelines, Axion-Sat enables automated detection, classification, and monitoring of critical infrastructure, environmental changes, and land use patterns at scale, empowering researchers, governments, and organizations to make data-driven decisions for sustainable development, disaster response, and strategic planning across diverse geographic regions.

## Project Structure

```
Axion-Sat/
├── app/                    # Main application code and entry points
├── vc_lib/                 # Version-controlled library modules and utilities
├── weights/                # Model weights and checkpoints (use Git LFS)
├── configs/                # Configuration files (YAML, JSON, TOML)
├── cache/                  # Temporary cache for data processing
├── data/                   # Dataset directory
│   └── raw/               # Raw satellite imagery (gitignored)
├── outputs/                # Training outputs and predictions
│   └── _tmp/              # Temporary outputs (gitignored)
├── logs/                   # Training and application logs
├── scripts/                # Setup and automation scripts
│   ├── 00_env_bootstrap.ps1    # Environment setup
│   ├── 01_python_env.ps1       # Python venv setup
│   └── 02_gpu_smoke.ps1        # GPU verification
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation and guides
├── notebooks/              # Jupyter notebooks for experiments
└── tools/                  # Development and utility tools
```

## Quick Start

### Prerequisites

- **OS:** Windows 11
- **GPU:** NVIDIA RTX 4050 (6 GB VRAM) or better
- **CUDA:** 12.1+
- **Python:** 3.11
- **Drivers:** NVIDIA GPU drivers ≥ 535

### Setup

1. **Bootstrap Environment** (Run as Administrator):
   ```powershell
   .\scripts\00_env_bootstrap.ps1
   ```

2. **Setup Python Environment**:
   ```powershell
   .\scripts\01_python_env.ps1
   ```

3. **Verify GPU**:
   ```powershell
   .\scripts\02_gpu_smoke.ps1
   ```

4. **Activate Virtual Environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

## Technology Stack

### Core ML Framework
- **PyTorch** 2.4.1 with CUDA 12.1
- **xformers** for efficient attention mechanisms
- **Transformers** (Hugging Face) for state-of-the-art models
- **Terratorch** for geospatial AI workflows
- **PyTorch Lightning** for scalable training

### Geospatial Processing
- **GDAL** 3.11.1 for raster operations
- **Rasterio** for satellite imagery I/O
- **GeoPandas** for vector data processing
- **STAC** (pystac-client, stackstac) for satellite data discovery
- **Xarray** for multidimensional arrays

### Supporting Libraries
- **Albumentations** for data augmentation
- **Kornia** for differentiable computer vision
- **FastAPI** for API services
- **MLflow** for experiment tracking

## System Specifications

- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB)
- **Compute Capability:** 8.9
- **CUDA Version:** 12.1
- **PyTorch Version:** 2.4.1+cu121

## Development

### Running Tests
```powershell
pytest tests/
```

### Starting Jupyter Lab
```powershell
jupyter lab notebooks/
```

### Training a Model
```powershell
python app/train.py --config configs/model_config.yaml
```

## Contributing

Please read our contribution guidelines in `docs/CONTRIBUTING.md` before submitting pull requests.

## License

[Specify your license here]

## Contact

[Add contact information or project links]

---

**Built with ❤️ for Earth observation and geospatial intelligence**
