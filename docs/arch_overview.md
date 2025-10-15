# Architecture Overview

This document provides a high-level overview of the Axion-Sat pipeline architecture, describing the flow of data from satellite imagery sources through the GAC (Geospatial AI Compute) processing pipeline to the end-user visualization.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Pipeline Flow Diagram](#pipeline-flow-diagram)
- [Component Details](#component-details)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Deployment Architecture](#deployment-architecture)

---

## System Architecture

The Axion-Sat pipeline implements a multi-stage geospatial AI processing system that:

1. **Ingests** satellite imagery from multiple STAC catalogs
2. **Preprocesses** and tiles imagery for efficient model processing
3. **Applies** a 3-stage deep learning pipeline for vegetation cover prediction
4. **Performs** quality control and outputs standardized geospatial formats
5. **Serves** predictions via a RESTful API
6. **Visualizes** results in an interactive web-based map viewer

---

## Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AXION-SAT GAC PIPELINE                              │
│                      Geospatial AI Compute Architecture                      │
└─────────────────────────────────────────────────────────────────────────────┘

                                 ┌─────────────┐
                                 │   User AOI  │
                                 │  (GeoJSON)  │
                                 └──────┬──────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  STAGE 0: DATA ACQUISITION                                                 │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                  │
│  │ STAC Earth   │   │ STAC MPC     │   │ STAC LPCLOUD │                  │
│  │ Search (AWS) │   │ (Microsoft)  │   │ (NASA)       │                  │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                  │
│         │                  │                  │                           │
│         └──────────────────┼──────────────────┘                           │
│                            │                                               │
│                     ┌──────▼───────┐                                       │
│                     │ STAC Client  │                                       │
│                     │ Query & Fetch│                                       │
│                     └──────┬───────┘                                       │
│                            │                                               │
│                            ▼                                               │
│                  ┌────────────────────┐                                    │
│                  │ Sentinel-2 / L8/9  │                                    │
│                  │ Tiles (COGs)       │                                    │
│                  └─────────┬──────────┘                                    │
└────────────────────────────┼──────────────────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  PREPROCESSING & TILING                                                    │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────┐        │
│  │ Band Selection  │──▶│ Atmospheric      │──▶│ Normalization   │        │
│  │ (RGB, NIR, etc.)│   │ Correction       │   │ (0-1 scaling)   │        │
│  └─────────────────┘   └──────────────────┘   └────────┬────────┘        │
│                                                         │                  │
│                                                         ▼                  │
│                                              ┌──────────────────┐          │
│                                              │ Tiling (512×512) │          │
│                                              │ Overlap: 64px    │          │
│                                              └────────┬─────────┘          │
│                                                       │                    │
│                                                       ▼                    │
│                                              ┌──────────────────┐          │
│                                              │ Cache to Disk    │          │
│                                              │ (./cache/)       │          │
│                                              └────────┬─────────┘          │
└──────────────────────────────────────────────────────┼─────────────────────┘
                                                       │
                                                       ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: FEATURE EXTRACTION                                               │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         ┌────────────────────┐                             │
│                         │   TerraMind S1     │                             │
│                         │ (Sentinel-1 ViT)   │                             │
│                         ├────────────────────┤                             │
│                         │ • Vision Trans.    │                             │
│                         │ • SAR features     │                             │
│                         │ • Texture analysis │                             │
│                         └─────────┬──────────┘                             │
│                                   │                                        │
│                                   ▼                                        │
│                         ┌────────────────────┐                             │
│                         │   TerraMind S2     │                             │
│                         │ (Sentinel-2 ViT)   │                             │
│                         ├────────────────────┤                             │
│                         │ • Multispectral    │                             │
│                         │ • NDVI/NDWI calc   │                             │
│                         │ • Feature fusion   │                             │
│                         └─────────┬──────────┘                             │
│                                   │                                        │
│                                   ▼                                        │
│                         ┌────────────────────┐                             │
│                         │ Feature Maps       │                             │
│                         │ (256-dim embeddings)│                            │
│                         └─────────┬──────────┘                             │
└───────────────────────────────────┼───────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: SEGMENTATION REFINEMENT                                          │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         ┌────────────────────┐                             │
│                         │  Prithvi Refiner   │                             │
│                         │  (Foundation Model)│                             │
│                         ├────────────────────┤                             │
│                         │ • Temporal fusion  │                             │
│                         │ • U-Net decoder    │                             │
│                         │ • Edge refinement  │                             │
│                         │ • Multi-scale pred │                             │
│                         └─────────┬──────────┘                             │
│                                   │                                        │
│                                   ▼                                        │
│                         ┌────────────────────┐                             │
│                         │ Segmentation Mask  │                             │
│                         │ (H×W, per-pixel)   │                             │
│                         └─────────┬──────────┘                             │
└───────────────────────────────────┼───────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: CONDITIONAL GROUNDING                                            │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                   ┌──────────────────────────────┐                         │
│                   │  TerraMind Conditional Model │                         │
│                   │  (Semantic Grounding)        │                         │
│                   ├──────────────────────────────┤                         │
│                   │ • Vegetation class mapping   │                         │
│                   │ • Confidence scoring         │                         │
│                   │ • Context-aware refinement   │                         │
│                   │ • Boundary smoothing         │                         │
│                   └──────────────┬───────────────┘                         │
│                                  │                                         │
│                                  ▼                                         │
│                   ┌──────────────────────────────┐                         │
│                   │ Final Prediction Map         │                         │
│                   │ • Per-pixel classification   │                         │
│                   │ • Confidence scores (0-1)    │                         │
│                   │ • Vegetation density (%)     │                         │
│                   └──────────────┬───────────────┘                         │
└──────────────────────────────────┼───────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  QUALITY CONTROL & OUTPUT FORMATTING                                       │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐                                                      │
│  │ QC Pipeline      │                                                      │
│  ├──────────────────┤                                                      │
│  │ • Confidence > 0.75 filter                                              │
│  │ • Cloud mask application                                                │
│  │ • Spatial consistency check                                             │
│  │ • Temporal anomaly detection                                            │
│  └────────┬─────────┘                                                      │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐      │
│  │ COG Generation   │   │ STAC Metadata    │   │ Stats & Metrics  │      │
│  │ (GeoTIFF)        │   │ (JSON)           │   │ (JSON)           │      │
│  └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘      │
│           │                      │                       │                 │
│           └──────────────────────┼───────────────────────┘                 │
│                                  │                                         │
│                                  ▼                                         │
│                       ┌────────────────────┐                               │
│                       │ ./outputs/         │                               │
│                       │ • predictions.tif  │                               │
│                       │ • metadata.json    │                               │
│                       │ • qc_report.json   │                               │
│                       └──────────┬─────────┘                               │
└──────────────────────────────────┼───────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  API SERVER (FastAPI)                                                      │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────┐              │
│  │  RESTful Endpoints                                      │              │
│  ├─────────────────────────────────────────────────────────┤              │
│  │  POST   /api/v1/predict          # Submit AOI          │              │
│  │  GET    /api/v1/status/{id}      # Check job status    │              │
│  │  GET    /api/v1/results/{id}     # Fetch results       │              │
│  │  GET    /api/v1/tiles/{z}/{x}/{y} # Map tiles          │              │
│  │  GET    /api/v1/health           # Health check        │              │
│  │  GET    /docs                    # Swagger UI          │              │
│  └────────────────────────┬────────────────────────────────┘              │
│                           │                                                │
│                           │  Port 7860                                     │
│                           │  CORS enabled                                  │
│                           │  JSON responses                                │
│                           │                                                │
└───────────────────────────┼───────────────────────────────────────────────┘
                            │
                            │  HTTP/REST
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  FRONTEND: LEAFLET MAP VIEWER                                              │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────┐              │
│  │  Interactive Web Map                                    │              │
│  ├─────────────────────────────────────────────────────────┤              │
│  │  ┌────────────────────────────────────────────────┐     │              │
│  │  │                                                 │     │              │
│  │  │  🗺️  Leaflet.js Map Canvas                     │     │              │
│  │  │                                                 │     │              │
│  │  │  • Base layers (OSM, satellite)                │     │              │
│  │  │  • Prediction overlay (colored)                │     │              │
│  │  │  • Confidence heatmap                          │     │              │
│  │  │  • AOI boundary                                │     │              │
│  │  │  • Legend & controls                           │     │              │
│  │  │                                                 │     │              │
│  │  └────────────────────────────────────────────────┘     │              │
│  │                                                          │              │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐   │              │
│  │  │ Draw AOI     │  │ Query        │  │ Export      │   │              │
│  │  │ Tool         │  │ Metadata     │  │ Results     │   │              │
│  │  └──────────────┘  └──────────────┘  └─────────────┘   │              │
│  │                                                          │              │
│  └──────────────────────────────────────────────────────────┘              │
│                                                                             │
│  Technologies: Leaflet.js, Leaflet.draw, GeoJSON, Vector tiles            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

                            ┌──────────────┐
                            │     USER     │
                            │ (Web Browser)│
                            └──────────────┘
```

---

## Component Details

### **1. Data Acquisition (Stage 0)**

**Purpose**: Fetch satellite imagery from STAC catalogs based on user-defined AOI.

**Components**:
- **STAC Clients**: Connect to Earth Search (AWS), Microsoft Planetary Computer, NASA LPCLOUD
- **Query Engine**: Filters by AOI, date range, cloud coverage, resolution
- **Tile Downloader**: Fetches Cloud-Optimized GeoTIFFs (COGs)

**Inputs**:
- User-defined Area of Interest (GeoJSON polygon)
- Date range (start, end)
- Satellite preferences (Sentinel-2, Landsat 8/9)

**Outputs**:
- Downloaded satellite tiles (COGs)
- Cached in `./cache/tiles/`

**Key Technologies**:
- `pystac-client` - STAC API interaction
- `rasterio` - Geospatial raster I/O
- `requests` - HTTP downloads

---

### **2. Preprocessing & Tiling**

**Purpose**: Prepare satellite imagery for model ingestion.

**Steps**:
1. **Band Selection**: Extract relevant bands (RGB, NIR, SWIR)
2. **Atmospheric Correction**: Apply Top-of-Atmosphere (TOA) or Surface Reflectance (SR) corrections
3. **Normalization**: Scale pixel values to [0, 1] range
4. **Tiling**: Divide large images into 512×512 tiles with 64-pixel overlap
5. **Caching**: Store preprocessed tiles for reuse

**Outputs**:
- Preprocessed tiles (NumPy arrays or GeoTIFFs)
- Metadata (tile coordinates, CRS, resolution)

**Key Technologies**:
- `numpy` - Array manipulation
- `rasterio` - Geospatial operations
- `affine` - Coordinate transformations

---

### **3. Stage 1: Feature Extraction (TerraMind S1/S2)**

**Purpose**: Extract semantic features from satellite imagery using Vision Transformers.

**Models**:
- **TerraMind S1**: Processes Sentinel-1 SAR data
  - Architecture: Vision Transformer (ViT)
  - Features: Texture, backscatter intensity, coherence
  - Output: 256-dimensional feature embeddings

- **TerraMind S2**: Processes Sentinel-2 multispectral data
  - Architecture: Vision Transformer (ViT)
  - Features: Spectral indices (NDVI, NDWI), reflectance patterns
  - Output: 256-dimensional feature embeddings

**Fusion**: Concatenate S1 and S2 features for multi-modal representation

**Outputs**:
- Feature maps (H×W×256)
- Attention weights (for visualization)

**Key Technologies**:
- `PyTorch` - Deep learning framework
- `timm` - Vision Transformer models
- `einops` - Tensor operations

---

### **4. Stage 2: Segmentation Refinement (Prithvi)**

**Purpose**: Generate dense per-pixel segmentation masks using a foundation model.

**Model**: Prithvi (IBM/NASA Foundation Model)
- **Architecture**: U-Net-style encoder-decoder
- **Capabilities**:
  - Temporal fusion (multi-date imagery)
  - Multi-scale prediction
  - Edge refinement
  - Transfer learning from large-scale pre-training

**Inputs**: Feature maps from Stage 1

**Outputs**:
- Segmentation mask (H×W, per-pixel class labels)
- Intermediate feature maps

**Key Technologies**:
- `PyTorch` - Model inference
- `torchvision` - Image transformations
- Prithvi checkpoint (HuggingFace or custom)

---

### **5. Stage 3: Conditional Grounding (TerraMind)**

**Purpose**: Apply semantic grounding and context-aware refinement to segmentation masks.

**Model**: TerraMind Conditional Model
- **Architecture**: Conditional GAN or transformer-based refiner
- **Capabilities**:
  - Vegetation class mapping (forest, grassland, cropland, etc.)
  - Confidence scoring (0-1 per pixel)
  - Context-aware boundary smoothing
  - Semantic consistency enforcement

**Inputs**: Segmentation mask from Stage 2

**Outputs**:
- Final prediction map (H×W, vegetation classes)
- Per-pixel confidence scores (H×W, 0-1 range)
- Vegetation density percentage

**Key Technologies**:
- `PyTorch` - Model inference
- Custom conditional layers

---

### **6. Quality Control & Output Formatting**

**Purpose**: Filter low-quality predictions and export standardized geospatial outputs.

**QC Pipeline**:
1. **Confidence Filtering**: Reject pixels with confidence < 0.75
2. **Cloud Masking**: Apply cloud/shadow masks from satellite metadata
3. **Spatial Consistency**: Detect and flag isolated anomalies
4. **Temporal Anomaly Detection**: Flag predictions inconsistent with historical data

**Output Formats**:
- **COG (Cloud-Optimized GeoTIFF)**: Prediction raster with internal tiling and overviews
- **STAC Metadata**: JSON catalog entry with spatiotemporal info
- **QC Report**: JSON with metrics (mean confidence, cloud coverage, etc.)

**Key Technologies**:
- `rasterio` - COG generation
- `pystac` - STAC metadata creation
- `json` - Structured outputs

---

### **7. API Server (FastAPI)**

**Purpose**: Serve predictions and tiles via RESTful API.

**Endpoints**:

| Method | Endpoint                    | Description                  |
|--------|-----------------------------|------------------------------|
| POST   | `/api/v1/predict`           | Submit AOI for processing    |
| GET    | `/api/v1/status/{id}`       | Check job status             |
| GET    | `/api/v1/results/{id}`      | Fetch prediction results     |
| GET    | `/api/v1/tiles/{z}/{x}/{y}` | Serve map tiles (XYZ)        |
| GET    | `/api/v1/health`            | Health check                 |
| GET    | `/docs`                     | Swagger UI (API docs)        |

**Features**:
- Asynchronous request handling
- Job queue management
- CORS enabled for web frontends
- Authentication (optional, via API keys)

**Key Technologies**:
- `FastAPI` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Request/response validation
- `aiofiles` - Async file I/O

---

### **8. Frontend: Leaflet Map Viewer**

**Purpose**: Interactive web-based visualization of predictions.

**Features**:
- **Base Layers**: OpenStreetMap, satellite imagery
- **Prediction Overlay**: Colored vegetation classification map
- **Confidence Heatmap**: Visual representation of prediction certainty
- **AOI Drawing**: Interactive polygon drawing for new queries
- **Legend**: Color-coded vegetation classes
- **Export**: Download results as GeoJSON, GeoTIFF, or PNG

**Interaction Flow**:
1. User draws AOI on map
2. AOI submitted to API (`/api/v1/predict`)
3. Job ID returned, polling for status (`/api/v1/status/{id}`)
4. On completion, tiles loaded (`/api/v1/tiles/{z}/{x}/{y}`)
5. Results displayed as overlay on map

**Key Technologies**:
- `Leaflet.js` - Mapping library
- `Leaflet.draw` - Drawing tools
- `GeoJSON` - Geometry format
- `HTML/CSS/JavaScript` - Frontend stack

---

## Data Flow

### **End-to-End Workflow**

```
User Input (AOI) 
  → STAC Query 
    → Download Tiles 
      → Preprocess & Tile 
        → Stage 1 (TerraMind S1/S2) 
          → Stage 2 (Prithvi) 
            → Stage 3 (TerraMind Conditional) 
              → QC & COG Export 
                → API Response 
                  → Leaflet Visualization
```

### **Data Formats by Stage**

| Stage                  | Format                     | Size (Approx.)    |
|------------------------|----------------------------|-------------------|
| Raw Satellite Tiles    | COG (GeoTIFF)              | 100-500 MB        |
| Preprocessed Tiles     | NumPy (.npy) or GeoTIFF    | 50-200 MB         |
| Feature Maps (S1)      | PyTorch Tensor (.pt)       | 10-50 MB          |
| Feature Maps (S2)      | PyTorch Tensor (.pt)       | 10-50 MB          |
| Segmentation Mask      | NumPy (.npy)               | 5-20 MB           |
| Final Prediction       | COG (GeoTIFF)              | 10-30 MB          |
| STAC Metadata          | JSON                       | 5-20 KB           |
| QC Report              | JSON                       | 1-5 KB            |

---

## Technology Stack

### **Core Libraries**

| Component              | Technology                          |
|------------------------|-------------------------------------|
| **Programming Language** | Python 3.11                       |
| **Deep Learning**      | PyTorch 2.1+, torchvision          |
| **Geospatial**         | rasterio, GDAL, geopandas, shapely |
| **STAC**               | pystac, pystac-client              |
| **API Framework**      | FastAPI, uvicorn                   |
| **Frontend**           | Leaflet.js, HTML/CSS/JS            |
| **Visualization**      | matplotlib, folium (optional)      |
| **Data Science**       | numpy, pandas, scikit-learn        |

### **Infrastructure**

| Component              | Technology                          |
|------------------------|-------------------------------------|
| **Compute**            | GPU (CUDA-enabled, 8+ GB VRAM)     |
| **Storage**            | Local disk (200+ GB recommended)   |
| **Web Server**         | Uvicorn (ASGI), Nginx (reverse proxy) |
| **Containerization**   | Docker (optional)                  |
| **Orchestration**      | Docker Compose (optional)          |

---

## Deployment Architecture

### **Single-Server Deployment**

```
┌────────────────────────────────────────────┐
│           Server (Windows/Linux)           │
├────────────────────────────────────────────┤
│                                            │
│  ┌──────────────────────────────────────┐ │
│  │  FastAPI (Port 7860)                 │ │
│  │  ├─ Uvicorn ASGI server              │ │
│  │  ├─ Worker pool (4 workers)          │ │
│  │  └─ GPU access via PyTorch           │ │
│  └──────────────────────────────────────┘ │
│                                            │
│  ┌──────────────────────────────────────┐ │
│  │  Static Frontend (HTML/JS)           │ │
│  │  └─ Served by FastAPI or Nginx       │ │
│  └──────────────────────────────────────┘ │
│                                            │
│  ┌──────────────────────────────────────┐ │
│  │  File System                          │ │
│  │  ├─ ./cache/     (50 GB)             │ │
│  │  ├─ ./outputs/   (100 GB)            │ │
│  │  ├─ ./weights/   (20 GB)             │ │
│  │  └─ ./temp/      (10 GB)             │ │
│  └──────────────────────────────────────┘ │
│                                            │
└────────────────────────────────────────────┘
```

### **Scalable Deployment (Optional)**

For high-throughput production workloads:

- **Load Balancer**: Nginx or HAProxy distributing requests
- **Worker Pool**: Multiple FastAPI instances (horizontal scaling)
- **Redis Queue**: Celery or RQ for async job processing
- **Distributed Storage**: S3 or MinIO for outputs
- **Model Serving**: TorchServe or TensorRT for optimized inference
- **Monitoring**: Prometheus + Grafana for metrics

---

## Performance Considerations

### **Throughput**

- **Single AOI Processing Time**: 5-15 minutes (depending on AOI size)
- **Concurrent Requests**: Up to 4 (limited by GPU memory)
- **Tile Processing Rate**: ~10-20 tiles/second (Stage 1)

### **Resource Requirements**

| Resource       | Minimum     | Recommended      |
|----------------|-------------|------------------|
| **GPU VRAM**   | 8 GB        | 16+ GB           |
| **RAM**        | 16 GB       | 32+ GB           |
| **Disk Space** | 100 GB      | 500+ GB          |
| **CPU**        | 4 cores     | 8+ cores         |

### **Optimization Strategies**

- **Model Quantization**: FP16 or INT8 for faster inference
- **Batch Processing**: Process multiple tiles in parallel
- **Caching**: Reuse preprocessed tiles for repeated queries
- **COG Optimization**: Enable overviews and internal tiling
- **Tile Prefetching**: Download tiles in background while processing

---

## Security & Access Control

- **API Authentication**: Optional API key or OAuth2
- **Rate Limiting**: Prevent abuse (60 requests/minute)
- **CORS Policy**: Restrict to trusted origins
- **Input Validation**: Reject invalid geometries or excessive AOI sizes
- **Output Sanitization**: Strip sensitive metadata before serving

---

## Future Enhancements

- **Real-time Processing**: WebSocket-based streaming of results
- **Multi-temporal Analysis**: Change detection across time series
- **3D Visualization**: Terrain-aware rendering with elevation data
- **Mobile App**: Native iOS/Android clients
- **Federated Learning**: Privacy-preserving model training
- **Blockchain Provenance**: Immutable prediction metadata tracking

---

## References

- **STAC Specification**: https://stacspec.org/
- **Prithvi Model**: IBM/NASA Geospatial Foundation Model
- **TerraMind**: Custom vegetation cover prediction models
- **Leaflet.js**: https://leafletjs.com/
- **FastAPI**: https://fastapi.tiangolo.com/

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-13  
**Author**: Axion-Sat Development Team  
**Contact**: See project repository for issues and discussions
