# Security & Resource Management

This document outlines security policies, resource constraints, and data management practices for the Axion-Sat pipeline.

---

## Table of Contents

- [Area of Interest (AOI) Limits](#area-of-interest-aoi-limits)
- [Disk Quota Management](#disk-quota-management)
- [Temporary File Cleanup](#temporary-file-cleanup)
- [Output Data Quality](#output-data-quality)
- [Security Best Practices](#security-best-practices)

---

## Area of Interest (AOI) Limits

To prevent excessive resource consumption and ensure fair usage, AOI processing requests are subject to the following limits:

### Spatial Constraints

| Constraint | Limit | Rationale |
|------------|-------|-----------|
| **Maximum AOI Area** | 10,000 km² | Prevents excessive memory usage and processing time |
| **Maximum Bounding Box Dimension** | 200 km (per side) | Ensures reasonable tile counts and download sizes |
| **Minimum AOI Area** | 0.01 km² | Prevents degenerate geometries |
| **Maximum Vertices** | 1,000 points | Limits geometry complexity |

### Temporal Constraints

| Constraint | Limit | Rationale |
|------------|-------|-----------|
| **Maximum Time Range** | 1 year | Controls data volume from satellite APIs |
| **Maximum Concurrent Requests** | 5 AOIs | Prevents API rate limit violations |
| **Request Timeout** | 30 minutes | Prevents hung processes |

### Enforcement

```python
# Example: AOI validation (pseudo-code)
def validate_aoi(geometry):
    area_km2 = calculate_area(geometry)
    
    if area_km2 > 10_000:
        raise ValueError("AOI exceeds maximum area of 10,000 km²")
    
    if area_km2 < 0.01:
        raise ValueError("AOI below minimum area of 0.01 km²")
    
    bbox = geometry.bounds
    width_km = haversine_distance(bbox[0], bbox[1], bbox[2], bbox[1])
    height_km = haversine_distance(bbox[0], bbox[1], bbox[0], bbox[3])
    
    if max(width_km, height_km) > 200:
        raise ValueError("AOI bounding box exceeds 200 km limit")
    
    if len(geometry.exterior.coords) > 1000:
        raise ValueError("AOI has too many vertices (max: 1000)")
```

### Override Policy

Administrative overrides for these limits require:
1. Written justification
2. Approval from project lead
3. Resource availability confirmation
4. Documented in processing logs

---

## Disk Quota Management

The pipeline implements disk quota guards to prevent storage exhaustion and ensure predictable resource usage.

### Quota Limits

| Directory | Quota | Enforcement |
|-----------|-------|-------------|
| `cache/` | 50 GB | Hard limit - oldest files deleted automatically |
| `outputs/` | 100 GB | Soft limit - warning at 80%, error at 100% |
| `weights/` | 20 GB | Hard limit - must manually clean |
| `temp/` | 10 GB | Hard limit - cleaned automatically |
| `data/raw/` | 200 GB | Soft limit - monitoring only |

### Quota Monitoring

```python
# Automatic quota checking before major operations
@enforce_quota(directory="outputs", limit_gb=100)
def generate_prediction_map(aoi, model, ...):
    # Processing happens here
    ...
```

### Disk Space Alerts

| Threshold | Action |
|-----------|--------|
| **< 20% free** | ⚠️ Warning logged, notification sent |
| **< 10% free** | ❌ New processing requests rejected |
| **< 5% free** | 🛑 Pipeline paused, emergency cleanup triggered |

### Quota Enforcement Strategy

1. **Proactive Checks**: Verify available space before starting operations
2. **Graceful Degradation**: Warn users and reduce cache size before hard failure
3. **Automatic Cleanup**: Remove old temporary files and expired cache entries
4. **User Notification**: Clear error messages with remediation steps

```plaintext
Example Error Message:
❌ Disk quota exceeded in outputs/ (105.3 GB / 100 GB limit)

Suggested actions:
  1. Archive or delete old predictions: outputs/predictions_2024-01/
  2. Compress large files: outputs/*.tif → *.tif.gz
  3. Move results to external storage
  4. Request quota increase (contact: admin@axion-sat.example)
```

---

## Temporary File Cleanup

The pipeline generates temporary files during processing. These are managed through an automated cleanup policy.

### Temporary File Locations

```plaintext
temp/
├── tiles/              # Downloaded satellite tiles
├── preprocessing/      # Intermediate processing artifacts
├── model_cache/        # Temporary model outputs
├── geospatial/         # Reprojection buffers, intermediate shapefiles
└── visualizations/     # Temporary rendering buffers
```

### Cleanup Policy

| File Type | Retention | Cleanup Trigger |
|-----------|-----------|-----------------|
| **Satellite Tiles** | 24 hours | Age-based, on-exit |
| **Preprocessing Artifacts** | 1 hour | On-success, on-exit |
| **Model Cache** | 6 hours | Age-based |
| **Geospatial Buffers** | 2 hours | On-success, on-exit |
| **Visualization Buffers** | 30 minutes | Immediate on-success |

### Cleanup Triggers

1. **Automatic Cleanup**: Runs every 4 hours via scheduled task
2. **On-Exit Cleanup**: Best-effort cleanup when pipeline exits normally
3. **On-Success Cleanup**: Immediate cleanup after successful operation
4. **Emergency Cleanup**: Triggered when `temp/` exceeds 10 GB quota

### Cleanup Implementation

```python
# Example cleanup configuration
CLEANUP_POLICY = {
    "tiles": {"max_age_hours": 24, "trigger": "age"},
    "preprocessing": {"max_age_hours": 1, "trigger": "on_success"},
    "model_cache": {"max_age_hours": 6, "trigger": "age"},
    "geospatial": {"max_age_hours": 2, "trigger": "on_success"},
    "visualizations": {"max_age_hours": 0.5, "trigger": "immediate"},
}

# Emergency cleanup: delete oldest 50% of temp files
def emergency_cleanup():
    files = sorted(temp_files_by_age(), key=lambda f: f.mtime)
    for file in files[:len(files) // 2]:
        safe_delete(file)
```

### Manual Cleanup

Users can manually trigger cleanup:

```bash
# Clean all temporary files
python scripts/cleanup_temp.py --all

# Clean specific category
python scripts/cleanup_temp.py --category tiles

# Clean files older than X hours
python scripts/cleanup_temp.py --older-than 6

# Dry run (show what would be deleted)
python scripts/cleanup_temp.py --all --dry-run
```

### Critical File Protection

Temporary files currently in use are protected via file locks:

```python
with FileLock(temp_file_path):
    # File is protected from cleanup while processing
    process_temp_file(temp_file_path)
# Lock released, file can be cleaned up
```

---

## Output Data Quality

### Synthetic Data Notice

⚠️ **IMPORTANT**: All pipeline outputs are **synthetic predictions** generated by machine learning models. They are **not** direct observations or ground truth.

### Output Characteristics

| Property | Description |
|----------|-------------|
| **Data Type** | Model predictions, not sensor measurements |
| **Confidence** | Includes uncertainty estimates and confidence scores |
| **Resolution** | May differ from input satellite resolution |
| **Temporal Lag** | Predictions based on historical satellite imagery |
| **Validation** | Subject to quality control layers (see below) |

### Quality Control (QC) Layers

All outputs include QC metadata to help users assess reliability:

#### 1. **Confidence Score**
- **Range**: 0.0 (low confidence) to 1.0 (high confidence)
- **Included**: Per-pixel confidence map
- **Interpretation**: Higher scores indicate more reliable predictions

#### 2. **Cloud Contamination Flag**
- **Values**: `clear` | `partial` | `cloudy`
- **Impact**: Cloudy areas have degraded prediction quality
- **Recommendation**: Mask predictions where `cloud_coverage > 0.3`

#### 3. **Model Version Tag**
- **Format**: `model_v{major}.{minor}_{training_date}`
- **Purpose**: Track model provenance and enable reproducibility
- **Example**: `model_v2.1_20241015`

#### 4. **Input Data Quality**
- **Metrics**: Satellite coverage percentage, pixel gaps, atmospheric correction status
- **Included**: Per-AOI summary statistics
- **Use**: Identify areas where input data was incomplete

#### 5. **Temporal Currency**
- **Timestamp**: Latest satellite acquisition used in prediction
- **Age Warning**: Flagged if input data > 30 days old
- **Use**: Assess prediction freshness

### QC Metadata Format

```json
{
  "prediction_id": "pred_20241013_aoi_001",
  "model_version": "model_v2.1_20241015",
  "timestamp": "2024-10-13T05:46:45Z",
  "qc": {
    "mean_confidence": 0.87,
    "confidence_below_threshold": 0.05,
    "cloud_contamination": "partial",
    "cloud_coverage_percent": 12.3,
    "input_data_age_days": 5,
    "input_data_completeness": 0.98,
    "spatial_coverage_percent": 99.7,
    "atmospheric_corrected": true
  },
  "warnings": [
    "Cloud coverage exceeds 10% in northern region",
    "Partial tile coverage in southwest corner"
  ],
  "quality_flags": {
    "reliable": true,
    "production_ready": true,
    "requires_review": false
  }
}
```

### User Guidelines

✅ **DO**:
- Check confidence scores before using predictions
- Review QC flags and warnings
- Validate predictions against ground truth when available
- Report anomalies or quality issues

❌ **DON'T**:
- Use predictions as ground truth without validation
- Ignore low confidence areas in critical applications
- Use outdated predictions (> 90 days) without review
- Assume 100% accuracy

### Validation Recommendations

For mission-critical applications:
1. Compare predictions against independent ground truth (if available)
2. Perform visual spot-checks of high-confidence areas
3. Cross-validate with alternative data sources (e.g., field surveys)
4. Monitor temporal consistency across multiple prediction runs
5. Document any discrepancies for model improvement

---

## Security Best Practices

### Data Access Control

- **Principle**: Least privilege access
- **Authentication**: Required for API access to satellite data providers
- **Credentials**: Stored in environment variables or secure vaults (never hardcoded)
- **API Keys**: Rotated every 90 days

### Network Security

- **HTTPS Only**: All external API calls use encrypted connections
- **Certificate Validation**: SSL certificate checking enabled
- **Timeouts**: Network requests timeout after 30 seconds
- **Rate Limiting**: Respect provider rate limits to avoid IP blocking

### Logging and Auditing

- **Access Logs**: All AOI requests logged with timestamp, user, and parameters
- **Error Logs**: Security-relevant errors logged separately
- **PII Handling**: No personally identifiable information in logs
- **Log Retention**: 90 days for operational logs, 1 year for security logs

### Dependency Management

- **Vulnerability Scanning**: Regular scans with `pip-audit` or `safety`
- **Update Policy**: Security patches applied within 7 days
- **Pinned Versions**: All dependencies version-pinned in `requirements.txt`
- **License Compliance**: Verify all dependencies have compatible licenses

### Incident Response

In case of security incidents:
1. **Isolate**: Stop affected services immediately
2. **Assess**: Determine scope and impact
3. **Notify**: Alert project lead and stakeholders
4. **Remediate**: Apply fixes and validate
5. **Document**: Post-mortem report with lessons learned

---

## Compliance & Reporting

### Regular Reviews

- **Monthly**: Disk quota usage review
- **Quarterly**: Security audit and dependency updates
- **Annually**: Full policy review and update

### Metrics & Monitoring

```plaintext
Key Performance Indicators:
- Average disk usage per AOI
- Cleanup job success rate
- Quota violation incidents
- QC flag distribution
- User-reported quality issues
```

### Contact

For questions or concerns about security and resource policies:
- **Technical Issues**: Open an issue in the project repository
- **Security Concerns**: Contact project lead directly
- **Quota Requests**: Submit via issue tracker with justification

---

**Document Version**: 1.0  
**Last Updated**: 2024-10-13  
**Next Review**: 2025-01-13  
**Owner**: Axion-Sat Development Team
