# OES/LIBS Spectral Analysis

Machine Learning pipeline for Optical Emission Spectroscopy (OES) and
Laser-Induced Breakdown Spectroscopy (LIBS) data analysis.

## Overview

This toolkit provides an end-to-end workflow for spectral data:

1. **Data Loading** — LIBS Benchmark (HDF5), Mesbah CAP (CSV), BOSCH Plasma (NetCDF)
2. **Preprocessing** — cosmic-ray removal, baseline correction, SNV normalisation
3. **Feature Extraction** — CWT peak detection, plasma descriptor features
4. **Classification & Regression** — SVM, Random Forest, 1D-CNN, LSTM
5. **Temporal Analysis** — PCA embedding, DTW clustering, LSTM prediction
6. **Spatial Analysis** — wafer uniformity, RBF interpolation heatmaps
7. **Evaluation** — cross-validation, ablation studies, calibration metrics

## Module Index

| Module | Description |
|--------|-------------|
| [`src.data_loader`](api/data_loader.md) | Dataset loaders for LIBS, Mesbah, BOSCH OES |
| [`src.preprocessing`](api/preprocessing.md) | Spectral preprocessing pipeline |
| [`src.features`](api/features.md) | Peak detection and plasma descriptors |
| [`src.spatial`](api/spatial.md) | Wafer spatial uniformity analysis |
| [`src.temporal`](api/temporal.md) | Temporal embedding and phase clustering |
| [`src.evaluation`](api/evaluation.md) | Model evaluation and comparison |
| [`src.guardrail`](api/guardrail.md) | Input validation and safety checks |
| [`src.target_transform`](api/target_transform.md) | Target variable transforms |
| [`src.models.traditional`](api/models_traditional.md) | SVM, RF, ensemble models |
| [`src.models.deep_learning`](api/models_deep_learning.md) | Conv1D, LSTM, Transformer |
| [`src.models.calibration`](api/models_calibration.md) | Probability calibration |

## Quick Start

```bash
pip install -e .
python main.py classify --data data/libs_benchmark/ --model svm
python scripts/plot_spatial.py --data data/bosch_oes/ --metric oxide_etch
```
