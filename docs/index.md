# Plasma OES Spectral Analysis

Machine Learning pipeline for Plasma Optical Emission Spectroscopy (OES) data analysis.

## Overview

This toolkit provides an end-to-end workflow for spectral data:

1. **Data Loading** — Mesbah CAP (CSV), BOSCH Plasma Etching (NetCDF)
2. **Preprocessing** — cosmic-ray removal, baseline correction, SNV normalisation
3. **Feature Extraction** — peak detection, plasma descriptor features
4. **Classification & Regression** — SVM, Random Forest, ANN, Ridge, PLS
5. **Temporal Analysis** — PCA embedding, DTW clustering, LSTM prediction
6. **Spatial Analysis** — wafer uniformity, RBF interpolation heatmaps
7. **Evaluation** — cross-validation, ablation studies, calibration metrics

## Module Index

| Module | Description |
|--------|-------------|
| [`src.data_loader`](api/data_loader.md) | Dataset loaders for Mesbah CAP, BOSCH OES |
| [`src.preprocessing`](api/preprocessing.md) | Spectral preprocessing pipeline |
| [`src.features`](api/features.md) | Peak detection and plasma descriptors |
| [`src.spatial`](api/spatial.md) | Wafer spatial uniformity analysis |
| [`src.temporal`](api/temporal.md) | Temporal embedding and phase clustering |
| [`src.evaluation`](api/evaluation.md) | Model evaluation and comparison |
| [`src.guardrail`](api/guardrail.md) | Input validation and safety checks |
| [`src.models.traditional`](api/models_traditional.md) | SVM, RF, ensemble models |
| [`src.models.deep_learning`](api/models_deep_learning.md) | Conv1D, LSTM, Transformer |
| [`src.models.calibration`](api/models_calibration.md) | Probability calibration |

## Quick Start

```bash
pip install -e .
python main.py --task regress --train data/mesbah_cap/dat_train.csv --target T_rot --model ann
python scripts/plot_spatial.py --data data/bosch_oes/ --metric oxide_etch
```
