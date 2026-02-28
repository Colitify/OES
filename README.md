# Plasma OES Spectral Analysis Toolkit

A Python toolkit for **machine learning on plasma optical emission spectroscopy (OES)** data —
substrate classification, temperature regression, and temporal evolution analysis.

> *"Apply ML techniques to analyse optical emission spectra from high voltage electrical discharge systems."*

**Primary dataset**: BOSCH Plasma Etching (Zenodo, 3648 channels, 25 Hz time-series)
**Secondary dataset**: Mesbah Lab CAP (N₂ OES, 51 channels, T_rot / T_vib)

---

## Prerequisites

- **Python** 3.10 or later (3.11 recommended)
- **conda** or **pip** for package management

---

## Installation

### 1 · Create and activate environment

```bash
# Using conda (recommended):
conda create -n oes_env python=3.11 -y
conda activate oes_env

# Or using venv:
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 2 · Install dependencies

```bash
pip install -r requirements.txt
```

### 3 · Download datasets

**Mesbah Lab CAP** (GitHub):

```bash
git clone https://github.com/Mesbah-Lab-UCB/Machine-Learning-for-Plasma-Diagnostics data/mesbah_cap_repo
cp data/mesbah_cap_repo/dat_*.csv data/mesbah_cap/
```

**BOSCH Plasma Etching** (Zenodo):

```bash
# Download from https://zenodo.org/records/17122442
# Place NetCDF files (Day_*.nc, Process_data.nc) in: data/bosch_oes/
mkdir -p data/bosch_oes
```

### 4 · Register Jupyter kernel (for tutorial notebooks)

```bash
python -m ipykernel install --user --name oes_env --display-name "Python (oes_env)"
```

---

## Quick Start

### Task 1 · Substrate classification (Mesbah CAP)

```bash
# SVM with 5-fold CV
python main.py \
  --task classify \
  --train data/mesbah_cap/dat_train.csv \
  --model svm --cv 5 --seed 42 \
  --metrics_out results/metrics.json
```

### Task 2 · Temperature regression (Mesbah Lab CAP)

```bash
python main.py \
  --task regress \
  --train data/mesbah_cap/dat_train.csv \
  --target T_rot \
  --model ann --per_target --cv 5 --seed 42 \
  --metrics_out results/metrics_cap.json

# Standalone benchmark (both T_rot and T_vib):
python scripts/evaluate_cap.py
```

Expected: T_rot RMSE ≤ 50 K, T_vib RMSE ≤ 200 K.

### Task 3 · Temporal OES analysis (BOSCH etching)

```bash
# PCA trajectory plot
python scripts/plot_temporal_pca.py --data data/bosch_oes/

# Discharge phase clustering (DTW K-means, k=4)
python scripts/plot_clusters.py --data data/bosch_oes/ --k 4

# LSTM next-step predictor
python scripts/train_temporal.py --data data/bosch_oes/ --epochs 50
```

---

## Tutorial Notebooks

Run the interactive tutorial notebooks (require dataset downloads):

```bash
# Preprocessing pipeline
jupyter nbconvert --to notebook --execute notebooks/01_preprocessing.ipynb --output-dir notebooks/

# Substrate classification
jupyter nbconvert --to notebook --execute notebooks/02_classification.ipynb --output-dir notebooks/

# Temporal analysis + DTW clustering
jupyter nbconvert --to notebook --execute notebooks/03_temporal_analysis.ipynb --output-dir notebooks/
```

Or launch interactively:
```bash
jupyter notebook notebooks/
```

---

## Run Tests

```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
libs-spectral-analysis/
├── data/
│   ├── mesbah_cap/              # Mesbah Lab CAP CSV files
│   └── bosch_oes/               # BOSCH NetCDF files
├── src/
│   ├── data_loader.py           # load_mesbah_cap, load_bosch_oes, load_wafer_spatial
│   ├── preprocessing.py         # Preprocessor (ALS, SavGol, SNV, cosmic-ray removal)
│   ├── features.py              # detect_peaks, PlasmaDescriptorExtractor, plasma emission lines
│   ├── temporal.py              # PCA embedding, DTW clustering, LSTM predictor
│   ├── spatial.py               # Wafer spatial uniformity, RBF interpolation
│   ├── evaluation.py            # evaluate_classifier, compute_ece, compute_snr_gain
│   ├── guardrail.py             # Regression guardrail for CI
│   ├── optimization.py          # Optuna HPO for PLS, Ridge, RF, etc.
│   └── models/
│       ├── traditional.py       # SVM, Ridge, PLS, RF
│       └── deep_learning.py     # Conv1DRegressor, LSTM, Transformer
├── notebooks/
│   ├── 01_preprocessing.ipynb   # Preprocessing tutorial
│   ├── 02_classification.ipynb  # Substrate classification demo
│   └── 03_temporal_analysis.ipynb # PCA trajectory + DTW tutorial
├── scripts/
│   ├── evaluate_cap.py          # Standalone CAP T_rot/T_vib benchmark
│   ├── plot_temporal_pca.py     # BOSCH PCA trajectory plot
│   ├── plot_clusters.py         # BOSCH DTW cluster plot
│   ├── train_temporal.py        # LSTM training on BOSCH data
│   ├── ablation.py              # Feature ablation study (OES)
│   ├── plot_shap.py             # SHAP attribution overlay
│   └── generate_report.py       # Final report generator
├── tests/
│   ├── test_preprocessing.py    # Preprocessor unit tests
│   ├── test_features.py         # Feature extraction unit tests
│   ├── test_classifier.py       # Classifier evaluation unit tests
│   └── test_temporal.py         # Temporal analysis unit tests
├── main.py                      # Unified CLI entry point
└── requirements.txt
```

---

## Preprocessing Pipeline

```
Raw Spectrum
    → Cosmic-ray removal  (Z-score median filter, threshold=5σ)
    → ALS baseline correction  (λ=1×10⁶, p=0.01)
    → Savitzky-Golay smoothing  (window=11, poly=3)
    → SNV normalization  (zero-mean, unit-variance per spectrum)
```

---

## CLI Reference

```
python main.py --help

usage: main.py [-h] --task {classify,regress,temporal} ...

Data arguments:
  --train PATH         Training data path (*.csv for classify/regress, dir/ for temporal)
  --target STR         Regression target column name (T_rot, T_vib, ...)

Model arguments:
  --model {svm,rf,ridge,pls,ann,xgb,lstm,dtw}
  --cv INT             Cross-validation folds (default: 5)
  --seed INT           Random seed (default: 42)

Temporal arguments:
  --n_clusters INT     DTW K-means clusters (default: 4)
  --seq_len INT        LSTM sequence length (default: 10)
  --n_temporal_components INT  PCA components for temporal embedding (default: 20)

Output arguments:
  --metrics_out PATH   Metrics JSON output path
```

---

## Performance Targets

| Task | Metric | Target | Status |
|------|--------|--------|--------|
| CAP T_rot regression | RMSE | ≤ 50 K | PASS |
| CAP T_vib regression | RMSE | ≤ 200 K | PASS |
| BOSCH LSTM forecasting | val MSE decrease | > 0 | PASS |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥ 1.21 | Array operations |
| scipy | ≥ 1.7 | Signal processing, interpolation |
| scikit-learn | ≥ 1.0 | ML models, preprocessing |
| torch | ≥ 1.10 | CNN, LSTM deep learning |
| netCDF4 | ≥ 1.6 | BOSCH NetCDF loading |
| tslearn | ≥ 0.6 | DTW K-means clustering |
| shap | ≥ 0.40 | SHAP wavelength importance |
| optuna | ≥ 3.0 | Hyperparameter optimisation |
| matplotlib | ≥ 3.5 | Plotting |
| nbformat | ≥ 5.9 | Notebook generation |

---

## License

MIT License
