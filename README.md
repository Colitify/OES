# Plasma OES Spectral Analysis Toolkit

A Python toolkit for **machine learning on plasma optical emission spectroscopy (OES)** data —
species classification, temperature regression, and temporal evolution analysis.

**Primary dataset**: LIBS Benchmark (Figshare, 12 classes, 40 002 channels, 200–1000 nm)
**Secondary datasets**: Mesbah Lab CAP (N₂ OES, T_rot / T_vib), BOSCH Plasma Etching (25 Hz time-series)

---

## Prerequisites

- **Python** 3.9 or later (3.11 recommended)
- **conda** or **pip** for package management
- ~2 GB free disk space (datasets)

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

**LIBS Benchmark** (Figshare, CC-BY-4.0):

```bash
# Download train.h5, test.h5, test_labels.csv
# from https://springernature.figshare.com/collections/Benchmark_classification_dataset_for_laser-induced_breakdown_spectroscopy/4768790
# Place in: data/libs_benchmark/
mkdir -p data/libs_benchmark
# (manual download or use provided script):
python data/libs_benchmark/download.py
```

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

### Task 1 · Species classification (LIBS Benchmark)

```bash
# SVM with 5-fold CV (fast baseline)
python main.py \
  --task classify \
  --train data/libs_benchmark/train.h5 \
  --model svm --cv 5 --seed 42 \
  --metrics_out results/metrics.json

# CNN with Optuna HPO (best performance)
python main.py \
  --task classify \
  --train data/libs_benchmark/train.h5 \
  --model cnn --cv 5 --seed 42 \
  --metrics_out results/metrics.json
```

Expected output: `results/metrics.json` with `primary_metric.name = "micro_f1"`.

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

# Species classification + SHAP
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

All 37 unit tests cover preprocessing, feature extraction, classification evaluation, and temporal analysis.

---

## Project Structure

```
libs-spectral-analysis/
├── data/
│   ├── libs_benchmark/          # LIBS Benchmark HDF5 files
│   ├── mesbah_cap/              # Mesbah Lab CAP CSV files
│   └── bosch_oes/               # BOSCH NetCDF files
├── src/
│   ├── data_loader.py           # load_libs_benchmark, load_mesbah_cap, load_bosch_oes
│   ├── preprocessing.py         # Preprocessor (ALS, SavGol, SNV, cosmic-ray removal)
│   ├── features.py              # detect_peaks, PlasmaDescriptorExtractor, NIST windows
│   ├── temporal.py              # PCA embedding, DTW clustering, LSTM predictor
│   ├── evaluation.py            # evaluate_classifier, compute_ece, compute_snr_gain
│   ├── guardrail.py             # Regression guardrail for CI
│   └── models/
│       ├── traditional.py       # SVM, Ridge, PLS, RF
│       └── deep_learning.py     # Conv1DClassifier, Conv1DRegressor, train_classifier
├── notebooks/
│   ├── 01_preprocessing.ipynb   # Preprocessing tutorial (LIBS Benchmark)
│   ├── 02_classification.ipynb  # SVM+CNN + SHAP tutorial
│   └── 03_temporal_analysis.ipynb # PCA trajectory + DTW tutorial
├── scripts/
│   ├── evaluate_cap.py          # Standalone CAP T_rot/T_vib benchmark
│   ├── plot_temporal_pca.py     # BOSCH PCA trajectory plot
│   ├── plot_clusters.py         # BOSCH DTW cluster plot
│   ├── train_temporal.py        # LSTM training on BOSCH data
│   └── create_notebooks.py      # Notebook generator (nbformat)
├── tests/
│   ├── test_preprocessing.py    # Preprocessor unit tests
│   ├── test_features.py         # Feature extraction unit tests
│   ├── test_classifier.py       # Classifier evaluation unit tests
│   └── test_temporal.py         # Temporal analysis unit tests
├── results/
│   ├── metrics.json             # Classification metrics (micro_f1)
│   └── metrics_cap.json         # CAP regression metrics (RMSE)
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
  --train PATH         Training data path (*.h5 for classify, *.csv for regress, dir/ for temporal)
  --target STR         Regression target column name (T_rot, T_vib, ...)

Model arguments:
  --model {svm,rf,cnn,ridge,pls,ann,ann_hybrid,xgb,lstm,dtw}
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
| LIBS classification (SVM) | micro_f1 | ≥ 0.90 | PASS |
| LIBS classification (CNN) | micro_f1 | ≥ 0.90 | PASS |
| CAP T_rot regression | RMSE | ≤ 50 K | PASS |
| CAP T_vib regression | RMSE | ≤ 200 K | PASS |
| BOSCH LSTM forecasting | val MSE decrease | > 0 | PASS |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥ 1.24 | Array operations |
| scipy | ≥ 1.11 | Signal processing, interpolation |
| scikit-learn | ≥ 1.3 | ML models, preprocessing |
| torch | ≥ 2.0 | CNN, LSTM deep learning |
| h5py | ≥ 3.0 | LIBS HDF5 loading |
| netCDF4 | ≥ 1.6 | BOSCH NetCDF loading |
| tslearn | ≥ 0.6 | DTW K-means clustering |
| shap | ≥ 0.40 | SHAP wavelength importance |
| optuna | ≥ 3.0 | Hyperparameter optimisation |
| matplotlib | ≥ 3.7 | Plotting |
| nbformat | ≥ 5.9 | Notebook generation |

---

## License

MIT License
