# OES — Machine Learning for Optical Emission Spectroscopy in Plasma Diagnostics

A Python toolkit for **automated plasma OES analysis** — species identification, process state classification, spatiotemporal evolution tracking, and semi-quantitative intensity analysis.

> *"Apply ML techniques to analyse optical emission spectra from high voltage electrical discharge systems."*

---

## Key Results

| Task | Method | Result |
|------|--------|--------|
| Plasma state classification | RF / SVM (balanced) | **94.2% accuracy**, F1=0.843 |
| Species identification | NMF + NIST matching | Ar I 69.8%, F I 68.4%, C2 23.8% detected |
| Temperature regression | ANN ensemble (CAP) | T_rot RMSE = **20.0 K**, T_vib = **102.0 K** |
| Temporal phase prediction | Attention-LSTM | **74.4% accuracy** |
| Excitation temperature | Boltzmann plot (Ar I) | T_exc = **13,334 K** median |
| Feature importance | SHAP (TreeExplainer) | F_I = 0.131 (top feature, physically correct) |

**6 models compared on real data:** SVM 94.2% > RF 94.2% > CNN 93.2% > Transformer 92.5% > LSTM 74.4%

---

## Prerequisites

- **Python** 3.10+ (3.11 recommended)
- **conda** or **pip**

---

## Installation

```bash
# 1. Clone
git clone https://github.com/Colitify/OES.git
cd OES

# 2. Create environment
conda create -n oes python=3.11 -y
conda activate oes

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -m pytest tests/ -v
# Expected: 74 passed
```

---

## Download Datasets

### Mesbah Lab CAP (0.7 MB)

N2 atmospheric plasma jet OES — 51 spectral channels, T_rot/T_vib labels.

```bash
mkdir -p data/mesbah_cap
git clone https://github.com/Mesbah-Lab-UCB/Machine-Learning-for-Plasma-Diagnostics /tmp/mesbah
cp /tmp/mesbah/dat_*.csv data/mesbah_cap/
```

### BOSCH Plasma Etching (~8.2 GB)

Reactive ion etching OES — 3,648 channels (185–884 nm), 25 Hz, 10 days x 10 wafers.

```bash
mkdir -p data/bosch_oes
# Download from: https://zenodo.org/records/17122442
# Place all .nc files and .csv files into data/bosch_oes/
```

Required files:
```
data/bosch_oes/
├── Day_2024_07_02.nc ... Day_2024_08_22.nc   (10 day files)
├── Process_data.nc                            (process parameters)
├── Dictionary_OES.nc                          (wavelength dictionary)
├── Dictionary_process.nc                      (parameter dictionary)
├── Si_Oxide_etch_89_points.csv                (spatial etch data)
└── Si_Oxide_etch_9_points.csv
```

---

## Quick Start — 6 Task Modes

### 1. Substrate Classification (Mesbah CAP)

```bash
python main.py \
  --task classify \
  --train data/mesbah_cap/dat_train.csv \
  --model svm --cv 5 \
  --metrics_out results/metrics_classify.json
```

Expected: accuracy > 0.90

### 2. Temperature Regression (Mesbah CAP)

```bash
python main.py \
  --task regress \
  --train data/mesbah_cap/dat_train.csv \
  --target T_rot --model ann --cv 5 \
  --metrics_out results/metrics_cap.json
```

Expected: T_rot RMSE ≤ 50 K, T_vib RMSE ≤ 200 K

### 3. Temporal Analysis (BOSCH)

```bash
python main.py \
  --task temporal \
  --train data/bosch_oes \
  --model dtw --n_clusters 4 \
  --metrics_out results/metrics_temporal.json
```

### 4. Species Classification (BOSCH) — NEW

Automated species detection + NMF decomposition + multi-model classification + SHAP interpretability.

```bash
python main.py \
  --task species \
  --train data/bosch_oes \
  --model rf --cv 5 \
  --max_wafers 3 --max_timesteps 5000 \
  --n_nmf_components 8 \
  --metrics_out results/metrics_species.json
```

Available models: `svm`, `rf`, `cnn`, `transformer`

Expected: accuracy > 0.90 (plasma ON/OFF classification)

### 5. Intensity Analysis (BOSCH) — NEW

Actinometry + Boltzmann temperature + OES-to-process regression.

```bash
python main.py \
  --task intensity \
  --train data/bosch_oes \
  --model ridge --cv 5 \
  --max_wafers 3 --max_timesteps 5000 \
  --metrics_out results/metrics_intensity.json
```

Available models: `ridge`, `pls`, `rf`, `ann`

### 6. Spatiotemporal Evolution (BOSCH) — NEW

Species time series extraction + Attention-LSTM phase transition prediction.

```bash
python main.py \
  --task spatiotemporal \
  --train data/bosch_oes \
  --cv 5 --seq_len 20 \
  --max_timesteps 5000 \
  --metrics_out results/metrics_spatiotemporal.json
```

---

## Full Reproducibility Check

Run all tasks in sequence to verify the complete pipeline:

```bash
# 1. Unit tests (no data required)
python -m pytest tests/ -v
# Expected: 74 passed

# 2. CAP temperature regression
python main.py --task regress \
  --train data/mesbah_cap/dat_train.csv \
  --target T_rot --model ann --cv 5 \
  --metrics_out results/check_cap.json
# Expected: T_rot RMSE ≤ 50 K

# 3. Substrate classification
python main.py --task classify \
  --train data/mesbah_cap/dat_train.csv \
  --model svm --cv 5 \
  --metrics_out results/check_classify.json
# Expected: accuracy > 0.90

# 4. Species classification (requires BOSCH data)
python main.py --task species \
  --train data/bosch_oes \
  --model rf --cv 5 \
  --max_wafers 3 --max_timesteps 5000 \
  --metrics_out results/check_species.json
# Expected: accuracy > 0.90

# 5. Intensity analysis
python main.py --task intensity \
  --train data/bosch_oes \
  --model ridge --cv 5 \
  --max_wafers 3 --max_timesteps 5000 \
  --metrics_out results/check_intensity.json

# 6. Spatiotemporal
python main.py --task spatiotemporal \
  --train data/bosch_oes \
  --max_timesteps 5000 \
  --metrics_out results/check_spatiotemporal.json
# Expected: Attention-LSTM accuracy > 0.60
```

All metrics are saved as JSON for automated verification.

---

## Project Structure

```
OES/
├── main.py                         # CLI entry point (6 task modes)
├── requirements.txt
├── LOGBOOK_DATED.md                # Complete laboratory logbook
│
├── src/
│   ├── data_loader.py              # 3 dataset loaders + multi-wafer batch + phase labeling
│   ├── preprocessing.py            # ALS baseline, SavGol, SNV, cosmic-ray removal
│   ├── features.py                 # PCA, NIST emission lines (13 species, 39 lines),
│   │                               #   PlasmaDescriptorExtractor, peak detection
│   ├── species.py                  # NMF decomposition, NIST matching, species detection,
│   │                               #   classifiers (SVM/RF/CNN/Transformer), SHAP
│   ├── intensity.py                # Line ratios, actinometry, Boltzmann T_e,
│   │                               #   OES-to-process regression
│   ├── temporal.py                 # PCA embedding, DTW clustering, LSTM predictor,
│   │                               #   species time series, Attention-LSTM
│   ├── spatial.py                  # Wafer uniformity, RBF interpolation,
│   │                               #   OES-to-etch prediction
│   ├── evaluation.py               # RMSE/R2/F1, GroupKFold, SHAP, per-target aggregation
│   ├── optimization.py             # Optuna two-stage hyperparameter search
│   ├── guardrail.py                # Automated regression testing
│   └── models/
│       ├── traditional.py          # Ridge, PLS, RF, XGBoost
│       ├── deep_learning.py        # Conv1DClassifier, Conv1DRegressor
│       ├── attention.py            # AttentionLSTM, SEConv1D, SpectralTransformer
│       └── calibration.py          # Temperature scaling (ECE)
│
├── tests/                          # 74 automated tests
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_classifier.py
│   ├── test_temporal.py
│   ├── test_spatial.py
│   ├── test_species.py
│   ├── test_intensity.py
│   └── test_attention.py
│
├── notebooks/                      # Interactive tutorials
│   ├── 01_preprocessing.ipynb
│   ├── 02_classification.ipynb
│   └── 03_temporal_analysis.ipynb
│
├── scripts/
│   ├── evaluate_cap.py
│   ├── plot_temporal_pca.py
│   ├── plot_clusters.py
│   └── make_poster.py              # A1 academic poster generator
│
├── docs/
│   └── superpowers/plans/
│       └── 2026-03-15-oes-species-analysis.md  # Phase 3 implementation plan (23 references)
│
└── data/                           # (not tracked — download separately)
    ├── mesbah_cap/
    └── bosch_oes/
```

---

## Emission Line Database

13 plasma species with 39 emission lines (NIST ASD verified):

| Species | Lines (nm) | Window | Origin |
|---------|-----------|--------|--------|
| N2 2nd positive | 315.9, 337.1, 357.7, 380.5 | ±2.0 | Atmospheric N2 plasma |
| N2+ 1st negative | 391.4, 427.8 | ±1.5 | High-energy electron marker |
| N I | 746.8, 818.5, 862.9 | ±1.0 | Atomic nitrogen |
| H alpha | 656.3 | ±2.0 | Hydrogen Balmer |
| H beta | 486.1 | ±2.0 | Hydrogen Balmer |
| O I | 777.4, 844.6, 926.6 | ±1.0 | Oxygen / air entrainment |
| Ar I | 696.5, 706.7, 738.4, 750.4, 763.5, 772.4 | ±1.0 | Noble gas carrier |
| **F I** | **685.6, 690.2, 703.7, 712.8, 739.9** | **±1.0** | **SF6 etchant radical** |
| **Si I** | **243.5, 250.7, 251.6, 252.4, 288.2** | **±1.0** | **Silicon etch product** |
| **CF2** | **251.9, 259.5** | **±1.5** | **C4F8 passivation** |
| **C2 Swan** | **473.7, 516.5, 563.6** | **±2.0** | **Carbon indicator** |
| **SiF** | **440.0, 442.5** | **±2.0** | **Si+F recombination** |
| **CO Angstrom** | **451.1, 519.8, 561.0** | **±1.5** | **C4F8+O2 product** |

Bold = BOSCH RIE species added in Phase 3.

---

## Model Comparison (BOSCH real data, 15,000 spectra)

| Model | Type | Accuracy | F1 macro | Notes |
|-------|------|----------|----------|-------|
| SVM (RBF, balanced) | Traditional | **94.2%** | **0.843** | Best overall |
| Random Forest (200) | Traditional | **94.2%** | **0.843** | + SHAP interpretability |
| RF (13 species features) | Traditional | 93.6% | 0.821 | Only 13 handcrafted features |
| CNN (3-layer Conv1D) | Deep Learning | 93.2% | 0.822 | Mini-batch, weighted CE loss |
| SpectralTransformer | Deep Learning | 92.5% | 0.802 | ViT-style patch embedding |
| Attention-LSTM | Deep Learning | 74.4% | — | Temporal sequence classification |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥ 1.21 | Array operations |
| scipy | ≥ 1.7 | Signal processing (ALS, SavGol, RBF) |
| scikit-learn | ≥ 1.0 | ML models, preprocessing, metrics |
| torch | ≥ 1.10 | CNN, LSTM, Attention, Transformer |
| netCDF4 | ≥ 1.6 | BOSCH NetCDF loading |
| tslearn | ≥ 0.6 | DTW K-means clustering |
| shap | ≥ 0.44 | SHAP feature importance |
| optuna | ≥ 3.0 | Hyperparameter optimisation |
| matplotlib | ≥ 3.5 | Plotting |
| pandas | ≥ 1.3 | Data manipulation |

---

## License

MIT License
