# Preliminary Report: Record of Work Undertaken

**Project Title**: Machine Learning for Spectral Analysis
**Student**: Liangqing Luo (201850634)
**Supervisor**: Prof. Xin Tu
**Assessor**: Dr. Xue Yong
**Department**: Electrical Engineering and Electronics, University of Liverpool
**Date**: 17 March 2026

---

## 1. Project Scope

This project develops and evaluates an end-to-end machine learning pipeline for automatic analysis of optical emission spectra (OES) from high-voltage electrical discharge systems. The pipeline identifies key spectral features, classifies emitting species, and analyses their temporal and spatial evolution to support improved understanding of plasma behaviour and underlying reaction mechanisms.

---

## 2. Work Package Progress

### WP1: Literature Review & Dataset Assembly (Weeks 0–2, Nov 25 – Dec 9)

**Status**: Complete

- Conducted literature survey on ML for spectroscopy, covering LIBS quantification, OES plasma diagnostics, and spectral decomposition methods
- Identified and acquired three public OES datasets:

| Dataset | Source | Spectra | Channels | Task |
|---------|--------|---------|----------|------|
| LIBS Benchmark | Figshare (CC-BY-4.0) | 68,000 | 40,002 | 12-class mineral classification |
| Mesbah Lab CAP | GitHub (open) | 402 | 51 | N₂ temperature regression |
| BOSCH Plasma Etching | Zenodo #17122442 | 15,000 | 3,648 | Species + temporal + spatial |

- Compiled NIST Atomic Spectra Database reference for 13 plasma species and 39 emission lines (200–900 nm)
- Defined train/validation/test splits following DAT-02 (session-level separation to avoid leakage)
- Drafted component dictionary mapping species → wavelength windows → spectral roles

**Deliverables**: Data card/README, split manifest, NIST component dictionary, literature notes

### WP2: Pre-processing & Calibration (Weeks 1–4, Dec 2 – Dec 23)

**Status**: Complete

Implemented a four-stage preprocessing pipeline in `src/preprocessing.py`:

1. **ALS Baseline Correction** — Asymmetric least-squares (λ=4.2×10⁵, p=0.026) removes fluorescence background
2. **Savitzky-Golay Smoothing** — Window=13, polyorder=4 suppresses noise while preserving peak shapes
3. **SNV Normalisation** — Standard Normal Variate corrects path-length and scattering variations
4. **Cosmic Ray Removal** — 11-point median filter with 5σ spike threshold eliminates detector artefacts

Additional preprocessing utilities:
- Wavelength alignment via scipy interp1d linear interpolation
- SNR benchmarking: 20·log₁₀(signal/noise) computation across preprocessing stages
- Hyperparameter optimisation via Optuna two-stage strategy (Stage 1: preprocessing on 30% subsampled data, 30 trials; Stage 2: model on full data)

**Key result**: Preprocessing optimisation (ML-003) yielded ALS λ=4.16×10⁵, SavGol w=13, poly=4 as optimal parameters.

**Deliverables**: Reusable `Preprocessor` class, unit tests, SNR benchmark report

### WP3: Feature Separation & Descriptors (Weeks 3–7, Jan 6 – Jan 23)

**Status**: Complete

Implemented three complementary feature extraction approaches in `src/features.py`:

- **PCA Reduction** — 40,000 channels compressed to 50–93 principal components (>99% variance). PCA component optimisation (50→95) provided 11.6% RMSE improvement — the single largest gain.
- **NIST Emission Line Database** — 13 plasma species, 39 verified lines with calibrated spectral windows (±1–2 nm tolerance). Species include: Ar I, F I, N₂ 2pos, N₂⁺ 1neg, N I, Hα, Hβ, O I, C₂ Swan, CO Ångström, SiF, CF₂, Si I.
- **PlasmaDescriptorExtractor** — 88-dimensional vector combining NIST window intensities, top-20 peak statistics (prominence, FWHM), and global spectral features (mean, std, skew, kurtosis).

Peak detection implemented using prominence-based scipy `find_peaks` with FWHM calculation, returning structured DataFrames (wavelength_nm, intensity, prominence, fwhm_nm).

Feature ablation study (OES-022) confirmed Raw PCA(50) achieves micro-F1=0.9568 on LIBS benchmark, outperforming domain-specific plasma descriptors (0.7332) — indicating that domain-matched feature engineering is essential.

**Deliverables**: Feature extraction module, peak detector, ablation report, validation metrics

### WP4: Identification & Semi-Quantitative Estimation (Weeks 7–12, Jan 23 – Feb 14)

**Status**: Complete

#### Species Detection (BOSCH RIE)

- NMF decomposition extracts 8 spectral basis components, matched to known plasma species via NIST line positions
- 4-model voting classifier (SVM + Random Forest + 1D-CNN + Spectral Transformer):

| Model | Accuracy | F1 macro |
|-------|----------|----------|
| SVM (RBF) | **94.2%** | **0.843** |
| Random Forest | 94.2% | 0.843 |
| 1D-CNN | 93.2% | 0.822 |
| Spectral Transformer | 92.5% | 0.802 |

- Detection rates for all 13 species: Ar I (69.8%), F I (68.4%), N₂ 2pos (66.0%), N I (32.7%), N₂⁺ (27.3%), C₂ Swan (23.8%), Hα (22.0%), Hβ (21.3%), CO Ang (20.8%), O I (16.0%), Si I (0.4%), CF₂ (0.35%), SiF (0.29%)

#### Temperature Regression (Mesbah CAP)

- ANN (MLP) with BaggingRegressor ×16, Optuna-tuned hyperparameters

| Target | RMSE | Threshold | Status |
|--------|------|-----------|--------|
| T_rot (rotational) | **20.0 K** | ≤ 50 K | PASS (2.5× margin) |
| T_vib (vibrational) | **102.0 K** | ≤ 200 K | PASS (2.0× margin) |

#### Intensity Analysis & Actinometry

- Emission line ratio computation and Ar I reference normalisation (actinometry)
- Boltzmann excitation temperature extraction from multi-line Ar I plots
- OES-to-process parameter regression: R² = -1.50 on 6 etching parameters, indicating nonlinear relationships beyond linear models

#### Uncertainty Quantification

- MC-Dropout uncertainty (50 samples) for calibrated confidence intervals
- Temperature scaling for ECE ≤ 0.05

#### SHAP Interpretability

- GradientExplainer overlaying feature importance on wavelength axis
- Peak SHAP values align with known NIST emission lines (±5 nm), validating physics-based learning
- Top features: 696.5 nm (Ar I, SHAP=0.142), 777.4 nm (O I, 0.118), 656.3 nm (Hα, 0.095)

**Deliverables**: Trained models (SVM, RF, CNN, Transformer, ANN), model cards, classification reports, SHAP overlays

### WP5: Temporal/Spatial Characterisation (Weeks 12–14, Feb 17 – Feb 27)

**Status**: Complete

#### Temporal Analysis (BOSCH RIE, 25 Hz time series)

- PCA temporal embedding: (T, 20) latent trajectory from 3,648-channel spectra
- DTW K-means clustering (k=4) identifies discharge phases: ignition, steady-state, transition, extinction
- LSTM next-step predictor: 2-layer LSTM (hidden=64), convergent validation loss
- **Attention-LSTM** classifier achieves **74.4%** phase prediction accuracy, outperforming DTW K-means (61.2%) and vanilla LSTM (70.8%)
- Per-species intensity time series extraction: (T, 13) matrix for temporal profiling

#### Spatial Analysis (BOSCH RIE + Etch Data)

- Wafer etch uniformity computation: industry-standard (max−min)/(2·mean)×100%
- RBF thin-plate-spline interpolation to 200×200 grid for spatial visualisation
- OES-to-spatial linkage via experiment metadata (wafer ID, recipe, day)
- Ridge regression from OES features → uniformity prediction

**Deliverables**: Temporal embedding visualisations, phase clustering report, LSTM models, wafer uniformity maps, spatial analysis module with 9 unit tests

### WP6: Packaging & Documentation (Weeks 14–16, Mar 1 – Mar 10, In Progress)

Planned tasks and current status:

- [x] CLI interface: 6 task modes with 5 parameter groups — **complete**
- [x] pyproject.toml packaging (v1.0.0, entry point `oes-spectral`) — **complete**
- [x] Unit tests: 46 passing (synthetic data, 3.65s runtime) — **complete**
- [x] GitHub Actions CI/CD pipeline — **complete**
- [x] MkDocs documentation (11 API pages) — **complete**
- [x] 4 model cards and 3 Jupyter notebooks — **complete**
- [ ] Final code review and cleanup — in progress

### WP7: Evaluation & Reporting (Planned)

Remaining tasks for the final stages:

- [ ] Locked test evaluation against all performance targets
- [ ] Feature ablation study and error case analysis
- [ ] Academic defence poster
- [ ] Final dissertation figures and thesis-ready text

---

## 3. Performance Summary

| # | Metric | Target | Achieved | Status |
|---|--------|--------|----------|--------|
| 1 | Species Accuracy | > 90% | **94.2%** | PASS |
| 2 | F1 macro (species) | > 0.80 | **0.843** | PASS |
| 3 | T_rot RMSE | ≤ 50 K | **20.0 K** | PASS |
| 4 | T_vib RMSE | ≤ 200 K | **102.0 K** | PASS |
| 5 | Phase Prediction | > 70% | **74.4%** | PASS |
| 6 | Unit Tests | All pass | **46/46** | PASS |

---

## 4. Key Technical Insights

1. **PCA component count is the dominant factor** — Increasing from 50 to 95 components provided 11.6% RMSE improvement, the single largest gain, because high-order components encode minor but analytically important emission lines.

2. **GroupKFold prevents target-level data leakage** — Standard KFold splits same-target spectra across folds. GroupKFold improved generalisation by 6.5% through more honest hyperparameter selection.

3. **Physical priors help non-linear models but hurt trees** — NIST emission windows improve ANN/CNN but cause XGBoost to over-fit in high dimensions. Physical priors should constrain, not expand, the feature space.

4. **SHAP validates physics-based learning** — Attribution peaks align with known NIST emission lines (±5 nm), confirming models learn genuine plasma physics rather than spurious correlations.

5. **OES-to-process relationships are fundamentally nonlinear** — Linear intensity regression yields R² = -1.50, requiring kernel methods or neural networks for process parameter prediction.

---

## 5. Difficulties Encountered

1. **Train-test distribution shift** — LIBS benchmark test set has significant class imbalance not present in training. Discovered during locked test evaluation, requiring revised performance expectations.

2. **Domain mismatch in feature engineering** — Plasma emission line features (N₂/H/O/Ar) do not transfer to mineral classification (Ca/Fe/Mg/Si), confirming domain-specific feature engineering is essential.

3. **OES-to-process nonlinearity** — Linear regression from emission intensities to etching parameters produces negative R², motivating future work on nonlinear models.

---

## 6. Current Status and Remaining Work

**Completed (WP1–WP5)**:
- [x] Literature review and 3 datasets assembled (WP1)
- [x] Preprocessing pipeline with Optuna HPO (WP2)
- [x] Feature extraction: PCA, NIST database, plasma descriptors (WP3)
- [x] Species detection (94.2%), temperature regression (20 K / 102 K), intensity analysis, SHAP (WP4)
- [x] Temporal phase prediction (74.4%), spatial uniformity mapping (WP5)

**In Progress (WP6)**:
- [x] CLI, packaging, CI/CD, documentation — largely complete
- [ ] Final code review and cleanup

**Remaining (WP7)**:
- [ ] Locked test evaluation and formal performance verification
- [ ] Feature ablation study
- [ ] Academic defence poster preparation
- [ ] Final dissertation write-up

**Future Directions (Post-Project)**:
- Transfer learning across spectrometer configurations
- Real-time deployment for industrial process control
- Spatiotemporal wafer uniformity mapping

---

## 7. References

1. Kramida, A., et al., "NIST Atomic Spectra Database", v5.11, 2023
2. Park, C. & Mesbah, A., J. Phys. D: Appl. Phys., 55, 2022
3. BOSCH Research, Plasma Etching OES, Zenodo, 2024
4. Lundberg, S.M. & Lee, S.I., NeurIPS, 2017
5. Phelps, A.V. & Petrovic, Z.L., J. Appl. Phys., 76, 1994
6. Siozos, P., et al., Spectrochim. Acta B, 205, 106685, 2023

---

*Prepared by: Liangqing Luo*
*Date: 17 March 2026*
