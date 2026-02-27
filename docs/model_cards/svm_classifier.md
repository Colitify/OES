# Model Card: SVM Classifier (LinearSVC + PCA)

## Model Details

| Field | Value |
|-------|-------|
| Model | `LinearSVC` (scikit-learn) with `PCA(n_components=50)` |
| Module | `src.models.traditional` |
| Task | 12-class LIBS material classification |
| Framework | scikit-learn |

## Architecture

- **Dimensionality reduction**: PCA with 50 components (40002 → 50)
- **Classifier**: LinearSVC with L2 penalty, one-vs-rest strategy
- **Pipeline**: StandardScaler → PCA(50) → LinearSVC(C=1.0)

## Intended Use

Baseline classification of LIBS spectra into 12 material categories.
Fast to train and evaluate; serves as the primary ablation study baseline.

## Training Data

- **Dataset**: LIBS Benchmark (Figshare), 12 classes, 40002 wavelength channels
- **Training split**: ~48,000 spectra (balanced ~4000/class)
- **Feature variant A** (ablation study best): Raw PCA(50)

## Metrics

Ablation study results (3-fold stratified CV):

| Variant | Features | Micro F1 | Macro F1 | Fit Time |
|---------|----------|----------|----------|----------|
| A — Raw PCA(50) | 50 | **0.957** | 0.957 | 7.2 s |
| B — Plasma descriptor | 88 | 0.733 | 0.731 | 77.2 s |
| C — NIST windows + PCA | 50 | 0.860 | 0.860 | 1.2 s |
| D — Combined (A+B+C) | 188 | 0.951 | 0.951 | 87.1 s |

Locked test set: micro_f1 = 0.686, macro_f1 = 0.590.

## Limitations

- Linear decision boundary may underfit complex class boundaries.
- **Distribution shift**: CV performance (0.957) significantly exceeds locked-test
  performance (0.686) due to class imbalance in the held-out test set.
- PCA is unsupervised and may discard discriminative spectral features.
- Domain-informed features (plasma descriptors) underperform raw PCA, suggesting
  the hand-crafted features do not capture the most discriminative information.
