# Model Card: Conv1D Classifier

## Model Details

| Field | Value |
|-------|-------|
| Model | `Conv1DClassifier` |
| Module | `src.models.deep_learning` |
| Task | 12-class LIBS material classification |
| Framework | PyTorch |

## Architecture

- **Input**: Raw spectrum (1, 40002) — single-channel 1D signal
- **Backbone**: 3 × (Conv1d → BatchNorm → ReLU), stride=4, kernel_size=7, 32 filters
- **Pooling**: AdaptiveAvgPool1d(1) (global average pooling)
- **Head**: Linear(32, 128) → ReLU → Dropout(0.3) → Linear(128, 12)
- **Output**: 12-class logits (softmax applied at inference)
- **Parameters**: ~140K trainable parameters

## Intended Use

Classification of laser-induced breakdown spectroscopy (LIBS) spectra into
12 ore/material categories. Designed for the LIBS Benchmark contest dataset.

## Training Data

- **Dataset**: LIBS Benchmark (Figshare), 12 classes, 40002 wavelength channels
- **Training split**: ~48,000 spectra (balanced, ~4000/class)
- **Preprocessing**: None (raw intensity), or PCA(50) feature reduction
- **Optimizer**: Adam, lr=1e-3, batch_size=64
- **Epochs**: Up to 50 with early stopping on validation loss

## Metrics

| Metric | CV (3-fold) | Locked Test |
|--------|-------------|-------------|
| Micro F1 | 0.957 | 0.686 |
| Macro F1 | 0.957 | 0.590 |

## Limitations

- **Distribution shift**: Significant gap between CV and locked-test performance
  due to class imbalance in the test set (514–3195 spectra/class vs balanced training).
- **No uncertainty quantification**: MC-dropout available but not enabled by default.
- Trained on a single LIBS instrument; may not generalise to different spectrometers
  without transfer learning or domain adaptation.
