# Model Card: Conv1D Regressor

## Model Details

| Field | Value |
|-------|-------|
| Model | `Conv1DRegressor` |
| Module | `src.models.deep_learning` |
| Task | Plasma temperature regression (T_rot, T_vib) |
| Framework | PyTorch |

## Architecture

- **Input**: OES spectrum (1, 51) — N2 2nd-positive transition channels
- **Backbone**: 3 × (Conv1d → BatchNorm → ReLU → MaxPool1d), channels [64, 128, 256]
- **Head**: Flatten → Linear(flat_dim, 128) → ReLU → Dropout(0.3) → Linear(128, output_dim)
- **Output**: Continuous temperature predictions (K)

## Intended Use

Regression of rotational temperature (T_rot) and vibrational temperature (T_vib)
from cold atmospheric plasma (CAP) optical emission spectra. Supports both
single-target and multi-target prediction.

## Training Data

- **Dataset**: Mesbah Lab CAP (UC Berkeley), 51 spectral channels, N2 OES
- **Targets**: T_rot (K), T_vib (K), optionally power/flow/substrate_type
- **Preprocessing**: Standard scaling of spectral features
- **Optimizer**: Adam, lr=1e-3
- **Epochs**: Up to 100 with early stopping

## Metrics

| Target | RMSE | Threshold |
|--------|------|-----------|
| T_rot | 20 K | ≤ 50 K |
| T_vib | 102 K | ≤ 200 K |

Both targets meet the project plan acceptance criteria.

## Limitations

- Trained on N2 2nd-positive band (296–422 nm); not suitable for other gas chemistries
  without retraining.
- Small input dimensionality (51 channels) limits spatial/spectral resolution.
- Assumes stable plasma conditions; rapid transient events may degrade accuracy.
