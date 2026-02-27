# Model Card: LSTM Temporal Predictor

## Model Details

| Field | Value |
|-------|-------|
| Model | `LSTMPredictor` |
| Module | `src.temporal` |
| Task | Next-step PCA embedding prediction for OES process monitoring |
| Framework | PyTorch |

## Architecture

- **Input**: Window of `seq_len` consecutive PCA embedding vectors (batch, seq_len, n_features)
- **LSTM**: 2-layer LSTM, hidden_size=64, dropout=0.1 between layers
- **Head**: Linear(64, n_features) applied to the last time step
- **Output**: Predicted next PCA embedding vector (batch, n_features)

## Intended Use

Temporal process monitoring for plasma etching. Predicts the next time step
in PCA-reduced OES spectral space, enabling anomaly detection when the
prediction error exceeds a threshold. Used in conjunction with DTW-based
discharge phase clustering.

## Training Data

- **Dataset**: BOSCH Plasma Etching OES (NetCDF, 3648 wavelength channels, 25 Hz)
- **Preprocessing**: PCA embedding of OES spectra (n_components=20)
- **Sequence construction**: Sliding window of length `seq_len` (default 50 time steps)
- **Loss**: MSE between predicted and actual next-step embedding

## Metrics

| Metric | Value |
|--------|-------|
| Training MSE | Dataset-dependent (varies by wafer/day) |
| Application | Anomaly detection via reconstruction error threshold |

Quantitative thresholds are set per-wafer using the training distribution
of prediction errors (e.g., 95th or 99th percentile).

## Limitations

- Requires sufficient temporal data (>100 time steps) for meaningful training.
- Assumes stationarity within a single etch process; model should be retrained
  or fine-tuned for significantly different process recipes.
- PCA dimensionality (n_features=20) may lose subtle spectral changes relevant
  to early fault detection.
- Real-time inference requires careful batching at 25 Hz OES sampling rate.
