"""Temporal analysis tools for OES time-series data.

Provides:
  - compute_temporal_embedding: PCA-based dimensionality reduction over time
  - cluster_discharge_phases: DTW K-means clustering of temporal OES segments
  - LSTMPredictor: PyTorch LSTM for next-step PCA embedding prediction
  - train_lstm: sliding-window training loop for LSTMPredictor
"""

from typing import Tuple, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_temporal_embedding(
    spectra: np.ndarray,
    n_components: int = 20,
    standardize: bool = True,
) -> Tuple[np.ndarray, PCA]:
    """Compute PCA temporal embedding of OES spectra time series.

    Reduces each time-step's high-dimensional spectrum to n_components PCA
    coordinates, yielding a (T, n_components) trajectory in latent space.

    Args:
        spectra: OES intensity array (T, n_wavelengths), float32 or float64
        n_components: Number of PCA components to retain (default 20)
        standardize: If True, apply StandardScaler before PCA (recommended)

    Returns:
        embedding: (T, n_components) float32 PCA embedding matrix
        pca: Fitted sklearn PCA object (for later transform / inverse_transform)
    """
    X = spectra.astype(np.float64)
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    embedding = pca.fit_transform(X).astype(np.float32)
    return embedding, pca


def cluster_discharge_phases(
    embedding: np.ndarray,
    k: int = 4,
    metric: str = "dtw",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Cluster temporal OES embedding using DTW K-means.

    Identifies discharge phase motifs (e.g., ignition, steady-state, extinction)
    by applying DTW-based K-means clustering on sliding windows of the PCA
    embedding.

    Args:
        embedding: (T, n_features) temporal PCA embedding
        k: Number of clusters (default 4)
        metric: Distance metric — 'dtw' (requires tslearn) or 'euclidean'
        seed: Random seed for reproducibility

    Returns:
        labels: (T,) integer cluster assignments per time step
        centroids: (k, n_features) cluster centroid time series
        inertia: Float inertia value for the clustering
    """
    if metric == "dtw":
        try:
            from tslearn.clustering import TimeSeriesKMeans
            # Reshape to (T, 1, n_features) as tslearn expects (n_samples, seq_len, n_features)
            X_ts = embedding.reshape(embedding.shape[0], 1, embedding.shape[1])
            km = TimeSeriesKMeans(
                n_clusters=k, metric="dtw",
                random_state=seed, n_jobs=-1, verbose=0,
            )
            labels = km.fit_predict(X_ts)
            centroids = km.cluster_centers_[:, 0, :]   # (k, n_features)
            inertia = float(km.inertia_)
        except ImportError:
            raise ImportError(
                "tslearn is required for DTW clustering. Install with: pip install tslearn"
            )
    else:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(embedding)
        centroids = km.cluster_centers_          # (k, n_features)
        inertia = float(km.inertia_)

    return labels, centroids, inertia


# ---------------------------------------------------------------------------
# LSTM temporal predictor (OES-017)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class LSTMPredictor(nn.Module if _TORCH_AVAILABLE else object):
    """LSTM model for next-step PCA embedding prediction.

    Given a window of seq_len consecutive PCA embedding vectors, predicts
    the next vector. Used for anomaly detection and process monitoring.

    Args:
        n_features: Dimensionality of the PCA embedding (default 20)
        hidden_size: LSTM hidden state size (default 64)
        n_layers: Number of LSTM layers (default 2)
        dropout: Dropout rate between LSTM layers (default 0.1)
    """

    def __init__(
        self,
        n_features: int = 20,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTMPredictor")
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, n_features)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass.

        Args:
            x: (batch, seq_len, n_features) input tensor

        Returns:
            (batch, n_features) predicted next step
        """
        out, _ = self.lstm(x)      # (batch, seq_len, hidden_size)
        return self.fc(out[:, -1])  # predict from last time step


def _build_sequences(
    embedding: np.ndarray,
    seq_len: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences from temporal embedding.

    Args:
        embedding: (T, n_features)
        seq_len: Window length

    Returns:
        X: (n_seqs, seq_len, n_features)
        y: (n_seqs, n_features) — next step targets
    """
    T = len(embedding)
    n = T - seq_len
    X = np.stack([embedding[i: i + seq_len] for i in range(n)])
    y = embedding[seq_len:]
    return X.astype(np.float32), y.astype(np.float32)


def train_lstm(
    model: "LSTMPredictor",
    sequences: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-3,
    val_split: float = 0.2,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> dict:
    """Train LSTMPredictor on sliding-window sequences.

    Args:
        model: LSTMPredictor instance
        sequences: (T, n_features) temporal embedding; sliding windows built internally
        epochs: Training epochs (default 100)
        lr: Learning rate (default 1e-3)
        val_split: Fraction of sequences reserved for validation (default 0.2)
        batch_size: Mini-batch size (default 64)
        device: 'cuda', 'cpu', or None (auto-detect)

    Returns:
        dict with 'train_loss' and 'val_loss' lists (length=epochs)
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for train_lstm")

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Normalise embedding to zero-mean unit-variance per feature before sliding windows
    # (PCA scores have unequal variances: PC1 >> PC20; scaling helps LSTM convergence)
    seq_mean = sequences.mean(axis=0, keepdims=True)
    seq_std = sequences.std(axis=0, keepdims=True).clip(1e-6)
    sequences_norm = ((sequences - seq_mean) / seq_std).astype(np.float32)

    # Build sliding-window sequences (seq_len from LSTMPredictor architecture)
    seq_len = 10
    X_all, y_all = _build_sequences(sequences_norm, seq_len=seq_len)

    # Random train/val split to avoid distribution shift from sequential split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_all))
    n_val = max(1, int(len(X_all) * val_split))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    n_train = len(X_train)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=batch_size, shuffle=True,
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(xb)
        train_losses.append(ep_loss / n_train)

        model.eval()
        with torch.no_grad():
            x_v = torch.FloatTensor(X_val).to(device)
            y_v = torch.FloatTensor(y_val).to(device)
            val_loss = criterion(model(x_v), y_v).item()
        val_losses.append(val_loss)

    return {"train_loss": train_losses, "val_loss": val_losses}
