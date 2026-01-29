"""Deep learning models for spectral regression."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, List, Tuple
from tqdm import tqdm


def _get_safe_device() -> str:
    """Get a device that is confirmed to work with PyTorch.

    Checks if CUDA is available and actually functional (can run a simple op).
    Falls back to CPU if CUDA is available but not compatible (e.g., GPU arch mismatch).

    Returns:
        "cuda" if CUDA is available and functional, "cpu" otherwise.
    """
    if not torch.cuda.is_available():
        return "cpu"

    try:
        # Try a simple CUDA operation to confirm it works
        test_tensor = torch.zeros(1, device="cuda")
        _ = test_tensor + 1
        return "cuda"
    except Exception:
        # CUDA is installed but not functional (e.g., incompatible GPU)
        return "cpu"


class Conv1DRegressor(nn.Module):
    """1D CNN for spectral regression."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        channels: List[int] = [64, 128, 256],
        kernel_size: int = 7,
        dropout: float = 0.3
    ):
        super().__init__()

        # Determine appropriate pooling factor based on input dimension
        # We need output_size >= 1 after all pooling layers
        # output_size = input_dim / (pool_size ^ n_layers)
        # Choose pool_size such that output remains >= 1
        n_layers = len(channels)
        pool_size = 4  # default
        if n_layers > 0:
            # Ensure output dimension >= 1 after all pooling
            max_pool_size = max(2, int(input_dim ** (1.0 / n_layers)))
            pool_size = min(4, max_pool_size)

        layers = []
        in_ch = 1
        current_dim = input_dim
        for ch in channels:
            # Dynamically adjust pool size if dimension is too small
            effective_pool = min(pool_size, max(1, current_dim // 2))
            if effective_pool < 1:
                effective_pool = 1

            layers += [
                nn.Conv1d(in_ch, ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
            ]
            # Only add pooling if it won't reduce to 0
            if effective_pool > 1:
                layers.append(nn.MaxPool1d(effective_pool))
                current_dim = current_dim // effective_pool

            in_ch = ch
        self.conv = nn.Sequential(*layers)

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            conv_out = self.conv(dummy)
            flat_size = conv_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dim
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class LSTMRegressor(nn.Module):
    """LSTM for spectral regression."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * (2 if bidirectional else 1)

        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(2)  # (batch, seq, 1)

        lstm_out, _ = self.lstm(x)
        # Use last time step
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


class TransformerRegressor(nn.Module):
    """Transformer for spectral regression."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        super().__init__()

        # Downsample input if too long
        self.downsample_factor = max(1, input_dim // max_seq_len)
        seq_len = input_dim // self.downsample_factor

        self.input_proj = nn.Linear(self.downsample_factor, d_model)

        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Reshape for downsampling
        if self.downsample_factor > 1:
            seq_len = x.size(1) // self.downsample_factor
            x = x[:, :seq_len * self.downsample_factor]
            x = x.view(batch_size, seq_len, self.downsample_factor)
        else:
            x = x.unsqueeze(2)

        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]

        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)
        return self.fc(x)


def train_deep_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device: Optional[str] = None,
    verbose: bool = True
) -> Tuple[nn.Module, dict]:
    """Train a deep learning model.

    Args:
        model: PyTorch model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience
        device: Device to use (auto-detect if None)
        verbose: Whether to print progress

    Returns:
        Tuple of (trained_model, history)
    """
    if device is None:
        device = _get_safe_device()

    model = model.to(device)

    # Prepare data
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if X_val is not None and y_val is not None:
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    else:
        val_loader = None

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience // 2)
    criterion = nn.MSELoss()

    # Training loop
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)

    for epoch in iterator:
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            if y_pred.dim() == 1:
                y_pred = y_pred.unsqueeze(1)
            if y_batch.dim() == 1:
                y_batch = y_batch.unsqueeze(1)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # Validation
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_pred = model(X_batch)
                    if y_pred.dim() == 1:
                        y_pred = y_pred.unsqueeze(1)
                    if y_batch.dim() == 1:
                        y_batch = y_batch.unsqueeze(1)
                    loss = criterion(y_pred, y_batch)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)
            history["val_loss"].append(avg_val_loss)
            scheduler.step(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            if verbose:
                iterator.set_postfix({
                    "train_loss": f"{avg_train_loss:.4f}",
                    "val_loss": f"{avg_val_loss:.4f}"
                })
        else:
            if verbose:
                iterator.set_postfix({"train_loss": f"{avg_train_loss:.4f}"})

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def predict(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 64,
    device: Optional[str] = None
) -> np.ndarray:
    """Make predictions with trained model.

    Args:
        model: Trained model
        X: Input features
        batch_size: Batch size
        device: Device to use

    Returns:
        Predictions array
    """
    if device is None:
        device = _get_safe_device()

    model = model.to(device)
    model.eval()

    X_t = torch.FloatTensor(X).to(device)
    dataset = TensorDataset(X_t)
    loader = DataLoader(dataset, batch_size=batch_size)

    predictions = []
    with torch.no_grad():
        for (X_batch,) in loader:
            y_pred = model(X_batch)
            predictions.append(y_pred.cpu().numpy())

    return np.vstack(predictions)


def save_model(model: nn.Module, filepath: str):
    """Save PyTorch model."""
    torch.save(model.state_dict(), filepath)


def load_model(model: nn.Module, filepath: str, device: Optional[str] = None) -> nn.Module:
    """Load PyTorch model."""
    if device is None:
        device = _get_safe_device()
    model.load_state_dict(torch.load(filepath, map_location=device))
    return model


from sklearn.base import BaseEstimator, RegressorMixin


class CNNRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper for Conv1DRegressor.

    Provides a sklearn-compatible interface for the 1D-CNN model,
    allowing it to be used with cross_val_predict and other sklearn utilities.

    Inherits from BaseEstimator and RegressorMixin for sklearn compatibility.
    """

    def __init__(
        self,
        channels: List[int] = None,
        kernel_size: int = 7,
        dropout: float = 0.3,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        verbose: bool = False,
    ):
        """Initialize CNN regressor.

        Args:
            channels: List of channel sizes for conv layers (default: [32, 64])
            kernel_size: Convolution kernel size
            dropout: Dropout rate
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            verbose: Whether to print training progress
        """
        self.channels = channels if channels is not None else [32, 64]
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.verbose = verbose
        self.model_ = None
        self.device_ = None
        self.input_dim_ = None
        self.output_dim_ = None

    def get_params(self, deep: bool = True):
        """Get parameters for sklearn."""
        return {
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        """Set parameters for sklearn."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the CNN model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target matrix (n_samples, n_targets) or (n_samples,)
        """
        self.device_ = _get_safe_device()
        self.input_dim_ = X.shape[1]

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.output_dim_ = y.shape[1]

        # Create model
        self.model_ = Conv1DRegressor(
            input_dim=self.input_dim_,
            output_dim=self.output_dim_,
            channels=self.channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )

        # Split for validation (10% of data)
        n_samples = X.shape[0]
        n_val = max(1, int(n_samples * 0.1))
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train
        self.model_, _ = train_deep_model(
            self.model_,
            X_train, y_train,
            X_val, y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            patience=self.patience,
            device=self.device_,
            verbose=self.verbose,
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        preds = predict(
            self.model_, X,
            batch_size=self.batch_size,
            device=self.device_,
        )

        # Squeeze if single output
        if preds.shape[1] == 1 and self.output_dim_ == 1:
            preds = preds.ravel()

        return preds

    def __repr__(self) -> str:
        return f"CNNRegressor(channels={self.channels}, kernel_size={self.kernel_size})"
