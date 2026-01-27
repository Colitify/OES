"""Deep learning models for spectral regression."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, List, Tuple
from tqdm import tqdm


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

        layers = []
        in_ch = 1
        for ch in channels:
            layers += [
                nn.Conv1d(in_ch, ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.MaxPool1d(4)
            ]
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
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(filepath, map_location=device))
    return model
