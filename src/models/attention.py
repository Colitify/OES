"""Attention-based models for spectral and temporal classification.

Provides:
  - AttentionLSTM: LSTM with additive temporal attention for sequence classification
  - SEConv1D: 1D-CNN with Squeeze-and-Excitation channel attention (SE-CNN component
    inspired by TrCSL architecture; does not include the LSTM temporal component)
"""

import torch
import torch.nn as nn
from typing import Tuple


class AttentionLSTM(nn.Module):
    """LSTM with temporal self-attention for OES sequence classification.

    Architecture: LSTM → Attention → FC.

    Args:
        n_features: Input feature dimension per timestep
        hidden_size: LSTM hidden state size
        n_layers: LSTM layers
        n_classes: Output dimension (n_classes for classification, 1 for regression)
        dropout: Dropout rate
    """

    def __init__(
        self,
        n_features: int = 20,
        hidden_size: int = 64,
        n_layers: int = 2,
        n_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.attn_fc = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        scores = self.attn_fc(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(scores, dim=1)
        context = (lstm_out * attn_weights.unsqueeze(-1)).sum(dim=1)
        output = self.fc(context)
        return output, attn_weights


class SEConv1D(nn.Module):
    """1D-CNN with Squeeze-and-Excitation attention for spectral classification.

    Inspired by SE-CNN component of TrCSL (JAAS 2025).

    Args:
        input_dim: Number of spectral channels
        n_classes: Output dimension
        channels: Conv1D channel sizes per layer
        se_reduction: SE block reduction ratio
    """

    def __init__(
        self,
        input_dim: int = 3648,
        n_classes: int = 3,
        channels: list = None,
        se_reduction: int = 16,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        layers = []
        in_ch = 1
        for ch in channels:
            layers.append(_SEBlock1D(in_ch, ch, se_reduction))
            layers.append(nn.MaxPool1d(4))
            in_ch = ch

        layers.append(nn.AdaptiveAvgPool1d(1))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.conv(x).squeeze(-1)
        return self.fc(feat)


class _SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block for 1D convolution."""

    def __init__(self, in_ch: int, out_ch: int, reduction: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 7, padding=3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
        mid = max(1, out_ch // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_ch, mid),
            nn.ReLU(),
            nn.Linear(mid, out_ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        w = self.se(out).unsqueeze(-1)
        return out * w
