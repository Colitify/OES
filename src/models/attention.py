"""Attention-based models for spectral and temporal classification.

Provides:
  - AttentionLSTM: LSTM with additive temporal attention for sequence classification
  - SEConv1D: 1D-CNN with Squeeze-and-Excitation channel attention (SE-CNN component
    inspired by TrCSL architecture; does not include the LSTM temporal component)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List


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
        channels: Optional[List[int]] = None,
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


class SpectralTransformer(nn.Module):
    """1D Transformer encoder for spectral classification.

    Splits the spectrum into patches, projects them via linear embedding,
    adds positional encoding, and processes with multi-head self-attention.

    Inspired by ViT (Dosovitskiy 2021) adapted for 1D spectral data.

    Args:
        input_dim: Number of spectral channels
        n_classes: Output classes
        patch_size: Size of each spectral patch
        d_model: Transformer embedding dimension
        n_heads: Number of attention heads
        n_layers: Number of Transformer encoder layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int = 3648,
        n_classes: int = 3,
        patch_size: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        n_patches = (input_dim + patch_size - 1) // patch_size  # ceil division
        self.n_patches = n_patches

        # Patch embedding: linear projection of each patch
        self.patch_embed = nn.Linear(patch_size, d_model)

        # Learnable [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) spectral input

        Returns:
            (batch, n_classes) class logits
        """
        B = x.shape[0]

        # Pad spectrum to multiple of patch_size
        if x.shape[1] % self.patch_size != 0:
            pad_len = self.patch_size - (x.shape[1] % self.patch_size)
            x = torch.nn.functional.pad(x, (0, pad_len))

        # Reshape into patches: (B, n_patches, patch_size)
        patches = x.view(B, -1, self.patch_size)
        n_p = patches.shape[1]

        # Patch embedding
        tokens = self.patch_embed(patches)  # (B, n_patches, d_model)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, n_patches+1, d_model)

        # Add positional embedding (handle variable n_patches)
        pos = self.pos_embed[:, :n_p + 1, :]
        tokens = tokens + pos

        # Transformer encoder
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        # Classification from [CLS] token
        cls_out = tokens[:, 0]
        return self.fc(cls_out)
