import numpy as np
import torch
import pytest


def test_attention_lstm_forward():
    """AttentionLSTM produces correct output shape."""
    from src.models.attention import AttentionLSTM

    batch, seq_len, n_feat = 8, 20, 10
    n_classes = 3
    model = AttentionLSTM(n_features=n_feat, hidden_size=32, n_classes=n_classes)
    x = torch.randn(batch, seq_len, n_feat)
    out, attn_weights = model(x)

    assert out.shape == (batch, n_classes)
    assert attn_weights.shape == (batch, seq_len)
    assert torch.allclose(attn_weights.sum(dim=1), torch.ones(batch), atol=1e-5)


def test_attention_lstm_regression():
    """AttentionLSTM works in regression mode (n_classes=1)."""
    from src.models.attention import AttentionLSTM

    model = AttentionLSTM(n_features=5, hidden_size=16, n_classes=1)
    x = torch.randn(4, 10, 5)
    out, attn = model(x)
    assert out.shape == (4, 1)


def test_se_conv1d_forward():
    """SE-Conv1D produces correct output shape."""
    from src.models.attention import SEConv1D

    model = SEConv1D(input_dim=200, n_classes=5)
    x = torch.randn(8, 200)
    out = model(x)
    assert out.shape == (8, 5)


def test_se_conv1d_bosch_dim():
    """SE-Conv1D works with actual BOSCH dimensionality (3648 channels)."""
    from src.models.attention import SEConv1D

    model = SEConv1D(input_dim=3648, n_classes=3)
    x = torch.randn(4, 3648)
    out = model(x)
    assert out.shape == (4, 3)
