import numpy as np
import torch
import pytest


def test_temperature_scaling_fit_transform():
    """TemperatureScaling fits and transforms logits."""
    from src.models.calibration import TemperatureScaling

    ts = TemperatureScaling()
    logits = np.random.randn(100, 5).astype(np.float32)
    labels = np.random.randint(0, 5, size=(100,))

    ts.fit(logits, labels)
    assert ts.T > 0

    calibrated = ts.transform(logits)
    assert calibrated.shape == logits.shape
    # Calibrated probabilities should sum to ~1
    row_sums = calibrated.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(100), atol=1e-5)
