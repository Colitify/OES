"""Post-hoc calibration models for classifier probability outputs."""

import numpy as np
import torch
import torch.nn as nn


class TemperatureScaling:
    """Temperature scaling calibration for neural network classifiers.

    Fits a single scalar temperature T > 0 on validation logits/labels by
    minimising NLL via LBFGS. Divides logits by T before softmax, which
    preserves argmax predictions (accuracy unchanged) while improving
    probability calibration (lower ECE).

    Usage:
        ts = TemperatureScaling()
        ts.fit(val_logits, y_val)          # fit on held-out val set
        cal_probs = ts.transform(test_logits)  # apply to any logits

    Args:
        lr: LBFGS learning rate (default 0.01).
        max_iter: Maximum LBFGS iterations (default 100).
    """

    def __init__(self, lr: float = 0.01, max_iter: int = 100):
        self.lr = lr
        self.max_iter = max_iter
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def fit(self, logits_val: np.ndarray, y_val: np.ndarray) -> "TemperatureScaling":
        """Optimise temperature on validation logits.

        Args:
            logits_val: Raw (un-softmaxed) logits (n_samples, n_classes).
            y_val: Integer class labels (n_samples,).

        Returns:
            self (fitted).
        """
        logits_t = torch.FloatTensor(logits_val)
        y_t = torch.LongTensor(y_val.astype(np.int64))
        criterion = nn.CrossEntropyLoss()

        # Reset temperature and build fresh optimizer each fit call
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=self.lr, max_iter=self.max_iter
        )

        def _eval():
            optimizer.zero_grad()
            loss = criterion(logits_t / self.temperature.clamp(min=0.01), y_t)
            loss.backward()
            return loss

        optimizer.step(_eval)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling and return calibrated probabilities.

        Args:
            logits: Raw logits array (n_samples, n_classes).

        Returns:
            Calibrated probability array (n_samples, n_classes), rows sum to 1.
        """
        with torch.no_grad():
            scaled = torch.FloatTensor(logits) / self.temperature.clamp(min=0.01)
            return torch.softmax(scaled, dim=1).numpy()

    @property
    def T(self) -> float:
        """Return the fitted temperature scalar."""
        return float(self.temperature.item())
