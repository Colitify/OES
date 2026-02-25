"""Unit tests for classifier-related functions in src.evaluation."""

import numpy as np
import pytest
from sklearn.svm import SVC
from sklearn.datasets import make_classification

from src.evaluation import evaluate_classifier, compute_ece, compute_snr_gain


class TestComputeEce:
    def test_perfect_calibration(self):
        """A perfectly calibrated model should have ECE close to 0."""
        n = 100
        # Perfect predictions: class 0 for all, probs = [1, 0]
        probs = np.zeros((n, 2), dtype=np.float64)
        probs[:, 0] = 1.0
        y_true = np.zeros(n, dtype=np.int64)
        ece = compute_ece(probs, y_true)
        assert ece < 0.05, f"Expected ECE ≈ 0 for perfect calibration, got {ece:.4f}"

    def test_returns_float_in_range(self):
        """ECE must be a float in [0, 1]."""
        rng = np.random.default_rng(42)
        n, n_cls = 200, 4
        probs = rng.dirichlet(alpha=[1.0] * n_cls, size=n)
        y_true = rng.integers(0, n_cls, size=n)
        ece = compute_ece(probs, y_true)
        assert isinstance(ece, float), "ECE must be a float"
        assert 0.0 <= ece <= 1.0, f"ECE out of [0,1]: {ece}"

    def test_uniform_probs_high_ece(self):
        """Uniform predicted probabilities with confident true labels → high ECE."""
        n = 200
        n_cls = 5
        probs = np.full((n, n_cls), 1.0 / n_cls)   # uniform: 0.2 per class
        y_true = np.zeros(n, dtype=np.int64)         # all class 0
        ece = compute_ece(probs, y_true)
        # ECE should be non-trivially positive (fraction wrong ~0.8)
        assert ece > 0.1, f"Expected higher ECE for uniform probs, got {ece:.4f}"


class TestComputeSnrGain:
    def test_denoised_signal_has_higher_snr(self):
        """SavGol-smoothed signal should have higher SNR than noisy input."""
        rng = np.random.default_rng(0)
        n_wl = 200
        t = np.linspace(0, 4 * np.pi, n_wl)
        clean = np.sin(t) + 2.0   # positive signal

        # Add noise
        noise = rng.normal(0, 0.3, (10, n_wl))
        X_raw = (np.tile(clean, (10, 1)) + noise).astype(np.float32)

        # Smooth with SavGol
        from scipy.signal import savgol_filter
        X_denoised = np.array(
            [savgol_filter(x, window_length=11, polyorder=3) for x in X_raw],
            dtype=np.float32,
        )

        snr_before, snr_after, gain, peak_loss = compute_snr_gain(X_raw, X_denoised)
        assert gain > 0, f"Expected SNR gain > 0, got {gain:.2f} dB"
        assert 0.0 <= peak_loss <= 1.0, f"peak_loss out of [0,1]: {peak_loss}"

    def test_returns_four_values(self):
        """compute_snr_gain must return a 4-tuple."""
        X = np.random.rand(5, 100).astype(np.float32) + 0.5
        result = compute_snr_gain(X, X)
        assert len(result) == 4, "Expected 4 return values"


class TestEvaluateClassifier:
    def test_returns_dict_with_primary_metric(self):
        """evaluate_classifier must return a dict containing 'micro_f1'."""
        X, y = make_classification(
            n_samples=100, n_features=20, n_classes=3, n_informative=5,
            n_redundant=3, random_state=42,
        )
        svc = SVC(kernel="linear", random_state=42)
        result = evaluate_classifier(svc, X, y, cv=3)
        assert isinstance(result, dict), "Result must be a dict"
        assert "micro_f1" in result, "Result must contain 'micro_f1'"

    def test_micro_f1_in_range(self):
        """micro_f1 must be in [0, 1]."""
        X, y = make_classification(n_samples=60, n_features=10, random_state=0)
        svc = SVC(kernel="linear", random_state=0)
        result = evaluate_classifier(svc, X, y, cv=3)
        assert 0.0 <= result["micro_f1"] <= 1.0, (
            f"micro_f1={result['micro_f1']} out of range"
        )

    def test_per_class_f1_present(self):
        """Result should contain per_class_f1 mapping."""
        X, y = make_classification(
            n_samples=90, n_features=15, n_classes=3, n_informative=6,
            n_redundant=3, random_state=7,
        )
        svc = SVC(kernel="linear", random_state=7)
        result = evaluate_classifier(svc, X, y, cv=3)
        assert "per_class_f1" in result, "Result must contain 'per_class_f1'"
