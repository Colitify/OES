"""Preprocessing module for spectral data."""

import numpy as np
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve
from scipy import sparse
from typing import Optional, Literal
from sklearn.base import BaseEstimator, TransformerMixin


def als_baseline(
    y: np.ndarray,
    lam: float = 1e6,
    p: float = 0.01,
    niter: int = 10
) -> np.ndarray:
    """Asymmetric Least Squares baseline correction.

    Args:
        y: Input spectrum
        lam: Smoothness parameter (larger = smoother)
        p: Asymmetry parameter (smaller = more asymmetric)
        niter: Number of iterations

    Returns:
        Estimated baseline
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def snv_normalize(spectrum: np.ndarray) -> np.ndarray:
    """Standard Normal Variate (SNV) normalization.

    Args:
        spectrum: Input spectrum

    Returns:
        SNV normalized spectrum
    """
    return (spectrum - np.mean(spectrum)) / np.std(spectrum)


def minmax_normalize(spectrum: np.ndarray) -> np.ndarray:
    """Min-Max normalization to [0, 1] range.

    Args:
        spectrum: Input spectrum

    Returns:
        Normalized spectrum
    """
    min_val = np.min(spectrum)
    max_val = np.max(spectrum)
    if max_val - min_val == 0:
        return np.zeros_like(spectrum)
    return (spectrum - min_val) / (max_val - min_val)


def l2_normalize(spectrum: np.ndarray) -> np.ndarray:
    """L2 (Euclidean) normalization.

    Args:
        spectrum: Input spectrum

    Returns:
        L2 normalized spectrum
    """
    norm = np.linalg.norm(spectrum)
    if norm == 0:
        return spectrum
    return spectrum / norm


def preprocess_spectrum(
    spectrum: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    baseline_correction: bool = True,
    smoothing: bool = True,
    normalization: str = "snv",
    baseline_lam: float = 1e6,
    baseline_p: float = 0.01,
    savgol_window: int = 11,
    savgol_polyorder: int = 3
) -> np.ndarray:
    """Standard preprocessing pipeline for a single spectrum.

    Args:
        spectrum: Input spectrum
        wavelengths: Wavelength array (optional, for future use)
        baseline_correction: Whether to apply ALS baseline correction
        smoothing: Whether to apply Savitzky-Golay smoothing
        normalization: Normalization method ("snv", "minmax", "l2", "none")
        baseline_lam: ALS smoothness parameter
        baseline_p: ALS asymmetry parameter
        savgol_window: Savitzky-Golay window length
        savgol_polyorder: Savitzky-Golay polynomial order

    Returns:
        Preprocessed spectrum
    """
    result = spectrum.copy()

    # 1. Baseline correction
    if baseline_correction:
        baseline = als_baseline(result, lam=baseline_lam, p=baseline_p)
        result = result - baseline

    # 2. Smoothing
    if smoothing:
        result = savgol_filter(result, window_length=savgol_window, polyorder=savgol_polyorder)

    # 3. Normalization
    if normalization == "snv":
        result = snv_normalize(result)
    elif normalization == "minmax":
        result = minmax_normalize(result)
    elif normalization == "l2":
        result = l2_normalize(result)

    return result


class Preprocessor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible preprocessor for spectral data."""

    def __init__(
        self,
        baseline: Literal["als", "none"] = "als",
        normalize: Literal["snv", "minmax", "l2", "none"] = "snv",
        denoise: Literal["savgol", "none"] = "savgol",
        baseline_lam: float = 1e6,
        baseline_p: float = 0.01,
        savgol_window: int = 11,
        savgol_polyorder: int = 3
    ):
        """Initialize preprocessor.

        Args:
            baseline: Baseline correction method
            normalize: Normalization method
            denoise: Denoising method
            baseline_lam: ALS smoothness parameter
            baseline_p: ALS asymmetry parameter
            savgol_window: Savitzky-Golay window length
            savgol_polyorder: Savitzky-Golay polynomial order
        """
        self.baseline = baseline
        self.normalize = normalize
        self.denoise = denoise
        self.baseline_lam = baseline_lam
        self.baseline_p = baseline_p
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder

    def fit(self, X: np.ndarray, y=None):
        """Fit the preprocessor (no-op for stateless preprocessing)."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform spectra matrix.

        Args:
            X: Spectra matrix of shape (n_samples, n_wavelengths)

        Returns:
            Preprocessed spectra matrix
        """
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(preprocess_spectrum)(
                X[i],
                baseline_correction=(self.baseline == "als"),
                smoothing=(self.denoise == "savgol"),
                normalization=self.normalize,
                baseline_lam=self.baseline_lam,
                baseline_p=self.baseline_p,
                savgol_window=self.savgol_window,
                savgol_polyorder=self.savgol_polyorder,
            )
            for i in range(X.shape[0])
        )
        return np.array(results, dtype=np.float32)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform spectra."""
        return self.fit(X, y).transform(X)
