"""Preprocessing module for spectral data."""

import numpy as np
from scipy.ndimage import median_filter
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
    DDT = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + DDT
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
    std = np.std(spectrum)
    if std < 1e-10:
        return spectrum - np.mean(spectrum)
    return (spectrum - np.mean(spectrum)) / std


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
    savgol_polyorder: int = 3,
    internal_standard: bool = False,
    internal_standard_idx: int = -1,
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
        internal_standard: Whether to apply internal standard normalization
            (divide by intensity at the reference line). Applied after baseline
            correction and smoothing, before SNV/minmax/l2 normalization.
        internal_standard_idx: Channel index of the reference spectral line.
            Typically Fe 259.94 nm. Ignored when internal_standard=False.

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

    # 3. Internal standard normalization (ML-018)
    #    Divide by a stable Fe line (physical, shot-to-shot correction).
    #    Complementary to the data-driven SNV that follows.
    if internal_standard and internal_standard_idx >= 0:
        ref = result[internal_standard_idx]
        if abs(ref) > 1e-10:
            result = result / ref

    # 4. Normalization
    if normalization == "snv":
        result = snv_normalize(result)
    elif normalization == "minmax":
        result = minmax_normalize(result)
    elif normalization == "l2":
        result = l2_normalize(result)

    return result


class Preprocessor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible preprocessor for spectral data."""

    @staticmethod
    def align_wavelengths(
        X: np.ndarray,
        src_wl: np.ndarray,
        tgt_wl: np.ndarray,
    ) -> np.ndarray:
        """Align spectra from source wavelength grid to target wavelength grid via interpolation.

        Uses scipy.interpolate.interp1d with linear interpolation and extrapolation
        to map each spectrum from src_wl to tgt_wl.

        Args:
            X: Spectra matrix of shape (n_samples, n_src_channels)
            src_wl: Source wavelength array of shape (n_src_channels,), in nm
            tgt_wl: Target wavelength array of shape (n_tgt_channels,), in nm

        Returns:
            X_aligned: ndarray of shape (n_samples, n_tgt_channels), float32
        """
        from scipy.interpolate import interp1d

        interpolator = interp1d(
            src_wl, X, kind="linear", axis=1,
            fill_value="extrapolate", bounds_error=False,
        )
        X_aligned = interpolator(tgt_wl).astype(np.float32)
        return X_aligned

    @staticmethod
    def cosmic_ray_removal(X: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        """Remove cosmic ray spikes from spectra using local median filtering.

        A spike is detected when |spectrum[j] - local_median[j]| / sigma > threshold,
        where local_median is computed over an 11-channel window and sigma is the
        standard deviation of the per-channel residuals (spectrum - local_median).
        Detected spikes are replaced by the local median value.

        Args:
            X: Spectra matrix of shape (n_samples, n_wavelengths), float32 or float64
            threshold: Z-score threshold for spike detection (default 5.0)

        Returns:
            X_cleaned: Array of same shape with cosmic ray spikes replaced
        """
        result = X.copy()
        for i in range(X.shape[0]):
            spec = X[i].astype(np.float64)
            local_med = median_filter(spec, size=11, mode="nearest")
            residuals = spec - local_med
            sigma = float(np.std(residuals))
            if sigma < 1e-10:
                continue
            z_score = np.abs(residuals) / sigma
            spike_mask = z_score > threshold
            result[i][spike_mask] = local_med[spike_mask].astype(result.dtype)
        return result

    def __init__(
        self,
        baseline: Literal["als", "none"] = "als",
        normalize: Literal["snv", "minmax", "l2", "none"] = "snv",
        denoise: Literal["savgol", "none"] = "savgol",
        baseline_lam: float = 1e6,
        baseline_p: float = 0.01,
        savgol_window: int = 11,
        savgol_polyorder: int = 3,
        cosmic_ray: bool = True,
        cosmic_ray_threshold: float = 5.0,
        internal_standard: bool = False,
        internal_standard_wl: float = 259.94,
        wavelengths: Optional[np.ndarray] = None,
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
            cosmic_ray: If True, apply cosmic ray removal as the first preprocessing step.
            cosmic_ray_threshold: Z-score threshold for spike detection (default 5.0).
            internal_standard: If True, divide each spectrum by the intensity
                of the reference line (Fe 259.94 nm by default) after baseline
                correction and smoothing. Physical normalization that corrects
                for shot-to-shot laser/plasma energy fluctuations. (ML-018)
            internal_standard_wl: Wavelength (nm) of the internal standard
                reference line. Default Fe 259.94 nm.
            wavelengths: Wavelength array (nm) of the spectra. Required when
                internal_standard=True to locate the reference channel index.
        """
        self.baseline = baseline
        self.normalize = normalize
        self.denoise = denoise
        self.baseline_lam = baseline_lam
        self.baseline_p = baseline_p
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        self.cosmic_ray = cosmic_ray
        self.cosmic_ray_threshold = cosmic_ray_threshold
        self.internal_standard = internal_standard
        self.internal_standard_wl = internal_standard_wl
        self.wavelengths = wavelengths
        self._internal_standard_idx: int = -1

    def fit(self, X: np.ndarray, y=None):
        """Fit the preprocessor.

        Locates the internal standard channel index if internal_standard=True.
        """
        if self.internal_standard and self.wavelengths is not None:
            self._internal_standard_idx = int(
                np.argmin(np.abs(self.wavelengths - self.internal_standard_wl))
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform spectra matrix.

        Args:
            X: Spectra matrix of shape (n_samples, n_wavelengths)

        Returns:
            Preprocessed spectra matrix
        """
        # Step 0: Cosmic ray removal (first step, before all other preprocessing)
        if self.cosmic_ray:
            X = self.cosmic_ray_removal(X, threshold=self.cosmic_ray_threshold)

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
                internal_standard=self.internal_standard,
                internal_standard_idx=self._internal_standard_idx,
            )
            for i in range(X.shape[0])
        )
        return np.array(results, dtype=np.float32)
