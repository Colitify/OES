"""Feature engineering module for spectral data."""

import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Tuple, List, Literal, Union


def extract_peaks(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    height_percentile: float = 90,
    distance: int = 10,
    prominence: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Extract significant peaks from spectrum.

    Args:
        spectrum: Input spectrum
        wavelengths: Wavelength array
        height_percentile: Minimum height as percentile of spectrum
        distance: Minimum distance between peaks (in indices)
        prominence: Minimum prominence of peaks

    Returns:
        Tuple of (peak_wavelengths, peak_intensities, peak_properties)
    """
    threshold = np.percentile(spectrum, height_percentile)
    peaks, properties = find_peaks(
        spectrum,
        height=threshold,
        distance=distance,
        prominence=prominence
    )
    return wavelengths[peaks], spectrum[peaks], properties


def select_wavelengths(
    X: np.ndarray,
    y: np.ndarray,
    n_wavelengths: int = 100,
    method: Literal["correlation", "variance", "f_score"] = "correlation"
) -> np.ndarray:
    """Select most important wavelengths.

    Args:
        X: Spectra matrix (n_samples, n_wavelengths)
        y: Target values (n_samples,) or (n_samples, n_targets)
        n_wavelengths: Number of wavelengths to select
        method: Selection method

    Returns:
        Indices of selected wavelengths
    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if method == "correlation":
        # Average absolute correlation across targets
        correlations = np.zeros(X.shape[1])
        for i in range(y.shape[1]):
            corr = np.abs(np.corrcoef(X.T, y[:, i])[:-1, -1])
            correlations += corr
        correlations /= y.shape[1]
        indices = np.argsort(correlations)[-n_wavelengths:]

    elif method == "variance":
        # Select high variance wavelengths
        variances = np.var(X, axis=0)
        indices = np.argsort(variances)[-n_wavelengths:]

    elif method == "f_score":
        from sklearn.feature_selection import f_regression
        scores = np.zeros(X.shape[1])
        for i in range(y.shape[1]):
            f_scores, _ = f_regression(X, y[:, i])
            scores += f_scores
        indices = np.argsort(scores)[-n_wavelengths:]

    else:
        raise ValueError(f"Unknown method: {method}")

    return np.sort(indices)


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Feature extraction and dimensionality reduction for spectral data."""

    def __init__(
        self,
        method: Literal["pca", "pls", "wavelength_selection", "none"] = "pca",
        n_components: int = 50,
        selection_method: Literal["correlation", "variance", "f_score"] = "correlation"
    ):
        """Initialize feature extractor.

        Args:
            method: Extraction method
            n_components: Number of components/features to extract
            selection_method: Method for wavelength selection
        """
        self.method = method
        self.n_components = n_components
        self.selection_method = selection_method
        self._model = None
        self._selected_indices = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the feature extractor.

        Args:
            X: Spectra matrix
            y: Target values (required for PLS and some selection methods)
        """
        if self.method == "pca":
            self._model = PCA(n_components=self.n_components)
            self._model.fit(X)

        elif self.method == "pls":
            if y is None:
                raise ValueError("y is required for PLS")
            self._model = PLSRegression(n_components=self.n_components)
            self._model.fit(X, y)

        elif self.method == "wavelength_selection":
            if y is None:
                raise ValueError("y is required for wavelength selection")
            self._selected_indices = select_wavelengths(
                X, y, self.n_components, self.selection_method
            )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform spectra to features.

        Args:
            X: Spectra matrix

        Returns:
            Feature matrix
        """
        if self.method == "none":
            return X

        if self.method in ["pca", "pls"]:
            return self._model.transform(X)

        elif self.method == "wavelength_selection":
            return X[:, self._selected_indices]

        raise ValueError(f"Unknown method: {self.method}")

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform spectra."""
        return self.fit(X, y).transform(X)

    @property
    def explained_variance_ratio_(self) -> Optional[np.ndarray]:
        """Get explained variance ratio (PCA only)."""
        if self.method == "pca" and self._model is not None:
            return self._model.explained_variance_ratio_
        return None

    @property
    def selected_wavelength_indices(self) -> Optional[np.ndarray]:
        """Get selected wavelength indices."""
        return self._selected_indices


class SpectralFeatures:
    """Utility class for computing spectral features."""

    @staticmethod
    def compute_statistics(spectrum: np.ndarray) -> dict:
        """Compute statistical features from spectrum.

        Args:
            spectrum: Input spectrum

        Returns:
            Dictionary of statistical features
        """
        return {
            "mean": np.mean(spectrum),
            "std": np.std(spectrum),
            "max": np.max(spectrum),
            "min": np.min(spectrum),
            "range": np.ptp(spectrum),
            "skewness": float(np.mean(((spectrum - np.mean(spectrum)) / np.std(spectrum)) ** 3)),
            "kurtosis": float(np.mean(((spectrum - np.mean(spectrum)) / np.std(spectrum)) ** 4) - 3),
        }

    @staticmethod
    def compute_spectral_indices(
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        indices: List[Tuple[float, float]]
    ) -> List[float]:
        """Compute ratios between spectral regions.

        Args:
            spectrum: Input spectrum
            wavelengths: Wavelength array
            indices: List of (numerator_wavelength, denominator_wavelength) tuples

        Returns:
            List of spectral index values
        """
        results = []
        for num_wl, denom_wl in indices:
            num_idx = np.argmin(np.abs(wavelengths - num_wl))
            denom_idx = np.argmin(np.abs(wavelengths - denom_wl))
            if spectrum[denom_idx] != 0:
                results.append(spectrum[num_idx] / spectrum[denom_idx])
            else:
                results.append(np.nan)
        return results
