"""Feature engineering module for spectral data."""

import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Tuple, List, Literal, Union


# NIST Atomic Spectra Database — strongest lines in 200–1000 nm for steel elements
# Source: NIST ASD (https://physics.nist.gov/asd), arc/spark conditions
NIST_EMISSION_LINES = {
    "C":  [247.856],
    "Si": [212.412, 243.515, 250.690, 251.432, 252.411, 252.852, 288.158],
    "Mn": [257.610, 259.373, 260.569, 279.482, 279.827, 280.106,
           403.076, 403.307, 403.449],
    "Cr": [205.552, 206.158, 267.716, 283.563, 284.325, 301.520,
           357.869, 359.349, 360.533, 425.433, 427.480, 428.972],
    "Mo": [202.030, 203.844, 281.615, 313.259, 315.816, 317.035,
           319.397, 320.883, 379.825, 386.411, 390.296],
    "Ni": [221.647, 227.021, 231.096, 232.003, 300.249,
           341.476, 344.626, 349.296, 361.939],
    "Cu": [213.598, 217.894, 224.261, 324.754, 327.396, 521.820],
    "Fe": [238.204, 239.562, 240.488, 248.814, 259.939, 260.709,
           271.442, 302.064, 358.119, 371.994, 373.487, 374.556,
           375.824, 381.584, 382.044, 404.581, 438.355],
}


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


def _correlation_column_wise(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute correlation between each column of X and y efficiently.

    Memory-efficient alternative to np.corrcoef(X.T, y) which creates
    a (n_features+1)x(n_features+1) matrix.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)

    Returns:
        Correlation coefficients (n_features,)
    """
    # Center X and y
    X_centered = X - X.mean(axis=0)
    y_centered = y - y.mean()

    # Compute covariance
    cov = (X_centered * y_centered[:, np.newaxis]).mean(axis=0)

    # Compute standard deviations
    X_std = X_centered.std(axis=0)
    y_std = y_centered.std()

    # Avoid division by zero
    X_std[X_std == 0] = 1e-10

    # Pearson correlation
    return cov / (X_std * y_std)


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
        # Average absolute correlation across targets (memory-efficient)
        correlations = np.zeros(X.shape[1])
        for i in range(y.shape[1]):
            corr = np.abs(_correlation_column_wise(X, y[:, i]))
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


def select_wavelengths_sdvs(
    X: np.ndarray,
    y: np.ndarray,
    n_per_element: int = 500,
    n_bins: int = 10,
) -> np.ndarray:
    """Select wavelengths using SDVS (Spectral Discriminant Variable Selection).

    Adapted from the SIA team (2nd place, LIBS 2022 competition) for single-
    measurement data.  The original SDVS uses repeat measurements to estimate
    within-class scatter; here, quantile bins over each target provide a
    pseudo-class approximation.

    For each target y_k, channels are scored by:
        SDVS_k = Sb_k / (Sw_k + eps)
    where Sb is the inter-class (between-bin) variance weighted by bin size,
    and Sw is the mean within-bin variance.  The per-channel score is averaged
    across targets.  The top-n SDVS indices are intersected with the top-n
    correlation indices; if the intersection is too small (<50 features), the
    union is used instead.

    Args:
        X: Spectra matrix (n_samples, n_wavelengths)
        y: Target matrix (n_samples, n_targets) or (n_samples,)
        n_per_element: Number of top wavelengths to keep per element
            (before merging).
        n_bins: Number of quantile bins used as pseudo-classes.

    Returns:
        Sorted array of selected wavelength indices.
    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_samples, n_wavelengths = X.shape
    n_targets = y.shape[1]
    eps = 1e-10

    sdvs_scores = np.zeros(n_wavelengths)
    corr_scores = np.zeros(n_wavelengths)

    for k in range(n_targets):
        yk = y[:, k]

        # --- SDVS via quantile bins ---
        bin_edges = np.quantile(yk, np.linspace(0, 1, n_bins + 1))
        bin_edges[-1] += 1e-6  # include the max value in the last bin
        bin_ids = np.digitize(yk, bin_edges[1:-1])  # 0 … n_bins-1

        overall_mean = X.mean(axis=0)
        sb = np.zeros(n_wavelengths)
        sw = np.zeros(n_wavelengths)

        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() < 2:
                continue
            Xb = X[mask]
            bin_mean = Xb.mean(axis=0)
            sb += mask.sum() * (bin_mean - overall_mean) ** 2
            sw += Xb.var(axis=0) * mask.sum()

        sb /= n_samples
        sw /= n_samples
        sdvs_k = sb / (sw + eps)
        sdvs_scores += sdvs_k

        # --- Correlation score ---
        corr_scores += np.abs(_correlation_column_wise(X, yk))

    sdvs_scores /= n_targets
    corr_scores /= n_targets

    # Top-n by SDVS and by correlation
    top_sdvs = set(np.argsort(sdvs_scores)[-n_per_element:].tolist())
    top_corr = set(np.argsort(corr_scores)[-n_per_element:].tolist())

    selected = top_sdvs & top_corr
    if len(selected) < 50:
        selected = top_sdvs | top_corr

    return np.sort(np.array(list(selected), dtype=int))


def select_wavelengths_nist(
    wavelengths: np.ndarray,
    target_names=None,
    delta_nm: float = 1.0,
) -> np.ndarray:
    """Select wavelength indices near NIST atomic emission lines.

    Args:
        wavelengths: Array of wavelength values in nm (length = n_wavelengths).
        target_names: Element names to include (e.g. ["Mo", "C"]). None = all.
        delta_nm: Half-window around each emission line in nm. Default 1.0.

    Returns:
        Sorted array of unique channel indices within the emission line windows.
    """
    if target_names is None:
        target_names = list(NIST_EMISSION_LINES.keys())

    selected = set()
    for name in target_names:
        for line_nm in NIST_EMISSION_LINES.get(name, []):
            mask = np.abs(wavelengths - line_nm) <= delta_nm
            selected.update(np.where(mask)[0].tolist())

    indices = np.sort(np.array(list(selected), dtype=int))
    if len(indices) == 0:
        raise ValueError(
            f"No channels found within ±{delta_nm} nm of NIST lines for "
            f"{target_names}. Check that wavelengths cover 200–1000 nm."
        )
    return indices


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Feature extraction and dimensionality reduction for spectral data."""

    def __init__(
        self,
        method: Literal["pca", "pls", "wavelength_selection", "none"] = "pca",
        n_components: int = 50,
        selection_method: Literal["correlation", "variance", "f_score", "sdvs", "nist"] = "correlation",
        wavelengths: Optional[np.ndarray] = None,
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
        self.wavelengths = wavelengths
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
            if self.selection_method == "nist":
                if self.wavelengths is None:
                    raise ValueError(
                        "wavelengths must be provided to FeatureExtractor for selection_method='nist'"
                    )
                self._selected_indices = select_wavelengths_nist(
                    self.wavelengths, delta_nm=1.0
                )
            elif self.selection_method == "sdvs":
                if y is None:
                    raise ValueError("y is required for sdvs wavelength selection")
                self._selected_indices = select_wavelengths_sdvs(
                    X, y, n_per_element=self.n_components
                )
            else:
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
