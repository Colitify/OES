"""Feature engineering module for spectral data."""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Tuple, List, Literal, Union, Dict


# OES-008: Plasma OES emission line dictionary for high-voltage electrical discharge.
# Species: N2 molecular bands, N2+ ionic bands, atomic N/H/O, noble gas Ar.
# Sources: NIST ASD (https://physics.nist.gov/asd), Pearse & Gaydon "Identification of
#   Molecular Spectra", Laux et al. (2003) Plasma Sources Sci. Technol.
PLASMA_EMISSION_LINES: Dict[str, list] = {
    # N2 2nd positive system (C³Πu → B³Πg): prominent UV-vis bands in N2/air plasma
    "N2_2pos":  [315.9, 337.1, 357.7, 380.5],
    # N2+ 1st negative system (B²Σu+ → X²Σg+): marker of high-energy electrons
    "N2p_1neg": [391.4, 427.8],
    # Atomic nitrogen (N I) — visible NIR lines
    "N_I":      [746.8, 818.5, 862.9],
    # Hydrogen Balmer series — discharge gas contamination / water vapor indicator
    "H_alpha":  [656.3],
    "H_beta":   [486.1],
    # Atomic oxygen (O I) — air entrainment / oxygen plasma marker
    "O_I":      [777.4, 844.6, 926.6],
    # Argon I (Ar I) — noble gas discharge / Paschen lines
    "Ar_I":     [696.5, 706.7, 738.4, 750.4, 763.5, 772.4],
    # --- BOSCH RIE Process Species (SF₆/C₄F₈ etching) ---
    # Fluorine atomic (F I) — primary etchant radical from SF₆ dissociation
    # Source: NIST ASD (F I), Donnelly & Kornblit (2013) J. Vac. Sci. Technol. A 31, 050825
    "F_I":      [685.6, 690.2, 703.7, 712.8, 739.9],
    # Silicon atomic (Si I) — etch product indicator
    # Source: NIST ASD (Si I), verified in Coburn (1982) Plasma Chem. Plasma Process.
    "Si_I":     [243.5, 250.7, 251.6, 252.4, 288.2],
    # CF₂ radical (A¹B₁–X¹A₁) — C₄F₈ passivation decomposition product
    # Source: d'Agostino et al. (1981) Plasma Chem. Plasma Process. 1, 365
    "CF2":      [251.9, 259.5],
    # C₂ Swan bands (d³Πg–a³Πu) — carbon-containing plasma indicator
    # Source: Pearse & Gaydon "Identification of Molecular Spectra" (5th ed.)
    "C2_swan":  [473.7, 516.5, 563.6],
    # SiF molecular (A²Σ+–X²Π) — etch product (Si + F recombination)
    # Source: NIST ASD, Chung et al. (1975) J. Chem. Phys. 62, 1645
    "SiF":      [440.0, 442.5],
    # CO Ångström system (B¹Σ+–A¹Π) — from C₄F₈ + O₂ reaction
    # Source: NIST ASD (CO), Herzberg "Molecular Spectra" Vol. I
    # Note: 451.1, 519.8, 561.0 nm are the strongest B-A band heads
    "CO_angstrom": [451.1, 519.8, 561.0],
}

# Per-species window half-widths (nm) for PLASMA_EMISSION_LINES.
# Wider windows for broad molecular bands (N2_2pos) and Balmer lines (H_alpha, H_beta).
PLASMA_DELTA_NM: Dict[str, float] = {
    "N2_2pos":  2.0,
    "N2p_1neg": 1.5,
    "N_I":      1.0,
    "H_alpha":  2.0,
    "H_beta":   2.0,
    "O_I":      1.0,
    "Ar_I":     1.0,
    "F_I":          1.0,
    "Si_I":         1.0,
    "CF2":          1.5,
    "C2_swan":      2.0,
    "SiF":          2.0,
    "CO_angstrom":  1.5,
}


def select_wavelengths_plasma(
    wavelengths: np.ndarray,
    species: Optional[List[str]] = None,
) -> np.ndarray:
    """Return a boolean mask selecting channels near plasma OES emission lines.

    Uses PLASMA_EMISSION_LINES and PLASMA_DELTA_NM to define ±window windows
    around each emission line for the requested plasma species.

    Args:
        wavelengths: Array of wavelength values in nm (n_wavelengths,).
        species: List of species keys to include (e.g. ['N2_2pos', 'Ar_I']).
            None selects all species in PLASMA_EMISSION_LINES.

    Returns:
        Boolean mask of shape (n_wavelengths,); True where the channel falls
        within a plasma emission window.
    """
    if species is None:
        species = list(PLASMA_EMISSION_LINES.keys())

    mask = np.zeros(len(wavelengths), dtype=bool)
    for sp in species:
        delta = PLASMA_DELTA_NM.get(sp, 1.0)
        for line_nm in PLASMA_EMISSION_LINES.get(sp, []):
            mask |= np.abs(wavelengths - line_nm) <= delta

    return mask


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


def detect_peaks(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    min_prominence: float = 0.02,
    min_width_nm: float = 0.1,
) -> pd.DataFrame:
    """Detect emission peaks in a spectrum using prominence filtering.

    Uses scipy.signal.find_peaks with prominence constraint and
    scipy.signal.peak_widths to compute FWHM for each detected peak.

    Args:
        spectrum: 1D spectrum array (n_wavelengths,)
        wavelengths: Wavelength array in nm (n_wavelengths,)
        min_prominence: Minimum peak prominence (absolute units)
        min_width_nm: Informational minimum width in nm (reported in output;
            not enforced as a hard filter to preserve narrow delta-function
            peaks during validation)

    Returns:
        DataFrame with columns [wavelength_nm, intensity, prominence, fwhm_nm],
        sorted by descending prominence. Empty DataFrame if no peaks found.
    """
    peaks, properties = find_peaks(spectrum, prominence=min_prominence)

    if len(peaks) == 0:
        return pd.DataFrame(columns=["wavelength_nm", "intensity", "prominence", "fwhm_nm"])

    # FWHM in samples → convert to nm using average wavelength spacing
    widths_samples, _, _, _ = peak_widths(spectrum, peaks, rel_height=0.5)
    nm_per_sample = (wavelengths[-1] - wavelengths[0]) / (len(wavelengths) - 1)
    fwhm_nm = widths_samples * nm_per_sample

    df = pd.DataFrame({
        "wavelength_nm": wavelengths[peaks],
        "intensity": spectrum[peaks],
        "prominence": properties["prominences"],
        "fwhm_nm": fwhm_nm,
    })
    return df.sort_values("prominence", ascending=False).reset_index(drop=True)


def batch_detect_peaks(
    X: np.ndarray,
    wavelengths: np.ndarray,
    min_prominence: float = 0.02,
    min_width_nm: float = 0.1,
) -> List[pd.DataFrame]:
    """Detect peaks in a batch of spectra.

    Args:
        X: Spectra matrix (n_samples, n_wavelengths)
        wavelengths: Wavelength array in nm (n_wavelengths,)
        min_prominence: Minimum peak prominence (absolute units)
        min_width_nm: Informational minimum width in nm

    Returns:
        List of DataFrames (one per spectrum), each with columns
        [wavelength_nm, intensity, prominence, fwhm_nm].
    """
    return [
        detect_peaks(X[i], wavelengths, min_prominence=min_prominence, min_width_nm=min_width_nm)
        for i in range(X.shape[0])
    ]


class PeakDetector(BaseEstimator, TransformerMixin):
    """Sklearn-compatible peak detector for spectral data.

    Detects emission peaks per spectrum using prominence filtering.
    transform() returns a list of DataFrames (one per spectrum) since
    peak counts vary per spectrum. For pipeline integration requiring a
    fixed-width numeric array, use PlasmaDescriptorExtractor instead.
    """

    def __init__(
        self,
        wavelengths: Optional[np.ndarray] = None,
        min_prominence: float = 0.02,
        min_width_nm: float = 0.1,
    ):
        self.wavelengths = wavelengths
        self.min_prominence = min_prominence
        self.min_width_nm = min_width_nm

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray) -> List[pd.DataFrame]:
        """Detect peaks in each spectrum.

        Args:
            X: Spectra matrix (n_samples, n_wavelengths)

        Returns:
            List of DataFrames (one per spectrum), each with columns
            [wavelength_nm, intensity, prominence, fwhm_nm].
        """
        if self.wavelengths is None:
            raise ValueError("wavelengths must be provided to PeakDetector")
        return batch_detect_peaks(
            X, self.wavelengths,
            min_prominence=self.min_prominence,
            min_width_nm=self.min_width_nm,
        )


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


class PlasmaDescriptorExtractor(BaseEstimator, TransformerMixin):
    """Assemble a fixed-width descriptor vector from plasma OES spectra.

    Three feature blocks:
    - Block A: Mean intensity in each PLASMA_EMISSION_LINES window.
      One value per emission LINE (not per species); total = 20 features.
    - Block B: Top-K peaks by prominence, each described by
      [wavelength_nm, intensity, fwhm_nm]; zero-padded to K×3 = 60 features.
    - Block C: Global statistics [mean, std, skew, kurtosis, max_intensity,
      argmax_wl, N2_Hα_ratio, N2_OI_ratio] = 8 features.

    Total descriptor length = 20 + 60 + 8 = 88 features (default K=20).
    """

    def __init__(
        self,
        top_k_peaks: int = 20,
        min_prominence: float = 0.02,
        min_width_nm: float = 0.1,
    ):
        self.top_k_peaks = top_k_peaks
        self.min_prominence = min_prominence
        self.min_width_nm = min_width_nm
        self._wavelengths: Optional[np.ndarray] = None
        self._block_a_masks: Optional[List[np.ndarray]] = None

    def _build_block_a_masks(self, wavelengths: np.ndarray) -> List[np.ndarray]:
        masks = []
        for sp, lines in PLASMA_EMISSION_LINES.items():
            delta = PLASMA_DELTA_NM.get(sp, 1.0)
            for line_nm in lines:
                masks.append(np.abs(wavelengths - line_nm) <= delta)
        return masks

    def fit(self, X: np.ndarray, wavelengths: Optional[np.ndarray] = None):
        if wavelengths is not None:
            self._wavelengths = wavelengths
            self._block_a_masks = self._build_block_a_masks(wavelengths)
        return self

    def transform(self, X: np.ndarray, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        wl = wavelengths if wavelengths is not None else self._wavelengths
        if wl is None:
            raise ValueError("wavelengths must be provided to PlasmaDescriptorExtractor")

        if wavelengths is not None:
            block_a_masks = self._build_block_a_masks(wl)
        elif self._block_a_masks is not None:
            block_a_masks = self._block_a_masks
        else:
            block_a_masks = self._build_block_a_masks(wl)

        n_samples = X.shape[0]
        top_k = self.top_k_peaks
        n_block_a = len(block_a_masks)   # 20
        n_block_b = top_k * 3            # 60
        n_block_c = 8
        n_total = n_block_a + n_block_b + n_block_c

        n2_mask = select_wavelengths_plasma(wl, species=["N2_2pos"])
        halpha_mask = select_wavelengths_plasma(wl, species=["H_alpha"])
        oi_mask = select_wavelengths_plasma(wl, species=["O_I"])

        descriptor = np.zeros((n_samples, n_total), dtype=np.float32)
        for i in range(n_samples):
            spec = X[i].astype(np.float64)

            # Block A: mean intensity per emission line window
            for j, mask in enumerate(block_a_masks):
                if mask.any():
                    descriptor[i, j] = float(spec[mask].mean())

            # Block B: top-K peaks [wavelength_nm, intensity, fwhm_nm], zero-padded
            peaks_df = detect_peaks(
                spec, wl,
                min_prominence=self.min_prominence,
                min_width_nm=self.min_width_nm,
            )
            n_detected = min(len(peaks_df), top_k)
            if n_detected > 0:
                for k in range(n_detected):
                    row = peaks_df.iloc[k]
                    base = n_block_a + k * 3
                    descriptor[i, base] = float(row["wavelength_nm"])
                    descriptor[i, base + 1] = float(row["intensity"])
                    descriptor[i, base + 2] = float(row["fwhm_nm"])

            # Block C: global statistics
            mu = spec.mean()
            sigma = spec.std()
            eps = 1e-10
            normed = (spec - mu) / (sigma + eps)
            skew = float(np.mean(normed ** 3))
            kurt = float(np.mean(normed ** 4) - 3)
            max_int = float(spec.max())
            argmax_wl = float(wl[int(np.argmax(spec))])
            n2_mean = float(spec[n2_mask].mean()) if n2_mask.any() else 0.0
            halpha_mean = float(spec[halpha_mask].mean()) if halpha_mask.any() else eps
            oi_mean = float(spec[oi_mask].mean()) if oi_mask.any() else eps
            n2_halpha = n2_mean / (abs(halpha_mean) + eps)
            n2_oi = n2_mean / (abs(oi_mean) + eps)
            descriptor[i, n_block_a + n_block_b:] = np.array(
                [mu, sigma, skew, kurt, max_int, argmax_wl, n2_halpha, n2_oi],
                dtype=np.float32,
            )

        return descriptor

    def fit_transform(self, X: np.ndarray, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, wavelengths).transform(X, wavelengths)


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Feature extraction and dimensionality reduction for spectral data."""

    def __init__(
        self,
        method: Literal["pca", "pls", "wavelength_selection", "none"] = "pca",
        n_components: int = 50,
        selection_method: Literal["correlation", "variance", "f_score", "plasma_descriptor"] = "correlation",
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
        self._pde: Optional[PlasmaDescriptorExtractor] = None

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
            if self.selection_method == "plasma_descriptor":
                self._pde = PlasmaDescriptorExtractor()
                self._pde.fit(X, self.wavelengths)
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
            if self.selection_method == "plasma_descriptor" and self._pde is not None:
                return self._pde.transform(X, self.wavelengths)
            return X[:, self._selected_indices]

        raise ValueError(f"Unknown method: {self.method}")

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
        mu = np.mean(spectrum)
        sigma = np.std(spectrum)
        normed = (spectrum - mu) / (sigma + 1e-10)
        return {
            "mean": mu,
            "std": sigma,
            "max": np.max(spectrum),
            "min": np.min(spectrum),
            "range": np.ptp(spectrum),
            "skewness": float(np.mean(normed ** 3)),
            "kurtosis": float(np.mean(normed ** 4) - 3),
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
