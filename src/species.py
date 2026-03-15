"""Species detection and classification for OES plasma spectra.

Provides:
  - nmf_decompose: NMF spectral decomposition into pure species components
  - match_lines_to_nist: Automated peak-to-emission-line matching
  - detect_species_presence: Threshold-based multi-label species detection
"""

import numpy as np
from sklearn.decomposition import NMF
from typing import Tuple, Optional, List, Dict


def nmf_decompose(
    X: np.ndarray,
    n_components: int = 5,
    max_iter: int = 500,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, NMF]:
    """Decompose spectral matrix into non-negative components via NMF.

    X ≈ W @ H, where each row of H is a "pure species" spectrum and
    columns of W are the corresponding abundances/concentrations.

    Args:
        X: Spectral matrix (n_samples, n_wavelengths), must be non-negative
        n_components: Number of spectral components to extract
        max_iter: Maximum NMF iterations
        random_state: Seed for reproducibility

    Returns:
        components: (n_components, n_wavelengths) — extracted spectral basis
        weights: (n_samples, n_components) — abundance/concentration matrix
        model: Fitted sklearn NMF object
    """
    X_nn = np.clip(X, 0, None)  # ensure non-negative

    model = NMF(
        n_components=n_components,
        init="nndsvda",
        max_iter=max_iter,
        random_state=random_state,
    )
    weights = model.fit_transform(X_nn)
    components = model.components_

    return components, weights, model


def match_lines_to_nist(
    detected_nm: np.ndarray,
    tolerance_nm: float = 1.5,
    line_db: Optional[Dict[str, list]] = None,
) -> List[Dict]:
    """Match detected peak wavelengths to known NIST emission lines.

    Searches all species in PLASMA_EMISSION_LINES for the closest match
    within tolerance_nm. Returns the best match (smallest residual) per peak.

    Args:
        detected_nm: Array of detected peak wavelengths (nm)
        tolerance_nm: Maximum allowed distance (nm) for a match
        line_db: Custom emission line dict; defaults to PLASMA_EMISSION_LINES

    Returns:
        List of dicts, one per detected peak:
        {detected, species, reference_nm, residual_nm} or species=None if unmatched
    """
    from src.features import PLASMA_EMISSION_LINES

    db = line_db or PLASMA_EMISSION_LINES
    results = []

    for peak in detected_nm:
        best = {"detected": float(peak), "species": None, "reference_nm": None, "residual_nm": None}
        best_dist = tolerance_nm + 1

        for species, lines in db.items():
            for ref_nm in lines:
                dist = abs(peak - ref_nm)
                if dist <= tolerance_nm and dist < best_dist:
                    best_dist = dist
                    best = {
                        "detected": float(peak),
                        "species": species,
                        "reference_nm": ref_nm,
                        "residual_nm": round(dist, 3),
                    }

        results.append(best)

    return results


def detect_species_presence(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    threshold_sigma: float = 3.0,
    line_db: Optional[Dict[str, list]] = None,
    delta_db: Optional[Dict[str, float]] = None,
) -> Dict[str, bool]:
    """Detect which plasma species are present in a single spectrum.

    For each species, computes max intensity in its emission line windows
    and compares to the spectrum's global mean + threshold_sigma * std.

    Args:
        spectrum: 1D intensity array (n_wavelengths,)
        wavelengths: Wavelength array in nm (n_wavelengths,)
        threshold_sigma: Number of std deviations above mean for detection
        line_db: Emission line dictionary (defaults to PLASMA_EMISSION_LINES)
        delta_db: Window widths (defaults to PLASMA_DELTA_NM)

    Returns:
        Dict mapping species name → bool (True = detected)
    """
    from src.features import PLASMA_EMISSION_LINES, PLASMA_DELTA_NM

    db = line_db or PLASMA_EMISSION_LINES
    deltas = delta_db or PLASMA_DELTA_NM

    global_mean = spectrum.mean()
    global_std = spectrum.std()
    threshold = global_mean + threshold_sigma * global_std

    presence = {}
    for species, lines in db.items():
        delta = deltas.get(species, 1.0)
        intensities = []
        for line_nm in lines:
            mask = np.abs(wavelengths - line_nm) <= delta
            if mask.any():
                intensities.append(spectrum[mask].max())

        if intensities:
            presence[species] = bool(max(intensities) > threshold)
        else:
            presence[species] = False

    return presence


def detect_species_presence_batch(
    X: np.ndarray,
    wavelengths: np.ndarray,
    threshold_sigma: float = 3.0,
) -> Tuple[np.ndarray, List[str]]:
    """Detect species presence across a batch of spectra.

    Args:
        X: Spectra matrix (n_samples, n_wavelengths)
        wavelengths: Wavelength array in nm
        threshold_sigma: Detection threshold in std deviations

    Returns:
        labels: Binary matrix (n_samples, n_species), 1=present
        species_names: List of species names (column order)
    """
    from src.features import PLASMA_EMISSION_LINES

    species_names = list(PLASMA_EMISSION_LINES.keys())
    labels = np.zeros((X.shape[0], len(species_names)), dtype=np.int64)

    for i in range(X.shape[0]):
        presence = detect_species_presence(X[i], wavelengths, threshold_sigma)
        for j, sp in enumerate(species_names):
            labels[i, j] = int(presence.get(sp, False))

    return labels, species_names
