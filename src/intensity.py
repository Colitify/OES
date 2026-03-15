"""Semi-quantitative intensity analysis for OES plasma spectra.

Provides:
  - compute_line_ratios: Emission line intensity ratios between species
  - actinometry: Normalize species intensity by Ar reference (absolute density proxy)
  - oes_to_process_regression: Regression from OES intensities to process parameters
"""

import numpy as np
from typing import Optional, Dict, Tuple
from src.features import PLASMA_EMISSION_LINES, PLASMA_DELTA_NM


def _species_mean_intensity(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    species: str,
) -> np.ndarray:
    """Compute mean intensity in a species' emission line windows."""
    lines = PLASMA_EMISSION_LINES.get(species, [])
    delta = PLASMA_DELTA_NM.get(species, 1.0)

    mask = np.zeros(len(wavelengths), dtype=bool)
    for line_nm in lines:
        mask |= np.abs(wavelengths - line_nm) <= delta

    if not mask.any():
        if spectra.ndim == 1:
            return 0.0
        return np.zeros(spectra.shape[0])

    if spectra.ndim == 1:
        return float(spectra[mask].mean())
    return spectra[:, mask].mean(axis=1)


def compute_line_ratios(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    numerator: str,
    denominator: str,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute emission line intensity ratio between two species.

    Ratio = mean_intensity(numerator) / mean_intensity(denominator).

    Args:
        spectra: (T, n_wavelengths) OES intensity matrix
        wavelengths: (n_wavelengths,) in nm
        numerator: Species key for numerator
        denominator: Species key for denominator
        eps: Small value to avoid division by zero

    Returns:
        (T,) intensity ratio time series
    """
    num = _species_mean_intensity(spectra, wavelengths, numerator)
    den = _species_mean_intensity(spectra, wavelengths, denominator)
    return num / (np.abs(den) + eps)


def actinometry(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    target_species: str,
    reference_species: str = "Ar_I",
    eps: float = 1e-10,
) -> np.ndarray:
    """Normalize target species intensity using actinometry.

    Actinometry divides the target emission by a reference gas (typically Ar)
    of known, constant concentration. The ratio is proportional to the target
    species' absolute density: n_target ~ I_target / I_Ar.

    Recommended Ar I pairings (similar upper-state energies for EEDF cancellation):
      F_I 703.7 nm (14.5 eV) <-> Ar_I 750.4 nm (13.5 eV) -- dE ~ 1.0 eV (good)
      O_I 777.4 nm (10.7 eV) <-> Ar_I 763.5 nm (13.2 eV) -- dE ~ 2.5 eV (marginal)

    Reference: Coburn & Chen (1980) J. Appl. Phys.

    Args:
        spectra: (T, n_wavelengths) OES intensity matrix
        wavelengths: (n_wavelengths,) in nm
        target_species: Species key to normalize
        reference_species: Reference species key (default "Ar_I")
        eps: Division safety

    Returns:
        (T,) actinometry-normalized intensity (proportional to abs. density)
    """
    return compute_line_ratios(spectra, wavelengths, target_species, reference_species, eps)


def oes_to_process_regression(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "ridge",
    cv: int = 5,
    seed: int = 42,
) -> Dict:
    """Train regression model from OES species intensities to process parameters.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)
        model_type: "ridge", "pls", "rf", or "ann"
        cv: CV folds
        seed: Random seed

    Returns:
        Dict with: rmse, r2, model, feature_importances (if available)
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import RandomForestRegressor

    if model_type == "ridge":
        model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    elif model_type == "pls":
        n_comp = min(X.shape[1], X.shape[0] // 2, 10)
        model = Pipeline([("scaler", StandardScaler()), ("reg", PLSRegression(n_components=max(1, n_comp)))])
    elif model_type == "rf":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)),
        ])
    elif model_type == "ann":
        from sklearn.neural_network import MLPRegressor
        from sklearn.ensemble import BaggingRegressor
        base = MLPRegressor(hidden_layer_sizes=(64,), activation="logistic",
                            solver="lbfgs", alpha=0.01, max_iter=2000, random_state=seed)
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", BaggingRegressor(estimator=base, n_estimators=16, random_state=seed, n_jobs=-1)),
        ])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    rmse_scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

    model.fit(X, y)

    result = {
        "rmse": float(-rmse_scores.mean()),
        "r2": float(r2_scores.mean()),
        "model": model,
    }

    if model_type == "rf":
        result["feature_importances"] = model.named_steps["reg"].feature_importances_.tolist()

    return result
