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
    """Detect species presence across a batch of spectra (vectorized).

    Args:
        X: Spectra matrix (n_samples, n_wavelengths)
        wavelengths: Wavelength array in nm
        threshold_sigma: Detection threshold in std deviations

    Returns:
        labels: Binary matrix (n_samples, n_species), 1=present
        species_names: List of species names (column order)
    """
    from src.features import PLASMA_EMISSION_LINES, PLASMA_DELTA_NM

    species_names = list(PLASMA_EMISSION_LINES.keys())
    labels = np.zeros((X.shape[0], len(species_names)), dtype=np.int64)

    # Per-sample threshold: mean + sigma * std
    global_mean = X.mean(axis=1)
    global_std = X.std(axis=1)
    threshold = global_mean + threshold_sigma * global_std  # (n_samples,)

    for j, sp in enumerate(species_names):
        delta = PLASMA_DELTA_NM.get(sp, 1.0)
        lines = PLASMA_EMISSION_LINES.get(sp, [])
        mask = np.zeros(len(wavelengths), dtype=bool)
        for line_nm in lines:
            mask |= np.abs(wavelengths - line_nm) <= delta

        if mask.any():
            max_intensity = X[:, mask].max(axis=1)  # (n_samples,)
            labels[:, j] = (max_intensity > threshold).astype(np.int64)

    return labels, species_names


def train_species_classifier(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "svm",
    cv: int = 5,
    seed: int = 42,
) -> Dict:
    """Train a species/phase classifier with cross-validation.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Integer class labels (n_samples,)
        model_type: "svm", "rf", or "cnn"
        cv: Number of CV folds
        seed: Random seed

    Returns:
        Dict with keys: accuracy, f1_macro, f1_per_class, model, y_pred
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score

    if model_type == "svm":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=seed)),
        ])
    elif model_type == "rf":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1)),
        ])
    elif model_type == "cnn":
        return _train_cnn_classifier(X, y, cv=cv, seed=seed)
    elif model_type == "transformer":
        return _train_transformer_classifier(X, y, cv=cv, seed=seed)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(clf, X, y, cv=skf)

    clf.fit(X, y)

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro")),
        "f1_per_class": f1_score(y, y_pred, average=None).tolist(),
        "model": clf,
        "y_pred": y_pred,
    }


def _train_cnn_classifier(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    seed: int = 42,
    epochs: int = 50,
    lr: float = 1e-3,
) -> Dict:
    """Train 1D-CNN classifier for species/phase classification."""
    import torch
    import torch.nn as nn
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from src.models import get_safe_device

    # Define CNN locally to avoid module-level torch dependency
    class _SimpleCNN(nn.Module):
        def __init__(self, input_dim: int, n_classes: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_classes))
        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            return self.fc(self.conv(x).squeeze(-1))

    device = get_safe_device()
    n_classes = len(np.unique(y))
    scaler = StandardScaler()

    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    cw_tensor = torch.FloatTensor(class_weights).to(device)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    y_pred = np.zeros_like(y)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])

        X_t = torch.FloatTensor(X_tr).to(device)
        y_t = torch.LongTensor(y[train_idx]).to(device)
        X_v = torch.FloatTensor(X_va).to(device)

        model = _SimpleCNN(X.shape[1], n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss(weight=cw_tensor)

        batch_size = min(512, len(X_tr))
        model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(X_t))
            for start in range(0, len(X_t), batch_size):
                batch_idx = perm[start:start+batch_size]
                optimizer.zero_grad()
                loss = criterion(model(X_t[batch_idx]), y_t[batch_idx])
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_v).argmax(dim=1).cpu().numpy()
        y_pred[val_idx] = preds

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro")),
        "f1_per_class": f1_score(y, y_pred, average=None).tolist(),
        "model": None,
        "y_pred": y_pred,
    }


def _train_transformer_classifier(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    seed: int = 42,
    epochs: int = 80,
    lr: float = 5e-4,
) -> Dict:
    """Train SpectralTransformer classifier."""
    import torch
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight
    from src.models import get_safe_device
    from src.models.attention import SpectralTransformer

    device = get_safe_device()
    n_classes = len(np.unique(y))
    scaler = StandardScaler()

    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    cw_tensor = torch.FloatTensor(class_weights).to(device)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    y_pred = np.zeros_like(y)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])

        X_t = torch.FloatTensor(X_tr).to(device)
        y_t = torch.LongTensor(y[train_idx]).to(device)
        X_v = torch.FloatTensor(X_va).to(device)

        model = SpectralTransformer(
            input_dim=X.shape[1], n_classes=n_classes,
            patch_size=64, d_model=128, n_heads=4, n_layers=3, dropout=0.1,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss(weight=cw_tensor)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        batch_size = min(256, len(X_tr))
        model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(X_t))
            for start in range(0, len(X_t), batch_size):
                batch_idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                loss = criterion(model(X_t[batch_idx]), y_t[batch_idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_v).argmax(dim=1).cpu().numpy()
        y_pred[val_idx] = preds

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro")),
        "f1_per_class": f1_score(y, y_pred, average=None).tolist(),
        "model": None,
        "y_pred": y_pred,
    }


def compute_species_shap(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "rf",
    n_background: int = 50,
    max_samples: int = 1000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SHAP feature importance for species classifier.

    Uses PCA(50) to reduce dimensionality when features > 100, then
    maps SHAP values back to original feature space. Subsamples to
    max_samples for computational tractability.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Class labels (n_samples,)
        model_type: "rf" or "svm"
        n_background: Background samples for KernelExplainer
        max_samples: Maximum samples for SHAP computation
        seed: Random seed

    Returns:
        shap_values: SHAP values array (in reduced or original space)
        feature_importance: Mean |SHAP| per original feature (n_features,)
    """
    import shap
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    n_orig_features = X.shape[1]
    use_pca = n_orig_features > 100

    # Subsample for tractability
    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        idx = rng.choice(len(X), max_samples, replace=False)
        X_sub, y_sub = X[idx], y[idx]
    else:
        X_sub, y_sub = X, y

    if use_pca:
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_sub)
        n_comp = min(50, X_sc.shape[0], X_sc.shape[1])
        pca = PCA(n_components=n_comp, random_state=seed)
        X_pca = pca.fit_transform(X_sc)
        # Train classifier on PCA features
        result = train_species_classifier(X_pca, y_sub, model_type=model_type, cv=3, seed=seed)
    else:
        X_pca = X_sub
        pca = None
        result = train_species_classifier(X_sub, y_sub, model_type=model_type, cv=3, seed=seed)

    model = result["model"]

    if model_type == "rf":
        clf = model.named_steps["clf"]
        model_scaler = model.named_steps["scaler"]
        X_for_shap = model_scaler.transform(X_pca)
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_for_shap, check_additivity=False)
    else:
        model_scaler = model.named_steps["scaler"]
        X_for_shap = model_scaler.transform(X_pca)
        background = shap.kmeans(X_for_shap, min(n_background, len(X_for_shap)))
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_for_shap[:min(100, len(X_for_shap))])

    # Handle shap.Explanation objects (shap >= 0.44)
    if hasattr(shap_values, 'values'):
        shap_values = shap_values.values

    # Compute feature importance in PCA space
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        pca_importance = np.abs(shap_values).mean(axis=(0, 2))
    elif isinstance(shap_values, list):
        stacked = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values])
        pca_importance = stacked.mean(axis=0)
    else:
        pca_importance = np.abs(shap_values).mean(axis=0)

    # Map back to original feature space via PCA components
    if use_pca and pca is not None:
        # importance_orig[j] = sum_i |pca_importance[i] * pca.components_[i, j]|
        feature_importance = np.abs(pca.components_.T @ pca_importance)
    else:
        feature_importance = pca_importance

    return shap_values, feature_importance
