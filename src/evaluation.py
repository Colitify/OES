"""Evaluation and visualization module for spectral analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict, GroupKFold
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path


def _add_overall_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Add overall aggregate metrics under metrics["_overall"].

    Overall metrics are computed across targets (mean/std of per-target RMSE, etc.).
    """
    # Collect per-target entries (skip reserved keys like "_overall")
    target_items = [(k, v) for k, v in metrics.items() if not str(k).startswith("_")]
    if not target_items:
        metrics["_overall"] = {
            "RMSE_mean": float("nan"),
            "RMSE_std": float("nan"),
            "MAE_mean": float("nan"),
            "R2_mean": float("nan"),
            "MAPE_mean": float("nan"),
        }
        return metrics

    rmse_list = [v.get("RMSE", np.nan) for _, v in target_items]
    mae_list = [v.get("MAE", np.nan) for _, v in target_items]
    r2_list = [v.get("R2", np.nan) for _, v in target_items]
    mape_list = [v.get("MAPE", np.nan) for _, v in target_items]

    metrics["_overall"] = {
        "RMSE_mean": float(np.nanmean(rmse_list)),
        "RMSE_std": float(np.nanstd(rmse_list)),
        "MAE_mean": float(np.nanmean(mae_list)),
        "R2_mean": float(np.nanmean(r2_list)),
        "MAPE_mean": float(np.nanmean(mape_list)),
    }
    return metrics


def make_target_groups(n_samples: int, n_per_target: int = 50) -> np.ndarray:
    """Create group labels for per-target GroupKFold.

    Assumes data is ordered: rows 0..n-1 = target 0, rows n..2n-1 = target 1, etc.
    """
    n_targets = n_samples // n_per_target
    return np.repeat(np.arange(n_targets), n_per_target)


def aggregate_per_target(arr: np.ndarray, n_per_target: int = 50) -> np.ndarray:
    """Average rows within each target block.

    arr shape: (n_samples,) or (n_samples, n_cols) → (n_groups, ...) after averaging.
    """
    n_groups = len(arr) // n_per_target
    if arr.ndim == 1:
        return arr.reshape(n_groups, n_per_target).mean(axis=1)
    return arr.reshape(n_groups, n_per_target, arr.shape[1]).mean(axis=1)


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    target_names: Optional[List[str]] = None,
    pred_transform=None,
    y_true: Optional[np.ndarray] = None,
    n_per_target: int = 0,
    normalize_sum100: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray]:
    """Comprehensive model evaluation with cross-validation.

    Args:
        model: Sklearn-compatible model
        X: Feature matrix
        y: Target matrix (may be transformed, e.g. logit-space)
        cv: Number of cross-validation folds
        target_names: Names of target variables
        pred_transform: Optional callable applied to cross_val_predict output
            before computing metrics (e.g. logit inverse transform).
        y_true: Ground-truth targets in original space. When provided, metrics
            are computed against y_true instead of y. Required when y has been
            transformed and pred_transform is supplied.
        n_per_target: If >0, use GroupKFold (groups = rows 0..n-1 = target 0, etc.)
            and additionally compute per-target aggregated RMSE as a secondary metric
            stored in metrics["_overall"]["per_target_RMSE_mean"].
        normalize_sum100: If True, apply closure constraint after prediction:
            rescale each sample so that all element predictions sum to 100 wt%.
            Based on the physical constraint that steel compositions sum to ≈100%.
            (ML-022; Noll et al. 2014)

    Returns:
        Tuple of (metrics_dict, predictions_in_original_space)

    Notes:
        - This function does NOT clip negative predictions by default.
          If you want non-negative concentrations, clip outside before computing metrics.
    """
    if n_per_target > 0:
        groups = make_target_groups(len(X), n_per_target)
        cv_splits = list(GroupKFold(n_splits=cv).split(X, y, groups))
        y_pred = cross_val_predict(model, X, y, cv=cv_splits)
    else:
        y_pred = cross_val_predict(model, X, y, cv=cv)

    # Inverse-transform predictions if requested (e.g. logit → wt%)
    if pred_transform is not None:
        y_pred = pred_transform(y_pred)

    # ML-022: Closure constraint — rescale predictions so Σ(wt%) = 100
    if normalize_sum100 and y_pred.ndim == 2 and y_pred.shape[1] > 1:
        row_sums = y_pred.sum(axis=1, keepdims=True)
        row_sums = np.where(np.abs(row_sums) < 1e-8, 1.0, row_sums)
        y_pred = y_pred / row_sums * 100.0

    # Use original-space ground truth for metrics when y was transformed
    y_for_metrics = y_true if y_true is not None else y

    if y_for_metrics.ndim == 1:
        y_for_metrics = y_for_metrics.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    elif y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_targets = y_for_metrics.shape[1]
    if target_names is None:
        target_names = [f"Target_{i}" for i in range(n_targets)]

    metrics: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(target_names):
        metrics[name] = {
            "RMSE": float(np.sqrt(mean_squared_error(y_for_metrics[:, i], y_pred[:, i]))),
            "MAE": float(mean_absolute_error(y_for_metrics[:, i], y_pred[:, i])),
            "R2": float(r2_score(y_for_metrics[:, i], y_pred[:, i])),
            "MAPE": float(np.mean(np.abs((y_for_metrics[:, i] - y_pred[:, i]) / (y_for_metrics[:, i] + 1e-8))) * 100.0),
        }

    metrics = _add_overall_metrics(metrics)

    # Secondary metric: per-target aggregated RMSE (50-spectra averaging)
    if n_per_target > 0:
        y_agg = aggregate_per_target(y_pred, n_per_target)
        y_ref_agg = aggregate_per_target(y_for_metrics, n_per_target)
        pt_rmse_list = [
            float(np.sqrt(mean_squared_error(y_ref_agg[:, i], y_agg[:, i])))
            for i in range(n_targets)
        ]
        metrics["_overall"]["per_target_RMSE_mean"] = float(np.mean(pt_rmse_list))
        metrics["_overall"]["per_target_RMSE_std"] = float(np.std(pt_rmse_list))
        metrics["_overall"]["per_target_n"] = n_per_target

    return metrics, y_pred


def compare_models(
    models_dict: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    target_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compare multiple models.

    Args:
        models_dict: Dictionary of {model_name: model}
        X: Feature matrix
        y: Target matrix
        cv: Number of cross-validation folds
        target_names: Names of target variables

    Returns:
        DataFrame with comparison results (per target only; excludes "_overall")
    """
    results = []

    for model_name, model in models_dict.items():
        print(f"Evaluating {model_name}.")
        metrics, _ = evaluate_model(model, X, y, cv, target_names)

        for target_name, target_metrics in metrics.items():
            # Skip "_overall" in model comparison table (keeps your existing behavior stable)
            if str(target_name).startswith("_"):
                continue

            for metric_name, value in target_metrics.items():
                results.append({
                    "Model": model_name,
                    "Target": target_name,
                    "Metric": metric_name,
                    "Value": value
                })

    df = pd.DataFrame(results)
    return df.pivot_table(
        index=["Model", "Target"],
        columns="Metric",
        values="Value"
    ).reset_index()


def plot_spectrum(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    peaks: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: str = "Spectrum",
    xlabel: str = "Wavelength (nm)",
    ylabel: str = "Intensity",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot single spectrum with optional peak markers."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(wavelengths, spectrum, "b-", lw=0.5)

    if peaks is not None:
        ax.scatter(peaks[0], peaks[1], c="r", s=20, zorder=5, label="Peaks")
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_spectra_comparison(
    wavelengths: np.ndarray,
    spectra_dict: Dict[str, np.ndarray],
    title: str = "Spectra Comparison",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot multiple spectra for comparison."""
    fig, ax = plt.subplots(figsize=figsize)

    for label, spectrum in spectra_dict.items():
        ax.plot(wavelengths, spectrum, lw=0.8, label=label)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_prediction_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    figsize_per_target: Tuple[int, int] = (4, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot true vs predicted for each target."""
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    n_targets = y_true.shape[1]
    if target_names is None:
        target_names = [f"Target_{i}" for i in range(n_targets)]

    fig, axes = plt.subplots(
        1, n_targets,
        figsize=(figsize_per_target[0] * n_targets, figsize_per_target[1])
    )

    if n_targets == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, target_names)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=10)

        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1)

        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")

        r2 = r2_score(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        ax.set_title(f"{name}\nR² = {r2:.3f}, RMSE = {rmse:.3f}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "RMSE",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot bar chart comparing models."""
    pivot_df = comparison_df.pivot(index="Target", columns="Model", values=metric)

    fig, ax = plt.subplots(figsize=figsize)
    pivot_df.plot(kind="bar", ax=ax)

    ax.set_xlabel("Target")
    ax.set_ylabel(metric)
    ax.set_title(f"Model Comparison - {metric}")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def generate_report(
    metrics: Dict[str, Dict[str, float]],
    model_name: str,
    save_path: Optional[str] = None
) -> str:
    """Generate text report of model performance.

    Notes:
        - This will include "_overall" if present (useful for quick reading).
    """
    lines = [
        "=" * 60,
        f"Model Evaluation Report: {model_name}",
        "=" * 60,
        ""
    ]

    # Print per-target first (skip overall), then overall at end if exists
    for target_name, target_metrics in metrics.items():
        if str(target_name).startswith("_"):
            continue
        lines.append(f"Target: {target_name}")
        lines.append("-" * 40)
        for metric_name, value in target_metrics.items():
            lines.append(f"  {metric_name}: {value:.4f}")
        lines.append("")

    if "_overall" in metrics:
        lines.append("Overall:")
        lines.append("-" * 40)
        for metric_name, value in metrics["_overall"].items():
            lines.append(f"  {metric_name}: {value:.4f}")
        lines.append("")

    report = "\n".join(lines)

    if save_path:
        Path(save_path).write_text(report)

    return report


def compute_snr_gain(
    X_raw: np.ndarray,
    X_denoised: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute SNR gain from denoising and peak loss fraction.

    SNR formula: signal = mean(X), noise = std(X - X_ref); dB = 20*log10(signal/noise)
    where X_ref is a smooth reference (local mean).

    Peak loss: fraction of peaks detected in X_raw (by scipy.signal.find_peaks) that
    are missing in X_denoised (peak centroid shifted > 5% of spectrum width).

    Args:
        X_raw: Raw spectra matrix of shape (n_samples, n_wavelengths)
        X_denoised: Denoised spectra of same shape

    Returns:
        snr_before_db: Mean SNR (dB) of raw spectra
        snr_after_db: Mean SNR (dB) of denoised spectra
        gain_db: snr_after_db - snr_before_db
        peak_loss_pct: Fraction of raw peaks missing in denoised spectra (0-1)
    """
    from scipy.signal import find_peaks, savgol_filter

    def _snr_db(X: np.ndarray) -> float:
        """Compute mean SNR in dB for a spectrum matrix."""
        snrs = []
        for i in range(X.shape[0]):
            spec = X[i].astype(np.float64)
            signal = float(np.mean(np.abs(spec)))
            if signal < 1e-10:
                continue
            # Smooth reference: 11-point Savitzky-Golay or simple local mean
            wl = min(11, len(spec) if len(spec) % 2 == 1 else len(spec) - 1)
            try:
                ref = savgol_filter(spec, window_length=wl, polyorder=3)
            except Exception:
                ref = spec
            noise = float(np.std(spec - ref))
            if noise < 1e-10:
                snrs.append(100.0)
            else:
                snrs.append(20.0 * np.log10(signal / noise))
        return float(np.mean(snrs)) if snrs else 0.0

    snr_before_db = _snr_db(X_raw)
    snr_after_db = _snr_db(X_denoised)
    gain_db = snr_after_db - snr_before_db

    # Peak loss: sample a subset of spectra to keep it fast
    n_check = min(50, X_raw.shape[0])
    indices = np.linspace(0, X_raw.shape[0] - 1, n_check, dtype=int)
    n_wavelengths = X_raw.shape[1]
    threshold_channels = int(0.05 * n_wavelengths)  # 5% of spectrum width

    total_raw_peaks = 0
    lost_peaks = 0
    for i in indices:
        raw_spec = X_raw[i].astype(np.float64)
        den_spec = X_denoised[i].astype(np.float64)
        raw_peaks, _ = find_peaks(raw_spec, prominence=0.01 * (raw_spec.max() - raw_spec.min()))
        den_peaks, _ = find_peaks(den_spec, prominence=0.01 * (den_spec.max() - den_spec.min()))
        total_raw_peaks += len(raw_peaks)
        for rp in raw_peaks:
            # Check if any denoised peak is within threshold_channels
            if len(den_peaks) == 0 or np.min(np.abs(den_peaks - rp)) > threshold_channels:
                lost_peaks += 1

    peak_loss_pct = float(lost_peaks / total_raw_peaks) if total_raw_peaks > 0 else 0.0

    return snr_before_db, snr_after_db, gain_db, peak_loss_pct


def compute_shap_spectrum(
    model: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    pca: Any = None,
) -> np.ndarray:
    """Compute SHAP values for a Conv1DClassifier using GradientExplainer.

    Uses shap.GradientExplainer to attribute each class prediction to input
    features. If pca is provided, SHAP values in PCA space are projected back
    to wavelength space via PCA component vectors.

    Args:
        model: Trained Conv1DClassifier (PyTorch nn.Module).
        X_background: Background spectra for SHAP baseline (n_bg, n_features).
        X_explain: Spectra to explain (n_explain, n_features).
        wavelengths: Wavelength array in nm (n_wavelengths,). Informational;
            used only when pca is None to determine output channels.
        pca: Optional fitted sklearn PCA object. If provided, SHAP values are
            projected from PCA space → wavelength space using pca.components_.

    Returns:
        shap_values: ndarray of shape (n_explain, n_out_features, n_classes),
            where n_out_features = n_wavelengths (if pca provided) or
            n_features (if pca is None).
    """
    import shap
    import torch

    device = "cpu"  # GradientExplainer runs on CPU for stability
    model_cpu = model.to(device).eval()

    # Feed as (batch, 1, n_features) for Conv1D forward compatibility
    X_bg_t = torch.FloatTensor(X_background[:, np.newaxis, :])
    X_exp_t = torch.FloatTensor(X_explain[:, np.newaxis, :])

    explainer = shap.GradientExplainer(model_cpu, X_bg_t)
    raw_shap = explainer.shap_values(X_exp_t)
    # raw_shap shape: (n_explain, 1, n_features, n_classes) — numpy array from GradientExplainer

    n_explain = X_explain.shape[0]
    n_features = X_explain.shape[1]

    if isinstance(raw_shap, list):
        # Older shap versions return list of (n_explain, 1, n_features) per class
        n_classes = len(raw_shap)
        shap_arr = np.stack([sv[:, 0, :] for sv in raw_shap], axis=-1)  # (n_explain, n_features, n_classes)
    else:
        # Newer shap versions return (n_explain, 1, n_features, n_classes)
        shap_arr = raw_shap[:, 0, :, :]  # (n_explain, n_features, n_classes)
        n_classes = shap_arr.shape[-1]

    if pca is not None:
        # Project from PCA space to wavelength space
        # pca.components_ is (n_components, n_wavelengths)
        n_wl = pca.components_.shape[1]
        shap_wl = np.zeros((n_explain, n_wl, n_classes), dtype=np.float32)
        for c in range(n_classes):
            # shap_arr[:, :, c] is (n_explain, n_pca)
            # components_.T is (n_wavelengths, n_pca)
            shap_wl[:, :, c] = shap_arr[:, :, c] @ pca.components_  # (n_explain, n_wavelengths)
        return shap_wl

    return shap_arr.astype(np.float32)


def compute_ece(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predictions by max confidence and measures the gap between
    average confidence and average accuracy in each bin.

    Args:
        probs: Predicted probability array (n_samples, n_classes); rows sum to 1.
        y_true: True integer class labels (n_samples,).
        n_bins: Number of equal-width bins in [0, 1] (default 15).

    Returns:
        ECE as a float in [0, 1]. Lower is better; 0 = perfect calibration.
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi) if i < n_bins - 1 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = float(accuracies[mask].mean())
        bin_conf = float(confidences[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_classifier(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate a classifier via stratified k-fold cross-validation.

    Args:
        model: Sklearn-compatible classifier (must implement fit/predict).
        X: Feature matrix (n_samples, n_features).
        y: Class label array (n_samples,).
        cv: Number of stratified CV folds (default 5).
        seed: Random seed for StratifiedKFold shuffling.

    Returns:
        Dict with keys: micro_f1, macro_f1, accuracy, confusion_matrix,
        per_class_f1 (dict mapping class label → f1 score).
    """
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix as cm_fn
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(model, X, y, cv=kf, n_jobs=1)

    micro_f1 = float(f1_score(y, y_pred, average="micro"))
    macro_f1 = float(f1_score(y, y_pred, average="macro"))
    accuracy = float(accuracy_score(y, y_pred))
    conf_mat = cm_fn(y, y_pred).tolist()
    classes = sorted(set(y.tolist()))
    per_class_f1 = {
        str(int(c)): float(f1_score(y, y_pred, labels=[c], average="micro"))
        for c in classes
    }

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "confusion_matrix": conf_mat,
        "per_class_f1": per_class_f1,
    }
