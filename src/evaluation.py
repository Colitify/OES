"""Evaluation and visualization module for spectral analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict
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


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    target_names: Optional[List[str]] = None,
    pred_transform=None,
    y_true: Optional[np.ndarray] = None,
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

    Returns:
        Tuple of (metrics_dict, predictions_in_original_space)

    Notes:
        - This function does NOT clip negative predictions by default.
          If you want non-negative concentrations, clip outside before computing metrics.
    """
    y_pred = cross_val_predict(model, X, y, cv=cv)

    # Inverse-transform predictions if requested (e.g. logit → wt%)
    if pred_transform is not None:
        y_pred = pred_transform(y_pred)

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
