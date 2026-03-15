"""Spatial analysis module for wafer-level etch uniformity (WP5, Objective 2.4).

Provides wafer uniformity computation, spatial interpolation for heatmap
visualisation, and linkage between OES time-series features and spatial
etch metrics.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def compute_wafer_uniformity(
    df: pd.DataFrame,
    metric_col: str,
) -> pd.DataFrame:
    """Compute per-wafer uniformity statistics for a given etch metric.

    Uniformity is defined as (max - min) / (2 * mean) * 100 [%], a standard
    semiconductor metric (semi-standard).

    Args:
        df: Spatial measurement DataFrame with at least 'experiment_key'
            and *metric_col*.
        metric_col: Column name of the etch metric (e.g. 'oxide_etch').

    Returns:
        DataFrame indexed by experiment_key with columns:
        mean, std, min, max, uniformity_pct.
    """
    if metric_col not in df.columns:
        raise ValueError(f"Column '{metric_col}' not found in DataFrame.")
    if "experiment_key" not in df.columns:
        raise ValueError("DataFrame must contain 'experiment_key' column.")

    grouped = df.groupby("experiment_key")[metric_col]
    stats = grouped.agg(["mean", "std", "min", "max"])
    stats["uniformity_pct"] = np.where(
        stats["mean"] != 0,
        (stats["max"] - stats["min"]) / (2 * stats["mean"]) * 100,
        0.0,
    )
    return stats.reset_index()


def interpolate_wafer_map(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_size: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate scattered wafer measurements onto a regular grid via RBF.

    Args:
        x: 1-D array of X coordinates (micrometres).
        y: 1-D array of Y coordinates (micrometres).
        values: 1-D array of measurement values at (x, y).
        grid_size: Number of grid points along each axis.

    Returns:
        Tuple of (grid_x, grid_y, grid_values) where grid_x and grid_y are
        1-D coordinate arrays and grid_values is a (grid_size, grid_size)
        interpolated map.
    """
    from scipy.interpolate import RBFInterpolator

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    grid_x = np.linspace(x.min(), x.max(), grid_size)
    grid_y = np.linspace(y.min(), y.max(), grid_size)
    gx, gy = np.meshgrid(grid_x, grid_y)

    coords = np.column_stack([x, y])
    grid_coords = np.column_stack([gx.ravel(), gy.ravel()])

    rbf = RBFInterpolator(coords, values, kernel="thin_plate_spline")
    grid_values = rbf(grid_coords).reshape(grid_size, grid_size)

    return grid_x, grid_y, grid_values


def link_oes_spatial(
    spatial_df: pd.DataFrame,
    oes_features: pd.DataFrame,
    metric_col: str,
) -> pd.DataFrame:
    """Merge OES time-series features with spatial etch metrics.

    Both DataFrames must share an 'experiment_key' column. The result
    aggregates spatial metrics per experiment and joins them with
    per-experiment OES features.

    Args:
        spatial_df: Spatial measurement DataFrame (from load_wafer_spatial).
        oes_features: DataFrame of OES-derived features with 'experiment_key'.
        metric_col: Spatial metric column to aggregate (e.g. 'oxide_etch').

    Returns:
        Merged DataFrame with OES features + spatial uniformity stats.
    """
    uniformity = compute_wafer_uniformity(spatial_df, metric_col)
    merged = oes_features.merge(uniformity, on="experiment_key", how="inner")
    return merged


def predict_etch_from_oes(
    oes_features: "pd.DataFrame",
    spatial_df: "pd.DataFrame",
    metric_col: str,
    cv: int = 5,
    seed: int = 42,
) -> dict:
    """Predict spatial etch uniformity from OES temporal features.

    Args:
        oes_features: DataFrame with 'experiment_key' + OES feature columns
        spatial_df: Spatial measurement DataFrame
        metric_col: Etch metric column (e.g., 'oxide_etch')
        cv: CV folds
        seed: Random seed

    Returns:
        Dict with: rmse, r2, model, feature_names
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_validate
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    merged = link_oes_spatial(spatial_df, oes_features, metric_col)

    exclude = {"experiment_key", "mean", "std", "min", "max", "uniformity_pct"}
    feat_cols = [c for c in merged.columns if c not in exclude and np.issubdtype(merged[c].dtype, np.number)]
    target_col = "uniformity_pct"

    if len(merged) == 0:
        raise ValueError("No overlapping experiment_keys between OES features and spatial data")

    X = merged[feat_cols].values
    y = merged[target_col].values

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])

    actual_cv = min(cv, len(X))
    cv_results = cross_validate(model, X, y, cv=actual_cv, scoring=["neg_root_mean_squared_error", "r2"])
    rmse = float(-cv_results["test_neg_root_mean_squared_error"].mean())
    r2 = float(cv_results["test_r2"].mean())

    model.fit(X, y)

    return {
        "rmse": rmse,
        "r2": r2,
        "model": model,
        "feature_names": feat_cols,
    }
