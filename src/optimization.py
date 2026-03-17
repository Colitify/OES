"""Hyperparameter optimization module for OES spectral models."""

import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any, Optional, Tuple
import warnings

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def _run_optuna_study(objective, n_trials: int, seed: int = 42) -> Tuple[Dict[str, Any], float]:
    """Run an Optuna study with standard boilerplate.

    Args:
        objective: Optuna objective function (trial -> float)
        n_trials: Number of optimization trials
        seed: Random seed for TPE sampler

    Returns:
        Tuple of (best_params, best_value)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optimization. Install with: pip install optuna")
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value


def optimize_pls(
    X: np.ndarray,
    y: np.ndarray,
    n_components_range: Tuple[int, int] = (5, 50),
    cv: int = 5,
    n_trials: int = 50
) -> Tuple[Dict[str, Any], float]:
    """Optimize PLS using Optuna."""
    def objective(trial):
        n_components = trial.suggest_int("n_components", *n_components_range)
        model = PLSRegression(n_components=n_components)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        return -scores.mean()

    return _run_optuna_study(objective, n_trials)


def optimize_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha_range: Tuple[float, float] = (1e-4, 1e4),
    cv: int = 5,
    n_trials: int = 100
) -> Tuple[Dict[str, Any], float]:
    """Optimize Ridge using Optuna."""
    def objective(trial):
        alpha = trial.suggest_float("alpha", *alpha_range, log=True)
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        return -scores.mean()

    return _run_optuna_study(objective, n_trials)


def optimize_lasso(
    X: np.ndarray,
    y: np.ndarray,
    alpha_range: Tuple[float, float] = (1e-6, 1e2),
    cv: int = 5,
    n_trials: int = 100
) -> Tuple[Dict[str, Any], float]:
    """Optimize Lasso using Optuna."""
    def objective(trial):
        alpha = trial.suggest_float("alpha", *alpha_range, log=True)
        model = Lasso(alpha=alpha, max_iter=10000)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        return -scores.mean()

    return _run_optuna_study(objective, n_trials)


def optimize_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    n_trials: int = 100
) -> Tuple[Dict[str, Any], float]:
    """Optimize ElasticNet using Optuna."""
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-6, 1e2, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        return -scores.mean()

    return _run_optuna_study(objective, n_trials)


def optimize_rf(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    n_trials: int = 50
) -> Tuple[Dict[str, Any], float]:
    """Optimize Random Forest using Optuna."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "n_jobs": -1,
            "random_state": 42
        }
        model = RandomForestRegressor(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        return -scores.mean()

    return _run_optuna_study(objective, n_trials)


def grid_search_model(
    model,
    param_grid: Dict[str, list],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5
) -> Tuple[Any, Dict[str, Any], float]:
    """Grid search for hyperparameter optimization."""
    gs = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_, -gs.best_score_


# Predefined parameter grids for grid search
PARAM_GRIDS = {
    "pls": {"n_components": [5, 10, 15, 20, 30, 50]},
    "ridge": {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
    "lasso": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1]},
    "elastic_net": {
        "alpha": [0.001, 0.01, 0.1, 1],
        "l1_ratio": [0.2, 0.5, 0.8]
    },
    "rf": {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10]
    }
}


def optimize_with_pca(
    X_raw: np.ndarray,
    y: np.ndarray,
    model_name: str = "ridge",
    n_components_range: Tuple[int, int] = (10, 100),
    cv: int = 5,
    n_trials: int = 100,
) -> Tuple[Dict[str, Any], float]:
    """Jointly optimize PCA n_components and model hyperparameters."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    def objective(trial):
        max_components = min(n_components_range[1], X_raw.shape[1], X_raw.shape[0] - 1)
        min_components = min(n_components_range[0], max_components)
        n_components = trial.suggest_int("n_components", min_components, max_components)

        if model_name == "ridge":
            alpha = trial.suggest_float("alpha", 1e-4, 1e4, log=True)
            model = Ridge(alpha=alpha)
        elif model_name == "lasso":
            alpha = trial.suggest_float("alpha", 1e-6, 1e2, log=True)
            model = Lasso(alpha=alpha, max_iter=10000)
        elif model_name == "pls":
            pls_components = trial.suggest_int("pls_n_components", 5, min(50, X_raw.shape[0] - 1))
            model = PLSRegression(n_components=pls_components)
            scores = cross_val_score(model, X_raw, y, cv=cv, scoring="neg_root_mean_squared_error")
            return -scores.mean()
        elif model_name == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "n_jobs": -1,
                "random_state": 42,
            }
            model = RandomForestRegressor(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        pipeline = Pipeline([
            ("pca", PCA(n_components=n_components)),
            ("model", model),
        ])

        scores = cross_val_score(pipeline, X_raw, y, cv=cv, scoring="neg_root_mean_squared_error")
        return -scores.mean()

    return _run_optuna_study(objective, n_trials)
