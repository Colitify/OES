"""Hyperparameter optimization module."""

import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any, Optional, Tuple, Callable
import warnings

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def optimize_pls(
    X: np.ndarray,
    y: np.ndarray,
    n_components_range: Tuple[int, int] = (5, 50),
    cv: int = 5,
    n_trials: int = 50
) -> Tuple[Dict[str, Any], float]:
    """Optimize PLS using Optuna.

    Args:
        X: Feature matrix
        y: Target matrix
        n_components_range: Range for n_components
        cv: Number of CV folds
        n_trials: Number of optimization trials

    Returns:
        Tuple of (best_params, best_score)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optimization. Install with: pip install optuna")

    def objective(trial):
        n_components = trial.suggest_int("n_components", *n_components_range)
        model = PLSRegression(n_components=n_components)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value


def optimize_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha_range: Tuple[float, float] = (1e-4, 1e4),
    cv: int = 5,
    n_trials: int = 100
) -> Tuple[Dict[str, Any], float]:
    """Optimize Ridge using Optuna.

    Args:
        X: Feature matrix
        y: Target matrix
        alpha_range: Range for alpha (log scale)
        cv: Number of CV folds
        n_trials: Number of optimization trials

    Returns:
        Tuple of (best_params, best_score)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optimization")

    def objective(trial):
        alpha = trial.suggest_float("alpha", *alpha_range, log=True)
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value


def optimize_lasso(
    X: np.ndarray,
    y: np.ndarray,
    alpha_range: Tuple[float, float] = (1e-6, 1e2),
    cv: int = 5,
    n_trials: int = 100
) -> Tuple[Dict[str, Any], float]:
    """Optimize Lasso using Optuna."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optimization")

    def objective(trial):
        alpha = trial.suggest_float("alpha", *alpha_range, log=True)
        model = Lasso(alpha=alpha, max_iter=10000)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value


def optimize_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    n_trials: int = 100
) -> Tuple[Dict[str, Any], float]:
    """Optimize ElasticNet using Optuna."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optimization")

    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-6, 1e2, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value


def optimize_rf(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    n_trials: int = 50
) -> Tuple[Dict[str, Any], float]:
    """Optimize Random Forest using Optuna."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optimization")

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

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value


def grid_search_model(
    model,
    param_grid: Dict[str, list],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5
) -> Tuple[Any, Dict[str, Any], float]:
    """Grid search for hyperparameter optimization.

    Args:
        model: Base model
        param_grid: Parameter grid
        X: Feature matrix
        y: Target matrix
        cv: Number of CV folds

    Returns:
        Tuple of (best_estimator, best_params, best_score)
    """
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
    """Jointly optimize PCA n_components and model hyperparameters.

    Args:
        X_raw: Raw/preprocessed spectra (before PCA)
        y: Target matrix
        model_name: Model to optimize ("ridge", "lasso", "pls", "rf")
        n_components_range: Range for PCA n_components search
        cv: Number of CV folds
        n_trials: Number of optimization trials

    Returns:
        Tuple of (best_params including n_components, best_score)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optimization")

    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    def objective(trial):
        # Suggest n_components within range, but cap at max features
        max_components = min(n_components_range[1], X_raw.shape[1], X_raw.shape[0] - 1)
        min_components = min(n_components_range[0], max_components)
        n_components = trial.suggest_int("n_components", min_components, max_components)

        # Model-specific hyperparameters
        if model_name == "ridge":
            alpha = trial.suggest_float("alpha", 1e-4, 1e4, log=True)
            model = Ridge(alpha=alpha)
        elif model_name == "lasso":
            alpha = trial.suggest_float("alpha", 1e-6, 1e2, log=True)
            model = Lasso(alpha=alpha, max_iter=10000)
        elif model_name == "pls":
            # For PLS, n_components applies to the model, not PCA
            # We skip PCA and use PLS directly
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

        # Create pipeline with PCA + model
        pipeline = Pipeline([
            ("pca", PCA(n_components=n_components)),
            ("model", model),
        ])

        scores = cross_val_score(pipeline, X_raw, y, cv=cv, scoring="neg_root_mean_squared_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value


def optimize_preprocessing_only(
    X_raw: np.ndarray,
    y: np.ndarray,
    n_components_range: Tuple[int, int] = (10, 100),
    cv: int = 5,
    n_trials: int = 30,
    subsample_ratio: float = 0.3,
    fixed_model: str = "ridge",
    fixed_alpha: float = 1.0,
) -> Tuple[Dict[str, Any], float]:
    """Stage 1: Optimize only preprocessing and PCA parameters.

    Uses a fixed simple model (Ridge with default alpha) to evaluate
    preprocessing configurations. This separates preprocessing optimization
    from model tuning.

    Args:
        X_raw: Raw spectra (before any preprocessing)
        y: Target matrix
        n_components_range: Range for PCA n_components search
        cv: Number of CV folds
        n_trials: Number of optimization trials
        subsample_ratio: Fraction of samples for faster optimization
        fixed_model: Model to use for evaluation (default "ridge")
        fixed_alpha: Fixed regularization strength

    Returns:
        Tuple of (best_preprocessing_params, best_score)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optimization")

    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from src.preprocessing import Preprocessor

    # Subsample for speed
    n_samples = X_raw.shape[0]
    subsample_size = max(int(n_samples * subsample_ratio), cv + 1)
    np.random.seed(42)
    subsample_idx = np.random.choice(n_samples, size=subsample_size, replace=False)
    X_subsample = X_raw[subsample_idx]
    y_subsample = y[subsample_idx] if y.ndim == 1 else y[subsample_idx, :]

    print(f"  [Stage 1] Using {subsample_size}/{n_samples} samples ({subsample_ratio*100:.0f}%)")

    # Fixed model for evaluation
    if fixed_model == "ridge":
        model = Ridge(alpha=fixed_alpha)
    elif fixed_model == "lasso":
        model = Lasso(alpha=0.1, max_iter=10000)
    else:
        model = Ridge(alpha=1.0)

    def objective(trial):
        # --- Preprocessing parameters only ---
        baseline_lam = trial.suggest_float("baseline_lam", 1e5, 1e7, log=True)  # Narrowed range
        baseline_p = trial.suggest_float("baseline_p", 0.001, 0.05, log=True)   # Narrowed range
        savgol_window_half = trial.suggest_int("savgol_window_half", 3, 10)     # 7-21
        savgol_window = 2 * savgol_window_half + 1
        savgol_polyorder = trial.suggest_int("savgol_polyorder", 2, 4)

        if savgol_polyorder >= savgol_window:
            savgol_polyorder = savgol_window - 1

        preprocessor = Preprocessor(
            baseline="als",
            normalize="snv",
            denoise="savgol",
            baseline_lam=baseline_lam,
            baseline_p=baseline_p,
            savgol_window=savgol_window,
            savgol_polyorder=savgol_polyorder,
        )
        try:
            X_preprocessed = preprocessor.fit_transform(X_subsample)
        except Exception:
            return 1e10

        # --- PCA n_components ---
        max_components = min(n_components_range[1], X_preprocessed.shape[1], X_preprocessed.shape[0] - 1)
        min_components = min(n_components_range[0], max_components)
        n_components = trial.suggest_int("n_components", min_components, max_components)

        # Fixed model pipeline
        pipeline = Pipeline([
            ("pca", PCA(n_components=n_components)),
            ("model", model),
        ])

        scores = cross_val_score(pipeline, X_preprocessed, y_subsample, cv=cv, scoring="neg_root_mean_squared_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params.copy()
    if "savgol_window_half" in best_params:
        best_params["savgol_window"] = 2 * best_params.pop("savgol_window_half") + 1

    return best_params, study.best_value


def optimize_model_only(
    X_preprocessed: np.ndarray,
    y: np.ndarray,
    model_name: str = "ridge",
    n_components: int = 50,
    cv: int = 5,
    n_trials: int = 30,
) -> Tuple[Dict[str, Any], float]:
    """Stage 2: Optimize only model hyperparameters.

    Uses pre-processed and PCA-transformed features to tune model params.

    Args:
        X_preprocessed: Preprocessed spectra (after preprocessing, before PCA)
        y: Target matrix
        model_name: Model to optimize
        n_components: Fixed PCA n_components from Stage 1
        cv: Number of CV folds
        n_trials: Number of optimization trials

    Returns:
        Tuple of (best_model_params, best_score)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optimization")

    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    print(f"  [Stage 2] Optimizing {model_name} with n_components={n_components}")

    def objective(trial):
        if model_name == "ridge":
            alpha = trial.suggest_float("alpha", 1e-4, 1e4, log=True)
            model = Ridge(alpha=alpha)
        elif model_name == "lasso":
            alpha = trial.suggest_float("alpha", 1e-6, 1e2, log=True)
            model = Lasso(alpha=alpha, max_iter=10000)
        elif model_name == "elastic_net":
            alpha = trial.suggest_float("alpha", 1e-6, 1e2, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
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
        elif model_name == "pls":
            pls_n = trial.suggest_int("n_components", 5, min(50, X_preprocessed.shape[0] - 1))
            model = PLSRegression(n_components=pls_n)
            scores = cross_val_score(model, X_preprocessed, y, cv=cv, scoring="neg_root_mean_squared_error")
            return -scores.mean()
        else:
            raise ValueError(f"Unknown model: {model_name}")

        pipeline = Pipeline([
            ("pca", PCA(n_components=n_components)),
            ("model", model),
        ])

        scores = cross_val_score(pipeline, X_preprocessed, y, cv=cv, scoring="neg_root_mean_squared_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value


def optimize_full_pipeline(
    X_raw: np.ndarray,
    y: np.ndarray,
    model_name: str = "ridge",
    n_components_range: Tuple[int, int] = (10, 100),
    cv: int = 5,
    n_trials: int = 100,
    subsample_ratio: float = 0.3,
) -> Tuple[Dict[str, Any], float]:
    """Jointly optimize preprocessing, PCA, and model hyperparameters.

    This function searches over:
    - Preprocessing: baseline_lam, baseline_p, savgol_window, savgol_polyorder
    - Feature extraction: PCA n_components
    - Model hyperparameters: model-specific params

    Uses subsampling during optimization to speed up preprocessing (the bottleneck).

    Args:
        X_raw: Raw spectra (before any preprocessing)
        y: Target matrix
        model_name: Model to optimize ("ridge", "lasso", "pls", "rf")
        n_components_range: Range for PCA n_components search
        cv: Number of CV folds
        n_trials: Number of optimization trials
        subsample_ratio: Fraction of samples to use during optimization (0.0-1.0)
                        Lower = faster but less accurate. Default 0.3 (30%)

    Returns:
        Tuple of (best_params including all pipeline params, best_score)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optimization")

    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from src.preprocessing import Preprocessor

    # Create subsampled dataset for faster optimization
    n_samples = X_raw.shape[0]
    subsample_size = max(int(n_samples * subsample_ratio), cv + 1)  # Ensure enough for CV
    np.random.seed(42)
    subsample_idx = np.random.choice(n_samples, size=subsample_size, replace=False)
    X_subsample = X_raw[subsample_idx]
    y_subsample = y[subsample_idx] if y.ndim == 1 else y[subsample_idx, :]

    print(f"  Using {subsample_size}/{n_samples} samples ({subsample_ratio*100:.0f}%) for optimization")

    def objective(trial):
        # --- Preprocessing parameters ---
        baseline_lam = trial.suggest_float("baseline_lam", 1e4, 1e8, log=True)
        baseline_p = trial.suggest_float("baseline_p", 0.001, 0.1, log=True)
        # savgol_window must be odd
        savgol_window_half = trial.suggest_int("savgol_window_half", 2, 15)
        savgol_window = 2 * savgol_window_half + 1  # ensures odd: 5, 7, ..., 31
        savgol_polyorder = trial.suggest_int("savgol_polyorder", 2, 5)

        # Ensure polyorder < window
        if savgol_polyorder >= savgol_window:
            savgol_polyorder = savgol_window - 1

        # Apply preprocessing on SUBSAMPLED data
        preprocessor = Preprocessor(
            baseline="als",
            normalize="snv",
            denoise="savgol",
            baseline_lam=baseline_lam,
            baseline_p=baseline_p,
            savgol_window=savgol_window,
            savgol_polyorder=savgol_polyorder,
        )
        try:
            X_preprocessed = preprocessor.fit_transform(X_subsample)
        except Exception:
            # If preprocessing fails, return a large penalty
            return 1e10

        # --- PCA n_components ---
        max_components = min(n_components_range[1], X_preprocessed.shape[1], X_preprocessed.shape[0] - 1)
        min_components = min(n_components_range[0], max_components)
        n_components = trial.suggest_int("n_components", min_components, max_components)

        # --- Model-specific hyperparameters ---
        if model_name == "ridge":
            alpha = trial.suggest_float("alpha", 1e-4, 1e4, log=True)
            model = Ridge(alpha=alpha)
        elif model_name == "lasso":
            alpha = trial.suggest_float("alpha", 1e-6, 1e2, log=True)
            model = Lasso(alpha=alpha, max_iter=10000)
        elif model_name == "pls":
            # PLS has its own latent variable decomposition, skip PCA
            pls_components = trial.suggest_int("pls_n_components", 5, min(50, X_preprocessed.shape[0] - 1))
            pls_model = PLSRegression(n_components=pls_components)
            scores = cross_val_score(pls_model, X_preprocessed, y_subsample, cv=cv, scoring="neg_root_mean_squared_error")
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

        # Create pipeline with PCA + model
        pipeline = Pipeline([
            ("pca", PCA(n_components=n_components)),
            ("model", model),
        ])

        scores = cross_val_score(pipeline, X_preprocessed, y_subsample, cv=cv, scoring="neg_root_mean_squared_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Post-process best params to convert savgol_window_half to actual window
    best_params = study.best_params.copy()
    if "savgol_window_half" in best_params:
        best_params["savgol_window"] = 2 * best_params.pop("savgol_window_half") + 1

    return best_params, study.best_value


def optimize_all_models(
    X: np.ndarray,
    y: np.ndarray,
    models: Optional[list] = None,
    method: str = "optuna",
    cv: int = 5,
    n_trials: int = 50
) -> Dict[str, Tuple[Dict[str, Any], float]]:
    """Optimize multiple models.

    Args:
        X: Feature matrix
        y: Target matrix
        models: List of model names to optimize
        method: "optuna" or "grid"
        cv: Number of CV folds
        n_trials: Number of trials (Optuna only)

    Returns:
        Dictionary of {model_name: (best_params, best_score)}
    """
    if models is None:
        models = ["pls", "ridge", "lasso", "rf"]

    results = {}

    for model_name in models:
        print(f"\nOptimizing {model_name}...")

        if method == "optuna":
            if model_name == "pls":
                params, score = optimize_pls(X, y, cv=cv, n_trials=n_trials)
            elif model_name == "ridge":
                params, score = optimize_ridge(X, y, cv=cv, n_trials=n_trials)
            elif model_name == "lasso":
                params, score = optimize_lasso(X, y, cv=cv, n_trials=n_trials)
            elif model_name == "elastic_net":
                params, score = optimize_elastic_net(X, y, cv=cv, n_trials=n_trials)
            elif model_name == "rf":
                params, score = optimize_rf(X, y, cv=cv, n_trials=n_trials)
            else:
                print(f"  Skipping unknown model: {model_name}")
                continue

        elif method == "grid":
            from .models.traditional import get_model_with_params
            base_model = get_model_with_params(model_name)
            _, params, score = grid_search_model(
                base_model, PARAM_GRIDS.get(model_name, {}), X, y, cv=cv
            )

        results[model_name] = (params, score)
        print(f"  Best params: {params}")
        print(f"  Best RMSE: {score:.4f}")

    return results
