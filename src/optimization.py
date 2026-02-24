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
    # Warm-start: first trial uses ML-003 known-best params so TPE explores around the optimum
    study.enqueue_trial({
        "baseline_lam": 416448.0,
        "baseline_p": 0.026,
        "savgol_window_half": 6,   # 2*6+1 = 13
        "savgol_polyorder": 4,
        "n_components": 86,
    })
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
    inverse_transform: Optional[Callable] = None,
    y_true: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], float]:
    """Stage 2: Optimize only model hyperparameters.

    Uses pre-processed spectra to tune model params (PCA applied inside).

    Args:
        X_preprocessed: Preprocessed spectra (after preprocessing, before PCA)
        y: Target matrix (may be transformed, e.g. logit)
        model_name: Model to optimize
        n_components: Fixed PCA n_components from Stage 1
        cv: Number of CV folds
        n_trials: Number of optimization trials
        inverse_transform: Optional callable applied to cross_val_predict output
            before computing the objective RMSE (e.g. logit inverse).
        y_true: Ground-truth in original scale; required when inverse_transform
            is provided.

    Returns:
        Tuple of (best_model_params, best_score)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optimization")

    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_predict as _cvp

    print(f"  [Stage 2] Optimizing {model_name} with n_components={n_components}")
    if inverse_transform is not None:
        print(f"  [Stage 2] Objective computed in original (inverse-transformed) space")

    def _score(model_or_pipeline, X_data, y_data):
        """Compute RMSE, in original space if inverse_transform is provided."""
        if inverse_transform is not None and y_true is not None:
            y_pred = _cvp(model_or_pipeline, X_data, y_data, cv=cv)
            y_pred_orig = inverse_transform(y_pred)
            return float(np.sqrt(np.mean((y_true - y_pred_orig) ** 2)))
        scores = cross_val_score(model_or_pipeline, X_data, y_data, cv=cv,
                                  scoring="neg_root_mean_squared_error")
        return -scores.mean()

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
            return _score(model, X_preprocessed, y)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        pipeline = Pipeline([
            ("pca", PCA(n_components=n_components)),
            ("model", model),
        ])
        return _score(pipeline, X_preprocessed, y)

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


def optimize_preprocessing_and_features(
    X_raw: np.ndarray,
    y: np.ndarray,
    n_components_range: Tuple[int, int] = (10, 100),
    n_wavelengths_range: Tuple[int, int] = (100, 1000),
    cv: int = 5,
    n_trials: int = 30,
    subsample_ratio: float = 0.3,
    fixed_model: str = "ridge",
    fixed_alpha: float = 1.0,
) -> Tuple[Dict[str, Any], float]:
    """Stage 1 with feature selection: Optimize preprocessing, feature method, and feature params.

    Extends optimize_preprocessing_only to include choice between PCA and wavelength selection.
    Optuna will search:
    - Preprocessing: baseline_lam, baseline_p, savgol_window, savgol_polyorder
    - Feature method: "pca" or "wavelength_selection"
    - If PCA: n_components
    - If wavelength_selection: n_wavelengths, selection_method

    Args:
        X_raw: Raw spectra (before any preprocessing)
        y: Target matrix
        n_components_range: Range for PCA n_components search
        n_wavelengths_range: Range for n_wavelengths search (wavelength selection)
        cv: Number of CV folds
        n_trials: Number of optimization trials
        subsample_ratio: Fraction of samples for faster optimization
        fixed_model: Model to use for evaluation (default "ridge")
        fixed_alpha: Fixed regularization strength

    Returns:
        Tuple of (best_params including feature_method, best_score)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optimization")

    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from src.preprocessing import Preprocessor
    from src.features import select_wavelengths

    # Subsample for speed
    n_samples = X_raw.shape[0]
    subsample_size = max(int(n_samples * subsample_ratio), cv + 1)
    np.random.seed(42)
    subsample_idx = np.random.choice(n_samples, size=subsample_size, replace=False)
    X_subsample = X_raw[subsample_idx]
    y_subsample = y[subsample_idx] if y.ndim == 1 else y[subsample_idx, :]

    print(f"  [Stage 1+Features] Using {subsample_size}/{n_samples} samples ({subsample_ratio*100:.0f}%)")

    # Fixed model for evaluation
    if fixed_model == "ridge":
        model = Ridge(alpha=fixed_alpha)
    elif fixed_model == "lasso":
        model = Lasso(alpha=0.1, max_iter=10000)
    else:
        model = Ridge(alpha=1.0)

    def objective(trial):
        # --- Preprocessing parameters ---
        baseline_lam = trial.suggest_float("baseline_lam", 1e5, 1e7, log=True)
        baseline_p = trial.suggest_float("baseline_p", 0.001, 0.05, log=True)
        savgol_window_half = trial.suggest_int("savgol_window_half", 3, 10)  # 7-21
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

        # --- Feature method selection ---
        feature_method = trial.suggest_categorical("feature_method", ["pca", "wavelength_selection"])

        if feature_method == "pca":
            # PCA feature extraction
            max_components = min(n_components_range[1], X_preprocessed.shape[1], X_preprocessed.shape[0] - 1)
            min_components = min(n_components_range[0], max_components)
            n_components = trial.suggest_int("n_components", min_components, max_components)

            pipeline = Pipeline([
                ("pca", PCA(n_components=n_components)),
                ("model", model),
            ])
            scores = cross_val_score(pipeline, X_preprocessed, y_subsample, cv=cv, scoring="neg_root_mean_squared_error")

        else:  # wavelength_selection
            selection_method = trial.suggest_categorical("selection_method", ["correlation", "variance", "f_score"])
            max_wavelengths = min(n_wavelengths_range[1], X_preprocessed.shape[1])
            min_wavelengths = min(n_wavelengths_range[0], max_wavelengths)
            n_wavelengths = trial.suggest_int("n_wavelengths", min_wavelengths, max_wavelengths)

            # Select wavelengths
            selected_indices = select_wavelengths(X_preprocessed, y_subsample, n_wavelengths, selection_method)
            X_selected = X_preprocessed[:, selected_indices]

            # Direct model on selected wavelengths (no PCA)
            scores = cross_val_score(model, X_selected, y_subsample, cv=cv, scoring="neg_root_mean_squared_error")

        return -scores.mean()

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params.copy()
    if "savgol_window_half" in best_params:
        best_params["savgol_window"] = 2 * best_params.pop("savgol_window_half") + 1

    return best_params, study.best_value


def optimize_model_with_features(
    X_preprocessed: np.ndarray,
    y: np.ndarray,
    model_name: str = "ridge",
    feature_method: str = "pca",
    n_components: int = 50,
    n_wavelengths: int = 500,
    selection_method: str = "correlation",
    cv: int = 5,
    n_trials: int = 30,
) -> Tuple[Dict[str, Any], float]:
    """Stage 2 with feature method: Optimize model hyperparameters with given feature config.

    Args:
        X_preprocessed: Preprocessed spectra (after preprocessing, before feature extraction)
        y: Target matrix
        model_name: Model to optimize
        feature_method: "pca" or "wavelength_selection"
        n_components: PCA n_components (if feature_method="pca")
        n_wavelengths: Number of wavelengths (if feature_method="wavelength_selection")
        selection_method: Wavelength selection method
        cv: Number of CV folds
        n_trials: Number of optimization trials

    Returns:
        Tuple of (best_model_params, best_score)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optimization")

    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from src.features import select_wavelengths

    print(f"  [Stage 2] Optimizing {model_name} with feature_method={feature_method}")

    # Prepare features based on method
    if feature_method == "pca":
        print(f"    Using PCA with n_components={n_components}")
        pca = PCA(n_components=n_components)
        X_features = pca.fit_transform(X_preprocessed)
    else:  # wavelength_selection
        print(f"    Using wavelength_selection ({selection_method}) with n_wavelengths={n_wavelengths}")
        selected_indices = select_wavelengths(X_preprocessed, y, n_wavelengths, selection_method)
        X_features = X_preprocessed[:, selected_indices]

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
            pls_n = trial.suggest_int("n_components", 5, min(50, X_features.shape[0] - 1))
            model = PLSRegression(n_components=pls_n)
            scores = cross_val_score(model, X_features, y, cv=cv, scoring="neg_root_mean_squared_error")
            return -scores.mean()
        else:
            raise ValueError(f"Unknown model: {model_name}")

        scores = cross_val_score(model, X_features, y, cv=cv, scoring="neg_root_mean_squared_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value


def optimize_per_target(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "ridge",
    target_names: Optional[list] = None,
    cv: int = 5,
    n_trials: int = 20,
    inverse_transforms: Optional[Dict[str, Callable]] = None,
    y_original: Optional[np.ndarray] = None,
    wavelengths: Optional[np.ndarray] = None,
) -> Tuple[list, Dict[str, Dict[str, Any]], Dict[str, float]]:
    """Optimize model hyperparameters separately for each target.

    Trains an independent model per target, allowing each to have different
    optimal hyperparameters. This can improve overall RMSE when targets have
    different characteristics.

    Args:
        X: Feature matrix
        y: Target matrix (n_samples, n_targets), may be transformed (e.g. logit)
        model_name: Model to optimize ("ridge", "lasso", "pls", "rf")
        target_names: Names of targets for reporting
        cv: Number of CV folds
        inverse_transforms: Optional dict mapping target_name -> inverse_transform callable.
            When provided for a target, the optimization objective is computed in
            original space (inverse_transform applied before RMSE).
        y_original: Original (untransformed) targets; required when inverse_transforms
            is provided.
        n_trials: Number of Optuna trials per target
        wavelengths: Wavelength array (nm). When provided and model_name="xgb", each
            target uses only its own NIST emission line channels for maximum SNR.

    Returns:
        Tuple of (list_of_models, params_per_target, scores_per_target)
        - list_of_models: List of trained models, one per target
        - params_per_target: Dict mapping target name to best params
        - scores_per_target: Dict mapping target name to best RMSE
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optimization")

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_targets = y.shape[1]
    if target_names is None:
        target_names = [f"Target_{i}" for i in range(n_targets)]

    models_list = []
    params_per_target = {}
    scores_per_target = {}
    per_element_indices: Dict[str, Any] = {}

    from sklearn.model_selection import cross_val_predict as _cvp

    for i, target_name in enumerate(target_names):
        print(f"\n  Optimizing for target: {target_name} ({i+1}/{n_targets})")
        y_single = y[:, i]
        inv_fn = (inverse_transforms or {}).get(target_name)
        y_orig_single = y_original[:, i] if (inv_fn is not None and y_original is not None) else None

        # Per-element NIST channel selection for XGBoost
        elem_indices = None
        if model_name == "xgb" and wavelengths is not None:
            from src.features import select_wavelengths_nist, NIST_EMISSION_LINES
            if target_name in NIST_EMISSION_LINES:
                elem_indices = select_wavelengths_nist(wavelengths, [target_name], delta_nm=1.0)
                print(f"    [NIST] {target_name}: {len(elem_indices)} channels selected")
        per_element_indices[target_name] = elem_indices
        X_fit = X[:, elem_indices] if elem_indices is not None else X

        def objective(trial, _y=y_single, _inv=inv_fn, _y_orig=y_orig_single, _X=X_fit):
            if model_name == "ridge":
                alpha = trial.suggest_float("alpha", 1e-2, 1e5, log=True)
                model = Ridge(alpha=alpha)
            elif model_name == "lasso":
                alpha = trial.suggest_float("alpha", 1e-6, 1e2, log=True)
                model = Lasso(alpha=alpha, max_iter=10000)
            elif model_name == "pls":
                n_components = trial.suggest_int("n_components", 5, min(50, _X.shape[0] - 1, _X.shape[1]))
                model = PLSRegression(n_components=n_components)
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
            elif model_name == "xgb":
                from xgboost import XGBRegressor
                from src.models.traditional import _cuda_ok
                xgb_params = {
                    "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
                    "max_depth":        trial.suggest_int("max_depth", 3, 8),
                    "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                    "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                }
                device = "cuda" if _cuda_ok() else "cpu"
                model = XGBRegressor(**xgb_params, tree_method="hist", device=device,
                                     n_jobs=-1, verbosity=0, random_state=42)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            if _inv is not None and _y_orig is not None:
                # Compute RMSE in original space (e.g., after inverse logit)
                y_pred_trans = _cvp(model, _X, _y, cv=cv)
                y_pred_orig = _inv(y_pred_trans)
                return float(np.sqrt(np.mean((_y_orig - y_pred_orig) ** 2)))
            else:
                scores = cross_val_score(model, _X, _y, cv=cv, scoring="neg_root_mean_squared_error")
                return -scores.mean()

        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42 + i))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        # Build final model with best params
        if model_name == "ridge":
            model = Ridge(**best_params)
        elif model_name == "lasso":
            model = Lasso(**best_params, max_iter=10000)
        elif model_name == "pls":
            model = PLSRegression(**best_params)
        elif model_name == "rf":
            model = RandomForestRegressor(**best_params, n_jobs=-1, random_state=42)
        elif model_name == "xgb":
            from xgboost import XGBRegressor
            from src.models.traditional import _cuda_ok
            device = "cuda" if _cuda_ok() else "cpu"
            model = XGBRegressor(**best_params, tree_method="hist", device=device,
                                 n_jobs=-1, verbosity=0, random_state=42)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_fit, y_single)
        models_list.append(model)
        params_per_target[target_name] = best_params
        scores_per_target[target_name] = best_score

        print(f"    Best RMSE: {best_score:.4f}")
        print(f"    Best params: {best_params}")

    return models_list, params_per_target, scores_per_target, per_element_indices


from sklearn.base import BaseEstimator, RegressorMixin, clone


class PerTargetRegressor(BaseEstimator, RegressorMixin):
    """Wrapper that uses separate models for each target.

    Provides a sklearn-compatible interface for multi-output regression
    where each target has its own independently optimized model.

    Inherits from sklearn's BaseEstimator and RegressorMixin for compatibility
    with sklearn's cross_val_predict and other utilities.
    """

    def __init__(self, models: list = None, target_names: Optional[list] = None,
                 per_element_indices: Optional[Dict[str, Any]] = None):
        """Initialize with a list of models, one per target.

        Args:
            models: List of fitted or unfitted models, one per target
            target_names: Names of targets for reporting
            per_element_indices: Optional dict mapping target name to channel index array.
                When provided, each model sees only its assigned channel subset.
        """
        self.models = models if models is not None else []
        self.target_names = target_names
        self.per_element_indices = per_element_indices or {}

    def __sklearn_clone__(self):
        """Custom clone that preserves model hyperparameters."""
        # Clone each model to preserve hyperparameters but reset fit state
        cloned_models = [clone(model) for model in self.models]
        return PerTargetRegressor(models=cloned_models, target_names=self.target_names,
                                  per_element_indices=self.per_element_indices)

    @property
    def n_targets(self):
        """Number of targets (models)."""
        return len(self.models) if self.models else 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit each model to its corresponding target.

        Args:
            X: Feature matrix
            y: Target matrix (n_samples, n_targets)
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_targets = len(self.models)
        if y.shape[1] != n_targets:
            raise ValueError(f"Expected {n_targets} targets, got {y.shape[1]}")

        for i, model in enumerate(self.models):
            name = self.target_names[i] if self.target_names else None
            indices = self.per_element_indices.get(name) if name else None
            X_in = X[:, indices] if indices is not None else X
            model.fit(X_in, y[:, i])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using each model for its target.

        Args:
            X: Feature matrix

        Returns:
            Predictions matrix (n_samples, n_targets)
        """
        predictions = []
        for i, model in enumerate(self.models):
            name = self.target_names[i] if self.target_names else None
            indices = self.per_element_indices.get(name) if name else None
            X_in = X[:, indices] if indices is not None else X
            pred = model.predict(X_in)
            if pred.ndim > 1:
                pred = pred.ravel()
            predictions.append(pred)

        return np.column_stack(predictions)

    def __repr__(self) -> str:
        return f"PerTargetRegressor(n_targets={self.n_targets})"


def optimize_cnn(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    n_trials: int = 20,
    epochs: int = 30,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], float]:
    """Optimize CNN hyperparameters using Optuna.

    Args:
        X: Feature matrix
        y: Target matrix
        cv: Number of CV folds
        n_trials: Number of optimization trials
        epochs: Training epochs per trial
        verbose: Whether to print training progress

    Returns:
        Tuple of (best_params, best_score)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optimization")

    from src.models.deep_learning import CNNRegressor

    # Calculate max layers based on input dimension
    # Each MaxPool1d(4) reduces dimension by 4x
    # We need output_size >= 1, so max_layers = floor(log4(input_dim))
    import math
    input_dim = X.shape[1]
    max_layers = max(1, int(math.log(input_dim, 4)))  # Ensure at least 1 layer

    def objective(trial):
        # Limit n_layers based on input dimension
        n_layers = trial.suggest_int("n_layers", 1, min(max_layers, 3))
        channels = []
        for i in range(n_layers):
            ch = trial.suggest_categorical(f"channels_{i}", [16, 32, 64, 128])
            channels.append(ch)

        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7, 9])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        model = CNNRegressor(
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=1e-4,
            patience=10,
            verbose=verbose,
        )

        try:
            # Use cross_val_score for evaluation with error handling
            scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
            return -scores.mean()
        except Exception:
            # Return a large penalty if model fails (e.g., invalid architecture)
            return 1e10

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Extract best params and rebuild channels list
    best_params = study.best_params.copy()
    n_layers = best_params.pop("n_layers")
    channels = [best_params.pop(f"channels_{i}") for i in range(n_layers)]
    best_params["channels"] = channels

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
            elif model_name == "cnn":
                params, score = optimize_cnn(X, y, cv=cv, n_trials=n_trials)
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
