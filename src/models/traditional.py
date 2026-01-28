"""Traditional machine learning models for spectral regression."""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional, Literal, Tuple
import joblib
from pathlib import Path


def get_traditional_models(
    n_targets: int = 1,
    include_slow: bool = False
) -> Dict[str, Any]:
    """Get dictionary of traditional ML models.

    Args:
        n_targets: Number of target variables
        include_slow: Whether to include slower models (SVR, GBR)

    Returns:
        Dictionary of {model_name: model}
    """
    models = {
        "pls": PLSRegression(n_components=20),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.01),
        "elastic_net": ElasticNet(alpha=0.01, l1_ratio=0.5),
        "rf": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
    }

    if include_slow:
        # SVR requires MultiOutputRegressor for multi-target
        models["svr"] = MultiOutputRegressor(
            SVR(kernel="rbf", C=100, gamma="scale")
        ) if n_targets > 1 else SVR(kernel="rbf", C=100, gamma="scale")

        models["gbr"] = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, random_state=42)
        ) if n_targets > 1 else GradientBoostingRegressor(n_estimators=100, random_state=42)

    return models


def get_model_with_params(
    model_name: Literal["pls", "ridge", "lasso", "elastic_net", "rf", "svr", "gbr"],
    params: Optional[Dict[str, Any]] = None,
    n_targets: int = 1
) -> Any:
    """Get a specific model with custom parameters.

    Args:
        model_name: Name of the model
        params: Custom parameters
        n_targets: Number of targets

    Returns:
        Configured model
    """
    params = params or {}

    if model_name == "pls":
        return PLSRegression(**{**{"n_components": 20}, **params})

    elif model_name == "ridge":
        return Ridge(**{**{"alpha": 1.0}, **params})

    elif model_name == "lasso":
        return Lasso(**{**{"alpha": 0.01}, **params})

    elif model_name == "elastic_net":
        return ElasticNet(**{**{"alpha": 0.01, "l1_ratio": 0.5}, **params})

    elif model_name == "rf":
        return RandomForestRegressor(**{
            **{"n_estimators": 100, "n_jobs": -1, "random_state": 42},
            **params
        })

    elif model_name == "svr":
        base_model = SVR(**{**{"kernel": "rbf", "C": 100, "gamma": "scale"}, **params})
        return MultiOutputRegressor(base_model) if n_targets > 1 else base_model

    elif model_name == "gbr":
        base_model = GradientBoostingRegressor(**{
            **{"n_estimators": 100, "random_state": 42},
            **params
        })
        return MultiOutputRegressor(base_model) if n_targets > 1 else base_model

    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_pipeline(
    model,
    scale: bool = True
) -> Pipeline:
    """Create sklearn pipeline with optional scaling.

    Args:
        model: ML model
        scale: Whether to add StandardScaler

    Returns:
        Pipeline
    """
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def train_traditional_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
) -> Any:
    """Train a traditional ML model.

    Args:
        model: ML model to train
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)

    Returns:
        Trained model
    """
    model.fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        from sklearn.metrics import r2_score, mean_squared_error
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f"Validation R²: {r2:.4f}, RMSE: {rmse:.4f}")

    return model


def save_model(model, filepath: str):
    """Save trained model to file.

    Args:
        model: Trained model
        filepath: Path to save
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str):
    """Load trained model from file.

    Args:
        filepath: Path to model file

    Returns:
        Loaded model
    """
    return joblib.load(filepath)


def get_ensemble_model(
    n_targets: int = 1,
    base_models: Optional[list] = None,
    ensemble_method: Literal["stacking", "voting"] = "stacking",
    final_estimator_alpha: float = 1.0,
) -> Any:
    """Create an ensemble model combining multiple base models.

    Args:
        n_targets: Number of target variables
        base_models: List of model names to include (default: ["pls", "ridge", "rf"])
        ensemble_method: "stacking" (meta-learner) or "voting" (average)
        final_estimator_alpha: Ridge alpha for stacking final estimator

    Returns:
        Ensemble model (StackingRegressor or VotingRegressor)
    """
    from sklearn.ensemble import StackingRegressor, VotingRegressor

    if base_models is None:
        base_models = ["pls", "ridge", "rf"]

    # Build base estimators list
    estimators = []
    for name in base_models:
        if name == "pls":
            estimators.append(("pls", PLSRegression(n_components=20)))
        elif name == "ridge":
            estimators.append(("ridge", Ridge(alpha=1000.0)))
        elif name == "lasso":
            estimators.append(("lasso", Lasso(alpha=0.01, max_iter=10000)))
        elif name == "elastic_net":
            estimators.append(("elastic_net", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)))
        elif name == "rf":
            estimators.append(("rf", RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)))

    if not estimators:
        raise ValueError("No valid base models specified for ensemble")

    if ensemble_method == "stacking":
        # StackingRegressor with Ridge as final estimator
        # For multi-output, use MultiOutputRegressor wrapper
        from sklearn.multioutput import MultiOutputRegressor

        base_ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=final_estimator_alpha),
            cv=5,
            n_jobs=-1,
        )

        if n_targets > 1:
            return MultiOutputRegressor(base_ensemble)
        return base_ensemble

    else:  # voting
        # VotingRegressor averages predictions
        from sklearn.multioutput import MultiOutputRegressor

        base_ensemble = VotingRegressor(
            estimators=estimators,
            n_jobs=-1,
        )

        if n_targets > 1:
            return MultiOutputRegressor(base_ensemble)
        return base_ensemble


def get_optimized_ensemble_model(
    X: np.ndarray,
    y: np.ndarray,
    n_targets: int = 1,
    base_models: Optional[list] = None,
    ensemble_method: Literal["stacking", "voting"] = "stacking",
    cv: int = 5,
    n_trials: int = 20,
) -> Tuple[Any, Dict[str, Any], float]:
    """Create an ensemble with optimized base model hyperparameters.

    First optimizes each base model individually, then combines them.

    Args:
        X: Feature matrix (already preprocessed and feature-extracted)
        y: Target matrix
        n_targets: Number of target variables
        base_models: List of model names to include
        ensemble_method: "stacking" or "voting"
        cv: Cross-validation folds for optimization
        n_trials: Optuna trials per base model

    Returns:
        Tuple of (ensemble_model, best_params_dict, best_ensemble_score)
    """
    from sklearn.ensemble import StackingRegressor, VotingRegressor
    from sklearn.model_selection import cross_val_score
    import warnings

    if base_models is None:
        base_models = ["pls", "ridge", "rf"]

    # Import optimization functions
    try:
        from src.optimization import optimize_pls, optimize_ridge, optimize_rf
        OPTUNA_AVAILABLE = True
    except ImportError:
        OPTUNA_AVAILABLE = False

    estimators = []
    all_params = {}

    for name in base_models:
        print(f"    Optimizing base model: {name}...")

        if OPTUNA_AVAILABLE:
            if name == "pls":
                params, _ = optimize_pls(X, y, cv=cv, n_trials=n_trials)
                model = PLSRegression(**params)
                all_params["pls"] = params
            elif name == "ridge":
                params, _ = optimize_ridge(X, y, cv=cv, n_trials=n_trials)
                model = Ridge(**params)
                all_params["ridge"] = params
            elif name == "rf":
                params, _ = optimize_rf(X, y, cv=cv, n_trials=n_trials)
                model = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
                all_params["rf"] = params
            elif name == "lasso":
                from src.optimization import optimize_lasso
                params, _ = optimize_lasso(X, y, cv=cv, n_trials=n_trials)
                model = Lasso(**params, max_iter=10000)
                all_params["lasso"] = params
            else:
                print(f"      Skipping unknown model: {name}")
                continue
        else:
            # Fall back to defaults if Optuna not available
            if name == "pls":
                model = PLSRegression(n_components=20)
            elif name == "ridge":
                model = Ridge(alpha=1000.0)
            elif name == "rf":
                model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
            elif name == "lasso":
                model = Lasso(alpha=0.01, max_iter=10000)
            else:
                continue

        estimators.append((name, model))

    if not estimators:
        raise ValueError("No valid base models for ensemble")

    # Create ensemble
    if ensemble_method == "stacking":
        from sklearn.multioutput import MultiOutputRegressor

        base_ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            n_jobs=-1,
        )

        if n_targets > 1:
            ensemble = MultiOutputRegressor(base_ensemble)
        else:
            ensemble = base_ensemble

    else:  # voting
        from sklearn.multioutput import MultiOutputRegressor

        base_ensemble = VotingRegressor(
            estimators=estimators,
            n_jobs=-1,
        )

        if n_targets > 1:
            ensemble = MultiOutputRegressor(base_ensemble)
        else:
            ensemble = base_ensemble

    # Evaluate ensemble
    print(f"    Evaluating {ensemble_method} ensemble...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(ensemble, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    ensemble_score = -scores.mean()

    all_params["ensemble_method"] = ensemble_method
    all_params["ensemble_rmse"] = ensemble_score

    return ensemble, all_params, ensemble_score
