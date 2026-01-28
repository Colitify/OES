"""Main script for OES/LIBS Spectral Analysis Pipeline."""

import argparse
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data_loader import SpectralDataset
from src.preprocessing import Preprocessor
from src.features import FeatureExtractor
from src.models.traditional import (
    get_traditional_models,
    train_traditional_model,
    save_model,
)
from src.models.deep_learning import (
    Conv1DRegressor,
    LSTMRegressor,
    TransformerRegressor,
    train_deep_model,
    predict as dl_predict,
)
from src.evaluation import (
    evaluate_model,
    compare_models,
    plot_prediction_comparison,
    generate_report,
)
import torch


def get_git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="OES/LIBS Spectral Analysis Pipeline")
    parser.add_argument("--train", type=str, required=True, help="Path to training data CSV")
    parser.add_argument("--test", type=str, default=None, help="Path to test data CSV")
    parser.add_argument("--n_wavelengths", type=int, default=40002, help="Number of wavelength columns")
    parser.add_argument("--target_cols", type=str, nargs="+", default=None, help="Target column names")
    parser.add_argument(
        "--model",
        type=str,
        default="ridge",
        choices=["pls", "ridge", "lasso", "rf", "all"],
        help="Model to train",
    )
    parser.add_argument("--optimize", action="store_true", help="Optimize hyperparameters")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")

    # Ralph-friendly additions
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--metrics_out", type=str, default=None, help="Write machine-readable metrics json")

    # Preprocessing parameters
    parser.add_argument(
        "--baseline", type=str, default="als", choices=["als", "none"],
        help="Baseline correction method"
    )
    parser.add_argument(
        "--normalize", type=str, default="snv", choices=["snv", "minmax", "l2", "none"],
        help="Normalization method"
    )
    parser.add_argument(
        "--denoise", type=str, default="savgol", choices=["savgol", "none"],
        help="Denoising method"
    )
    parser.add_argument("--baseline_lam", type=float, default=1e6, help="ALS baseline smoothness parameter")
    parser.add_argument("--baseline_p", type=float, default=0.01, help="ALS baseline asymmetry parameter")
    parser.add_argument("--savgol_window", type=int, default=11, help="Savitzky-Golay window length")
    parser.add_argument("--savgol_polyorder", type=int, default=3, help="Savitzky-Golay polynomial order")

    # Feature extraction parameters
    parser.add_argument("--n_components", type=int, default=50, help="Number of PCA components for feature extraction")
    parser.add_argument("--n_components_min", type=int, default=10, help="Min n_components for optimization search")
    parser.add_argument("--n_components_max", type=int, default=100, help="Max n_components for optimization search")
    parser.add_argument("--optimize_n_components", action="store_true", help="Include n_components in hyperparameter optimization")

    args = parser.parse_args()

    # Fix randomness for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directories
    output_dir = Path(args.output_dir)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OES/LIBS Spectral Analysis Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data...")
    train_data = SpectralDataset.from_csv(
        args.train,
        n_wavelengths=args.n_wavelengths,
        target_cols=args.target_cols,
    )
    print(f"  Loaded: {train_data}")

    test_data = None
    if args.test:
        test_data = SpectralDataset.from_csv(
            args.test,
            n_wavelengths=args.n_wavelengths,
            target_cols=args.target_cols,
        )
        print(f"  Test data: {test_data}")

    # 2. Preprocess
    print("\n[2/5] Preprocessing...")
    preprocessor = Preprocessor(
        baseline=args.baseline,
        normalize=args.normalize,
        denoise=args.denoise,
        baseline_lam=args.baseline_lam,
        baseline_p=args.baseline_p,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
    )
    X_train = preprocessor.fit_transform(train_data.spectra)
    print(f"  Preprocessed shape: {X_train.shape}")

    X_test = None
    if test_data is not None:
        X_test = preprocessor.transform(test_data.spectra)

    # 3. Feature extraction (optional PCA)
    print("\n[3/5] Feature extraction...")
    feature_extractor = FeatureExtractor(method="pca", n_components=args.n_components)
    X_train_feat = feature_extractor.fit_transform(X_train, train_data.targets)
    print(f"  Features shape: {X_train_feat.shape}")

    if feature_extractor.explained_variance_ratio_ is not None:
        print(f"  Explained variance: {sum(feature_extractor.explained_variance_ratio_):.2%}")

    X_test_feat = None
    if X_test is not None:
        X_test_feat = feature_extractor.transform(X_test)

    # 4. Train model(s)
    print("\n[4/5] Training model(s)...")
    y_train = train_data.targets
    target_names = train_data.target_names

    models = get_traditional_models(n_targets=train_data.n_targets)

    if args.model == "all":
        comparison_df = compare_models(models, X_train_feat, y_train, cv=args.cv, target_names=target_names)
        print("\nModel Comparison:")
        print(comparison_df.to_string())
        comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)

        if "RMSE" not in comparison_df.columns:
            raise ValueError(f"compare_models output missing RMSE column. Columns={comparison_df.columns}")

        best_model_name = comparison_df.groupby("Model")["RMSE"].mean().idxmin()
        print(f"\nBest model: {best_model_name}")
        best_model = models[best_model_name]
    else:
        best_model_name = args.model
        best_model = models[best_model_name]

    # Hyperparameter optimization
    if args.optimize:
        print(f"\n  Optimizing {best_model_name}...")

        if args.optimize_n_components:
            # Joint optimization of n_components + model hyperparameters
            from src.optimization import optimize_with_pca
            print(f"  Including n_components in search range [{args.n_components_min}, {args.n_components_max}]...")
            best_params, _ = optimize_with_pca(
                X_train, y_train,
                model_name=best_model_name,
                n_components_range=(args.n_components_min, args.n_components_max),
                cv=args.cv,
                n_trials=100,
            )

            # Extract optimal n_components and rebuild feature extractor
            opt_n_components = best_params.pop("n_components", args.n_components)
            print(f"  Optimal n_components: {opt_n_components}")
            feature_extractor = FeatureExtractor(method="pca", n_components=opt_n_components)
            X_train_feat = feature_extractor.fit_transform(X_train, train_data.targets)
            print(f"  Updated features shape: {X_train_feat.shape}")
            if X_test is not None:
                X_test_feat = feature_extractor.transform(X_test)
        else:
            # Standard model-only optimization (fixed n_components)
            from src.optimization import optimize_all_models
            opt_results = optimize_all_models(
                X_train_feat, y_train,
                models=[best_model_name],
                cv=args.cv
            )
            best_params, _ = opt_results[best_model_name]

        from src.models.traditional import get_model_with_params
        best_model = get_model_with_params(best_model_name, best_params, train_data.n_targets)

    # Train final model
    print(f"\n  Training {best_model_name}...")
    best_model = train_traditional_model(best_model, X_train_feat, y_train)

    # Save model
    model_path = output_dir / "models" / f"{best_model_name}_model.joblib"
    save_model(best_model, str(model_path))

    # 5. Evaluate
    print("\n[5/5] Evaluation...")
    metrics, y_pred = evaluate_model(best_model, X_train_feat, y_train, cv=args.cv, target_names=target_names)

    report = generate_report(metrics, best_model_name, str(output_dir / "evaluation_report.txt"))
    print(report)

    # Plot predictions
    fig = plot_prediction_comparison(
        y_train, y_pred, target_names,
        save_path=str(output_dir / "figures" / "predictions.png")
    )
    print(f"  Saved prediction plot to {output_dir / 'figures' / 'predictions.png'}")

    # Write machine-readable metrics (for Ralph/guardrail)
    if args.metrics_out:
        # Primary metric used by guardrail: overall RMSE_mean
        rmse_mean = None
        if isinstance(metrics, dict):
            rmse_mean = metrics.get("_overall", {}).get("RMSE_mean")
            if rmse_mean is None:
                # Fallback: compute mean RMSE across targets if missing
                per_target_rmse = [
                    v.get("RMSE") for k, v in metrics.items()
                    if not str(k).startswith("_") and isinstance(v, dict)
                ]
                if per_target_rmse:
                    rmse_mean = float(np.mean(per_target_rmse))

        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "git_sha": get_git_sha(),
            "model": best_model_name,
            "cv_folds": args.cv,
            "seed": args.seed,
            "primary_metric": {"name": "RMSE_mean", "value": rmse_mean},
            "metrics": metrics,
            "pipeline": {
                "preprocess": {
                    "baseline": preprocessor.baseline,
                    "normalize": preprocessor.normalize,
                    "denoise": preprocessor.denoise,
                    "baseline_lam": preprocessor.baseline_lam,
                    "baseline_p": preprocessor.baseline_p,
                    "savgol_window": preprocessor.savgol_window,
                    "savgol_polyorder": preprocessor.savgol_polyorder,
                },
                "features": {
                    "method": feature_extractor.method,
                    "n_components": feature_extractor.n_components,
                    "selection_method": getattr(feature_extractor, "selection_method", None),
                },
            },
            "paths": {
                "train": str(args.train),
                "test": str(args.test) if args.test else None,
                "output_dir": str(output_dir),
                "model_path": str(model_path),
            },
        }

        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Wrote metrics json to {out_path}")

    # Test set predictions
    if test_data is not None and X_test_feat is not None:
        print("\n[Bonus] Test set predictions...")
        y_test_pred = best_model.predict(X_test_feat)

        # Ensure non-negative predictions for concentrations
        y_test_pred = np.maximum(y_test_pred, 0)

        # Save predictions
        import pandas as pd
        pred_df = pd.DataFrame(y_test_pred, columns=target_names)
        pred_df.to_csv(output_dir / "predictions" / "test_predictions.csv", index=False)
        print(f"  Saved predictions to {output_dir / 'predictions' / 'test_predictions.csv'}")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
