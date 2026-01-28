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
    get_ensemble_model,
    get_optimized_ensemble_model,
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

    # Feature selection parameters (ML-004)
    parser.add_argument(
        "--feature_method", type=str, default="pca",
        choices=["pca", "wavelength_selection"],
        help="Feature extraction method (pca or wavelength_selection)"
    )
    parser.add_argument(
        "--selection_method", type=str, default="correlation",
        choices=["correlation", "variance", "f_score"],
        help="Wavelength selection method (used when --feature_method=wavelength_selection)"
    )
    parser.add_argument("--n_selected_wavelengths", type=int, default=500, help="Number of wavelengths to select (used when --feature_method=wavelength_selection)")
    parser.add_argument("--n_selected_min", type=int, default=100, help="Min n_selected_wavelengths for optimization search")
    parser.add_argument("--n_selected_max", type=int, default=1000, help="Max n_selected_wavelengths for optimization search")
    parser.add_argument("--optimize_features", action="store_true", help="Include feature method (PCA vs wavelength selection) in optimization")
    parser.add_argument("--optimize_preprocess", action="store_true", help="Include preprocessing params in optimization (baseline_lam, baseline_p, savgol_window, savgol_polyorder)")
    parser.add_argument("--two_stage", action="store_true", help="Two-stage optimization: Stage1=preprocessing+PCA, Stage2=model params (recommended)")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials for optimization (default 20)")
    parser.add_argument("--subsample_ratio", type=float, default=0.3, help="Subsample ratio for preprocessing optimization (0.0-1.0). Lower=faster. Default 0.3")

    # Ensemble parameters (ML-005)
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble model combining PLS + Ridge + RF")
    parser.add_argument(
        "--ensemble_method", type=str, default="stacking",
        choices=["stacking", "voting"],
        help="Ensemble method: stacking (meta-learner) or voting (average)"
    )
    parser.add_argument(
        "--ensemble_models", type=str, nargs="+", default=["pls", "ridge", "rf"],
        help="Base models for ensemble (default: pls ridge rf)"
    )

    # Per-target optimization (ML-006)
    parser.add_argument("--per_target", action="store_true", help="Train separate optimized model per target element")

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

    # 3. Feature extraction (PCA or wavelength selection)
    print("\n[3/5] Feature extraction...")
    if args.feature_method == "wavelength_selection":
        feature_extractor = FeatureExtractor(
            method="wavelength_selection",
            n_components=args.n_selected_wavelengths,
            selection_method=args.selection_method,
        )
    else:
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

    # Per-target optimization mode (ML-006)
    if args.per_target:
        print(f"\n  === Per-Target Optimization Mode ===")
        print(f"  Model: {args.model}")
        print(f"  Training separate optimized model per target element")

        from src.optimization import optimize_per_target, PerTargetRegressor

        per_target_models, per_target_params, per_target_scores = optimize_per_target(
            X_train_feat, y_train,
            model_name=args.model,
            target_names=target_names,
            cv=args.cv,
            n_trials=args.n_trials,
        )

        best_model = PerTargetRegressor(per_target_models, target_names=target_names)
        best_model_name = f"{args.model}_per_target"

        # Report overall score
        overall_rmse = np.mean(list(per_target_scores.values()))
        print(f"\n  Per-Target Optimization Complete:")
        print(f"    Overall RMSE (mean of per-target): {overall_rmse:.4f}")
        for tname, score in per_target_scores.items():
            print(f"    {tname}: RMSE={score:.4f}, params={per_target_params[tname]}")

    # Ensemble mode (ML-005)
    elif args.ensemble:
        print(f"\n  === Ensemble Mode ({args.ensemble_method}) ===")
        print(f"  Base models: {args.ensemble_models}")

        if args.optimize:
            print("  Optimizing base models before ensemble...")
            best_model, ensemble_params, ensemble_score = get_optimized_ensemble_model(
                X_train_feat, y_train,
                n_targets=train_data.n_targets,
                base_models=args.ensemble_models,
                ensemble_method=args.ensemble_method,
                cv=args.cv,
                n_trials=args.n_trials,
            )
            best_model_name = f"ensemble_{args.ensemble_method}"
            print(f"  Ensemble RMSE: {ensemble_score:.4f}")
            print(f"  Base model params: {ensemble_params}")
        else:
            best_model = get_ensemble_model(
                n_targets=train_data.n_targets,
                base_models=args.ensemble_models,
                ensemble_method=args.ensemble_method,
            )
            best_model_name = f"ensemble_{args.ensemble_method}"

    elif args.model == "all":
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

    # Hyperparameter optimization (skip if ensemble or per_target mode - already optimized)
    if args.optimize and not args.ensemble and not args.per_target:
        print(f"\n  Optimizing {best_model_name}...")

        if args.two_stage:
            # Two-stage optimization (recommended for large search space)
            print("\n  === Two-Stage Optimization ===")

            if args.optimize_features:
                # Extended Stage 1: Also optimize feature method (PCA vs wavelength selection)
                from src.optimization import optimize_preprocessing_and_features, optimize_model_with_features

                print("  Stage 1: Optimizing preprocessing + feature method (PCA vs wavelength selection)...")
                print(f"  Searching: baseline_lam (1e5-1e7), baseline_p (0.001-0.05)")
                print(f"  Searching: savgol_window (7-21 odd), savgol_polyorder (2-4)")
                print(f"  Searching: feature_method [pca, wavelength_selection]")
                print(f"  If PCA: n_components [{args.n_components_min}, {args.n_components_max}]")
                print(f"  If wavelength_selection: n_selected_wavelengths [{args.n_selected_min}, {args.n_selected_max}], selection_method [correlation, variance, f_score]")

                stage1_params, stage1_score = optimize_preprocessing_and_features(
                    train_data.spectra,
                    y_train,
                    n_components_range=(args.n_components_min, args.n_components_max),
                    n_wavelengths_range=(args.n_selected_min, args.n_selected_max),
                    cv=args.cv,
                    n_trials=args.n_trials,
                    subsample_ratio=args.subsample_ratio,
                )
                print(f"  Stage 1 Best RMSE: {stage1_score:.4f}")
                print(f"  Stage 1 params: {stage1_params}")

                # Extract optimal parameters
                opt_baseline_lam = stage1_params.get("baseline_lam", args.baseline_lam)
                opt_baseline_p = stage1_params.get("baseline_p", args.baseline_p)
                opt_savgol_window = stage1_params.get("savgol_window", args.savgol_window)
                opt_savgol_polyorder = stage1_params.get("savgol_polyorder", args.savgol_polyorder)
                opt_feature_method = stage1_params.get("feature_method", "pca")
                opt_n_components = stage1_params.get("n_components", args.n_components)
                opt_n_wavelengths = stage1_params.get("n_wavelengths", args.n_selected_wavelengths)
                opt_selection_method = stage1_params.get("selection_method", args.selection_method)

                print(f"\n  Optimal preprocessing: lam={opt_baseline_lam:.2e}, p={opt_baseline_p:.4f}, "
                      f"window={opt_savgol_window}, polyorder={opt_savgol_polyorder}")
                print(f"  Optimal feature method: {opt_feature_method}")

                # Apply optimal preprocessing
                preprocessor = Preprocessor(
                    baseline=args.baseline,
                    normalize=args.normalize,
                    denoise=args.denoise,
                    baseline_lam=opt_baseline_lam,
                    baseline_p=opt_baseline_p,
                    savgol_window=opt_savgol_window,
                    savgol_polyorder=opt_savgol_polyorder,
                )
                X_train = preprocessor.fit_transform(train_data.spectra)
                if test_data is not None:
                    X_test = preprocessor.transform(test_data.spectra)

                # Stage 2: Optimize model params with optimal feature method
                print(f"\n  Stage 2: Optimizing {best_model_name} hyperparameters (full data)...")
                best_params, stage2_score = optimize_model_with_features(
                    X_train,
                    y_train,
                    model_name=best_model_name,
                    feature_method=opt_feature_method,
                    n_components=opt_n_components,
                    n_wavelengths=opt_n_wavelengths,
                    selection_method=opt_selection_method,
                    cv=args.cv,
                    n_trials=args.n_trials,
                )
                print(f"  Stage 2 Best RMSE: {stage2_score:.4f}")
                print(f"  Stage 2 params: {best_params}")

                # Build final feature extractor
                if opt_feature_method == "pca":
                    print(f"\n  Final feature method: PCA with n_components={opt_n_components}")
                    feature_extractor = FeatureExtractor(method="pca", n_components=opt_n_components)
                else:
                    print(f"\n  Final feature method: wavelength_selection ({opt_selection_method}) with n_wavelengths={opt_n_wavelengths}")
                    feature_extractor = FeatureExtractor(
                        method="wavelength_selection",
                        n_components=opt_n_wavelengths,
                        selection_method=opt_selection_method,
                    )
                X_train_feat = feature_extractor.fit_transform(X_train, train_data.targets)
                print(f"  Final features shape: {X_train_feat.shape}")
                if X_test is not None:
                    X_test_feat = feature_extractor.transform(X_test)

            else:
                # Original two-stage: preprocessing + PCA only
                from src.optimization import optimize_preprocessing_only, optimize_model_only

                print("  Stage 1: Optimizing preprocessing + PCA (fixed model)...")
                print(f"  Searching: baseline_lam (1e5-1e7), baseline_p (0.001-0.05)")
                print(f"  Searching: savgol_window (7-21 odd), savgol_polyorder (2-4)")
                print(f"  Searching: n_components [{args.n_components_min}, {args.n_components_max}]")

                stage1_params, stage1_score = optimize_preprocessing_only(
                    train_data.spectra,
                    y_train,
                    n_components_range=(args.n_components_min, args.n_components_max),
                    cv=args.cv,
                    n_trials=args.n_trials,
                    subsample_ratio=args.subsample_ratio,
                )
                print(f"  Stage 1 Best RMSE: {stage1_score:.4f}")
                print(f"  Stage 1 params: {stage1_params}")

                # Apply Stage 1 optimal preprocessing
                opt_baseline_lam = stage1_params.get("baseline_lam", args.baseline_lam)
                opt_baseline_p = stage1_params.get("baseline_p", args.baseline_p)
                opt_savgol_window = stage1_params.get("savgol_window", args.savgol_window)
                opt_savgol_polyorder = stage1_params.get("savgol_polyorder", args.savgol_polyorder)
                opt_n_components = stage1_params.get("n_components", args.n_components)

                print(f"\n  Applying optimal preprocessing: lam={opt_baseline_lam:.2e}, p={opt_baseline_p:.4f}, "
                      f"window={opt_savgol_window}, polyorder={opt_savgol_polyorder}")

                preprocessor = Preprocessor(
                    baseline=args.baseline,
                    normalize=args.normalize,
                    denoise=args.denoise,
                    baseline_lam=opt_baseline_lam,
                    baseline_p=opt_baseline_p,
                    savgol_window=opt_savgol_window,
                    savgol_polyorder=opt_savgol_polyorder,
                )
                X_train = preprocessor.fit_transform(train_data.spectra)
                if test_data is not None:
                    X_test = preprocessor.transform(test_data.spectra)

                # Stage 2: Optimize model params with full data
                print(f"\n  Stage 2: Optimizing {best_model_name} hyperparameters (full data)...")
                best_params, stage2_score = optimize_model_only(
                    X_train,  # Preprocessed with optimal params
                    y_train,
                    model_name=best_model_name,
                    n_components=opt_n_components,
                    cv=args.cv,
                    n_trials=args.n_trials,
                )
                print(f"  Stage 2 Best RMSE: {stage2_score:.4f}")
                print(f"  Stage 2 params: {best_params}")

                # Update feature extractor with optimal n_components
                print(f"\n  Final n_components: {opt_n_components}")
                feature_extractor = FeatureExtractor(method="pca", n_components=opt_n_components)
                X_train_feat = feature_extractor.fit_transform(X_train, train_data.targets)
                print(f"  Final features shape: {X_train_feat.shape}")
                if X_test is not None:
                    X_test_feat = feature_extractor.transform(X_test)

        elif args.optimize_preprocess:
            # Full pipeline optimization: preprocessing + PCA + model hyperparameters
            from src.optimization import optimize_full_pipeline
            print("  Full pipeline optimization (preprocessing + PCA + model)...")
            print(f"  Searching: baseline_lam (1e4-1e8), baseline_p (0.001-0.1)")
            print(f"  Searching: savgol_window (5-31 odd), savgol_polyorder (2-5)")
            print(f"  Searching: n_components [{args.n_components_min}, {args.n_components_max}]")

            best_params, best_score = optimize_full_pipeline(
                train_data.spectra,  # Raw spectra, before preprocessing
                y_train,
                model_name=best_model_name,
                n_components_range=(args.n_components_min, args.n_components_max),
                cv=args.cv,
                n_trials=args.n_trials,
                subsample_ratio=args.subsample_ratio,
            )
            print(f"  Best RMSE: {best_score:.4f}")
            print(f"  Best params: {best_params}")

            # Extract and apply optimal preprocessing params
            opt_baseline_lam = best_params.pop("baseline_lam", args.baseline_lam)
            opt_baseline_p = best_params.pop("baseline_p", args.baseline_p)
            opt_savgol_window = best_params.pop("savgol_window", args.savgol_window)
            opt_savgol_polyorder = best_params.pop("savgol_polyorder", args.savgol_polyorder)

            print(f"  Optimal preprocessing: lam={opt_baseline_lam:.2e}, p={opt_baseline_p:.4f}, "
                  f"window={opt_savgol_window}, polyorder={opt_savgol_polyorder}")

            # Rebuild preprocessor with optimal params
            preprocessor = Preprocessor(
                baseline=args.baseline,
                normalize=args.normalize,
                denoise=args.denoise,
                baseline_lam=opt_baseline_lam,
                baseline_p=opt_baseline_p,
                savgol_window=opt_savgol_window,
                savgol_polyorder=opt_savgol_polyorder,
            )
            X_train = preprocessor.fit_transform(train_data.spectra)
            if test_data is not None:
                X_test = preprocessor.transform(test_data.spectra)

            # Extract and apply optimal n_components
            opt_n_components = best_params.pop("n_components", args.n_components)
            print(f"  Optimal n_components: {opt_n_components}")
            feature_extractor = FeatureExtractor(method="pca", n_components=opt_n_components)
            X_train_feat = feature_extractor.fit_transform(X_train, train_data.targets)
            print(f"  Updated features shape: {X_train_feat.shape}")
            if X_test is not None:
                X_test_feat = feature_extractor.transform(X_test)

        elif args.optimize_n_components:
            # Joint optimization of n_components + model hyperparameters
            from src.optimization import optimize_with_pca
            print(f"  Including n_components in search range [{args.n_components_min}, {args.n_components_max}]...")
            best_params, _ = optimize_with_pca(
                X_train, y_train,
                model_name=best_model_name,
                n_components_range=(args.n_components_min, args.n_components_max),
                cv=args.cv,
                n_trials=args.n_trials,
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
                cv=args.cv,
                n_trials=args.n_trials
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
