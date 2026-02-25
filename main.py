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
from src.target_transform import LogitTargetTransformer
import torch


def get_git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return None


def run_classify(args) -> None:
    """Execute classification pipeline (--task classify)."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from src.data_loader import load_libs_benchmark

    print("=" * 60)
    print("OES Classification Pipeline")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/3] Loading data...")
    train_path = Path(args.train)
    if train_path.suffix in (".h5", ".hdf5"):
        X, y = load_libs_benchmark(str(train_path.parent), split="train")
    else:
        raise ValueError(f"Unsupported file format for classify task: {train_path.suffix}. Use .h5 files.")
    print(f"  Loaded: X={X.shape}, classes={len(set(y.tolist()))}")

    # --- Feature extraction ---
    print("\n[2/3] Feature extraction (PCA)...")
    n_components = args.n_components
    pipeline_steps = [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=args.seed)),
    ]

    if args.model == "svm":
        clf = SVC(kernel="rbf", C=10, gamma="scale", probability=False, random_state=args.seed)
    elif args.model == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1)
    else:
        raise ValueError(f"Unknown classify model: {args.model}. Use 'svm' or 'rf'.")

    model = Pipeline(pipeline_steps + [("clf", clf)])

    # --- Cross-validation ---
    print(f"\n[3/3] {args.cv}-fold stratified CV with {args.model.upper()}...")
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1)

    micro_f1 = float(f1_score(y, y_pred, average="micro"))
    macro_f1 = float(f1_score(y, y_pred, average="macro"))
    acc = float(accuracy_score(y, y_pred))
    classes = sorted(set(y.tolist()))
    per_class_f1 = {
        f"class_{int(c):02d}": float(f1_score(y, y_pred, labels=[c], average="micro"))
        for c in classes
    }

    print(f"\n  micro_f1  = {micro_f1:.4f}")
    print(f"  macro_f1  = {macro_f1:.4f}")
    print(f"  accuracy  = {acc:.4f}")

    # --- Write metrics.json ---
    if args.metrics_out:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "git_sha": get_git_sha(),
            "model": args.model,
            "task": "classify",
            "cv_folds": args.cv,
            "seed": args.seed,
            "primary_metric": {"name": "micro_f1", "value": micro_f1},
            "metrics": {
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "accuracy": acc,
                "per_class_f1": per_class_f1,
            },
        }
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Wrote metrics json to {out_path}")

    print("\n" + "=" * 60)
    print("Classification pipeline completed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="OES/LIBS Spectral Analysis Pipeline")
    parser.add_argument("--train", type=str, required=True, help="Path to training data")
    parser.add_argument("--test", type=str, default=None, help="Path to test data")
    parser.add_argument(
        "--task",
        type=str,
        default="regress",
        choices=["classify", "regress", "temporal"],
        help="Task type: classify (LIBS 12-class), regress (CAP temperatures), temporal (BOSCH time-series)",
    )
    parser.add_argument("--n_wavelengths", type=int, default=40002, help="Number of wavelength columns")
    parser.add_argument("--target_cols", type=str, nargs="+", default=None, help="Target column names")
    parser.add_argument("--target", type=str, default=None,
                        help="Target column name for regress task (e.g. T_rot, T_vib, substrate_type)")
    parser.add_argument(
        "--model",
        type=str,
        default="ridge",
        choices=["pls", "ridge", "lasso", "rf", "svm", "cnn", "xgb", "hybrid", "ann", "ann_hybrid", "all"],
        help="Model to train (svm/rf for classify; ridge/pls/ann etc. for regress)",
    )
    parser.add_argument("--optimize", action="store_true", help="Optimize hyperparameters")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")

    # Ralph-friendly additions
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--metrics_out", type=str, default=None, help="Write machine-readable metrics json")

    # Preprocessing parameters
    parser.add_argument("--cosmic_ray", action="store_true", default=False,
                        help="Enable cosmic ray removal (Z-score spike filter, first preprocessing step)")
    parser.add_argument("--cosmic_ray_threshold", type=float, default=5.0,
                        help="Z-score threshold for cosmic ray detection (default 5.0)")
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
        choices=["correlation", "variance", "f_score", "sdvs", "nist"],
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

    # Logit target transform (ML-008)
    parser.add_argument("--logit_transform", action="store_true", help="Apply logit transform to low-concentration targets before fitting")
    parser.add_argument("--logit_elements", type=str, nargs="+", default=["C", "Si", "Mo", "Cu"],
                        help="Elements to apply logit transform to (default: C Si Mo Cu)")

    # CNN weighted loss + augmentation (ML-009)
    parser.add_argument("--cnn_weighted_loss", action="store_true",
                        help="Enable per-element weighted MSE in Phase 1 CNN training (C:2.0, Mo:2.0, Cu:1.5, Si:1.5)")
    parser.add_argument("--cnn_augment", action="store_true",
                        help="Enable spectral data augmentation during CNN Phase 2 training")

    # SDVS feature selection (ML-010): extend existing --selection_method choices
    # (sdvs is already registered in FeatureExtractor; expose it here)

    # Per-target aggregation evaluation (ML-013)
    parser.add_argument("--n_per_target", type=int, default=0,
        help="If >0, use GroupKFold + per-target aggregation in evaluation "
             "(set to 50 for this dataset: 50 spectra per target)")

    # ANN ensemble (ML-014)
    parser.add_argument("--n_ann_ensemble", type=int, default=16,
        help="Number of ANN models in ensemble for --model ann (default 16, like FORTH)")

    # Internal standard normalization (ML-018)
    parser.add_argument("--internal_standard", action="store_true",
        help="Divide each spectrum by the Fe 259.94 nm reference line intensity after "
             "baseline correction + smoothing (physical shot-to-shot correction)")
    parser.add_argument("--internal_standard_wl", type=float, default=259.94,
        help="Wavelength (nm) of the internal standard reference line (default: Fe 259.94 nm)")

    # Cr clean line selection (ML-019)
    parser.add_argument("--cr_clean_lines", action="store_true",
        help="Use reduced Cr NIST line set (267.716, 357.869, 425.433 nm only) "
             "to reduce Fe-matrix interference on Cr channels")

    # Cr model routing in ann_hybrid (ML-020)
    parser.add_argument("--cr_model", type=str, default="ridge",
        choices=["ridge", "pls"],
        help="Model for Cr in ann_hybrid: 'ridge' (PCA features) or "
             "'pls' (PLS on Cr NIST channels). Default: ridge (ML-016 baseline)")

    # Closure constraint post-processing (ML-022)
    parser.add_argument("--normalize_sum100", action="store_true",
        help="After prediction, rescale element concentrations so Σ(wt%%)=100 "
             "(closure constraint; Noll et al. 2014)")

    args = parser.parse_args()

    # Fix randomness for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Route to task-specific handler
    if args.task == "classify":
        run_classify(args)
        return

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
        cosmic_ray=args.cosmic_ray,
        cosmic_ray_threshold=args.cosmic_ray_threshold,
        internal_standard=args.internal_standard,
        internal_standard_wl=args.internal_standard_wl,
        wavelengths=train_data.wavelengths,
    )
    if args.internal_standard:
        print(f"  Internal standard: Fe {args.internal_standard_wl:.3f} nm")
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
            wavelengths=train_data.wavelengths,
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

    # ML-008: Logit transform of low-concentration targets
    logit_transformer = None
    y_for_opt = y_train.copy()  # original targets for preprocessing/stage-1 optimization
    if args.logit_transform:
        logit_transformer = LogitTargetTransformer(elements=args.logit_elements)
        y_train = logit_transformer.fit_transform(y_train, target_names)
        print(f"  Logit transform applied to: {args.logit_elements}")

    # Include CNN in models if requested
    include_deep = args.model == "cnn" or args.model == "all"
    models = get_traditional_models(n_targets=train_data.n_targets, include_deep=include_deep)

    # X passed to final fit/eval (updated to X_combined in hybrid per-target mode)
    X_train_for_model = X_train_feat

    # Per-target optimization mode (ML-006)
    if args.per_target:
        print(f"\n  === Per-Target Optimization Mode ===")
        print(f"  Model: {args.model}")
        print(f"  Training separate optimized model per target element")

        from src.optimization import optimize_per_target, PerTargetRegressor

        # Build per-target inverse transforms for logit elements so optimization
        # is performed in original space (prevents error amplification by inverse logit).
        per_target_inv = None
        if logit_transformer is not None and logit_transformer._transform_mask is not None:
            per_target_inv = {}
            for idx, tname in enumerate(target_names):
                if logit_transformer._transform_mask[idx]:
                    per_target_inv[tname] = lambda yp: 100.0 / (1.0 + np.exp(-yp))

        # Default hybrid map: XGBoost for low-concentration elements,
        # Ridge for high-concentration elements (ML-012)
        DEFAULT_HYBRID_MAP = {
            "C": "xgb", "Si": "xgb", "Mo": "xgb", "Cu": "xgb",
            "Mn": "ridge", "Cr": "ridge", "Ni": "ridge", "Fe": "ridge",
        }

        # Set up per-element routing, wavelengths, and combined features
        feat_wavelengths = None
        n_pca_cols = 0
        model_map = None

        if args.model == "hybrid":
            # Build NIST feature columns for XGBoost elements (hstacked after PCA)
            print("  [Hybrid] Building NIST feature columns for XGBoost elements...")
            fe_nist = FeatureExtractor(
                method="wavelength_selection",
                selection_method="nist",
                wavelengths=train_data.wavelengths,
            )
            X_nist = fe_nist.fit_transform(X_train, y_train)
            n_pca_cols = X_train_feat.shape[1]
            X_train_for_model = np.hstack([X_train_feat, X_nist])
            feat_wavelengths = train_data.wavelengths[fe_nist._selected_indices]
            model_map = DEFAULT_HYBRID_MAP
            print(f"  [Hybrid] X_pca={X_train_feat.shape}, X_nist={X_nist.shape}, "
                  f"X_combined={X_train_for_model.shape}")
        elif args.model == "xgb":
            if (args.feature_method == "wavelength_selection"
                    and feature_extractor._selected_indices is not None):
                feat_wavelengths = train_data.wavelengths[feature_extractor._selected_indices]
            elif args.feature_method == "none":
                feat_wavelengths = train_data.wavelengths

        elif args.model == "ann":
            # ANN (FORTH replication): NIST features, per-element channel routing, BaggingRegressor
            # ML-019: optionally use reduced Cr NIST lines
            print("  [ANN] Building NIST feature columns for per-element ANN...")
            fe_nist = FeatureExtractor(
                method="wavelength_selection",
                selection_method="nist",
                wavelengths=train_data.wavelengths,
            )
            if args.cr_clean_lines:
                from src.features import NIST_EMISSION_LINES_CR_CLEAN, select_wavelengths_nist, NIST_DELTA_NM
                _nist_idx = select_wavelengths_nist(
                    train_data.wavelengths,
                    delta_nm=NIST_DELTA_NM,
                    nist_lines=NIST_EMISSION_LINES_CR_CLEAN,
                )
                fe_nist._selected_indices = _nist_idx
                X_nist = fe_nist.transform(X_train)
                print("  [ANN] Using reduced Cr NIST lines (ML-019)")
            else:
                X_nist = fe_nist.fit_transform(X_train, y_train)
            X_train_for_model = X_nist
            feat_wavelengths = train_data.wavelengths[fe_nist._selected_indices]
            print(f"  [ANN] X_nist={X_nist.shape}")

        elif args.model == "ann_hybrid":
            # ML-016: ANN+NIST for all elements except Cr which reverts to Ridge+PCA
            # Cr degrades with ANN (5.56→6.12), all others benefit from ANN
            # ML-019: optionally use reduced Cr NIST line set
            # ML-020: Cr can use PLS on its NIST channels instead of Ridge+PCA
            print("  [ANN-Hybrid] Building combined PCA+NIST feature matrix...")
            _nist_lines_override = None
            if args.cr_clean_lines:
                from src.features import NIST_EMISSION_LINES_CR_CLEAN
                _nist_lines_override = NIST_EMISSION_LINES_CR_CLEAN
                print("  [ANN-Hybrid] Using reduced Cr NIST lines (ML-019: clean lines only)")
            fe_nist = FeatureExtractor(
                method="wavelength_selection",
                selection_method="nist",
                wavelengths=train_data.wavelengths,
            )
            # If clean Cr lines requested, temporarily patch the NIST dict via fit
            if _nist_lines_override is not None:
                from src.features import select_wavelengths_nist, NIST_DELTA_NM
                _nist_idx = select_wavelengths_nist(
                    train_data.wavelengths,
                    delta_nm=NIST_DELTA_NM,
                    nist_lines=_nist_lines_override,
                )
                fe_nist._selected_indices = _nist_idx
                X_nist = fe_nist.transform(X_train)
            else:
                X_nist = fe_nist.fit_transform(X_train, y_train)
            n_pca_cols = X_train_feat.shape[1]
            X_train_for_model = np.hstack([X_train_feat, X_nist])
            feat_wavelengths = train_data.wavelengths[fe_nist._selected_indices]
            # Cr model routing: ridge (default, ML-016) or pls (ML-020)
            cr_model_choice = args.cr_model  # "ridge" or "pls"
            model_map = {"Cr": cr_model_choice}
            print(f"  [ANN-Hybrid] X_pca={X_train_feat.shape}, X_nist={X_nist.shape}, "
                  f"X_combined={X_train_for_model.shape}")
            print(f"  [ANN-Hybrid] Routing: Cr→{cr_model_choice.upper()}, others→ANN+NIST")

        # Fallback model_name for elements not in model_map
        if args.model == "hybrid":
            eff_model_name = "ridge"
        elif args.model == "ann_hybrid":
            eff_model_name = "ann"
        else:
            eff_model_name = args.model

        per_target_models, per_target_params, per_target_scores, per_element_indices = optimize_per_target(
            X_train_for_model, y_train,
            model_name=eff_model_name,
            target_names=target_names,
            cv=args.cv,
            n_trials=args.n_trials,
            inverse_transforms=per_target_inv,
            y_original=y_for_opt if logit_transformer else None,
            wavelengths=feat_wavelengths,
            model_map=model_map,
            n_pca_cols=n_pca_cols,
            n_ann_ensemble=args.n_ann_ensemble,
        )

        best_model = PerTargetRegressor(
            per_target_models,
            target_names=target_names,
            per_element_indices=per_element_indices,
        )
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
        comparison_df = compare_models(models, X_train_feat, y_for_opt, cv=args.cv, target_names=target_names)
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
                    y_for_opt,  # use original targets for preprocessing optimization
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
                    internal_standard=args.internal_standard,
                    internal_standard_wl=args.internal_standard_wl,
                    wavelengths=train_data.wavelengths,
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
                    y_for_opt,  # use original targets for preprocessing optimization
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
                    internal_standard=args.internal_standard,
                    internal_standard_wl=args.internal_standard_wl,
                    wavelengths=train_data.wavelengths,
                )
                X_train = preprocessor.fit_transform(train_data.spectra)
                if test_data is not None:
                    X_test = preprocessor.transform(test_data.spectra)

                # Stage 2: Optimize model params with full data
                print(f"\n  Stage 2: Optimizing {best_model_name} hyperparameters (full data)...")
                best_params, stage2_score = optimize_model_only(
                    X_train,  # Preprocessed with optimal params
                    y_train,  # may be logit-transformed
                    model_name=best_model_name,
                    n_components=opt_n_components,
                    cv=args.cv,
                    n_trials=args.n_trials,
                    inverse_transform=logit_transformer.inverse_transform if logit_transformer else None,
                    y_true=y_for_opt if logit_transformer else None,
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
                internal_standard=args.internal_standard,
                internal_standard_wl=args.internal_standard_wl,
                wavelengths=train_data.wavelengths,
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

    # ML-009: Apply CNN weighted loss / augmentation flags if CNN model selected
    if "cnn" in best_model_name.lower():
        from src.models.deep_learning import CNNRegressor
        if isinstance(best_model, CNNRegressor):
            if args.cnn_weighted_loss:
                best_model.element_weights = {"C": 2.0, "Mo": 2.0, "Cu": 1.5, "Si": 1.5}
                best_model.target_names = target_names
            if args.cnn_augment:
                best_model.augment = True

    # Train final model
    print(f"\n  Training {best_model_name}...")
    best_model = train_traditional_model(best_model, X_train_for_model, y_train)

    # Save model
    model_path = output_dir / "models" / f"{best_model_name}_model.joblib"
    save_model(best_model, str(model_path))

    # 5. Evaluate
    print("\n[5/5] Evaluation...")
    metrics, y_pred = evaluate_model(
        best_model, X_train_for_model, y_train,
        cv=args.cv, target_names=target_names,
        pred_transform=logit_transformer.inverse_transform if logit_transformer else None,
        y_true=y_for_opt if logit_transformer else None,
        n_per_target=args.n_per_target,
        normalize_sum100=args.normalize_sum100,
    )

    report = generate_report(metrics, best_model_name, str(output_dir / "evaluation_report.txt"))
    print(report)

    # Plot predictions (always in original wt% space)
    fig = plot_prediction_comparison(
        y_for_opt, y_pred, target_names,
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

        per_target_rmse_mean = metrics.get("_overall", {}).get("per_target_RMSE_mean") if isinstance(metrics, dict) else None

        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "git_sha": get_git_sha(),
            "model": best_model_name,
            "cv_folds": args.cv,
            "seed": args.seed,
            "primary_metric": {"name": "RMSE_mean", "value": rmse_mean},
            "per_target_rmse_mean": per_target_rmse_mean,
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
