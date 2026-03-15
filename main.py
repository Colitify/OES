"""Main script for Plasma OES Spectral Analysis Pipeline."""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from src.preprocessing import Preprocessor
from src.features import FeatureExtractor
from src.models.traditional import (
    get_traditional_models,
    train_traditional_model,
    save_model,
    get_ensemble_model,
    get_optimized_ensemble_model,
)
from src.evaluation import (
    evaluate_model,
    compare_models,
    plot_prediction_comparison,
    generate_report,
)


from src.utils import get_git_sha

_CAP_TARGETS = ("T_rot", "T_vib", "power", "flow", "substrate_type")
_CAP_THRESHOLDS = {"T_rot": 50.0, "T_vib": 200.0}


def run_classify(args) -> None:
    """Execute OES substrate classification pipeline (--task classify).

    Uses Mesbah CAP substrate_type (glass=0, metal=1) as binary classification
    demo with SVM(rbf) or RandomForest on the 51 OES channels.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from src.data_loader import load_mesbah_cap

    print("=" * 60)
    print("OES Substrate Classification Pipeline")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/3] Loading Mesbah CAP data (substrate_type)...")
    X, y, wavelengths = load_mesbah_cap(args.train, target="substrate_type")
    print(f"  Loaded: X={X.shape}, classes={len(set(y.tolist()))}")

    # --- Build pipeline ---
    print("\n[2/3] Building classifier pipeline...")
    if args.model == "svm":
        clf = SVC(kernel="rbf", C=10, gamma="scale", probability=False, random_state=args.seed)
    elif args.model == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1)
    else:
        raise ValueError(f"Unknown classify model: {args.model}. Use 'svm' or 'rf'.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

    # --- Cross-validation ---
    print(f"\n[3/3] {args.cv}-fold stratified CV with {args.model.upper()}...")
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1)

    micro_f1 = float(f1_score(y, y_pred, average="micro"))
    macro_f1 = float(f1_score(y, y_pred, average="macro"))
    acc = float(accuracy_score(y, y_pred))

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
            },
        }
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Wrote metrics json to {out_path}")

    print("\n" + "=" * 60)
    print("Classification pipeline completed!")
    print("=" * 60)


def _is_mesbah_cap(train_path: str, target: str | None) -> bool:
    """Detect Mesbah Lab CAP dataset by path name or target column."""
    if target in _CAP_TARGETS:
        return True
    p = Path(train_path)
    return "mesbah_cap" in str(p) or p.name in ("dat_train.csv", "dat_test.csv")


def run_cap_regress(args) -> None:
    """Execute Mesbah Lab CAP temperature regression pipeline (--task regress with CAP data).

    Uses ANN (MLPRegressor in BaggingRegressor) on the raw 51-channel OES features.
    """
    import warnings
    warnings.filterwarnings("ignore")

    from sklearn.ensemble import BaggingRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    from src.data_loader import load_mesbah_cap

    # Determine which targets to fit (CAP always runs per-target)
    if args.target:
        targets = [args.target]
    else:
        targets = ["T_rot", "T_vib"]

    print("=" * 60)
    print("Mesbah Lab CAP Temperature Regression (ANN per-target)")
    print("=" * 60)

    n_ens = getattr(args, "n_ann_ensemble", 16)
    cv_folds = getattr(args, "cv", 5)

    results = {}
    for target in targets:
        if target not in _CAP_TARGETS:
            print(f"  [SKIP] Unknown CAP target: {target}")
            continue
        print(f"\n--- Target: {target} ---")
        X, y, wavelengths = load_mesbah_cap(args.train, target=target)
        print(f"  X={X.shape}, y: mean={y.mean():.1f}, std={y.std():.1f}")

        # StandardScaler (ALS + SNV not needed for narrow-band N2 OES channels)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        # ANN: single hidden layer (64) proved best on 51-channel CAP data
        base_ann = MLPRegressor(
            hidden_layer_sizes=(64,),
            activation="logistic",
            solver="lbfgs",
            alpha=0.01,
            max_iter=2000,
            random_state=args.seed,
        )
        model = BaggingRegressor(
            estimator=base_ann,
            n_estimators=n_ens,
            bootstrap=True,
            n_jobs=-1,
            random_state=args.seed,
        )

        scores = cross_val_score(
            model, X_sc, y, cv=cv_folds,
            scoring="neg_root_mean_squared_error", n_jobs=1
        )
        rmse = float(-scores.mean())
        rmse_std = float(scores.std())
        print(f"  {cv_folds}-fold CV RMSE: {rmse:.2f} ± {rmse_std:.2f} K")

        threshold = _CAP_THRESHOLDS.get(target)
        if threshold is not None:
            status = "PASS" if rmse <= threshold else "FAIL"
            print(f"  Guardrail ({target} RMSE <= {threshold} K): {status}")

        results[target] = {"rmse": rmse, "rmse_std": rmse_std}

    # Summary
    print("\n" + "-" * 40)
    for tgt, res in results.items():
        threshold = _CAP_THRESHOLDS.get(tgt, None)
        thr_str = f" / target <= {threshold} K" if threshold else ""
        print(f"  {tgt}: RMSE = {res['rmse']:.2f} K{thr_str}")

    # Write metrics_cap.json
    if args.metrics_out:
        rmse_values = [v["rmse"] for v in results.values()]
        rmse_mean = sum(rmse_values) / len(rmse_values)
        payload = {
            "primary_metric": {"name": "RMSE_mean", "value": round(rmse_mean, 4)},
            "per_target_rmse": {k: round(v["rmse"], 4) for k, v in results.items()},
            "per_target_rmse_std": {k: round(v["rmse_std"], 4) for k, v in results.items()},
        }
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Wrote metrics to {out_path}")

    print("\n" + "=" * 60)
    print("CAP regression pipeline completed!")
    print("=" * 60)


def run_temporal(args) -> None:
    """Execute BOSCH temporal analysis pipeline (--task temporal)."""
    from src.data_loader import load_bosch_oes
    from src.temporal import cluster_discharge_phases, compute_temporal_embedding

    n_comp = getattr(args, "n_temporal_components", 20)
    n_clusters = getattr(args, "n_clusters", 4)
    model_type = args.model  # 'lstm' or 'dtw'

    print("=" * 60)
    print("BOSCH Plasma Etching Temporal Analysis")
    print(f"  Model: {model_type}  |  n_clusters: {n_clusters}  |  PCA components: {n_comp}")
    print("=" * 60)

    print(f"\nLoading BOSCH OES data from {args.train}...")
    data = load_bosch_oes(args.train)
    spectra = data["spectra"]
    print(f"  Loaded: {spectra.shape} spectra, {data['day_file']} / {data['wafer_key']}")

    print(f"Computing PCA({n_comp}) temporal embedding...")
    embedding, pca = compute_temporal_embedding(spectra, n_components=n_comp)
    print(f"  Embedding: {embedding.shape}")

    if model_type in ("dtw", "all"):
        print(f"\nDTW K-means clustering (k={n_clusters})...")
        labels, centroids, inertia = cluster_discharge_phases(
            embedding, k=n_clusters, metric="euclidean"
        )
        print(f"  Inertia: {inertia:.2f}")
        print(f"  Cluster sizes: {[(labels == c).sum() for c in range(n_clusters)]}")

    if model_type in ("lstm", "all"):
        import torch
        from src.temporal import LSTMPredictor, train_lstm

        seq_len = getattr(args, "seq_len", 10)
        epochs = 50
        print(f"\nTraining LSTM (seq_len={seq_len}, epochs={epochs})...")
        model = LSTMPredictor(n_features=n_comp, hidden_size=64, n_layers=2)
        history = train_lstm(model, embedding, epochs=epochs, lr=1e-3)
        print(f"  Val MSE: {history['val_loss'][0]:.4f} → {history['val_loss'][-1]:.4f}")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "n_features": n_comp},
                   str(out_dir / "lstm_temporal.pt"))
        print(f"  Saved model to {out_dir}/lstm_temporal.pt")

    print("\n" + "=" * 60)
    print("Temporal analysis pipeline completed!")
    print("=" * 60)


def run_intensity(args) -> None:
    """Execute semi-quantitative intensity analysis (--task intensity)."""
    from src.data_loader import load_bosch_multi_wafer
    from src.intensity import actinometry, compute_line_ratios, oes_to_process_regression
    from src.temporal import extract_species_timeseries
    from src.features import PLASMA_EMISSION_LINES

    print("=" * 60)
    print("Semi-quantitative Intensity Analysis Pipeline")
    print(f"  Model: {args.model}  |  CV: {args.cv}")
    print("=" * 60)

    print("\n[1/4] Loading BOSCH multi-wafer data...")
    data = load_bosch_multi_wafer(
        args.train,
        max_wafers=getattr(args, "max_wafers", 5),
        max_timesteps=getattr(args, "max_timesteps", None),
    )
    spectra = data["spectra"]
    wl = data["wavelengths"]
    proc = data["process_params"]
    feat_names = data["feature_names"]
    print(f"  Spectra: {spectra.shape}, Process params: {proc.shape}")

    print("\n[2/4] Extracting species intensity time series...")
    ts, species_names = extract_species_timeseries(spectra, wl)
    print(f"  Extracted {len(species_names)} species: {species_names}")

    print("\n[3/4] Computing actinometry ratios (ref: Ar_I)...")
    actin_species = [s for s in species_names if s != "Ar_I"]
    actin_matrix = np.column_stack([
        actinometry(spectra, wl, sp, "Ar_I") for sp in actin_species
    ])
    print(f"  Actinometry matrix: {actin_matrix.shape}")

    print(f"\n[4/4] OES → process parameter regression ({args.model})...")
    reg_model = args.model if args.model in ("ridge", "pls", "rf", "ann") else "ridge"
    results = {}
    for j, pname in enumerate(feat_names[:min(8, len(feat_names))]):
        y_proc = proc[:, j]
        if y_proc.std() < 1e-6:
            continue
        res = oes_to_process_regression(actin_matrix, y_proc, model_type=reg_model, cv=args.cv)
        results[pname] = {"rmse": res["rmse"], "r2": res["r2"]}
        status = "OK" if res["r2"] > 0.3 else "low"
        print(f"  {pname}: RMSE={res['rmse']:.3f}, R²={res['r2']:.3f} [{status}]")

    if args.metrics_out:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "git_sha": get_git_sha(),
            "task": "intensity",
            "model": reg_model,
            "primary_metric": {"name": "mean_r2", "value": float(np.mean([r["r2"] for r in results.values()])) if results else 0.0},
            "per_parameter": results,
            "species_used": actin_species,
        }
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  Wrote metrics to {out_path}")

    print("\n" + "=" * 60)
    print("Intensity analysis pipeline completed!")
    print("=" * 60)


def run_spatiotemporal(args) -> None:
    """Execute spatiotemporal evolution analysis (--task spatiotemporal)."""
    from src.data_loader import load_bosch_oes, find_gas_column, label_process_phases
    from src.temporal import extract_species_timeseries, train_attention_classifier
    from src.temporal import compute_temporal_embedding

    print("=" * 60)
    print("Spatiotemporal Evolution Analysis Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading BOSCH OES data...")
    data = load_bosch_oes(args.train)
    spectra = data["spectra"]
    wl = data["wavelengths"]

    max_ts = getattr(args, "max_timesteps", None)
    if max_ts and len(spectra) > max_ts:
        idx = np.linspace(0, len(spectra) - 1, max_ts, dtype=int)
        spectra = spectra[idx]
        if data["process_params"].shape[1] > 0:
            data["process_params"] = data["process_params"][idx]

    print(f"  Spectra: {spectra.shape}, {data['day_file']} / {data['wafer_key']}")

    print("\n[2/4] Extracting species intensity time series...")
    ts, species_names = extract_species_timeseries(spectra, wl)
    print(f"  Species: {species_names}")
    print(f"  Time series shape: {ts.shape}")

    print("\n[3/4] Computing PCA embedding + training Attention-LSTM...")
    n_comp = getattr(args, "n_temporal_components", 20)
    embedding, pca = compute_temporal_embedding(spectra, n_components=n_comp)

    proc = data["process_params"]
    result = {"accuracy": 0.0, "attn_weights": np.zeros((1, 1))}
    if proc.shape[1] > 0:
        feat_names = data["process_feature_names"]
        sf6_idx = find_gas_column(feat_names, None, ["sf6", "SF6"])
        c4f8_idx = find_gas_column(feat_names, None, ["c4f8", "C4F8"])

        if sf6_idx is not None and c4f8_idx is not None:
            labels = label_process_phases(proc[:, sf6_idx], proc[:, c4f8_idx])
            unique_labels = np.unique(labels)

            if len(unique_labels) >= 2:
                seq_len = getattr(args, "seq_len", 20)
                result = train_attention_classifier(
                    embedding, labels, seq_len=seq_len, epochs=50
                )
                print(f"  Attention-LSTM val accuracy: {result['accuracy']:.4f}")
                print(f"  Attention weights shape: {result['attn_weights'].shape}")
            else:
                print(f"  Only {len(unique_labels)} phase(s) found — skipping Attention-LSTM")
        else:
            print("  Gas flow columns not found — skipping phase classification")
    else:
        print("  No process params — skipping phase classification")

    print("\n[4/4] Spatial coupling: use --task intensity for OES → etch uniformity prediction")

    if args.metrics_out:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "git_sha": get_git_sha(),
            "task": "spatiotemporal",
            "primary_metric": {"name": "attention_lstm_accuracy", "value": result["accuracy"]},
            "species_count": len(species_names),
            "species_names": species_names,
            "n_timesteps": int(spectra.shape[0]),
        }
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  Wrote metrics to {out_path}")

    print("\n" + "=" * 60)
    print("Spatiotemporal analysis pipeline completed!")
    print("=" * 60)


def run_species(args) -> None:
    """Execute species classification pipeline (--task species)."""
    from src.data_loader import load_bosch_multi_wafer
    from src.species import (
        train_species_classifier,
        detect_species_presence_batch,
        nmf_decompose,
        compute_species_shap,
    )

    print("=" * 60)
    print("OES Species Classification Pipeline")
    print(f"  Model: {args.model}  |  CV: {args.cv}")
    print("=" * 60)

    print("\n[1/5] Loading BOSCH multi-wafer data...")
    data = load_bosch_multi_wafer(
        args.train,
        max_wafers=getattr(args, "max_wafers", 10),
        max_timesteps=getattr(args, "max_timesteps", None),
    )
    X = data["spectra"]
    y = data["labels"]
    wl = data["wavelengths"]
    print(f"  Loaded: X={X.shape}, classes={np.unique(y).tolist()}")

    mask = y > 0
    if mask.sum() > 100:
        X_clf, y_clf = X[mask], y[mask]
        print(f"  After removing idle: {X_clf.shape[0]} samples")
    else:
        X_clf, y_clf = X, y

    n_nmf = getattr(args, "n_nmf_components", 5)
    print(f"\n[2/5] NMF decomposition (k={n_nmf})...")
    components, weights, nmf_model = nmf_decompose(X_clf, n_components=n_nmf)
    print(f"  Reconstruction error: {nmf_model.reconstruction_err_:.2f}")

    print("\n[3/5] Species presence detection...")
    species_labels, species_names = detect_species_presence_batch(X_clf, wl)
    for i, sp in enumerate(species_names):
        pct = 100 * species_labels[:, i].mean()
        print(f"  {sp}: {pct:.1f}% of spectra")

    model_type = args.model if args.model in ("svm", "rf", "cnn") else "svm"
    print(f"\n[4/5] Training {model_type.upper()} classifier ({args.cv}-fold CV)...")
    result = train_species_classifier(X_clf, y_clf, model_type=model_type, cv=args.cv, seed=args.seed)
    print(f"  Accuracy:  {result['accuracy']:.4f}")
    print(f"  F1 macro:  {result['f1_macro']:.4f}")
    print(f"  F1/class:  {result['f1_per_class']}")

    if model_type in ("svm", "rf"):
        print(f"\n[5/5] Computing SHAP feature importance...")
        _, feat_imp = compute_species_shap(X_clf, y_clf, model_type=model_type, seed=args.seed)
        top_k = 10
        top_idx = np.argsort(feat_imp)[-top_k:][::-1]
        print(f"  Top-{top_k} wavelengths (nm): {wl[top_idx].tolist()}")
    else:
        print("\n[5/5] SHAP skipped for CNN (use RF for interpretability)")

    if args.metrics_out:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "git_sha": get_git_sha(),
            "task": "species",
            "model": model_type,
            "primary_metric": {"name": "f1_macro", "value": result["f1_macro"]},
            "metrics": {
                "accuracy": result["accuracy"],
                "f1_macro": result["f1_macro"],
                "f1_per_class": result["f1_per_class"],
            },
            "nmf_reconstruction_err": float(nmf_model.reconstruction_err_),
            "species_detection_pct": {
                sp: float(100 * species_labels[:, i].mean())
                for i, sp in enumerate(species_names)
            },
        }
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  Wrote metrics to {out_path}")

    print("\n" + "=" * 60)
    print("Species classification pipeline completed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plasma OES Spectral Analysis Pipeline\n\n"
            "Tasks:\n"
            "  classify       — Substrate classification on Mesbah CAP OES data (*.csv)\n"
            "  regress        — T_rot/T_vib temperature regression on Mesbah CAP (*.csv)\n"
            "  temporal       — Temporal clustering/forecasting on BOSCH plasma etching (dir/)\n"
            "  species        — Species classification on BOSCH multi-wafer OES data (dir/)\n"
            "  intensity      — Semi-quantitative intensity regression (dir/)\n"
            "  spatiotemporal — Spatial etch prediction from temporal OES (dir/)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Data arguments ──────────────────────────────────────────────────────
    data_grp = parser.add_argument_group("Data arguments")
    data_grp.add_argument("--train", type=str, required=True,
                          help="Path to training data: *.csv (classify/regress) "
                               "or directory (temporal)")
    data_grp.add_argument("--test", type=str, default=None,
                          help="Path to test data (same format as --train)")
    data_grp.add_argument(
        "--task", type=str, default="regress",
        choices=["classify", "regress", "temporal", "species", "intensity", "spatiotemporal"],
        help="Task mode (default: regress)",
    )
    data_grp.add_argument("--target", type=str, default=None,
                          help="Single target for regress task "
                               "(e.g. T_rot, T_vib, substrate_type)")

    # ── Model arguments ─────────────────────────────────────────────────────
    model_grp = parser.add_argument_group("Model arguments")
    model_grp.add_argument(
        "--model", type=str, default="ridge",
        choices=["pls", "ridge", "lasso", "rf", "svm", "xgb",
                 "ann", "cnn", "lstm", "dtw", "all"],
        help="Model to train:\n"
             "  classify : svm, rf\n"
             "  regress  : ridge, pls, lasso, rf, ann, xgb\n"
             "  temporal : lstm, dtw\n"
             "  species  : svm, rf, cnn\n"
             "  intensity: ridge, pls, rf, ann\n"
             "(default: ridge)",
    )
    model_grp.add_argument("--optimize", action="store_true",
                           help="Run Optuna hyperparameter optimisation")
    model_grp.add_argument("--per_target", action="store_true",
                           help="Train separate optimised model per target (regress only)")
    model_grp.add_argument("--n_trials", type=int, default=20,
                           help="Optuna trials per target (default 20)")
    model_grp.add_argument("--n_ann_ensemble", type=int, default=16,
                           help="BaggingRegressor ensemble size for --model ann (default 16)")
    # Temporal-specific
    model_grp.add_argument("--n_clusters", type=int, default=4,
                           help="Number of DTW K-means clusters for --task temporal "
                                "(default 4)")
    model_grp.add_argument("--seq_len", type=int, default=10,
                           help="LSTM sequence length (history window) for --task temporal "
                                "(default 10)")
    model_grp.add_argument("--n_temporal_components", type=int, default=20,
                           help="PCA components for BOSCH temporal embedding (default 20)")
    model_grp.add_argument("--max_wafers", type=int, default=10,
                            help="Max wafers to load for species/intensity tasks (default 10)")
    model_grp.add_argument("--max_timesteps", type=int, default=None,
                            help="Max timesteps per wafer (default None = all)")
    model_grp.add_argument("--n_nmf_components", type=int, default=5,
                            help="NMF components for species decomposition (default 5)")

    # ── Preprocessing arguments ─────────────────────────────────────────────
    pre_grp = parser.add_argument_group("Preprocessing arguments")
    pre_grp.add_argument("--cosmic_ray", action="store_true", default=False,
                         help="Enable cosmic-ray removal (Z-score spike filter)")
    pre_grp.add_argument("--cosmic_ray_threshold", type=float, default=5.0,
                         help="Z-score threshold for cosmic-ray detection (default 5.0)")
    pre_grp.add_argument("--baseline", type=str, default="als",
                         choices=["als", "none"],
                         help="Baseline correction method (default: als)")
    pre_grp.add_argument("--normalize", type=str, default="snv",
                         choices=["snv", "minmax", "l2", "none"],
                         help="Normalisation method (default: snv)")
    pre_grp.add_argument("--denoise", type=str, default="savgol",
                         choices=["savgol", "none"],
                         help="Denoising method (default: savgol)")
    pre_grp.add_argument("--baseline_lam", type=float, default=1e6,
                         help="ALS baseline smoothness λ (default 1e6)")
    pre_grp.add_argument("--baseline_p", type=float, default=0.01,
                         help="ALS baseline asymmetry p (default 0.01)")
    pre_grp.add_argument("--savgol_window", type=int, default=11,
                         help="Savitzky-Golay window length (default 11)")
    pre_grp.add_argument("--savgol_polyorder", type=int, default=3,
                         help="Savitzky-Golay polynomial order (default 3)")

    # ── Feature extraction arguments ─────────────────────────────────────────
    feat_grp = parser.add_argument_group("Feature extraction arguments")
    feat_grp.add_argument("--n_components", type=int, default=50,
                          help="PCA components for feature extraction (default 50)")
    feat_grp.add_argument("--feature_method", type=str, default="pca",
                          choices=["pca", "wavelength_selection"],
                          help="Feature extraction method (default: pca)")
    feat_grp.add_argument("--selection_method", type=str, default="correlation",
                          choices=["correlation", "variance", "f_score"],
                          help="Wavelength selection method (when --feature_method=wavelength_selection)")
    feat_grp.add_argument("--n_selected_wavelengths", type=int, default=500,
                          help="Wavelengths to select (default 500)")

    # ── Evaluation arguments ─────────────────────────────────────────────────
    eval_grp = parser.add_argument_group("Evaluation arguments")
    eval_grp.add_argument("--cv", type=int, default=5,
                          help="Cross-validation folds (default 5)")
    eval_grp.add_argument("--seed", type=int, default=42,
                          help="Random seed (default 42)")

    # ── Ensemble arguments ────────────────────────────────────────────────────
    ens_grp = parser.add_argument_group("Ensemble arguments (regress only)")
    ens_grp.add_argument("--ensemble", action="store_true",
                         help="Ensemble model: PLS + Ridge + RF")
    ens_grp.add_argument("--ensemble_method", type=str, default="stacking",
                         choices=["stacking", "voting"],
                         help="Ensemble method (default: stacking)")
    ens_grp.add_argument("--ensemble_models", type=str, nargs="+",
                         default=["pls", "ridge", "rf"],
                         help="Base models for ensemble (default: pls ridge rf)")

    # ── Output arguments ──────────────────────────────────────────────────────
    out_grp = parser.add_argument_group("Output arguments")
    out_grp.add_argument("--output_dir", type=str, default="outputs",
                         help="Output directory for models, figures, predictions "
                              "(default: outputs)")
    out_grp.add_argument("--metrics_out", type=str, default=None,
                         help="Write machine-readable metrics JSON to this path")

    args = parser.parse_args()

    # ── Cross-task argument validation ────────────────────────────────────────
    if args.task == "temporal" and args.model not in ("lstm", "dtw", "all"):
        parser.error(
            f"--task temporal requires --model in {{lstm, dtw}}, got '{args.model}'"
        )
    if args.task == "species" and args.model not in ("svm", "rf", "cnn"):
        parser.error(f"--task species requires --model in {{svm, rf, cnn}}, got '{args.model}'")
    if args.task == "intensity" and args.model not in ("ridge", "pls", "rf", "ann"):
        parser.error(f"--task intensity requires --model in {{ridge, pls, rf, ann}}, got '{args.model}'")

    # Fix randomness for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Route to task-specific handler
    if args.task == "classify":
        run_classify(args)
        return

    if args.task == "species":
        run_species(args)
        return

    if args.task == "temporal":
        run_temporal(args)
        return

    if args.task == "intensity":
        run_intensity(args)
        return

    if args.task == "spatiotemporal":
        run_spatiotemporal(args)
        return

    if args.task == "regress":
        run_cap_regress(args)
        return


if __name__ == "__main__":
    main()
