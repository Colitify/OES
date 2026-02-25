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
        X, y, _wavelengths = load_libs_benchmark(str(train_path.parent), split="train")
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

    if args.model == "cnn":
        # CNN classification path: PCA(200) preprocessing → Conv1DClassifier → Optuna HPO
        import optuna
        import torch
        from sklearn.model_selection import train_test_split
        from src.models.deep_learning import Conv1DClassifier, train_classifier, _get_safe_device

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Remap labels to 0-indexed contiguous integers (required by CrossEntropyLoss)
        classes_arr = np.unique(y)
        n_classes = int(len(classes_arr))
        label_map = {int(old): int(new) for new, old in enumerate(classes_arr)}
        y_mapped = np.array([label_map[int(lbl)] for lbl in y], dtype=np.int64)

        # PCA preprocessing: reduce 40002 → n_pca dims for fast CNN convergence
        n_pca = min(200, n_components)
        print(f"  Applying PCA({n_pca}) before CNN...")
        pca_pre = PCA(n_components=n_pca, random_state=args.seed)
        scaler_pre = StandardScaler()
        X_pca = pca_pre.fit_transform(scaler_pre.fit_transform(X.astype(np.float32)))
        print(f"  PCA reduced to {X_pca.shape}, {n_classes} classes")

        device = _get_safe_device()
        X_tr_hp, X_val_hp, y_tr_hp, y_val_hp = train_test_split(
            X_pca, y_mapped, test_size=0.2, stratify=y_mapped, random_state=args.seed
        )

        def _cnn_objective(trial):
            n_filters = trial.suggest_categorical("n_filters", [32, 64, 128])
            kernel_size = trial.suggest_categorical("kernel_size", [3, 7, 11])
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            clf_model = Conv1DClassifier(
                n_classes=n_classes, n_filters=n_filters, kernel_size=kernel_size,
                dropout=dropout, lr=lr,
            )
            trained = train_classifier(
                clf_model, X_tr_hp, y_tr_hp, X_val_hp, y_val_hp,
                epochs=20, batch_size=64, device=device,
            )
            trained.eval()
            with torch.no_grad():
                preds = trained(torch.FloatTensor(X_val_hp).to(device)).argmax(dim=1).cpu().numpy()
            return float(f1_score(y_val_hp, preds, average="micro"))

        n_hpo_trials = getattr(args, "n_trials", 20)
        print(f"\n[3/4] Optuna HPO: {n_hpo_trials} trials...")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
        study.optimize(_cnn_objective, n_trials=n_hpo_trials, show_progress_bar=False)
        best_params = study.best_params
        print(f"  Best HPO params: {best_params}  (val micro_f1={study.best_value:.4f})")

        # 5-fold CV with best params
        print(f"\n[4/4] {args.cv}-fold stratified CV with best CNN params...")
        cv_kf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
        y_pred = np.zeros(len(y_mapped), dtype=np.int64)
        for fold_i, (tr_idx, val_idx) in enumerate(cv_kf.split(X_pca, y_mapped)):
            clf_model = Conv1DClassifier(
                n_classes=n_classes, n_filters=best_params["n_filters"],
                kernel_size=best_params["kernel_size"], dropout=best_params["dropout"],
                lr=best_params["lr"],
            )
            trained = train_classifier(
                clf_model, X_pca[tr_idx], y_mapped[tr_idx],
                X_pca[val_idx], y_mapped[val_idx],
                epochs=30, batch_size=64, device=device,
            )
            trained.eval()
            with torch.no_grad():
                preds = trained(torch.FloatTensor(X_pca[val_idx]).to(device)).argmax(dim=1).cpu().numpy()
            y_pred[val_idx] = preds
            fold_f1 = float(f1_score(y_mapped[val_idx], preds, average="micro"))
            print(f"  Fold {fold_i+1}/{args.cv}: micro_f1={fold_f1:.4f}")

        micro_f1 = float(f1_score(y_mapped, y_pred, average="micro"))
        macro_f1 = float(f1_score(y_mapped, y_pred, average="macro"))
        acc = float(accuracy_score(y_mapped, y_pred))
        per_class_f1 = {
            f"class_{int(c):02d}": float(f1_score(y_mapped, y_pred, labels=[c], average="micro"))
            for c in sorted(set(y_mapped.tolist()))
        }

        # --- SHAP interpretability (OES-013) ---
        if getattr(args, "explain", False):
            import torch
            from sklearn.model_selection import train_test_split
            from src.evaluation import compute_shap_spectrum

            print("\n[SHAP] Training final model for interpretability...")
            X_sh_tr, X_sh_val, y_sh_tr, y_sh_val = train_test_split(
                X_pca, y_mapped, test_size=0.2, stratify=y_mapped, random_state=args.seed + 2
            )
            shap_model = Conv1DClassifier(
                n_classes=n_classes, n_filters=best_params["n_filters"],
                kernel_size=best_params["kernel_size"], dropout=best_params["dropout"],
                lr=best_params["lr"],
            )
            shap_model = train_classifier(
                shap_model, X_sh_tr, y_sh_tr, X_sh_val, y_sh_val,
                epochs=30, batch_size=64, device=device,
            )
            shap_model.eval()

            # Save checkpoint for plot_shap.py
            Path("outputs").mkdir(parents=True, exist_ok=True)
            ckpt = {
                "model_state_dict": shap_model.state_dict(),
                "n_classes": n_classes,
                "n_pca": n_pca,
                "best_params": best_params,
                "pca_components": pca_pre.components_,   # (n_pca, n_wavelengths)
                "pca_mean": pca_pre.mean_,                # (n_wavelengths,)
                "scaler_mean": scaler_pre.mean_,          # (n_wavelengths,)
                "scaler_scale": scaler_pre.scale_,        # (n_wavelengths,)
                "wavelengths": _wavelengths,
                "label_map": label_map,
            }
            torch.save(ckpt, "outputs/best_model.pt")
            print("  Saved model checkpoint to outputs/best_model.pt")

            # Compute SHAP on explain_samples
            n_exp = min(getattr(args, "explain_samples", 50), len(X_pca))
            n_bg = min(50, len(X_pca))
            rng_sh = np.random.default_rng(args.seed)
            bg_idx = rng_sh.choice(len(X_pca), size=n_bg, replace=False)
            exp_idx = rng_sh.choice(len(X_pca), size=n_exp, replace=False)

            class _PCAProxy:
                def __init__(self, comp):
                    self.components_ = comp
            pca_proxy = _PCAProxy(pca_pre.components_)
            print(f"  Computing SHAP ({n_bg} background, {n_exp} explain)...")
            shap_vals = compute_shap_spectrum(
                shap_model, X_pca[bg_idx], X_pca[exp_idx],
                wavelengths=_wavelengths, pca=pca_proxy,
            )
            mean_abs = np.abs(shap_vals).mean(axis=0)  # (n_wavelengths, n_classes)

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from src.features import PLASMA_EMISSION_LINES

            dominant_cls = int(mean_abs.max(axis=0).argmax())
            shap_1d = mean_abs[:, dominant_cls]
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(_wavelengths, shap_1d, lw=0.6, color="steelblue",
                    label=f"Mean |SHAP| (class {dominant_cls})")
            colors = plt.cm.tab10.colors
            for si, (sp, lines) in enumerate(PLASMA_EMISSION_LINES.items()):
                col = colors[si % len(colors)]
                for li, ln in enumerate(lines):
                    ax.axvline(ln, color=col, alpha=0.5, lw=0.8, linestyle="--",
                               label=sp if li == 0 else None)
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Mean |SHAP value|")
            ax.set_title("SHAP Spectrum Attribution with Plasma Emission Line Overlay")
            ax.legend(loc="upper right", fontsize=7, ncol=2)
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            fig_path = Path("outputs/shap_overlay.png")
            fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved SHAP overlay to {fig_path}")
    else:
        if args.model == "svm":
            clf = SVC(kernel="rbf", C=10, gamma="scale", probability=False, random_state=args.seed)
        elif args.model == "rf":
            clf = RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1)
        else:
            raise ValueError(f"Unknown classify model: {args.model}. Use 'svm', 'rf', or 'cnn'.")

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

    # --- MC-Dropout + Temperature Scaling (OES-012, CNN only) ---
    ece_val: float | None = None
    calibrated_micro_f1: float | None = None
    if getattr(args, "mc_dropout", False) and args.model == "cnn":
        from sklearn.model_selection import train_test_split
        from src.models.deep_learning import predict_with_uncertainty, _get_safe_device
        from src.models.calibration import TemperatureScaling
        from src.evaluation import compute_ece
        import torch

        device_mc = _get_safe_device()
        print("\n[MC-Dropout] Training final model for uncertainty/calibration...")
        # Train final model on 80% of PCA data
        X_fin_tr, X_fin_val, y_fin_tr, y_fin_val = train_test_split(
            X_pca, y_mapped, test_size=0.2, stratify=y_mapped, random_state=args.seed + 1
        )
        final_model = Conv1DClassifier(
            n_classes=n_classes, n_filters=best_params["n_filters"],
            kernel_size=best_params["kernel_size"], dropout=best_params["dropout"],
            lr=best_params["lr"],
        )
        final_model = train_classifier(
            final_model, X_fin_tr, y_fin_tr, X_fin_val, y_fin_val,
            epochs=30, batch_size=64, device=device_mc,
        )

        # MC-Dropout uncertainty on val set
        n_mc = getattr(args, "n_mc_samples", 50)
        print(f"  MC-Dropout: {n_mc} samples...")
        mean_probs, std_probs = predict_with_uncertainty(final_model, X_fin_val, n_samples=n_mc, device=device_mc)
        ece_before = compute_ece(mean_probs, y_fin_val)
        print(f"  ECE before calibration: {ece_before:.4f}")

        # Collect logits on val set for temperature scaling
        final_model.eval()
        with torch.no_grad():
            val_logits = final_model(torch.FloatTensor(X_fin_val).to(device_mc)).cpu().numpy()
        ts = TemperatureScaling()
        ts.fit(val_logits, y_fin_val)
        cal_probs = ts.transform(val_logits)
        ece_val = compute_ece(cal_probs, y_fin_val)
        print(f"  ECE after calibration (T={ts.T:.3f}): {ece_val:.4f}")
        calibrated_micro_f1 = float(f1_score(y_fin_val, cal_probs.argmax(axis=1), average="micro"))
        print(f"  Calibrated micro_f1: {calibrated_micro_f1:.4f}")

    # --- Write metrics.json ---
    if args.metrics_out:
        metrics_dict: dict = {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "accuracy": acc,
            "per_class_f1": per_class_f1,
        }
        if ece_val is not None:
            metrics_dict["ece"] = ece_val
        if calibrated_micro_f1 is not None:
            metrics_dict["calibrated_micro_f1"] = calibrated_micro_f1

        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "git_sha": get_git_sha(),
            "model": args.model,
            "task": "classify",
            "cv_folds": args.cv,
            "seed": args.seed,
            "primary_metric": {"name": "micro_f1", "value": micro_f1},
            "metrics": metrics_dict,
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
    _CAP_TARGETS = {"T_rot", "T_vib", "power", "flow", "substrate_type"}
    if target in _CAP_TARGETS:
        return True
    p = Path(train_path)
    return "mesbah_cap" in str(p) or p.name in ("dat_train.csv", "dat_test.csv")


def run_cap_regress(args) -> None:
    """Execute Mesbah Lab CAP temperature regression pipeline (--task regress with CAP data).

    Uses ANN (MLPRegressor in BaggingRegressor) on the raw 51-channel OES features.
    No NIST wavelength routing — the N2 2nd-positive channels are used directly.
    """
    import warnings
    warnings.filterwarnings("ignore")

    from sklearn.ensemble import BaggingRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    from src.data_loader import load_mesbah_cap

    _CAP_TARGETS = ("T_rot", "T_vib", "power", "flow", "substrate_type")
    _TARGETS_THRESHOLDS = {"T_rot": 50.0, "T_vib": 200.0}

    # Determine which targets to fit
    if args.target:
        targets = [args.target]
    elif args.per_target:
        targets = ["T_rot", "T_vib"]
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

        threshold = _TARGETS_THRESHOLDS.get(target)
        if threshold is not None:
            status = "PASS" if rmse <= threshold else "FAIL"
            print(f"  Guardrail ({target} RMSE <= {threshold} K): {status}")

        results[target] = {"rmse": rmse, "rmse_std": rmse_std}

    # Summary
    print("\n" + "-" * 40)
    for tgt, res in results.items():
        threshold = _TARGETS_THRESHOLDS.get(tgt, None)
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "OES/LIBS Spectral Analysis Pipeline\n\n"
            "Tasks:\n"
            "  classify  — 12-class ore-type classification on LIBS Benchmark (*.h5)\n"
            "  regress   — T_rot/T_vib temperature regression on Mesbah CAP (*.csv)\n"
            "  temporal  — Temporal clustering/forecasting on BOSCH plasma etching (dir/)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Data arguments ──────────────────────────────────────────────────────
    data_grp = parser.add_argument_group("Data arguments")
    data_grp.add_argument("--train", type=str, required=True,
                          help="Path to training data: *.h5 (classify), *.csv (regress), "
                               "or directory (temporal)")
    data_grp.add_argument("--test", type=str, default=None,
                          help="Path to test data (same format as --train)")
    data_grp.add_argument(
        "--task", type=str, default="regress",
        choices=["classify", "regress", "temporal"],
        help="Task mode (default: regress)",
    )
    data_grp.add_argument("--n_wavelengths", type=int, default=40002,
                          help="Number of wavelength columns in CSV (default 40002; "
                               "auto-detected for CAP/BOSCH data)")
    data_grp.add_argument("--target_cols", type=str, nargs="+", default=None,
                          help="Target column names (regress, auto-detected if unset)")
    data_grp.add_argument("--target", type=str, default=None,
                          help="Single target for regress task "
                               "(e.g. T_rot, T_vib, substrate_type)")
    data_grp.add_argument("--target_wavelengths", type=str, default=None,
                          help="Path to CSV with target wavelength grid (nm). "
                               "If not specified, uses dataset native grid.")

    # ── Model arguments ─────────────────────────────────────────────────────
    model_grp = parser.add_argument_group("Model arguments")
    model_grp.add_argument(
        "--model", type=str, default="ridge",
        choices=["pls", "ridge", "lasso", "rf", "svm", "cnn", "xgb",
                 "hybrid", "ann", "ann_hybrid", "lstm", "dtw", "all"],
        help="Model to train:\n"
             "  classify : svm, rf, cnn\n"
             "  regress  : ridge, pls, lasso, rf, ann, ann_hybrid, xgb, hybrid\n"
             "  temporal : lstm, dtw\n"
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

    # ── Preprocessing arguments ─────────────────────────────────────────────
    pre_grp = parser.add_argument_group("Preprocessing arguments (regress/classify)")
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
    pre_grp.add_argument("--internal_standard", action="store_true",
                         help="Divide each spectrum by Fe 259.94 nm reference line intensity")
    pre_grp.add_argument("--internal_standard_wl", type=float, default=259.94,
                         help="Internal standard wavelength (nm) (default 259.94)")

    # ── Feature extraction arguments ─────────────────────────────────────────
    feat_grp = parser.add_argument_group(
        "Feature extraction arguments (regress only)")
    feat_grp.add_argument("--n_components", type=int, default=50,
                          help="PCA components for feature extraction (default 50)")
    feat_grp.add_argument("--n_components_min", type=int, default=10,
                          help="Min n_components for optimisation search")
    feat_grp.add_argument("--n_components_max", type=int, default=100,
                          help="Max n_components for optimisation search")
    feat_grp.add_argument("--optimize_n_components", action="store_true",
                          help="Include n_components in hyperparameter optimisation")
    feat_grp.add_argument("--feature_method", type=str, default="pca",
                          choices=["pca", "wavelength_selection"],
                          help="Feature extraction method (default: pca)")
    feat_grp.add_argument("--selection_method", type=str, default="correlation",
                          choices=["correlation", "variance", "f_score", "sdvs", "nist"],
                          help="Wavelength selection method (when --feature_method=wavelength_selection)")
    feat_grp.add_argument("--n_selected_wavelengths", type=int, default=500,
                          help="Wavelengths to select (default 500)")
    feat_grp.add_argument("--n_selected_min", type=int, default=100,
                          help="Min selected wavelengths for optimisation")
    feat_grp.add_argument("--n_selected_max", type=int, default=1000,
                          help="Max selected wavelengths for optimisation")
    feat_grp.add_argument("--optimize_features", action="store_true",
                          help="Include feature method in optimisation")
    feat_grp.add_argument("--optimize_preprocess", action="store_true",
                          help="Include preprocessing params in optimisation")
    feat_grp.add_argument("--two_stage", action="store_true",
                          help="Two-stage optimisation: Stage1=preprocessing+PCA, Stage2=model")
    feat_grp.add_argument("--subsample_ratio", type=float, default=0.3,
                          help="Subsample ratio for preprocessing optimisation (default 0.3)")
    feat_grp.add_argument("--cr_clean_lines", action="store_true",
                          help="Use reduced Cr NIST line set for ann/ann_hybrid (ML-019)")
    feat_grp.add_argument("--cr_model", type=str, default="ridge",
                          choices=["ridge", "pls"],
                          help="Cr model in ann_hybrid: ridge (default) or pls (ML-020)")

    # ── Evaluation arguments ─────────────────────────────────────────────────
    eval_grp = parser.add_argument_group("Evaluation arguments")
    eval_grp.add_argument("--cv", type=int, default=5,
                          help="Cross-validation folds (default 5)")
    eval_grp.add_argument("--seed", type=int, default=42,
                          help="Random seed (default 42)")
    eval_grp.add_argument("--n_per_target", type=int, default=0,
                          help="GroupKFold aggregation: spectra per target sample "
                               "(set to 50 for LIBS; 0 = disabled)")
    eval_grp.add_argument("--mc_dropout", action="store_true",
                          help="MC-Dropout uncertainty + temperature-scaling calibration "
                               "(--task classify --model cnn only)")
    eval_grp.add_argument("--n_mc_samples", type=int, default=50,
                          help="MC-Dropout forward passes (default 50)")
    eval_grp.add_argument("--explain", action="store_true",
                          help="SHAP attribution overlay on CNN classifier "
                               "(--task classify --model cnn only)")
    eval_grp.add_argument("--explain_samples", type=int, default=50,
                          help="SHAP explain samples (default 50)")

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
    ens_grp.add_argument("--logit_transform", action="store_true",
                         help="Logit-transform low-concentration targets before fitting")
    ens_grp.add_argument("--logit_elements", type=str, nargs="+",
                         default=["C", "Si", "Mo", "Cu"],
                         help="Elements to logit-transform (default: C Si Mo Cu)")
    ens_grp.add_argument("--cnn_weighted_loss", action="store_true",
                         help="Per-element weighted MSE in CNN Phase 1 training")
    ens_grp.add_argument("--cnn_augment", action="store_true",
                         help="Spectral augmentation during CNN Phase 2 training")
    ens_grp.add_argument("--normalize_sum100", action="store_true",
                         help="Post-hoc closure constraint: rescale Σ(wt%%)=100")

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

    # Fix randomness for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Route to task-specific handler
    if args.task == "classify":
        run_classify(args)
        return

    if args.task == "temporal":
        run_temporal(args)
        return

    if args.task == "regress" and _is_mesbah_cap(args.train, args.target):
        run_cap_regress(args)
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
