"""OES-022: Ablation study — raw PCA vs plasma descriptor vs NIST windows vs combined.

Runs 4 SVM(LinearSVC) variants with different feature inputs on the LIBS Benchmark
5-fold stratified CV. Saves results to results/ablation.csv and outputs/ablation_bar.png.

Usage:
    python scripts/ablation.py [--data data/libs_benchmark/] [--cv 5] [--seed 42]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_libs_benchmark
from src.features import PlasmaDescriptorExtractor, select_wavelengths_plasma


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def build_features(X, wl, variant: str, fitted=None):
    """Build feature matrix for the given variant.

    variant:
        A  — PCA(50) raw spectrum
        B  — PlasmaDescriptor only
        C  — NIST window intensities (select_wavelengths_plasma) + PCA
        D  — Combined: PCA(50) + descriptor + NIST-PCA

    Returns (X_feat, fit_artifacts) where fit_artifacts can be passed
    as `fitted` on the transform call to re-use the same fitted objects.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if variant == "A":
        # Raw spectrum → StandardScaler → PCA(50)
        if fitted is None:
            sc = StandardScaler(); pca = PCA(n_components=50, random_state=0)
            X_s = sc.fit_transform(X)
            X_f = pca.fit_transform(X_s)
            return X_f, (sc, pca)
        sc, pca = fitted
        return pca.transform(sc.transform(X)), fitted

    elif variant == "B":
        # PlasmaDescriptor only
        if fitted is None:
            fe = PlasmaDescriptorExtractor()
            X_d = fe.fit_transform(X, wl)
            sc = StandardScaler()
            X_f = sc.fit_transform(X_d)
            return X_f, (fe, sc)
        fe, sc = fitted
        X_d = fe.transform(X)
        return sc.transform(X_d), fitted

    elif variant == "C":
        # NIST windows → StandardScaler → PCA(50)
        mask = select_wavelengths_plasma(wl)
        X_n = X[:, mask]
        if fitted is None:
            sc = StandardScaler(); pca = PCA(n_components=min(50, X_n.shape[1]), random_state=0)
            X_s = sc.fit_transform(X_n)
            X_f = pca.fit_transform(X_s)
            return X_f, (mask, sc, pca)
        _, sc, pca = fitted
        return pca.transform(sc.transform(X_n)), fitted

    elif variant == "D":
        # Combined: PCA(50) raw + descriptor (standardised) + NIST-PCA(50)
        # A: PCA(50)
        X_a, arts_a = build_features(X, wl, "A", fitted[0] if fitted else None)
        # B: descriptor
        X_b, arts_b = build_features(X, wl, "B", fitted[1] if fitted else None)
        # C: NIST-PCA
        X_c, arts_c = build_features(X, wl, "C", fitted[2] if fitted else None)
        X_f = np.hstack([X_a, X_b, X_c])
        if fitted is None:
            return X_f, (arts_a, arts_b, arts_c)
        return X_f, fitted
    else:
        raise ValueError(f"Unknown variant: {variant}")


def run_cv(X, y, variant: str, wl, cv: int, seed: int):
    """Stratified k-fold CV for one variant. Returns metrics dict."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.svm import LinearSVC

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    micro_scores, macro_scores = [], []
    t0 = time.time()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        X_tr_f, arts = build_features(X_tr, wl, variant)
        X_val_f, _ = build_features(X_val, wl, variant, arts)

        clf = LinearSVC(C=0.1, max_iter=3000, random_state=seed)
        clf.fit(X_tr_f, y_tr)
        y_pred = clf.predict(X_val_f)
        micro_scores.append(f1_score(y_val, y_pred, average="micro"))
        macro_scores.append(f1_score(y_val, y_pred, average="macro"))
        print(f"    fold {fold+1}/{cv}: micro_f1={micro_scores[-1]:.4f}")

    # Measure final feature size with one full fit
    X_f_full, _ = build_features(X, wl, variant)
    n_features = X_f_full.shape[1]
    fit_time = time.time() - t0

    return {
        "micro_f1": float(np.mean(micro_scores)),
        "macro_f1": float(np.mean(macro_scores)),
        "n_features": n_features,
        "fit_time_s": round(fit_time, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="OES-022: Feature ablation study")
    parser.add_argument("--data", default="data/libs_benchmark/")
    parser.add_argument("--cv", type=int, default=3, help="CV folds (default 3 for speed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics_out", default="results/metrics.json")
    args = parser.parse_args()

    print("=" * 60)
    print("OES-022: Ablation Study")
    print("=" * 60)

    data_dir = Path(args.data)
    results_dir = ROOT / "results"
    outputs_dir = ROOT / "outputs"
    results_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    print("\nLoading training data...")
    X, y, wl = load_libs_benchmark(str(data_dir), split="train")
    # Use a balanced 6000-sample subset for speed (500 per class)
    # to avoid the slow descriptor extraction on 48000 samples
    max_per_class = 500
    from collections import Counter
    counts = Counter(y.tolist())
    idx_keep = []
    rng = np.random.default_rng(args.seed)
    for cls in sorted(counts.keys()):
        idx_cls = np.where(y == cls)[0]
        n = min(max_per_class, len(idx_cls))
        idx_keep.extend(rng.choice(idx_cls, n, replace=False).tolist())
    idx_keep = np.array(sorted(idx_keep))
    X_sub, y_sub = X[idx_keep], y[idx_keep]
    print(f"Using {len(X_sub)} balanced training samples ({max_per_class}/class max)")

    variants = {
        "A": "Raw PCA(50)",
        "B": "Plasma descriptor",
        "C": "NIST windows + PCA",
        "D": "Combined (A+B+C)",
    }

    rows = []
    best_micro = -1.0
    best_variant = "A"
    for code, name in variants.items():
        print(f"\nVariant {code}: {name}")
        metrics = run_cv(X_sub, y_sub, code, wl, args.cv, args.seed)
        row = {"variant": code, "name": name, **metrics}
        rows.append(row)
        print(f"  → micro_f1={metrics['micro_f1']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
              f"n_features={metrics['n_features']}, fit_time={metrics['fit_time_s']}s")
        if metrics["micro_f1"] > best_micro:
            best_micro = metrics["micro_f1"]
            best_variant = code

    # Check acceptance: D >= A
    d_mf1 = next(r["micro_f1"] for r in rows if r["variant"] == "D")
    a_mf1 = next(r["micro_f1"] for r in rows if r["variant"] == "A")
    print(f"\nCombined(D) vs PCA-only(A): {d_mf1:.4f} vs {a_mf1:.4f} → "
          f"{'OK' if d_mf1 >= a_mf1 else 'WARN: combined worse than PCA-only'}")

    # Save CSV
    import csv
    csv_path = results_dir / "ablation.csv"
    fieldnames = ["variant", "name", "micro_f1", "macro_f1", "n_features", "fit_time_s"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {csv_path}")

    # Save bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [f"{r['variant']}: {r['name']}" for r in rows]
        micro_vals = [r["micro_f1"] for r in rows]
        macro_vals = [r["macro_f1"] for r in rows]

        x = np.arange(len(rows))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        bars1 = ax.bar(x - width / 2, micro_vals, width, label="micro_f1", color="steelblue")
        bars2 = ax.bar(x + width / 2, macro_vals, width, label="macro_f1", color="coral")
        ax.set_ylabel("F1 Score")
        ax.set_title("Ablation Study: Feature Block Contribution (LIBS Benchmark, LinearSVC)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 1.0)
        ax.legend()
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        png_path = outputs_dir / "ablation_bar.png"
        fig.savefig(str(png_path), dpi=120)
        plt.close(fig)
        print(f"Saved {png_path}")
    except Exception as exc:
        print(f"Warning: could not save ablation bar chart: {exc}")

    # Update results/metrics.json with best variant micro_f1
    metrics_out = Path(args.metrics_out)
    if metrics_out.exists():
        payload = json.loads(metrics_out.read_text(encoding="utf-8"))
    else:
        payload = {}
    payload["ablation_best_variant"] = best_variant
    payload["ablation_best_micro_f1"] = best_micro
    payload["ablation_rows"] = rows
    metrics_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated {metrics_out} with ablation results")

    print(f"\nOES-022 COMPLETE. Best variant: {best_variant} ({best_micro:.4f} micro_f1)")


if __name__ == "__main__":
    main()
