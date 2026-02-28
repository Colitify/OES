"""SHAP overlay plot for OES regression model on Mesbah CAP data.

Trains an SVM regressor on the CAP dataset (T_rot), computes SHAP values
via KernelExplainer, and plots mean |SHAP| per wavelength overlaid with
vertical lines at PLASMA_EMISSION_LINES positions.

Usage:
    python scripts/plot_shap.py --data data/mesbah_cap/dat_train.csv --target T_rot

Output:
    outputs/shap_overlay.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import PLASMA_EMISSION_LINES, PLASMA_DELTA_NM


def main():
    parser = argparse.ArgumentParser(description="SHAP overlay plot for OES regression")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to Mesbah CAP CSV (e.g. data/mesbah_cap/dat_train.csv)")
    parser.add_argument("--target", type=str, default="T_rot",
                        help="Regression target (default: T_rot)")
    parser.add_argument("--n_background", type=int, default=50,
                        help="Number of background samples for KernelExplainer (default 50)")
    parser.add_argument("--n_explain", type=int, default=30,
                        help="Number of samples to explain (default 30)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path (default: outputs/shap_overlay.png)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import shap
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from src.data_loader import load_mesbah_cap

    # --- Load data ---
    print(f"Loading Mesbah CAP data ({args.target})...")
    X, y, wavelengths = load_mesbah_cap(args.data, target=args.target)
    print(f"  Loaded: X={X.shape}, y range=[{y.min():.1f}, {y.max():.1f}]")

    # --- Train SVR model ---
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    svr = SVR(kernel="rbf", C=100, gamma="scale")
    svr.fit(X_sc, y)
    print(f"  SVR trained (R² on train: {svr.score(X_sc, y):.4f})")

    # --- Sample background and explain sets ---
    rng = np.random.default_rng(args.seed)
    bg_idx = rng.choice(len(X_sc), size=min(args.n_background, len(X_sc)), replace=False)
    exp_idx = rng.choice(len(X_sc), size=min(args.n_explain, len(X_sc)), replace=False)

    X_bg = X_sc[bg_idx]
    X_exp = X_sc[exp_idx]

    # --- Compute SHAP values ---
    print(f"Computing SHAP values (background={len(X_bg)}, explain={len(X_exp)})...")
    explainer = shap.KernelExplainer(svr.predict, X_bg)
    shap_values = explainer.shap_values(X_exp, nsamples=100)
    # shap_values: (n_explain, n_features)
    print(f"  SHAP values shape: {shap_values.shape}")

    # --- Plot mean |SHAP| per wavelength ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)  # (n_features,)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(wavelengths, mean_abs_shap, width=(wavelengths[-1] - wavelengths[0]) / len(wavelengths),
           color="steelblue", alpha=0.7, label=f"Mean |SHAP| ({args.target})")

    # Overlay PLASMA_EMISSION_LINES vertical lines
    colors = plt.cm.tab10.colors
    for si, (sp, lines) in enumerate(PLASMA_EMISSION_LINES.items()):
        col = colors[si % len(colors)]
        for li, line_nm in enumerate(lines):
            label = sp if li == 0 else None
            ax.axvline(line_nm, color=col, alpha=0.5, lw=0.8, linestyle="--", label=label)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Attribution for {args.target} Regression with Plasma Emission Line Overlay")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    out_path = Path(args.out) if args.out else Path("outputs/shap_overlay.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SHAP overlay to {out_path}")


if __name__ == "__main__":
    main()
