"""SHAP overlay plot for Conv1DClassifier on spectral data.

Loads a saved model checkpoint (outputs/best_model.pt), computes SHAP values
via GradientExplainer on a subset of training spectra, and plots mean |SHAP|
per wavelength overlaid with vertical lines at PLASMA_EMISSION_LINES positions.

Usage:
    python scripts/plot_shap.py --model outputs/best_model.pt --data data/libs_benchmark/train.h5

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
    parser = argparse.ArgumentParser(description="SHAP overlay plot for Conv1DClassifier")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved model checkpoint (.pt dict)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to train.h5 or its directory")
    parser.add_argument("--n_background", type=int, default=50,
                        help="Number of background samples for GradientExplainer (default 50)")
    parser.add_argument("--n_explain", type=int, default=20,
                        help="Number of samples to explain (default 20)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path (default: outputs/shap_overlay.png)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from src.models.deep_learning import Conv1DClassifier
    from src.evaluation import compute_shap_spectrum
    from src.data_loader import load_libs_benchmark

    # --- Load checkpoint ---
    ckpt_path = Path(args.model)
    if not ckpt_path.exists():
        print(f"ERROR: Model checkpoint not found at {ckpt_path}")
        print("Run: python main.py --task classify --model cnn --explain ... first.")
        sys.exit(1)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    n_classes = ckpt["n_classes"]
    n_pca = ckpt["n_pca"]
    best_params = ckpt["best_params"]
    pca_components = ckpt["pca_components"]    # (n_pca, n_wavelengths)
    pca_mean = ckpt["pca_mean"]               # (n_wavelengths,)
    scaler_mean = ckpt["scaler_mean"]         # (n_wavelengths,)
    scaler_scale = ckpt["scaler_scale"]       # (n_wavelengths,)
    wavelengths = ckpt.get("wavelengths", None)  # (n_wavelengths,)

    # Reconstruct model
    model = Conv1DClassifier(
        n_classes=n_classes,
        n_filters=best_params["n_filters"],
        kernel_size=best_params["kernel_size"],
        dropout=best_params["dropout"],
        lr=best_params.get("lr", 1e-3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --- Load data ---
    data_path = Path(args.data)
    data_dir = str(data_path.parent) if data_path.suffix in (".h5", ".hdf5") else str(data_path)
    print(f"Loading data from {data_dir}...")
    X_raw, y_raw, wl = load_libs_benchmark(data_dir, split="train")
    if wavelengths is None:
        wavelengths = wl
    print(f"  Loaded: X={X_raw.shape}, wavelengths={wavelengths.shape}")

    # --- Apply preprocessing (scaler + PCA) ---
    X_scaled = (X_raw.astype(np.float64) - scaler_mean) / scaler_scale
    X_pca = (X_scaled - pca_mean) @ pca_components.T   # (n_samples, n_pca)
    X_pca = X_pca.astype(np.float32)

    # --- Sample background and explain sets ---
    rng = np.random.default_rng(args.seed)
    bg_idx = rng.choice(len(X_pca), size=min(args.n_background, len(X_pca)), replace=False)
    exp_idx = rng.choice(len(X_pca), size=min(args.n_explain, len(X_pca)), replace=False)

    X_bg = X_pca[bg_idx]
    X_exp = X_pca[exp_idx]
    y_exp = y_raw[exp_idx]

    # --- Compute SHAP values ---
    class _PCAProxy:
        """Minimal PCA-like object for compute_shap_spectrum projection."""
        def __init__(self, components, mean):
            self.components_ = components  # (n_pca, n_wavelengths)

    pca_proxy = _PCAProxy(pca_components, pca_mean)
    print(f"Computing SHAP values (background={len(X_bg)}, explain={len(X_exp)})...")
    shap_vals = compute_shap_spectrum(model, X_bg, X_exp, wavelengths=wavelengths, pca=pca_proxy)
    # shap_vals: (n_explain, n_wavelengths, n_classes)
    print(f"  SHAP values shape: {shap_vals.shape}")

    # --- Plot mean |SHAP| per wavelength for the dominant class ---
    mean_abs_shap_per_class = np.abs(shap_vals).mean(axis=0)  # (n_wavelengths, n_classes)
    dominant_class = int(mean_abs_shap_per_class.max(axis=0).argmax())
    shap_1d = mean_abs_shap_per_class[:, dominant_class]       # (n_wavelengths,)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(wavelengths, shap_1d, color="steelblue", lw=0.6, label=f"Mean |SHAP| (class {dominant_class})")

    # Overlay PLASMA_EMISSION_LINES vertical lines
    colors = plt.cm.tab10.colors
    species_list = list(PLASMA_EMISSION_LINES.keys())
    for si, (sp, lines) in enumerate(PLASMA_EMISSION_LINES.items()):
        col = colors[si % len(colors)]
        for li, line_nm in enumerate(lines):
            label = sp if li == 0 else None
            ax.axvline(line_nm, color=col, alpha=0.5, lw=0.8, linestyle="--", label=label)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title("SHAP Spectrum Attribution with Plasma Emission Line Overlay")
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
