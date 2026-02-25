"""Plot PCA temporal embedding trajectory for BOSCH OES data.

Loads BOSCH NetCDF data, computes PCA(20) embedding per time step,
and plots the 2D trajectory (PC1 vs PC2) coloured by time index.
Saves to outputs/temporal_pca.png.

Usage:
    python scripts/plot_temporal_pca.py --data data/bosch_oes/
    python scripts/plot_temporal_pca.py --data data/bosch_oes/ --n_components 20 --out outputs/temporal_pca.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(
        description="BOSCH OES PCA temporal trajectory plot"
    )
    parser.add_argument("--data", type=str, default="data/bosch_oes/",
                        help="Directory containing BOSCH NetCDF files")
    parser.add_argument("--n_components", type=int, default=20,
                        help="PCA components for temporal embedding (default 20)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path (default: outputs/temporal_pca.png)")
    parser.add_argument("--subsample", type=int, default=500,
                        help="Plot every N-th time step for clarity (default 500)")
    args = parser.parse_args()

    from src.data_loader import load_bosch_oes
    from src.temporal import compute_temporal_embedding

    # Load data
    print(f"Loading BOSCH OES data from {args.data}...")
    data = load_bosch_oes(args.data)
    spectra = data["spectra"]       # (T, 3648)
    timestamps = data["timestamps"]  # (T,)
    print(f"  Loaded: {spectra.shape} spectra, wavelengths={len(data['wavelengths'])} channels")
    print(f"  Day: {data['day_file']}, Wafer: {data['wafer_key']}")
    print(f"  Duration: {(timestamps[-1] - timestamps[0]):.1f} s at ~{1 / np.diff(timestamps).mean():.1f} Hz")

    # Compute PCA temporal embedding
    print(f"Computing PCA({args.n_components}) temporal embedding...")
    embedding, pca = compute_temporal_embedding(spectra, n_components=args.n_components)
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Explained variance (first 2 PCs): {pca.explained_variance_ratio_[:2].sum():.1%}")

    # Subsample for plotting clarity
    step = max(1, args.subsample)
    idx = np.arange(0, len(embedding), step)
    emb_sub = embedding[idx]
    t_sub = timestamps[idx] - timestamps[0]  # relative seconds from start

    # Plot PC1 vs PC2 trajectory coloured by time
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: 2D trajectory (PC1 vs PC2) ---
    ax = axes[0]
    sc = ax.scatter(emb_sub[:, 0], emb_sub[:, 1],
                    c=t_sub, cmap="viridis", s=6, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Time (s)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("PCA Temporal Trajectory (PC1 vs PC2)")
    ax.grid(True, alpha=0.3)

    # --- Right: PC1, PC2, PC3 over time ---
    ax2 = axes[1]
    t_plot = (timestamps - timestamps[0])  # full relative time
    for pc_idx, color, label in zip([0, 1, 2], ["steelblue", "tomato", "forestgreen"],
                                    ["PC1", "PC2", "PC3"]):
        if pc_idx < embedding.shape[1]:
            ax2.plot(t_plot[::step], embedding[::step, pc_idx],
                     color=color, lw=0.6, label=label, alpha=0.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("PCA Score")
    ax2.set_title("PCA Scores over Time")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"BOSCH Plasma Etching OES — Temporal PCA Embedding\n"
        f"{data['day_file']} / {data['wafer_key']}  |  "
        f"T={len(spectra)} steps, n_comp={args.n_components}",
        fontsize=10,
    )
    plt.tight_layout()

    out_path = Path(args.out) if args.out else Path("outputs/temporal_pca.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved temporal PCA plot to {out_path}")


if __name__ == "__main__":
    main()
