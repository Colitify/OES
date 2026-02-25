"""DTW K-means discharge phase clustering of BOSCH OES temporal embedding.

Loads BOSCH NetCDF data, computes PCA(20) temporal embedding, runs K-means
clustering (with DTW or euclidean metric), and saves:
  outputs/discharge_clusters.png — time series colored by cluster label

Also plots an elbow curve (inertia vs k) for k=2..6 to aid selection.

Usage:
    python scripts/plot_clusters.py --data data/bosch_oes/ --k 4
    python scripts/plot_clusters.py --data data/bosch_oes/ --k 4 --metric euclidean
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
        description="DTW K-means clustering of BOSCH OES discharge phases"
    )
    parser.add_argument("--data", type=str, default="data/bosch_oes/",
                        help="Directory containing BOSCH NetCDF files")
    parser.add_argument("--k", type=int, default=4,
                        help="Number of clusters (default 4)")
    parser.add_argument("--metric", type=str, default="euclidean",
                        choices=["euclidean", "dtw"],
                        help="Distance metric for K-means (default euclidean)")
    parser.add_argument("--n_components", type=int, default=20,
                        help="PCA components for temporal embedding (default 20)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path (default: outputs/discharge_clusters.png)")
    parser.add_argument("--subsample", type=int, default=10,
                        help="Subsample every N-th time step for plotting (default 10)")
    args = parser.parse_args()

    from src.data_loader import load_bosch_oes
    from src.temporal import cluster_discharge_phases, compute_temporal_embedding

    # Load data
    print(f"Loading BOSCH OES data from {args.data}...")
    data = load_bosch_oes(args.data)
    spectra = data["spectra"]       # (T, 3648)
    timestamps = data["timestamps"]  # (T,)
    wavelengths = data["wavelengths"]
    print(f"  Loaded: {spectra.shape} spectra")

    # Compute PCA temporal embedding
    print(f"Computing PCA({args.n_components}) temporal embedding...")
    embedding, pca = compute_temporal_embedding(spectra, n_components=args.n_components)
    print(f"  Embedding: {embedding.shape}")

    # Compute inertia for k=2..6 (elbow plot)
    print("Computing elbow curve (k=2..6)...")
    k_range = list(range(2, 7))
    inertias = []
    for k_try in k_range:
        _, _, inertia_k = cluster_discharge_phases(embedding, k=k_try, metric="euclidean")
        inertias.append(inertia_k)
        print(f"  k={k_try}: inertia={inertia_k:.0f}")

    # Verify monotonic decrease
    is_monotone = all(inertias[i] >= inertias[i + 1] for i in range(len(inertias) - 1))
    print(f"  Monotonically decreasing: {is_monotone}")

    # Main clustering
    print(f"\nClustering with k={args.k}, metric={args.metric}...")
    labels, centroids, inertia = cluster_discharge_phases(
        embedding, k=args.k, metric=args.metric
    )
    print(f"  Inertia: {inertia:.2f}")
    print(f"  Cluster sizes: {[int((labels == c).sum()) for c in range(args.k)]}")

    # Check interpretability: highest-variance emission channel intensity per cluster
    high_var_ch = int(spectra.std(axis=0).argmax())
    high_var_wl = wavelengths[high_var_ch]
    ch_intensity = spectra[:, high_var_ch].astype(float)
    cluster_means = [ch_intensity[labels == c].mean() for c in range(args.k)]
    max_mean = max(cluster_means)
    min_mean = min(cluster_means)
    ratio = max_mean / (min_mean + 1e-6)
    print(f"\n  Emission channel: {high_var_wl:.1f} nm (ch {high_var_ch})")
    for c in range(args.k):
        print(f"    Cluster {c}: mean intensity = {cluster_means[c]:.0f}")
    print(f"  Max/min cluster intensity ratio: {ratio:.2f}x (need >= 2x)")
    status = "PASS" if ratio >= 2.0 else "FAIL"
    print(f"  2x criterion: {status}")

    # --- Build figure ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    colors = plt.cm.tab10.colors
    t_rel = (timestamps - timestamps[0])  # relative seconds
    step = max(1, args.subsample)

    # 1. Elbow plot
    ax = axes[0]
    ax.plot(k_range, inertias, "o-", color="steelblue", lw=2)
    ax.set_xlabel("Number of Clusters k")
    ax.set_ylabel("Inertia (within-cluster SSE)")
    ax.set_title("Elbow Curve: Inertia vs k (PCA Embedding, Euclidean)")
    ax.grid(True, alpha=0.3)
    ax.axvline(args.k, color="tomato", linestyle="--", alpha=0.7,
               label=f"Selected k={args.k}")
    ax.legend()

    # 2. PC1 over time coloured by cluster
    ax2 = axes[1]
    for c in range(args.k):
        mask = labels == c
        t_c = t_rel[mask]
        pc1_c = embedding[mask, 0]
        ax2.scatter(t_c, pc1_c, s=4, color=colors[c % len(colors)],
                    alpha=0.5, label=f"Cluster {c} (n={mask.sum()})")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("PC1 Score")
    ax2.set_title(f"Discharge Phase Clusters (k={args.k}) — PC1 over Time")
    ax2.legend(fontsize=8, ncol=args.k)
    ax2.grid(True, alpha=0.3)

    # 3. High-variance emission line intensity over time coloured by cluster
    ax3 = axes[2]
    for c in range(args.k):
        mask = labels == c
        t_c = t_rel[mask][::step]
        int_c = ch_intensity[mask][::step]
        ax3.scatter(t_c, int_c, s=4, color=colors[c % len(colors)],
                    alpha=0.5, label=f"Cluster {c}")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel(f"Intensity at {high_var_wl:.1f} nm (counts)")
    ax3.set_title(f"Emission Intensity over Time (Highest-Variance Line: {high_var_wl:.1f} nm)")
    ax3.legend(fontsize=8, ncol=args.k)
    ax3.grid(True, alpha=0.3)

    plt.suptitle(
        f"BOSCH Plasma Etching OES — Discharge Phase Clustering\n"
        f"{data['day_file']} / {data['wafer_key']}  |  "
        f"k={args.k}, metric={args.metric}",
        fontsize=10,
    )
    plt.tight_layout()

    out_path = Path(args.out) if args.out else Path("outputs/discharge_clusters.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved discharge clusters plot to {out_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Elbow monotonic: {is_monotone}")
    print(f"2x emission ratio: {ratio:.2f}x → {status}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
