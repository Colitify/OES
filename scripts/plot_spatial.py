"""Spatial visualisation script — wafer heatmaps and uniformity bar charts.

Usage:
    python scripts/plot_spatial.py --data data/bosch_oes/ --metric oxide_etch
    python scripts/plot_spatial.py --data data/bosch_oes/ --metric si_etch --wafer 2024-07-02_01
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_wafer_spatial
from src.spatial import compute_wafer_uniformity, interpolate_wafer_map


def plot_wafer_heatmap(df_wafer: pd.DataFrame, metric_col: str, title: str, ax=None):
    """Plot a single wafer heatmap with contourf + scatter overlay."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    x = df_wafer["X"].values / 1000  # μm → mm
    y = df_wafer["Y"].values / 1000
    values = df_wafer[metric_col].values

    grid_x, grid_y, grid_z = interpolate_wafer_map(
        df_wafer["X"].values, df_wafer["Y"].values, values, grid_size=200
    )
    grid_x_mm = grid_x / 1000
    grid_y_mm = grid_y / 1000

    cf = ax.contourf(grid_x_mm, grid_y_mm, grid_z, levels=20, cmap="RdYlBu_r")
    ax.scatter(x, y, c=values, edgecolors="k", linewidths=0.5, s=30,
               cmap="RdYlBu_r", vmin=grid_z.min(), vmax=grid_z.max(), zorder=5)
    plt.colorbar(cf, ax=ax, label=metric_col)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title)
    ax.set_aspect("equal")
    return ax


def plot_uniformity_bars(uniformity_df: pd.DataFrame, metric_col: str, ax=None):
    """Plot uniformity bar chart across wafers."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    keys = uniformity_df["experiment_key"]
    pct = uniformity_df["uniformity_pct"]

    ax.bar(range(len(keys)), pct, color="steelblue", edgecolor="navy")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Uniformity (%)")
    ax.set_title(f"Wafer Uniformity — {metric_col}")
    ax.axhline(y=pct.mean(), color="red", linestyle="--", label=f"mean = {pct.mean():.2f}%")
    ax.legend()
    return ax


def main():
    parser = argparse.ArgumentParser(
        description="Wafer spatial analysis visualisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Path to bosch_oes data directory")
    parser.add_argument("--metric", type=str, default="oxide_etch",
                        help="Etch metric column to visualise")
    parser.add_argument("--wafer", type=str, default=None,
                        help="Specific experiment_key for single-wafer heatmap")
    parser.add_argument("--points", type=int, default=89, choices=[9, 89],
                        help="Measurement point layout (89 or 9)")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Output directory for figures")
    args = parser.parse_args()

    data_dir = Path(args.data)
    csv_name = f"Si_Oxide_etch_{args.points}_points.csv"
    csv_path = data_dir / csv_name
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_wafer_spatial(csv_path, points=args.points)

    # --- Single wafer heatmap ---
    if args.wafer:
        wafer_keys = [args.wafer]
    else:
        wafer_keys = df["experiment_key"].unique()[:4]  # first 4 wafers

    for key in wafer_keys:
        df_w = df[df["experiment_key"] == key]
        if df_w.empty:
            print(f"No data for experiment_key='{key}', skipping.")
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_wafer_heatmap(df_w, args.metric, f"Wafer {key} — {args.metric}", ax=ax)
        fig.tight_layout()
        fname = out_dir / f"wafer_heatmap_{key.replace('-', '')}_{args.metric}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved: {fname}")

    # --- Multi-wafer uniformity bar chart ---
    uniformity = compute_wafer_uniformity(df, args.metric)
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_uniformity_bars(uniformity, args.metric, ax=ax)
    fig.tight_layout()
    fname = out_dir / f"uniformity_{args.metric}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


if __name__ == "__main__":
    main()
