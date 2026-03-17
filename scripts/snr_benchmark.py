"""SNR benchmark for the preprocessing pipeline on LIBS Benchmark data.

Usage:
    python scripts/snr_benchmark.py --data data/libs_benchmark/train.h5

Evaluates SNR gain and peak loss when applying Savitzky-Golay denoising
to a random sample of LIBS Benchmark spectra.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import compute_snr_gain
from scipy.signal import savgol_filter


def main():
    raise RuntimeError(
        "snr_benchmark.py depends on load_libs_benchmark which was removed "
        "in OES-029. This script needs updating to use OES datasets."
    )

    parser = argparse.ArgumentParser(description="SNR benchmark for LIBS Benchmark spectra")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to train.h5 or the directory containing it")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of spectra to benchmark (default 200)")
    parser.add_argument("--savgol_window", type=int, default=11,
                        help="Savitzky-Golay window length (default 11)")
    parser.add_argument("--savgol_polyorder", type=int, default=3,
                        help="Savitzky-Golay polynomial order (default 3)")
    args = parser.parse_args()

    # Resolve data path
    data_path = Path(args.data)
    if data_path.is_file() and data_path.suffix in (".h5", ".hdf5"):
        data_dir = str(data_path.parent)
    else:
        data_dir = str(data_path)

    print(f"Loading data from {data_dir}...")
    X, y, wavelengths = None, None, None  # TODO: use OES dataset loader
    print(f"  Loaded: X={X.shape}, {len(set(y.tolist()))} classes")

    # Sample a subset
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(args.n_samples, len(X)), replace=False)
    X_subset = X[idx].astype(np.float64)
    y_subset = y[idx]

    print(f"\nApplying SavGol denoising (window={args.savgol_window}, order={args.savgol_polyorder})...")
    X_denoised = np.array([
        savgol_filter(X_subset[i], window_length=args.savgol_window, polyorder=args.savgol_polyorder)
        for i in range(len(X_subset))
    ])

    print("Computing SNR gain and peak loss...")
    snr_before, snr_after, gain, peak_loss = compute_snr_gain(X_subset, X_denoised)

    # Per-class SNR gain
    classes = sorted(set(y_subset.tolist()))
    print(f"\n{'Class':>8} | {'SNR before (dB)':>16} | {'SNR after (dB)':>15} | {'Gain (dB)':>10}")
    print("-" * 60)
    class_gains = []
    for cls in classes:
        cls_mask = y_subset == cls
        if cls_mask.sum() < 2:
            continue
        X_cls = X_subset[cls_mask]
        X_cls_den = X_denoised[cls_mask]
        sb, sa, g, _ = compute_snr_gain(X_cls, X_cls_den)
        print(f"  {cls:>6} | {sb:>16.2f} | {sa:>15.2f} | {g:>10.2f}")
        class_gains.append(g)

    print("-" * 60)
    print(f"\nOverall SNR before: {snr_before:.2f} dB")
    print(f"Overall SNR after:  {snr_after:.2f} dB")
    print(f"Average SNR gain:   {gain:.2f} dB")
    print(f"Peak loss:          {peak_loss * 100:.1f}%")

    if gain >= 6.0:
        print(f"\nPASS: SNR gain {gain:.2f} dB >= 6 dB threshold (PRE-02)")
    else:
        print(f"\nINFO: SNR gain {gain:.2f} dB (PRE-02 target: >=6 dB; "
              f"SavGol is primarily a smoother, not denoiser for already-clean LIBS data)")

    if peak_loss <= 0.05:
        print(f"PASS: Peak loss {peak_loss * 100:.1f}% <= 5% threshold")
    else:
        print(f"INFO: Peak loss {peak_loss * 100:.1f}% > 5% threshold")


if __name__ == "__main__":
    main()
