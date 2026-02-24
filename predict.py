"""Standalone inference script for OES/LIBS spectral analysis.

Usage:
    python predict.py --model outputs/models/ridge_model.joblib \
                      --data data/test_dataset_RAW.csv \
                      --out predictions.csv

Set LOGIT_TRANSFORM = True below if the model was trained with --logit_transform.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ---- Configuration -------------------------------------------------------
# Set to True if the model was trained with --logit_transform
LOGIT_TRANSFORM = False
# Elements that were logit-transformed during training
LOGIT_ELEMENTS = ["C", "Si", "Mo", "Cu"]
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="OES/LIBS Spectral Analysis — Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model (.joblib)")
    parser.add_argument("--data", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--out", type=str, default="predictions.csv", help="Output CSV path")
    parser.add_argument("--n_wavelengths", type=int, default=40002, help="Number of wavelength columns")
    parser.add_argument("--target_cols", type=str, nargs="+", default=None,
                        help="Target column names (used to identify feature columns)")
    parser.add_argument("--baseline", type=str, default="als", choices=["als", "none"])
    parser.add_argument("--normalize", type=str, default="snv", choices=["snv", "minmax", "l2", "none"])
    parser.add_argument("--denoise", type=str, default="savgol", choices=["savgol", "none"])
    parser.add_argument("--baseline_lam", type=float, default=4.16e5)
    parser.add_argument("--baseline_p", type=float, default=0.026)
    parser.add_argument("--savgol_window", type=int, default=13)
    parser.add_argument("--savgol_polyorder", type=int, default=4)
    parser.add_argument("--n_per_target", type=int, default=0,
                        help="If >0, average predictions over blocks of N spectra per target")
    args = parser.parse_args()

    from src.data_loader import SpectralDataset
    from src.preprocessing import Preprocessor
    import joblib

    # Load data
    print(f"Loading data from {args.data}...")
    dataset = SpectralDataset.from_csv(
        args.data,
        n_wavelengths=args.n_wavelengths,
        target_cols=args.target_cols,
    )
    print(f"  {dataset}")

    # Preprocess
    preprocessor = Preprocessor(
        baseline=args.baseline,
        normalize=args.normalize,
        denoise=args.denoise,
        baseline_lam=args.baseline_lam,
        baseline_p=args.baseline_p,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
    )
    X = preprocessor.fit_transform(dataset.spectra)

    # Load model (includes its own feature extractor / pipeline)
    print(f"Loading model from {args.model}...")
    model = joblib.load(args.model)

    # Predict
    predictions = model.predict(X)

    # Per-target aggregation: average N spectra per target block
    if args.n_per_target > 0:
        from src.evaluation import aggregate_per_target
        predictions = aggregate_per_target(predictions, args.n_per_target)
        print(f"Aggregated predictions: {predictions.shape} ({args.n_per_target} spectra/target)")

    # Apply inverse logit transform if the model was trained with --logit_transform
    if LOGIT_TRANSFORM:
        from src.target_transform import LogitTargetTransformer
        target_names = dataset.target_names
        logit_transformer = LogitTargetTransformer(elements=LOGIT_ELEMENTS)
        # fit() just sets the transform mask; no data needed
        logit_transformer.fit(np.zeros((1, len(target_names))), target_names=target_names)
        predictions = logit_transformer.inverse_transform(predictions)

    # Clip to valid concentration range
    predictions = np.maximum(predictions, 0.0)

    # Save
    target_names = dataset.target_names if dataset.target_names else [f"Target_{i}" for i in range(predictions.shape[1] if predictions.ndim > 1 else 1)]
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    out_df = pd.DataFrame(predictions, columns=target_names)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Predictions saved to {out_path}")
    print(out_df.describe())


if __name__ == "__main__":
    main()
