"""Standalone Mesbah Lab CAP T_rot / T_vib regression evaluator.

Loads dat_train.csv, runs 5-fold CV for both T_rot and T_vib with an ANN
(MLPRegressor in BaggingRegressor), prints a RMSE table, and writes
results/metrics_cap.json.

Usage:
    python scripts/evaluate_cap.py
    python scripts/evaluate_cap.py --train data/mesbah_cap/dat_train.csv --cv 5
    python scripts/evaluate_cap.py --metrics_out results/metrics_cap.json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore")


def evaluate_cap(train_path: str, cv: int = 5, seed: int = 42, n_ens: int = 16) -> dict:
    """Run 5-fold CV regression for T_rot and T_vib on Mesbah CAP data.

    Args:
        train_path: Path to dat_train.csv
        cv: Number of cross-validation folds
        seed: Random seed
        n_ens: Number of ANN ensemble members (BaggingRegressor n_estimators)

    Returns:
        dict mapping target name -> {'rmse': float, 'rmse_std': float}
    """
    from sklearn.ensemble import BaggingRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    from src.data_loader import load_mesbah_cap

    targets = ["T_rot", "T_vib"]
    results = {}

    for target in targets:
        X, y, _ = load_mesbah_cap(train_path, target=target)
        X_sc = StandardScaler().fit_transform(X)

        base_ann = MLPRegressor(
            hidden_layer_sizes=(64,),
            activation="logistic",
            solver="lbfgs",
            alpha=0.01,
            max_iter=2000,
            random_state=seed,
        )
        model = BaggingRegressor(
            estimator=base_ann,
            n_estimators=n_ens,
            bootstrap=True,
            n_jobs=-1,
            random_state=seed,
        )

        scores = cross_val_score(
            model, X_sc, y,
            cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1
        )
        rmse = float(-scores.mean())
        rmse_std = float(scores.std())
        results[target] = {"rmse": rmse, "rmse_std": rmse_std}

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Mesbah Lab CAP T_rot/T_vib regression benchmark"
    )
    parser.add_argument(
        "--train", type=str,
        default="data/mesbah_cap/dat_train.csv",
        help="Path to dat_train.csv (default: data/mesbah_cap/dat_train.csv)"
    )
    parser.add_argument("--cv", type=int, default=5,
                        help="Cross-validation folds (default 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_ens", type=int, default=16,
                        help="ANN ensemble size (default 16)")
    parser.add_argument(
        "--metrics_out", type=str,
        default="results/metrics_cap.json",
        help="Output metrics JSON path (default: results/metrics_cap.json)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Mesbah Lab CAP Regression Benchmark")
    print(f"  Data: {args.train}")
    print(f"  CV folds: {args.cv}  |  Ensemble size: {args.n_ens}")
    print("=" * 60)

    results = evaluate_cap(args.train, cv=args.cv, seed=args.seed, n_ens=args.n_ens)

    # Print table
    thresholds = {"T_rot": 50.0, "T_vib": 200.0}
    print(f"\n{'Target':<12} {'RMSE (K)':>10} {'± std':>8}  {'Threshold':>12}  {'Status':>6}")
    print("-" * 55)
    for tgt, res in results.items():
        thr = thresholds.get(tgt, float("inf"))
        status = "PASS" if res["rmse"] <= thr else "FAIL"
        print(f"{tgt:<12} {res['rmse']:>10.2f} {res['rmse_std']:>8.2f}  {thr:>12.1f}  {status:>6}")

    # Write metrics
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
    print(f"\nMetrics saved to {out_path}")

    # Exit code: 0 if all guardrails pass
    all_pass = all(
        results[t]["rmse"] <= thresholds[t]
        for t in thresholds if t in results
    )
    print(f"\nOverall: {'ALL PASS' if all_pass else 'GUARDRAIL FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
