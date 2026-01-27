"""
Guardrail script for Ralph-style iteration.

Usage:
    python -m src.guardrail results/metrics.json

Behavior:
- Reads a machine-readable metrics.json produced by main.py
- Extracts primary metric (RMSE_mean, smaller is better)
- Appends one line to results/runs.csv
- Updates results/best.json if improved
- Exit code:
    0 => PASS (improved or initialized)
    1 => FAIL (no improvement)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def extract_rmse_mean(payload: Dict[str, Any]) -> Optional[float]:
    # 1) Prefer primary_metric.value
    pm = payload.get("primary_metric", {})
    if isinstance(pm, dict) and pm.get("name") == "RMSE_mean":
        v = pm.get("value")
        if isinstance(v, (int, float)):
            return float(v)

    # 2) Fallback to metrics._overall.RMSE_mean
    metrics = payload.get("metrics", {})
    if isinstance(metrics, dict):
        overall = metrics.get("_overall", {})
        if isinstance(overall, dict):
            v = overall.get("RMSE_mean")
            if isinstance(v, (int, float)):
                return float(v)

    return None


def append_runs_csv(runs_path: Path, payload: Dict[str, Any], rmse_mean: float) -> None:
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    header = "timestamp,rmse_mean,model,cv_folds,seed,git_sha\n"

    timestamp = payload.get("timestamp")
    model = payload.get("model")
    cv_folds = payload.get("cv_folds")
    seed = payload.get("seed")
    git_sha = payload.get("git_sha")

    line = f"{timestamp},{rmse_mean},{model},{cv_folds},{seed},{git_sha}\n"

    if not runs_path.exists():
        runs_path.write_text(header, encoding="utf-8")

    with runs_path.open("a", encoding="utf-8") as f:
        f.write(line)


def main():
    parser = argparse.ArgumentParser(description="Guardrail for RMSE_mean improvements.")
    parser.add_argument("metrics_json", type=str, help="Path to metrics.json")
    parser.add_argument("--best_path", type=str, default="results/best.json", help="Path to best.json")
    parser.add_argument("--runs_path", type=str, default="results/runs.csv", help="Path to runs.csv")
    parser.add_argument("--tol", type=float, default=0.0, help="Tolerance for improvement (minimize metric).")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json)
    if not metrics_path.exists():
        raise SystemExit(f"metrics.json not found: {metrics_path}")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rmse_mean = extract_rmse_mean(payload)
    if rmse_mean is None:
        raise SystemExit("Could not extract RMSE_mean from metrics.json (primary_metric or metrics._overall.RMSE_mean).")

    best_path = Path(args.best_path)
    runs_path = Path(args.runs_path)

    # Always log the run
    append_runs_csv(runs_path, payload, rmse_mean)

    # Initialize best.json if missing
    if not best_path.exists():
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_payload = {
            "best_rmse_mean": rmse_mean,
            "git_sha": payload.get("git_sha"),
            "timestamp": payload.get("timestamp"),
            "model": payload.get("model"),
        }
        best_path.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"PASS: initialized best.json with rmse_mean={rmse_mean}")
        raise SystemExit(0)

    best = json.loads(best_path.read_text(encoding="utf-8"))
    best_rmse = best.get("best_rmse_mean")
    if not isinstance(best_rmse, (int, float)):
        raise SystemExit("best.json missing valid best_rmse_mean")

    best_rmse = float(best_rmse)

    # Minimize RMSE_mean
    if rmse_mean <= best_rmse + float(args.tol):
        best_payload = {
            "best_rmse_mean": rmse_mean,
            "git_sha": payload.get("git_sha"),
            "timestamp": payload.get("timestamp"),
            "model": payload.get("model"),
        }
        best_path.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"PASS: improved rmse_mean {best_rmse} -> {rmse_mean}")
        raise SystemExit(0)

    print(f"FAIL: no improvement. best={best_rmse}, current={rmse_mean}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
