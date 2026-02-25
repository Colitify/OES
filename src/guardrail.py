"""
Guardrail script for Ralph-style iteration.

Usage:
    python -m src.guardrail results/metrics.json [--tol 0.02]

Behavior:
- Reads a machine-readable metrics.json produced by main.py
- Extracts primary metric and direction from primary_metric.name:
    "RMSE_mean" → minimize (pass if current <= best + tol)
    "micro_f1"  → maximize (pass if current >= best - tol)
    "setup_ok"  → always pass (infrastructure stories)
- Appends one line to results/runs.csv
- Updates results/best.json if improved
- Exit code:
    0 => PASS (improved or initialized)
    1 => FAIL (no improvement)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def extract_primary_metric(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    """Return (metric_name, metric_value) from primary_metric block, with fallbacks."""
    pm = payload.get("primary_metric", {})
    if isinstance(pm, dict):
        name = pm.get("name")
        v = pm.get("value")
        if isinstance(name, str) and isinstance(v, (int, float)):
            return name, float(v)

    # Legacy fallback: RMSE_mean in metrics._overall
    metrics = payload.get("metrics", {})
    if isinstance(metrics, dict):
        overall = metrics.get("_overall", {})
        if isinstance(overall, dict):
            v = overall.get("RMSE_mean")
            if isinstance(v, (int, float)):
                return "RMSE_mean", float(v)

    return None, None


# Keep legacy name for backward compatibility
def extract_rmse_mean(payload: Dict[str, Any]) -> Optional[float]:
    _, v = extract_primary_metric(payload)
    return v


def append_runs_csv(runs_path: Path, payload: Dict[str, Any], metric_name: str, metric_value: float) -> None:
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    header = "timestamp,metric_name,metric_value,model,task,cv_folds,seed,git_sha\n"

    timestamp = payload.get("timestamp")
    model = payload.get("model")
    task = payload.get("task", "regress")
    cv_folds = payload.get("cv_folds")
    seed = payload.get("seed")
    git_sha = payload.get("git_sha")

    line = f"{timestamp},{metric_name},{metric_value},{model},{task},{cv_folds},{seed},{git_sha}\n"

    if not runs_path.exists():
        runs_path.write_text(header, encoding="utf-8")

    with runs_path.open("a", encoding="utf-8") as f:
        f.write(line)


def main():
    parser = argparse.ArgumentParser(
        description="Guardrail for Ralph iteration. Supports minimize (RMSE_mean) and maximize (micro_f1) modes."
    )
    parser.add_argument("metrics_json", type=str, help="Path to metrics.json")
    parser.add_argument("--best_path", type=str, default="results/best.json", help="Path to best.json")
    parser.add_argument("--runs_path", type=str, default="results/runs.csv", help="Path to runs.csv")
    parser.add_argument("--tol", type=float, default=0.0, help="Allowed degradation tolerance.")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json)
    if not metrics_path.exists():
        raise SystemExit(f"metrics.json not found: {metrics_path}")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metric_name, metric_value = extract_primary_metric(payload)

    if metric_name is None or metric_value is None:
        raise SystemExit(
            "Could not extract primary_metric from metrics.json. "
            "Expected: primary_metric.name and primary_metric.value, "
            "or metrics._overall.RMSE_mean."
        )

    # setup_ok: infrastructure stories always pass
    if metric_name == "setup_ok":
        print(f"PASS: setup_ok story — guardrail always passes.")
        raise SystemExit(0)

    # Determine optimization direction
    maximize = metric_name in ("micro_f1", "macro_f1", "accuracy", "f1")

    best_path = Path(args.best_path)
    runs_path = Path(args.runs_path)

    # Always log the run
    append_runs_csv(runs_path, payload, metric_name, metric_value)

    # Initialize best.json if missing or metric changed
    if not best_path.exists():
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_payload = {
            "metric_name": metric_name,
            "best_value": metric_value,
            "maximize": maximize,
            "git_sha": payload.get("git_sha"),
            "timestamp": payload.get("timestamp"),
            "model": payload.get("model"),
        }
        best_path.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"PASS: initialized best.json with {metric_name}={metric_value:.4f} ({'maximize' if maximize else 'minimize'})")
        raise SystemExit(0)

    best = json.loads(best_path.read_text(encoding="utf-8"))

    # If metric type changed (e.g. switching from RMSE to micro_f1), reinitialize
    if best.get("metric_name") != metric_name:
        best_payload = {
            "metric_name": metric_name,
            "best_value": metric_value,
            "maximize": maximize,
            "git_sha": payload.get("git_sha"),
            "timestamp": payload.get("timestamp"),
            "model": payload.get("model"),
        }
        best_path.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"PASS: metric changed to {metric_name}={metric_value:.4f} — reinitializing best.json")
        raise SystemExit(0)

    best_value = float(best.get("best_value", best.get("best_rmse_mean", 0.0)))
    tol = float(args.tol)

    if maximize:
        # Higher is better (micro_f1): pass if current >= best - tol
        passed = metric_value >= best_value - tol
        direction_str = f"{best_value:.4f} -> {metric_value:.4f} (maximize, tol={tol})"
    else:
        # Lower is better (RMSE_mean): pass if current <= best + tol
        passed = metric_value <= best_value + tol
        direction_str = f"{best_value:.4f} -> {metric_value:.4f} (minimize, tol={tol})"

    if passed:
        # Update best if genuinely improved
        improved = (metric_value > best_value) if maximize else (metric_value < best_value)
        if improved:
            best_payload = {
                "metric_name": metric_name,
                "best_value": metric_value,
                "maximize": maximize,
                "git_sha": payload.get("git_sha"),
                "timestamp": payload.get("timestamp"),
                "model": payload.get("model"),
            }
            best_path.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"PASS: {metric_name} {direction_str}")
        raise SystemExit(0)

    print(f"FAIL: {metric_name} {direction_str}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
