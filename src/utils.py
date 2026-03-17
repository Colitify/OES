"""Shared utility functions."""

import json
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def get_git_sha(cwd: str = None) -> str:
    """Return the current git HEAD SHA, or 'unknown' on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or str(_ROOT),
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def write_metrics_json(path, payload) -> None:
    """Write a metrics dict as formatted JSON with consistent encoding."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path, default=None):
    """Load a JSON file, returning *default* if the file does not exist."""
    p = Path(path)
    if not p.exists():
        return default if default is not None else {}
    return json.loads(p.read_text(encoding="utf-8"))
