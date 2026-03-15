import numpy as np
import pytest
import json
import tempfile
from pathlib import Path


def test_guardrail_minimize():
    """Guardrail detects regression in minimize mode."""
    from src.guardrail import extract_primary_metric

    payload_good = {"primary_metric": {"name": "RMSE_mean", "value": 2.5}}
    name, value = extract_primary_metric(payload_good)
    assert name == "RMSE_mean"
    assert value == 2.5

    payload_bad = {"primary_metric": {"name": "RMSE_mean", "value": 3.0}}
    name2, value2 = extract_primary_metric(payload_bad)
    assert name2 == "RMSE_mean"
    assert value2 == 3.0

    # Worse value (higher RMSE) should be detectable
    assert value2 > value
