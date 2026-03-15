import numpy as np
import pytest


def test_label_process_phases_basic():
    """Label process phases from gas flow data."""
    from src.data_loader import label_process_phases

    n_timesteps = 100
    sf6_flow = np.zeros(n_timesteps)
    c4f8_flow = np.zeros(n_timesteps)
    sf6_flow[:40] = 100.0
    c4f8_flow[50:90] = 80.0

    labels = label_process_phases(sf6_flow, c4f8_flow, threshold_pct=10.0)

    assert labels.shape == (n_timesteps,)
    assert set(np.unique(labels)).issubset({0, 1, 2})
    assert (labels[:40] == 1).all()
    assert (labels[40:50] == 0).all()
    assert (labels[50:90] == 2).all()
    assert (labels[90:] == 0).all()


def test_label_process_phases_overlap():
    """Both gases flowing = etch (SF6 dominant in Bosch process)."""
    from src.data_loader import label_process_phases

    n = 50
    sf6 = np.full(n, 100.0)
    c4f8 = np.full(n, 50.0)
    labels = label_process_phases(sf6, c4f8)
    assert (labels == 1).all()


def test_load_bosch_multi_wafer_returns_dict():
    """Multi-wafer loader returns expected keys."""
    from src.data_loader import load_bosch_multi_wafer
    import os
    bosch_dir = "data/bosch_oes"
    if not os.path.isdir(bosch_dir):
        pytest.skip("BOSCH data not available")

    result = load_bosch_multi_wafer(bosch_dir, max_wafers=1, max_timesteps=500)
    assert "spectra" in result
    assert "wavelengths" in result
    assert "labels" in result
    assert "wafer_ids" in result
    assert result["spectra"].shape[0] == result["labels"].shape[0]
    assert result["spectra"].shape[1] == 3648
