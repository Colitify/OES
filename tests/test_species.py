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


def test_nmf_spectral_decomposition():
    """NMF decomposes synthetic spectra into known components."""
    from src.species import nmf_decompose

    np.random.seed(42)
    n_wavelengths = 200
    n_components = 3
    n_samples = 100

    wl = np.linspace(200, 800, n_wavelengths)
    pure = np.zeros((n_components, n_wavelengths))
    pure[0] = np.exp(-0.5 * ((wl - 400) / 10) ** 2)
    pure[1] = np.exp(-0.5 * ((wl - 550) / 15) ** 2)
    pure[2] = np.exp(-0.5 * ((wl - 700) / 8) ** 2)

    concentrations = np.random.rand(n_samples, n_components)
    X = concentrations @ pure + 0.01 * np.random.rand(n_samples, n_wavelengths)

    components, weights, model = nmf_decompose(X, n_components=n_components)

    assert components.shape == (n_components, n_wavelengths)
    assert weights.shape == (n_samples, n_components)
    assert (components >= 0).all()
    assert (weights >= 0).all()

    peak_positions = [wl[np.argmax(c)] for c in components]
    peak_positions.sort()
    assert abs(peak_positions[0] - 400) < 30
    assert abs(peak_positions[1] - 550) < 30
    assert abs(peak_positions[2] - 700) < 30


def test_match_lines_to_nist():
    """Match detected peaks to NIST emission lines."""
    from src.species import match_lines_to_nist

    detected_peaks_nm = np.array([685.5, 696.6, 750.3, 777.5, 500.0])
    matches = match_lines_to_nist(detected_peaks_nm, tolerance_nm=1.5)

    assert any(m["species"] == "F_I" and abs(m["detected"] - 685.5) < 0.1 for m in matches)
    assert any(m["species"] == "Ar_I" and abs(m["detected"] - 696.6) < 0.1 for m in matches)
    assert any(m["species"] == "O_I" and abs(m["detected"] - 777.5) < 0.1 for m in matches)


def test_match_lines_to_nist_returns_unmatched():
    """Unmatched peaks are returned with species=None."""
    from src.species import match_lines_to_nist

    detected = np.array([500.0, 600.0])
    matches = match_lines_to_nist(detected, tolerance_nm=1.0)
    assert len(matches) == 2
    assert all(m["species"] is None for m in matches)


def test_detect_species_presence():
    """Detect species presence by emission line intensity thresholding."""
    from src.species import detect_species_presence

    np.random.seed(42)
    n_wavelengths = 500
    wl = np.linspace(186, 884, n_wavelengths)

    spectrum = np.random.rand(n_wavelengths) * 0.1
    f_idx = np.argmin(np.abs(wl - 685.6))
    ar_idx = np.argmin(np.abs(wl - 750.4))
    spectrum[f_idx - 2: f_idx + 3] = 5.0
    spectrum[ar_idx - 2: ar_idx + 3] = 3.0

    presence = detect_species_presence(spectrum, wl, threshold_sigma=2.0)

    assert isinstance(presence, dict)
    assert "F_I" in presence
    assert presence["F_I"] is True
    assert presence.get("Si_I", False) is False


def test_detect_species_presence_batch():
    """Batch detection returns (n_samples, n_species) binary matrix."""
    from src.species import detect_species_presence_batch

    np.random.seed(42)
    n_samples, n_wl = 10, 500
    wl = np.linspace(186, 884, n_wl)
    X = np.random.rand(n_samples, n_wl) * 0.1

    labels, species_names = detect_species_presence_batch(X, wl)
    assert labels.shape[0] == n_samples
    assert labels.shape[1] == len(species_names)
    assert labels.dtype == np.int64
    assert set(np.unique(labels)).issubset({0, 1})
