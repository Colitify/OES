import numpy as np
import pytest


def test_compute_line_ratios():
    """Compute emission line intensity ratios between species."""
    from src.intensity import compute_line_ratios

    np.random.seed(42)
    T, n_wl = 50, 500
    wl = np.linspace(186, 884, n_wl)
    spectra = np.random.rand(T, n_wl).astype(np.float32) * 0.1

    f_idx = np.argmin(np.abs(wl - 685.6))
    ar_idx = np.argmin(np.abs(wl - 750.4))
    spectra[:, f_idx] = 3.0
    spectra[:, ar_idx] = 1.5

    ratios = compute_line_ratios(spectra, wl, numerator="F_I", denominator="Ar_I")
    assert ratios.shape == (T,)
    assert np.all(np.isfinite(ratios))


def test_compute_line_ratios_zero_denominator():
    """Line ratios handle zero-intensity denominator via eps floor."""
    from src.intensity import compute_line_ratios

    np.random.seed(42)
    T, n_wl = 20, 500
    wl = np.linspace(186, 884, n_wl)
    spectra = np.zeros((T, n_wl), dtype=np.float32)

    f_idx = np.argmin(np.abs(wl - 685.6))
    spectra[:, f_idx] = 2.0

    ratios = compute_line_ratios(spectra, wl, numerator="F_I", denominator="Si_I")
    assert ratios.shape == (T,)
    assert np.all(np.isfinite(ratios))
    assert np.all(ratios > 0)


def test_actinometry():
    """Actinometry normalizes species intensity by Ar reference."""
    from src.intensity import actinometry

    np.random.seed(42)
    T, n_wl = 30, 500
    wl = np.linspace(186, 884, n_wl)
    spectra = np.random.rand(T, n_wl).astype(np.float32) * 0.1

    ar_idx = np.argmin(np.abs(wl - 750.4))
    spectra[:, ar_idx] = 2.0

    normalized = actinometry(spectra, wl, target_species="F_I", reference_species="Ar_I")
    assert normalized.shape == (T,)
    assert np.all(np.isfinite(normalized))


def test_oes_to_process_regression():
    """Regression from OES intensities to process parameters."""
    from src.intensity import oes_to_process_regression

    np.random.seed(42)
    n = 200
    n_species = 5
    X = np.random.randn(n, n_species).astype(np.float32)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n) * 0.5

    result = oes_to_process_regression(X, y, model_type="ridge", cv=3)
    assert "rmse" in result
    assert "r2" in result
    assert result["r2"] > 0.5


def test_boltzmann_temperature():
    """Boltzmann plot estimates T_exc from synthetic Ar lines."""
    from src.intensity import boltzmann_temperature, AR_LINE_DATA

    np.random.seed(42)
    T_true = 5000.0  # K
    k_B = 8.617333e-5  # eV/K

    n_wl = 1000
    wl = np.linspace(650, 800, n_wl)
    n_timesteps = 20
    spectra = np.ones((n_timesteps, n_wl)) * 100  # baseline

    # Inject Ar I lines with Boltzmann-distributed intensities
    for nm, data in AR_LINE_DATA.items():
        I = data["g"] * data["A"] / nm * np.exp(-data["E_upper"] / (k_B * T_true))
        idx = np.argmin(np.abs(wl - nm))
        if 0 <= idx < n_wl:
            spectra[:, idx] = I * 1e12 + 100  # scale + baseline

    T_est = boltzmann_temperature(spectra, wl)
    valid = np.isfinite(T_est)
    assert valid.sum() > 0, "No valid T_exc estimates"
    # Should be within ~50% of true T (Boltzmann plot is rough)
    median_T = np.median(T_est[valid])
    assert 2000 < median_T < 15000, f"T_exc={median_T:.0f}K out of expected range"
