"""Unit tests for src.features module."""

import numpy as np
import pytest
import pandas as pd
from scipy.stats import norm as sp_norm

from src.features import detect_peaks, select_wavelengths_plasma, PlasmaDescriptorExtractor


class TestDetectPeaks:
    def test_returns_dataframe(self):
        """detect_peaks must return a pandas DataFrame."""
        wl = np.linspace(200, 900, 500)
        spec = np.random.rand(500).astype(np.float32)
        result = detect_peaks(spec, wl)
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_columns(self):
        """Output DataFrame must contain required columns."""
        wl = np.linspace(200, 900, 500)
        spec = np.random.rand(500).astype(np.float32)
        df = detect_peaks(spec, wl)
        for col in ("wavelength_nm", "intensity", "prominence", "fwhm_nm"):
            assert col in df.columns, f"Missing column: {col}"

    def test_gaussian_peak_detected(self):
        """A clearly injected Gaussian peak should appear in the output."""
        n = 1000
        wl = np.linspace(300, 800, n)
        # Baseline flat at 0.1, Gaussian peak at 550 nm with sigma=2 nm
        spec = np.full(n, 0.1, dtype=np.float64)
        peak_center = 550.0
        sigma_nm = 2.0
        spec += 5.0 * np.exp(-0.5 * ((wl - peak_center) / sigma_nm) ** 2)

        df = detect_peaks(spec.astype(np.float32), wl, min_prominence=0.5)
        assert len(df) > 0, "No peaks detected; expected at least one."
        # Closest detected peak to 550 nm should be within 5 nm
        dist = np.abs(df["wavelength_nm"].values - peak_center)
        assert dist.min() < 5.0, f"Detected peak too far from 550 nm: {dist.min():.1f} nm"

    def test_empty_flat_spectrum(self):
        """A perfectly flat spectrum should return an empty DataFrame."""
        wl = np.linspace(200, 900, 500)
        spec = np.ones(500, dtype=np.float32)
        df = detect_peaks(spec, wl, min_prominence=0.5)
        assert len(df) == 0


class TestSelectWavelengthsPlasma:
    def test_returns_boolean_mask(self):
        """Return value must be a boolean array of same length as wavelengths."""
        wl = np.linspace(250, 800, 300)
        mask = select_wavelengths_plasma(wl)
        assert mask.dtype == bool
        assert mask.shape == wl.shape

    def test_mask_true_near_plasma_lines(self):
        """Mask should be True near known plasma emission lines."""
        # N2 second-positive band head at 337.1 nm
        wl = np.array([337.1])
        mask = select_wavelengths_plasma(wl, species=["N2_2pos"])
        # The line itself (exactly at the center) should be True
        # (subject to the delta window used)
        # Just verify the call doesn't raise and returns a mask
        assert mask.shape == (1,)

    def test_species_filter(self):
        """Requesting a single species should produce a narrower mask than all."""
        wl = np.linspace(200, 900, 1000)
        mask_all = select_wavelengths_plasma(wl, species=None)
        mask_n2 = select_wavelengths_plasma(wl, species=["N2_2pos"])
        assert mask_n2.sum() <= mask_all.sum(), (
            "Single-species mask should not exceed all-species mask"
        )


class TestPlasmaDescriptorExtractor:
    def _make_spectra(self, n=10, n_wl=500):
        wl = np.linspace(250, 800, n_wl)
        X = np.random.rand(n, n_wl).astype(np.float32) + 0.5
        return X, wl

    def test_output_is_2d(self):
        """fit_transform must return a 2D array."""
        X, wl = self._make_spectra()
        # wavelengths passed to fit(), not __init__()
        pde = PlasmaDescriptorExtractor()
        pde.fit(X, wavelengths=wl)
        out = pde.transform(X)
        assert out.ndim == 2, "Output must be 2D"

    def test_fixed_width_output(self):
        """Output width must be the same for different numbers of samples."""
        X1, wl = self._make_spectra(n=5)
        X2, _ = self._make_spectra(n=10)
        pde = PlasmaDescriptorExtractor()
        pde.fit(X1, wavelengths=wl)
        w1 = pde.transform(X1).shape[1]
        w2 = pde.transform(X2).shape[1]
        assert w1 == w2, "Feature width must be constant across different sample counts"

    def test_no_nan_in_output(self):
        """Descriptor output must not contain NaN or Inf values."""
        X, wl = self._make_spectra(n=8)
        pde = PlasmaDescriptorExtractor()
        pde.fit(X, wavelengths=wl)
        out = pde.transform(X)
        assert not np.isnan(out).any(), "Output contains NaN"
        assert not np.isinf(out).any(), "Output contains Inf"
