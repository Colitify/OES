"""Unit tests for src.preprocessing module."""

import numpy as np
import pytest

from src.preprocessing import Preprocessor


class TestCosmicRayRemoval:
    def test_spike_removed(self):
        """Injected spike should be replaced by local median value."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0.5, 1.0, (5, 200)).astype(np.float32)
        spike_row, spike_col = 2, 100
        original_val = float(X[spike_row, spike_col])
        X[spike_row, spike_col] = original_val * 10   # 10× spike

        X_clean = Preprocessor.cosmic_ray_removal(X, threshold=5.0)

        assert float(X_clean[spike_row, spike_col]) <= original_val * 2, (
            "Spike should be reduced to <= 2× original after removal"
        )

    def test_flat_spectrum_unchanged(self):
        """Perfectly flat spectrum should not be modified."""
        X = np.ones((3, 100), dtype=np.float32)
        X_clean = Preprocessor.cosmic_ray_removal(X, threshold=5.0)
        np.testing.assert_array_equal(X, X_clean)

    def test_output_shape_preserved(self):
        """Output shape must match input shape."""
        X = np.random.rand(8, 300).astype(np.float32)
        X_clean = Preprocessor.cosmic_ray_removal(X)
        assert X_clean.shape == X.shape


class TestAlignWavelengths:
    def test_output_shape(self):
        """Aligned array must have (n_samples, n_tgt_channels) shape."""
        X = np.random.rand(10, 100).astype(np.float32)
        src_wl = np.linspace(200, 900, 100)
        tgt_wl = np.linspace(200, 900, 200)
        X_aligned = Preprocessor.align_wavelengths(X, src_wl, tgt_wl)
        assert X_aligned.shape == (10, 200)

    def test_identity_mapping(self):
        """Aligning to the same grid should return (nearly) the same values."""
        X = np.random.rand(5, 50).astype(np.float32)
        wl = np.linspace(300, 800, 50)
        X_aligned = Preprocessor.align_wavelengths(X, wl, wl)
        np.testing.assert_allclose(X_aligned, X, rtol=1e-4, atol=1e-5)

    def test_interpolation_midpoint(self):
        """Values at midpoints should be interpolated correctly."""
        # Simple linear spectrum: intensity = wavelength
        wl_src = np.array([100.0, 200.0, 300.0])
        X = np.array([[100.0, 200.0, 300.0]])   # identity: intensity == wavelength
        wl_tgt = np.array([150.0, 250.0])
        X_aligned = Preprocessor.align_wavelengths(X, wl_src, wl_tgt)
        np.testing.assert_allclose(X_aligned, [[150.0, 250.0]], rtol=1e-5)

    def test_dtype_float32(self):
        """Output dtype should be float32."""
        X = np.random.rand(4, 80)
        X_aligned = Preprocessor.align_wavelengths(
            X, np.linspace(200, 800, 80), np.linspace(200, 800, 40)
        )
        assert X_aligned.dtype == np.float32


class TestPreprocessorPipeline:
    def test_fit_transform_shape(self):
        """Preprocessor.fit_transform must return same shape as input."""
        X = np.random.rand(12, 500).astype(np.float32) + 1.0
        pp = Preprocessor(baseline="none", normalize="snv", denoise="savgol",
                          cosmic_ray=False)
        X_pp = pp.fit_transform(X)
        assert X_pp.shape == X.shape

    def test_snv_zero_mean(self):
        """After SNV normalization, each spectrum must have zero mean."""
        X = np.random.rand(6, 100).astype(np.float32) * 10 + 5
        pp = Preprocessor(baseline="none", normalize="snv", denoise="none",
                          cosmic_ray=False)
        X_pp = pp.fit_transform(X)
        np.testing.assert_allclose(X_pp.mean(axis=1), 0.0, atol=1e-5)
