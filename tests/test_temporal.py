"""Unit tests for src.temporal module."""

import numpy as np
import pytest

from src.temporal import compute_temporal_embedding, cluster_discharge_phases


class TestComputeTemporalEmbedding:
    def test_output_shape(self):
        """Embedding must have shape (T, n_components)."""
        T, n_wl, n_comp = 100, 50, 5
        spectra = np.random.rand(T, n_wl).astype(np.float32)
        embedding, pca = compute_temporal_embedding(spectra, n_components=n_comp)
        assert embedding.shape == (T, n_comp), (
            f"Expected shape ({T}, {n_comp}), got {embedding.shape}"
        )

    def test_pca_object_returned(self):
        """Second return value must be a fitted PCA object."""
        from sklearn.decomposition import PCA
        spectra = np.random.rand(50, 100).astype(np.float32)
        embedding, pca = compute_temporal_embedding(spectra, n_components=3)
        assert isinstance(pca, PCA), "Second return must be a PCA object"
        assert hasattr(pca, "explained_variance_ratio_"), "PCA must be fitted"

    def test_embedding_dtype_float32(self):
        """Embedding dtype should be float32."""
        spectra = np.random.rand(40, 60).astype(np.float32)
        embedding, _ = compute_temporal_embedding(spectra, n_components=4)
        assert embedding.dtype == np.float32

    def test_explained_variance_positive(self):
        """All explained variance ratios must be positive."""
        spectra = np.random.rand(80, 30).astype(np.float64)
        _, pca = compute_temporal_embedding(spectra, n_components=5)
        assert (pca.explained_variance_ratio_ > 0).all(), (
            "All explained variance ratios must be positive"
        )

    def test_n_components_clipped_to_min_dim(self):
        """Requesting more components than min(T, n_wl) should not raise."""
        spectra = np.random.rand(10, 8).astype(np.float32)
        embedding, pca = compute_temporal_embedding(spectra, n_components=100)
        # Should succeed and return min(10, 8) = 8 components
        assert embedding.shape[1] <= 8


class TestClusterDischargePhases:
    def _make_embedding(self, T=200, n_feat=10):
        return np.random.rand(T, n_feat).astype(np.float32)

    def test_labels_shape(self):
        """Labels array must have length equal to number of time steps."""
        emb = self._make_embedding(T=150)
        labels, centroids, inertia = cluster_discharge_phases(emb, k=4, metric="euclidean")
        assert labels.shape == (150,), f"Expected (150,), got {labels.shape}"

    def test_centroids_shape(self):
        """Centroids array must have shape (k, n_features)."""
        n_feat = 10
        emb = self._make_embedding(T=100, n_feat=n_feat)
        k = 3
        labels, centroids, inertia = cluster_discharge_phases(emb, k=k, metric="euclidean")
        assert centroids.shape == (k, n_feat), (
            f"Expected centroids shape ({k}, {n_feat}), got {centroids.shape}"
        )

    def test_labels_in_range(self):
        """All labels must be in [0, k-1]."""
        k = 4
        emb = self._make_embedding(T=80)
        labels, _, _ = cluster_discharge_phases(emb, k=k, metric="euclidean")
        assert labels.min() >= 0, "Labels contain negative values"
        assert labels.max() <= k - 1, f"Labels contain values >= k={k}"

    def test_inertia_positive(self):
        """Clustering inertia must be strictly positive."""
        emb = self._make_embedding(T=60)
        _, _, inertia = cluster_discharge_phases(emb, k=3, metric="euclidean")
        assert inertia > 0, f"Expected positive inertia, got {inertia}"

    def test_inertia_decreases_with_more_clusters(self):
        """Inertia should decrease (or stay equal) as k increases."""
        emb = self._make_embedding(T=100)
        _, _, inertia_k2 = cluster_discharge_phases(emb, k=2, metric="euclidean")
        _, _, inertia_k4 = cluster_discharge_phases(emb, k=4, metric="euclidean")
        assert inertia_k4 <= inertia_k2, (
            f"Inertia should decrease with more clusters: k=2 {inertia_k2:.2f}, k=4 {inertia_k4:.2f}"
        )


def test_extract_species_timeseries():
    """Extract per-species intensity time series from spectra."""
    from src.temporal import extract_species_timeseries

    np.random.seed(42)
    T, n_wl = 100, 500
    wl = np.linspace(186, 884, n_wl)
    spectra = np.random.rand(T, n_wl).astype(np.float32) * 0.1

    f_idx = np.argmin(np.abs(wl - 685.6))
    spectra[:, f_idx] = np.linspace(1, 5, T)

    ts, names = extract_species_timeseries(spectra, wl)
    assert ts.shape[0] == T
    assert len(names) == ts.shape[1]
    assert "F_I" in names

    f_col = names.index("F_I")
    assert ts[-1, f_col] > ts[0, f_col]


def test_extract_species_timeseries_out_of_range():
    """Species with lines outside wavelength range return zeros."""
    from src.temporal import extract_species_timeseries

    np.random.seed(42)
    T, n_wl = 50, 200
    wl = np.linspace(186, 400, n_wl)
    spectra = np.random.rand(T, n_wl).astype(np.float32)

    ts, names = extract_species_timeseries(spectra, wl, species=["F_I", "N2_2pos"])
    f_col = names.index("F_I")
    n2_col = names.index("N2_2pos")

    assert np.allclose(ts[:, f_col], 0.0)
    assert ts[:, n2_col].sum() > 0
