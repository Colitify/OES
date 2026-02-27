"""Unit tests for src.spatial module — wafer spatial analysis."""

import numpy as np
import pandas as pd
import pytest

from src.spatial import compute_wafer_uniformity, interpolate_wafer_map, link_oes_spatial


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_spatial_df(n_wafers=3, n_points=10, seed=42):
    """Synthesize a spatial DataFrame mimicking Si_Oxide_etch_89_points.csv."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_wafers):
        key = f"2024-07-{2 + i:02d}_01"
        for j in range(n_points):
            rows.append({
                "experiment_key": key,
                "lot_number": 1,
                "wafer_number": i + 1,
                "X": rng.uniform(-95000, 95000),
                "Y": rng.uniform(-95000, 95000),
                "oxide_etch": rng.uniform(0.4, 0.6),
                "si_etch": rng.uniform(50, 55),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TestComputeWaferUniformity
# ---------------------------------------------------------------------------

class TestComputeWaferUniformity:
    def test_returns_dataframe(self):
        df = _make_spatial_df()
        result = compute_wafer_uniformity(df, "oxide_etch")
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self):
        df = _make_spatial_df()
        result = compute_wafer_uniformity(df, "oxide_etch")
        for col in ("experiment_key", "mean", "std", "min", "max", "uniformity_pct"):
            assert col in result.columns, f"Missing column: {col}"

    def test_constant_values_give_zero_uniformity(self):
        """If all measurements on a wafer are identical, uniformity should be 0."""
        df = pd.DataFrame({
            "experiment_key": ["w1"] * 5,
            "oxide_etch": [1.0] * 5,
        })
        result = compute_wafer_uniformity(df, "oxide_etch")
        assert result["uniformity_pct"].iloc[0] == pytest.approx(0.0)

    def test_number_of_rows_equals_unique_wafers(self):
        df = _make_spatial_df(n_wafers=4)
        result = compute_wafer_uniformity(df, "oxide_etch")
        assert len(result) == 4

    def test_missing_column_raises(self):
        df = _make_spatial_df()
        with pytest.raises(ValueError, match="not found"):
            compute_wafer_uniformity(df, "nonexistent_col")


# ---------------------------------------------------------------------------
# TestInterpolateWaferMap
# ---------------------------------------------------------------------------

class TestInterpolateWaferMap:
    def test_output_shape(self):
        x = np.array([0, 1, 2, 0, 1, 2], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        v = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        gx, gy, gz = interpolate_wafer_map(x, y, v, grid_size=50)
        assert gx.shape == (50,)
        assert gy.shape == (50,)
        assert gz.shape == (50, 50)

    def test_custom_grid_size(self):
        x = np.array([0, 1, 0, 1], dtype=float)
        y = np.array([0, 0, 1, 1], dtype=float)
        v = np.array([1, 2, 3, 4], dtype=float)
        gx, gy, gz = interpolate_wafer_map(x, y, v, grid_size=100)
        assert gz.shape == (100, 100)

    def test_values_within_input_range(self):
        """Interpolated values should stay reasonably close to input range."""
        rng = np.random.RandomState(0)
        x = rng.uniform(-1, 1, 20)
        y = rng.uniform(-1, 1, 20)
        v = rng.uniform(10, 20, 20)
        _, _, gz = interpolate_wafer_map(x, y, v, grid_size=30)
        # RBF can slightly overshoot, allow generous margin
        assert gz.min() > v.min() - 20
        assert gz.max() < v.max() + 20


# ---------------------------------------------------------------------------
# TestLinkOesSpatial
# ---------------------------------------------------------------------------

class TestLinkOesSpatial:
    def test_merge_produces_expected_columns(self):
        spatial = _make_spatial_df(n_wafers=2, n_points=5)
        oes = pd.DataFrame({
            "experiment_key": spatial["experiment_key"].unique(),
            "pca_1": [0.1, 0.2],
            "pca_2": [0.3, 0.4],
        })
        merged = link_oes_spatial(spatial, oes, "oxide_etch")
        assert "pca_1" in merged.columns
        assert "uniformity_pct" in merged.columns
        assert len(merged) == 2
