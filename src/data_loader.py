"""Data loading module for OES/LIBS spectral data."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Union


class SpectralDataset:
    """Container for spectral data with wavelengths and targets."""

    def __init__(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        targets: Optional[np.ndarray] = None,
        target_names: Optional[List[str]] = None,
        sample_ids: Optional[np.ndarray] = None
    ):
        self.spectra = spectra
        self.wavelengths = wavelengths
        self.targets = targets
        self.target_names = target_names or []
        self.sample_ids = sample_ids

    @property
    def n_samples(self) -> int:
        return self.spectra.shape[0]

    @property
    def n_wavelengths(self) -> int:
        return self.spectra.shape[1]

    @property
    def n_targets(self) -> int:
        return self.targets.shape[1] if self.targets is not None else 0

    @classmethod
    def from_csv(
        cls,
        filepath: Union[str, Path],
        n_wavelengths: int = 40002,
        target_cols: Optional[List[str]] = None,
        id_col: Optional[str] = None
    ) -> "SpectralDataset":
        """Load dataset from CSV file (LIBS contest format).

        Args:
            filepath: Path to CSV file
            n_wavelengths: Number of wavelength columns at the beginning
            target_cols: List of target column names (auto-detect if None)
            id_col: Sample ID column name

        Returns:
            SpectralDataset instance
        """
        df = pd.read_csv(filepath)

        # Extract wavelengths from column names (handle 'X' prefix)
        wavelength_cols = df.columns[:n_wavelengths]
        wavelengths = np.array([
            float(col.lstrip('X')) if isinstance(col, str) and col.startswith('X')
            else float(col)
            for col in wavelength_cols
        ])

        # Extract spectra
        spectra = df.iloc[:, :n_wavelengths].values.astype(np.float32)

        # Extract targets
        if target_cols is None:
            # Auto-detect numeric columns after wavelengths
            remaining_cols = df.columns[n_wavelengths:]
            target_cols = [c for c in remaining_cols
                          if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
                          and c != id_col]

        targets = df[target_cols].values.astype(np.float32) if target_cols else None

        # Extract sample IDs
        sample_ids = df[id_col].values if id_col and id_col in df.columns else None

        return cls(spectra, wavelengths, targets, target_cols, sample_ids)

    def split(
        self,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple["SpectralDataset", "SpectralDataset"]:
        """Split dataset into train and validation sets."""
        from sklearn.model_selection import train_test_split

        indices = np.arange(self.n_samples)
        train_idx, val_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )

        train_data = SpectralDataset(
            spectra=self.spectra[train_idx],
            wavelengths=self.wavelengths,
            targets=self.targets[train_idx] if self.targets is not None else None,
            target_names=self.target_names,
            sample_ids=self.sample_ids[train_idx] if self.sample_ids is not None else None
        )

        val_data = SpectralDataset(
            spectra=self.spectra[val_idx],
            wavelengths=self.wavelengths,
            targets=self.targets[val_idx] if self.targets is not None else None,
            target_names=self.target_names,
            sample_ids=self.sample_ids[val_idx] if self.sample_ids is not None else None
        )

        return train_data, val_data

    def __repr__(self) -> str:
        return (
            f"SpectralDataset(n_samples={self.n_samples}, "
            f"n_wavelengths={self.n_wavelengths}, "
            f"n_targets={self.n_targets})"
        )


def load_libs_data(
    filepath: Union[str, Path],
    n_wavelengths: int = 40002
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    """Load LIBS contest format CSV.

    Args:
        filepath: Path to CSV file
        n_wavelengths: Number of wavelength columns

    Returns:
        Tuple of (wavelengths, spectra, targets, target_names)
    """
    df = pd.read_csv(filepath)

    # Extract wavelengths (handle 'X' prefix)
    wavelength_cols = df.columns[:n_wavelengths]
    wavelengths = np.array([
        float(col.lstrip('X')) if isinstance(col, str) and col.startswith('X')
        else float(col)
        for col in wavelength_cols
    ])
    X = df.iloc[:, :n_wavelengths].values

    target_cols = [c for c in df.columns[n_wavelengths:]
                   if df[c].dtype in [np.float64, np.int64]]
    y = df[target_cols].values if target_cols else None

    return wavelengths, X, y, target_cols


def load_large_csv(
    filepath: Union[str, Path],
    chunksize: int = 500,
    dtype: type = np.float32
) -> np.ndarray:
    """Memory-efficient loading with chunking.

    Args:
        filepath: Path to CSV file
        chunksize: Number of rows per chunk
        dtype: Data type for the array

    Returns:
        Stacked numpy array
    """
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        chunks.append(chunk.values.astype(dtype))
    return np.vstack(chunks)
