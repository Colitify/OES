"""Data loading module for OES/LIBS spectral data."""

import json
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


def load_libs_benchmark(
    path: Union[str, Path],
    split: str = "train",
    max_spectra_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load LIBS Benchmark HDF5 dataset (Figshare, 12 classes, 40002 channels).

    File structure (from official reader):
      train.h5:
        Wavelengths/<key>  -> (40002,) float64 wavelengths in nm
        Spectra/<material> -> (40002, n_spectra) float64 intensities
        Class/1            -> (n_total_spectra,) int class labels (1-based)

    Args:
        path: Directory containing train.h5 (and optionally test.h5, test_labels.csv)
        split: "train" or "test"
        max_spectra_per_class: If set, limit spectra per class (max 500)

    Returns:
        X: ndarray of shape (n_samples, 40002), float32
        y: ndarray of shape (n_samples,), int64 class labels 0–11
    """
    import h5py

    path = Path(path)
    class_map_path = path / "class_map.json"

    if split == "train":
        h5_path = path / "train.h5"
        if not h5_path.exists():
            raise FileNotFoundError(
                f"{h5_path} not found. Run data/libs_benchmark/download.py to fetch the dataset."
            )
        with h5py.File(h5_path, "r") as f:
            # --- class labels (1-based ore-type IDs from Class/1 array) ---
            raw_labels = np.array(f["Class"]["1"]).astype(np.int64)  # (n_total,) 1-based

            # --- spectra (concatenate all sample spectra in sorted order) ---
            spectra_group = f["Spectra"]
            sample_keys = sorted(spectra_group.keys())
            spectra_list = []
            label_list = []
            n_per_key = spectra_group[sample_keys[0]].shape[1]  # 500 per key
            for i, key in enumerate(sample_keys):
                key_labels = raw_labels[i * n_per_key: (i + 1) * n_per_key]
                try:
                    mat_data = spectra_group[key][:, :]  # (40002, n_spectra)
                    n = mat_data.shape[1]
                    if max_spectra_per_class is not None:
                        n = min(n, max_spectra_per_class)
                    spectra_list.append(mat_data[:, :n].T.astype(np.float32))
                    label_list.append(key_labels[:n])
                except OSError:
                    # Skip corrupted/unreadable samples
                    continue

            X = np.vstack(spectra_list)  # (n_total, 40002)
            raw_labels = np.concatenate(label_list)
            y = (raw_labels - 1)  # convert 1-based to 0-based

            # Collect class names from support tables or use integer labels
            unique_class_ids = sorted(set(raw_labels.tolist()))
            class_map = {str(int(c) - 1): f"class_{int(c):02d}" for c in unique_class_ids}

        # Save class_map.json
        with open(class_map_path, "w") as cm:
            json.dump(class_map, cm, indent=2)

        return X, y

    else:  # test split
        h5_path = path / "test.h5"
        labels_path = path / "test_labels.csv"
        if not h5_path.exists():
            raise FileNotFoundError(f"{h5_path} not found.")
        with h5py.File(h5_path, "r") as f:
            unknown_group = f["UNKNOWN"]
            sample_keys = sorted(unknown_group.keys())
            spectra_list = []
            for key in sample_keys:
                mat_data = np.array(unknown_group[key])  # (40002, n_spectra)
                spectra_list.append(mat_data.T.astype(np.float32))
            X = np.vstack(spectra_list)

        if labels_path.exists():
            y = pd.read_csv(labels_path, header=None).values.squeeze().astype(np.int64) - 1
        else:
            y = np.full(X.shape[0], -1, dtype=np.int64)

        return X, y


def load_mesbah_cap(
    path: Union[str, Path],
    target: str = "T_rot",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load Mesbah Lab CAP dataset (N2 OES + Trot/Tvib/operating conditions).

    CSV structure (no named header — first row IS data):
        Columns 0-50  : 51 OES spectral channels (N2 2nd-positive transition)
        Column  51    : power  (W)
        Column  52    : flow   (slm)
        Column  53    : T_rot  (K, rotational temperature)
        Column  54    : T_vib  (K, vibrational temperature)
        Column  55    : substrate_type (0 = glass, 1 = metal)

    Args:
        path: Path to the CSV file (dat_train.csv or dat_test.csv)
        target: Target column name; one of 'T_rot', 'T_vib', 'substrate_type',
                'power', 'flow'

    Returns:
        X: ndarray of shape (n_samples, 51), spectral features, float32
        y: ndarray of shape (n_samples,), target values
    """
    # Column index mapping (0-based, after reading without header)
    _TARGET_COL = {
        "power": 51,
        "flow": 52,
        "T_rot": 53,
        "T_vib": 54,
        "substrate_type": 55,
    }
    _NON_SPECTRAL = set(_TARGET_COL.values())  # columns 51-55
    N_SPECTRAL = 51  # columns 0-50

    if target not in _TARGET_COL:
        raise ValueError(
            f"Unknown target '{target}'. Choose from: {list(_TARGET_COL.keys())}"
        )

    df = pd.read_csv(path, header=None)  # read without header (first row is data)
    X = df.iloc[:, :N_SPECTRAL].values.astype(np.float32)
    y = df.iloc[:, _TARGET_COL[target]].values

    if target == "substrate_type":
        y = y.astype(np.int64)
    else:
        y = y.astype(np.float32)

    return X, y


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
