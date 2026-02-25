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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        wavelengths_nm: ndarray of shape (40002,), float64, wavelengths in nm
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
            # --- wavelengths ---
            wl_group = f["Wavelengths"]
            wl_key = list(wl_group.keys())[0]
            wavelengths_nm = np.array(wl_group[wl_key], dtype=np.float64)  # (40002,)

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

        return X, y, wavelengths_nm

    else:  # test split
        h5_path = path / "test.h5"
        labels_path = path / "test_labels.csv"
        if not h5_path.exists():
            raise FileNotFoundError(f"{h5_path} not found.")
        with h5py.File(h5_path, "r") as f:
            # Read wavelengths from test.h5 if available (same grid as train)
            try:
                wl_group = f["Wavelengths"]
                wl_key = list(wl_group.keys())[0]
                wavelengths_nm = np.array(wl_group[wl_key], dtype=np.float64)
            except KeyError:
                wavelengths_nm = np.linspace(200.0, 1000.0, 40002)
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

        return X, y, wavelengths_nm


def load_mesbah_cap(
    path: Union[str, Path],
    target: str = "T_rot",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        wavelengths_nm: ndarray of shape (51,), approximate N2 2nd-positive band wavelengths
    """
    # Column index mapping (0-based, after reading without header)
    _TARGET_COL = {
        "power": 51,
        "flow": 52,
        "T_rot": 53,
        "T_vib": 54,
        "substrate_type": 55,
    }
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

    # N2 2nd-positive transition wavelengths (approximate range 300-420 nm, 51 channels)
    wavelengths_nm = np.linspace(296.2, 421.6, N_SPECTRAL)

    return X, y, wavelengths_nm


def load_bosch_oes(
    path: Union[str, Path],
    wafer_key: Optional[str] = None,
    day_file: Optional[str] = None,
) -> dict:
    """Load BOSCH plasma etching OES dataset from NetCDF files.

    File structure:
      <day>.nc / <WaferGroup>:
        times       : (T,)       Unix epoch timestamps (s)
        wavelengths : (3648,)    nm, range ~185.9–884.0 nm
        data        : (T, 3648)  uint16 OES intensity counts at 25 Hz

      Process_data.nc / <day>_<WaferGroup>:
        times       : (Tp,)      relative timestamps (s from experiment start)
        feature     : (n_feat,)  string feature names
        data        : (Tp, n_feat) uint16 process parameter counts

    Args:
        path: Directory containing Day_*.nc and Process_data.nc files.
        wafer_key: Wafer group name (e.g. 'Wafer_01'). Defaults to first group.
        day_file: Name of the Day NetCDF file (e.g. 'Day_2024_07_02.nc').
                  Defaults to the first Day_*.nc file found.

    Returns:
        dict with keys:
          'spectra'       : (T, 3648) float32 OES intensities
          'wavelengths'   : (3648,) float64 wavelengths in nm
          'timestamps'    : (T,) float64 Unix epoch timestamps
          'process_params': (T, n_features) float32 process params interpolated
                            to OES time grid (nearest-neighbour)
          'process_feature_names': list[str] — names of process parameter columns
          'day_file'      : str — loaded day filename
          'wafer_key'     : str — loaded wafer group key
    """
    try:
        import netCDF4 as nc_lib
    except ImportError:
        raise ImportError("netCDF4 is required. Install with: pip install netCDF4")

    path = Path(path)

    # Find Day NetCDF file
    if day_file is None:
        day_files = sorted(path.glob("Day_*.nc"))
        if not day_files:
            raise FileNotFoundError(f"No Day_*.nc files found in {path}")
        day_path = day_files[0]
    else:
        day_path = path / day_file

    # Load OES spectra from Day file
    with nc_lib.Dataset(str(day_path)) as ds:
        if wafer_key is None:
            wafer_key = sorted(ds.groups.keys())[0]
        grp = ds[wafer_key]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            times = np.array(grp["times"]).astype(np.float64)
            wavelengths = np.array(grp["wavelengths"]).astype(np.float64)
            spectra = np.array(grp["data"]).astype(np.float32)  # (T, 3648)

    # Build process_params key name
    stem = day_path.stem  # e.g. 'Day_2024_07_02'
    proc_key = f"{stem}_{wafer_key}"  # e.g. 'Day_2024_07_02_Wafer_01'

    proc_params = None
    proc_feature_names: List[str] = []

    proc_path = path / "Process_data.nc"
    if proc_path.exists():
        try:
            with nc_lib.Dataset(str(proc_path)) as pds:
                if proc_key in pds.groups:
                    pg = pds[proc_key]
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        p_times = np.array(pg["times"]).astype(np.float64)
                        p_data = np.array(pg["data"]).astype(np.float32)  # (Tp, n_feat)
                        p_feat = list(np.array(pg["feature"]).astype(str))
                    proc_feature_names = p_feat

                    # Interpolate process params to OES time grid using nearest-neighbour.
                    # OES uses Unix epoch; process data uses experiment-relative seconds.
                    # Align by normalising both to [0, 1] fractional experiment time.
                    from scipy.interpolate import interp1d
                    oes_rel = times - times[0]              # 0 .. duration_s
                    p_rel_norm = (p_times - p_times[0])     # already relative

                    # Scale process time to match OES duration span
                    oes_dur = oes_rel[-1]
                    p_dur = p_rel_norm[-1]
                    if p_dur > 0:
                        p_rel_scaled = p_rel_norm * (oes_dur / p_dur)
                    else:
                        p_rel_scaled = p_rel_norm

                    proc_params = np.zeros((len(times), p_data.shape[1]), dtype=np.float32)
                    for j in range(p_data.shape[1]):
                        interp = interp1d(
                            p_rel_scaled, p_data[:, j],
                            kind="nearest",
                            fill_value=(p_data[0, j], p_data[-1, j]),
                            bounds_error=False,
                        )
                        proc_params[:, j] = interp(oes_rel)
        except Exception:
            proc_params = None

    if proc_params is None:
        proc_params = np.zeros((len(times), 0), dtype=np.float32)

    return {
        "spectra": spectra,
        "wavelengths": wavelengths,
        "timestamps": times,
        "process_params": proc_params,
        "process_feature_names": proc_feature_names,
        "day_file": day_path.name,
        "wafer_key": wafer_key,
    }


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
