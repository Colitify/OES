"""Data loading module for OES spectral data."""

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


def load_wafer_spatial(
    path: Union[str, Path],
    points: int = 89,
) -> pd.DataFrame:
    """Load wafer spatial etch-measurement CSV (89-point or 9-point layout).

    Expected columns: experiment_key, lot_number, wafer_number, X, Y,
    preox_thickness, postox_thickness, postox_thickness_nan, stepheight,
    oxide_etch, si_etch.

    Args:
        path: Path to CSV file (e.g. Si_Oxide_etch_89_points.csv).
        points: Expected number of measurement points per wafer (89 or 9).

    Returns:
        DataFrame with all columns, coordinates in micrometres.
    """
    path = Path(path)
    df = pd.read_csv(path)
    required = {"experiment_key", "X", "Y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def parse_experiment_key(key: str) -> Tuple[str, str]:
    """Parse experiment_key into (day_filename, wafer_group).

    Example:
        '2024-07-02_01' -> ('Day_2024_07_02.nc', 'Wafer_01')

    Args:
        key: Experiment key string in format 'YYYY-MM-DD_WW'.

    Returns:
        Tuple of (NetCDF day filename, wafer group name).
    """
    parts = key.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid experiment_key format: '{key}'. Expected 'YYYY-MM-DD_WW'.")
    date_str, wafer_num = parts
    day_file = f"Day_{date_str.replace('-', '_')}.nc"
    wafer_group = f"Wafer_{wafer_num}"
    return day_file, wafer_group


def label_process_phases(
    sf6_flow: np.ndarray,
    c4f8_flow: np.ndarray,
    threshold_pct: float = 5.0,
) -> np.ndarray:
    """Label each timestep as etch (1), passivation (2), or idle (0).

    Uses gas flow rates as proxy: SF₆ = etch gas, C₄F₈ = passivation gas.
    When both gases flow, the dominant gas determines the phase.

    Args:
        sf6_flow: SF₆ gas flow rate array (T,)
        c4f8_flow: C₄F₈ gas flow rate array (T,)
        threshold_pct: Minimum flow as % of max to consider gas "active"

    Returns:
        Integer labels (T,): 0=idle, 1=etch, 2=passivation
    """
    sf6_max = sf6_flow.max() if sf6_flow.max() > 0 else 1.0
    c4f8_max = c4f8_flow.max() if c4f8_flow.max() > 0 else 1.0

    sf6_active = sf6_flow > (threshold_pct / 100.0 * sf6_max)
    c4f8_active = c4f8_flow > (threshold_pct / 100.0 * c4f8_max)

    labels = np.zeros(len(sf6_flow), dtype=np.int64)
    # Etch where SF6 is active and dominant (or C4F8 inactive)
    labels[sf6_active & ~c4f8_active] = 1
    labels[sf6_active & c4f8_active & (sf6_flow >= c4f8_flow)] = 1
    # Passivation where C4F8 is active and dominant
    labels[c4f8_active & ~sf6_active] = 2
    labels[c4f8_active & sf6_active & (c4f8_flow > sf6_flow)] = 2

    return labels


def find_gas_column(
    feature_names: List[str],
    explicit_name: Optional[str],
    search_patterns: List[str],
) -> Optional[int]:
    """Find gas flow column index by name or pattern matching."""
    if explicit_name:
        try:
            return feature_names.index(explicit_name)
        except ValueError:
            return None
    for pattern in search_patterns:
        for i, name in enumerate(feature_names):
            if pattern.lower() in name.lower():
                return i
    return None


def _detect_bosch_gas_pair(
    proc: np.ndarray,
    feature_names: List[str],
) -> Tuple[Optional[int], Optional[int]]:
    """Auto-detect etch/passivation gas columns from anti-correlated flow pairs.

    In the Bosch process, SF6 (etch) and C4F8 (passivation) alternate.
    Finds the pair of GasNFlow columns with the strongest negative correlation
    and sufficient variability (std > 5% of max).

    Returns:
        (etch_col_idx, passivation_col_idx) or (None, None) if not found
    """
    gas_cols = [i for i, n in enumerate(feature_names) if "gas" in str(n).lower() and "flow" in str(n).lower()]
    if len(gas_cols) < 2:
        return None, None

    # Filter to variable gas channels (std > 5% of max)
    variable = [i for i in gas_cols if proc[:, i].std() > 0.05 * proc[:, i].max() and proc[:, i].max() > proc[:, i].min()]
    if len(variable) < 2:
        return None, None

    # Find most anti-correlated pair
    best_corr, best_pair = 0.0, (None, None)
    for a in range(len(variable)):
        for b in range(a + 1, len(variable)):
            corr = np.corrcoef(proc[:, variable[a]], proc[:, variable[b]])[0, 1]
            if corr < best_corr:
                best_corr = corr
                best_pair = (variable[a], variable[b])

    if best_corr > -0.3:  # require meaningful anti-correlation
        return None, None

    # The channel active more often is likely C4F8 (passivation dominates in Bosch)
    # The channel active less often is likely SF6 (etch steps are shorter)
    i, j = best_pair
    frac_i = (proc[:, i] > 0.1 * proc[:, i].max()).mean()
    frac_j = (proc[:, j] > 0.1 * proc[:, j].max()).mean()
    if frac_i < frac_j:
        return i, j  # i=etch(SF6), j=passivation(C4F8)
    else:
        return j, i  # j=etch(SF6), i=passivation(C4F8)


def load_bosch_multi_wafer(
    path: Union[str, Path],
    max_wafers: int = 10,
    max_timesteps: Optional[int] = None,
    gas_sf6_col: Optional[str] = None,
    gas_c4f8_col: Optional[str] = None,
) -> dict:
    """Load BOSCH OES data from multiple wafers with process phase labels.

    Iterates over Day_*.nc files and wafer groups, concatenates spectra,
    and generates process phase labels from gas flow data.

    Args:
        path: Directory containing Day_*.nc and Process_data.nc
        max_wafers: Maximum number of wafers to load (for memory)
        max_timesteps: If set, subsample each wafer to this many timesteps
        gas_sf6_col: Column name for SF6 flow in process params (auto-detected if None)
        gas_c4f8_col: Column name for C4F8 flow (auto-detected if None)

    Returns:
        dict with keys: spectra (N, 3648), wavelengths (3648,), labels (N,),
        wafer_ids (N,), process_params (N, n_feat), feature_names list[str]
    """
    path = Path(path)
    day_files = sorted(path.glob("Day_*.nc"))
    if not day_files:
        raise FileNotFoundError(f"No Day_*.nc in {path}")

    all_spectra, all_labels, all_wafer_ids, all_params = [], [], [], []
    wavelengths = None
    feature_names = []
    wafer_count = 0

    for day_path in day_files:
        if wafer_count >= max_wafers:
            break

        # Enumerate wafer groups in this day file
        import netCDF4 as nc_lib
        with nc_lib.Dataset(str(day_path)) as ds:
            wafer_keys = sorted(ds.groups.keys())

        for wk in wafer_keys:
            if wafer_count >= max_wafers:
                break
            data = load_bosch_oes(str(path), day_file=day_path.name, wafer_key=wk)

            if wavelengths is None:
                wavelengths = data["wavelengths"]
                feature_names = data["process_feature_names"]

            spectra = data["spectra"]
            proc = data["process_params"]

            if max_timesteps and len(spectra) > max_timesteps:
                idx = np.linspace(0, len(spectra) - 1, max_timesteps, dtype=int)
                spectra = spectra[idx]
                proc = proc[idx] if proc.shape[1] > 0 else proc

            sf6_col_idx = find_gas_column(feature_names, gas_sf6_col, ["sf6", "SF6", "sf_6"])
            c4f8_col_idx = find_gas_column(feature_names, gas_c4f8_col, ["c4f8", "C4F8", "c_4f_8"])

            # Fallback: auto-detect etch/passivation gases from anti-correlated flow pairs
            if (sf6_col_idx is None or c4f8_col_idx is None) and proc.shape[1] > 0:
                sf6_col_idx, c4f8_col_idx = _detect_bosch_gas_pair(proc, feature_names)

            if sf6_col_idx is not None and c4f8_col_idx is not None and proc.shape[1] > 0:
                labels = label_process_phases(proc[:, sf6_col_idx], proc[:, c4f8_col_idx])
            else:
                labels = np.zeros(len(spectra), dtype=np.int64)

            all_spectra.append(spectra)
            all_labels.append(labels)
            all_wafer_ids.append(np.full(len(spectra), wafer_count, dtype=np.int64))
            all_params.append(proc)
            wafer_count += 1

    return {
        "spectra": np.concatenate(all_spectra),
        "wavelengths": wavelengths,
        "labels": np.concatenate(all_labels),
        "wafer_ids": np.concatenate(all_wafer_ids),
        "process_params": np.concatenate(all_params) if all_params[0].shape[1] > 0 else np.zeros((sum(len(s) for s in all_spectra), 0)),
        "feature_names": feature_names,
    }


