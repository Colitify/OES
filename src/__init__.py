"""OES Spectral Analysis Toolkit."""

from .data_loader import SpectralDataset, load_mesbah_cap, load_bosch_oes
from .preprocessing import Preprocessor, als_baseline, preprocess_spectrum
from .features import FeatureExtractor, extract_peaks
from .evaluation import evaluate_model, compare_models
from .spatial import compute_wafer_uniformity, interpolate_wafer_map, link_oes_spatial
from .data_loader import load_wafer_spatial, parse_experiment_key

__version__ = "1.0.0"
__all__ = [
    "SpectralDataset",
    "load_mesbah_cap",
    "load_bosch_oes",
    "load_wafer_spatial",
    "parse_experiment_key",
    "Preprocessor",
    "als_baseline",
    "preprocess_spectrum",
    "FeatureExtractor",
    "extract_peaks",
    "evaluate_model",
    "compare_models",
    "compute_wafer_uniformity",
    "interpolate_wafer_map",
    "link_oes_spatial",
]
