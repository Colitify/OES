"""OES/LIBS Spectral Analysis Toolkit."""

from .data_loader import SpectralDataset, load_libs_data
from .preprocessing import Preprocessor, als_baseline, preprocess_spectrum
from .features import FeatureExtractor, extract_peaks
from .evaluation import evaluate_model, compare_models

__version__ = "1.0.0"
__all__ = [
    "SpectralDataset",
    "load_libs_data",
    "Preprocessor",
    "als_baseline",
    "preprocess_spectrum",
    "FeatureExtractor",
    "extract_peaks",
    "evaluate_model",
    "compare_models",
]
