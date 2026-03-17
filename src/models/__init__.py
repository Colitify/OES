"""Machine learning models for spectral analysis."""

import functools

from .traditional import get_traditional_models, train_traditional_model
from .deep_learning import Conv1DRegressor, LSTMRegressor, TransformerRegressor, train_deep_model


@functools.lru_cache(maxsize=1)
def get_safe_device() -> str:
    """Get a device confirmed to work with PyTorch. Result is cached.

    Returns "cuda" if CUDA is available and functional, "cpu" otherwise.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        test_tensor = torch.zeros(1, device="cuda")
        _ = test_tensor + 1
        return "cuda"
    except Exception:
        return "cpu"


__all__ = [
    "get_traditional_models",
    "train_traditional_model",
    "Conv1DRegressor",
    "LSTMRegressor",
    "TransformerRegressor",
    "train_deep_model",
    "get_safe_device",
]
