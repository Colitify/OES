"""Target variable transformation for low-concentration elements (ML-008).

Based on the FORTH team (1st place, LIBS 2022 competition, Siozos et al. 2023).
Logit transformation maps [0, 100] wt% to (-inf, +inf), expanding the
low-concentration region and preventing negative predictions.
"""

import numpy as np
from typing import List, Optional


class LogitTargetTransformer:
    """Applies logit transform to selected target columns.

    For each designated element, maps concentration y in (0, 100) wt% via:
        logit(y) = ln(y / (100 - y))

    Inverse (logistic) maps back:
        inv(s) = 100 / (1 + exp(-s))

    Args:
        elements: Element names to transform. If None, transform all targets.
        eps_low:  Clip values below eps_low before transform (avoids log(0)).
        eps_high: Clip values above eps_high before transform (avoids log(inf)).
    """

    def __init__(
        self,
        elements: Optional[List[str]] = None,
        eps_low: float = 1e-3,
        eps_high: float = 99.999,
    ):
        self.elements = elements
        self.eps_low = eps_low
        self.eps_high = eps_high
        self._transform_mask: Optional[np.ndarray] = None  # bool array, set in fit()

    def fit(self, y: np.ndarray, target_names: Optional[List[str]] = None):
        """Determine which columns to transform.

        Args:
            y: Target matrix (n_samples, n_targets) or (n_samples,)
            target_names: Names of target columns.
        """
        if y.ndim == 1:
            n_targets = 1
        else:
            n_targets = y.shape[1]

        if self.elements is None or target_names is None:
            # Transform all columns
            self._transform_mask = np.ones(n_targets, dtype=bool)
        else:
            self._transform_mask = np.zeros(n_targets, dtype=bool)
            for i, name in enumerate(target_names):
                if name in self.elements:
                    self._transform_mask[i] = True

        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Apply logit transform to designated columns.

        Args:
            y: Target matrix (n_samples, n_targets) or (n_samples,)

        Returns:
            Transformed matrix of same shape.
        """
        if self._transform_mask is None:
            raise ValueError("Call fit() before transform()")

        y_out = y.astype(float).copy()
        single = y_out.ndim == 1

        if single:
            y_out = y_out.reshape(-1, 1)

        for j, do_transform in enumerate(self._transform_mask):
            if do_transform:
                col = np.clip(y_out[:, j], self.eps_low, self.eps_high)
                y_out[:, j] = np.log(col / (100.0 - col))

        return y_out.ravel() if single else y_out

    def inverse_transform(self, y_logit: np.ndarray) -> np.ndarray:
        """Apply logistic (inverse logit) to designated columns.

        Args:
            y_logit: Logit-space matrix (n_samples, n_targets) or (n_samples,)

        Returns:
            Concentrations in [0, 100] wt%.
        """
        if self._transform_mask is None:
            raise ValueError("Call fit() before inverse_transform()")

        y_out = y_logit.astype(float).copy()
        single = y_out.ndim == 1

        if single:
            y_out = y_out.reshape(-1, 1)

        for j, do_transform in enumerate(self._transform_mask):
            if do_transform:
                y_out[:, j] = 100.0 / (1.0 + np.exp(-y_out[:, j]))

        return y_out.ravel() if single else y_out

    def fit_transform(self, y: np.ndarray, target_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(y, target_names).transform(y)
