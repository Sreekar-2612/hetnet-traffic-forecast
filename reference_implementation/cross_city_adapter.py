"""
Cross-city evaluation (Phase 4 stub).

Implement a second loader that maps another grid (e.g. China Mobile CDR) to the same
tensor shapes (T, N) and activity definition, then zero-shot or fine-tune TASTF.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class CityTrafficLoader(Protocol):
    def load(self, data_dir: str) -> tuple[np.ndarray, dict]:
        """Returns (T, N) raw activity matrix and metadata dict."""


def not_implemented_loader(data_dir: str) -> tuple[np.ndarray, dict]:
    raise NotImplementedError(
        "Add a dataset-specific parser for your CDR export; "
        "return a (timesteps, nodes) float32 matrix aligned with Milan preprocessing."
    )
