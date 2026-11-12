"""Hierarchical layer assignment via exponential decay distribution.

Pure Python standard library and NumPy only (browser/Pyodide safe).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def compute_default_ml(m: int) -> float:
    """Compute default level decay parameter mL = 1 / ln(M)."""
    if m <= 1:
        raise ValueError(f"m must be > 1 to compute m_l, got {m}")
    return 1.0 / math.log(m)


def assign_level(u: float, m_l: float) -> int:
    """Compute level = floor(-ln(U) * m_L) for U in (0, 1].

    Parameters
    ----------
    u : float
        Uniform random value in half-open interval (0, 1].
    m_l : float
        Exponential decay scale factor (m_l > 0).

    Returns
    -------
    int
        Non-negative integer layer level.
    """
    if not (0.0 < u <= 1.0):
        raise ValueError(f"U must be in (0, 1], got {u}")
    if m_l <= 0.0:
        raise ValueError(f"m_l must be positive, got {m_l}")
    return int(math.floor(-math.log(u) * m_l))


class LevelGenerator:
    """Generates hierarchical levels using exponential decay distribution.

    Formula:
        level = floor(-ln(U) * m_L)
    where U is uniformly drawn from (0, 1], and default m_L = 1 / ln(M).
    """

    def __init__(
        self,
        m: int = 16,
        m_l: float | None = None,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if m <= 1:
            raise ValueError(f"m must be > 1, got {m}")
        self._m = int(m)
        self._m_l = float(m_l) if m_l is not None else compute_default_ml(self._m)
        if self._m_l <= 0.0:
            raise ValueError(f"m_l must be positive, got {self._m_l}")

        if isinstance(seed, np.random.Generator):
            self._rng = seed
        else:
            self._rng = np.random.default_rng(seed)

    @property
    def m(self) -> int:
        """Maximum upper-layer out-degree parameter M."""
        return self._m

    @property
    def m_l(self) -> float:
        """Exponential decay scale parameter m_L."""
        return self._m_l

    def sample(self) -> int:
        """Sample a single layer level as a non-negative integer."""
        # rng.random() returns float in [0.0, 1.0).
        # 1.0 - rng.random() yields U in (0.0, 1.0], strictly excluding 0.0.
        u = 1.0 - float(self._rng.random())
        while u <= 0.0:
            u = 1.0 - float(self._rng.random())
        return assign_level(u, self._m_l)

    def __call__(self) -> int:
        """Convenience invocation to sample a level."""
        return self.sample()

    def sample_many(self, size: int) -> Sequence[int]:
        """Sample a sequence of layer levels."""
        if size < 0:
            raise ValueError(f"size must be non-negative, got {size}")
        return [self.sample() for _ in range(size)]
