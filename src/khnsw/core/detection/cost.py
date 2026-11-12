"""Cost metering and distance accounting for K-HNSW kinetic detection."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class CostMeter:
    """Monotonic counter and computational meter for geometric distance evaluations.

    Every pairwise vector distance calculated during graph certification and affected-edge
    detection must pass through an instance of this meter to enforce exact algorithmic
    budget tracking.
    """

    def __init__(self) -> None:
        """Initialize cost meter with zero counted operations."""
        self._count: int = 0

    @property
    def count(self) -> int:
        """Total cumulative pairwise distance computations executed through this meter."""
        return self._count

    def dist(
        self,
        a: NDArray[np.floating[Any]],
        b: NDArray[np.floating[Any]],
    ) -> float:
        """Compute squared Euclidean distance between two vectors in float64 and increment count.

        Parameters
        ----------
        a : NDArray[np.floating]
            First spatial vector, 1D array.
        b : NDArray[np.floating]
            Second spatial vector, 1D array.

        Returns
        -------
        float
            Squared L2 norm ||a - b||^2 computed in float64 precision.
        """
        self._count += 1
        va = np.asarray(a, dtype=np.float64)
        vb = np.asarray(b, dtype=np.float64)
        diff = va - vb
        return float(np.dot(diff, diff))

    def reset(self) -> None:
        """Reset the internal operation counter to zero."""
        self._count = 0

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable dictionary snapshot of current meter metrics."""
        return {"count": self._count}
