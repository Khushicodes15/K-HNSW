"""Adapter stub wrapping the external hnswlib C++ library.

Heavy external libraries are permitted only in `khnsw.adapters`.
All algorithm methods remain unimplemented stubs per project scaffold rules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import hnswlib
except ImportError:  # pragma: no cover
    hnswlib = None  # type: ignore[assignment]


class HnswlibAdapter:
    """Baseline adapter interfacing with hnswlib.Index."""

    def __init__(self, space: str = "l2", dim: int = 128) -> None:
        self._space = space
        self._dim = dim
        self._index: Any = None

    @property
    def dim(self) -> int:
        """Return the vector dimensionality."""
        return self._dim

    def add(self, ids: NDArray[np.int64], vectors: NDArray[np.float32]) -> None:
        """Add vectors into the hnswlib index."""
        raise NotImplementedError("HnswlibAdapter.add is a stub; algorithm not yet implemented.")

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
        ef: int = 50,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Query top-k approximate nearest neighbors using hnswlib."""
        raise NotImplementedError("HnswlibAdapter.search is a stub; algorithm not yet implemented.")

    def move(self, node_id: int, new_vector: NDArray[np.float32]) -> None:
        """Update node coordinates (e.g. via mark-deleted and re-insert)."""
        raise NotImplementedError("HnswlibAdapter.move is a stub; algorithm not yet implemented.")
