"""Adapter stub wrapping the external faiss library.

Heavy external libraries are permitted only in `khnsw.adapters`.
All algorithm methods remain unimplemented stubs per project scaffold rules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment]


class FaissAdapter:
    """Baseline adapter interfacing with faiss.IndexHNSWFlat."""

    def __init__(self, dim: int = 128, m: int = 16) -> None:
        self._dim = dim
        self._m = m
        self._index: Any = None

    @property
    def dim(self) -> int:
        """Return the vector dimensionality."""
        return self._dim

    def add(self, ids: NDArray[np.int64], vectors: NDArray[np.float32]) -> None:
        """Add vectors into the faiss index."""
        raise NotImplementedError("FaissAdapter.add is a stub; algorithm not yet implemented.")

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
        ef: int = 50,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Query top-k approximate nearest neighbors using faiss."""
        raise NotImplementedError("FaissAdapter.search is a stub; algorithm not yet implemented.")

    def move(self, node_id: int, new_vector: NDArray[np.float32]) -> None:
        """Update node coordinates in faiss."""
        raise NotImplementedError("FaissAdapter.move is a stub; algorithm not yet implemented.")
