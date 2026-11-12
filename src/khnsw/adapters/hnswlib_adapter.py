"""Adapter wrapping the external hnswlib C++ library.

Heavy external libraries are permitted only in `khnsw.adapters`.
Conforms to the ANNIndex protocol using hnswlib.Index with L2 distance.
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

    def __init__(
        self,
        space: str = "l2",
        dim: int = 128,
        m: int = 16,
        ef_construction: int = 200,
        seed: int | None = None,
        max_elements: int = 10000,
    ) -> None:
        self._space = space
        self._dim = dim
        self._m = m
        self._ef_construction = ef_construction
        self._seed = seed
        self._active_ids: set[int] = set()
        self._index: Any = None

        if hnswlib is not None:
            self._index = hnswlib.Index(space=self._space, dim=self._dim)
            self._index.init_index(
                max_elements=max_elements,
                ef_construction=self._ef_construction,
                M=self._m,
                random_seed=seed if seed is not None else 100,
                allow_replace_deleted=True,
            )

    @property
    def dim(self) -> int:
        """Return the vector dimensionality."""
        return self._dim

    def add(self, ids: NDArray[np.int64], vectors: NDArray[np.float32]) -> None:
        """Add vectors into the hnswlib index."""
        if hnswlib is None:
            raise ImportError("hnswlib is not installed.")
        if len(ids) != len(vectors):
            raise ValueError(f"Length mismatch: {len(ids)} ids vs {len(vectors)} vectors")
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(f"Vectors shape {vectors.shape} does not match dim {self._dim}")

        current_max = self._index.get_max_elements()
        if len(self._active_ids) + len(ids) > current_max:
            self._index.resize_index(max(current_max * 2, len(self._active_ids) + len(ids)))

        ids_arr = np.ascontiguousarray(ids, dtype=np.int64)
        vecs_arr = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index.add_items(vecs_arr, ids_arr)
        self._active_ids.update(ids_arr.tolist())

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
        ef: int = 50,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Query top-k approximate nearest neighbors using hnswlib."""
        if hnswlib is None:
            raise ImportError("hnswlib is not installed.")
        if self.size() == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        k_actual = min(k, self.size())
        self._index.set_ef(max(ef, k_actual))
        q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        labels, dists = self._index.knn_query(q, k=k_actual)
        return labels[0].astype(np.int64), dists[0].astype(np.float32)

    def move(self, node_id: int, new_vector: NDArray[np.float32]) -> None:
        """Update node coordinates via delete-and-reinsert."""
        if hnswlib is None:
            raise ImportError("hnswlib is not installed.")
        if node_id not in self._active_ids:
            raise KeyError(f"Node {node_id} not found in index")
        if new_vector.shape != (self._dim,):
            raise ValueError(f"Vector shape {new_vector.shape} != ({self._dim},)")

        self._index.mark_deleted(node_id)
        vec = np.ascontiguousarray(new_vector.reshape(1, -1), dtype=np.float32)
        self._index.add_items(vec, np.array([node_id], dtype=np.int64), replace_deleted=True)

    def size(self) -> int:
        """Return count of active vectors in the index."""
        return len(self._active_ids)

    def contains(self, node_id: int) -> bool:
        """Check if node_id exists in the index."""
        return node_id in self._active_ids

    def remove(self, node_id: int) -> None:
        """Remove a node from the index."""
        if hnswlib is None:
            raise ImportError("hnswlib is not installed.")
        if node_id not in self._active_ids:
            raise KeyError(f"Node {node_id} not found in index")
        self._index.mark_deleted(node_id)
        self._active_ids.remove(node_id)
