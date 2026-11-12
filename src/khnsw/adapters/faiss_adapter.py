"""Adapter wrapping the external FAISS library.

Heavy external libraries are permitted only in `khnsw.adapters`.
Conforms to the ANNIndex protocol using faiss.IndexHNSWFlat wrapped in faiss.IndexIDMap2.
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

    def __init__(
        self,
        dim: int = 128,
        m: int = 16,
        ef_construction: int = 200,
    ) -> None:
        self._dim = dim
        self._m = m
        self._ef_construction = ef_construction
        self._vectors: dict[int, NDArray[np.float32]] = {}
        self._base_index: Any = None
        self._index: Any = None
        if faiss is not None:
            self._rebuild()

    @property
    def dim(self) -> int:
        """Return the vector dimensionality."""
        return self._dim

    def add(self, ids: NDArray[np.int64], vectors: NDArray[np.float32]) -> None:
        """Add vectors into the faiss index."""
        if len(ids) != len(vectors):
            raise ValueError(f"Length mismatch: {len(ids)} ids vs {len(vectors)} vectors")
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(f"Vectors shape {vectors.shape} does not match dim {self._dim}")

        for node_id, vec in zip(ids, vectors, strict=True):
            node_int = int(node_id)
            if node_int in self._vectors:
                raise ValueError(f"Node id {node_int} already exists in index")
            self._vectors[node_int] = np.asarray(vec, dtype=np.float32)

        ids_arr = np.ascontiguousarray(ids, dtype=np.int64)
        vecs_arr = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index.add_with_ids(vecs_arr, ids_arr)

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
        ef: int = 50,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Query top-k approximate nearest neighbors using faiss."""
        if self.size() == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        k_actual = min(k, self.size())
        self._base_index.hnsw.efSearch = max(ef, k_actual)
        q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        dists, indices = self._index.search(q, k_actual)
        return indices[0].astype(np.int64), dists[0].astype(np.float32)

    def move(self, node_id: int, new_vector: NDArray[np.float32]) -> None:
        """Update node coordinates in faiss via index rebuild."""
        if node_id not in self._vectors:
            raise KeyError(f"Node {node_id} not found in index")
        if new_vector.shape != (self._dim,):
            raise ValueError(f"Vector shape {new_vector.shape} != ({self._dim},)")
        self._vectors[node_id] = np.asarray(new_vector, dtype=np.float32)
        self._rebuild()

    def size(self) -> int:
        """Return count of active vectors in the index."""
        return len(self._vectors)

    def contains(self, node_id: int) -> bool:
        """Check if node_id exists in the index."""
        return node_id in self._vectors

    def remove(self, node_id: int) -> None:
        """Remove a node from the index via rebuild."""
        if node_id not in self._vectors:
            raise KeyError(f"Node {node_id} not found in index")
        del self._vectors[node_id]
        self._rebuild()

    def _rebuild(self) -> None:
        """Reconstruct faiss index from active vectors."""
        if faiss is None:
            raise ImportError("faiss is required for FaissAdapter but is not installed.")
        self._base_index = faiss.IndexHNSWFlat(self._dim, self._m)
        self._base_index.hnsw.efConstruction = self._ef_construction
        self._index = faiss.IndexIDMap2(self._base_index)
        if self._vectors:
            ids = np.array(list(self._vectors.keys()), dtype=np.int64)
            vecs = np.ascontiguousarray(
                np.stack(list(self._vectors.values())),
                dtype=np.float32,
            )
            self._index.add_with_ids(vecs, ids)
