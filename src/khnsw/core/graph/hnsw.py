"""HNSW graph index implementation wrapping GraphStore.

Implements HNSW Algorithms 1 (insertion) and 5 (k-NN search) from
Malkov & Yashunin 2018, using the existing search_layer (Algorithm 2)
and select_neighbors_heuristic (Algorithm 4).

Pure Python standard library and NumPy only (browser/Pyodide safe).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from khnsw.core.graph.levels import LevelGenerator
from khnsw.core.graph.store import GraphStore
from khnsw.core.search.heuristic import select_neighbors_heuristic
from khnsw.core.search.search_layer import DistanceCounter, search_layer


class HNSWIndex:
    """Hierarchical Navigable Small World graph index.

    Wraps a :class:`GraphStore` with HNSW construction (Algorithm 1) and
    k-NN search (Algorithm 5).  Conforms to the :class:`GraphIndex` protocol
    defined in ``khnsw.core.interfaces``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        dim: int,
        m: int = 16,
        ef_construction: int = 200,
        seed: int | None = None,
        m0: int | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if m <= 1:
            raise ValueError(f"m must be > 1, got {m}")
        if ef_construction < 1:
            raise ValueError(f"ef_construction must be >= 1, got {ef_construction}")

        resolved_m0 = m0 if m0 is not None else 2 * m
        if resolved_m0 <= 0:
            raise ValueError(f"m0 must be positive, got {resolved_m0}")

        self._store = GraphStore(dim=dim, m=m, m0=resolved_m0)
        self._level_gen = LevelGenerator(m=m, seed=seed)
        self._ef_construction = ef_construction
        self._dist_counter = DistanceCounter(0)
        self._versions: dict[int, int] = {}

    # ------------------------------------------------------------------
    # ANNIndex.dim (property)
    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        """Vector dimensionality of the index."""
        return self._store.dim

    # ------------------------------------------------------------------
    # ANNIndex.add
    # ------------------------------------------------------------------
    def add(self, ids: NDArray[np.int64], vectors: NDArray[np.float32]) -> None:
        """Insert a batch of vectors into the HNSW index (Algorithm 1).

        Preconditions
        -------------
        - ``len(ids) == len(vectors)``
        - All ids non-negative and not already in the index.
        - ``vectors.shape[1] == self.dim``
        """
        ids_arr = np.asarray(ids, dtype=np.int64).ravel()
        vecs_arr = np.asarray(vectors, dtype=np.float32)

        if vecs_arr.ndim == 1 and len(ids_arr) == 1:
            vecs_arr = vecs_arr.reshape(1, -1)

        if ids_arr.shape[0] != vecs_arr.shape[0]:
            raise ValueError(f"len(ids) ({ids_arr.shape[0]}) != len(vectors) ({vecs_arr.shape[0]})")
        if vecs_arr.ndim != 2 or vecs_arr.shape[1] != self._store.dim:
            raise ValueError(
                f"vectors must have shape (N, {self._store.dim}), got {vecs_arr.shape}"
            )

        # Validate ids: non-negative, unique, and not already in the index
        if np.any(ids_arr < 0):
            raise ValueError("All ids must be non-negative")
        if len(np.unique(ids_arr)) != len(ids_arr):
            raise ValueError("ids must be unique within the batch")
        for nid in ids_arr:
            if int(nid) in self._store:
                raise ValueError(f"node_id {int(nid)} already exists in the index")

        for i in range(len(ids_arr)):
            self._insert_one(int(ids_arr[i]), vecs_arr[i])

    def _insert_one(self, node_id: int, vector: NDArray[np.float32]) -> None:
        """Insert a single node following HNSW Algorithm 1."""
        node_level = self._level_gen.sample()
        self._store.add_node(node_id, vector, node_level)
        self._versions[node_id] = 0

        ep = self._store.entry_point()

        # If this is the first node (or the only node), nothing to connect.
        if self._store.__len__() == 1:
            return

        # After add_node, entry_point may have changed to node_id if its
        # level is the new maximum.  For wiring we need the *previous* EP.
        if ep == node_id:
            # This node became the new entry point.  We still need to wire
            # it using the old entry point.  Unfortunately GraphStore already
            # set the EP to node_id.  Find the old EP: the node with the
            # highest level that is NOT node_id.
            old_ep = self._find_old_entry_point(node_id)
            if old_ep is None:
                # Should not happen since size > 1, but be safe.
                return  # pragma: no cover
            ep = old_ep

        current_max_level = self._store.node_level(ep)

        # Phase 1: Greedy descent from current entry point down to
        # node_level + 1 (layers above the insertion level).
        current_ep = ep
        for lc in range(current_max_level, node_level, -1):
            results = search_layer(
                self._store,
                vector,
                current_ep,
                ef=1,
                level=lc,
                counter=self._dist_counter,
            )
            if results:
                current_ep = results[0].node_id

        # Phase 2: At each level from min(node_level, current_max_level)
        # down to 0, run search_layer with ef_construction, then select
        # neighbors using the heuristic, and wire edges.
        for lc in range(min(node_level, current_max_level), -1, -1):
            results = search_layer(
                self._store,
                vector,
                current_ep,
                ef=self._ef_construction,
                level=lc,
                counter=self._dist_counter,
            )

            # Select neighbors using the heuristic (Algorithm 4)
            candidate_ids = [r.node_id for r in results]
            max_m = self._store.max_m_for_level(lc)
            selected = select_neighbors_heuristic(
                self._store,
                node_id,
                candidate_ids,
                m=max_m,
                keep_pruned_connections=True,
            )

            # Set forward neighbors for the new node at this level
            self._store.set_neighbors(node_id, lc, selected)

            # Add reverse connections: for each selected neighbor, add
            # node_id to their neighbor list.  If that makes them overfull,
            # shrink with the heuristic via set_neighbors.
            for nbr in selected:
                nbr_neighbors = list(self._store.neighbors(nbr, lc))
                if node_id not in nbr_neighbors:
                    nbr_neighbors.append(node_id)

                if len(nbr_neighbors) > max_m:
                    # Shrink using heuristic
                    pruned = select_neighbors_heuristic(
                        self._store,
                        nbr,
                        nbr_neighbors,
                        m=max_m,
                        keep_pruned_connections=True,
                    )
                    self._store.set_neighbors(nbr, lc, pruned)
                else:
                    self._store.set_neighbors(nbr, lc, nbr_neighbors)

            # Update entry points for next level down
            if results:
                current_ep = results[0].node_id

    def _find_old_entry_point(self, exclude_id: int) -> int | None:
        """Find the node with the highest level that is not *exclude_id*."""
        best_id: int | None = None
        best_level = -1
        for nid in self._store.ids():
            if nid == exclude_id:
                continue
            nl = self._store.node_level(nid)
            if nl > best_level:
                best_level = nl
                best_id = nid
        return best_id

    # ------------------------------------------------------------------
    # ANNIndex.search
    # ------------------------------------------------------------------
    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        ef: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """k-NN search following HNSW Algorithm 5.

        Preconditions
        -------------
        - ``ef >= k >= 1``
        - ``query.shape == (self.dim,)``

        Returns at most ``min(k, size())`` results sorted by ascending
        squared L2 distance.
        """
        q = np.asarray(query, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != self._store.dim:
            raise ValueError(f"query.shape must be ({self._store.dim},), got {q.shape}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if ef < k:
            raise ValueError(f"ef must be >= k, got ef={ef}, k={k}")

        # Empty index → return empty arrays
        ep = self._store.entry_point()
        if ep is None:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
            )

        current_max_level = self._store.node_level(ep)

        # Phase 1: Greedy descent from top level down to level 1
        current_ep = ep
        for lc in range(current_max_level, 0, -1):
            results = search_layer(
                self._store,
                q,
                current_ep,
                ef=1,
                level=lc,
                counter=self._dist_counter,
            )
            if results:
                current_ep = results[0].node_id

        # Phase 2: Search at level 0 with full ef
        results = search_layer(
            self._store,
            q,
            current_ep,
            ef=ef,
            level=0,
            counter=self._dist_counter,
        )

        # Return top-k (or fewer if index has fewer points)
        actual_k = min(k, len(results))
        top = results[:actual_k]

        result_ids = np.array([r.node_id for r in top], dtype=np.int64)
        result_dists = np.array([r.distance for r in top], dtype=np.float32)
        return result_ids, result_dists

    # ------------------------------------------------------------------
    # ANNIndex.move
    # ------------------------------------------------------------------
    def move(self, node_id: int, new_vector: NDArray[np.float32]) -> None:
        """Update spatial coordinates of an existing node (no topology rewiring).

        Preconditions
        -------------
        - ``node_id`` exists in the index.
        - ``new_vector.shape == (self.dim,)``

        Postconditions
        --------------
        - ``vector(node_id)`` returns ``new_vector``.
        - ``version(node_id)`` incremented by 1.
        """
        node_id = int(node_id)
        if node_id not in self._store:
            raise KeyError(f"node_id {node_id} does not exist in the index")
        nv = np.asarray(new_vector, dtype=np.float32)
        if nv.ndim != 1 or nv.shape[0] != self._store.dim:
            raise ValueError(f"new_vector.shape must be ({self._store.dim},), got {nv.shape}")
        self._store.set_vector(node_id, nv)
        self._versions[node_id] += 1

    # ------------------------------------------------------------------
    # ANNIndex.size / contains / remove
    # ------------------------------------------------------------------
    def size(self) -> int:
        """Return the total number of indexed points."""
        return len(self._store)

    def contains(self, node_id: int) -> bool:
        """Check whether a node_id is currently present in the index."""
        return int(node_id) in self._store

    def remove(self, node_id: int) -> None:
        """Remove a node from the index (not yet implemented)."""
        raise NotImplementedError("remove is not yet implemented")

    # ------------------------------------------------------------------
    # GraphIndex read-only topology methods
    # ------------------------------------------------------------------
    def neighbors(self, node_id: int, level: int) -> list[int]:
        """Return outgoing neighbor identifiers at a specific layer."""
        return self._store.neighbors(node_id, level)

    def in_neighbors(self, node_id: int, level: int) -> list[int]:
        """Return incoming neighbor identifiers (reverse edges) at a layer."""
        return self._store.in_neighbors(node_id, level)

    def node_level(self, node_id: int) -> int:
        """Return the highest layer level assigned to the given node."""
        return self._store.node_level(node_id)

    def max_level(self) -> int:
        """Return the current maximum layer level (-1 if empty)."""
        return self._store.max_level()

    def entry_point(self) -> int | None:
        """Return the entry point node ID, or None if empty."""
        return self._store.entry_point()

    def vector(self, node_id: int) -> NDArray[np.float32]:
        """Retrieve the current vector coordinates for a node."""
        return self._store.vector(node_id)

    def version(self, node_id: int) -> int:
        """Return the monotonic update version of a node."""
        node_id = int(node_id)
        if node_id not in self._versions:
            raise KeyError(f"node_id {node_id} does not exist in the index")
        return self._versions[node_id]

    def ids(self) -> list[int]:
        """Return all active node identifiers."""
        return self._store.ids()

    def max_degree(self, level: int) -> int:
        """Return maximum allowable degree at a given layer level."""
        return self._store.max_m_for_level(level)

    @property
    def distance_computations(self) -> int:
        """Cumulative counter of distance computations performed.

        Note: counts only distances evaluated inside ``search_layer``.
        Distances computed internally by ``select_neighbors_heuristic``
        (candidate ranking and diversity checks) are *not* counted because
        the heuristic does not expose a counter.
        """
        return self._dist_counter.value

    def snapshot(self) -> dict[str, Any]:
        """Export a serializable snapshot of graph topology and metadata."""
        return self._store.snapshot()
