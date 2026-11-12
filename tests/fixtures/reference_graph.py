"""TEST DOUBLE. O(n^2) construction. NOT the real HNSW. Do not use for benchmarks.

This module provides a pure in-memory brute-force reference graph test double implementing
the MutableGraphIndex protocol. It constructs ground-truth graphs following HNSW's
heuristic neighbor selection rule and optionally mimics reverse-link appending to generate
predictable certificate violations for test suites.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from khnsw.core.interfaces import MutableGraphIndex


class ReferenceGraph(MutableGraphIndex):
    """Brute-force in-memory reference graph test double.

    TEST DOUBLE. O(n^2) construction. NOT the real HNSW. Do not use for benchmarks.
    """

    def __init__(self, dim: int, M: int = 16) -> None:
        """Initialize empty reference graph.

        Parameters
        ----------
        dim : int
            Vector dimension of indexed points.
        M : int, default=16
            Base maximum out-degree (level 0 allows 2*M, level > 0 allows M).
        """
        self._dim = int(dim)
        self._M = int(M)
        self._vectors: dict[int, NDArray[np.float32]] = {}
        self._levels: dict[int, int] = {}
        self._versions: dict[int, int] = {}
        # level -> u -> list of outgoing neighbors v
        self._adj: dict[int, dict[int, list[int]]] = {}
        # level -> v -> list of incoming neighbors u
        self._rev: dict[int, dict[int, list[int]]] = {}
        self._dist_comps: int = 0
        self._entry_point: int | None = None

    @classmethod
    def from_points(
        cls,
        X: NDArray[np.floating[Any]] | Sequence[Sequence[float]],
        M: int = 16,
        seed: int = 42,
        levels: dict[int, int] | Sequence[int] | NDArray[np.integer[Any]] | None = None,
        mimic_reverse_append: bool = False,
    ) -> ReferenceGraph:
        """Construct reference graph from points using brute-force heuristic neighbor selection.

        Level Assignment:
        -----------------
        When `levels` is None, levels are sampled deterministically using the standard HNSW
        geometric decay distribution:
            u ~ Uniform(0, 1)
            mL = 1.0 / ln(M)  (or 1.0 if M <= 1)
            level = floor(-ln(u) * mL)
        using `np.random.default_rng(seed)`.

        Neighbor Selection Heuristic:
        -----------------------------
        For each node q and each level L from 0 to node_level(q):
        - Candidates C are all other nodes present at layer L (level >= L, c != q).
        - Candidates are evaluated nearest-first by squared Euclidean distance d(q, c).
          Ties in distance are broken deterministically by candidate node ID.
        - Candidate e is accepted if and only if NO already-accepted neighbor r has:
              d(r, e) < d(q, e)   (strict inequality; ties d(r, e) == d(q, e) are accepted).
        - Acceptance continues until max_degree(L) is reached (2*M for L=0, M for L>0).

        Reverse Link Appending (hnswlib mutuallyConnectNewElement mimic):
        -----------------------------------------------------------------
        When `mimic_reverse_append=True`, after forward lists are built across all levels:
        Iterate level-by-level (L = 0..max_level), then over nodes q in ascending ID order,
        and over q's forward neighbors e in forward list order:
        - If q is not already in e's neighbor list at level L:
          - If len(e's neighbors) < max_degree(L):
            q is APPENDED to e's list without running the heuristic.
          - If len(e's neighbors) == max_degree(L):
            The heuristic is rerun over (e's current neighbors + [q]), keeping the top
            accepted candidates up to max_degree(L).
        This mirrors hnswlib and reproduces the effect where fresh graphs already contain
        edges violating kinetic certificates.

        Parameters
        ----------
        X : array-like of shape (N, dim)
            Coordinates of points to index.
        M : int, default=16
            Maximum out-degree parameter.
        seed : int, default=42
            Random seed for deterministic level assignment.
        levels : dict, sequence, or None, default=None
            Optional explicit level assignments per node ID.
        mimic_reverse_append : bool, default=False
            Whether to mimic hnswlib's asymmetric reverse link appending.

        Returns
        -------
        ReferenceGraph
            Constructed reference graph test double.
        """
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n, dim = arr.shape
        graph = cls(dim=dim, M=M)

        if n == 0:
            return graph

        rng = np.random.default_rng(seed)

        # 1. Determine level for each node
        node_levels: dict[int, int] = {}
        if levels is not None:
            if isinstance(levels, dict):
                for i in range(n):
                    node_levels[i] = int(levels.get(i, 0))
            else:
                for i, lvl in enumerate(levels):
                    node_levels[i] = int(lvl)
        else:
            m_l = 1.0 / np.log(M) if M > 1 else 1.0
            for i in range(n):
                u_val = float(rng.uniform(0.0, 1.0))
                u_val = max(u_val, 1e-12)
                lvl = int(np.floor(-np.log(u_val) * m_l))
                node_levels[i] = lvl

        # Initialize node storage
        for i in range(n):
            graph._vectors[i] = arr[i].copy()
            graph._levels[i] = node_levels[i]
            graph._versions[i] = 0

        max_lvl = max(node_levels.values()) if node_levels else 0
        for lvl in range(max_lvl + 1):
            graph._adj[lvl] = {i: [] for i in range(n) if node_levels[i] >= lvl}
            graph._rev[lvl] = {i: [] for i in range(n) if node_levels[i] >= lvl}

        # 2. Forward neighbor selection by heuristic
        for lvl in range(max_lvl + 1):
            nodes_at_lvl = sorted(i for i in range(n) if node_levels[i] >= lvl)
            degree_limit = graph.max_degree(lvl)

            for q in nodes_at_lvl:
                # Candidates: all other nodes at layer >= lvl
                candidates = [c for c in nodes_at_lvl if c != q]
                if not candidates:
                    continue

                # Sort candidates nearest-first by squared L2; tie-break by node ID
                cand_dists = [(graph._dist_sq(q, c), c) for c in candidates]
                cand_dists.sort(key=lambda item: (item[0], item[1]))

                accepted: list[int] = []
                for d_qc, e in cand_dists:
                    # Check rejection condition:
                    # Reject e if any already-accepted r has d(r, e) < d(q, e)
                    reject = False
                    for r in accepted:
                        d_re = graph._dist_sq(r, e)
                        if d_re < d_qc:  # strict inequality: ties are accepted
                            reject = True
                            break
                    if not reject:
                        accepted.append(e)
                        if len(accepted) == degree_limit:
                            break

                graph._adj[lvl][q] = accepted
                for e in accepted:
                    graph._rev[lvl][e].append(q)

        # 3. Optional reverse append pass (mimicking hnswlib mutuallyConnectNewElement)
        if mimic_reverse_append:
            for lvl in range(max_lvl + 1):
                nodes_at_lvl = sorted(i for i in range(n) if node_levels[i] >= lvl)
                degree_limit = graph.max_degree(lvl)

                for q in nodes_at_lvl:
                    forward_nbrs = list(graph._adj[lvl][q])
                    for e in forward_nbrs:
                        if q in graph._adj[lvl][e]:
                            continue  # already connected

                        if len(graph._adj[lvl][e]) < degree_limit:
                            # Append without running heuristic
                            graph._adj[lvl][e].append(q)
                            graph._rev[lvl][q].append(e)
                        else:
                            # Full list: rerun heuristic over (current neighbors + q)
                            cand_list = list(graph._adj[lvl][e]) + [q]
                            cand_dists = [(graph._dist_sq(e, c), c) for c in cand_list]
                            cand_dists.sort(key=lambda item: (item[0], item[1]))

                            new_accepted: list[int] = []
                            for d_ec, c in cand_dists:
                                reject = False
                                for r in new_accepted:
                                    d_rc = graph._dist_sq(r, c)
                                    if d_rc < d_ec:
                                        reject = True
                                        break
                                if not reject:
                                    new_accepted.append(c)
                                    if len(new_accepted) == degree_limit:
                                        break

                            graph.set_neighbors(e, lvl, new_accepted)

        graph._update_entry_point()
        return graph

    @classmethod
    def from_adjacency(
        cls,
        vectors: NDArray[np.floating[Any]] | dict[int, NDArray[np.floating[Any]]],
        adjacency: dict[tuple[int, int], list[int]],
        levels: dict[int, int] | Sequence[int] | None = None,
        M: int = 16,
    ) -> ReferenceGraph:
        """Construct reference graph from explicit vectors and adjacency dictionary.

        Parameters
        ----------
        vectors : NDArray or dict[int, NDArray]
            Spatial coordinates of nodes.
        adjacency : dict[(node_id, level), list[neighbor_ids]]
            Explicit directed neighbor lists per (node, level).
        levels : dict, sequence, or None, default=None
            Explicit level per node. If None, inferred from maximum level in adjacency.
        M : int, default=16
            Maximum out-degree parameter.

        Returns
        -------
        ReferenceGraph
            Constructed reference graph test double.

        Raises
        ------
        ValueError
            If endpoints or degree limits are invalid.
        """
        # Parse vectors
        vec_dict: dict[int, NDArray[np.float32]] = {}
        if isinstance(vectors, dict):
            for nid, vec in vectors.items():
                vec_dict[int(nid)] = np.asarray(vec, dtype=np.float32)
        else:
            arr = np.asarray(vectors, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            for i in range(len(arr)):
                vec_dict[i] = arr[i].copy()

        dim = next(iter(vec_dict.values())).shape[0] if vec_dict else 0
        graph = cls(dim=dim, M=M)

        if not vec_dict:
            return graph

        # Determine levels
        node_levels: dict[int, int] = {}
        if levels is not None:
            if isinstance(levels, dict):
                for nid in vec_dict:
                    node_levels[nid] = int(levels.get(nid, 0))
            else:
                for idx, lvl in enumerate(levels):
                    if idx in vec_dict:
                        node_levels[idx] = int(lvl)
        else:
            for nid in vec_dict:
                node_levels[nid] = 0
            for (u, lvl), _ in adjacency.items():
                node_levels[u] = max(node_levels.get(u, 0), int(lvl))

        # Initialize graph structures
        for nid, vec in vec_dict.items():
            graph._vectors[nid] = vec
            graph._levels[nid] = node_levels[nid]
            graph._versions[nid] = 0

        max_lvl = max(node_levels.values()) if node_levels else 0
        for lvl in range(max_lvl + 1):
            graph._adj[lvl] = {nid: [] for nid in vec_dict if node_levels[nid] >= lvl}
            graph._rev[lvl] = {nid: [] for nid in vec_dict if node_levels[nid] >= lvl}

        # Populate adjacency with strict validation
        for (u, lvl), nbrs in adjacency.items():
            u_int = int(u)
            lvl_int = int(lvl)
            if u_int not in vec_dict:
                msg = f"Source node {u_int} not in vectors"
                raise ValueError(msg)
            if lvl_int < 0 or lvl_int > node_levels[u_int]:
                msg = f"Source node {u_int} level {node_levels[u_int]} < edge level {lvl_int}"
                raise ValueError(msg)

            limit = graph.max_degree(lvl_int)
            if len(nbrs) > limit:
                msg = (
                    f"Node {u_int} at level {lvl_int} has {len(nbrs)} neighbors; "
                    f"exceeds limit {limit}"
                )
                raise ValueError(msg)

            for v in nbrs:
                v_int = int(v)
                if v_int == u_int:
                    msg = f"Self-loop {u_int} -> {v_int} at level {lvl_int} is forbidden"
                    raise ValueError(msg)
                if v_int not in vec_dict:
                    msg = f"Target neighbor {v_int} not in vectors"
                    raise ValueError(msg)
                if lvl_int > node_levels[v_int]:
                    msg = f"Target node {v_int} level {node_levels[v_int]} < edge level {lvl_int}"
                    raise ValueError(msg)

            graph._adj[lvl_int][u_int] = [int(v) for v in nbrs]
            for v in nbrs:
                graph._rev[lvl_int][int(v)].append(u_int)

        graph._update_entry_point()
        return graph

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _dist_sq(self, u: int, v: int) -> float:
        """Compute squared L2 distance between node u and node v and increment counter."""
        self._dist_comps += 1
        diff = self._vectors[u] - self._vectors[v]
        return float(np.dot(diff, diff))

    def _dist_sq_vec(self, q: NDArray[np.float32], u: int) -> float:
        """Compute squared L2 distance between query vector and node u and increment counter."""
        self._dist_comps += 1
        diff = q - self._vectors[u]
        return float(np.dot(diff, diff))

    def _update_entry_point(self) -> None:
        """Update top-layer search entry point.

        Rule: None if empty; otherwise the lowest-id node at max_level.
        """
        if not self._vectors:
            self._entry_point = None
            return
        top_level = max(self._levels.values())
        top_nodes = [nid for nid, lvl in self._levels.items() if lvl == top_level]
        self._entry_point = min(top_nodes)

    # =========================================================================
    # ANNIndex Protocol Implementation
    # =========================================================================

    @property
    def dim(self) -> int:
        """Vector dimensionality of the index."""
        return self._dim

    def add(self, ids: NDArray[np.int64], vectors: NDArray[np.float32]) -> None:
        """Insert a batch of vectors with given node identifiers into the index."""
        id_arr = np.asarray(ids, dtype=np.int64)
        vec_arr = np.asarray(vectors, dtype=np.float32)

        if len(id_arr) != len(vec_arr):
            msg = f"Length mismatch: {len(id_arr)} ids != {len(vec_arr)} vectors"
            raise ValueError(msg)

        for nid_raw, vec in zip(id_arr, vec_arr, strict=True):
            node_id = int(nid_raw)
            if node_id < 0:
                msg = f"Node ID must be non-negative, got {node_id}"
                raise ValueError(msg)
            if node_id in self._vectors:
                msg = f"Node ID {node_id} already exists in graph"
                raise ValueError(msg)
            if vec.shape != (self._dim,):
                msg = f"Vector shape {vec.shape} does not match index dim ({self._dim},)"
                raise ValueError(msg)

            self._vectors[node_id] = vec.copy()
            self._levels[node_id] = 0
            self._versions[node_id] = 0
            self._adj.setdefault(0, {})[node_id] = []
            self._rev.setdefault(0, {})[node_id] = []

        self._update_entry_point()

    def move(self, node_id: int, new_vector: NDArray[np.float32]) -> None:
        """Update the spatial coordinates of an existing node in the index.

        Replaces the vector and increments version(node_id) by EXACTLY 1.
        Does NOT change any edges.
        """
        nid = int(node_id)
        if nid not in self._vectors:
            msg = f"Node {nid} not in index"
            raise KeyError(msg)

        vec = np.asarray(new_vector, dtype=np.float32)
        if vec.shape != (self._dim,):
            msg = f"Vector shape {vec.shape} does not match graph dim ({self._dim},)"
            raise ValueError(msg)

        self._vectors[nid] = vec.copy()
        self._versions[nid] += 1

    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        ef: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Exact brute-force top-k search by squared L2 distance.

        Stand-in implementation because this test double focuses on topology.
        """
        if self.size() == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self._dim,):
            msg = f"Query shape {q.shape} does not match graph dim ({self._dim},)"
            raise ValueError(msg)

        scored: list[tuple[float, int]] = []
        for nid in self._vectors:
            d2 = self._dist_sq_vec(q, nid)
            scored.append((d2, nid))

        # Sort by squared L2 distance ascending, breaking ties deterministically by node ID
        scored.sort(key=lambda item: (item[0], item[1]))
        top = scored[: min(int(k), len(scored))]

        top_ids = np.array([nid for _, nid in top], dtype=np.int64)
        top_dists = np.array([d2 for d2, _ in top], dtype=np.float32)
        return top_ids, top_dists

    def size(self) -> int:
        """Return the total number of indexed points."""
        return len(self._vectors)

    def contains(self, node_id: int) -> bool:
        """Check whether a node_id is currently present in the index."""
        return int(node_id) in self._vectors

    def remove(self, node_id: int) -> None:
        """Remove a node and ALL its forward and reverse edges across all levels."""
        nid = int(node_id)
        if nid not in self._vectors:
            msg = f"Node {nid} not in index"
            raise KeyError(msg)

        del self._vectors[nid]
        del self._versions[nid]
        node_lvl = self._levels.pop(nid)

        for lvl in range(node_lvl + 1):
            # Clean up outgoing connections
            out_nbrs = self._adj.get(lvl, {}).pop(nid, [])
            for v in out_nbrs:
                if v in self._rev.get(lvl, {}) and nid in self._rev[lvl][v]:
                    self._rev[lvl][v].remove(nid)

            # Clean up incoming connections
            in_nbrs = self._rev.get(lvl, {}).pop(nid, [])
            for u in in_nbrs:
                if u in self._adj.get(lvl, {}) and nid in self._adj[lvl][u]:
                    self._adj[lvl][u].remove(nid)

        self._update_entry_point()

    # =========================================================================
    # GraphIndex Protocol Implementation
    # =========================================================================

    def neighbors(self, node_id: int, level: int) -> list[int]:
        """Return outgoing neighbor identifiers for a node at a specific layer."""
        return list(self._adj.get(int(level), {}).get(int(node_id), []))

    def in_neighbors(self, node_id: int, level: int) -> list[int]:
        """Return incoming neighbor identifiers for a node at a specific layer."""
        return list(self._rev.get(int(level), {}).get(int(node_id), []))

    def node_level(self, node_id: int) -> int:
        """Return the highest layer level assigned to the given node."""
        nid = int(node_id)
        if nid not in self._levels:
            msg = f"Node {nid} not in graph"
            raise KeyError(msg)
        return self._levels[nid]

    def max_level(self) -> int:
        """Return the current maximum layer level present in the entire graph (-1 if empty)."""
        if not self._levels:
            return -1
        return max(self._levels.values())

    def entry_point(self) -> int | None:
        """Return the node identifier of the search entry point (lowest-id at max level)."""
        return self._entry_point

    def vector(self, node_id: int) -> NDArray[np.float32]:
        """Retrieve current spatial vector coordinates for a given node."""
        nid = int(node_id)
        if nid not in self._vectors:
            msg = f"Node {nid} not in graph"
            raise KeyError(msg)
        return self._vectors[nid].copy()

    def version(self, node_id: int) -> int:
        """Return the monotonic update version of a node."""
        nid = int(node_id)
        if nid not in self._versions:
            msg = f"Node {nid} not in graph"
            raise KeyError(msg)
        return self._versions[nid]

    def ids(self) -> list[int]:
        """Return all active node identifiers in the graph index."""
        return sorted(self._vectors.keys())

    def max_degree(self, level: int) -> int:
        """Return the maximum allowable degree at a given layer level."""
        return 2 * self._M if int(level) == 0 else self._M

    @property
    def distance_computations(self) -> int:
        """Cumulative counter of distance computations performed by the index."""
        return self._dist_comps

    def snapshot(self) -> dict[str, Any]:
        """Export a serializable snapshot of graph topology and metadata.

        Keys:
        -----
        size : int
            Total number of nodes.
        dim : int
            Vector dimensionality.
        M : int
            Base degree parameter.
        max_level : int
            Maximum hierarchy layer level (-1 if empty).
        entry_point : int | None
            Search entry point node ID.
        distance_computations : int
            Total distance computations executed.
        nodes : list[int]
            List of active node IDs.
        levels : dict[int, int]
            Mapping of node ID to maximum assigned level.
        degrees : dict[int, dict[int, int]]
            Mapping of level to {node_id: degree}.
        """
        top_lvl = self.max_level()
        degrees_map: dict[int, dict[int, int]] = {}
        if top_lvl >= 0:
            for lvl in range(top_lvl + 1):
                degrees_map[lvl] = {
                    nid: len(self.neighbors(nid, lvl))
                    for nid in self._vectors
                    if self._levels[nid] >= lvl
                }

        return {
            "size": self.size(),
            "dim": self._dim,
            "M": self._M,
            "max_level": top_lvl,
            "entry_point": self.entry_point(),
            "distance_computations": self.distance_computations,
            "nodes": self.ids(),
            "levels": {nid: self._levels[nid] for nid in self.ids()},
            "degrees": degrees_map,
        }

    # =========================================================================
    # MutableGraphIndex Protocol Implementation
    # =========================================================================

    def set_neighbors(self, node_id: int, level: int, neighbors: list[int]) -> None:
        """Replace the outgoing neighbor list of a node at a given level.

        Validates degree limits, self-loops, and endpoint existence at level.
        Keeps reverse index in_neighbors symmetrically consistent.
        """
        nid = int(node_id)
        lvl = int(level)
        if nid not in self._vectors:
            msg = f"Node {nid} does not exist in graph"
            raise ValueError(msg)
        if lvl < 0 or lvl > self._levels[nid]:
            msg = f"Level {lvl} invalid for node {nid} (assigned level {self._levels[nid]})"
            raise ValueError(msg)

        limit = self.max_degree(lvl)
        if len(neighbors) > limit:
            msg = f"Neighbor count {len(neighbors)} exceeds max_degree({lvl})={limit}"
            raise ValueError(msg)

        int_nbrs = [int(v) for v in neighbors]
        if nid in int_nbrs:
            msg = f"Self-loop {nid} -> {nid} is forbidden"
            raise ValueError(msg)

        for v in int_nbrs:
            if v not in self._vectors:
                msg = f"Neighbor {v} does not exist in graph"
                raise ValueError(msg)
            if lvl > self._levels[v]:
                msg = f"Neighbor {v} (level {self._levels[v]}) does not exist at layer {lvl}"
                raise ValueError(msg)

        level_adj = self._adj.setdefault(lvl, {})
        level_rev = self._rev.setdefault(lvl, {})

        old_nbrs = level_adj.get(nid, [])
        # Remove reverse links for discarded neighbors
        for v in old_nbrs:
            if v not in int_nbrs and v in level_rev and nid in level_rev[v]:
                level_rev[v].remove(nid)

        # Add reverse links for newly introduced neighbors
        for v in int_nbrs:
            rev_list = level_rev.setdefault(v, [])
            if nid not in rev_list:
                rev_list.append(nid)

        level_adj[nid] = list(int_nbrs)

    def add_edge(self, u: int, v: int, level: int) -> None:
        """Add a directed edge from u to v at the specified level.

        Validates endpoints existence at level, degree limits, and forbids self-loops.
        Updates reverse index in_neighbors symmetrically.
        """
        u_int = int(u)
        v_int = int(v)
        lvl = int(level)

        if u_int == v_int:
            msg = f"Self-loop {u_int} -> {v_int} is forbidden"
            raise ValueError(msg)
        if u_int not in self._vectors or v_int not in self._vectors:
            msg = f"Endpoints {u_int} and {v_int} must exist in graph"
            raise ValueError(msg)
        if lvl < 0 or lvl > self._levels[u_int] or lvl > self._levels[v_int]:
            msg = f"Endpoints {u_int} and {v_int} must exist at layer {lvl}"
            raise ValueError(msg)

        level_adj = self._adj.setdefault(lvl, {})
        level_rev = self._rev.setdefault(lvl, {})

        u_nbrs = level_adj.setdefault(u_int, [])
        if v_int not in u_nbrs:
            limit = self.max_degree(lvl)
            if len(u_nbrs) >= limit:
                msg = f"Cannot add edge {u_int} -> {v_int}: degree limit {limit} reached"
                raise ValueError(msg)
            u_nbrs.append(v_int)
            level_rev.setdefault(v_int, []).append(u_int)
            self._dist_comps += 1

    def remove_edge(self, u: int, v: int, level: int) -> None:
        """Remove the directed edge from u to v at the specified level."""
        u_int = int(u)
        v_int = int(v)
        lvl = int(level)

        level_adj = self._adj.get(lvl, {})
        level_rev = self._rev.get(lvl, {})

        if u_int in level_adj and v_int in level_adj[u_int]:
            level_adj[u_int].remove(v_int)
        if v_int in level_rev and u_int in level_rev[v_int]:
            level_rev[v_int].remove(u_int)
