"""Low-level graph storage and coupled forward/reverse adjacency container.

Pure Python standard library and NumPy only (browser/Pyodide safe).
Does not implement search, heuristics, or dynamic level assignment.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


class GraphStore:
    """Foundational graph container maintaining multi-layer forward and reverse edge sets.

    Guarantees:
        1. Layer-hierarchy invariant: a node at level L is present at all layers 0 <= l <= L.
        2. Bidirectional edge invariant: u in in_neighbors(v, l) <=> v in neighbors(u, l).
        3. Degree bounds: out-degree <= M0 for layer 0, and <= M for layers l > 0.
    """

    def __init__(self, dim: int, m: int = 16, m0: int | None = None) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if m <= 0:
            raise ValueError(f"m must be positive, got {m}")

        self._dim = int(dim)
        self._m = int(m)
        self._m0 = int(m0) if m0 is not None else int(2 * m)
        if self._m0 <= 0:
            raise ValueError(f"m0 must be positive, got {self._m0}")

        # Node storage: node_id -> float32 1D array of shape (dim,)
        self._vectors: dict[int, NDArray[np.float32]] = {}
        # Node maximum level: node_id -> level (int >= 0)
        self._levels: dict[int, int] = {}
        # Entry point: node_id at max_level, or None if empty
        self._entry_point: int | None = None

        # Adjacency structures: node_id -> dict of level -> set of neighbor node_ids
        # _forward[u][level] contains outgoing neighbors v (edge u -> v)
        # _reverse[v][level] contains incoming neighbors u (edge u -> v)
        self._forward: dict[int, dict[int, set[int]]] = {}
        self._reverse: dict[int, dict[int, set[int]]] = {}

    @property
    def dim(self) -> int:
        """Vector dimensionality of the index."""
        return self._dim

    @property
    def m(self) -> int:
        """Maximum out-degree for upper layers (l > 0)."""
        return self._m

    @property
    def m0(self) -> int:
        """Maximum out-degree for base layer (l = 0)."""
        return self._m0

    def __len__(self) -> int:
        """Total number of nodes currently stored in the graph."""
        return len(self._levels)

    def __contains__(self, node_id: int) -> bool:
        """Check whether node_id exists in the graph store."""
        return node_id in self._levels

    def max_m_for_level(self, level: int) -> int:
        """Return maximum permitted out-degree for a specific level."""
        return self._m0 if level == 0 else self._m

    def max_level(self) -> int:
        """Return current top layer index in the graph, or -1 if empty."""
        if not self._levels:
            return -1
        if self._entry_point is not None:
            return self._levels[self._entry_point]
        return max(self._levels.values())

    def entry_point(self) -> int | None:
        """Return identifier of the top-layer entry point node, or None if empty."""
        return self._entry_point

    def set_entry_point(self, node_id: int | None) -> None:
        """Explicitly set or reset the entry point node."""
        if node_id is None:
            if len(self._levels) > 0:
                raise ValueError("Cannot set entry_point to None while graph contains nodes")
            self._entry_point = None
            return

        if node_id not in self._levels:
            raise KeyError(f"node_id {node_id} does not exist in graph")

        curr_max = max(self._levels.values())
        if self._levels[node_id] != curr_max:
            raise ValueError(
                f"Entry point {node_id} level {self._levels[node_id]} != max_level {curr_max}"
            )
        self._entry_point = node_id

    def node_level(self, node_id: int) -> int:
        """Return the highest layer assigned to the given node."""
        if node_id not in self._levels:
            raise KeyError(f"node_id {node_id} does not exist in graph")
        return self._levels[node_id]

    def ids(self) -> list[int]:
        """Return all indexed node identifiers in sorted order."""
        return sorted(self._levels.keys())

    def vector(self, node_id: int) -> NDArray[np.float32]:
        """Retrieve vector coordinates for a node as float32 array of shape (dim,)."""
        if node_id not in self._vectors:
            raise KeyError(f"node_id {node_id} does not exist in graph")
        return self._vectors[node_id]

    def set_vector(self, node_id: int, vector: NDArray[np.float32]) -> None:
        """Update spatial vector coordinates for an existing node with a read-only copy."""
        if node_id not in self._levels:
            raise KeyError(f"node_id {node_id} does not exist in graph")

        vec = np.array(vector, dtype=np.float32, copy=True)
        if vec.shape != (self._dim,):
            raise ValueError(f"vector shape must be ({self._dim},), got {vec.shape}")
        vec.flags.writeable = False

        self._vectors[node_id] = vec

    def add_node(self, node_id: int, vector: NDArray[np.float32], level: int) -> None:
        """Add a single node with given vector coordinates and maximum level.

        Preconditions:
            - node_id is non-negative and not already present.
            - vector has shape (dim,) and is float32-compatible.
            - level is non-negative integer.

        Postconditions:
            - Node is initialized at all layers 0 <= l <= level (layer-hierarchy invariant).
            - Entry point is initialized or updated if level exceeds current max_level.
        """
        if not isinstance(node_id, (int, np.integer)):
            raise TypeError(f"node_id must be an integer, got {type(node_id).__name__}")
        node_id = int(node_id)
        if node_id < 0:
            raise ValueError(f"node_id must be non-negative, got {node_id}")
        if node_id in self._levels:
            raise ValueError(f"node_id {node_id} already exists in graph")

        if not isinstance(level, (int, np.integer)):
            raise TypeError(f"level must be an integer, got {type(level).__name__}")
        level = int(level)
        if level < 0:
            raise ValueError(f"level must be non-negative, got {level}")

        vec = np.array(vector, dtype=np.float32, copy=True)
        if vec.shape != (self._dim,):
            raise ValueError(f"vector shape must be ({self._dim},), got {vec.shape}")
        vec.flags.writeable = False

        # Store vector and level
        self._vectors[node_id] = vec
        self._levels[node_id] = level

        # Initialize empty forward and reverse adjacency sets for all layers 0 <= l <= level
        self._forward[node_id] = {layer: set() for layer in range(level + 1)}
        self._reverse[node_id] = {layer: set() for layer in range(level + 1)}

        # Update entry point if graph was empty or new node achieves strictly higher level
        if self._entry_point is None or level > self._levels[self._entry_point]:
            self._entry_point = node_id

    def neighbors(self, node_id: int, level: int) -> list[int]:
        """Return sorted list of outgoing neighbor identifiers at given level."""
        if node_id not in self._levels:
            raise KeyError(f"node_id {node_id} does not exist in graph")
        node_lvl = self._levels[node_id]
        if not (0 <= level <= node_lvl):
            raise ValueError(
                f"level {level} out of bounds for node {node_id} (0 <= level <= {node_lvl})"
            )
        return sorted(self._forward[node_id][level])

    def in_neighbors(self, node_id: int, level: int) -> list[int]:
        """Return sorted list of incoming neighbor identifiers (reverse edges) at given level."""
        if node_id not in self._levels:
            raise KeyError(f"node_id {node_id} does not exist in graph")
        node_lvl = self._levels[node_id]
        if not (0 <= level <= node_lvl):
            raise ValueError(
                f"level {level} out of bounds for node {node_id} (0 <= level <= {node_lvl})"
            )
        return sorted(self._reverse[node_id][level])

    def add_edge(self, u: int, v: int, level: int) -> None:
        """Add a directed edge u -> v at a level, updating forward and reverse together.

        Enforces degree bounds (M0 for level 0, M for level > 0).
        """
        u, v = int(u), int(v)
        if u == v:
            raise ValueError(f"Self-loops are not permitted: {u} -> {v}")
        if u not in self._levels:
            raise KeyError(f"Source node {u} does not exist in graph")
        if v not in self._levels:
            raise KeyError(f"Target node {v} does not exist in graph")

        if not (0 <= level <= self._levels[u]):
            raise ValueError(f"Level {level} exceeds source node {u} max level {self._levels[u]}")
        if not (0 <= level <= self._levels[v]):
            raise ValueError(f"Level {level} exceeds target node {v} max level {self._levels[v]}")

        fwd_u = self._forward[u][level]
        if v in fwd_u:
            return  # Edge already exists; idempotent

        max_deg = self.max_m_for_level(level)
        if len(fwd_u) >= max_deg:
            raise ValueError(
                f"Cannot add edge: node {u} at level {level} reached max out-degree {max_deg}"
            )

        fwd_u.add(v)
        self._reverse[v][level].add(u)

    def remove_edge(self, u: int, v: int, level: int) -> None:
        """Remove edge u -> v at a level, updating forward and reverse together."""
        u, v = int(u), int(v)
        if u not in self._levels:
            raise KeyError(f"Source node {u} does not exist in graph")
        if v not in self._levels:
            raise KeyError(f"Target node {v} does not exist in graph")

        if not (0 <= level <= self._levels[u]):
            raise ValueError(f"Level {level} exceeds source node {u} max level {self._levels[u]}")
        if not (0 <= level <= self._levels[v]):
            raise ValueError(f"Level {level} exceeds target node {v} max level {self._levels[v]}")

        fwd_u = self._forward[u][level]
        if v not in fwd_u:
            raise KeyError(f"Edge {u} -> {v} does not exist at level {level}")

        fwd_u.remove(v)
        self._reverse[v][level].remove(u)

    def remove_node(self, node_id: int) -> None:
        """Remove a node and all its incident edges across all levels.

        Symmetrically severs connections from predecessors (forward) and successors
        (reverse). If node_id was the entry point, reassigns it to an existing node
        at the new max level, or None if the graph becomes empty.
        """
        node_id = int(node_id)
        if node_id not in self._levels:
            raise KeyError(f"node_id {node_id} does not exist in graph")

        node_lvl = self._levels[node_id]

        # Symmetrically remove from all incident forward and reverse edge sets
        for lvl in range(node_lvl + 1):
            # Predecessors u -> node_id: remove node_id from forward of u
            for u in list(self._reverse[node_id][lvl]):
                self._forward[u][lvl].discard(node_id)

            # Successors node_id -> v: remove node_id from reverse of v
            for v in list(self._forward[node_id][lvl]):
                self._reverse[v][lvl].discard(node_id)

        # Delete node records
        del self._vectors[node_id]
        del self._levels[node_id]
        del self._forward[node_id]
        del self._reverse[node_id]

        # Update entry point if deleted node was the entry point
        if self._entry_point == node_id:
            if not self._levels:
                self._entry_point = None
            else:
                new_max = max(self._levels.values())
                for candidate, lvl in self._levels.items():
                    if lvl == new_max:
                        self._entry_point = candidate
                        break

    def set_neighbors(self, node_id: int, level: int, neighbors: Sequence[int] | set[int]) -> None:
        """Replace all outgoing neighbors of node_id at given level.

        Synchronously updates both forward and reverse adjacency.
        Enforces degree bounds and absence of self-loops.
        """
        node_id = int(node_id)
        if node_id not in self._levels:
            raise KeyError(f"node_id {node_id} does not exist in graph")

        node_lvl = self._levels[node_id]
        if not (0 <= level <= node_lvl):
            raise ValueError(
                f"level {level} out of bounds for node {node_id} (0 <= level <= {node_lvl})"
            )

        new_nbr_set = {int(n) for n in neighbors}
        if node_id in new_nbr_set:
            raise ValueError(f"Cannot set neighbor list containing self-loop: {node_id}")

        max_deg = self.max_m_for_level(level)
        if len(new_nbr_set) > max_deg:
            raise ValueError(
                f"Neighbor count {len(new_nbr_set)} exceeds max out-degree {max_deg} at {level}"
            )

        # Validate that all targets exist and reach this level
        for target in new_nbr_set:
            if target not in self._levels:
                raise KeyError(f"Target neighbor {target} does not exist in graph")
            if level > self._levels[target]:
                raise ValueError(
                    f"Target neighbor {target} max level {self._levels[target]} < level {level}"
                )

        old_nbr_set = self._forward[node_id][level]

        # Edges to remove: old - new
        removed = old_nbr_set - new_nbr_set
        for r in removed:
            self._reverse[r][level].remove(node_id)

        # Edges to add: new - old
        added = new_nbr_set - old_nbr_set
        for a in added:
            self._reverse[a][level].add(node_id)

        # Replace forward set
        self._forward[node_id][level] = new_nbr_set

    def snapshot(self) -> dict[str, Any]:
        """Export a serializable snapshot of graph topology and metadata."""
        max_lvl = self.max_level()
        layer_stats: list[dict[str, Any]] = []

        if max_lvl >= 0:
            for layer in range(max_lvl + 1):
                nodes_at_l = [nid for nid, lvl in self._levels.items() if lvl >= layer]
                edge_count = sum(len(self._forward[nid][layer]) for nid in nodes_at_l)
                layer_stats.append(
                    {
                        "level": layer,
                        "num_nodes": len(nodes_at_l),
                        "num_edges": edge_count,
                        "max_degree": self.max_m_for_level(layer),
                    }
                )

        return {
            "dim": self._dim,
            "m": self._m,
            "m0": self._m0,
            "num_nodes": len(self._levels),
            "max_level": max_lvl,
            "entry_point": self._entry_point,
            "layers": layer_stats,
        }
