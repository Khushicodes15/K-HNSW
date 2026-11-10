"""Core interface contracts for K-HNSW kinetic graph index, movement models, and repairers.

This module defines typing.Protocol specifications for core components.
Implementations must conform to these protocols and maintain pure dependencies
(stdlib + numpy + khnsw only).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from khnsw.core.events.schema import EventSink

# Re-export EventSink for convenience
__all__ = [
    "EventSink",
    "GraphIndex",
    "MovementModel",
    "Repairer",
]


@runtime_checkable
class GraphIndex(Protocol):
    """Protocol defining the Hierarchical Navigable Small World index contract.

    Supports vector insertion, dynamic node movement, multi-layer neighbor exploration,
    reverse-edge tracking, and approximate nearest neighbor search.
    """

    @property
    def dim(self) -> int:
        """Vector dimensionality of the index.

        Returns
        -------
        int
            Dimension of vector embeddings stored in the graph.
        """
        ...

    def add(self, ids: NDArray[np.int64], vectors: NDArray[np.float32]) -> None:
        """Insert a batch of vectors with given node identifiers into the index.

        Parameters
        ----------
        ids : NDArray[np.int64]
            1D array of unique integer node identifiers, shape (N,).
        vectors : NDArray[np.float32]
            2D array of vector coordinates, shape (N, dim).

        Preconditions:
            - TODO (Owner: Member 2): len(ids) == len(vectors).
            - TODO (Owner: Member 2): All ids non-negative and not already in index.
            - TODO (Owner: Member 2): vectors.shape[1] == self.dim.
        Postconditions:
            - TODO (Owner: Member 2): Added nodes assigned layers via decay distribution.
            - TODO (Owner: Member 2): Out-degree at each layer obeys maximum M / M0 bounds.
        Complexity:
            - Expected: O(N * log(total_nodes) * M) distance computations.
        """
        ...

    def move(self, node_id: int, new_vector: NDArray[np.float32]) -> None:
        """Update the spatial coordinates of an existing node in the index.

        Parameters
        ----------
        node_id : int
            Identifier of the moving node.
        new_vector : NDArray[np.float32]
            Updated 1D vector position, shape (dim,).

        Preconditions:
            - TODO (Owner: Member 2): node_id exists in the graph.
            - TODO (Owner: Member 2): new_vector.shape == (self.dim,).
        Postconditions:
            - TODO (Owner: Member 2): vector(node_id) returns new_vector.
            - Note: graph edges may become kinetically invalid or sub-optimal until repaired.
        Complexity:
            - Expected: O(1) coordinate update or O(deg(node_id)) certificate evaluation.
        """
        ...

    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        ef: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Perform greedy / beam search for approximate nearest neighbors.

        Parameters
        ----------
        query : NDArray[np.float32]
            1D query vector of shape (dim,).
        k : int
            Number of nearest neighbors to retrieve.
        ef : int
            Size of dynamic candidate exploration list (ef >= k).

        Returns
        -------
        tuple[NDArray[np.int64], NDArray[np.float32]]
            Tuple of (top_k_ids, top_k_distances).

        Preconditions:
            - TODO (Owner: Member 2): ef >= k >= 1.
            - TODO (Owner: Member 2): query.shape == (self.dim,).
        Postconditions:
            - TODO (Owner: Member 2): Returns <= min(k, total_nodes) elements sorted by distance.
        Complexity:
            - Expected: O(ef * log(ef) * log(N)) distance computations.
        """
        ...

    def neighbors(self, node_id: int, level: int) -> list[int]:
        """Return the list of outgoing neighbor identifiers for a node at a specific layer.

        Parameters
        ----------
        node_id : int
            Target node identifier.
        level : int
            Layer level index (0 <= level <= max_level).

        Returns
        -------
        list[int]
            List of outgoing neighbor IDs.

        Preconditions:
            - TODO (Owner: Member 2): node_id exists and level <= node_level(node_id).
        Postconditions:
            - TODO (Owner: Member 2): Output list length <= max_m_for_level.
        Complexity:
            - Expected: O(1) or O(degree).
        """
        ...

    def in_neighbors(self, node_id: int, level: int) -> list[int]:
        """Return the list of incoming neighbor identifiers (reverse edges) for a node at a layer.

        Parameters
        ----------
        node_id : int
            Target node identifier.
        level : int
            Layer level index.

        Returns
        -------
        list[int]
            List of incoming neighbor IDs that have an edge directed to node_id at this layer.

        Preconditions:
            - TODO (Owner: Member 2): Reverse-edge tracking enabled on graph.
        Postconditions:
            - TODO (Owner: Member 2): Consistent with forward adjacency lists.
        Complexity:
            - Expected: O(1) lookup in reverse adjacency index.
        """
        ...

    def node_level(self, node_id: int) -> int:
        """Return the highest layer assigned to the given node.

        Parameters
        ----------
        node_id : int
            Target node identifier.

        Returns
        -------
        int
            Maximum level index where this node appears.

        Preconditions:
            - TODO (Owner: Member 2): node_id exists in the graph.
        Postconditions:
            - TODO (Owner: Member 2): 0 <= level <= max_level().
        Complexity:
            - Expected: O(1).
        """
        ...

    def max_level(self) -> int:
        """Return the current maximum layer level present in the entire graph.

        Returns
        -------
        int
            Top-level index of the hierarchy (e.g. 0 for single layer, >=0 for multi-layer).

        Preconditions:
            - TODO (Owner: Member 2): Graph must have at least one node, else returns -1 or 0.
        Complexity:
            - Expected: O(1).
        """
        ...

    def entry_point(self) -> int | None:
        """Return the node identifier of the top-layer search entry point.

        Returns
        -------
        int | None
            Node ID of the entry point, or None if the graph is empty.

        Preconditions:
            - None.
        Postconditions:
            - TODO (Owner: Member 2): If not None, node exists at level == max_level().
        Complexity:
            - Expected: O(1).
        """
        ...

    def vector(self, node_id: int) -> NDArray[np.float32]:
        """Retrieve the current spatial vector coordinates for a given node.

        Parameters
        ----------
        node_id : int
            Target node identifier.

        Returns
        -------
        NDArray[np.float32]
            Current vector representation, shape (dim,).

        Preconditions:
            - TODO (Owner: Member 2): node_id exists.
        Complexity:
            - Expected: O(1).
        """
        ...

    def snapshot(self) -> dict[str, Any]:
        """Export a serializable snapshot of the graph topology and metadata.

        Returns
        -------
        dict[str, Any]
            Dictionary containing node counts, layer statistics, entry point, and edge distribution.

        Preconditions:
            - None.
        Complexity:
            - Expected: O(V + E) for complete structural export.
        """
        ...


@runtime_checkable
class MovementModel(Protocol):
    """Protocol for synthetic or dataset-driven kinetic vector trajectories."""

    @property
    def name(self) -> str:
        """Descriptive identifier of the kinetic movement model.

        Returns
        -------
        str
            Model name (e.g. 'random_walk', 'cluster_drift', 'brownian').
        """
        ...

    @property
    def seed(self) -> int:
        """Random seed for deterministic trajectory reproducibility.

        Returns
        -------
        int
            Random generator seed.
        """
        ...

    def positions(self, t: float) -> NDArray[np.float32]:
        """Compute the spatial positions of all indexed nodes at simulation timestamp t.

        Parameters
        ----------
        t : float
            Simulation time parameter (t >= 0.0).

        Returns
        -------
        NDArray[np.float32]
            2D array of positions for all n points, shape (n, dim).

        Preconditions:
            - TODO (Owner: Member 3): t >= 0.0 and monotonically non-decreasing in simulation loops.
        Postconditions:
            - TODO (Owner: Member 3): Output shape is strictly (n, dim) with finite float32 values.
        Complexity:
            - Expected: O(n * dim) per simulation step.
        """
        ...


@runtime_checkable
class Repairer(Protocol):
    """Protocol for topology repair policies under kinetic node updates."""

    def on_move(
        self,
        index: GraphIndex,
        node_id: int,
        old_vector: NDArray[np.float32],
        new_vector: NDArray[np.float32],
        sink: EventSink,
    ) -> None:
        """Handle a single node movement notification and trigger kinetic certificates/queues.

        Parameters
        ----------
        index : GraphIndex
            Target graph index instance.
        node_id : int
            Identifier of the moved node.
        old_vector : NDArray[np.float32]
            Previous spatial coordinates.
        new_vector : NDArray[np.float32]
            Updated spatial coordinates.
        sink : EventSink
            Event sink to emit certificate invalidations and queue events.

        Preconditions:
            - TODO (Owner: Member 3): node_id exists in index.
        Postconditions:
            - TODO (Owner: Member 3): Invalidated edges are enqueued in repair heap.
        Complexity:
            - Expected: O(deg(node_id) * dim) certificate checks.
        """
        ...

    def run_repairs(
        self,
        index: GraphIndex,
        budget: float,
        sink: EventSink,
    ) -> dict[str, Any]:
        """Execute scheduled topological repairs within an allotted computational budget.

        Parameters
        ----------
        index : GraphIndex
            Target graph index instance.
        budget : float
            Computational repair budget (e.g. maximum distance calculations or operations).
        sink : EventSink
            Event sink to log edge modifications and repair metrics.

        Returns
        -------
        dict[str, Any]
            Dictionary summarizing repair actions taken (edges added, removed, budget consumed).

        Preconditions:
            - TODO (Owner: Member 3): budget > 0.0.
        Postconditions:
            - TODO (Owner: Member 3): Graph invariants (degree bounds, reachability) preserved.
        Complexity:
            - Expected: Consumes <= budget operations.
        """
        ...
