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

__all__ = [
    "ANNIndex",
    "EventSink",
    "GraphIndex",
    "LinearTrajectory",
    "MovementModel",
    "MutableGraphIndex",
    "Repairer",
]


@runtime_checkable
class ANNIndex(Protocol):
    """General Approximate Nearest Neighbor (ANN) index protocol."""

    @property
    def dim(self) -> int:
        """Vector dimensionality of the index.

        Returns
        -------
        int
            Dimension of vector embeddings stored in the index.
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
            - TODO (Owner: Member 2 - Arpanjot Kaur): len(ids) == len(vectors).
            - TODO (Owner: Member 2 - Arpanjot Kaur): All ids non-negative and not in index.
            - TODO (Owner: Member 2 - Arpanjot Kaur): vectors.shape[1] == self.dim.
        Postconditions:
            - TODO (Owner: Member 2 - Arpanjot Kaur): All points are queryable.
        Complexity:
            - Expected: O(N * log(total_nodes)) for tree/graph structures.
        """
        ...

    def move(self, node_id: int, new_vector: NDArray[np.float32]) -> None:
        """Update the spatial coordinates of an existing node in the index.

        Contract & Lifecycle:
            - Call order: read old vector -> index.move -> repairer.on_move.
            - Version increments: Calling move increments version(node_id) monotonically.

        Parameters
        ----------
        node_id : int
            Identifier of the moving node.
        new_vector : NDArray[np.float32]
            Updated 1D vector position, shape (dim,).

        Preconditions:
            - TODO (Owner: Member 2 - Arpanjot Kaur): node_id exists in the index.
            - TODO (Owner: Member 2 - Arpanjot Kaur): new_vector.shape == (self.dim,).
        Postconditions:
            - TODO (Owner: Member 2 - Arpanjot Kaur): vector(node_id) returns new_vector.
            - TODO (Owner: Member 2 - Arpanjot Kaur): version(node_id) incremented by 1.
        Complexity:
            - Expected: O(1) coordinate and version update.
        """
        ...

    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        ef: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Perform search for approximate nearest neighbors.

        Distance Metric Contract:
            - Search returns squared L2 distances (||q - v||^2).
            - For cosine similarity data, vectors must be normalized beforehand
              (cosine distance on unit vectors equals 0.5 * squared L2).

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
            Tuple of (top_k_ids, top_k_squared_l2_distances).

        Preconditions:
            - TODO (Owner: Member 2 - Arpanjot Kaur): ef >= k >= 1.
            - TODO (Owner: Member 2 - Arpanjot Kaur): query.shape == (self.dim,).
        Postconditions:
            - TODO (Owner: Member 2 - Arpanjot Kaur): Returns <= min(k, size()) sorted elements.
        Complexity:
            - Expected: O(ef * log(ef) * log(N)) distance computations.
        """
        ...

    def size(self) -> int:
        """Return the total number of indexed points.

        Returns
        -------
        int
            Total count of active vectors currently in the index.
        """
        ...

    def contains(self, node_id: int) -> bool:
        """Check whether a node_id is currently present in the index.

        Parameters
        ----------
        node_id : int
            Node identifier to query.

        Returns
        -------
        bool
            True if the node exists in the index, False otherwise.
        """
        ...

    def remove(self, node_id: int) -> None:
        """Remove a node from the index.

        Parameters
        ----------
        node_id : int
            Identifier of the node to remove.

        Preconditions:
            - TODO (Owner: Member 2 - Arpanjot Kaur): node_id exists in the index.
        Postconditions:
            - TODO (Owner: Member 2 - Arpanjot Kaur): contains(node_id) is False.
        """
        ...


@runtime_checkable
class GraphIndex(ANNIndex, Protocol):
    """Protocol defining the Hierarchical Navigable Small World graph index contract.

    Extends ANNIndex with multi-layer topology inspection, reverse-edge tracking,
    and distance computation counters.
    """

    def neighbors(self, node_id: int, level: int) -> list[int]:
        """Return outgoing neighbor identifiers for a node at a specific layer.

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
            - TODO (Owner: Member 2 - Arpanjot Kaur): node_id exists, level <= node_level(node_id).
        Postconditions:
            - TODO (Owner: Member 2 - Arpanjot Kaur): len(neighbors) <= max_degree(level).
        Complexity:
            - Expected: O(1) or O(degree).
        """
        ...

    def in_neighbors(self, node_id: int, level: int) -> list[int]:
        """Return incoming neighbor identifiers (reverse edges) for a node at a layer.

        Parameters
        ----------
        node_id : int
            Target node identifier.
        level : int
            Layer level index.

        Returns
        -------
        list[int]
            List of incoming neighbor IDs that have a directed edge to node_id.

        Preconditions:
            - TODO (Owner: Member 2 - Arpanjot Kaur): Reverse-edge tracking enabled.
        Postconditions:
            - TODO (Owner: Member 2 - Arpanjot Kaur): u in in_neighbors <=> v in neighbors.
        Complexity:
            - Expected: O(1) lookup in reverse adjacency table.
        """
        ...

    def node_level(self, node_id: int) -> int:
        """Return the highest layer level assigned to the given node.

        Parameters
        ----------
        node_id : int
            Target node identifier.

        Returns
        -------
        int
            Maximum level index where this node appears.

        Preconditions:
            - TODO (Owner: Member 2 - Arpanjot Kaur): node_id exists in the graph.
        Postconditions:
            - TODO (Owner: Member 2 - Arpanjot Kaur): 0 <= level <= max_level().
        Complexity:
            - Expected: O(1).
        """
        ...

    def max_level(self) -> int:
        """Return the current maximum layer level present in the entire graph.

        Returns
        -------
        int
            Top-level index of the hierarchy (>= 0 for non-empty graphs, -1 for empty graph).
        """
        ...

    def entry_point(self) -> int | None:
        """Return the node identifier of the top-layer search entry point.

        Returns
        -------
        int | None
            Node ID of the entry point, or None if the graph is empty.
        """
        ...

    def vector(self, node_id: int) -> NDArray[np.float32]:
        """Retrieve current spatial vector coordinates for a given node.

        Parameters
        ----------
        node_id : int
            Target node identifier.

        Returns
        -------
        NDArray[np.float32]
            Current vector representation, shape (dim,).

        Preconditions:
            - TODO (Owner: Member 2 - Arpanjot Kaur): node_id exists.
        Complexity:
            - Expected: O(1).
        """
        ...

    def version(self, node_id: int) -> int:
        """Return the monotonic update version of a node.

        Increments on each call to move(node_id, ...).

        Parameters
        ----------
        node_id : int
            Target node identifier.

        Returns
        -------
        int
            Monotonic version counter (starts at 0 on add, increments on move).
        """
        ...

    def ids(self) -> list[int]:
        """Return all active node identifiers in the graph index.

        Returns
        -------
        list[int]
            List of all node IDs present in the graph.
        """
        ...

    def max_degree(self, level: int) -> int:
        """Return the maximum allowable degree at a given layer level.

        Parameters
        ----------
        level : int
            Layer index (level 0 allows M0; level > 0 allows M).

        Returns
        -------
        int
            Maximum allowable outgoing edge count.
        """
        ...

    @property
    def distance_computations(self) -> int:
        """Cumulative counter of distance computations performed by the index.

        Returns
        -------
        int
            Total number of pairwise distance calculations executed.
        """
        ...

    def snapshot(self) -> dict[str, Any]:
        """Export a serializable snapshot of graph topology and metadata.

        Returns
        -------
        dict[str, Any]
            Dictionary containing node counts, layer statistics, entry point, and edge distribution.
        """
        ...


@runtime_checkable
class MutableGraphIndex(GraphIndex, Protocol):
    """Protocol defining graph mutation operations for repair and construction.

    Each mutation method must keep the reverse index consistent and count its
    distance computations.
    """

    def set_neighbors(self, node_id: int, level: int, neighbors: list[int]) -> None:
        """Replace the outgoing neighbor list of a node at a given level.

        Contract:
            - Must keep the reverse index (`in_neighbors`) consistent for both added
              and removed connections.
            - Must count any distance computations performed during pruning/selection.

        Parameters
        ----------
        node_id : int
            Source node identifier.
        level : int
            Layer level index.
        neighbors : list[int]
            New list of outgoing neighbor IDs.

        Preconditions:
            - TODO (Owner: Member 2 - Arpanjot Kaur): len(neighbors) <= max_degree(level).
        Postconditions:
            - TODO (Owner: Member 2 - Arpanjot Kaur): Reverse index updated symmetrically.
        """
        ...

    def add_edge(self, u: int, v: int, level: int) -> None:
        """Add a directed edge from u to v at the specified level.

        Contract:
            - Must add u to in_neighbors(v, level) consistently.
            - Must count any distance computations performed.

        Parameters
        ----------
        u : int
            Source node identifier.
        v : int
            Target node identifier.
        level : int
            Layer level index.

        Preconditions:
            - TODO (Owner: Member 2 - Arpanjot Kaur): u and v exist at this level.
        Postconditions:
            - TODO (Owner: Member 2 - Arpanjot Kaur): v in neighbors(u, level).
            - TODO (Owner: Member 2 - Arpanjot Kaur): u in in_neighbors(v, level).
        """
        ...

    def remove_edge(self, u: int, v: int, level: int) -> None:
        """Remove the directed edge from u to v at the specified level.

        Contract:
            - Must remove u from in_neighbors(v, level) consistently.

        Parameters
        ----------
        u : int
            Source node identifier.
        v : int
            Target node identifier.
        level : int
            Layer level index.

        Postconditions:
            - TODO (Owner: Member 2 - Arpanjot Kaur): v not in neighbors(u, level).
            - TODO (Owner: Member 2 - Arpanjot Kaur): u not in in_neighbors(v, level).
        """
        ...


@runtime_checkable
class LinearTrajectory(Protocol):
    """Optional protocol for linear kinematic trajectories where p(t) = a + b * t."""

    def trajectory(self, node_id: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the linear trajectory coefficients (a, b) for a given node.

        Position at simulation time t is parameterized as p(t) = a + b * t.
        Note: returns float64 arrays because certificate evaluation and quadratic root
        finding require double-precision arithmetic to avoid numerical cancellation.

        Parameters
        ----------
        node_id : int
            Identifier of the target node.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            Tuple of (initial_offset_a, velocity_vector_b), each of shape (dim,).
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
            - TODO (Owner: Member 3 - Khushi Dhona): t >= 0.0 and monotonically non-decreasing.
        Postconditions:
            - TODO (Owner: Member 3 - Khushi Dhona): Output shape is strictly (n, dim).
        Complexity:
            - Expected: O(n * dim) per simulation step.
        """
        ...

    def changed_nodes(self, t_prev: float, t_curr: float) -> list[int]:
        """Report node identifiers whose positions changed between t_prev and t_curr.

        Parameters
        ----------
        t_prev : float
            Previous simulation timestamp.
        t_curr : float
            Current simulation timestamp.

        Returns
        -------
        list[int]
            List of node IDs that moved during this time step.
        """
        ...


@runtime_checkable
class Repairer(Protocol):
    """Protocol for topology repair policies under kinetic node updates."""

    def on_move(
        self,
        index: MutableGraphIndex,
        node_id: int,
        old_vector: NDArray[np.float32],
        new_vector: NDArray[np.float32],
        sink: EventSink,
    ) -> None:
        """Handle a single node movement notification and trigger kinetic certificates/queues.

        Execution Protocol:
            - Call order is strictly: read old vector -> index.move -> repairer.on_move.

        Parameters
        ----------
        index : MutableGraphIndex
            Target mutable graph index instance.
        node_id : int
            Identifier of the moved node.
        old_vector : NDArray[np.float32]
            Previous spatial coordinates.
        new_vector : NDArray[np.float32]
            Updated spatial coordinates.
        sink : EventSink
            Event sink to emit certificate invalidations and queue events.

        Preconditions:
            - TODO (Owner: Member 3 - Khushi Dhona): node_id exists in index.
        Postconditions:
            - TODO (Owner: Member 3 - Khushi Dhona): Invalidated edges enqueued in repair heap.
        Complexity:
            - Expected: O(deg(node_id) * dim) certificate checks.
        """
        ...

    def run_repairs(
        self,
        index: MutableGraphIndex,
        budget: float,
        sink: EventSink,
    ) -> dict[str, Any]:
        """Execute scheduled topological repairs within an allotted computational budget.

        Budget Metric Contract:
            - Budget unit = distance computations. Every pairwise distance calculation
              performed during neighbor search or rewiring consumes 1 unit of budget.

        Parameters
        ----------
        index : MutableGraphIndex
            Target mutable graph index instance.
        budget : float
            Computational repair budget in distance computations (budget > 0.0).
        sink : EventSink
            Event sink to log edge modifications and repair metrics.

        Returns
        -------
        dict[str, Any]
            Dictionary summarizing repair actions taken (edges added, removed, budget consumed).

        Preconditions:
            - TODO (Owner: Member 3 - Khushi Dhona): budget > 0.0.
        Postconditions:
            - TODO (Owner: Member 3 - Khushi Dhona): Graph degree bounds are maintained;
              reachability is measured and reported.
        Complexity:
            - Expected: Consumes <= budget distance computations.
        """
        ...
