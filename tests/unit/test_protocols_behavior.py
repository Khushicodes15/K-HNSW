"""Behavioral verification tests exercising fake implementations of core protocols."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from khnsw.core.events.schema import Event, EventSink, EventType
from khnsw.core.interfaces import (
    ANNIndex,
    GraphIndex,
    LinearTrajectory,
    MovementModel,
    MutableGraphIndex,
    Repairer,
)


class FakeMutableGraphIndex:
    """In-memory concrete implementation exercising MutableGraphIndex protocol."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self._vectors: dict[int, NDArray[np.float32]] = {}
        self._levels: dict[int, int] = {}
        self._versions: dict[int, int] = {}
        # level -> u -> list of v
        self._adj: dict[int, dict[int, list[int]]] = {0: {}}
        # level -> v -> list of u
        self._rev: dict[int, dict[int, list[int]]] = {0: {}}
        self._dist_comps: int = 0
        self._entry_point: int | None = None

    @property
    def dim(self) -> int:
        return self._dim

    def add(self, ids: NDArray[np.int64], vectors: NDArray[np.float32]) -> None:
        for nid, vec in zip(ids, vectors, strict=True):
            node_id = int(nid)
            self._vectors[node_id] = np.asarray(vec, dtype=np.float32)
            self._levels[node_id] = 0
            self._versions[node_id] = 0
            self._adj[0][node_id] = []
            self._rev[0][node_id] = []
            if self._entry_point is None:
                self._entry_point = node_id

    def move(self, node_id: int, new_vector: NDArray[np.float32]) -> None:
        if node_id not in self._vectors:
            raise KeyError(f"Node {node_id} not in index")
        self._vectors[node_id] = np.asarray(new_vector, dtype=np.float32)
        self._versions[node_id] += 1

    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        ef: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        q = np.asarray(query, dtype=np.float32)
        node_ids: list[int] = []
        dists: list[float] = []
        for nid, vec in self._vectors.items():
            diff = q - vec
            d2 = float(np.dot(diff, diff))  # squared L2
            self._dist_comps += 1
            node_ids.append(nid)
            dists.append(d2)

        order = np.argsort(dists)[:k]
        top_ids = np.array([node_ids[i] for i in order], dtype=np.int64)
        top_dists = np.array([dists[i] for i in order], dtype=np.float32)
        return top_ids, top_dists

    def size(self) -> int:
        return len(self._vectors)

    def contains(self, node_id: int) -> bool:
        return node_id in self._vectors

    def remove(self, node_id: int) -> None:
        if node_id in self._vectors:
            del self._vectors[node_id]
            del self._levels[node_id]
            del self._versions[node_id]
            if node_id in self._adj.get(0, {}):
                del self._adj[0][node_id]
            if node_id in self._rev.get(0, {}):
                del self._rev[0][node_id]
            if self._entry_point == node_id:
                self._entry_point = next(iter(self._vectors)) if self._vectors else None

    def neighbors(self, node_id: int, level: int) -> list[int]:
        return list(self._adj.get(level, {}).get(node_id, []))

    def in_neighbors(self, node_id: int, level: int) -> list[int]:
        return list(self._rev.get(level, {}).get(node_id, []))

    def node_level(self, node_id: int) -> int:
        return self._levels.get(node_id, 0)

    def max_level(self) -> int:
        if not self._vectors:
            return -1
        return max(self._levels.values(), default=0)

    def entry_point(self) -> int | None:
        return self._entry_point

    def vector(self, node_id: int) -> NDArray[np.float32]:
        return self._vectors[node_id]

    def version(self, node_id: int) -> int:
        return self._versions[node_id]

    def ids(self) -> list[int]:
        return list(self._vectors.keys())

    def max_degree(self, level: int) -> int:
        return 32 if level == 0 else 16

    @property
    def distance_computations(self) -> int:
        return self._dist_comps

    def snapshot(self) -> dict[str, Any]:
        return {
            "size": self.size(),
            "max_level": self.max_level(),
            "entry_point": self.entry_point(),
            "distance_computations": self.distance_computations,
        }

    def set_neighbors(self, node_id: int, level: int, neighbors: list[int]) -> None:
        level_adj = self._adj.setdefault(level, {})
        level_rev = self._rev.setdefault(level, {})

        # Remove old incoming links
        old_neighbors = level_adj.get(node_id, [])
        for v in old_neighbors:
            if v in level_rev and node_id in level_rev[v]:
                level_rev[v].remove(node_id)

        # Set new
        level_adj[node_id] = list(neighbors)
        for v in neighbors:
            level_rev.setdefault(v, [])
            if node_id not in level_rev[v]:
                level_rev[v].append(node_id)

    def add_edge(self, u: int, v: int, level: int) -> None:
        level_adj = self._adj.setdefault(level, {})
        level_rev = self._rev.setdefault(level, {})

        u_neighbors = level_adj.setdefault(u, [])
        if v not in u_neighbors:
            u_neighbors.append(v)

        v_in = level_rev.setdefault(v, [])
        if u not in v_in:
            v_in.append(u)

        # Counting distance computation for edge weight evaluation
        self._dist_comps += 1

    def remove_edge(self, u: int, v: int, level: int) -> None:
        level_adj = self._adj.setdefault(level, {})
        level_rev = self._rev.setdefault(level, {})

        if u in level_adj and v in level_adj[u]:
            level_adj[u].remove(v)
        if v in level_rev and u in level_rev[v]:
            level_rev[v].remove(u)


class FakeMovementModel:
    """In-memory concrete implementation exercising MovementModel protocol."""

    def __init__(self, name: str = "fake_drift", seed: int = 123) -> None:
        self._name = name
        self._seed = seed

    @property
    def name(self) -> str:
        return self._name

    @property
    def seed(self) -> int:
        return self._seed

    def positions(self, t: float) -> NDArray[np.float32]:
        # Simple deterministic coordinates shifting with t
        return np.array([[float(t), float(t) + 1.0]], dtype=np.float32)

    def changed_nodes(self, t_prev: float, t_curr: float) -> list[int]:
        if t_curr > t_prev:
            return [0]
        return []


class FakeLinearTrajectory:
    """In-memory concrete implementation exercising LinearTrajectory protocol."""

    def trajectory(self, node_id: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        a = np.array([float(node_id), 0.0], dtype=np.float64)
        b = np.array([0.1, 0.2], dtype=np.float64)
        return a, b


class FakeRepairer:
    """In-memory concrete implementation exercising Repairer protocol."""

    def on_move(
        self,
        index: MutableGraphIndex,
        node_id: int,
        old_vector: NDArray[np.float32],
        new_vector: NDArray[np.float32],
        sink: EventSink,
    ) -> None:
        # Emits a cert_fail event
        event = Event(
            run_id="test-run",
            seq=1,
            t=0.1,
            type=EventType.cert_fail,
            node=node_id,
            level=0,
            version=index.version(node_id),
            payload={"neighbor_id": 99, "reason": "relative_distance_violation"},
        )
        sink.emit(event)

    def run_repairs(
        self,
        index: MutableGraphIndex,
        budget: float,
        sink: EventSink,
    ) -> dict[str, Any]:
        # Performs a simple repair within budget
        consumed = min(budget, 10.0)
        index.add_edge(0, 1, level=0)
        return {"repairs_executed": 1, "budget_consumed": consumed}


class FakeEventSink:
    """In-memory concrete implementation exercising EventSink protocol."""

    def __init__(self) -> None:
        self.received: list[Event] = []

    def emit(self, event: Event) -> None:
        self.received.append(event)


def test_ann_and_graph_interfaces_behavior() -> None:
    """Exercise ANNIndex, GraphIndex, and MutableGraphIndex methods and properties."""
    idx = FakeMutableGraphIndex(dim=2)

    # Empty graph properties
    assert idx.dim == 2
    assert idx.size() == 0
    assert idx.max_level() == -1
    assert idx.entry_point() is None

    # Add points
    ids = np.array([10, 20], dtype=np.int64)
    vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    idx.add(ids, vectors)

    assert idx.size() == 2
    assert idx.contains(10)
    assert idx.contains(20)
    assert not idx.contains(999)
    assert set(idx.ids()) == {10, 20}
    assert idx.max_level() == 0
    assert idx.entry_point() == 10
    assert idx.version(10) == 0

    # Search returns squared L2 distances
    q = np.array([1.0, 2.0], dtype=np.float32)
    top_ids, top_dists = idx.search(q, k=2, ef=10)
    assert top_ids[0] == 10
    assert np.isclose(top_dists[0], 0.0)
    assert idx.distance_computations >= 2

    # Edge mutations and symmetry
    idx.add_edge(10, 20, level=0)
    assert 20 in idx.neighbors(10, 0)
    assert 10 in idx.in_neighbors(20, 0)

    idx.remove_edge(10, 20, level=0)
    assert 20 not in idx.neighbors(10, 0)
    assert 10 not in idx.in_neighbors(20, 0)

    idx.set_neighbors(10, 0, [20])
    assert idx.neighbors(10, 0) == [20]
    assert idx.in_neighbors(20, 0) == [10]

    # Lifecycle: read old vector -> move -> repairer.on_move
    old_vec = idx.vector(10)
    new_vec = np.array([1.1, 2.1], dtype=np.float32)
    idx.move(10, new_vec)
    assert idx.version(10) == 1
    assert np.allclose(idx.vector(10), new_vec)

    # Repairer interaction
    sink = FakeEventSink()
    repairer = FakeRepairer()
    repairer.on_move(idx, 10, old_vec, new_vec, sink)
    assert len(sink.received) == 1
    assert sink.received[0].type == EventType.cert_fail
    assert sink.received[0].version == 1

    rep_summary = repairer.run_repairs(idx, budget=50.0, sink=sink)
    assert rep_summary["repairs_executed"] == 1
    assert rep_summary["budget_consumed"] <= 50.0

    # Removal
    idx.remove(20)
    assert not idx.contains(20)
    assert idx.size() == 1

    # Snapshot
    snap = idx.snapshot()
    assert snap["size"] == 1


def test_movement_model_and_trajectory_behavior() -> None:
    """Exercise MovementModel and LinearTrajectory protocol implementations."""
    model = FakeMovementModel(name="test_model", seed=42)
    assert model.name == "test_model"
    assert model.seed == 42

    pos = model.positions(1.5)
    assert pos.shape == (1, 2)
    changed = model.changed_nodes(1.0, 1.5)
    assert changed == [0]

    traj = FakeLinearTrajectory()
    a, b = traj.trajectory(5)
    assert a.shape == (2,)
    assert b.shape == (2,)


def test_protocol_types_registered() -> None:
    """Confirm protocol type relationships with FakeMutableGraphIndex."""
    idx = FakeMutableGraphIndex(dim=2)
    assert isinstance(idx, ANNIndex)
    assert isinstance(idx, GraphIndex)
    assert isinstance(idx, MutableGraphIndex)

    repairer = FakeRepairer()
    assert isinstance(repairer, Repairer)

    sink = FakeEventSink()
    assert isinstance(sink, EventSink)

    model = FakeMovementModel()
    assert isinstance(model, MovementModel)

    traj = FakeLinearTrajectory()
    assert isinstance(traj, LinearTrajectory)
