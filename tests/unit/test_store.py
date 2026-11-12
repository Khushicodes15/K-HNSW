"""Unit tests and Hypothesis property tests for GraphStore."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from khnsw.core.graph.store import GraphStore


def test_store_initialization() -> None:
    store = GraphStore(dim=16, m=8, m0=16)
    assert store.dim == 16
    assert store.m == 8
    assert store.m0 == 16
    assert store.max_m_for_level(0) == 16
    assert store.max_m_for_level(1) == 8

    # Default m0 = 2 * m
    store_default = GraphStore(dim=32, m=12)
    assert store_default.m0 == 24

    with pytest.raises(ValueError, match="dim must be positive"):
        GraphStore(dim=0)
    with pytest.raises(ValueError, match="m must be positive"):
        GraphStore(dim=8, m=0)
    with pytest.raises(ValueError, match="m0 must be positive"):
        GraphStore(dim=8, m=4, m0=-1)


def test_store_empty_state() -> None:
    store = GraphStore(dim=4)
    assert len(store) == 0
    assert 0 not in store
    assert store.max_level() == -1
    assert store.entry_point() is None

    snap = store.snapshot()
    assert snap["num_nodes"] == 0
    assert snap["max_level"] == -1
    assert snap["entry_point"] is None
    assert snap["layers"] == []


def test_store_add_node_validation() -> None:
    store = GraphStore(dim=3)
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    store.add_node(0, vec, level=2)
    assert 0 in store
    assert len(store) == 1
    assert store.node_level(0) == 2
    np.testing.assert_array_equal(store.vector(0), vec)

    # Duplicate node_id
    with pytest.raises(ValueError, match="already exists"):
        store.add_node(0, vec, level=0)

    # Negative node_id
    with pytest.raises(ValueError, match="node_id must be non-negative"):
        store.add_node(-1, vec, level=0)

    # Negative level
    with pytest.raises(ValueError, match="level must be non-negative"):
        store.add_node(1, vec, level=-1)

    # Dimension mismatch
    with pytest.raises(ValueError, match="vector shape must be"):
        store.add_node(2, np.zeros(4, dtype=np.float32), level=0)


def test_store_layer_hierarchy_invariant() -> None:
    store = GraphStore(dim=2)
    vec = np.array([0.5, -0.5], dtype=np.float32)

    # Node added at level 2 must be present and queryable at layers 0, 1, and 2
    store.add_node(10, vec, level=2)
    for lvl in (0, 1, 2):
        assert store.neighbors(10, lvl) == []
        assert store.in_neighbors(10, lvl) == []

    # Level 3 is out of bounds for node 10
    with pytest.raises(ValueError, match="out of bounds"):
        store.neighbors(10, 3)
    with pytest.raises(ValueError, match="out of bounds"):
        store.in_neighbors(10, 3)


def test_store_entry_point_tracking() -> None:
    store = GraphStore(dim=2)
    vec = np.zeros(2, dtype=np.float32)

    store.add_node(1, vec, level=1)
    assert store.max_level() == 1
    assert store.entry_point() == 1

    # Adding node at equal or lower level preserves current entry point
    store.add_node(2, vec, level=0)
    assert store.entry_point() == 1
    store.add_node(3, vec, level=1)
    assert store.entry_point() == 1

    # Adding node at strictly higher level updates entry point
    store.add_node(4, vec, level=3)
    assert store.max_level() == 3
    assert store.entry_point() == 4

    # Explicit set_entry_point validation
    with pytest.raises(KeyError):
        store.set_entry_point(999)

    # Cannot set node at level 1 when max_level is 3
    with pytest.raises(ValueError, match="!= max_level"):
        store.set_entry_point(1)


def test_store_add_and_remove_edge() -> None:
    store = GraphStore(dim=2, m=4, m0=8)
    vec = np.zeros(2, dtype=np.float32)
    store.add_node(0, vec, level=1)
    store.add_node(1, vec, level=1)

    # Add directed edge 0 -> 1 at level 0
    store.add_edge(0, 1, level=0)
    assert store.neighbors(0, level=0) == [1]
    assert store.in_neighbors(1, level=0) == [0]
    assert store.neighbors(1, level=0) == []
    assert store.in_neighbors(0, level=0) == []

    # Adding duplicate edge is idempotent
    store.add_edge(0, 1, level=0)
    assert store.neighbors(0, level=0) == [1]
    assert store.in_neighbors(1, level=0) == [0]

    # Remove edge
    store.remove_edge(0, 1, level=0)
    assert store.neighbors(0, level=0) == []
    assert store.in_neighbors(1, level=0) == []

    # Removing non-existent edge raises KeyError
    with pytest.raises(KeyError, match="does not exist"):
        store.remove_edge(0, 1, level=0)

    # Self-loops prohibited
    with pytest.raises(ValueError, match="Self-loops are not permitted"):
        store.add_edge(0, 0, level=0)


def test_store_degree_bounds() -> None:
    store = GraphStore(dim=1, m=2, m0=3)
    vec = np.zeros(1, dtype=np.float32)

    for i in range(10):
        store.add_node(i, vec, level=1)

    # Layer 0 has bound M0 = 3
    store.add_edge(0, 1, level=0)
    store.add_edge(0, 2, level=0)
    store.add_edge(0, 3, level=0)
    with pytest.raises(ValueError, match="reached max out-degree 3"):
        store.add_edge(0, 4, level=0)

    # Layer 1 has bound M = 2
    store.add_edge(0, 1, level=1)
    store.add_edge(0, 2, level=1)
    with pytest.raises(ValueError, match="reached max out-degree 2"):
        store.add_edge(0, 3, level=1)


def test_store_set_neighbors() -> None:
    store = GraphStore(dim=2, m=4, m0=6)
    vec = np.zeros(2, dtype=np.float32)
    for i in range(5):
        store.add_node(i, vec, level=1)

    # Initial set
    store.set_neighbors(0, level=0, neighbors=[1, 2, 3])
    assert set(store.neighbors(0, 0)) == {1, 2, 3}
    assert store.in_neighbors(1, 0) == [0]
    assert store.in_neighbors(2, 0) == [0]
    assert store.in_neighbors(3, 0) == [0]

    # Partial replacement: keep 2, drop 1 and 3, add 4
    store.set_neighbors(0, level=0, neighbors=[2, 4])
    assert set(store.neighbors(0, 0)) == {2, 4}
    assert store.in_neighbors(1, 0) == []
    assert store.in_neighbors(2, 0) == [0]
    assert store.in_neighbors(3, 0) == []
    assert store.in_neighbors(4, 0) == [0]

    # Exceeding degree limit raises ValueError
    with pytest.raises(ValueError, match="exceeds max out-degree"):
        store.set_neighbors(0, level=1, neighbors=[1, 2, 3, 4, 5])


def test_store_snapshot() -> None:
    store = GraphStore(dim=3, m=4, m0=8)
    vec = np.zeros(3, dtype=np.float32)
    store.add_node(0, vec, level=1)
    store.add_node(1, vec, level=0)
    store.add_edge(0, 1, level=0)

    snap = store.snapshot()
    assert snap["dim"] == 3
    assert snap["m"] == 4
    assert snap["m0"] == 8
    assert snap["num_nodes"] == 2
    assert snap["max_level"] == 1
    assert snap["entry_point"] == 0
    assert len(snap["layers"]) == 2
    assert snap["layers"][0]["num_nodes"] == 2
    assert snap["layers"][0]["num_edges"] == 1
    assert snap["layers"][1]["num_nodes"] == 1
    assert snap["layers"][1]["num_edges"] == 0


# Operation types for Hypothesis stateful property test
OpType = Literal["add_edge", "remove_edge", "set_neighbors"]


@settings(max_examples=50)
@given(
    nodes=st.lists(
        st.tuples(st.integers(min_value=0, max_value=20), st.integers(min_value=0, max_value=3)),
        min_size=2,
        max_size=10,
        unique_by=lambda x: x[0],
    ),
    operations=st.lists(
        st.tuples(
            st.sampled_from(["add_edge", "remove_edge", "set_neighbors"]),
            st.integers(min_value=0, max_value=15),  # u index
            st.integers(min_value=0, max_value=15),  # v index
            st.integers(min_value=0, max_value=3),  # level
        ),
        min_size=1,
        max_size=30,
    ),
)
def test_store_forward_reverse_sync_property(
    nodes: list[tuple[int, int]],
    operations: list[tuple[str, int, int, int]],
) -> None:
    """Property test: forward and reverse adjacency always remain strictly in sync."""
    dim = 4
    m = 5
    m0 = 8
    store = GraphStore(dim=dim, m=m, m0=m0)
    dummy_vec = np.zeros(dim, dtype=np.float32)

    for nid, lvl in nodes:
        store.add_node(nid, dummy_vec, level=lvl)

    node_ids = [nid for nid, _ in nodes]
    n_count = len(node_ids)

    for op, u_idx, v_idx, lvl in operations:
        u = node_ids[u_idx % n_count]
        v = node_ids[v_idx % n_count]

        try:
            if op == "add_edge":
                store.add_edge(u, v, level=lvl)
            elif op == "remove_edge":
                store.remove_edge(u, v, level=lvl)
            elif op == "set_neighbors":
                targets = [node_ids[(v_idx + i) % n_count] for i in range(min(3, n_count))]
                targets = [t for t in targets if t != u]
                store.set_neighbors(u, level=lvl, neighbors=targets)
        except (ValueError, KeyError):
            pass

    max_lvl = store.max_level()
    assert max_lvl >= 0

    for layer in range(max_lvl + 1):
        active_nodes = [nid for nid, lvl in nodes if lvl >= layer]
        for u in active_nodes:
            fwd = store.neighbors(u, layer)
            max_allowed = m0 if layer == 0 else m
            assert len(fwd) <= max_allowed

            for v in fwd:
                assert u in store.in_neighbors(v, layer)

        for v in active_nodes:
            rev = store.in_neighbors(v, layer)
            for u in rev:
                assert v in store.neighbors(u, layer)
