"""Unit tests for beam search_layer (HNSW Algorithm 2)."""

from __future__ import annotations

import numpy as np
import pytest

from khnsw.core.graph.store import GraphStore
from khnsw.core.search.search_layer import search_layer


def test_search_layer_single_node() -> None:
    store = GraphStore(dim=2)
    vec = np.array([1.0, 2.0], dtype=np.float32)
    store.add_node(0, vec, level=0)

    query = np.array([1.0, 5.0], dtype=np.float32)
    # Distance: (1-1)^2 + (5-2)^2 = 9.0
    results = search_layer(store, query, entry_points=[0], ef=5, level=0)

    assert len(results) == 1
    assert results[0].node_id == 0
    assert pytest.approx(results[0].distance) == 9.0
    # Also verify tuple unpacking works: (distance, node_id)
    dist, nid = results[0]
    assert dist == results[0].distance
    assert nid == results[0].node_id


def test_search_layer_matches_brute_force_on_hand_built_graph() -> None:
    # 8 nodes in 2D arranged in a line: (0,0), (1,0), (2,0), ..., (7,0)
    # Fully connected as a bidirectional line graph: 0 <-> 1 <-> 2 ... <-> 7
    dim = 2
    store = GraphStore(dim=dim, m=4, m0=8)
    coords = [np.array([float(i), 0.0], dtype=np.float32) for i in range(8)]

    for i, c in enumerate(coords):
        store.add_node(i, c, level=0)

    for i in range(7):
        store.add_edge(i, i + 1, level=0)
        store.add_edge(i + 1, i, level=0)

    # Query near node 3: (3.2, 0.0)
    query = np.array([3.2, 0.0], dtype=np.float32)

    # Brute-force ground truth: squared Euclidean distances
    bf_dists = [(float(np.sum((c - query) ** 2)), i) for i, c in enumerate(coords)]
    bf_sorted = sorted(bf_dists, key=lambda x: (x[0], x[1]))

    # Search with ef=8 starting from entry point 0
    results = search_layer(store, query, entry_points=[0], ef=8, level=0)

    assert len(results) == 8
    for res, (expected_d, expected_id) in zip(results, bf_sorted, strict=True):
        assert res.node_id == expected_id
        assert pytest.approx(res.distance) == expected_d


def test_search_layer_ef_limits_result_size() -> None:
    # 15 nodes in a line graph
    dim = 2
    store = GraphStore(dim=dim, m=4, m0=8)
    for i in range(15):
        store.add_node(i, np.array([float(i), 0.0], dtype=np.float32), level=0)
    for i in range(14):
        store.add_edge(i, i + 1, level=0)
        store.add_edge(i + 1, i, level=0)

    query = np.array([0.0, 0.0], dtype=np.float32)

    # ef = 1 returns 1 closest
    res1 = search_layer(store, query, entry_points=[0], ef=1, level=0)
    assert len(res1) == 1
    assert res1[0].node_id == 0

    # ef = 4 returns 4 closest
    res4 = search_layer(store, query, entry_points=[0], ef=4, level=0)
    assert len(res4) == 4
    assert [r.node_id for r in res4] == [0, 1, 2, 3]

    # ef = 50 on 15 nodes returns all 15
    res50 = search_layer(store, query, entry_points=[0], ef=50, level=0)
    assert len(res50) == 15


def test_search_layer_multiple_entry_points() -> None:
    # Two disconnected components: {0, 1, 2} and {10, 11, 12}
    store = GraphStore(dim=1, m=4, m0=8)
    for i in [0, 1, 2]:
        store.add_node(i, np.array([float(i)], dtype=np.float32), level=0)
    store.add_edge(0, 1, level=0)
    store.add_edge(1, 2, level=0)

    for i in [10, 11, 12]:
        store.add_node(i, np.array([float(i)], dtype=np.float32), level=0)
    store.add_edge(10, 11, level=0)
    store.add_edge(11, 12, level=0)

    # Query is at 10.5 (closer to component 2)
    query = np.array([10.5], dtype=np.float32)

    # Starting only at 0 cannot reach 10..12
    res_single = search_layer(store, query, entry_points=[0], ef=5, level=0)
    assert {r.node_id for r in res_single}.issubset({0, 1, 2})

    # Providing both [0, 10] as entry points explores both components
    res_multi = search_layer(store, query, entry_points=[0, 10], ef=3, level=0)
    assert [r.node_id for r in res_multi] == [10, 11, 12]


def test_search_layer_at_upper_level() -> None:
    # 2-layer graph:
    # Level 0 has 0, 1, 2, 3, 4
    # Level 1 has only 0, 2, 4
    store = GraphStore(dim=1, m=4, m0=8)
    store.add_node(0, np.array([0.0], dtype=np.float32), level=1)
    store.add_node(1, np.array([1.0], dtype=np.float32), level=0)
    store.add_node(2, np.array([2.0], dtype=np.float32), level=1)
    store.add_node(3, np.array([3.0], dtype=np.float32), level=0)
    store.add_node(4, np.array([4.0], dtype=np.float32), level=1)

    # Connect upper layer: 0 <-> 2 <-> 4
    store.add_edge(0, 2, level=1)
    store.add_edge(2, 0, level=1)
    store.add_edge(2, 4, level=1)
    store.add_edge(4, 2, level=1)

    query = np.array([2.1], dtype=np.float32)

    # Search at level 1 should only return nodes present at level 1: {0, 2, 4}
    results = search_layer(store, query, entry_points=[0], ef=3, level=1)
    returned_ids = [r.node_id for r in results]
    assert returned_ids == [2, 0, 4] or returned_ids == [2, 4, 0]
    assert results[0].node_id == 2


def test_search_layer_validation_and_edge_cases() -> None:
    store = GraphStore(dim=2)
    store.add_node(0, np.zeros(2, dtype=np.float32), level=0)

    # ef < 1
    with pytest.raises(ValueError, match="ef must be >= 1"):
        search_layer(store, np.zeros(2, dtype=np.float32), entry_points=[0], ef=0, level=0)

    # level < 0
    with pytest.raises(ValueError, match="level must be non-negative"):
        search_layer(store, np.zeros(2, dtype=np.float32), entry_points=[0], ef=1, level=-1)

    # query dimension mismatch
    with pytest.raises(ValueError, match="query shape must be"):
        search_layer(store, np.zeros(3, dtype=np.float32), entry_points=[0], ef=1, level=0)

    # non-existent entry point
    with pytest.raises(KeyError, match="does not exist"):
        search_layer(store, np.zeros(2, dtype=np.float32), entry_points=[999], ef=1, level=0)

    # entry point level lower than search level
    with pytest.raises(ValueError, match="< search level"):
        search_layer(store, np.zeros(2, dtype=np.float32), entry_points=[0], ef=1, level=2)

    # empty entry points list
    assert search_layer(store, np.zeros(2, dtype=np.float32), entry_points=[], ef=1, level=0) == []
