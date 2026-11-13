"""Unit tests for reverse-links additions during K-HNSW kinetic graph repair.

Tests use ReferenceGraph test doubles to verify reverse edge formation, capacity checks,
pruning eviction, budget safety, event replay, and time shift invariance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from khnsw.core.detection.certificate_store import CertificateStore
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.events.schema import EventEmitter, EventType, ListSink
from khnsw.core.graph.audit import check_degree_bounds, validate_invariants
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.repair.loop import repair_loop
from khnsw.core.scheduling.event_heap import KineticEventHeap
from khnsw.core.search.search_layer import search_layer
from tests.fixtures.reference_graph import ReferenceGraph


def test_reverse_links_basic_and_spare_capacity() -> None:
    """After repair of a moved node, in-degree >= 1 and spare capacity neighbors point to it."""
    rng = np.random.default_rng(42)
    dim = 2
    n = 20
    X = rng.standard_normal((n, dim)).astype(np.float32)
    # Give all nodes level 0 for this unit test
    levels = {i: 0 for i in range(n)}
    rg = ReferenceGraph.from_points(X, M=6, seed=42, levels=levels)

    store = CertificateStore()
    meter = CostMeter()
    store.build(rg, meter)

    # Initial graph checks
    assert validate_invariants(rg) == [], "Initial invariants failed"
    assert check_degree_bounds(rg) == [], "Initial degree bounds failed"

    # Move node 0 far away so all its old edges become invalid and it finds a new neighborhood
    moved = 0
    new_pos = X[moved] + np.array([50.0, 50.0], dtype=np.float32)
    rg.move(moved, new_pos)
    store.on_move(rg, moved, meter)

    res = repair_loop(rg, store, moved_node=moved, search_layer=search_layer)
    assert res.stopped_reason in {"converged", "max_rounds"}

    new_out = rg.neighbors(moved, 0)
    assert len(new_out) > 0, "Precondition failed: moved node must have out-neighbors after repair"

    # In-degree of moved node at layer 0 must be >= 1
    in_nbrs_0 = rg.in_neighbors(moved, 0)
    assert len(in_nbrs_0) >= 1, f"Moved node in-degree {len(in_nbrs_0)} must be >= 1"

    # Any new out-neighbor that had spare capacity (degree < max_degree) must now point to moved
    cap = rg.max_degree(0)
    for w in new_out:
        if len(rg.neighbors(w, 0)) < cap:
            assert moved in rg.neighbors(w, 0), (
                f"Neighbor {w} had spare capacity but does not point to moved node {moved}"
            )


def test_reverse_links_full_neighbor_prune_and_replay() -> None:
    """Full neighbor (deg == cap) keeps deg <= cap, evicts prune drop, and events replay exactly."""
    rng = np.random.default_rng(100)
    dim = 2
    n = 15
    X = rng.standard_normal((n, dim)).astype(np.float32)
    levels = {i: 0 for i in range(n)}
    # Use small cap M=3 (layer 0 cap = 2*M = 6)
    rg = ReferenceGraph.from_points(X, M=3, seed=100, levels=levels)

    store = CertificateStore()
    meter = CostMeter()
    store.build(rg, meter)

    # Fill neighbor 1 to exactly capacity (cap=6) if not already full
    cap_0 = rg.max_degree(0)
    cur_nbrs_1 = list(rg.neighbors(1, 0))
    if len(cur_nbrs_1) < cap_0:
        for candidate in range(2, n):
            if candidate not in cur_nbrs_1:
                cur_nbrs_1.append(candidate)
            if len(cur_nbrs_1) == cap_0:
                break
        rg.set_neighbors(1, 0, cur_nbrs_1)
        store.on_neighbor_change(rg, 1, 0, meter)

    assert len(rg.neighbors(1, 0)) == cap_0, "Precondition failed: node 1 must be full"

    # Move node 0 very close to node 1 so node 1 is its closest neighbor and vice versa
    pos_1 = rg.vector(1)
    new_pos_0 = pos_1 + np.array([0.001, 0.001], dtype=np.float32)
    rg.move(0, new_pos_0)
    store.on_move(rg, 0, meter)

    # Record initial adjacency before repair for replay
    initial_adj: dict[tuple[int, int], set[int]] = {}
    for u in rg.ids():
        for lvl in range(rg.node_level(u) + 1):
            initial_adj[(u, lvl)] = set(rg.neighbors(u, lvl))

    sink = ListSink()
    emitter = EventEmitter(run_id="unit-test-replay", sink=sink)

    res = repair_loop(rg, store, moved_node=0, search_layer=search_layer, emitter=emitter)
    assert res.stopped_reason in {"converged", "max_rounds"}

    # Degree bounds must strictly hold
    assert check_degree_bounds(rg) == [], "Degree bounds failed after full neighbor prune"
    assert validate_invariants(rg) == [], "Graph invariants failed after full neighbor prune"
    deg_1 = len(rg.neighbors(1, 0))
    assert deg_1 <= cap_0, f"Node 1 degree {deg_1} exceeds cap {cap_0}"

    # Verify that an edge_remove event was emitted for the evicted neighbor of node 1
    remove_events = [
        ev
        for ev in sink.events
        if ev.type == EventType.edge_remove and ev.node == 1 and ev.level == 0
    ]
    add_events = [
        ev for ev in sink.events if ev.type == EventType.edge_add and ev.node == 1 and ev.level == 0
    ]
    assert len(add_events) >= 1, "Expected edge_add event on node 1 for reverse link"
    assert any(ev.payload["target_id"] == 0 for ev in add_events)
    assert len(remove_events) >= 1, "Expected edge_remove event for evicted edge on full node 1"

    # Event replay: applying edge_add / edge_remove must reproduce exact final adjacency
    replayed: dict[tuple[int, int], set[int]] = {k: set(v) for k, v in initial_adj.items()}
    for ev in sink.events:
        if ev.type == EventType.edge_add:
            assert ev.node is not None and ev.level is not None
            replayed[(ev.node, ev.level)].add(ev.payload["target_id"])
        elif ev.type == EventType.edge_remove:
            assert ev.node is not None and ev.level is not None
            replayed[(ev.node, ev.level)].discard(ev.payload["target_id"])

    for u in rg.ids():
        for lvl in range(rg.node_level(u) + 1):
            assert set(rg.neighbors(u, lvl)) == replayed[(u, lvl)], (
                f"Replay mismatch on ({u}, {lvl})"
            )


def test_reverse_links_budget_exhaustion_safety() -> None:
    """Budget exhaustion must never leave half-applied reverse links; still_violated reported."""
    rng = np.random.default_rng(42)
    dim = 2
    n = 20
    X = rng.standard_normal((n, dim)).astype(np.float32)
    levels = {i: 0 for i in range(n)}
    rg = ReferenceGraph.from_points(X, M=6, seed=42, levels=levels)

    store = CertificateStore()
    meter = CostMeter()
    store.build(rg, meter)

    rg.move(0, X[0] + np.array([20.0, 20.0], dtype=np.float32))
    store.on_move(rg, 0, meter)

    # Capture adjacency before attempting repair with zero budget
    adj_before = {u: set(rg.neighbors(u, 0)) for u in rg.ids()}

    # Call with budget=0 (exhausted immediately)
    res = repair_loop(rg, store, moved_node=0, search_layer=search_layer, budget=0)
    assert res.stopped_reason == "budget"
    assert res.rounds == 0

    # No half-applied reverse links or modified adjacency
    for u in rg.ids():
        assert set(rg.neighbors(u, 0)) == adj_before[u], f"Node {u} modified under budget=0"

    # Repeat with a very small budget that runs planning/reverse links but stops at safe point
    res2 = repair_loop(rg, store, moved_node=0, search_layer=search_layer, budget=5)
    assert res2.stopped_reason == "budget"
    # Invariants and degree bounds must hold
    assert validate_invariants(rg) == []
    assert check_degree_bounds(rg) == []


def test_reverse_links_time_shift_invariance() -> None:
    """t0 != 0 behaves identically to t0 == 0 up to a time shift."""
    dim = 2
    n = 25
    rng = np.random.default_rng(77)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    levels = {i: 0 for i in range(n)}

    # Run 1: t = 0.0
    rg1 = ReferenceGraph.from_points(X, M=6, seed=77, levels=levels)
    store1 = CertificateStore()
    meter1 = CostMeter()
    store1.build(rg1, meter1)
    rg1.move(0, X[0] + np.array([15.0, 15.0], dtype=np.float32))
    store1.on_move(rg1, 0, meter1)

    res1 = repair_loop(rg1, store1, moved_node=0, t=0.0, search_layer=search_layer)

    # Run 2: t = 100.0 (identical vectors, simply shifted in timestamp)
    rg2 = ReferenceGraph.from_points(X, M=6, seed=77, levels=levels)
    store2 = CertificateStore()
    meter2 = CostMeter()
    store2.build(rg2, meter2)
    rg2.move(0, X[0] + np.array([15.0, 15.0], dtype=np.float32))
    store2.on_move(rg2, 0, meter2)

    res2 = repair_loop(rg2, store2, moved_node=0, t=100.0, search_layer=search_layer)

    assert res1.rounds == res2.rounds
    assert res1.edges_repaired == res2.edges_repaired
    for u in rg1.ids():
        for lvl in range(rg1.node_level(u) + 1):
            assert rg1.neighbors(u, lvl) == rg2.neighbors(u, lvl)


def test_reverse_links_sync_nodes_called(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detects mutation (c): detector.sync_nodes called with w and N(w) before reverse prune."""
    dim = 2
    n = 20
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    levels = {i: 0 for i in range(n)}
    rg = ReferenceGraph.from_points(X, M=6, seed=42, levels=levels)

    store = CertificateStore()
    meter = CostMeter()
    store.build(rg, meter)

    drift = LinearDrift(X, seed=42, speed=2.0)
    heap = KineticEventHeap(version_of=rg.version)

    def sync_vec(node_id: int, pos: np.ndarray) -> None:
        rg._vectors[node_id] = pos.astype(np.float32)

    detector = KineticDetector(
        index=rg,
        store=store,
        heap=heap,
        trajectory_of=drift.trajectory,
        meter=meter,
        sync_vector=sync_vec,
        horizon=1.0,
    )
    detector.schedule_all(t0=0.0)

    post_search_rev_syncs: list[set[int]] = []
    in_rebuild = False

    def tracked_search(*args: Any, **kwargs: Any) -> Any:
        nonlocal in_rebuild
        in_rebuild = True
        return search_layer(*args, **kwargs)

    orig_sync_nodes = detector.sync_nodes

    def tracking_sync_nodes(nodes: Any, t: float) -> None:
        if in_rebuild:
            post_search_rev_syncs.append(set(nodes))
        orig_sync_nodes(nodes, t)

    monkeypatch.setattr(detector, "sync_nodes", tracking_sync_nodes)

    # Ensure no node already points to 0 so new neighbors will require reverse links
    for u in rg.ids():
        if 0 in rg.neighbors(u, 0):
            rg.set_neighbors(u, 0, [v for v in rg.neighbors(u, 0) if v != 0])

    # Move node 0 far away
    rg.move(0, X[0] + np.array([25.0, 25.0], dtype=np.float32))
    store.on_move(rg, 0, meter)

    res = repair_loop(
        rg,
        store,
        detector=detector,
        moved_node=0,
        t=0.5,
        search_layer=tracked_search,
        max_rounds=1,
    )
    assert res.stopped_reason in {"converged", "max_rounds"}

    # Verify that sync_nodes was invoked specifically for reverse-links after search_layer
    assert len(post_search_rev_syncs) >= 1, (
        "detector.sync_nodes was not called for reverse-link candidate nodes before pruning!"
    )
    new_out = rg.neighbors(0, 0)
    assert len(new_out) > 0
    assert any(w in post_search_rev_syncs[0] for w in new_out), (
        "Reverse-link sync_nodes did not include the new out-neighbor candidates!"
    )


def test_reverse_links_executed_only_once_per_moved_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detects mutation (g): reverse-link step must be executed once in round 0, not repeated."""
    from khnsw.core.graph.hnsw import HNSWIndex

    seed = 42
    n = 60
    d = 8
    m = 4
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    idx = HNSWIndex(dim=d, m=m, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X)

    store = CertificateStore()
    meter = CostMeter()
    store.build(idx, meter)

    # Move node 0 far away so it creates cascade violations across rounds
    idx.move(0, idx.vector(0) + np.ones(d, dtype=np.float32) * 15.0)
    store.on_move(idx, 0, meter)

    import khnsw.core.repair.loop as loop_mod

    rebuild_calls = 0
    orig_rebuild = loop_mod._rebuild_moved_out_links

    def tracking_rebuild(*args: Any, **kwargs: Any) -> Any:
        nonlocal rebuild_calls
        rebuild_calls += 1
        return orig_rebuild(*args, **kwargs)

    monkeypatch.setattr(loop_mod, "_rebuild_moved_out_links", tracking_rebuild)

    res = repair_loop(
        idx,
        store,
        moved_node=0,
        search_layer=search_layer,
        max_rounds=5,
    )
    # The scenario naturally takes 2 rounds to repair
    assert res.rounds >= 2, f"Expected at least 2 rounds, got {res.rounds}"
    # Rebuild and reverse-links must be executed strictly once, not repeated in subsequent rounds
    assert rebuild_calls == 1, (
        f"Reverse-link / rebuild was executed {rebuild_calls} times; expected strictly 1!"
    )
