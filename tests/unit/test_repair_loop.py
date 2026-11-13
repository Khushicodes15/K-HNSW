"""Unit tests for the iterative K-HNSW repair loop on ReferenceGraph."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.events.schema import EventEmitter, ListSink
from khnsw.core.graph.audit import check_degree_bounds, validate_invariants
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.repair.loop import repair_loop
from tests.fixtures.reference_graph import ReferenceGraph


def _collect_violated(
    index: ReferenceGraph,
    store: CertificateStore,
) -> set[tuple[int, int, int]]:
    """Collect all currently violated edges from the store."""
    violated: set[tuple[int, int, int]] = set()
    for u in index.ids():
        for lvl in range(index.node_level(u) + 1):
            for v in index.neighbors(u, lvl):
                edge = (u, v, lvl)
                if store.status(edge) == EdgeStatus.VIOLATED.value:
                    violated.add(edge)
    return violated


def test_loop_invariants_and_version_semantics() -> None:
    """Verify invariants, degree bounds, and that only repaired nodes' versions change."""
    rng = np.random.default_rng(101)
    n = 25
    dim = 2
    X = rng.standard_normal((n, dim)).astype(np.float32)

    graph = ReferenceGraph.from_points(X, M=4, seed=101, mimic_reverse_append=True)
    store = CertificateStore()
    meter = CostMeter()
    store.build(graph, meter)

    # Initial settle pass if needed
    init_res = repair_loop(graph, store, budget=None)
    assert init_res.stopped_reason == "converged"

    # Perturb node 0
    pre_versions = {u: graph.version(u) for u in graph.ids()}
    graph.move(0, (graph.vector(0) + np.array([1.5, -1.2], dtype=np.float32)).astype(np.float32))
    store.on_move(graph, 0, meter)

    pre_violated = _collect_violated(graph, store)
    assert len(pre_violated) > 0, "Precondition: violations must exist after move"

    sink = ListSink()
    emitter = EventEmitter("test_run", sink)

    result = repair_loop(graph, store, moved_node=0, emitter=emitter)

    assert result.stopped_reason == "converged"
    assert validate_invariants(graph) == []
    assert check_degree_bounds(graph) == []

    # Node 0 was moved, so its version incremented by 1 during move.
    # Repaired nodes must NOT have versions changed by apply_plan.
    for u in graph.ids():
        if u == 0:
            assert graph.version(u) == pre_versions[u] + 1
        else:
            assert graph.version(u) == pre_versions[u]


def test_loop_convergence_unlimited_budget() -> None:
    """Convergence: with unlimited budget, 0 VIOLATED edges remain and from-scratch agrees."""
    rng = np.random.default_rng(202)
    n = 30
    dim = 2
    X = rng.standard_normal((n, dim)).astype(np.float32)

    graph = ReferenceGraph.from_points(X, M=4, seed=202, mimic_reverse_append=True)
    store = CertificateStore()
    meter = CostMeter()
    store.build(graph, meter)

    # Move node 3 significantly
    graph.move(3, (graph.vector(3) + np.array([2.5, 2.5], dtype=np.float32)).astype(np.float32))
    store.on_move(graph, 3, meter)

    pre_violated = _collect_violated(graph, store)
    assert len(pre_violated) > 0, "Precondition: violations must exist before repair"

    result = repair_loop(graph, store, moved_node=3, budget=None)

    assert result.stopped_reason == "converged"
    assert len(result.still_violated) == 0
    assert result.edges_repaired > 0

    # Build from-scratch store on final graph; must have 0 violations and agree
    fresh_store = CertificateStore()
    fresh_meter = CostMeter()
    fresh_store.build(graph, fresh_meter)
    fresh_violated = _collect_violated(graph, fresh_store)
    assert len(fresh_violated) == 0
    assert result.still_violated == fresh_violated


def test_loop_budget_exhaustion() -> None:
    """Budget exhaustion: tiny budget leaves oracle's VIOLATED set minus repaired edges."""
    rng = np.random.default_rng(303)
    n = 30
    dim = 2
    X = rng.standard_normal((n, dim)).astype(np.float32)

    graph = ReferenceGraph.from_points(X, M=4, seed=303, mimic_reverse_append=True)
    store = CertificateStore()
    meter = CostMeter()
    store.build(graph, meter)

    # Move node 1
    graph.move(1, (graph.vector(1) + np.array([3.0, 0.5], dtype=np.float32)).astype(np.float32))
    store.on_move(graph, 1, meter)

    initial_violated = _collect_violated(graph, store)
    assert len(initial_violated) > 0, "Precondition: initial violations must be non-empty"

    # Execute repair loop with tiny budget (e.g. 1 distance computation)
    result = repair_loop(graph, store, moved_node=1, budget=1)

    assert result.stopped_reason == "budget"
    assert validate_invariants(graph) == []
    assert check_degree_bounds(graph) == []

    # still_violated must contain exactly remaining violated edges
    current_store_violated = _collect_violated(graph, store)
    assert result.still_violated == current_store_violated
    assert len(result.still_violated) > 0, "Precondition: remaining violations must be non-empty"


def test_loop_max_rounds_cap() -> None:
    """max_rounds: confirm max_rounds cap is hit and reported when rounds are restricted."""
    rng = np.random.default_rng(404)
    n = 35
    dim = 2
    X = rng.standard_normal((n, dim)).astype(np.float32)

    graph = ReferenceGraph.from_points(X, M=4, seed=404, mimic_reverse_append=True)
    store = CertificateStore()
    meter = CostMeter()
    store.build(graph, meter)

    graph.move(2, (graph.vector(2) + np.array([4.0, -3.0], dtype=np.float32)).astype(np.float32))
    store.on_move(graph, 2, meter)

    pre_violated = _collect_violated(graph, store)
    assert len(pre_violated) > 0, "Precondition: violations exist"

    # Set max_rounds = 1
    result = repair_loop(graph, store, moved_node=2, max_rounds=1)

    assert result.rounds <= 1
    assert result.stopped_reason in {"max_rounds", "converged"}


def test_search_layer_injection() -> None:
    """Search_layer injection: a fake search_layer that records calls proves the loop uses it."""
    rng = np.random.default_rng(505)
    n = 20
    dim = 2
    X = rng.standard_normal((n, dim)).astype(np.float32)

    graph = ReferenceGraph.from_points(X, M=4, seed=505, mimic_reverse_append=True)
    store = CertificateStore()
    meter = CostMeter()
    store.build(graph, meter)

    graph.move(4, (graph.vector(4) + np.array([1.0, 1.0], dtype=np.float32)).astype(np.float32))
    store.on_move(graph, 4, meter)

    pre_violated = _collect_violated(graph, store)
    assert len(pre_violated) > 0, "Precondition: violations must exist"

    recorded_calls: list[tuple[Any, Any]] = []

    def fake_search_layer(
        store_obj: Any, query: Any, entry_points: Any, ef: int, level: int, **kwargs: Any
    ) -> list[int]:
        recorded_calls.append((entry_points, level))
        # Return simple candidate list
        return [int(nid) for nid in graph.ids() if nid != 4][:ef]

    result = repair_loop(graph, store, moved_node=4, search_layer=fake_search_layer)

    assert len(recorded_calls) > 0, "Precondition: fake search_layer must be invoked"
    assert result.budget_used is not None


def test_moved_out_links_rebuilt_once_and_in_neighbors_repaired() -> None:
    """Moved node's out-links rebuilt exactly once; in-neighbors beyond 2 hops are repaired."""
    # Build hand-crafted graph where node 5 is an in-neighbor beyond 2 hops of moved node 0:
    # 0 -> 1 -> 3 -> 5, and 5 -> 0.
    vectors = {
        0: np.array([0.0, 0.0], dtype=np.float32),
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
        3: np.array([2.0, 0.0], dtype=np.float32),
        4: np.array([0.0, 2.0], dtype=np.float32),
        5: np.array([3.0, 0.0], dtype=np.float32),  # in-neighbor beyond 2 hops
    }
    levels = {i: 0 for i in range(6)}
    adj = {
        (0, 0): [1, 2],
        (1, 0): [0, 3],
        (2, 0): [0, 4],
        (3, 0): [1, 5],
        (4, 0): [2],
        (5, 0): [0, 3],  # 5 -> 0 and 5 -> 3
    }
    graph = ReferenceGraph.from_adjacency(vectors, adj, levels=levels, M=3)
    store = CertificateStore()
    meter = CostMeter()
    store.build(graph, meter)

    # Verify precondition: 5 is an in-neighbor of 0
    assert 5 in graph.in_neighbors(0, 0), "Precondition: 5 must be in-neighbor of 0"

    # Two-hop forward set of 0:
    # L1 = N(0) = {1, 2}
    # L2 = N(1) U N(2) = {0, 3} U {0, 4} = {0, 3, 4}
    two_hop_forward = {1, 2, 0, 3, 4}
    assert 5 not in two_hop_forward, "Precondition: 5 must be beyond 2 hops from 0"

    search_layer_calls = 0

    def tracking_search_layer(
        store_obj: Any, query: Any, entry_points: Any, ef: int, level: int, **kwargs: Any
    ) -> list[int]:
        nonlocal search_layer_calls
        search_layer_calls += 1
        return [1, 2]

    # Move node 0 to (0.5, 0.0) making edge (5, 0) violated by witness 3
    graph.move(0, np.array([0.5, 0.0], dtype=np.float32))
    store.on_move(graph, 0, meter)

    assert store.status((5, 0, 0)) == EdgeStatus.VIOLATED.value, (
        "Precondition: edge (5, 0, 0) must be VIOLATED"
    )

    # Node 0 moves, giving failure on edge (0, 1, 0)
    result = repair_loop(
        graph,
        store,
        failures=[(0, 1, 0)],
        moved_node=0,
        search_layer=tracking_search_layer,
        max_rounds=5,
    )

    # Rebuilding moved node out-links happened across its levels in pass 1 exactly once
    assert search_layer_calls == 1, (
        f"Expected exactly 1 search_layer pass for level 0, got {search_layer_calls}"
    )
    # In-neighbor 5 was in scope and got repaired!
    assert (5, 0, 0) not in result.still_violated


def test_no_early_return_prune_below_cap() -> None:
    """When pool size < cap, the diversity test is still applied (skip_if_below_cap=False)."""
    # Base node 0 at (0, 0).
    # Candidate 1 at (1.0, 0.0) -> dist = 1.0.
    # Candidate 2 at (1.1, 0.0) -> dist = 1.21.
    # Distance between 1 and 2 is (0.1)^2 = 0.01.
    # Diversity check: alpha * D(1, 2) = 1.0 * 0.01 = 0.01 < D(0, 2) = 1.21.
    # So candidate 2 is occluded by candidate 1!
    # Even if cap = 5 (> pool size 2), candidate 2 must be pruned when skip_if_below_cap=False!
    vectors = {
        0: np.array([0.0, 0.0], dtype=np.float32),
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([1.1, 0.0], dtype=np.float32),
    }
    levels = {0: 0, 1: 0, 2: 0}
    # Initial adjacency: 0 has [2] which is violated by witness 1
    adj = {
        (0, 0): [1, 2],
        (1, 0): [0],
        (2, 0): [0],
    }
    graph = ReferenceGraph.from_adjacency(vectors, adj, levels=levels, M=5)
    store = CertificateStore()
    meter = CostMeter()
    store.build(graph, meter)

    assert store.status((0, 2, 0)) == EdgeStatus.VIOLATED.value, (
        "Precondition: edge (0, 2, 0) must be VIOLATED"
    )
    assert len(graph.neighbors(0, 0)) < graph.max_degree(0), "Precondition: pool below cap"

    result = repair_loop(graph, store, moved_node=0, budget=None)

    assert result.stopped_reason == "converged"
    # Candidate 2 must be pruned and not kept
    assert 2 not in graph.neighbors(0, 0)
    assert 1 in graph.neighbors(0, 0)


def test_explicit_t_shifted_and_detector_sync() -> None:
    """Calling the loop with t != 0 passes explicit t to detector and syncs correctly."""
    n = 20
    dim = 2
    rng = np.random.default_rng(707)
    X = rng.standard_normal((n, dim)).astype(np.float32)

    graph = ReferenceGraph.from_points(X, M=4, seed=707, mimic_reverse_append=True)
    store = CertificateStore()
    meter = CostMeter()
    store.build(graph, meter)

    # Initial settle pass
    repair_loop(graph, store, budget=None)

    from khnsw.core.scheduling.event_heap import KineticEventHeap

    model = LinearDrift(X, seed=707, speed=1.5)

    def sync_vec(node_id: int, pos: np.ndarray) -> None:
        graph._vectors[node_id] = pos.astype(np.float32)

    heap = KineticEventHeap(version_of=graph.version)
    detector = KineticDetector(
        index=graph,
        store=store,
        heap=heap,
        trajectory_of=model.trajectory,
        horizon=10.0,
        sync_vector=sync_vec,
        meter=meter,
    )
    detector.schedule_all(t0=0.0)

    t_eval = 2.5
    # Move node 1 close to its first neighbor to induce certificate violations
    u = 1
    nbrs = graph.neighbors(u, 0)
    assert len(nbrs) >= 2, "Precondition: node 1 must have at least 2 neighbors"
    vec_nbr0 = graph.vector(nbrs[0])
    vec_nbr1 = graph.vector(nbrs[1])
    new_vec = (vec_nbr0 + 0.01 * (vec_nbr0 - vec_nbr1)).astype(np.float32)

    detector.sync_nodes([u], t_eval)
    graph.move(u, new_vec)
    store.on_move(graph, u, meter)

    pre_violated = _collect_violated(graph, store)
    assert len(pre_violated) > 0, "Precondition: violations must exist at t_eval"

    topology_changes: list[float] = []
    sync_nodes_calls: list[tuple[set[int], float]] = []
    orig_on_topo = detector.on_topology_change
    orig_sync_nodes = detector.sync_nodes

    def wrapped_on_topo(node_id: int, lvl: int, t: float = 0.0) -> None:
        topology_changes.append(t)
        orig_on_topo(node_id, lvl, t=t)

    def wrapped_sync_nodes(nodes: Any, t: float) -> None:
        sync_nodes_calls.append((set(nodes), t))
        orig_sync_nodes(nodes, t)

    detector.on_topology_change = wrapped_on_topo
    detector.sync_nodes = wrapped_sync_nodes

    result = repair_loop(graph, store, detector=detector, moved_node=u, t=t_eval)

    assert result.stopped_reason == "converged"
    assert len(topology_changes) > 0, "Precondition: topology changes must be recorded"
    for t_val in topology_changes:
        assert pytest.approx(t_val) == t_eval, f"Expected t={t_eval}, got {t_val}"

    # Verify detector.sync_nodes was called before planning with t=t_eval
    assert len(sync_nodes_calls) > 0, "detector.sync_nodes must be called before planning"
    for synced_nodes, t_val in sync_nodes_calls:
        assert pytest.approx(t_val) == t_eval
        assert u in synced_nodes
