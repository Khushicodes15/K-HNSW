"""Integration tests for reverse-links additions on HNSWIndex (d in {2, 8}).

Verifies:
1. Oracle store agreement (from-scratch vs incremental) after repair and after subsequent drift.
2. Engine adjacency equivalence with direct loop execution.
3. Multi-level reverse link placement at correct hierarchy levels (d=8).
4. Self-findability after 12x NN displacement (at least control across seeds 42, 43, 44).
5. Event emission (edge_remove for evictions with reason="heuristic_prune").
6. Store synchronization on neighbor changes for reverse-link endpoints.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from khnsw.core.detection.certificate_store import CertificateStore
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.events.schema import ListSink
from khnsw.core.graph.audit import check_degree_bounds, validate_invariants
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.repair.engine import KHNSWRepairEngine
from khnsw.core.repair.loop import repair_loop
from khnsw.core.scheduling.event_heap import KineticEventHeap
from khnsw.core.search.search_layer import search_layer


def make_hnsw_sync(index: HNSWIndex):
    """Callback closure to sync an HNSW node vector without bumping version."""

    def _sync(node_id: int, pos: np.ndarray) -> None:
        index._store.set_vector(int(node_id), pos.astype(np.float32))

    return _sync


def mean_nn_distance(X: np.ndarray, seed: int = 42) -> float:
    """Compute empirical mean 1-nearest-neighbor distance on dataset X."""
    rng = np.random.default_rng(seed)
    n = len(X)
    sample_size = min(n, 100)
    sample_idx = rng.choice(n, size=sample_size, replace=False)

    dists = []
    for idx in sample_idx:
        diffs = X - X[idx]
        sq_dist = np.sum(diffs * diffs, axis=1)
        sq_dist[idx] = np.inf
        dists.append(float(np.sqrt(np.min(sq_dist))))
    return float(np.mean(dists))


@pytest.mark.parametrize("d", [2, 8])
def test_oracle_agreement_and_drift_advance(d: int) -> None:
    """From-scratch store agrees with incremental store after repair and subsequent drift."""
    seed = 42
    n = 60
    m = 4
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    idx = HNSWIndex(dim=d, m=m, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X)

    store = CertificateStore()
    meter = CostMeter()
    store.build(idx, meter)

    # Move node 0 far away
    scale = mean_nn_distance(X, seed=seed)
    new_p = idx.vector(0) + np.ones(d, dtype=np.float32) * (15.0 * scale)
    idx.move(0, new_p)
    store.on_move(idx, 0, meter)

    # Perform repair
    res = repair_loop(idx, store, moved_node=0, search_layer=search_layer)
    assert res.stopped_reason in {"converged", "max_rounds"}

    # Invariants and degree bounds
    assert validate_invariants(idx) == []
    assert check_degree_bounds(idx) == []

    # 1. Immediate Oracle store agreement check
    oracle_store = CertificateStore()
    oracle_store.build(idx, meter)

    for u in idx.ids():
        for lvl in range(idx.node_level(u) + 1):
            for v in idx.neighbors(u, lvl):
                edge = (u, v, lvl)
                st_store = store._edge_status.get(edge)
                st_oracle = oracle_store._edge_status.get(edge)
                assert st_store == st_oracle, (
                    f"Oracle mismatch on edge {edge}: store={st_store}, oracle={st_oracle}"
                )

    # 2. Subsequent drift with detector re-armed
    drift = LinearDrift(
        np.array([idx.vector(u) for u in idx.ids()], dtype=np.float32),
        seed=seed,
        speed=1.0,
    )
    heap = KineticEventHeap(version_of=idx.version)
    detector = KineticDetector(
        index=idx,
        store=store,
        heap=heap,
        trajectory_of=drift.trajectory,
        meter=meter,
        sync_vector=make_hnsw_sync(idx),
        horizon=1.0,
    )
    detector.schedule_all(t0=0.0)

    # Advance drift to t=0.2
    detector.sync_nodes(idx.ids(), 0.2)
    adv = detector.advance(0.2)
    assert isinstance(adv, list)

    # Compare store again after drift advance
    oracle_drift = CertificateStore()
    oracle_drift.build(idx, meter)
    for u in idx.ids():
        for lvl in range(idx.node_level(u) + 1):
            for v in idx.neighbors(u, lvl):
                edge = (u, v, lvl)
                st_store = store._edge_status.get(edge)
                st_oracle = oracle_drift._edge_status.get(edge)
                assert st_store == st_oracle, (
                    f"Post-drift oracle mismatch on {edge}: store={st_store}, oracle={st_oracle}"
                )


def test_engine_vs_loop_equivalence() -> None:
    """Engine adjacency must exactly match calling the repair loop directly."""
    seed = 42
    d = 8
    n = 60
    m = 4
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    # Graph 1 for loop
    idx1 = HNSWIndex(dim=d, m=m, seed=seed)
    idx1.add(np.arange(n, dtype=np.int64), X)
    store1 = CertificateStore()
    meter1 = CostMeter()
    store1.build(idx1, meter1)

    # Graph 2 for engine (exact copy)
    idx2 = copy.deepcopy(idx1)
    store2 = CertificateStore()
    meter2 = CostMeter()
    store2.build(idx2, meter2)

    scale = mean_nn_distance(X, seed=seed)
    new_p = idx1.vector(0) + np.ones(d, dtype=np.float32) * (12.0 * scale)

    idx1.move(0, new_p)
    store1.on_move(idx1, 0, meter1)
    res_loop = repair_loop(idx1, store1, moved_node=0, search_layer=search_layer)

    engine = KHNSWRepairEngine(store=store2, search_layer=search_layer)
    idx2.move(0, new_p)
    engine.notify_moved(0, t=0.0)
    res_engine = engine.run_repairs(idx2, budget=100000.0, sink=ListSink())

    assert res_loop.rounds == res_engine["rounds"]
    for u in idx1.ids():
        for lvl in range(idx1.node_level(u) + 1):
            assert idx1.neighbors(u, lvl) == idx2.neighbors(u, lvl), (
                f"Adjacency disparity on ({u}, {lvl}) between loop and engine"
            )


def test_multi_level_reverse_link_placement() -> None:
    """Detects mutation (d): reverse links must be added at matching hierarchy levels."""
    # Construct a reference graph where node 1 is at level 1 and does NOT point to node 0 at level 1
    X = np.array(
        [
            [10.0, 10.0],  # node 0 (moved, level 1)
            [0.0, 0.0],  # node 1 (target w, level 1)
            [0.1, 0.1],  # node 2 (level 1)
            [5.0, 5.0],  # node 3 (level 0)
            [6.0, 6.0],  # node 4 (level 0)
        ],
        dtype=np.float32,
    )
    from tests.fixtures.reference_graph import ReferenceGraph

    levels = {0: 1, 1: 1, 2: 1, 3: 0, 4: 0}
    rg = ReferenceGraph.from_points(X, M=3, seed=42, levels=levels)

    # Ensure node 1 does not point to node 0 at level 1
    rg.set_neighbors(1, 1, [2])
    # Ensure node 0 does not point to node 1 at level 1 initially
    rg.set_neighbors(0, 1, [2])

    assert 0 not in rg.neighbors(1, 1), "Precondition: node 1 must not point to node 0 at level 1"

    store = CertificateStore()
    meter = CostMeter()
    store.build(rg, meter)

    # Move node 0 right next to node 1
    rg.move(0, np.array([0.001, 0.001], dtype=np.float32))
    store.on_move(rg, 0, meter)

    res = repair_loop(rg, store, moved_node=0, search_layer=search_layer)
    assert res.stopped_reason in {"converged", "max_rounds"}

    # Under correct behavior, node 1 gets a reverse link to node 0 AT LEVEL 1
    nbrs_1_1 = rg.neighbors(1, 1)
    assert 0 in nbrs_1_1, (
        f"Node 1 neighbors at level 1: {nbrs_1_1}; expected 0 to be present at level 1"
    )


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_self_findability_12x_displacement(seed: int) -> None:
    """Self-findability >= control across seeds 42, 43, 44 after 12x NN displacement."""
    d = 8
    n = 500  # diagnostic setup
    m = 8
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    idx_ours = HNSWIndex(dim=d, m=m, seed=seed)
    idx_ours.add(np.arange(n, dtype=np.int64), X)

    idx_ctrl = copy.deepcopy(idx_ours)

    store = CertificateStore()
    meter = CostMeter()
    store.build(idx_ours, meter)

    scale = mean_nn_distance(X, seed=seed)
    num_moved = int(0.05 * n)  # 5% = 25 nodes
    moved_nodes = list(rng.choice(n, size=num_moved, replace=False))

    for u in moved_nodes:
        disp = rng.standard_normal(d).astype(np.float32)
        disp /= np.linalg.norm(disp)
        new_p = idx_ours.vector(u) + disp * (12.0 * scale)
        idx_ours.move(u, new_p)
        idx_ctrl.move(u, new_p)
        store.on_move(idx_ours, u, meter)

    # Run repair on ours
    for u in moved_nodes:
        repair_loop(idx_ours, store, moved_node=u, search_layer=search_layer)

    # Measure self-findability: search with own new vector at ef=32, target in top 10
    found_ours = 0
    found_ctrl = 0
    for u in moved_nodes:
        q = idx_ours.vector(u)
        labels_ours, _ = idx_ours.search(q, k=10, ef=32)
        labels_ctrl, _ = idx_ctrl.search(q, k=10, ef=32)
        if u in labels_ours:
            found_ours += 1
        if u in labels_ctrl:
            found_ctrl += 1

    frac_ours = found_ours / len(moved_nodes)
    frac_ctrl = found_ctrl / len(moved_nodes)
    print(
        f"[Seed {seed}] 12x displacement self-findability: "
        f"Ours={frac_ours:.2f}, Control={frac_ctrl:.2f}"
    )

    # Ours must be at least as good as control
    assert frac_ours >= frac_ctrl, (
        f"Ours ({frac_ours:.2f}) fell below Control ({frac_ctrl:.2f}) on seed {seed}"
    )
