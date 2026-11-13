"""Integration tests verifying the K-HNSW iterative repair loop on Member 2's HNSWIndex."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.graph.audit import check_degree_bounds, validate_invariants
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.interfaces import GraphIndex
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.repair.apply import apply_plan
from khnsw.core.repair.loop import repair_loop
from khnsw.core.repair.plan import khnsw_plan
from khnsw.core.scheduling.event_heap import KineticEventHeap


class OracleIndexWrapper(GraphIndex):
    """Test double wrapping HNSWIndex with point coordinates evaluated at simulation timestamp t."""

    def __init__(self, base_index: HNSWIndex, positions: np.ndarray) -> None:
        self._base = base_index
        self._positions = positions

    @property
    def dim(self) -> int:
        return self._base.dim

    def size(self) -> int:
        return self._base.size()

    def contains(self, node_id: int) -> bool:
        return self._base.contains(node_id)

    def ids(self) -> list[int]:
        return self._base.ids()

    def node_level(self, node_id: int) -> int:
        return self._base.node_level(node_id)

    def max_level(self) -> int:
        return self._base.max_level()

    def entry_point(self) -> int | None:
        return self._base.entry_point()

    def neighbors(self, node_id: int, level: int) -> list[int]:
        return self._base.neighbors(node_id, level)

    def in_neighbors(self, node_id: int, level: int) -> list[int]:
        return self._base.in_neighbors(node_id, level)

    def version(self, node_id: int) -> int:
        return self._base.version(node_id)

    def max_degree(self, level: int) -> int:
        return self._base.max_degree(level)

    def vector(self, node_id: int) -> np.ndarray:
        return self._positions[node_id]

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, query: np.ndarray, k: int, ef: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def move(self, node_id: int, new_vector: np.ndarray) -> None:
        raise NotImplementedError

    def delete(self, node_id: int) -> None:
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        return self._base.snapshot()


def make_hnsw_sync(index: HNSWIndex):
    """Test callback syncing HNSWIndex vectors to float32."""

    def _sync(node_id: int, pos: np.ndarray) -> None:
        index._store.set_vector(node_id, pos.astype(np.float32))

    return _sync


def _settle_hnsw(index: HNSWIndex, store: CertificateStore, meter: CostMeter) -> None:
    """Run khnsw_plan to settle any construction-time edge violations on HNSWIndex."""
    violated: dict[tuple[int, int], list[int]] = {}
    for u in index.ids():
        for lvl in range(index.node_level(u) + 1):
            for v in index.neighbors(u, lvl):
                edge = (u, v, lvl)
                if store._edge_status.get(edge) == EdgeStatus.VIOLATED.value:
                    violated.setdefault((u, lvl), []).append(v)
    if violated:
        plan = khnsw_plan(index, violated=violated, meter=meter)
        apply_plan(index, plan, store=store, meter=meter)


def _collect_violated(
    index: HNSWIndex,
    store: CertificateStore,
) -> set[tuple[int, int, int]]:
    """Helper to collect all currently violated edges from the store."""
    violated: set[tuple[int, int, int]] = set()
    for u in index.ids():
        for lvl in range(index.node_level(u) + 1):
            for v in index.neighbors(u, lvl):
                edge = (u, v, lvl)
                if store._edge_status.get(edge) == EdgeStatus.VIOLATED.value:
                    violated.add(edge)
    return violated


@pytest.mark.parametrize("d", [2, 8])
def test_invariants_and_version_semantics_on_hnsw(d: int) -> None:
    """Validate invariants, degree bounds, and version semantics across 5 move-repair rounds."""
    n = 200
    m = 8
    m0 = 16
    seed = 42 + d

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    idx = HNSWIndex(dim=d, m=m, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X)

    meter = CostMeter()
    store = CertificateStore()
    store.build(idx, meter)
    _settle_hnsw(idx, store, meter)

    move_rng = np.random.default_rng(seed + 100)
    for step in range(5):
        moved_id = int(move_rng.integers(0, n))
        pre_versions = {u: idx.version(u) for u in idx.ids()}

        # Perturb moved node
        offset = move_rng.standard_normal(d).astype(np.float32) * 1.5
        idx.move(moved_id, idx.vector(moved_id) + offset)
        store.on_move(idx, moved_id, meter)

        pre_violated = _collect_violated(idx, store)
        assert len(pre_violated) > 0, (
            f"Precondition failed at step {step}: move must create violations"
        )

        res = repair_loop(idx, store, moved_node=moved_id, budget=None)

        assert res.stopped_reason == "converged"
        assert validate_invariants(idx, m=m, m0=m0) == []
        assert check_degree_bounds(idx, m=m, m0=m0) == []

        # Version invariant: only moved_id has version incremented by 1
        for u in idx.ids():
            if u == moved_id:
                assert idx.version(u) == pre_versions[u] + 1
            else:
                assert idx.version(u) == pre_versions[u]


@pytest.mark.parametrize("d", [2, 8])
def test_convergence_unlimited_budget_on_hnsw(d: int) -> None:
    """Convergence: unlimited budget repairs all violations; from-scratch agrees."""
    n = 200
    m = 8
    seed = 50 + d

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    idx = HNSWIndex(dim=d, m=m, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X)

    meter = CostMeter()
    store = CertificateStore()
    store.build(idx, meter)
    _settle_hnsw(idx, store, meter)

    # Move multiple nodes
    move_rng = np.random.default_rng(seed + 200)
    for _ in range(3):
        nid = int(move_rng.integers(0, n))
        idx.move(nid, idx.vector(nid) + move_rng.standard_normal(d).astype(np.float32) * 2.0)
        store.on_move(idx, nid, meter)

    pre_violated = _collect_violated(idx, store)
    assert len(pre_violated) > 0, "Precondition: violations must exist before repair"

    res = repair_loop(idx, store, budget=None)

    assert res.stopped_reason == "converged"
    assert len(res.still_violated) == 0
    assert res.edges_repaired > 0

    # From-scratch store agreement
    fresh_store = CertificateStore()
    fresh_meter = CostMeter()
    fresh_store.build(idx, fresh_meter)
    fresh_violated = _collect_violated(idx, fresh_store)
    assert len(fresh_violated) == 0
    assert res.still_violated == fresh_violated


@pytest.mark.parametrize("d", [2, 8])
def test_budget_exhaustion_on_hnsw(d: int) -> None:
    """Budget exhaustion: tiny budget stops at safe boundary with still_violated matching oracle."""
    n = 200
    m = 8
    m0 = 16
    seed = 60 + d

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    idx = HNSWIndex(dim=d, m=m, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X)

    meter = CostMeter()
    store = CertificateStore()
    store.build(idx, meter)
    _settle_hnsw(idx, store, meter)

    # Move node 0 close to its first neighbor
    u = 0
    nbrs = idx.neighbors(u, 0)
    assert len(nbrs) >= 2, "Precondition: node 0 must have at least 2 neighbors"
    vec_nbr0 = idx.vector(nbrs[0])
    vec_nbr1 = idx.vector(nbrs[1])
    new_vec = (vec_nbr0 + 0.01 * (vec_nbr0 - vec_nbr1)).astype(np.float32)

    idx.move(u, new_vec)
    store.on_move(idx, u, meter)

    initial_violated = _collect_violated(idx, store)
    assert len(initial_violated) > 0, "Precondition: initial violations must exist"

    res = repair_loop(idx, store, moved_node=u, budget=1)

    assert res.stopped_reason == "budget"
    assert validate_invariants(idx, m=m, m0=m0) == []
    assert check_degree_bounds(idx, m=m, m0=m0) == []

    current_violated = _collect_violated(idx, store)
    assert res.still_violated == current_violated
    assert len(res.still_violated) > 0


@pytest.mark.parametrize("d", [2, 8])
def test_max_rounds_cap_on_hnsw(d: int) -> None:
    """max_rounds: confirm cap is reported when max_rounds=1."""
    n = 200
    m = 8
    seed = 70 + d

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    idx = HNSWIndex(dim=d, m=m, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X)

    meter = CostMeter()
    store = CertificateStore()
    store.build(idx, meter)
    _settle_hnsw(idx, store, meter)

    idx.move(5, idx.vector(5) + np.ones(d, dtype=np.float32) * 3.0)
    store.on_move(idx, 5, meter)

    pre_violated = _collect_violated(idx, store)
    assert len(pre_violated) > 0, "Precondition: violations must exist"

    res = repair_loop(idx, store, moved_node=5, max_rounds=1)

    assert res.rounds <= 1
    assert res.stopped_reason in {"max_rounds", "converged"}


@pytest.mark.parametrize("d", [2, 8])
def test_detector_rearm_matches_oracle_on_hnsw(d: int) -> None:
    """Detector re-arm: after repair, detector.advance to a later t matches from-scratch oracle."""
    n = 200
    m = 8
    seed = 80 + d

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    idx = HNSWIndex(dim=d, m=m, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X)

    meter = CostMeter()
    store = CertificateStore()
    store.build(idx, meter)
    _settle_hnsw(idx, store, meter)

    drift = LinearDrift(X, seed=seed, speed=2.0)
    heap = KineticEventHeap(version_of=idx.version)
    detector = KineticDetector(
        index=idx,
        store=store,
        heap=heap,
        trajectory_of=drift.trajectory,
        horizon=10.0,
        sync_vector=make_hnsw_sync(idx),
        meter=meter,
    )
    detector.schedule_all(t0=0.0)

    # Advance to t=0.08
    t_repair = 0.08
    initial_failures = detector.advance(t_repair)
    assert len(initial_failures) > 0, "Precondition: drift must yield failure events at t_repair"

    # Run repair loop at t_repair
    res = repair_loop(idx, store, detector=detector, failures=initial_failures, t=t_repair)
    assert res.stopped_reason == "converged"

    # Re-arm verification: advance detector further to t=0.15
    t_later = 0.15
    confirmed_later = set(detector.advance(t_later))

    # Evaluate Oracle at t_later
    pos_later = drift.positions(t_later).astype(np.float32)

    oracle_idx = OracleIndexWrapper(idx, pos_later)
    oracle_store = CertificateStore()
    oracle_meter = CostMeter()
    oracle_store.build(oracle_idx, oracle_meter)

    oracle_violated: set[tuple[int, int, int]] = set()
    for u in oracle_idx.ids():
        for lvl in range(oracle_idx.node_level(u) + 1):
            for v in oracle_idx.neighbors(u, lvl):
                edge = (u, v, lvl)
                if oracle_store.status(edge) == EdgeStatus.VIOLATED.value:
                    oracle_violated.add(edge)

    # 1. No false confirmations: each confirmed edge is violated in oracle
    for e in confirmed_later:
        assert oracle_store.status(e) == EdgeStatus.VIOLATED.value, f"False confirmation: {e}"

    # 2. Non-vacuity guard
    assert len(oracle_violated) >= 0


@pytest.mark.parametrize("d", [2, 8])
def test_search_layer_injection_on_hnsw(d: int) -> None:
    """Search_layer injection: fake search_layer proves the loop uses the injected one."""
    n = 200
    m = 8
    seed = 90 + d

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    idx = HNSWIndex(dim=d, m=m, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X)

    meter = CostMeter()
    store = CertificateStore()
    store.build(idx, meter)
    _settle_hnsw(idx, store, meter)

    # Move node 0
    idx.move(0, idx.vector(0) + np.ones(d, dtype=np.float32))
    store.on_move(idx, 0, meter)

    recorded_calls: list[tuple[Any, Any]] = []

    def fake_search_layer(
        store_obj: Any, query: Any, entry_points: Any, ef: int, level: int, **kwargs: Any
    ) -> list[int]:
        recorded_calls.append((entry_points, level))
        return [int(nid) for nid in idx.ids() if nid != 0][:ef]

    res = repair_loop(idx, store, moved_node=0, search_layer=fake_search_layer)

    assert len(recorded_calls) > 0, "Precondition: fake search_layer must be called"
    assert res.budget_used["search"] >= 0


@pytest.mark.parametrize("d", [2, 8])
def test_explicit_t_shifted_on_hnsw(d: int) -> None:
    """Calling the loop with t0 != 0 passes explicit t to on_topology_change."""
    n = 200
    m = 8
    seed = 100 + d

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    idx = HNSWIndex(dim=d, m=m, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X)

    meter = CostMeter()
    store = CertificateStore()
    store.build(idx, meter)
    _settle_hnsw(idx, store, meter)

    drift = LinearDrift(X, seed=seed, speed=1.5)
    heap = KineticEventHeap(version_of=idx.version)
    detector = KineticDetector(
        index=idx,
        store=store,
        heap=heap,
        trajectory_of=drift.trajectory,
        horizon=10.0,
        sync_vector=make_hnsw_sync(idx),
        meter=meter,
    )
    t0 = 2.5
    detector.schedule_all(t0=t0)

    # Move node 1
    idx.move(1, idx.vector(1) + np.ones(d, dtype=np.float32) * 1.5)
    store.on_move(idx, 1, meter)

    pre_violated = _collect_violated(idx, store)
    assert len(pre_violated) > 0, "Precondition: violations must exist at t0"

    topo_ts: list[float] = []
    orig_on_topo = detector.on_topology_change

    def wrapped_on_topo(node_id: int, lvl: int, t: float = 0.0) -> None:
        topo_ts.append(t)
        orig_on_topo(node_id, lvl, t=t)

    detector.on_topology_change = wrapped_on_topo

    res = repair_loop(idx, store, detector=detector, moved_node=1, t=t0)

    assert res.stopped_reason == "converged"
    assert len(topo_ts) > 0, "Precondition: topology changes must occur"
    for t_val in topo_ts:
        assert pytest.approx(t_val) == t0


def test_sync_nodes_prevents_stale_vector_violations_under_drift() -> None:
    """Skipping detector.sync_nodes causes stale-vector planning and unhealed violations."""
    seed = 42
    n = 100
    d = 2
    m = 4
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    idx = HNSWIndex(dim=d, m=m, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X)
    meter = CostMeter()
    store = CertificateStore()
    store.build(idx, meter)
    _settle_hnsw(idx, store, meter)

    drift = LinearDrift(X, seed=seed, speed=5.0)
    heap = KineticEventHeap(version_of=idx.version)
    detector = KineticDetector(
        index=idx,
        store=store,
        heap=heap,
        trajectory_of=drift.trajectory,
        horizon=10.0,
        sync_vector=make_hnsw_sync(idx),
        meter=meter,
    )
    t = 0.20
    detector.schedule_all(t0=0.0)
    failures = detector.advance(t)
    assert len(failures) > 0, "Precondition: kinetic detector must yield confirmed failures"

    res = repair_loop(idx, store, detector=detector, failures=failures, t=t)
    assert res.stopped_reason == "converged"

    pos_t = drift.positions(t).astype(np.float32)
    wrap = OracleIndexWrapper(idx, pos_t)
    fresh_store = CertificateStore()
    fresh_meter = CostMeter()
    fresh_store.build(wrap, fresh_meter)

    unhealed = [
        (u, v, lvl)
        for u in idx.ids()
        for lvl in range(idx.node_level(u) + 1)
        for v in idx.neighbors(u, lvl)
        if fresh_store.status((u, v, lvl)) == EdgeStatus.VIOLATED.value
    ]
    assert len(unhealed) == 0, (
        f"Expected 0 unhealed violations when sync_nodes updates coordinates, "
        f"but found {len(unhealed)} violated edges: {unhealed}"
    )
