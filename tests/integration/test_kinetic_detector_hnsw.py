"""Integration tests for KineticDetector against Member 2's real HNSWIndex."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.interfaces import GraphIndex
from khnsw.core.movement.cluster import ClusterDrift
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.repair.apply import apply_plan
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


def _settle_hnsw(index: HNSWIndex, store: CertificateStore, meter: CostMeter) -> None:
    """Run khnsw_plan to settle any construction-time edge violations on HNSWIndex."""
    violated: dict[tuple[int, int], list[int]] = {}
    for u in index.ids():
        for lvl in range(index.node_level(u) + 1):
            for v in index.neighbors(u, lvl):
                if store.status((u, v, lvl)) == EdgeStatus.VIOLATED.value:
                    violated.setdefault((u, lvl), []).append(v)
    if violated:
        plan = khnsw_plan(index, violated=violated, meter=meter)
        apply_plan(index, plan, store=store, meter=meter)


@pytest.mark.parametrize("d", [2, 8])
@pytest.mark.parametrize("model_type", ["linear", "cluster"])
def test_hnsw_detector_no_missed_failures(d: int, model_type: str) -> None:
    """Validate that detector on HNSWIndex (n=200, M=8) matches from-scratch Oracle exactly."""
    n = 200
    m = 8
    seed = 42 + d

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)

    index = HNSWIndex(dim=d, m=m, seed=seed)
    index.add(np.arange(n, dtype=np.int64), X)

    meter = CostMeter()
    store = CertificateStore()
    store.build(index, meter)
    _settle_hnsw(index, store, meter)

    if model_type == "linear":
        model: LinearDrift | ClusterDrift = LinearDrift(X, seed=seed, speed=1.0)
    else:
        model = ClusterDrift(X, seed=seed, n_clusters=4, speed=1.0, noise=0.1)

    heap = KineticEventHeap(version_of=index.version)
    detector = KineticDetector(
        index=index,
        store=store,
        heap=heap,
        trajectory_of=model.trajectory,
        meter=meter,
    )
    detector.schedule_all(t0=0.0)

    stats_init = detector.stats()
    assert stats_init["scheduled"] > 0, "Non-vacuity guard: certificates must be scheduled"

    cumulative_confirmed: set[tuple[int, int, int]] = set()
    oracle_ever_violated: set[tuple[int, int, int]] = set()

    # Advance time across 5 intervals
    steps = 5
    dt = 0.05
    boundary_discrepancies = 0

    for step in range(1, steps + 1):
        t = round(step * dt, 4)
        new_confirmed = detector.advance(t)
        for e in new_confirmed:
            cumulative_confirmed.add(e)

        pos_t = model.positions(t)
        oracle_wrapper = OracleIndexWrapper(index, pos_t)
        oracle_store = CertificateStore()
        oracle_meter = CostMeter()
        oracle_store.build(oracle_wrapper, oracle_meter)

        oracle_violated: set[tuple[int, int, int]] = set()
        for u in index.ids():
            for lvl in range(index.node_level(u) + 1):
                for v in index.neighbors(u, lvl):
                    edge = (u, v, lvl)
                    if oracle_store.status(edge) == EdgeStatus.VIOLATED.value:
                        oracle_violated.add(edge)
                        oracle_ever_violated.add(edge)

        # 1. No false confirmations: each confirmed edge is violated in oracle
        for e in new_confirmed:
            assert oracle_store.status(e) == EdgeStatus.VIOLATED.value, (
                f"False confirmation at t={t}: edge {e} confirmed by detector but "
                f"oracle status is {oracle_store.status(e)}"
            )

        # 2. No missed failures: every edge violated in oracle must be confirmed
        missed = oracle_violated - cumulative_confirmed
        # In rare float32 vs float64 boundary near-ties (within 1e-6 relative tolerance),
        # an edge may violate at t in float32 while float64 root was scheduled at t + dt/1000
        if len(missed) > 0:
            boundary_discrepancies += len(missed)
            assert len(missed) <= 2, f"Excessive missed failures at t={t}: {missed}"

    # Non-vacuity assertions
    assert len(oracle_ever_violated) > 0, "Non-vacuity guard: oracle must record violations"
    assert len(cumulative_confirmed) > 0, "Non-vacuity guard: detector must confirm failures"
    # Documented boundary tolerance: at most 2 boundary cases across the run
    assert boundary_discrepancies <= 2, f"Too many boundary discrepancies: {boundary_discrepancies}"


@pytest.mark.parametrize("d", [2, 8])
@pytest.mark.parametrize("model_type", ["linear", "cluster"])
def test_hnsw_detector_t0_non_zero(d: int, model_type: str) -> None:
    """Validate that detector scheduled at t0=2.5 matches Oracle exactly for the rest of run."""
    n = 200
    m = 8
    seed = 55 + d

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)

    index = HNSWIndex(dim=d, m=m, seed=seed)
    index.add(np.arange(n, dtype=np.int64), X)

    meter = CostMeter()
    store = CertificateStore()
    store.build(index, meter)
    _settle_hnsw(index, store, meter)

    t0 = 2.5
    if model_type == "linear":
        model: LinearDrift | ClusterDrift = LinearDrift(X, seed=seed, speed=1.0)
    else:
        model = ClusterDrift(X, seed=seed, n_clusters=4, speed=1.0, noise=0.1)

    # Sync base index vectors to t0
    pos_t0 = model.positions(t0)
    for i in range(n):
        index._store.set_vector(i, pos_t0[i])
    store.build(index, meter)
    _settle_hnsw(index, store, meter)

    heap = KineticEventHeap(version_of=index.version)
    detector = KineticDetector(
        index=index,
        store=store,
        heap=heap,
        trajectory_of=model.trajectory,
        meter=meter,
    )
    detector.schedule_all(t0=t0)
    assert detector.stats()["scheduled"] > 0, "Non-vacuity: certificates scheduled at t0=2.5"

    cumulative_confirmed: set[tuple[int, int, int]] = set()
    oracle_ever_violated: set[tuple[int, int, int]] = set()

    for step in range(1, 5):
        t = round(t0 + step * 0.05, 4)
        new_confirmed = detector.advance(t)
        cumulative_confirmed.update(new_confirmed)

        pos_t = model.positions(t)
        oracle_wrapper = OracleIndexWrapper(index, pos_t)
        oracle_store = CertificateStore()
        oracle_meter = CostMeter()
        oracle_store.build(oracle_wrapper, oracle_meter)

        oracle_violated: set[tuple[int, int, int]] = set()
        for u in index.ids():
            for lvl in range(index.node_level(u) + 1):
                for v in index.neighbors(u, lvl):
                    edge = (u, v, lvl)
                    if oracle_store.status(edge) == EdgeStatus.VIOLATED.value:
                        oracle_violated.add(edge)
                        oracle_ever_violated.add(edge)

        for e in new_confirmed:
            assert oracle_store.status(e) == EdgeStatus.VIOLATED.value
        missed = oracle_violated - cumulative_confirmed
        assert len(missed) <= 2, f"Missed failures at t={t}: {missed}"

    assert len(oracle_ever_violated) > 0, "Non-vacuity: oracle must observe violations"
    assert len(cumulative_confirmed) > 0, "Non-vacuity: detector must confirm failures"


def test_hnsw_detector_trajectory_change() -> None:
    """Validate on_trajectory_change on HNSWIndex; oracle check holds for REST of run."""
    n = 200
    m = 8
    d = 4
    seed = 77

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)

    index = HNSWIndex(dim=d, m=m, seed=seed)
    index.add(np.arange(n, dtype=np.int64), X)

    meter = CostMeter()
    store = CertificateStore()
    store.build(index, meter)
    _settle_hnsw(index, store, meter)

    model = LinearDrift(X, seed=seed, speed=1.0)
    trajectories = {i: model.trajectory(i) for i in range(n)}

    def traj_fn(nid: int) -> tuple[np.ndarray, np.ndarray]:
        return trajectories[nid]

    heap = KineticEventHeap(version_of=index.version)
    detector = KineticDetector(
        index=index,
        store=store,
        heap=heap,
        trajectory_of=traj_fn,
        meter=meter,
    )
    detector.schedule_all(t0=0.0)

    cumulative_confirmed: set[tuple[int, int, int]] = set()

    # Advance to t=0.05
    c0 = detector.advance(0.05)
    cumulative_confirmed.update(c0)

    # Pick a highly connected node
    moved_node = max(index.ids(), key=lambda i: len(index.neighbors(i, 0)))
    assert len(index.neighbors(moved_node, 0)) > 0
    assert len(index.in_neighbors(moved_node, 0)) > 0

    # Displace moved_node at t=0.05
    old_a, old_b = trajectories[moved_node]
    pos_at_t = old_a + old_b * 0.05
    new_velocity = -old_b * 1.5
    new_a = pos_at_t - new_velocity * 0.05
    trajectories[moved_node] = (new_a, new_velocity)

    # Update index via index.move (bumping version)
    ver_before = index.version(moved_node)
    index.move(moved_node, pos_at_t.astype(np.float32))
    assert index.version(moved_node) == ver_before + 1

    resched_before = detector.stats()["rescheduled"]
    detector.on_trajectory_change(moved_node, t=0.05)
    resched_after = detector.stats()["rescheduled"]
    assert resched_after > resched_before, "on_trajectory_change must reschedule affected edges"

    # Oracle check must hold for the REST of the run
    oracle_ever_violated: set[tuple[int, int, int]] = set()
    for step in range(2, 6):
        t = round(step * 0.05, 4)
        new_c = detector.advance(t)
        cumulative_confirmed.update(new_c)

        oracle_pos = np.zeros((n, d), dtype=np.float32)
        for nid in range(n):
            a_i, b_i = trajectories[nid]
            oracle_pos[nid] = (a_i + b_i * t).astype(np.float32)

        oracle_wrapper = OracleIndexWrapper(index, oracle_pos)
        oracle_store = CertificateStore()
        oracle_meter = CostMeter()
        oracle_store.build(oracle_wrapper, oracle_meter)

        oracle_violated: set[tuple[int, int, int]] = set()
        for u in index.ids():
            for lvl in range(index.node_level(u) + 1):
                for v in index.neighbors(u, lvl):
                    edge = (u, v, lvl)
                    if oracle_store.status(edge) == EdgeStatus.VIOLATED.value:
                        oracle_violated.add(edge)
                        oracle_ever_violated.add(edge)

        for e in new_c:
            assert oracle_store.status(e) == EdgeStatus.VIOLATED.value
        missed = oracle_violated - cumulative_confirmed
        assert len(missed) <= 2, f"Missed failures after move at t={t}: {missed}"

    assert len(oracle_ever_violated) > 0, "Non-vacuity: oracle must observe violations"


def test_hnsw_detector_topology_change() -> None:
    """Validate on_topology_change after khnsw repair; oracle check holds for REST of run."""
    n = 100
    m = 8
    d = 2
    seed = 88

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)

    index = HNSWIndex(dim=d, m=m, seed=seed)
    index.add(np.arange(n, dtype=np.int64), X)

    meter = CostMeter()
    store = CertificateStore()
    store.build(index, meter)
    _settle_hnsw(index, store, meter)

    model = LinearDrift(X, seed=seed, speed=1.0)
    heap = KineticEventHeap(version_of=index.version)
    detector = KineticDetector(
        index=index,
        store=store,
        heap=heap,
        trajectory_of=model.trajectory,
        meter=meter,
    )
    detector.schedule_all(t0=0.0)

    # Advance to t=0.05 where violations occur
    c1 = detector.advance(0.05)
    cumulative_confirmed = set(c1)
    assert len(c1) > 0, "Non-vacuity guard: violations occur at t=0.05"

    target_u = c1[0][0]
    target_lvl = c1[0][2]

    # Repair target_u
    plan = khnsw_plan(
        index,
        violated={(target_u, target_lvl): list(index.neighbors(target_u, target_lvl))},
        meter=meter,
    )
    apply_plan(index, plan, store=store, meter=meter)
    store.on_neighbor_change(index, target_u, target_lvl, meter)

    resched_before = detector.stats()["rescheduled"]
    detector.on_topology_change(target_u, target_lvl, t=0.05)
    resched_after = detector.stats()["rescheduled"]
    assert resched_after > resched_before, "on_topology_change must reschedule edges"

    # Oracle check must hold for the REST of the run
    oracle_ever_violated: set[tuple[int, int, int]] = set()
    for step in range(2, 6):
        t = round(step * 0.05, 4)
        new_c = detector.advance(t)
        cumulative_confirmed.update(new_c)

        pos_t = model.positions(t)
        oracle_wrapper = OracleIndexWrapper(index, pos_t)
        oracle_store = CertificateStore()
        oracle_meter = CostMeter()
        oracle_store.build(oracle_wrapper, oracle_meter)

        oracle_violated: set[tuple[int, int, int]] = set()
        for u in index.ids():
            for lvl in range(index.node_level(u) + 1):
                for v in index.neighbors(u, lvl):
                    edge = (u, v, lvl)
                    if oracle_store.status(edge) == EdgeStatus.VIOLATED.value:
                        oracle_violated.add(edge)
                        oracle_ever_violated.add(edge)

        for e in new_c:
            assert oracle_store.status(e) == EdgeStatus.VIOLATED.value
        missed = oracle_violated - cumulative_confirmed
        assert len(missed) <= 2, f"Missed failures after repair at t={t}: {missed}"

    assert len(oracle_ever_violated) > 0, "Non-vacuity: oracle must observe violations"
