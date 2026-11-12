"""Unit tests for KineticDetector using ReferenceGraph test double."""

from __future__ import annotations

import numpy as np

from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.certificates import first_violation_time
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.events.schema import EventEmitter, EventType, ListSink
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.repair.apply import apply_plan
from khnsw.core.repair.plan import khnsw_plan
from khnsw.core.scheduling.event_heap import KineticEventHeap
from tests.fixtures.reference_graph import ReferenceGraph


def make_refgraph_sync(graph: ReferenceGraph):
    """Test callback syncing ReferenceGraph vectors to float32. Test code may use private fields."""

    def _sync(node_id: int, pos: np.ndarray) -> None:
        graph._vectors[node_id] = pos.astype(np.float32)

    return _sync


def _settle_graph(graph: ReferenceGraph, store: CertificateStore, meter: CostMeter) -> None:
    """Run khnsw_plan to settle any construction-time edge violations."""
    violated: dict[tuple[int, int], list[int]] = {}
    for u in graph.ids():
        for lvl in range(graph.node_level(u) + 1):
            for v in graph.neighbors(u, lvl):
                if store.status((u, v, lvl)) == EdgeStatus.VIOLATED.value:
                    violated.setdefault((u, lvl), []).append(v)
    if violated:
        plan = khnsw_plan(graph, violated=violated, meter=meter)
        apply_plan(graph, plan, store=store, meter=meter)


def _build_oracle_store(graph: ReferenceGraph, positions: np.ndarray) -> CertificateStore:
    """Build a ground-truth from-scratch CertificateStore on a copy of graph with vectors at t."""
    n = len(positions)
    dim = positions.shape[1]
    oracle_graph = ReferenceGraph(dim=dim, M=graph._M)
    for i in range(n):
        oracle_graph._vectors[i] = positions[i].copy()
        oracle_graph._levels[i] = graph._levels[i]
        oracle_graph._versions[i] = graph._versions[i]
    for lvl in graph._adj:
        oracle_graph._adj[lvl] = {u: list(nbrs) for u, nbrs in graph._adj[lvl].items()}
        oracle_graph._rev[lvl] = {u: list(nbrs) for u, nbrs in graph._rev[lvl].items()}
    oracle_graph._update_entry_point()

    oracle_meter = CostMeter()
    oracle_store = CertificateStore()
    oracle_store.build(oracle_graph, oracle_meter)
    return oracle_store


def test_oracle_agreement_and_no_missed_failures() -> None:
    """Core property: every edge violated in the oracle at t was confirmed by the detector <= t."""
    n = 25
    d = 4
    M = 4
    seed = 42

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)
    graph = ReferenceGraph.from_points(X, M=M, seed=seed)

    meter = CostMeter()
    store = CertificateStore()
    store.build(graph, meter)
    _settle_graph(graph, store, meter)

    model = LinearDrift(X, seed=seed, speed=1.5)
    heap = KineticEventHeap(version_of=graph.version)
    detector = KineticDetector(
        index=graph,
        store=store,
        heap=heap,
        trajectory_of=model.trajectory,
        meter=meter,
        sync_vector=make_refgraph_sync(graph),
    )

    detector.schedule_all(t0=0.0)
    stats_init = detector.stats()
    assert stats_init["scheduled"] > 0, "Non-vacuity guard: certificates must be scheduled"

    cumulative_confirmed: set[tuple[int, int, int]] = set()
    oracle_ever_violated: set[tuple[int, int, int]] = set()

    # Advance time in 10 steps to T=1.0
    steps = 10
    dt = 0.1
    for step in range(1, steps + 1):
        t = round(step * dt, 4)
        new_confirmed = detector.advance(t)
        for e in new_confirmed:
            cumulative_confirmed.add(e)

        pos_t = model.positions(t)
        oracle_store = _build_oracle_store(graph, pos_t)

        oracle_violated: set[tuple[int, int, int]] = set()
        for u in graph.ids():
            for lvl in range(graph.node_level(u) + 1):
                for v in graph.neighbors(u, lvl):
                    edge = (u, v, lvl)
                    if oracle_store.status(edge) == EdgeStatus.VIOLATED.value:
                        oracle_violated.add(edge)
                        oracle_ever_violated.add(edge)

        # 1. No false confirmations: each confirmed failure is violated in oracle
        for e in new_confirmed:
            assert oracle_store.status(e) == EdgeStatus.VIOLATED.value, (
                f"False confirmation at t={t}: edge {e} confirmed but oracle says "
                f"{oracle_store.status(e)}"
            )

        # 2. No missed failures: every currently violated edge in oracle must be in
        # cumulative confirmed
        missed = oracle_violated - cumulative_confirmed
        assert len(missed) == 0, (
            f"Missed failures at t={t}: {missed} are VIOLATED in oracle but unconfirmed"
        )

    # Non-vacuity assertions: violations must have occurred
    assert len(oracle_ever_violated) > 0, "Non-vacuity guard: oracle must have observed violations"
    assert len(cumulative_confirmed) > 0, "Non-vacuity guard: detector must have confirmed failures"
    stats_end = detector.stats()
    assert stats_end["confirmed"] == len(cumulative_confirmed)


def test_t0_non_zero_oracle_agreement() -> None:
    """A run with t0 != 0 (e.g. t0=2.5) verifies correct window offset and oracle agreement."""
    n = 25
    d = 4
    M = 4
    seed = 45

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)
    graph = ReferenceGraph.from_points(X, M=M, seed=seed)

    meter = CostMeter()
    store = CertificateStore()
    store.build(graph, meter)
    _settle_graph(graph, store, meter)

    t0 = 2.5
    model = LinearDrift(X, seed=seed, speed=1.5)
    # Sync graph vectors to t0
    pos_t0 = model.positions(t0)
    for i in range(n):
        graph._vectors[i] = pos_t0[i].copy()
    store.build(graph, meter)
    _settle_graph(graph, store, meter)

    heap = KineticEventHeap(version_of=graph.version)
    detector = KineticDetector(
        index=graph,
        store=store,
        heap=heap,
        trajectory_of=model.trajectory,
        meter=meter,
        horizon=1.0,
        sync_vector=make_refgraph_sync(graph),
    )
    detector.schedule_all(t0=t0)
    assert detector.stats()["scheduled"] > 0, (
        "Non-vacuity: certificates must be scheduled at t0=2.5"
    )

    cumulative_confirmed: set[tuple[int, int, int]] = set()
    oracle_ever_violated: set[tuple[int, int, int]] = set()

    for step in range(1, 6):
        t = round(t0 + step * 0.05, 4)
        new_confirmed = detector.advance(t)
        for e in new_confirmed:
            cumulative_confirmed.add(e)

        pos_t = model.positions(t)
        oracle_store = _build_oracle_store(graph, pos_t)
        oracle_violated: set[tuple[int, int, int]] = set()
        for u in graph.ids():
            for lvl in range(graph.node_level(u) + 1):
                for v in graph.neighbors(u, lvl):
                    edge = (u, v, lvl)
                    if oracle_store.status(edge) == EdgeStatus.VIOLATED.value:
                        oracle_violated.add(edge)
                        oracle_ever_violated.add(edge)

        for e in new_confirmed:
            assert oracle_store.status(e) == EdgeStatus.VIOLATED.value
        missed = oracle_violated - cumulative_confirmed
        assert len(missed) == 0, f"Missed failures at t={t}: {missed}"

    assert len(oracle_ever_violated) > 0, "Non-vacuity: oracle must observe violations"
    assert len(cumulative_confirmed) > 0, "Non-vacuity: detector must confirm failures"


def test_event_emission_contract() -> None:
    """Event emission: cert_fail emitted exactly once per confirmed failure with valid envelope."""
    n = 20
    d = 4
    M = 4
    seed = 101

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)
    graph = ReferenceGraph.from_points(X, M=M, seed=seed)

    meter = CostMeter()
    store = CertificateStore()
    store.build(graph, meter)
    _settle_graph(graph, store, meter)

    sink = ListSink()
    emitter = EventEmitter("unit_test_trace", sink)

    model = LinearDrift(X, seed=seed, speed=2.0)
    heap = KineticEventHeap(version_of=graph.version, emitter=emitter)
    detector = KineticDetector(
        index=graph,
        store=store,
        heap=heap,
        trajectory_of=model.trajectory,
        meter=meter,
        sync_vector=make_refgraph_sync(graph),
        emitter=emitter,
    )

    detector.schedule_all(t0=0.0)
    confirmed_all: list[tuple[int, int, int]] = []
    for step in range(1, 11):
        t = round(step * 0.1, 4)
        c = detector.advance(t)
        confirmed_all.extend(c)

    assert len(confirmed_all) > 0, "Non-vacuity guard: confirmed edges must be non-empty"

    events = sink.events
    cert_fail_events = [e for e in events if e.type == EventType.cert_fail]

    # Exactly one cert_fail event per confirmed failure
    assert len(cert_fail_events) == len(confirmed_all)
    assert len(cert_fail_events) == detector.stats()["confirmed"]

    for ev in cert_fail_events:
        assert ev.node is not None
        assert ev.level is not None
        assert ev.version is not None
        assert ev.version == graph.version(ev.node)
        assert "neighbor_id" in ev.payload
        assert ev.payload["reason"] == "relative_distance_violation"
        assert "witness_id" in ev.payload


def test_trajectory_change_reschedule() -> None:
    """Trajectory change updates certificates; oracle agreement holds for the REST of the run."""
    n = 25
    d = 4
    M = 4
    seed = 202

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)
    graph = ReferenceGraph.from_points(X, M=M, seed=seed)

    meter = CostMeter()
    store = CertificateStore()
    store.build(graph, meter)
    _settle_graph(graph, store, meter)

    model = LinearDrift(X, seed=seed, speed=1.0)
    trajectories = {i: model.trajectory(i) for i in range(n)}

    def traj_fn(nid: int) -> tuple[np.ndarray, np.ndarray]:
        return trajectories[nid]

    heap = KineticEventHeap(version_of=graph.version)
    detector = KineticDetector(
        index=graph,
        store=store,
        heap=heap,
        trajectory_of=traj_fn,
        meter=meter,
        sync_vector=make_refgraph_sync(graph),
    )
    detector.schedule_all(t0=0.0)

    cumulative_confirmed: set[tuple[int, int, int]] = set()

    # Advance to t=0.2
    c0 = detector.advance(0.2)
    cumulative_confirmed.update(c0)

    # Select node with high degree to modify trajectory
    moved_node = max(graph.ids(), key=lambda i: len(graph.neighbors(i, 0)))
    assert len(graph.neighbors(moved_node, 0)) > 0, "Non-vacuity: node must have neighbors"
    assert len(graph.in_neighbors(moved_node, 0)) > 0, "Non-vacuity: node must have in-neighbors"

    # Modify trajectory of moved_node at t=0.2
    old_a, old_b = trajectories[moved_node]
    pos_at_02 = old_a + old_b * 0.2
    new_velocity = -old_b * 2.0  # Reverse and accelerate
    new_a = pos_at_02 - new_velocity * 0.2
    trajectories[moved_node] = (new_a, new_velocity)

    # Driver updates index via index.move (bumping version)
    graph.move(moved_node, pos_at_02.astype(np.float32))
    assert graph.version(moved_node) == 1

    # Call on_trajectory_change
    resched_before = detector.stats()["rescheduled"]
    detector.on_trajectory_change(moved_node, t=0.2)
    resched_after = detector.stats()["rescheduled"]
    assert resched_after > resched_before, "Non-vacuity guard: affected edges must be rescheduled"

    # Oracle check must hold for the REST of the run
    oracle_ever_violated: set[tuple[int, int, int]] = set()
    for step in range(3, 7):
        t = round(step * 0.1, 4)
        new_c = detector.advance(t)
        cumulative_confirmed.update(new_c)

        # Oracle positions computed from updated trajectories
        oracle_pos = np.zeros((n, d), dtype=np.float32)
        for nid in range(n):
            a_i, b_i = trajectories[nid]
            oracle_pos[nid] = (a_i + b_i * t).astype(np.float32)

        oracle_store = _build_oracle_store(graph, oracle_pos)
        oracle_violated: set[tuple[int, int, int]] = set()
        for u in graph.ids():
            for lvl in range(graph.node_level(u) + 1):
                for v in graph.neighbors(u, lvl):
                    edge = (u, v, lvl)
                    if oracle_store.status(edge) == EdgeStatus.VIOLATED.value:
                        oracle_violated.add(edge)
                        oracle_ever_violated.add(edge)

        # No false confirmations
        for e in new_c:
            assert oracle_store.status(e) == EdgeStatus.VIOLATED.value
        # No missed failures
        missed = oracle_violated - cumulative_confirmed
        assert len(missed) == 0, f"Missed failures after move at t={t}: {missed}"

    assert len(oracle_ever_violated) > 0, "Non-vacuity: oracle must observe violations"


def test_topology_change_reschedule() -> None:
    """Topology changes (repair) drop old certificates; oracle check holds for the REST of run."""
    n = 20
    d = 4
    M = 4
    seed = 303

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)
    graph = ReferenceGraph.from_points(X, M=M, seed=seed)

    meter = CostMeter()
    store = CertificateStore()
    store.build(graph, meter)
    _settle_graph(graph, store, meter)

    model = LinearDrift(X, seed=seed, speed=1.0)
    heap = KineticEventHeap(version_of=graph.version)
    detector = KineticDetector(
        index=graph,
        store=store,
        heap=heap,
        trajectory_of=model.trajectory,
        meter=meter,
        sync_vector=make_refgraph_sync(graph),
    )
    detector.schedule_all(t0=0.0)

    # Advance to t=0.1 where violations occur
    c1 = detector.advance(0.1)
    cumulative_confirmed = set(c1)
    assert len(c1) > 0, "Non-vacuity guard: violations must occur at t=0.1"

    # Pick a violated source node
    target_u = c1[0][0]
    target_lvl = c1[0][2]

    # Apply khnsw repair to target_u
    plan = khnsw_plan(
        graph,
        violated={(target_u, target_lvl): list(graph.neighbors(target_u, target_lvl))},
        meter=meter,
    )
    apply_plan(graph, plan, store=store, meter=meter)
    store.on_neighbor_change(graph, target_u, target_lvl, meter)

    resched_before = detector.stats()["rescheduled"]
    detector.on_topology_change(target_u, target_lvl, t=0.1)
    resched_after = detector.stats()["rescheduled"]
    assert resched_after > resched_before, "Non-vacuity: on_topology_change must reschedule edges"

    # Oracle check must hold for the REST of the run
    oracle_ever_violated: set[tuple[int, int, int]] = set()
    for step in range(2, 5):
        t = round(step * 0.1, 4)
        new_c = detector.advance(t)
        cumulative_confirmed.update(new_c)

        pos_t = model.positions(t)
        oracle_store = _build_oracle_store(graph, pos_t)
        oracle_violated: set[tuple[int, int, int]] = set()
        for u in graph.ids():
            for lvl in range(graph.node_level(u) + 1):
                for v in graph.neighbors(u, lvl):
                    edge = (u, v, lvl)
                    if oracle_store.status(edge) == EdgeStatus.VIOLATED.value:
                        oracle_violated.add(edge)
                        oracle_ever_violated.add(edge)

        for e in new_c:
            assert oracle_store.status(e) == EdgeStatus.VIOLATED.value
        missed = oracle_violated - cumulative_confirmed
        assert len(missed) == 0, f"Missed failures after repair at t={t}: {missed}"

    assert len(oracle_ever_violated) > 0, "Non-vacuity: oracle must observe violations"


def test_edge_cases() -> None:
    """Validate zero velocity, lookahead horizon, and compaction mid-run."""
    n = 15
    d = 4
    M = 3
    seed = 404

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)
    graph = ReferenceGraph.from_points(X, M=M, seed=seed)

    meter = CostMeter()
    store = CertificateStore()
    store.build(graph, meter)
    _settle_graph(graph, store, meter)

    # 1. Zero-velocity nodes: no failure events scheduled
    model_zero = LinearDrift(X, seed=seed, speed=0.0)
    heap_zero = KineticEventHeap(version_of=graph.version)
    detector_zero = KineticDetector(
        index=graph,
        store=store,
        heap=heap_zero,
        trajectory_of=model_zero.trajectory,
        meter=meter,
        sync_vector=make_refgraph_sync(graph),
    )
    detector_zero.schedule_all(t0=0.0)
    assert heap_zero.raw_size == 0
    confirmed_zero = detector_zero.advance(10.0)
    assert len(confirmed_zero) == 0

    # 2. Lookahead horizon truncation
    model_moving = LinearDrift(X, seed=seed, speed=1.5)
    heap_inf = KineticEventHeap(version_of=graph.version)
    det_inf = KineticDetector(
        index=graph,
        store=store,
        heap=heap_inf,
        trajectory_of=model_moving.trajectory,
        meter=meter,
        sync_vector=make_refgraph_sync(graph),
        horizon=float("inf"),
    )
    det_inf.schedule_all(t0=0.0)
    sched_inf = det_inf.stats()["scheduled"]

    heap_short = KineticEventHeap(version_of=graph.version)
    det_short = KineticDetector(
        index=graph,
        store=store,
        heap=heap_short,
        trajectory_of=model_moving.trajectory,
        meter=meter,
        sync_vector=make_refgraph_sync(graph),
        horizon=0.05,
    )
    det_short.schedule_all(t0=0.0)
    sched_short = det_short.stats()["scheduled"]

    assert sched_inf > sched_short, "Lookahead horizon must truncate distant failure events"
    assert sched_short >= 0

    # 3. Mid-run compaction does not affect correctness
    det_inf.advance(0.1)
    heap_inf.compact()
    det_inf.advance(0.2)
    assert det_inf.stats()["popped"] > 0


def test_failure_already_true_at_t0() -> None:
    """Failure already true at t0 is marked VIOLATED by store and receives no certificate."""
    dim = 2
    graph = ReferenceGraph(dim=dim, M=4)
    # At t=0: u=(0,0), v=(2,0), w=(1,0). d(u,w)=1 <= d(u,v)=4, d(v,w)=1 < d(u,v)=4.
    # Therefore, 0 -> 1 is violated by 2 at t=0.
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    graph.add(np.arange(3, dtype=np.int64), pts)
    graph.set_neighbors(0, 0, [1, 2])

    meter = CostMeter()
    store = CertificateStore()
    store.build(graph, meter)
    assert store.status((0, 1, 0)) == EdgeStatus.VIOLATED.value

    trajs = {i: (pts[i].astype(np.float64), np.zeros(2, dtype=np.float64)) for i in range(3)}
    heap = KineticEventHeap(version_of=graph.version)
    detector = KineticDetector(
        graph, store, heap, lambda nid: trajs[nid], meter, sync_vector=make_refgraph_sync(graph)
    )
    detector.schedule_all(t0=0.0)

    # VIOLATED edges get no certificate in schedule_all
    assert (0, 1, 0) not in heap._live_edges
    assert detector.stats()["scheduled"] == 0


def test_tangent_degenerate_touch_no_event() -> None:
    """Isolated tangent touch does not satisfy open-interval policy (no event scheduled)."""
    dim = 2
    graph = ReferenceGraph(dim=dim, M=4)
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, -1.0]], dtype=np.float32)
    graph.add(np.arange(3, dtype=np.int64), pts)
    graph.set_neighbors(0, 0, [1, 2])

    meter = CostMeter()
    store = CertificateStore()
    store.build(graph, meter)
    assert store.status((0, 1, 0)) == EdgeStatus.VALID.value

    # Trajectories where g1 touches 0 at single instant t=1.0 with discriminant=0
    trajs = {
        0: (np.array([0.0, 0.0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)),
        1: (np.array([1.0, 0.0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)),
        2: (np.array([1.0, -1.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64)),
    }

    t_fail = first_violation_time(trajs[0], trajs[1], trajs[2], t0=0.0, t_end=2.0)
    assert t_fail is None, "Open interval policy must return None for tangent touch"

    heap = KineticEventHeap(version_of=graph.version)
    detector = KineticDetector(
        graph, store, heap, lambda nid: trajs[nid], meter, sync_vector=make_refgraph_sync(graph)
    )
    detector.schedule_all(t0=0.0)
    assert detector.stats()["scheduled"] == 0


def test_two_events_at_identical_time_deterministic_order() -> None:
    """Two events at identical failure time t pop and confirm in deterministic order."""
    dim = 2
    graph = ReferenceGraph(dim=dim, M=4)
    # Pair 1: 0 -> 1 with witness 2
    # Pair 2: 3 -> 4 with witness 5
    pts = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 10.0],
            [1.0, 10.0],
            [2.0, 10.0],
        ],
        dtype=np.float32,
    )
    graph.add(np.arange(6, dtype=np.int64), pts)
    graph.set_neighbors(0, 0, [1, 2])
    graph.set_neighbors(3, 0, [4, 5])

    meter = CostMeter()
    store = CertificateStore()
    store.build(graph, meter)
    assert store.status((0, 1, 0)) == EdgeStatus.VALID.value
    assert store.status((3, 4, 0)) == EdgeStatus.VALID.value

    trajs = {
        0: (np.array([0.0, 0.0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)),
        1: (np.array([1.0, 0.0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)),
        2: (np.array([2.0, 0.0], dtype=np.float64), np.array([-1.0, 0.0], dtype=np.float64)),
        3: (np.array([0.0, 10.0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)),
        4: (np.array([1.0, 10.0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)),
        5: (np.array([2.0, 10.0], dtype=np.float64), np.array([-1.0, 0.0], dtype=np.float64)),
    }

    heap = KineticEventHeap(version_of=graph.version)
    detector = KineticDetector(
        graph, store, heap, lambda nid: trajs[nid], meter, sync_vector=make_refgraph_sync(graph)
    )
    detector.schedule_all(t0=0.0)
    assert detector.stats()["scheduled"] == 2

    # Deterministic tie order check
    due = heap.pop_due(1.1)
    assert len(due) == 2
    assert due[0].edge == (0, 1, 0)
    assert due[1].edge == (3, 4, 0)

    # Put back into heap for advance verification
    heap.push((0, 1, 0), 1.0, 2, now=0.0)
    heap.push((3, 4, 0), 1.0, 5, now=0.0)

    confirmed = detector.advance(1.1)
    assert (0, 1, 0) in confirmed
    assert (3, 4, 0) in confirmed
    assert detector.stats()["confirmed"] == 2


def test_false_alarm_rescheduling() -> None:
    """When predicted event is a false alarm in float32, detector reschedules strictly forward."""
    dim = 2
    graph = ReferenceGraph(dim=dim, M=4)
    u_vec = np.array([0.0, 0.0], dtype=np.float32)
    v_vec = np.array([1.0, 0.0], dtype=np.float32)
    w_vec = np.array([2.0, 0.0], dtype=np.float32)

    graph.add(np.array([0, 1, 2], dtype=np.int64), np.stack([u_vec, v_vec, w_vec]))
    graph.set_neighbors(0, 0, [1, 2])

    meter = CostMeter()
    store = CertificateStore()
    store.build(graph, meter)
    assert store.status((0, 1, 0)) == EdgeStatus.VALID.value

    trajs = {
        0: (np.array([0.0, 0.0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)),
        1: (np.array([1.0, 0.0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)),
        2: (np.array([2.0, 0.0], dtype=np.float64), np.array([-1.0, 0.0], dtype=np.float64)),
    }

    t_fail = first_violation_time(trajs[0], trajs[1], trajs[2], t0=0.0, t_end=5.0)
    assert t_fail is not None
    assert abs(t_fail - 1.0) < 1e-6

    # Test callback where at t=1.0, w is slightly further away in float32 (simulating near-tie),
    # so float32 store evaluates VALID (false alarm at boundary t=1.0).
    def _sync(node_id: int, pos: np.ndarray) -> None:
        if node_id == 2 and abs(pos[0] - 1.0) < 1e-5:
            graph._vectors[node_id] = np.array([1.01, 0.0], dtype=np.float32)
        else:
            graph._vectors[node_id] = pos.astype(np.float32)

    heap = KineticEventHeap(version_of=graph.version)
    detector = KineticDetector(
        index=graph,
        store=store,
        heap=heap,
        trajectory_of=lambda nid: trajs[nid],
        meter=meter,
        sync_vector=_sync,
    )
    # Schedule certificates: event for (0, 1, 0) scheduled at t=1.0
    detector.schedule_all(t0=0.0)
    assert detector.stats()["scheduled"] == 1
    assert heap.peek_time() == 1.0

    # Advance to t=1.0: event pops, evaluates as VALID (false alarm at t=1.0)
    c1 = detector.advance(1.0)
    assert len(c1) == 0, "False alarm at t=1.0 must not be confirmed"
    assert detector.stats()["false_alarms"] == 1
    assert detector.stats()["rescheduled"] == 1

    # Strengthen to kill Mutation (h):
    # After false alarm at t_curr=1.0, heap's next event MUST be strictly > 1.0
    next_event_t = heap.peek_time()
    assert next_event_t is not None and next_event_t > 1.0, (
        f"Heap next event time {next_event_t} must be strictly > t_curr (1.0)"
    )

    # A second advance(1.0) must NOT pop or create a second false alarm
    c1_repeat = detector.advance(1.0)
    assert len(c1_repeat) == 0
    assert detector.stats()["false_alarms"] == 1, (
        "Second advance(t_curr) must not produce another false alarm or pop"
    )

    # At t=1.1: trajectories continue, violation occurs and is confirmed
    c2 = detector.advance(1.1)
    assert (0, 1, 0) in c2, "Rescheduled event must be confirmed when violation occurs"
    assert detector.stats()["confirmed"] == 1


def test_advance_past_horizon_without_refresh() -> None:
    """T > horizon demonstrates distant failures are missed without window refresh."""
    n = 25
    d = 4
    M = 4
    seed = 42

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)
    graph = ReferenceGraph.from_points(X, M=M, seed=seed)

    meter = CostMeter()
    store = CertificateStore()
    store.build(graph, meter)
    _settle_graph(graph, store, meter)

    model = LinearDrift(X, seed=seed, speed=1.5)
    heap = KineticEventHeap(version_of=graph.version)
    horizon = 0.05
    detector = KineticDetector(
        index=graph,
        store=store,
        heap=heap,
        trajectory_of=model.trajectory,
        meter=meter,
        horizon=horizon,
        sync_vector=make_refgraph_sync(graph),
    )
    detector.schedule_all(t0=0.0)
    assert detector.stats()["scheduled"] > 0, "Non-vacuity: certificates must be scheduled"

    # Advance past horizon to T = 0.30
    cumulative_confirmed: set[tuple[int, int, int]] = set()
    for step in range(1, 7):
        t = round(step * 0.05, 4)
        c = detector.advance(t)
        cumulative_confirmed.update(c)

    pos_t = model.positions(0.30)
    oracle_store = _build_oracle_store(graph, pos_t)
    oracle_violated = {
        e for e, s in oracle_store._edge_status.items() if s == EdgeStatus.VIOLATED.value
    }
    assert len(oracle_violated) > 0, "Precondition: distant violations exist"
    missed = oracle_violated - cumulative_confirmed
    assert len(missed) == 0, f"Distant failures missed after window refreshes: {missed}"
    assert detector.stats()["window_refreshes"] >= 5, "Window refreshes must have occurred"


def test_sync_nodes_updates_stored_vectors_without_bumping_version() -> None:
    """Public sync_nodes applies callback at time t without modifying node versions."""
    n = 10
    d = 2
    rng = np.random.default_rng(123)
    X = rng.standard_normal((n, d), dtype=np.float32)
    graph = ReferenceGraph.from_points(X, M=4, seed=123)
    meter = CostMeter()
    store = CertificateStore()
    store.build(graph, meter)

    model = LinearDrift(X, seed=123, speed=2.0)
    heap = KineticEventHeap(version_of=graph.version)
    detector = KineticDetector(
        index=graph,
        store=store,
        heap=heap,
        trajectory_of=model.trajectory,
        meter=meter,
        sync_vector=make_refgraph_sync(graph),
    )

    t = 1.75
    nodes_to_sync = [2, 5, 8]
    versions_before = {u: graph.version(u) for u in nodes_to_sync}

    detector.sync_nodes(nodes_to_sync, t=t)

    expected_pos = model.positions(t)
    for u in nodes_to_sync:
        np.testing.assert_allclose(graph.vector(u), expected_pos[u].astype(np.float32))
        assert graph.version(u) == versions_before[u], f"Version of node {u} must not bump"


def test_transient_violation_sampled_time_semantics() -> None:
    """Transient violations resolving between sample steps are counted as false alarms."""
    dim = 2
    graph = ReferenceGraph(dim=dim, M=4)
    # Static u=(0,0), static v=(1,0). Distance = 1.0
    u_vec = np.array([0.0, 0.0], dtype=np.float32)
    v_vec = np.array([1.0, 0.0], dtype=np.float32)
    # w moves along y: w(t) = (0.5, 2.0 - 4.0 * t)
    # At t=0: w=(0.5, 2.0). D(u, w)=4.25 > 1.0. VALID.
    # At t=0.5: w=(0.5, 0.0). D(u, w)=0.25 < 1.0, D(v, w)=0.25 < 1.0. VIOLATED.
    # At t=1.0: w=(0.5, -2.0). D(u, w)=4.25 > 1.0. VALID again.
    w_vec = np.array([0.5, 2.0], dtype=np.float32)

    graph.add(np.arange(3, dtype=np.int64), np.stack([u_vec, v_vec, w_vec]))
    graph.set_neighbors(0, 0, [1, 2])

    meter = CostMeter()
    store = CertificateStore()
    store.build(graph, meter)
    assert store.status((0, 1, 0)) == EdgeStatus.VALID.value

    trajs = {
        0: (np.array([0.0, 0.0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)),
        1: (np.array([1.0, 0.0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)),
        2: (np.array([0.5, 2.0], dtype=np.float64), np.array([0.0, -4.0], dtype=np.float64)),
    }

    heap = KineticEventHeap(version_of=graph.version)
    detector = KineticDetector(
        index=graph,
        store=store,
        heap=heap,
        trajectory_of=lambda nid: trajs[nid],
        meter=meter,
        sync_vector=make_refgraph_sync(graph),
    )
    detector.schedule_all(t0=0.0)
    assert detector.stats()["scheduled"] == 1, "Transient violation event must be scheduled"

    # Advance directly to t=1.0, jumping over the transient violation interval [0.28, 0.72]
    confirmed = detector.advance(1.0)

    # Sampled-time semantics: at t=1.0, edge is VALID.
    # Counted as false alarm, NOT confirmed failure.
    assert len(confirmed) == 0, "Transient violation must not be confirmed at t=1.0"
    assert detector.stats()["confirmed"] == 0
    assert detector.stats()["false_alarms"] == 1, (
        "Transient violation must be logged as false alarm"
    )


def test_window_refresh_with_trajectory_and_topology_change() -> None:
    """Window refresh preserves oracle agreement across mutations after refresh."""
    n = 25
    d = 4
    M = 4
    seed = 505

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)
    graph = ReferenceGraph.from_points(X, M=M, seed=seed)

    meter = CostMeter()
    store = CertificateStore()
    store.build(graph, meter)
    _settle_graph(graph, store, meter)

    model = LinearDrift(X, seed=seed, speed=1.0)
    trajectories = {i: model.trajectory(i) for i in range(n)}

    def traj_fn(nid: int) -> tuple[np.ndarray, np.ndarray]:
        return trajectories[nid]

    heap = KineticEventHeap(version_of=graph.version)
    horizon = 0.05
    detector = KineticDetector(
        index=graph,
        store=store,
        heap=heap,
        trajectory_of=traj_fn,
        meter=meter,
        horizon=horizon,
        sync_vector=make_refgraph_sync(graph),
    )
    detector.schedule_all(t0=0.0)

    cumulative_confirmed: set[tuple[int, int, int]] = set()

    # Step 1: Advance to 0.05
    c1 = detector.advance(0.05)
    cumulative_confirmed.update(c1)

    # Step 2: Advance to 0.10 (triggers first window refresh from [0, 0.05] to [0.05, 0.10])
    c2 = detector.advance(0.10)
    cumulative_confirmed.update(c2)
    assert detector.stats()["window_refreshes"] >= 1, "Window refresh must have occurred"

    # Mutate trajectory of a node at t=0.10
    moved_node = 0
    old_a, old_b = trajectories[moved_node]
    pos_at_t = old_a + old_b * 0.10
    new_velocity = -old_b * 2.0
    trajectories[moved_node] = (pos_at_t - new_velocity * 0.10, new_velocity)
    graph.move(moved_node, pos_at_t.astype(np.float32))
    detector.on_trajectory_change(moved_node, t=0.10)

    # Mutate topology: repair a node if any violations exist
    if cumulative_confirmed:
        target_u = list(cumulative_confirmed)[0][0]
        target_lvl = list(cumulative_confirmed)[0][2]
        plan = khnsw_plan(
            graph,
            violated={(target_u, target_lvl): list(graph.neighbors(target_u, target_lvl))},
            meter=meter,
        )
        apply_plan(graph, plan, store=store, meter=meter)
        store.on_neighbor_change(graph, target_u, target_lvl, meter)
        detector.on_topology_change(target_u, target_lvl, t=0.10)

    # Step 3: Advance past next window boundaries to 0.20 (triggers additional window refreshes)
    for step in range(3, 5):
        t = round(step * 0.05, 4)
        c = detector.advance(t)
        cumulative_confirmed.update(c)

    assert detector.stats()["window_refreshes"] >= 3, "Multiple refreshes must occur"

    # Oracle agreement check at t=0.20
    pos_20 = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        a_i, b_i = trajectories[i]
        pos_20[i] = (a_i + b_i * 0.20).astype(np.float32)
    oracle_store = _build_oracle_store(graph, pos_20)
    oracle_violated = {
        e for e, s in oracle_store._edge_status.items() if s == EdgeStatus.VIOLATED.value
    }
    missed = oracle_violated - cumulative_confirmed
    assert len(missed) == 0, f"Missed failures after refresh and mutations: {missed}"
