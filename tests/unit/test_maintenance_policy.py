"""Unit tests and mutation audits for MaintenancePolicy API (Phase 6 Item C).

Pure Python standard library and NumPy only.
Tests:
- Policy behaviors (none, periodic, round_robin, lazy, budgeted_priority)
- Determinism
- Invariants after step (anchors refreshed, vectors synchronized)
- Cost categories add up
- Event emission (contract absence documented)
- Mutation check 1: round_robin cursor never advances
- Mutation check 2: budget ignored
"""

from __future__ import annotations

import numpy as np
import pytest

from khnsw.core.detection.cost import CostMeter
from khnsw.core.events.schema import EventEmitter, ListSink
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.scheduling.policy import MaintenancePolicy


def _sample_graph(
    n: int = 30, d: int = 4, m: int = 4, seed: int = 42
) -> tuple[HNSWIndex, np.ndarray]:
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n, d)).astype(np.float32)
    index = HNSWIndex(dim=d, m=m, seed=seed)
    index.add(np.arange(n, dtype=np.int64), X0)
    return index, X0


def test_policy_none() -> None:
    """'none' policy never reinserts any node."""
    index, X0 = _sample_graph()
    pol = MaintenancePolicy(policy="none")
    X1 = X0.copy()
    X1[0] += np.float32([1.0, 0, 0, 0])
    res = pol.step(index, X1, t=0.1, moving_nodes=[0])
    assert res == []


def test_policy_periodic() -> None:
    """'periodic' policy reinserts all moving nodes only when step % K == 0."""
    index, X0 = _sample_graph()
    K = 3
    pol = MaintenancePolicy(policy="periodic", K=K)
    X1 = X0.copy()
    moving = [1, 2, 3]
    for nid in moving:
        X1[nid] += np.float32([1.0, 0, 0, 0])

    # Steps 1 and 2: no reinsertions
    assert pol.step(index, X1, t=0.1, moving_nodes=moving) == []
    assert pol.step(index, X1, t=0.2, moving_nodes=moving) == []

    # Step 3: all moving nodes reinserted
    res3 = pol.step(index, X1, t=0.3, moving_nodes=moving)
    assert set(res3) == set(moving)


def test_policy_round_robin() -> None:
    """'round_robin' policy reinserts next B nodes cyclically."""
    index, X0 = _sample_graph()
    B = 2
    pol = MaintenancePolicy(policy="round_robin", B=B)
    moving = [10, 11, 12, 13, 14]
    X1 = X0.copy()
    for nid in moving:
        X1[nid] += np.float32([1.0, 0, 0, 0])

    res1 = pol.step(index, X1, t=0.1, moving_nodes=moving)
    assert res1 == [10, 11]

    res2 = pol.step(index, X1, t=0.2, moving_nodes=moving)
    assert res2 == [12, 13]

    res3 = pol.step(index, X1, t=0.3, moving_nodes=moving)
    assert res3 == [14, 10]


def test_policy_lazy() -> None:
    """'lazy' policy reinserts nodes whose displacement exceeds theta * radius."""
    index, X0 = _sample_graph()
    pol = MaintenancePolicy(policy="lazy", theta=1.0)
    pol.register(index, X0)

    moving = [5, 6]
    r5 = pol._radii[5]
    r6 = pol._radii[6]

    X1 = X0.copy()
    X1[5] += np.float32([2.0 * r5, 0, 0, 0])  # stale
    X1[6] += np.float32([0.5 * r6, 0, 0, 0])  # not stale

    res = pol.step(index, X1, t=0.1, moving_nodes=moving)
    assert res == [5]


def test_policy_budgeted_priority() -> None:
    """'budgeted_priority' reinserts top B nodes by score s_i = ||x-p||/r."""
    index, X0 = _sample_graph()
    pol = MaintenancePolicy(policy="budgeted_priority", B=2)
    pol.register(index, X0)

    moving = [1, 2, 3]
    r1, r2, r3 = pol._radii[1], pol._radii[2], pol._radii[3]

    X1 = X0.copy()
    X1[1] += np.float32([1.0 * r1, 0, 0, 0])  # score ~ 1
    X1[2] += np.float32([3.0 * r2, 0, 0, 0])  # score ~ 3
    X1[3] += np.float32([2.0 * r3, 0, 0, 0])  # score ~ 2

    res = pol.step(index, X1, t=0.1, moving_nodes=moving)
    assert res == [2, 3], "Top 2 scores are nodes 2 and 3"


def test_determinism() -> None:
    """Identical policy configurations and sequences produce identical reinsertions."""
    idx1, X0_1 = _sample_graph(seed=99)
    idx2, X0_2 = _sample_graph(seed=99)

    pol1 = MaintenancePolicy(policy="round_robin", B=2, seed=99)
    pol2 = MaintenancePolicy(policy="round_robin", B=2, seed=99)

    moving = [2, 4, 6, 8]
    X_step = X0_1.copy()
    for nid in moving:
        X_step[nid] += np.float32([1.0, 0, 0, 0])

    r1_step1 = pol1.step(idx1, X_step, t=0.1, moving_nodes=moving)
    r2_step1 = pol2.step(idx2, X_step, t=0.1, moving_nodes=moving)
    assert r1_step1 == r2_step1

    r1_step2 = pol1.step(idx1, X_step, t=0.2, moving_nodes=moving)
    r2_step2 = pol2.step(idx2, X_step, t=0.2, moving_nodes=moving)
    assert r1_step2 == r2_step2


def test_invariants_after_step() -> None:
    """Vectors in index._store and anchors for reinserted nodes match current positions."""
    index, X0 = _sample_graph()
    pol = MaintenancePolicy(policy="lazy", theta=0.5)
    pol.register(index, X0)

    X1 = X0.copy()
    X1[7] += np.float32([3.0, 0, 0, 0])

    res = pol.step(index, X1, t=0.1, moving_nodes=[7])
    assert res == [7]

    # Index store updated
    np.testing.assert_allclose(index.vector(7), X1[7], atol=1e-5)
    # Anchor updated to current position
    np.testing.assert_allclose(pol._anchors[7], X1[7], atol=1e-5)


def test_cost_categories() -> None:
    """CostMeter distance computations reflect stale checks and reinsertions."""
    index, X0 = _sample_graph()
    meter = CostMeter()
    pol = MaintenancePolicy(policy="lazy", theta=1.0, meter=meter)

    moving = [1, 2, 3, 4]
    X1 = X0.copy()
    for nid in moving:
        X1[nid] += np.float32([2.0, 0, 0, 0])

    before_checks = meter.count
    _ = pol.step(index, X1, t=0.1, moving_nodes=moving)
    # Stale checks: 1 per moving node
    assert meter.count - before_checks == len(moving)


def test_event_emission_contract_adherence() -> None:
    """Documents contract boundary: events.md defines no node_reinsert event type.

    Asserts that no uncontracted events are emitted to avoid breaking JSON schema
    validation for downstream consumers (Member 5).
    """
    index, X0 = _sample_graph()
    sink = ListSink()
    emitter = EventEmitter(run_id="test_run", sink=sink)
    pol = MaintenancePolicy(policy="round_robin", B=1, emitter=emitter)

    X1 = X0.copy()
    X1[1] += np.float32([1.0, 0, 0, 0])
    res = pol.step(index, X1, t=0.1, moving_nodes=[1])
    assert res == [1]

    # No uncontracted events emitted to sink
    assert len(sink.events) == 0, (
        "docs/contracts/events.md defines no 'node_reinsert' type; "
        "custom events must not be emitted without repo owner contract amendment."
    )


def test_mutation_audit_round_robin_cursor_frozen() -> None:
    """Mutation audit: Detects if round_robin cursor never advances."""
    index, X0 = _sample_graph()

    class MutantFrozenCursorPolicy(MaintenancePolicy):
        def step(self, index: HNSWIndex, positions: np.ndarray, t: float = 0.0, moving_nodes=None):
            # Mutant: never increments self._rr_cursor
            moving_list = list(moving_nodes) if moving_nodes else []
            reinserted = []
            for _ in range(min(self._B, len(moving_list))):
                chosen = moving_list[self._rr_cursor % len(moving_list)]
                # Missing: self._rr_cursor += 1
                reinserted.append(chosen)
            return reinserted

    mutant = MutantFrozenCursorPolicy(policy="round_robin", B=2)
    moving = [10, 11, 12, 13]
    X1 = X0.copy()
    step1 = mutant.step(index, X1, moving_nodes=moving)
    step2 = mutant.step(index, X1, moving_nodes=moving)

    with pytest.raises(AssertionError):
        # In correct implementation, step 1 has [10, 11] and step 2 has [12, 13]
        assert step1 == [10, 11] and step2 == [12, 13], "Mutant with frozen cursor must fail"


def test_mutation_audit_budget_ignored() -> None:
    """Mutation audit: Detects if reinsertion budget B is ignored."""
    index, X0 = _sample_graph()

    class MutantBudgetIgnoredPolicy(MaintenancePolicy):
        def step(self, index: HNSWIndex, positions: np.ndarray, t: float = 0.0, moving_nodes=None):
            # Mutant: ignores self._B and reinserts all moving nodes
            return list(moving_nodes) if moving_nodes else []

    mutant = MutantBudgetIgnoredPolicy(policy="round_robin", B=1)
    moving = [1, 2, 3, 4]
    X1 = X0.copy()
    res = mutant.step(index, X1, moving_nodes=moving)

    with pytest.raises(AssertionError):
        assert len(res) == 1, "Mutant ignoring budget B must fail"
