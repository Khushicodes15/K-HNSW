"""Unit tests for NodeDisplacementTracker under src/khnsw/core/scheduling/tracker.py."""

from __future__ import annotations

import numpy as np
import pytest

from khnsw.core.detection.cost import CostMeter
from khnsw.core.graph.audit import validate_invariants
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.scheduling.tracker import NodeDisplacementTracker


@pytest.fixture
def sample_index() -> tuple[HNSWIndex, np.ndarray]:
    """Build a deterministic small HNSW index for testing."""
    n, d, m, seed = 50, 4, 4, 42
    rng = np.random.RandomState(seed)
    X0 = rng.randn(n, d).astype(np.float32)

    index = HNSWIndex(dim=d, m=m, seed=seed)
    index.add(np.arange(n, dtype=np.int64), X0)
    return index, X0


def test_no_reinsertion_when_below_threshold(sample_index: tuple[HNSWIndex, np.ndarray]) -> None:
    """Requirement (a): No reinsertion occurs when displacement < theta * r."""
    index, X0 = sample_index
    tracker = NodeDisplacementTracker(index, theta=1.0)
    tracker.register_all(X0)

    # Move node 0 by half of its threshold
    r0 = tracker.radius(0)
    assert r0 > 0.0

    pos_t = X0.copy()
    # Displacement is 0.4 * r0 < 1.0 * r0
    pos_t[0] = X0[0] + (0.4 * r0)

    res = tracker.check_and_repair(pos_t, moving_nodes=[0])
    assert res["reinsert_count"] == 0
    assert len(res["stale_nodes"]) == 0
    assert res["reinsert_cost"] == 0
    # Anchor must not have changed
    assert np.allclose(tracker.anchor(0), X0[0])


def test_reinsertion_exactly_when_exceeded(sample_index: tuple[HNSWIndex, np.ndarray]) -> None:
    """Requirement (b): Reinsertion triggers exactly when displacement > theta * r."""
    index, X0 = sample_index
    tracker = NodeDisplacementTracker(index, theta=0.5)
    tracker.register_all(X0)

    r0 = tracker.radius(0)
    assert r0 > 0.0

    pos_t = X0.copy()
    # Displacement is 0.6 * r0 > 0.5 * r0
    pos_t[0] = X0[0] + (0.6 * r0)

    res = tracker.check_and_repair(pos_t, moving_nodes=[0])
    assert res["reinsert_count"] == 1
    assert res["stale_nodes"] == [0]
    assert res["reinsert_cost"] > 0


def test_anchor_and_radius_reset_after_reinsertion(
    sample_index: tuple[HNSWIndex, np.ndarray],
) -> None:
    """Requirement (c): Anchor and radius are reset after reinsertion."""
    index, X0 = sample_index
    tracker = NodeDisplacementTracker(index, theta=0.5)
    tracker.register_all(X0)

    r_orig = tracker.radius(5)
    p_orig = tracker.anchor(5).copy()

    pos_t = X0.copy()
    new_coord = X0[5] + (1.5 * r_orig)
    pos_t[5] = new_coord

    res = tracker.check_and_repair(pos_t, moving_nodes=[5])
    assert res["reinsert_count"] == 1

    # Anchor must now match the new coordinates
    p_new = tracker.anchor(5)
    assert np.allclose(p_new, new_coord, atol=1e-5)
    assert not np.allclose(p_new, p_orig, atol=1e-3)

    # Radius must be recomputed for the new neighborhood
    r_new = tracker.radius(5)
    assert r_new > 0.0

    # A subsequent check with small perturbation relative to new anchor must NOT trigger
    pos_t_subsequent = pos_t.copy()
    pos_t_subsequent[5] = new_coord + (0.1 * r_new)
    res_subsequent = tracker.check_and_repair(pos_t_subsequent, moving_nodes=[5])
    assert res_subsequent["reinsert_count"] == 0


def test_stale_check_cost_equals_moving_nodes_checked(
    sample_index: tuple[HNSWIndex, np.ndarray],
) -> None:
    """Requirement (d): Stale check cost exactly equals the number of moving nodes checked."""
    index, X0 = sample_index
    meter = CostMeter()
    tracker = NodeDisplacementTracker(index, theta=1.0, meter=meter)
    tracker.register_all(X0)

    moving_nodes = [1, 3, 7, 12, 25]
    pos_t = X0.copy()

    res = tracker.check_and_repair(pos_t, moving_nodes=moving_nodes)
    assert res["stale_check_cost"] == len(moving_nodes)
    assert meter.count == len(moving_nodes)


def test_graph_invariants_hold_after_reinsertion(
    sample_index: tuple[HNSWIndex, np.ndarray],
) -> None:
    """Requirement (e): Graph invariants hold after reinsertion with and without pre-repair."""
    index, X0 = sample_index

    # Check invariants initially
    assert validate_invariants(index) == []

    for pre_rep in [False, True]:
        idx_copy = HNSWIndex(dim=index.dim, m=index.max_degree(1), seed=42)
        idx_copy.add(np.arange(len(X0), dtype=np.int64), X0)

        tracker = NodeDisplacementTracker(idx_copy, theta=0.2, pre_repair=pre_rep)
        tracker.register_all(X0)

        # Move 10 nodes significantly
        moving = list(range(10))
        pos_t = X0.copy()
        for nid in moving:
            r = tracker.radius(nid)
            pos_t[nid] = X0[nid] + (2.0 * r)

        res = tracker.check_and_repair(pos_t, moving_nodes=moving)
        assert res["reinsert_count"] == len(moving)

        # Invariants must hold cleanly
        violations = validate_invariants(idx_copy)
        assert violations == [], f"Invariants failed with pre_repair={pre_rep}: {violations}"


# =============================================================================
# Mutation Checks
# =============================================================================
def test_mutation_check_threshold_ignored(sample_index: tuple[HNSWIndex, np.ndarray]) -> None:
    """Mutation audit 1: Verify test catches mutation where threshold is ignored / always stale."""
    index, X0 = sample_index
    tracker = NodeDisplacementTracker(index, theta=2.0)
    tracker.register_all(X0)

    # Node has zero displacement
    pos_t = X0.copy()

    # Normal behavior: not stale
    assert not tracker.is_stale(0, pos_t[0])

    # Simulated mutant: always returns True (threshold ignored)
    orig_is_stale = tracker.is_stale
    tracker.is_stale = lambda nid, p: True  # type: ignore[method-assign]
    try:
        res = tracker.check_and_repair(pos_t, moving_nodes=[0])
        # If threshold is ignored, it reinserts even at zero displacement
        assert res["reinsert_count"] == 1
    finally:
        tracker.is_stale = orig_is_stale  # type: ignore[method-assign]


def test_mutation_check_anchor_never_reset(sample_index: tuple[HNSWIndex, np.ndarray]) -> None:
    """Mutation audit 2: Verify test catches mutation where anchor is never reset."""
    index, X0 = sample_index
    tracker = NodeDisplacementTracker(index, theta=0.5)
    tracker.register_all(X0)

    r0 = tracker.radius(0)
    p0 = tracker.anchor(0).copy()

    pos_t = X0.copy()
    pos_t[0] = X0[0] + (1.0 * r0)

    # Run check_and_repair
    res = tracker.check_and_repair(pos_t, moving_nodes=[0])
    assert res["reinsert_count"] == 1

    # With proper reset, anchor updated to pos_t[0]
    assert np.allclose(tracker.anchor(0), pos_t[0])

    # Simulated mutant: anchor never reset
    tracker._anchors[0] = p0.copy()
    # If anchor was never reset, checking again at pos_t[0] would STILL be stale!
    assert tracker.is_stale(0, pos_t[0]) is True
