"""Unit tests and mutation audits for BudgetedPriorityTracker.

Pure Python standard library and NumPy only.
Tests:
- Highest score first
- At most B reinsertions per step
- Score reset after reinsertion
- Cost accounting: cost = stale_checks + reinsertions only
- Mutation check 1: Ordering reversed
- Mutation check 2: Budget ignored
"""

from __future__ import annotations

import numpy as np
import pytest

from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.scheduling.tracker import BudgetedPriorityTracker


def _create_sample_index(
    n: int = 20, d: int = 4, m: int = 4, seed: int = 42
) -> tuple[HNSWIndex, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    index = HNSWIndex(dim=d, m=m, seed=seed)
    index.add(np.arange(n, dtype=np.int64), X)
    return index, X


def test_highest_score_first() -> None:
    """Nodes with the highest score s_i = ||x_i - p_i|| / r_i are reinserted first."""
    index, X0 = _create_sample_index()
    tracker = BudgetedPriorityTracker(index, budget=3)
    tracker.register_all(X0)

    # Move nodes 1, 2, 3 with different displacements
    X1 = X0.copy()
    r1 = tracker.radius(1)
    r2 = tracker.radius(2)
    r3 = tracker.radius(3)

    # Set displacements so that score(2) > score(3) > score(1)
    X1[1] += np.float32([1.0 * r1, 0, 0, 0])  # score ~ 1.0
    X1[2] += np.float32([3.0 * r2, 0, 0, 0])  # score ~ 3.0
    X1[3] += np.float32([2.0 * r3, 0, 0, 0])  # score ~ 2.0

    moving_nodes = [1, 2, 3]
    res = tracker.step(X1, moving_nodes=moving_nodes)

    assert res["reinserted_nodes"] == [2, 3, 1], "Must select highest score first"


def test_at_most_b_per_step() -> None:
    """Number of reinsertions per step does not exceed budget B."""
    index, X0 = _create_sample_index()
    budget = 2
    tracker = BudgetedPriorityTracker(index, budget=budget)
    tracker.register_all(X0)

    X1 = X0.copy()
    for nid in range(1, 10):
        X1[nid] += np.float32([2.0, 0, 0, 0])

    moving_nodes = list(range(1, 10))
    res = tracker.step(X1, moving_nodes=moving_nodes)

    assert len(res["reinserted_nodes"]) == budget
    assert res["reinsert_count"] == budget
    assert res["reinsert_count"] <= budget


def test_score_reset_after_reinsertion() -> None:
    """Reinserting a node resets its anchor p_i to current position, resetting score to 0."""
    index, X0 = _create_sample_index()
    tracker = BudgetedPriorityTracker(index, budget=1)
    tracker.register_all(X0)

    X1 = X0.copy()
    X1[5] += np.float32([2.0, 0, 0, 0])
    X1[6] += np.float32([1.0, 0, 0, 0])

    # Step 1: node 5 has highest displacement -> reinserted
    res1 = tracker.step(X1, moving_nodes=[5, 6])
    assert res1["reinserted_nodes"] == [5]

    # Without moving node 5 again, its displacement from anchor is now 0
    score_after = tracker.compute_score(5, X1[5])
    assert np.isclose(score_after, 0.0, atol=1e-5), "Score must reset to 0 after reinsertion"

    # In Step 2, node 6 should now have higher score than node 5
    res2 = tracker.step(X1, moving_nodes=[5, 6])
    assert res2["reinserted_nodes"] == [6]


def test_cost_accounting() -> None:
    """Cost must equal stale checks + reinsertions only."""
    index, X0 = _create_sample_index()
    tracker = BudgetedPriorityTracker(index, budget=2)
    tracker.register_all(X0)

    X1 = X0.copy()
    moving = [1, 2, 3, 4, 5]
    for nid in moving:
        X1[nid] += np.float32([1.5, 0, 0, 0])

    res = tracker.step(X1, moving_nodes=moving)

    assert res["stale_check_cost"] == len(moving), "Exactly 1 stale check per moving node"
    assert res["reinsert_cost"] > 0
    assert res["total_cost"] == res["stale_check_cost"] + res["reinsert_cost"]


def test_mutation_audit_ordering_reversed() -> None:
    """Mutation audit: Detects if score sorting order is accidentally reversed (lowest first)."""
    index, X0 = _create_sample_index()

    class MutantReversedTracker(BudgetedPriorityTracker):
        def step(self, current_positions: np.ndarray, moving_nodes: list[int]) -> dict:
            pos_mat = np.asarray(current_positions, dtype=np.float32)
            scores = [(self.compute_score(nid, pos_mat[nid]), nid) for nid in moving_nodes]
            # Mutant: sorting ascending (lowest first)
            scores.sort(key=lambda item: item[0], reverse=False)
            selected = [nid for s, nid in scores if s > self._min_score][: self._budget]
            return {"reinserted_nodes": selected}

    mutant = MutantReversedTracker(index, budget=1)
    mutant.register_all(X0)

    X1 = X0.copy()
    r1 = mutant.radius(1)
    r2 = mutant.radius(2)
    X1[1] += np.float32([1.0 * r1, 0, 0, 0])  # lower score
    X1[2] += np.float32([5.0 * r2, 0, 0, 0])  # higher score

    res = mutant.step(X1, moving_nodes=[1, 2])
    with pytest.raises(AssertionError):
        assert res["reinserted_nodes"] == [2], "Mutant with reversed order must be caught"


def test_mutation_audit_budget_ignored() -> None:
    """Mutation audit: Detects if reinsertion budget B is ignored (all eligible reinserted)."""
    index, X0 = _create_sample_index()

    class MutantBudgetIgnoredTracker(BudgetedPriorityTracker):
        def step(self, current_positions: np.ndarray, moving_nodes: list[int]) -> dict:
            pos_mat = np.asarray(current_positions, dtype=np.float32)
            scores = [(self.compute_score(nid, pos_mat[nid]), nid) for nid in moving_nodes]
            scores.sort(key=lambda item: item[0], reverse=True)
            # Mutant: ignores budget B and reinserts all eligible
            selected = [nid for s, nid in scores if s > self._min_score]
            return {"reinserted_nodes": selected}

    mutant = MutantBudgetIgnoredTracker(index, budget=1)
    mutant.register_all(X0)

    X1 = X0.copy()
    for nid in [1, 2, 3]:
        X1[nid] += np.float32([2.0, 0, 0, 0])

    res = mutant.step(X1, moving_nodes=[1, 2, 3])
    with pytest.raises(AssertionError):
        assert len(res["reinserted_nodes"]) == 1, "Mutant ignoring budget must be caught"
