"""Unit and property tests for repair pruning and ranking primitives."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from khnsw.core.detection.cost import CostMeter
from khnsw.core.repair.prune import prune, rank_pool


def _reference_prune(
    ranked: list[tuple[float, int]],
    base: int,
    cap: int,
    alpha: float,
    skip_if_below_cap: bool,
    vector_of: Any,
) -> tuple[list[tuple[float, int]], int]:
    """Naive reference implementation of alpha-RNG heuristic prune with eval counting."""
    if cap <= 0 or not ranked:
        return [], 0

    if skip_if_below_cap and len(ranked) < cap:
        return list(ranked), 0

    selected: list[tuple[float, int]] = []
    selected_vecs: list[NDArray[np.floating[Any]]] = []
    eval_count = 0

    for d_base_e, e_id in ranked:
        if len(selected) >= cap:
            break

        vec_e = vector_of(e_id)
        reject = False

        for r_vec in selected_vecs:
            eval_count += 1
            diff = r_vec - vec_e
            d_re = float(np.dot(diff, diff))
            if alpha * d_re < d_base_e:
                reject = True
                break

        if not reject:
            selected.append((d_base_e, e_id))
            selected_vecs.append(vec_e)

    return selected, eval_count


# ---------------------------------------------------------------------------
# T1: Hypothesis Oracle Test (>= 200 examples)
# ---------------------------------------------------------------------------


@settings(max_examples=250, deadline=None)
@given(
    d=st.integers(min_value=2, max_value=8),
    n_candidates=st.integers(min_value=0, max_value=25),
    cap=st.integers(min_value=1, max_value=15),
    alpha=st.sampled_from([1.0, 1.1, 1.2]),
    skip_if_below_cap=st.booleans(),
    seed=st.integers(min_value=0, max_value=1000000),
)
def test_prune_hypothesis_oracle(
    d: int,
    n_candidates: int,
    cap: int,
    alpha: float,
    skip_if_below_cap: bool,
    seed: int,
) -> None:
    """T1: Property-based oracle test comparing prune() against reference implementation."""
    rng = np.random.default_rng(seed)

    base = 0
    base_vec = rng.standard_normal(d).astype(np.float64)

    # Generate random candidate vectors
    cand_ids = list(range(1, n_candidates + 1))
    cand_vecs = {cid: rng.standard_normal(d).astype(np.float64) for cid in cand_ids}
    vectors = {base: base_vec, **cand_vecs}

    def vector_of(nid: int) -> NDArray[np.float64]:
        return vectors[nid]

    # Pre-rank candidates
    rank_meter = CostMeter()
    ranked = rank_pool(
        base=base,
        pool=cand_ids,
        ef=None,
        meter=rank_meter,
        vector_of=vector_of,
    )

    prune_meter = CostMeter()
    actual_selected = prune(
        ranked=ranked,
        base=base,
        cap=cap,
        alpha=alpha,
        skip_if_below_cap=skip_if_below_cap,
        meter=prune_meter,
        vector_of=vector_of,
    )

    expected_selected, expected_evals = _reference_prune(
        ranked=ranked,
        base=base,
        cap=cap,
        alpha=alpha,
        skip_if_below_cap=skip_if_below_cap,
        vector_of=vector_of,
    )

    # 1. Output neighbor lists must match exactly
    assert actual_selected == expected_selected

    # 2. Distance evaluation meter counts must match exactly
    assert prune_meter.count == expected_evals

    # 3. If skip_if_below_cap applied, eval count must be exactly 0
    if skip_if_below_cap and len(ranked) < cap:
        assert prune_meter.count == 0


# ---------------------------------------------------------------------------
# T2: Hand-built Golden Tests
# ---------------------------------------------------------------------------


def test_strict_inequality_rejection_boundary() -> None:
    """T2.1: Candidates where alpha * D(r, e) == D(base, e) are ACCEPTED, not rejected.

    Construct exact integer coordinates in R^2:
    base at (0.0, 0.0)
    r (node 1) at (2.0, 0.0) -> D(base, r) = 2^2 + 0^2 = 4.0
    e (node 2) at (1.0, 2.0) -> D(base, e) = 1^2 + 2^2 = 5.0
    D(r, e) = (1 - 2)^2 + (2 - 0)^2 = 1 + 4 = 5.0

    Under alpha = 1.0:
    alpha * D(r, e) = 1.0 * 5.0 = 5.0 == D(base, e).
    Strict inequality alpha * D(r, e) < D(base, e) is FALSE (5.0 < 5.0 is False).
    Therefore, candidate e MUST NOT be rejected, and both r and e are selected.
    """
    vectors = {
        0: np.array([0.0, 0.0], dtype=np.float64),
        1: np.array([2.0, 0.0], dtype=np.float64),
        2: np.array([1.0, 2.0], dtype=np.float64),
    }

    def vector_of(nid: int) -> NDArray[np.float64]:
        return vectors[nid]

    # Pre-rank candidates 1 and 2
    meter = CostMeter()
    ranked = rank_pool(base=0, pool=[1, 2], ef=None, meter=meter, vector_of=vector_of)

    # Node 1 (d=4.0) ranks before Node 2 (d=5.0)
    assert [cid for _, cid in ranked] == [1, 2]

    meter_prune = CostMeter()
    selected = prune(
        ranked=ranked,
        base=0,
        cap=2,
        alpha=1.0,
        skip_if_below_cap=False,
        meter=meter_prune,
        vector_of=vector_of,
    )

    # Because D(1, 2) == D(0, 2), candidate 2 is NOT rejected by candidate 1
    selected_ids = [cid for _, cid in selected]
    assert selected_ids == [1, 2]


def test_alpha_relaxation_difference() -> None:
    """T2.2: Candidate is REJECTED under alpha=1.0 but ACCEPTED under alpha=1.1.

    Points in R^2:
    base (0) at (0.0, 0.0)
    r (1) at (1.0, 0.0) -> D(base, r) = 1.0
    e (2) at (0.525, sqrt(0.774375))
      -> D(base, e) = 0.525^2 + 0.774375 = 0.275625 + 0.774375 = 1.05
      -> D(r, e) = (0.525 - 1.0)^2 + 0.774375 = 0.225625 + 0.774375 = 1.00

    Comparison for candidate e against accepted neighbor r:
    - Under alpha = 1.0:
      1.0 * D(r, e) = 1.00 < 1.05 -> REJECTED!
    - Under alpha = 1.1:
      1.1 * D(r, e) = 1.10 < 1.05 is FALSE -> ACCEPTED!
    """
    y = math.sqrt(0.774375)
    vectors = {
        0: np.array([0.0, 0.0], dtype=np.float64),
        1: np.array([1.0, 0.0], dtype=np.float64),
        2: np.array([0.525, y], dtype=np.float64),
    }

    def vector_of(nid: int) -> NDArray[np.float64]:
        return vectors[nid]

    meter_rank = CostMeter()
    ranked = rank_pool(base=0, pool=[1, 2], ef=None, meter=meter_rank, vector_of=vector_of)

    # Ranked order: node 1 (d=1.0), then node 2 (d=1.05)
    assert [cid for _, cid in ranked] == [1, 2]

    # Prune with alpha=1.0: node 2 is rejected
    m1 = CostMeter()
    sel_1 = prune(
        ranked=ranked,
        base=0,
        cap=2,
        alpha=1.0,
        skip_if_below_cap=False,
        meter=m1,
        vector_of=vector_of,
    )
    assert [cid for _, cid in sel_1] == [1]

    # Prune with alpha=1.1: node 2 is ACCEPTED
    m2 = CostMeter()
    sel_2 = prune(
        ranked=ranked,
        base=0,
        cap=2,
        alpha=1.1,
        skip_if_below_cap=False,
        meter=m2,
        vector_of=vector_of,
    )
    assert [cid for _, cid in sel_2] == [1, 2]


def test_skip_if_below_cap_leaves_certificate_violations() -> None:
    """T2.3: skip_if_below_cap=True returns without pruning, leaving certificate violations.

    Collinear points along x-axis:
    base at 0.0, r (1) at 1.0, e (2) at 2.0.
    D(base, r) = 1.0, D(base, e) = 4.0, D(r, e) = 1.0.
    r violates the empty-ellipse certificate between base and e.

    With cap = 5:
    - skip_if_below_cap=True returns [1, 2] without checking (leaves violating edge).
    - skip_if_below_cap=False checks e against r, rejects e (D(r, e) = 1 < 4), returns [1].
    """
    vectors = {
        0: np.array([0.0], dtype=np.float64),
        1: np.array([1.0], dtype=np.float64),
        2: np.array([2.0], dtype=np.float64),
    }

    def vector_of(nid: int) -> NDArray[np.float64]:
        return vectors[nid]

    ranked = [(1.0, 1), (4.0, 2)]

    # skip_if_below_cap=True: returns both without pruning
    m_skip = CostMeter()
    sel_skip = prune(
        ranked=ranked,
        base=0,
        cap=5,
        alpha=1.0,
        skip_if_below_cap=True,
        meter=m_skip,
        vector_of=vector_of,
    )
    assert [cid for _, cid in sel_skip] == [1, 2]
    assert m_skip.count == 0

    # skip_if_below_cap=False: prunes candidate 2
    m_noskip = CostMeter()
    sel_noskip = prune(
        ranked=ranked,
        base=0,
        cap=5,
        alpha=1.0,
        skip_if_below_cap=False,
        meter=m_noskip,
        vector_of=vector_of,
    )
    assert [cid for _, cid in sel_noskip] == [1]
    assert m_noskip.count == 1


def test_rank_pool_deterministic_tie_breaking() -> None:
    """T2.4: rank_pool breaks equidistant ties by candidate ID descending."""
    # Points at unit circle: all equidistant (d=1.0) to base at (0, 0)
    vectors = {
        0: np.array([0.0, 0.0], dtype=np.float64),
        2: np.array([0.0, 1.0], dtype=np.float64),
        5: np.array([1.0, 0.0], dtype=np.float64),
        8: np.array([-1.0, 0.0], dtype=np.float64),
        1: np.array([0.0, -1.0], dtype=np.float64),
    }

    def vector_of(nid: int) -> NDArray[np.float64]:
        return vectors[nid]

    meter = CostMeter()
    ranked = rank_pool(
        base=0,
        pool=[2, 5, 8, 1],
        ef=None,
        meter=meter,
        vector_of=vector_of,
    )

    # All distances are 1.0; candidates must be sorted by ID descending: [8, 5, 2, 1]
    assert [cid for _, cid in ranked] == [8, 5, 2, 1]
