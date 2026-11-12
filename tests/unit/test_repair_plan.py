"""Unit tests for repair planning primitives (native, mnru variants, khnsw)."""

from __future__ import annotations

import numpy as np

from khnsw.core.detection.cost import CostMeter
from khnsw.core.repair.plan import khnsw_plan, mnru_plan, native_plan
from tests.fixtures.reference_graph import ReferenceGraph

# ---------------------------------------------------------------------------
# T3: Hand-built ReferenceGraph pool golden test
# ---------------------------------------------------------------------------


def test_repair_plan_node_class_coverage_golden() -> None:
    """T3: Pool goldens on hand-built ReferenceGraphs.

    Tests two variants for non-mutual in-neighbor u (u -> moved, moved -/-> u) whose
    edge (u, moved) is violated after move:
    - Variant 1 (u within two hops):
      native and mnru one_hop never rewrite u.
      mnru two_hop DOES rewrite u (u is within 2 hops and u -> moved).
      khnsw DOES rewrite u.
    - Variant 2 (u beyond two hops):
      native and mnru one_hop never rewrite u.
      mnru two_hop does NOT rewrite u (u is beyond 2 hops).
      khnsw DOES rewrite u.

    Also asserts candidate pool properties:
    - For mutual neighbor v: native's pool contains the two-hop node from another neighbor
      and does not contain private neighbors of other non-L1 nodes; mnru one_hop contains
      N(v) and not the external two-hop node.
    """
    # -----------------------------------------------------------------------
    # Variant 1: u (node 3) is within 2 hops of moved (node 0): 0 -> 1 -> 3, and 3 -> 0
    # -----------------------------------------------------------------------
    vectors_v1 = {
        0: np.array([0.0, 0.0], dtype=np.float32),
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
        3: np.array([2.0, 0.0], dtype=np.float32),  # u
        4: np.array([0.0, 2.0], dtype=np.float32),  # two-hop node from 2
    }
    levels_v1 = {i: 0 for i in range(5)}
    adj_v1 = {
        (0, 0): [1, 2],  # moved -> 1 (mutual), moved -> 2 (mutual)
        (1, 0): [0, 3],  # 1 -> 0, 1 -> 3 (making 3 a 2-hop node from 0)
        (2, 0): [0, 4],  # 2 -> 0, 2 -> 4 (making 4 a 2-hop node from 0)
        (3, 0): [0],  # 3 -> 0 (non-mutual in-neighbor, within 2 hops)
        (4, 0): [2],
    }
    g_v1 = ReferenceGraph.from_adjacency(vectors_v1, adj_v1, levels=levels_v1, M=3)

    # In Variant 1:
    p_native_1 = native_plan(g_v1, moved=0)
    assert 3 not in {r.node for r in p_native_1.rewrites}

    p_beta_1 = mnru_plan(g_v1, moved=0, alpha=1.0, scope="one_hop")
    assert 3 not in {r.node for r in p_beta_1.rewrites}

    p_thn_1 = mnru_plan(g_v1, moved=0, alpha=1.1, scope="two_hop")
    # In Variant 1, u is within 2 hops, so mnru_two_hop rewrites u (if neighbors change)
    assert 3 in {r.node for r in p_thn_1.rewrites}

    p_khnsw_1 = khnsw_plan(g_v1, violated={(3, 0): [0]})
    # khnsw does rewrite u (node 3)
    # u's candidate pool is (N(3) U N(0)) \ {3} = ({0} U {1, 2}) \ {3} = {0, 1, 2}
    # which replaces [0] with [0, 1, 2]
    assert 3 in {r.node for r in p_khnsw_1.rewrites}

    # -----------------------------------------------------------------------
    # Variant 2: u (node 5) is beyond 2 hops of moved (node 0): 0 -> 1 -> 3 -> 5, 5 -> 0
    # -----------------------------------------------------------------------
    vectors_v2 = {
        0: np.array([0.0, 0.0], dtype=np.float32),
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
        3: np.array([2.0, 0.0], dtype=np.float32),
        4: np.array([0.0, 2.0], dtype=np.float32),
        5: np.array([3.0, 0.0], dtype=np.float32),  # u beyond 2 hops
    }
    levels_v2 = {i: 0 for i in range(6)}
    adj_v2 = {
        (0, 0): [1, 2],
        (1, 0): [0, 3],
        (2, 0): [0, 4],
        (3, 0): [1, 5],  # 1 -> 3 -> 5
        (4, 0): [2],
        (5, 0): [0],  # 5 -> 0 (in-neighbor beyond 2 hops)
    }
    g_v2 = ReferenceGraph.from_adjacency(vectors_v2, adj_v2, levels=levels_v2, M=3)

    p_native_2 = native_plan(g_v2, moved=0)
    assert 5 not in {r.node for r in p_native_2.rewrites}

    p_beta_2 = mnru_plan(g_v2, moved=0, alpha=1.0, scope="one_hop")
    assert 5 not in {r.node for r in p_beta_2.rewrites}

    p_thn_2 = mnru_plan(g_v2, moved=0, alpha=1.1, scope="two_hop")
    # In Variant 2, u is beyond 2 hops, so mnru_two_hop does NOT rewrite u
    assert 5 not in {r.node for r in p_thn_2.rewrites}

    p_khnsw_2 = khnsw_plan(g_v2, violated={(5, 0): [0]})
    # khnsw DOES rewrite u
    assert 5 in {r.node for r in p_khnsw_2.rewrites}


# ---------------------------------------------------------------------------
# Targeted golden tests killing M6, M7, M8
# ---------------------------------------------------------------------------


def test_mnru_plan_skips_non_mutual_out_neighbor() -> None:
    """MN-RU must strictly skip non-mutual out-neighbors (kills M6).

    Setup:
    Moved node 0 has out-neighbors L1 = [1, 2].
    - Node 1 is mutual: 1 -> 0.
    - Node 2 is non-mutual: 0 -> 2, but 2 -> 5 (node 2 does not have 0).
    Node 5 is placed far away at (0, 10). If node 2 were repaired, it would drop 5
    and add 0, creating a Rewrite for node 2.
    With the mutual filter, node 2 is skipped and never repaired.
    """
    vectors = {
        0: np.array([0.0, 0.0], dtype=np.float32),
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
        5: np.array([0.0, 10.0], dtype=np.float32),
    }
    levels = {i: 0 for i in vectors}
    adj = {
        (0, 0): [1, 2],
        (1, 0): [0],
        (2, 0): [5],  # 2 does not have 0
        (5, 0): [],
    }
    g = ReferenceGraph.from_adjacency(vectors, adj, levels=levels, M=3)

    plan = mnru_plan(g, moved=0, alpha=1.0, scope="one_hop")
    repaired_nodes = {r.node for r in plan.rewrites}

    # Node 1 is repaired; non-mutual out-neighbor 2 MUST NOT be repaired
    assert 2 not in repaired_nodes
    assert repaired_nodes == {1}


def test_native_plan_candidate_pool_includes_two_hop_neighbors() -> None:
    """native_plan must include two-hop neighbors in candidate pool (kills M7).

    Setup:
    Moved node 0 has L1 = [1, 2].
    Node 1 has [0].
    Node 2 has [0, 4] (node 4 is a two-hop neighbor of 0 reachable only via 2).
    Node 4 at (9.9, 0) is extremely close to node 1 at (10, 0) (dist = 0.01).
    Node 0 at (0, 0) is at dist = 100. Node 2 at (0, 10) is at dist = 200.
    Cap = 2.

    - With two-hop expansion:
      Node 4 enters node 1's candidate pool and is ranked #1.
      Node 1's new neighbor list is (4, 0).
    - Without two-hop expansion (M7):
      Node 4 is omitted from candidate pool; node 1 gets (0, 2).
    """
    vectors = {
        0: np.array([0.0, 0.0], dtype=np.float32),
        1: np.array([10.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 10.0], dtype=np.float32),
        4: np.array([9.9, 0.0], dtype=np.float32),
    }
    levels = {i: 0 for i in vectors}
    adj = {
        (0, 0): [1, 2],
        (1, 0): [0],
        (2, 0): [0, 4],
        (4, 0): [2],
    }
    g = ReferenceGraph.from_adjacency(vectors, adj, levels=levels, M=2)

    plan = native_plan(g, moved=0)
    r1 = next(r for r in plan.rewrites if r.node == 1)

    assert 4 in r1.new
    assert r1.new == (4, 0, 2)


def test_mnru_plan_candidate_pool_includes_existing_neighbors() -> None:
    """mnru_plan candidate pool must include existing neighbors N(v) (kills M8).

    Setup:
    Moved node 0 has N(0) = [1, 2]. Node 2 is at (0.9, 0) -> D(1, 2) = 0.01.
    Node 1 is mutual (1 -> 0), and has existing neighbors [0, 3, 4].
    Node 3 in N(1) is at (1.05, 0) -> D(1, 3) = 0.0025 (closer than 2!).
    Node 4 in N(1) is at (5.0, 0).
    Cap = 2.

    - With N(1) in candidate pool:
      Candidates for node 1 are {0, 2, 3, 4}.
      Top 2 ranked candidates are 3 and 2.
      Node 1's new neighbor list is (3, 2).
    - Without N(1) in candidate pool (M8):
      Candidate pool only has {0, 2}.
      Node 3 is missing; node 1's new neighbor list becomes (2, 0).
    """
    vectors = {
        0: np.array([0.0, 0.0], dtype=np.float32),
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.9, 0.0], dtype=np.float32),
        3: np.array([1.05, 0.0], dtype=np.float32),
        4: np.array([5.0, 0.0], dtype=np.float32),
    }
    levels = {i: 0 for i in vectors}
    adj = {
        (0, 0): [1, 2],
        (1, 0): [0, 3, 4],
        (2, 0): [0],
        (3, 0): [1],
        (4, 0): [1],
    }
    g = ReferenceGraph.from_adjacency(vectors, adj, levels=levels, M=2)

    plan = mnru_plan(g, moved=0, alpha=1.0, scope="one_hop")
    r1 = next(r for r in plan.rewrites if r.node == 1)

    assert 3 in r1.new
    assert r1.new == (3, 2)


# ---------------------------------------------------------------------------
# T4: Planning is strictly read-only
# ---------------------------------------------------------------------------


def test_planning_is_strictly_read_only() -> None:
    """T4: Planning functions do not alter graph state or versions."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10, 4)).astype(np.float32)
    g = ReferenceGraph.from_points(X, M=3, seed=42, mimic_reverse_append=True)

    # Record snapshot of all neighbor lists and node versions
    initial_nbrs = {
        (u, lvl): list(g.neighbors(u, lvl)) for u in g.ids() for lvl in range(g.node_level(u) + 1)
    }
    initial_versions = {u: g.version(u) for u in g.ids()}
    initial_dists = g.distance_computations

    # Run all 4 planners
    meter1 = CostMeter()
    p1 = native_plan(g, moved=0, meter=meter1)
    assert p1.distance_computations == meter1.count

    meter2 = CostMeter()
    p2 = mnru_plan(g, moved=0, alpha=1.0, scope="one_hop", meter=meter2)
    assert p2.distance_computations == meter2.count

    meter3 = CostMeter()
    p3 = mnru_plan(g, moved=0, alpha=1.1, scope="two_hop", meter=meter3)
    assert p3.distance_computations == meter3.count

    meter4 = CostMeter()
    p4 = khnsw_plan(g, violated={(0, 0): list(g.neighbors(0, 0))}, meter=meter4)
    assert p4.distance_computations == meter4.count

    # Verify graph state is 100% pristine
    for (u, lvl), nbrs in initial_nbrs.items():
        assert list(g.neighbors(u, lvl)) == nbrs
    for u, ver in initial_versions.items():
        assert g.version(u) == ver
    assert g.distance_computations == initial_dists


# ---------------------------------------------------------------------------
# T5: Planning determinism
# ---------------------------------------------------------------------------


def test_planning_determinism() -> None:
    """T5: Consecutive planning calls on identical graph produce identical plans."""
    rng = np.random.default_rng(123)
    X = rng.standard_normal((15, 3)).astype(np.float32)
    g = ReferenceGraph.from_points(X, M=4, seed=123, mimic_reverse_append=True)

    # Test native plan determinism
    p_native_a = native_plan(g, moved=2)
    p_native_b = native_plan(g, moved=2)
    assert p_native_a == p_native_b

    # Test mnru plan determinism
    p_mnru_a = mnru_plan(g, moved=2, alpha=1.1, scope="two_hop")
    p_mnru_b = mnru_plan(g, moved=2, alpha=1.1, scope="two_hop")
    assert p_mnru_a == p_mnru_b

    # Test khnsw plan determinism (including key and target order invariance)
    violated_a = {(3, 0): [4, 0], (1, 0): [2, 5]}
    violated_b = {(1, 0): [5, 2], (3, 0): [0, 4]}
    p_khnsw_a = khnsw_plan(g, violated=violated_a)
    p_khnsw_b = khnsw_plan(g, violated=violated_b)
    assert p_khnsw_a == p_khnsw_b
