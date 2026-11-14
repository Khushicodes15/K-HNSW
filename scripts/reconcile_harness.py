"""Unified benchmark harness for Phase 5h & 5i reconciliation (Items A-F).

Runs on HNSWIndex (n=500, M=8, d=8) under 10% kinetic drift at 10x and 30x speeds
(T=0.30, 10 steps, seeds 42, 43, 44).
Unified harness specification:
- Drift: LinearDrift(X0, speed=speed), 50 drifting nodes chosen via rng.choice.
- Ground truth: Brute-force Euclidean top-10 at t=0.30.
- Queries: 50 drifting queries (all drifting nodes at t=0.30) and 100 static queries.
- Cost: Full CostMeter counting detector evaluations
  + rebuild search/planning/store/reverse evaluations.
- Arms:
  - no_repair
  - ours_after (ef=16 in rebuild)
  - arm3_rebuild_ef64 (rebuild with ef=64)
  - fresh_insert_reference (drifting nodes built at final positions into fresh static index)
  - arm_inplace_reinsert (in-place delete from no_repair graph + add once via index add)
  - arm_inplace_reinsert_heuristic (in-place delete with static in-nbr heuristic repair + add once)
  - arm4_exact_top_m (exact top-M candidates, reporting cost with and without brute-force search)
  - arm5_oracle_failures (oracle reinsert of confirmed failure drifting nodes, fixed levels)
  - oracle_control_rebuilt (reinsert exactly the nodes our loop rebuilt each step, fixed levels)
  - oracle_control_all_drift (reinsert all 50 drifting nodes every step, fixed levels)
- Static search path tracing (Item D): fraction of static queries visiting >=1 drifting node,
  and mean drifting nodes visited.
- In-degree analysis (Item E): level-0 in-degree and in-degree from true top-10 neighbors.
- Static node edge audit (Phase 5i Item C): mean out-degree of static nodes, fraction of out-edges
  pointing to drifting nodes, and fraction of those that are stale.
- Headroom sweep (Phase 5i Item D): no_repair vs fresh_insert_ref across drift fractions
  10%, 30%, 50% and d=32 at 30x.
"""

from __future__ import annotations

import argparse
import copy
import heapq
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

import khnsw.core.repair.loop as loop_mod
from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.events.schema import ListSink
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.repair.engine import KHNSWRepairEngine
from khnsw.core.repair.loop import prune, rank_pool
from khnsw.core.repair.plan import Rewrite
from khnsw.core.scheduling.event_heap import KineticEventHeap
from khnsw.core.search.heuristic import select_neighbors_heuristic
from khnsw.core.search.search_layer import search_layer


def settle_hnsw(index: HNSWIndex, store: CertificateStore) -> None:
    """Run native_plan repairs until all violated edges are resolved."""
    from khnsw.core.repair.plan import native_plan

    max_rounds = 50
    meter_p = CostMeter()
    meter_s = CostMeter()
    for _ in range(max_rounds):
        violated = [
            edge
            for edge, status in store._edge_status.items()
            if status == EdgeStatus.VIOLATED.value
        ]
        if not violated:
            break
        plan = native_plan(index, violated, meter_p)
        for rewrite in plan.rewrites:
            store.set_neighbors(index, rewrite.node, rewrite.level, rewrite.new, meter_s)


def brute_force_knn(
    vectors: np.ndarray, queries: np.ndarray, k: int = 10, exclude_self: bool = False
) -> list[set[int]]:
    """Compute ground-truth k-NN via brute force squared Euclidean distances."""
    gt: list[set[int]] = []
    for i, q in enumerate(queries):
        dists = np.sum((vectors - q) ** 2, axis=1)
        order = np.argsort(dists)
        if exclude_self:
            selected = [int(idx) for idx in order if int(idx) != i][:k]
        else:
            selected = [int(idx) for idx in order[:k]]
        gt.append(set(selected))
    return gt


def evaluate_recall(
    index: HNSWIndex, queries: np.ndarray, ground_truth: list[set[int]], k: int = 10, ef: int = 16
) -> float:
    """Compute mean recall@k against ground truth sets."""
    recalls: list[float] = []
    for q, gt_set in zip(queries, ground_truth, strict=False):
        ids, _ = index.search(q, k=k, ef=ef)
        retrieved = {int(x) for x in ids[:k]}
        overlap = len(retrieved & gt_set)
        denom = min(k, len(gt_set))
        recalls.append(overlap / float(denom) if denom > 0 else 1.0)
    return float(np.mean(recalls))


def trace_search_path(index: HNSWIndex, query: np.ndarray, ef: int = 16) -> set[int]:
    """Trace all nodes visited during HNSW search."""
    visited: set[int] = set()
    ep = index.entry_point()
    if ep is None:
        return visited

    curr_ep = ep
    visited.add(curr_ep)
    top_lvl = index.max_level()

    # 1. Greedy descent from top level to 1
    for lc in range(top_lvl, 0, -1):
        for nb in index.neighbors(curr_ep, lc):
            visited.add(nb)
        res = search_layer(index._store, query, [curr_ep], 1, lc)
        if res:
            first = res[0]
            nid = int(first.node_id if hasattr(first, "node_id") else first)
            curr_ep = nid
            visited.add(nid)

    # 2. Level 0 beam search exploration
    v_layer0: set[int] = set()
    ep_vec = index.vector(curr_ep)
    d_ep = float(np.sum((ep_vec - query) ** 2))
    cand: list[tuple[float, int]] = [(d_ep, curr_ep)]
    w: list[tuple[float, int]] = [(-d_ep, curr_ep)]
    v_layer0.add(curr_ep)

    while cand:
        d_c, c = heapq.heappop(cand)
        if d_c > -w[0][0]:
            break
        for nb in index.neighbors(c, 0):
            if nb not in v_layer0:
                v_layer0.add(nb)
                d_nb = float(np.sum((index.vector(nb) - query) ** 2))
                if d_nb < -w[0][0] or len(w) < ef:
                    heapq.heappush(cand, (d_nb, nb))
                    heapq.heappush(w, (-d_nb, nb))
                    if len(w) > ef:
                        heapq.heappop(w)

    visited.update(v_layer0)
    return visited


def run_headroom(seed: int) -> list[dict[str, Any]]:
    """Evaluate headroom between no_repair and fresh_insert_ref across drift regimes."""
    headroom_results: list[dict[str, Any]] = []
    n = 500
    m = 8
    speed = 30.0
    t_final = 0.30

    regimes = [
        {"dim": 8, "drift_frac": 0.10, "label": "d8_frac10"},
        {"dim": 8, "drift_frac": 0.30, "label": "d8_frac30"},
        {"dim": 8, "drift_frac": 0.50, "label": "d8_frac50"},
        {"dim": 32, "drift_frac": 0.10, "label": "d32_frac10"},
    ]

    for reg in regimes:
        d = int(reg["dim"])
        frac = float(reg["drift_frac"])
        rng = np.random.default_rng(seed)
        X0 = rng.standard_normal((n, d)).astype(np.float32)

        base_index = HNSWIndex(dim=d, m=m, seed=seed)
        base_index.add(np.arange(n, dtype=np.int64), X0)
        base_store = CertificateStore()
        base_store.build(base_index, CostMeter())
        settle_hnsw(base_index, base_store)

        n_drift = int(frac * n)
        drifting_nodes = sorted(rng.choice(n, size=n_drift, replace=False).tolist())
        static_nodes = [i for i in range(n) if i not in set(drifting_nodes)]

        drift = LinearDrift(X0, seed=seed, speed=speed)
        drift._velocities[static_nodes] = 0.0
        pos_t_final = drift.positions(t_final).astype(np.float32)

        q_drift = pos_t_final[drifting_nodes]
        exact_drift = brute_force_knn(pos_t_final, q_drift, k=10)

        # 1. no_repair
        idx_no_rep = copy.deepcopy(base_index)
        for nid in range(n):
            idx_no_rep._store.set_vector(nid, pos_t_final[nid])
        rec16_no_rep = evaluate_recall(idx_no_rep, q_drift, exact_drift, k=10, ef=16)
        rec32_no_rep = evaluate_recall(idx_no_rep, q_drift, exact_drift, k=10, ef=32)

        # 2. fresh_insert_ref
        idx_fresh = HNSWIndex(dim=d, m=m, seed=seed)
        static_arr = np.array(static_nodes, dtype=np.int64)
        idx_fresh.add(static_arr, pos_t_final[static_arr])
        drift_arr = np.array(drifting_nodes, dtype=np.int64)
        idx_fresh.add(drift_arr, pos_t_final[drift_arr])
        rec16_fresh = evaluate_recall(idx_fresh, q_drift, exact_drift, k=10, ef=16)
        rec32_fresh = evaluate_recall(idx_fresh, q_drift, exact_drift, k=10, ef=32)

        headroom_results.append(
            {
                "seed": seed,
                "label": reg["label"],
                "dim": d,
                "drift_frac": frac,
                "no_repair_rec16": rec16_no_rep,
                "no_repair_rec32": rec32_no_rep,
                "fresh_insert_rec16": rec16_fresh,
                "fresh_insert_rec32": rec32_fresh,
                "headroom_rec16": rec16_fresh - rec16_no_rep,
                "headroom_rec32": rec32_fresh - rec32_no_rep,
            }
        )

    return headroom_results


def run_seed(seed: int) -> dict[str, Any]:
    n = 500
    m = 8
    d = 8
    t_final = 0.30
    steps = 10
    dt = t_final / steps
    speeds = [10.0, 30.0]

    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n, d)).astype(np.float32)

    base_index = HNSWIndex(dim=d, m=m, seed=seed)
    base_index.add(np.arange(n, dtype=np.int64), X0)
    base_store = CertificateStore()
    base_store.build(base_index, CostMeter())
    settle_hnsw(base_index, base_store)

    drifting_nodes = sorted(rng.choice(n, size=int(0.10 * n), replace=False).tolist())
    static_nodes = [i for i in range(n) if i not in drifting_nodes]
    drifting_set = set(drifting_nodes)
    static_set = set(static_nodes)

    rng_s = np.random.default_rng(seed + 500)
    eval_static_nodes = sorted(rng_s.choice(static_nodes, size=100, replace=False).tolist())

    results: dict[str, Any] = {
        "seed": seed,
        "runs": [],
        "static_tracing": {},
        "in_degree": {},
        "static_edge_audit": {},
        "headroom": run_headroom(seed),
    }

    def make_hnsw_sync(index: HNSWIndex) -> Callable[[int, np.ndarray], None]:
        def _sync(node_id: int, vec: np.ndarray) -> None:
            index._store.set_vector(node_id, np.asarray(vec, dtype=np.float32))

        return _sync

    for speed in speeds:
        drift = LinearDrift(X0, seed=seed, speed=speed)
        drift._velocities[static_nodes] = 0.0

        def make_traj(
            x_mat: np.ndarray = X0, v_mat: np.ndarray = drift._velocities
        ) -> Callable[[int], tuple[np.ndarray, np.ndarray]]:
            def _traj(node_id: int) -> tuple[np.ndarray, np.ndarray]:
                return x_mat[node_id].astype(np.float64), v_mat[node_id]

            return _traj

        pos_t_final = drift.positions(t_final).astype(np.float32)
        q_drift = pos_t_final[drifting_nodes]
        exact_drift = brute_force_knn(pos_t_final, q_drift, k=10)

        q_static = pos_t_final[eval_static_nodes]
        exact_static = brute_force_knn(pos_t_final, q_static, k=10)

        # Tracing on no_repair for Item D
        idx_no_rep = copy.deepcopy(base_index)
        for nid in range(n):
            idx_no_rep._store.set_vector(nid, pos_t_final[nid])

        paths_visit_any_16 = 0
        nodes_visited_counts_16 = []
        for q in q_static:
            path = trace_search_path(idx_no_rep, q, ef=16)
            drift_in_path = path & drifting_set
            if len(drift_in_path) > 0:
                paths_visit_any_16 += 1
            nodes_visited_counts_16.append(len(drift_in_path))

        results["static_tracing"][f"{int(speed)}x"] = {
            "frac_paths_visiting_drifting": paths_visit_any_16 / float(len(q_static)),
            "mean_drifting_nodes_visited": float(np.mean(nodes_visited_counts_16)),
        }

        # Arms to evaluate:
        arms = [
            "no_repair",
            "ours_after",
            "arm3_rebuild_ef64",
            "fresh_insert_reference",
            "arm_inplace_reinsert",
            "arm_inplace_reinsert_heuristic",
            "arm4_exact_top_m",
            "arm5_oracle_failures",
            "oracle_control_rebuilt",
            "oracle_control_all_drift",
        ]

        # First, run ours_after to capture rebuilt nodes per step for control (i)
        rebuilt_per_step: dict[int, list[int]] = {}
        idx_ours = copy.deepcopy(base_index)
        store_ours = copy.deepcopy(base_store)
        det_meter_ours = CostMeter()
        heap_ours = KineticEventHeap(version_of=idx_ours.version)
        det_ours = KineticDetector(
            index=idx_ours,
            store=store_ours,
            heap=heap_ours,
            trajectory_of=make_traj(),
            meter=det_meter_ours,
            horizon=0.50,
            sync_vector=make_hnsw_sync(idx_ours),
        )
        det_ours.schedule_all(t0=0.0)
        engine_ours = KHNSWRepairEngine(
            store=store_ours,
            detector=det_ours,
            time_source=0.0,
            search_layer=search_layer,
            ef=16,
        )

        categories_ours = {
            "search": 0,
            "planning": 0,
            "store": 0,
            "detector": 0,
            "reverse_link": 0,
        }
        cost_ours = 0
        rebuilt_this_step: list[int] = []
        orig_rebuild_fn = loop_mod._rebuild_moved_out_links

        def _recording_rebuild(
            *args_r: Any,
            _rec: list[int] = rebuilt_this_step,
            _fn: Any = orig_rebuild_fn,
            **kwargs_r: Any,
        ) -> Any:
            m = kwargs_r.get("moved", args_r[1] if len(args_r) > 1 else None)
            if m is not None:
                _rec.append(int(m))
            return _fn(*args_r, **kwargs_r)

        loop_mod._rebuild_moved_out_links = _recording_rebuild
        try:
            for step in range(1, steps + 1):
                t_curr = round(step * dt, 5)
                pos_step = drift.positions(t_curr)
                for nid in range(n):
                    idx_ours._store.set_vector(nid, pos_step[nid].astype(np.float32))
                engine_ours._time_source = t_curr

                rebuilt_this_step.clear()
                det_before = det_meter_ours.count
                res = engine_ours.run_repairs(idx_ours, budget=1e9, sink=ListSink())
                det_cost = det_meter_ours.count - det_before
                categories_ours["detector"] += det_cost
                for k in ["search", "planning", "store", "reverse_link"]:
                    categories_ours[k] += res["budget_used"].get(k, 0)
                rebuild_cost = sum(
                    res["budget_used"].get(k, 0)
                    for k in ["search", "planning", "store", "reverse_link"]
                )
                cost_ours += det_cost + rebuild_cost
                rebuilt_per_step[step] = list(rebuilt_this_step)
        finally:
            loop_mod._rebuild_moved_out_links = orig_rebuild_fn

        # Store in-degree analysis for Item E
        for nid in range(n):
            idx_ours._store.set_vector(nid, pos_t_final[nid])

        in_deg_no_rep = [
            sum(1 for u in idx_no_rep.ids() if d_id in idx_no_rep.neighbors(u, 0))
            for d_id in drifting_nodes
        ]
        in_deg_ours = [
            sum(1 for u in idx_ours.ids() if d_id in idx_ours.neighbors(u, 0))
            for d_id in drifting_nodes
        ]

        # In-degree from true top-10 neighbors
        in_deg_true10_no_rep = []
        for d_id in drifting_nodes:
            dists = np.sum((pos_t_final - pos_t_final[d_id]) ** 2, axis=1)
            order = np.argsort(dists)
            true_10 = [int(u) for u in order if u != d_id][:10]
            in_deg_true10_no_rep.append(
                sum(1 for u in true_10 if d_id in idx_no_rep.neighbors(u, 0))
            )

        in_deg_true10_ours = []
        for d_id in drifting_nodes:
            dists = np.sum((pos_t_final - pos_t_final[d_id]) ** 2, axis=1)
            order = np.argsort(dists)
            true_10 = [int(u) for u in order if u != d_id][:10]
            in_deg_true10_ours.append(sum(1 for u in true_10 if d_id in idx_ours.neighbors(u, 0)))

        results["in_degree"][f"{int(speed)}x"] = {
            "no_repair": {
                "mean_in_deg": float(np.mean(in_deg_no_rep)),
                "mean_in_deg_from_true10": float(np.mean(in_deg_true10_no_rep)),
            },
            "ours_after": {
                "mean_in_deg": float(np.mean(in_deg_ours)),
                "mean_in_deg_from_true10": float(np.mean(in_deg_true10_ours)),
            },
        }

        idx_fresh_ref: HNSWIndex | None = None

        # Evaluate each arm
        for arm in arms:
            arm_categories: dict[str, int] = {
                "search": 0,
                "planning": 0,
                "store": 0,
                "detector": 0,
                "reverse_link": 0,
            }
            brute_force_extra_cost = 0

            if arm == "no_repair":
                idx_arm = idx_no_rep
                total_cost = 0
            elif arm == "ours_after":
                idx_arm = idx_ours
                total_cost = cost_ours
                arm_categories = dict(categories_ours)
            elif arm == "fresh_insert_reference":
                # Build index with static nodes, then insert drifting nodes at final positions
                idx_arm = HNSWIndex(dim=d, m=m, seed=seed)
                static_arr = np.array(static_nodes, dtype=np.int64)
                idx_arm.add(static_arr, pos_t_final[static_arr])
                before_add = idx_arm.distance_computations
                drift_arr = np.array(drifting_nodes, dtype=np.int64)
                idx_arm.add(drift_arr, pos_t_final[drift_arr])
                total_cost = idx_arm.distance_computations - before_add
                arm_categories["search"] = total_cost
                if speed == 30.0:
                    idx_fresh_ref = idx_arm
            elif arm == "arm_inplace_reinsert":
                idx_arm = copy.deepcopy(base_index)
                for nid in range(n):
                    idx_arm._store.set_vector(nid, pos_t_final[nid])
                before_add = idx_arm.distance_computations
                for nid in drifting_nodes:
                    idx_arm.delete(nid)
                drift_arr = np.array(drifting_nodes, dtype=np.int64)
                idx_arm.add(drift_arr, pos_t_final[drift_arr])
                total_cost = idx_arm.distance_computations - before_add
                arm_categories["search"] = total_cost
            elif arm == "arm_inplace_reinsert_heuristic":
                idx_arm = copy.deepcopy(base_index)
                for nid in range(n):
                    idx_arm._store.set_vector(nid, pos_t_final[nid])
                heuristic_dists = 0
                for d_id in drifting_nodes:
                    top_lvl = idx_arm.node_level(d_id)
                    for lc in range(top_lvl + 1):
                        static_in = [u for u in idx_arm.in_neighbors(d_id, lc) if u in static_set]
                        d_nbrs = set(idx_arm.neighbors(d_id, lc))
                        for u in static_in:
                            u_nbrs = set(idx_arm.neighbors(u, lc))
                            cands = (u_nbrs | d_nbrs) - {u, d_id}
                            max_m = idx_arm.max_degree(lc)
                            cand_list = list(cands)
                            if cand_list:
                                heuristic_dists += len(cand_list)
                                pruned = select_neighbors_heuristic(
                                    idx_arm._store, u, cand_list, max_m, level=lc
                                )
                                heuristic_dists += (len(pruned) * (len(pruned) - 1)) // 2
                                idx_arm._store.set_neighbors(u, lc, pruned)
                before_add = idx_arm.distance_computations
                for nid in drifting_nodes:
                    idx_arm.delete(nid)
                drift_arr = np.array(drifting_nodes, dtype=np.int64)
                idx_arm.add(drift_arr, pos_t_final[drift_arr])
                add_cost = idx_arm.distance_computations - before_add
                total_cost = heuristic_dists + add_cost
                arm_categories["search"] = add_cost
                arm_categories["planning"] = heuristic_dists
            else:
                idx_arm = copy.deepcopy(base_index)
                store_arm = copy.deepcopy(base_store)
                det_meter_arm = CostMeter()
                heap_arm = KineticEventHeap(version_of=idx_arm.version)
                det_arm = KineticDetector(
                    index=idx_arm,
                    store=store_arm,
                    heap=heap_arm,
                    trajectory_of=make_traj(),
                    meter=det_meter_arm,
                    horizon=0.50,
                    sync_vector=make_hnsw_sync(idx_arm),
                )
                det_arm.schedule_all(t0=0.0)

                total_cost = 0
                oracle_dist_used = 0
                oracle_det_cost = 0

                # Rebuild hooks for arm3 and arm4
                orig_rebuild = loop_mod._rebuild_moved_out_links

                def _build_hook(arm_name: str, orig_fn: Any) -> Any:
                    def _custom(
                        index: Any,
                        moved: int,
                        search_layer_fn: Any,
                        store_or_adapter: Any,
                        meter_search: CostMeter,
                        meter_planning: CostMeter,
                        ef: int | None,
                        meter_reverse_link: CostMeter | None = None,
                        detector: KineticDetector | None = None,
                        t: float = 0.0,
                    ) -> list[Rewrite]:
                        if arm_name == "arm3_rebuild_ef64":
                            return orig_fn(
                                index,
                                moved,
                                search_layer_fn,
                                store_or_adapter,
                                meter_search,
                                meter_planning,
                                64,
                                meter_reverse_link,
                                detector,
                                t,
                            )
                        elif arm_name == "arm4_exact_top_m":
                            moved_id = int(moved)
                            top_lvl = index.node_level(moved_id)
                            rewrites = []
                            v_moved = index.vector(moved_id)
                            for lc in range(top_lvl, -1, -1):
                                eligible = [
                                    int(u)
                                    for u in index.ids()
                                    if index.node_level(u) >= lc and u != moved_id
                                ]
                                diffs = np.array([index.vector(u) - v_moved for u in eligible])
                                dists = np.sum(diffs**2, axis=1)
                                top_k = np.argpartition(dists, min(m * 2, len(eligible) - 1))[
                                    : min(m * 2, len(eligible))
                                ]
                                cand_ids = {eligible[k] for k in top_k}
                                cap = index.max_degree(lc)
                                ranked = rank_pool(
                                    base=moved_id,
                                    pool=cand_ids,
                                    ef=16,
                                    meter=meter_planning,
                                    vector_of=index.vector,
                                )
                                selected = prune(
                                    ranked=ranked,
                                    base=moved_id,
                                    cap=cap,
                                    alpha=1.0,
                                    skip_if_below_cap=False,
                                    meter=meter_planning,
                                    vector_of=index.vector,
                                )
                                new_nbrs = tuple(c for _, c in selected)
                                old_nbrs = tuple(index.neighbors(moved_id, lc))
                                rewrites.append(
                                    Rewrite(
                                        node=moved_id,
                                        level=lc,
                                        old=old_nbrs,
                                        new=new_nbrs,
                                    )
                                )
                            return rewrites
                        else:
                            return orig_fn(
                                index,
                                moved,
                                search_layer_fn,
                                store_or_adapter,
                                meter_search,
                                meter_planning,
                                ef,
                                meter_reverse_link,
                                detector,
                                t,
                            )

                    return _custom

                loop_mod._rebuild_moved_out_links = _build_hook(arm, orig_rebuild)

                try:
                    engine_arm = KHNSWRepairEngine(
                        store=store_arm,
                        detector=det_arm,
                        time_source=0.0,
                        search_layer=search_layer,
                        ef=16,
                    )

                    for step in range(1, steps + 1):
                        t_curr = round(step * dt, 5)
                        pos_step = drift.positions(t_curr)
                        for nid in range(n):
                            idx_arm._store.set_vector(nid, pos_step[nid].astype(np.float32))

                        if arm in {
                            "arm5_oracle_failures",
                            "oracle_control_rebuilt",
                            "oracle_control_all_drift",
                        }:
                            det_before = det_meter_arm.count
                            failures = det_arm.advance(t_curr)
                            det_cost = det_meter_arm.count - det_before
                            total_cost += det_cost
                            oracle_det_cost += det_cost

                            if arm == "arm5_oracle_failures":
                                target_nodes = set()
                                for edge in failures:
                                    for nid in (edge[0], edge[1]):
                                        if nid in drifting_set:
                                            target_nodes.add(nid)
                                nodes_to_reinsert = sorted(target_nodes)
                            elif arm == "oracle_control_rebuilt":
                                nodes_to_reinsert = rebuilt_per_step.get(step, [])
                            else:  # oracle_control_all_drift
                                nodes_to_reinsert = list(drifting_nodes)

                            for nid in nodes_to_reinsert:
                                orig_lvl = idx_arm.node_level(nid)
                                dist_before = idx_arm.distance_computations
                                v_node = idx_arm.vector(nid).copy()
                                idx_arm.delete(nid)

                                orig_sample = idx_arm._level_gen.sample
                                idx_arm._level_gen.sample = (  # type: ignore[method-assign]
                                    lambda lvl=orig_lvl: lvl
                                )
                                try:
                                    idx_arm._insert_one(nid, v_node)
                                finally:
                                    idx_arm._level_gen.sample = (  # type: ignore[method-assign]
                                        orig_sample
                                    )

                                dist_used = idx_arm.distance_computations - dist_before
                                total_cost += dist_used
                                oracle_dist_used += dist_used

                            store_arm.build(idx_arm, CostMeter())
                            det_arm.schedule_all(t0=t_curr)
                        else:
                            engine_arm._time_source = t_curr
                            det_before = det_meter_arm.count
                            res_engine = engine_arm.run_repairs(
                                idx_arm, budget=1e9, sink=ListSink()
                            )
                            det_cost = det_meter_arm.count - det_before
                            arm_categories["detector"] += det_cost
                            for k in ["search", "planning", "store", "reverse_link"]:
                                arm_categories[k] += res_engine["budget_used"].get(k, 0)
                            rebuild_cost = sum(
                                res_engine["budget_used"].get(k, 0)
                                for k in ["search", "planning", "store", "reverse_link"]
                            )
                            total_cost += det_cost + rebuild_cost
                            if arm == "arm4_exact_top_m":
                                brute_force_extra_cost += len(drifting_nodes) * n
                finally:
                    loop_mod._rebuild_moved_out_links = orig_rebuild

                if arm in {
                    "arm5_oracle_failures",
                    "oracle_control_rebuilt",
                    "oracle_control_all_drift",
                }:
                    arm_categories["search"] = oracle_dist_used
                    arm_categories["detector"] = oracle_det_cost

            # Synchronize to final positions for evaluation
            for nid in range(n):
                idx_arm._store.set_vector(nid, pos_t_final[nid])

            # Violated edge check
            fresh_store = CertificateStore()
            fresh_store.build(idx_arm, CostMeter())
            violated = sum(
                1 for s in fresh_store._edge_status.values() if s == EdgeStatus.VIOLATED.value
            )

            rec16_drift = evaluate_recall(idx_arm, q_drift, exact_drift, k=10, ef=16)
            rec32_drift = evaluate_recall(idx_arm, q_drift, exact_drift, k=10, ef=32)
            rec16_static = evaluate_recall(idx_arm, q_static, exact_static, k=10, ef=16)
            rec32_static = evaluate_recall(idx_arm, q_static, exact_static, k=10, ef=32)

            results["runs"].append(
                {
                    "speed": speed,
                    "seed": seed,
                    "arm": arm,
                    "drift_rec16": rec16_drift,
                    "drift_rec32": rec32_drift,
                    "static_rec16": rec16_static,
                    "static_rec32": rec32_static,
                    "violated": violated,
                    "cost": total_cost,
                    "cost_with_brute_force": total_cost + brute_force_extra_cost
                    if arm == "arm4_exact_top_m"
                    else total_cost,
                    "cost_by_category": arm_categories,
                    "initial_schedule_all": 0,
                }
            )

        # Audit static node edges at 30x drift
        if speed == 30.0:
            top_16_dict: dict[int, set[int]] = {}
            for s in static_nodes:
                diffs = pos_t_final - pos_t_final[s]
                dists = np.sum(diffs**2, axis=1)
                order = np.argsort(dists)
                top_16 = [int(u) for u in order if u != s][:16]
                top_16_dict[s] = set(top_16)

            def audit_edges(
                idx_to_audit: HNSWIndex | None,
                _top16: dict[int, set[int]] = top_16_dict,
                _drifting: set[int] = drifting_set,
                _static: list[int] = static_nodes,
            ) -> dict[str, float]:
                if idx_to_audit is None:
                    return {"mean_out_deg": 0.0, "frac_out_to_drift": 0.0, "frac_stale": 0.0}
                out_degs = []
                edges_to_drift = 0
                stale_edges = 0
                total_edges = 0
                for s in _static:
                    nbrs = idx_to_audit.neighbors(s, 0)
                    out_degs.append(len(nbrs))
                    total_edges += len(nbrs)
                    for nb in nbrs:
                        if nb in _drifting:
                            edges_to_drift += 1
                            if nb not in _top16[s]:
                                stale_edges += 1
                mean_deg = float(np.mean(out_degs))
                frac_drift = edges_to_drift / total_edges if total_edges > 0 else 0.0
                frac_stale = stale_edges / edges_to_drift if edges_to_drift > 0 else 0.0
                return {
                    "mean_out_deg": mean_deg,
                    "frac_out_to_drift": frac_drift,
                    "frac_stale": frac_stale,
                }

            results["static_edge_audit"] = {
                "no_repair": audit_edges(idx_no_rep),
                "ours_after": audit_edges(idx_ours),
                "fresh_insert_ref": audit_edges(idx_fresh_ref),
            }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified reconciliation harness for Phase 5h & 5i")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--output", type=str, default="docs/data/reconcile_benchmark.json")
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    if args.parallel:
        seeds = [42, 43, 44] if args.seeds is None else args.seeds
        processes = []
        out_files = []
        repo_root = Path(__file__).resolve().parent.parent
        py_exe = repo_root / ".venv" / "Scripts" / "python.exe"

        for s in seeds:
            out_f = repo_root / "docs" / "data" / f"recon_{s}.json"
            out_files.append(out_f)
            cmd = [
                str(py_exe),
                str(Path(__file__).resolve()),
                "--seeds",
                str(s),
                "--output",
                str(out_f),
            ]
            p = subprocess.Popen(cmd, cwd=repo_root)
            processes.append((s, p))

        for s, p in processes:
            p.wait()
            if p.returncode != 0:
                print(f"Error on seed {s}: exit {p.returncode}")
                sys.exit(p.returncode)

        # Merge results
        combined_runs = []
        static_tracing: dict[str, list[Any]] = {"10x": [], "30x": []}
        in_degrees: dict[str, dict[str, list[Any]]] = {
            "10x": {"no_repair": [], "ours_after": []},
            "30x": {"no_repair": [], "ours_after": []},
        }
        loaded_data = []

        for f in out_files:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            loaded_data.append(data)
            combined_runs.extend(data["runs"])
            for spd in ["10x", "30x"]:
                static_tracing[spd].append(data["static_tracing"][spd])
                for cond in ["no_repair", "ours_after"]:
                    in_degrees[spd][cond].append(data["in_degree"][spd][cond])

        # Headroom runs
        headroom_runs = []
        for d in loaded_data:
            headroom_runs.extend(d.get("headroom", []))

        final_data = {
            "runs": combined_runs,
            "static_tracing": {
                spd: {
                    "frac_paths_visiting_drifting_mean": float(
                        np.mean(
                            [
                                x.get(
                                    "frac_paths_visiting_drifting_mean",
                                    x.get("frac_paths_visiting_drifting", 0.0),
                                )
                                for x in static_tracing[spd]
                            ]
                        )
                    ),
                    "mean_drifting_nodes_visited_mean": float(
                        np.mean(
                            [
                                x.get(
                                    "mean_drifting_nodes_visited_mean",
                                    x.get("mean_drifting_nodes_visited", 0.0),
                                )
                                for x in static_tracing[spd]
                            ]
                        )
                    ),
                }
                for spd in ["10x", "30x"]
            },
            "in_degree": {
                spd: {
                    cond: {
                        "mean_in_deg": float(
                            np.mean([x["mean_in_deg"] for x in in_degrees[spd][cond]])
                        ),
                        "mean_in_deg_from_true10": float(
                            np.mean([x["mean_in_deg_from_true10"] for x in in_degrees[spd][cond]])
                        ),
                    }
                    for cond in ["no_repair", "ours_after"]
                }
                for spd in ["10x", "30x"]
            },
            "static_edge_audit": {
                "per_seed": [d["static_edge_audit"] for d in loaded_data],
                "mean": {
                    cond: {
                        "mean_out_deg": float(
                            np.mean(
                                [d["static_edge_audit"][cond]["mean_out_deg"] for d in loaded_data]
                            )
                        ),
                        "frac_out_to_drift": float(
                            np.mean(
                                [
                                    d["static_edge_audit"][cond]["frac_out_to_drift"]
                                    for d in loaded_data
                                ]
                            )
                        ),
                        "frac_stale": float(
                            np.mean(
                                [d["static_edge_audit"][cond]["frac_stale"] for d in loaded_data]
                            )
                        ),
                    }
                    for cond in ["no_repair", "ours_after", "fresh_insert_ref"]
                },
            },
            "headroom": headroom_runs,
        }

        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(final_data, fp, indent=2)

        for f in out_files:
            if f.exists():
                f.unlink()
        print(f"Parallel reconciliation run completed. Wrote {args.output}")

    else:
        seeds = [42, 43, 44] if args.seeds is None else args.seeds
        combined_runs = []
        static_tracing: dict[str, list[Any]] = {"10x": [], "30x": []}
        in_degrees = {
            "10x": {"no_repair": [], "ours_after": []},
            "30x": {"no_repair": [], "ours_after": []},
        }
        loaded_res = []

        for s in seeds:
            res = run_seed(s)
            loaded_res.append(res)
            combined_runs.extend(res["runs"])
            for spd in ["10x", "30x"]:
                static_tracing[spd].append(res["static_tracing"][spd])
                for cond in ["no_repair", "ours_after"]:
                    in_degrees[spd][cond].append(res["in_degree"][spd][cond])

        headroom_runs = []
        for r in loaded_res:
            headroom_runs.extend(r.get("headroom", []))

        final_data = {
            "runs": combined_runs,
            "static_tracing": {
                spd: {
                    "frac_paths_visiting_drifting_mean": float(
                        np.mean([x["frac_paths_visiting_drifting"] for x in static_tracing[spd]])
                    ),
                    "mean_drifting_nodes_visited_mean": float(
                        np.mean([x["mean_drifting_nodes_visited"] for x in static_tracing[spd]])
                    ),
                }
                for spd in ["10x", "30x"]
            },
            "in_degree": {
                spd: {
                    cond: {
                        "mean_in_deg": float(
                            np.mean([x["mean_in_deg"] for x in in_degrees[spd][cond]])
                        ),
                        "mean_in_deg_from_true10": float(
                            np.mean([x["mean_in_deg_from_true10"] for x in in_degrees[spd][cond]])
                        ),
                    }
                    for cond in ["no_repair", "ours_after"]
                }
                for spd in ["10x", "30x"]
            },
            "static_edge_audit": {
                "per_seed": [r["static_edge_audit"] for r in loaded_res],
                "mean": {
                    cond: {
                        "mean_out_deg": float(
                            np.mean(
                                [r["static_edge_audit"][cond]["mean_out_deg"] for r in loaded_res]
                            )
                        ),
                        "frac_out_to_drift": float(
                            np.mean(
                                [
                                    r["static_edge_audit"][cond]["frac_out_to_drift"]
                                    for r in loaded_res
                                ]
                            )
                        ),
                        "frac_stale": float(
                            np.mean(
                                [r["static_edge_audit"][cond]["frac_stale"] for r in loaded_res]
                            )
                        ),
                    }
                    for cond in ["no_repair", "ours_after", "fresh_insert_ref"]
                },
            },
            "headroom": headroom_runs,
        }

        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(final_data, fp, indent=2)
        print(f"Reconciliation run completed. Wrote {args.output}")


if __name__ == "__main__":
    main()
