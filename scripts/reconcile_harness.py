"""Unified benchmark harness for Phase 5h reconciliation (Items B, C, D, E).

Runs on HNSWIndex (n=500, M=8, d=8) under 10% kinetic drift at 10x and 30x speeds
(T=0.30, 10 steps, seeds 42, 43, 44).
Unified harness specification:
- Drift: LinearDrift(X0, speed=speed), 50 drifting nodes chosen via rng.choice.
- Ground truth: Brute-force Euclidean top-10 at t=0.30.
- Queries: 50 drifting queries (all drifting nodes at t=0.30) and 100 static queries.
- Cost: Full CostMeter counting detector evaluations (including initial scheduling)
  + rebuild search/planning/store/reverse evaluations.
- Rows:
  - no_repair
  - ours-after (ef=16 in rebuild)
  - arm 3 (rebuild with ef=64)
  - fresh-insert reference (drifting nodes built at final positions into settled static index)
  - arm 4 (exact top-M candidates, reporting cost with and without brute-force search)
  - arm 5 (oracle reinsert of confirmed failure drifting nodes, fixed levels)
  - oracle control (i): reinsert exactly the nodes our loop rebuilt each step, fixed levels
  - oracle control (ii): reinsert all 50 drifting nodes every step, fixed levels
- Static search path tracing (Item D): fraction of static queries visiting >=1 drifting node,
  and mean drifting nodes visited.
- In-degree analysis (Item E): level-0 in-degree and in-degree from true top-10 neighbors.
"""

from __future__ import annotations

import argparse
import copy
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
from khnsw.core.repair.loop import prune, rank_pool, repair_loop
from khnsw.core.repair.plan import Rewrite
from khnsw.core.scheduling.event_heap import KineticEventHeap
from khnsw.core.search.search_layer import search_layer


def make_hnsw_sync(index: HNSWIndex) -> Callable[[int, np.ndarray], None]:
    def _sync(node_id: int, pos: np.ndarray) -> None:
        index._store.set_vector(node_id, pos.astype(np.float32))

    return _sync


def settle_hnsw(index: HNSWIndex, store: CertificateStore) -> None:
    repair_loop(index, store)


def brute_force_knn(X: np.ndarray, queries: np.ndarray, k: int = 10) -> list[set[int]]:
    exact_neighbors: list[set[int]] = []
    for q in queries:
        diffs = X - q
        dists = np.sum(diffs**2, axis=1)
        top_k = np.argpartition(dists, k)[:k]
        exact_neighbors.append({int(idx) for idx in top_k})
    return exact_neighbors


def evaluate_recall(
    index: HNSWIndex, queries: np.ndarray, exact_sets: list[set[int]], k: int = 10, ef: int = 16
) -> float:
    recalls: list[float] = []
    for q, exact in zip(queries, exact_sets, strict=False):
        ids, _ = index.search(q, k=k, ef=ef)
        found = {int(nid) for nid in ids[:k]}
        recalls.append(len(found & exact) / float(k))
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
        res = search_layer(index, query, [curr_ep], 1, lc)
        if res:
            first = res[0]
            nid = int(first.node_id if hasattr(first, "node_id") else first)
            curr_ep = nid
            visited.add(nid)

    # 2. Level 0 beam search
    res0 = search_layer(index, query, [curr_ep], ef, 0)
    for r in res0:
        nid = int(r.node_id if hasattr(r, "node_id") else r)
        visited.add(nid)

    return visited


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

    rng_s = np.random.default_rng(seed + 500)
    eval_static_nodes = sorted(rng_s.choice(static_nodes, size=100, replace=False).tolist())

    results: dict[str, Any] = {
        "seed": seed,
        "runs": [],
        "static_tracing": {},
        "in_degree": {},
    }

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
        # 1. no_repair
        # 2. ours_after (ef=16)
        # 3. fresh_insert_reference
        # 4. arm3_rebuild_ef64
        # 5. arm4_exact_top_m
        # 6. arm5_oracle_failures
        # 7. oracle_control_rebuilt_nodes (i)
        # 8. oracle_control_all_drifting_every_step (ii)

        arms = [
            "no_repair",
            "ours_after",
            "arm3_rebuild_ef64",
            "fresh_insert_reference",
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

        cost_ours = 0
        for step in range(1, steps + 1):
            t_curr = round(step * dt, 5)
            pos_step = drift.positions(t_curr)
            for nid in range(n):
                idx_ours._store.set_vector(nid, pos_step[nid].astype(np.float32))
            engine_ours._time_source = t_curr

            det_before = det_meter_ours.count
            res = engine_ours.run_repairs(idx_ours, budget=1e9, sink=ListSink())
            det_cost = det_meter_ours.count - det_before

            # Identify nodes that were rebuilt at this step
            rebuilt_per_step[step] = list(res.get("unrepaired_nodes", []))
            rebuild_cost = sum(
                res["budget_used"].get(k, 0)
                for k in ["search", "planning", "store", "reverse_link"]
            )
            cost_ours += det_cost + rebuild_cost
        cost_ours += det_meter_ours.count  # Include initial scheduling

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

        # Evaluate each arm
        for arm in arms:
            if arm == "no_repair":
                idx_arm = idx_no_rep
                total_cost = 0
            elif arm == "ours_after":
                idx_arm = idx_ours
                total_cost = cost_ours
            elif arm == "fresh_insert_reference":
                # Build index with static nodes, then insert drifting nodes at final positions
                idx_arm = HNSWIndex(dim=d, m=m, seed=seed)
                # Add static nodes first
                static_arr = np.array(static_nodes, dtype=np.int64)
                idx_arm.add(static_arr, pos_t_final[static_arr])
                # Add drifting nodes at their final positions
                before_add = idx_arm.distance_computations
                drift_arr = np.array(drifting_nodes, dtype=np.int64)
                idx_arm.add(drift_arr, pos_t_final[drift_arr])
                total_cost = idx_arm.distance_computations - before_add
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

                total_cost = det_meter_arm.count
                brute_force_extra_cost = 0

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
                        det: KineticDetector | None = None,
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
                                det,
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
                                det,
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

                            store_arm.build(idx_arm, CostMeter())
                            det_arm.schedule_all(t0=t_curr)
                        else:
                            engine_arm._time_source = t_curr
                            det_before = det_meter_arm.count
                            res_engine = engine_arm.run_repairs(
                                idx_arm, budget=1e9, sink=ListSink()
                            )
                            det_cost = det_meter_arm.count - det_before
                            rebuild_cost = sum(
                                res_engine["budget_used"].get(k, 0)
                                for k in ["search", "planning", "store", "reverse_link"]
                            )
                            total_cost += det_cost + rebuild_cost
                            if arm == "arm4_exact_top_m":
                                # Brute-force: 50 drifting nodes * 500 distances per step
                                brute_force_extra_cost += len(drifting_nodes) * n
                finally:
                    loop_mod._rebuild_moved_out_links = orig_rebuild

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
                }
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified reconciliation harness for Phase 5h")
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
        static_tracing = {"10x": [], "30x": []}
        in_degrees = {
            "10x": {"no_repair": [], "ours_after": []},
            "30x": {"no_repair": [], "ours_after": []},
        }

        for f in out_files:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            combined_runs.extend(data["runs"])
            for spd in ["10x", "30x"]:
                static_tracing[spd].append(data["static_tracing"][spd])
                for cond in ["no_repair", "ours_after"]:
                    in_degrees[spd][cond].append(data["in_degree"][spd][cond])
            f.unlink()

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
        }

        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(final_data, fp, indent=2)
        print(f"Parallel reconciliation run completed. Wrote {args.output}")

    else:
        seeds = [42, 43, 44] if args.seeds is None else args.seeds
        combined_runs = []
        static_tracing: dict[str, list[Any]] = {"10x": [], "30x": []}
        in_degrees: dict[str, dict[str, list[Any]]] = {
            "10x": {"no_repair": [], "ours_after": []},
            "30x": {"no_repair": [], "ours_after": []},
        }

        for s in seeds:
            res = run_seed(s)
            combined_runs.extend(res["runs"])
            for spd in ["10x", "30x"]:
                static_tracing[spd].append(res["static_tracing"][spd])
                for cond in ["no_repair", "ours_after"]:
                    in_degrees[spd][cond].append(res["in_degree"][spd][cond])

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
        }

        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(final_data, fp, indent=2)
        print(f"Reconciliation run completed. Wrote {args.output}")


if __name__ == "__main__":
    main()
