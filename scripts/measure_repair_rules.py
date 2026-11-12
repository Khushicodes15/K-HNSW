"""Measurement script evaluating repair rules on Member 2's HNSWIndex.

Evaluates:
- Pre-settle construction violation counts and settle pass computational cost.
- Across 50 continuous moves after settle pass:
  - Mean and max distance computations per step.
  - Mean modified edges per step (rewrites).
  - Mean post-repair violation count remaining in the graph.
"""

from __future__ import annotations

import numpy as np

from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.cost import CostMeter
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.repair.apply import apply_plan
from khnsw.core.repair.plan import khnsw_plan, mnru_plan, native_plan


def _collect_violated(
    index: HNSWIndex, store: CertificateStore
) -> dict[tuple[int, int], list[int]]:
    violated: dict[tuple[int, int], list[int]] = {}
    for u in index.ids():
        for lvl in range(index.node_level(u) + 1):
            for v in index.neighbors(u, lvl):
                if store.status((u, v, lvl)) == EdgeStatus.VIOLATED:
                    violated.setdefault((u, lvl), []).append(v)
    return violated


def measure() -> None:
    n = 200
    m = 8
    rules = ["native", "mnru_beta", "mnru_gamma", "mnru_thn", "khnsw"]

    for d in [8, 2]:
        print("\n=======================================================")
        print(f"  Repair Rule Measurement (n={n}, M={m}, d={d}, 50 moves)")
        print("=======================================================")

        # 1. Baseline setup and settle pass measurement
        rng_settle = np.random.default_rng(42)
        X_settle = rng_settle.standard_normal((n, d)).astype(np.float32)
        idx_settle = HNSWIndex(dim=d, m=m, seed=42)
        idx_settle.add(np.arange(n, dtype=np.int64), X_settle)

        store_settle = CertificateStore()
        meter_settle = CostMeter()
        store_settle.build(idx_settle, meter_settle)

        pre_settle_violated = _collect_violated(idx_settle, store_settle)
        pre_settle_count = sum(len(v) for v in pre_settle_violated.values())

        c_before = meter_settle.count
        settle_plan = khnsw_plan(idx_settle, violated=pre_settle_violated, meter=meter_settle)
        apply_plan(idx_settle, settle_plan, store=store_settle, meter=meter_settle)
        settle_cost = meter_settle.count - c_before

        header = (
            f"{'Rule':<12} | {'Mean Dist':>10} | {'Max Dist':>9} | "
            f"{'Mean Mod':>9} | {'Mean Viol':>10}"
        )
        print(f"Pre-settle Violated Edges: {pre_settle_count}")
        print(f"Settle Pass Distance Computations: {settle_cost:,}")
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        # 2. Per-rule evaluation over 50 identical moves
        for rule in rules:
            rng = np.random.default_rng(42)
            X = rng.standard_normal((n, d)).astype(np.float32)
            idx = HNSWIndex(dim=d, m=m, seed=42)
            idx.add(np.arange(n, dtype=np.int64), X)

            store = CertificateStore()
            meter = CostMeter()
            store.build(idx, meter)

            # Settle pass
            init_v = _collect_violated(idx, store)
            s_plan = khnsw_plan(idx, violated=init_v, meter=meter)
            apply_plan(idx, s_plan, store=store, meter=meter)

            # Deterministic move sequence
            move_rng = np.random.default_rng(100)
            moves = [
                (
                    int(move_rng.choice(idx.ids())),
                    move_rng.normal(scale=0.1, size=d).astype(np.float32),
                )
                for _ in range(50)
            ]

            dist_history: list[int] = []
            mod_history: list[int] = []
            viol_history: list[int] = []

            for moved, delta in moves:
                c_start = meter.count
                idx.move(moved, idx.vector(moved) + delta)
                store.on_move(idx, moved, meter)

                if rule == "native":
                    plan = native_plan(idx, moved, meter=meter)
                elif rule == "mnru_beta":
                    plan = mnru_plan(idx, moved, alpha=1.0, scope="one_hop", meter=meter)
                elif rule == "mnru_gamma":
                    plan = mnru_plan(idx, moved, alpha=1.1, scope="one_hop", meter=meter)
                elif rule == "mnru_thn":
                    plan = mnru_plan(idx, moved, alpha=1.1, scope="two_hop", meter=meter)
                elif rule == "khnsw":
                    v_dict = _collect_violated(idx, store)
                    plan = khnsw_plan(idx, violated=v_dict, meter=meter)
                else:
                    raise ValueError(f"Unknown rule {rule}")

                apply_plan(idx, plan, store=store, meter=meter)

                step_cost = meter.count - c_start
                dist_history.append(step_cost)

                step_mod = sum(len(set(r.old) ^ set(r.new)) for r in plan.rewrites)
                mod_history.append(step_mod)

                current_violations = sum(
                    1
                    for u in idx.ids()
                    for lvl in range(idx.node_level(u) + 1)
                    for v in idx.neighbors(u, lvl)
                    if store.status((u, v, lvl)) == EdgeStatus.VIOLATED
                )
                viol_history.append(current_violations)

            row = (
                f"{rule:<12} | {np.mean(dist_history):10.1f} | {np.max(dist_history):9d} | "
                f"{np.mean(mod_history):9.2f} | {np.mean(viol_history):10.2f}"
            )
            print(row)
        print("-" * len(header))


if __name__ == "__main__":
    measure()
