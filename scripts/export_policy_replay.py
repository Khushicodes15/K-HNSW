"""Export deterministic policy replay trace events and snapshots for visualization.

Generates:
- initial_snapshot.json: Initial graph topology, layers, coordinates, and adjacency at t=0.
- events.jsonl: Line-delimited trace events conforming to event.schema.json.
- final_snapshot.json: Final graph topology, coordinates, and adjacency after drift & repair.

Usage:
    python scripts/export_policy_replay.py --policy round_robin --steps 5 --n 200 --seed 42
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import numpy as np

from khnsw.core.events.schema import EventEmitter, EventType, ListSink
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.movement.base import mean_nn_distance
from khnsw.core.scheduling.policy import MaintenancePolicy


def setup_drifting_data(
    n: int,
    d: int,
    seed: int,
    drift_frac: float = 0.20,
    displacement_mult: float = 8.0,
    t_final: float = 0.30,
) -> dict[str, Any]:
    """Generate initial points, node partitions, and isotropic velocities (indep_dir)."""
    rng_x = np.random.default_rng(seed)
    rng_id = np.random.default_rng(seed + 1)
    rng_v = np.random.default_rng(seed + 2)

    X0 = rng_x.standard_normal((n, d)).astype(np.float32)
    n_drift = int(round(n * drift_frac))
    drift_nodes = sorted(rng_id.choice(n, size=n_drift, replace=False).tolist())
    drift_set = set(drift_nodes)
    static_nodes = [i for i in range(n) if i not in drift_set]

    scale = mean_nn_distance(X0, seed=seed)
    normals = rng_v.standard_normal((n, d), dtype=np.float64)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit_dirs = (normals / norms).astype(np.float32)

    speed = (displacement_mult * scale) / t_final
    vels = np.zeros((n, d), dtype=np.float32)
    for u in drift_nodes:
        vels[u] = unit_dirs[u] * speed

    return {
        "X0": X0,
        "drift_nodes": drift_nodes,
        "static_nodes": static_nodes,
        "vels": vels,
        "scale": scale,
    }


def export_snapshot(index: HNSWIndex, filepath: str | Path) -> dict[str, Any]:
    """Serialize complete graph structure, layers, coordinates, and adjacency to JSON."""
    snap = index.snapshot()
    snap["vectors"] = {str(u): index.vector(u).tolist() for u in index.ids()}
    snap["adjacency"] = {
        str(u): {str(lvl): index.neighbors(u, lvl) for lvl in range(index.node_level(u) + 1)}
        for u in index.ids()
    }
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)
    return snap


def run_export(
    policy_name: str = "round_robin",
    steps: int = 5,
    n: int = 200,
    d: int = 8,
    m: int = 8,
    seed: int = 42,
    out_dir: str | Path = "replay_output",
    drift_frac: float = 0.20,
    displacement_mult: float = 8.0,
    t_final: float = 0.30,
    B: int | None = None,
    K: int = 2,
    theta: float = 1.0,
    pre_repair: bool = False,
) -> dict[str, Any]:
    """Execute policy replay export and serialize snapshots and events."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Setup data and build HNSW index
    data = setup_drifting_data(
        n=n,
        d=d,
        seed=seed,
        drift_frac=drift_frac,
        displacement_mult=displacement_mult,
        t_final=t_final,
    )
    X0 = data["X0"]
    vels = data["vels"]
    drift_nodes = data["drift_nodes"]

    idx = HNSWIndex(dim=d, m=m, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X0)

    # 2. Export initial snapshot
    initial_snapshot_path = out_path / "initial_snapshot.json"
    export_snapshot(idx, initial_snapshot_path)

    # 3. Setup event emission
    run_id = f"policy-replay-{policy_name}-seed{seed}"
    sink = ListSink()
    emitter = EventEmitter(run_id=run_id, sink=sink)

    # Emit run_start
    emitter.emit(
        type=EventType.run_start,
        node=None,
        level=None,
        version=None,
        payload={
            "total_nodes": n,
            "dim": d,
            "dataset": "gaussian_synthetic",
            "policy": policy_name,
        },
        t=0.0,
    )

    # 4. Setup maintenance policy
    policy_kwargs: dict[str, Any] = {"emitter": emitter, "pre_repair": pre_repair}
    if policy_name in ("round_robin", "budgeted_priority"):
        if B is None:
            n_drift = len(drift_nodes)
            B = max(1, int(np.ceil(n_drift / steps)))
        policy_kwargs["B"] = B
    elif policy_name == "periodic":
        policy_kwargs["K"] = K
    elif policy_name == "lazy":
        policy_kwargs["theta"] = theta

    maintenance = MaintenancePolicy(policy=policy_name, **policy_kwargs)

    # 5. Simulation steps
    dt = t_final / steps
    current_X = X0.copy()

    for s in range(1, steps + 1):
        t = round(s * dt, 6)
        delta_step = vels * dt
        current_X += delta_step

        # Move nodes and emit move events
        for u in drift_nodes:
            disp_norm = float(np.linalg.norm(delta_step[u]))
            idx.move(u, current_X[u])
            emitter.emit(
                type=EventType.move,
                node=u,
                level=0,
                version=idx.version(u),
                payload={
                    "delta_norm": disp_norm,
                    "new_vector": current_X[u].tolist(),
                },
                t=t,
            )

        # Step maintenance policy
        maintenance.step(idx, current_X, t=t)

    # Emit run_end
    emitter.emit(
        type=EventType.run_end,
        node=None,
        level=None,
        version=None,
        payload={"status": "completed", "total_events": len(sink.events) + 1},
        t=t_final,
    )

    # 6. Export final snapshot
    final_snapshot_path = out_path / "final_snapshot.json"
    export_snapshot(idx, final_snapshot_path)

    # 7. Write events.jsonl
    events_path = out_path / "events.jsonl"
    with open(events_path, "w", encoding="utf-8") as f:
        for ev in sink.events:
            line = json.dumps(ev.to_dict(), allow_nan=False)
            f.write(line + "\n")

    # 8. Event counts summary
    counts: collections.Counter[str] = collections.Counter()
    for ev in sink.events:
        type_str = ev.type.value if isinstance(ev.type, EventType) else str(ev.type)
        counts[type_str] += 1

    return {
        "initial_snapshot": initial_snapshot_path,
        "final_snapshot": final_snapshot_path,
        "events": events_path,
        "total_events": len(sink.events),
        "counts": dict(counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export policy replay events and snapshots for visualization"
    )
    parser.add_argument(
        "--policy",
        choices=["round_robin", "periodic", "lazy", "budgeted_priority"],
        default="round_robin",
        help="Maintenance policy type",
    )
    parser.add_argument("--steps", type=int, default=5, help="Number of simulation steps")
    parser.add_argument("--n", type=int, default=200, help="Number of nodes in the graph")
    parser.add_argument("--d", type=int, default=8, help="Vector dimensionality")
    parser.add_argument("--m", type=int, default=8, help="HNSW max degree")
    parser.add_argument("--seed", type=int, default=42, help="Random generator seed")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="replay_output",
        help="Output directory for snapshot and event files",
    )
    parser.add_argument(
        "--drift-frac",
        type=float,
        default=0.20,
        help="Fraction of nodes that drift",
    )
    parser.add_argument(
        "--displacement-mult",
        type=float,
        default=8.0,
        help="Displacement multiplier in units of mean 1-NN distance",
    )
    parser.add_argument(
        "--t-final",
        type=float,
        default=0.30,
        help="Total simulation duration",
    )
    parser.add_argument(
        "--B",
        type=int,
        default=None,
        help="Reinsertion budget B per step",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=2,
        help="Periodic reinsertion period K",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=1.0,
        help="Displacement threshold for lazy reinsertion",
    )
    parser.add_argument(
        "--pre-repair",
        action="store_true",
        help="Enable in-neighbor pre-repair",
    )
    args = parser.parse_args()

    res = run_export(
        policy_name=args.policy,
        steps=args.steps,
        n=args.n,
        d=args.d,
        m=args.m,
        seed=args.seed,
        out_dir=args.out_dir,
        drift_frac=args.drift_frac,
        displacement_mult=args.displacement_mult,
        t_final=args.t_final,
        B=args.B,
        K=args.K,
        theta=args.theta,
        pre_repair=args.pre_repair,
    )

    print(f"Exported {res['initial_snapshot']}")
    print(f"Exported {res['final_snapshot']}")
    print(f"Exported {res['events']} ({res['total_events']} events)")
    print("\nPolicy Replay Trace Event Summary:")
    print("-" * 35)
    print(f"{'Event Type':<20} | {'Count':>10}")
    print("-" * 35)
    for etype in sorted(res["counts"].keys()):
        print(f"{etype:<20} | {res['counts'][etype]:>10}")
    print("-" * 35)
    print(f"{'Total Events':<20} | {res['total_events']:>10}")


if __name__ == "__main__":
    main()
