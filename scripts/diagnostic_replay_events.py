"""Diagnostic script verifying whether MaintenancePolicy.step emits replay events."""

import collections

import numpy as np

from khnsw.core.events.schema import EventEmitter, ListSink
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.scheduling.policy import MaintenancePolicy


def main() -> None:
    rng = np.random.default_rng(42)
    X0 = rng.normal(size=(10, 4)).astype(np.float32)
    idx = HNSWIndex(dim=4, m=4, ef_construction=16, seed=42)
    idx.add(np.arange(10, dtype=np.int64), X0)

    sink = ListSink()
    emitter = EventEmitter(run_id="diagnostic_replay_events", sink=sink)
    policy = MaintenancePolicy(
        policy="round_robin",
        B=2,
        emitter=emitter,
        seed=42,
    )
    policy.register(idx, X0)

    # Shift positions and perform step
    X1 = (X0 + 0.1).astype(np.float32)
    reinserted = policy.step(idx, X1, t=1)

    print(f"Policy maintained nodes: {reinserted}")
    print(f"Total events captured by sink: {len(sink.events)}")
    counts = collections.Counter(
        ev.type.value if hasattr(ev.type, "value") else str(ev.type) for ev in sink.events
    )
    print("Event counts by type:")
    for ev_type, count in sorted(counts.items()):
        print(f"  {ev_type}: {count}")
    for ev in sink.events:
        print(ev)


if __name__ == "__main__":
    main()
