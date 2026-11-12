"""Kinetic certificate detector coordinating failure prediction and confirmation.

This module is strictly pure Python standard library and NumPy only.
Coordinates KineticEventHeap, CertificateStore, and continuous kinematic trajectories.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.certificates import edge_failure_time
from khnsw.core.detection.cost import CostMeter
from khnsw.core.events.schema import EventEmitter, EventType
from khnsw.core.interfaces import GraphIndex
from khnsw.core.scheduling.event_heap import KineticEventHeap

__all__ = ["KineticDetector"]


class KineticDetector:
    """Coordinates kinetic certificate evaluation, priority scheduling, and confirmation."""

    def __init__(
        self,
        index: GraphIndex,
        store: CertificateStore,
        heap: KineticEventHeap,
        trajectory_of: Callable[[int], tuple[NDArray[np.float64], NDArray[np.float64]]],
        meter: CostMeter,
        emitter: EventEmitter | None = None,
        horizon: float = float("inf"),
        sync_vector: Callable[[int, float], None] | None = None,
    ) -> None:
        """Initialize kinetic detector.

        Parameters
        ----------
        index : GraphIndex
            Underlying graph index holding topology and vectors.
        store : CertificateStore
            Store holding ground-truth edge validity status.
        heap : KineticEventHeap
            Priority queue for kinetic certificate invalidation events.
        trajectory_of : Callable[[int], tuple[NDArray, NDArray]]
            Function returning (offset_a, velocity_b) for a node ID in float64.
        meter : CostMeter
            Cost meter for distance computation accounting during confirmation.
        emitter : EventEmitter | None, default=None
            Optional event emitter for streaming trace events.
        horizon : float, default=infinity
            Lookahead time horizon duration. Only events within [t, t + horizon] are scheduled.
        sync_vector : Callable[[int, float], None] | None, default=None
            Optional callback to update a node's vector to time t without bumping version.
        """
        self._index = index
        self._store = store
        self._heap = heap
        self._trajectory_of = trajectory_of
        self._meter = meter
        self._emitter = emitter
        self._horizon = float(horizon)
        self._sync_vector = sync_vector

        # Diagnostic and accounting counters
        self._scheduled: int = 0
        self._popped: int = 0
        self._confirmed: int = 0
        self._false_alarms: int = 0
        self._rescheduled: int = 0
        self._fail_time_evals: int = 0

    def _sync_node(self, node_id: int, t: float) -> None:
        """Synchronize node vector to simulation timestamp t without bumping version."""
        if self._sync_vector is not None:
            self._sync_vector(node_id, t)
            return

        a, b = self._trajectory_of(node_id)
        pos = np.asarray(a + b * float(t), dtype=np.float32)

        # Non-versioning vector update
        if hasattr(self._index, "_store") and hasattr(self._index._store, "set_vector"):
            self._index._store.set_vector(node_id, pos)
        elif hasattr(self._index, "_vectors") and isinstance(self._index._vectors, dict):
            self._index._vectors[node_id] = pos
        elif hasattr(self._index, "set_vector") and callable(self._index.set_vector):
            self._index.set_vector(node_id, pos)

    def _evaluate_and_schedule_edge(
        self,
        edge: tuple[int, int, int],
        t0: float,
        now: float,
    ) -> bool:
        """Evaluate failure time for edge and push to heap if valid and finite."""
        u, v, level = edge
        if self._store.status(edge) != EdgeStatus.VALID.value:
            return False

        nbrs = list(self._index.neighbors(u, level))
        witness_ids = [w for w in nbrs if w != v]
        if not witness_ids:
            return False

        u_traj = self._trajectory_of(u)
        v_traj = self._trajectory_of(v)
        w_trajs = [self._trajectory_of(w) for w in witness_ids]

        t_end = t0 + self._horizon if math.isfinite(self._horizon) else t0 + 1e6
        res = edge_failure_time(u_traj, v_traj, w_trajs, t0=t0, t_end=t_end)
        self._fail_time_evals += len(w_trajs)

        if res is None:
            return False

        t_fail, w_idx = res
        if not math.isfinite(t_fail) or t_fail > t_end:
            return False

        witness_id = witness_ids[w_idx]
        push_time = max(t_fail, now)
        self._heap.push(edge, push_time, witness_id, now=now)
        return True

    def schedule_all(self, t0: float = 0.0) -> None:
        """Schedule kinetic certificates for all VALID edges across all hierarchy layers."""
        t0_f = float(t0)
        max_lvl = self._index.max_level()
        for level in range(max_lvl + 1):
            for u in self._index.ids():
                if self._index.node_level(u) < level:
                    continue
                for v in self._index.neighbors(u, level):
                    edge = (int(u), int(v), int(level))
                    if self._evaluate_and_schedule_edge(edge, t0=t0_f, now=t0_f):
                        self._scheduled += 1

    def advance(self, t: float) -> list[tuple[int, int, int]]:
        """Advance simulation time to t, popping due events and verifying failures.

        Parameters
        ----------
        t : float
            Current simulation timestamp.

        Returns
        -------
        list[tuple[int, int, int]]
            List of confirmed violated edges (u, v, level).
        """
        t_curr = float(t)
        due = self._heap.pop_due(t_curr)
        self._popped += len(due)

        confirmed_edges: list[tuple[int, int, int]] = []
        evaluated_sources: dict[
            tuple[int, int], dict[tuple[int, int, int], tuple[str, int | None]]
        ] = {}

        for item in due:
            edge = item.edge
            u, v, level = edge

            # Avoid duplicates if multiple events for same edge popped
            if edge in confirmed_edges:
                continue

            # Edge must still exist in graph
            nbrs = list(self._index.neighbors(u, level))
            if v not in nbrs:
                continue

            source_key = (u, level)
            if source_key not in evaluated_sources:
                # Sync source and outgoing neighbors to current time in float32
                self._sync_node(u, t_curr)
                for w in nbrs:
                    self._sync_node(w, t_curr)
                # Recheck via CertificateStore using real float32 stored vectors
                results = self._store.certify_source(self._index, u, level, self._meter)
                evaluated_sources[source_key] = results
            else:
                results = evaluated_sources[source_key]

            status, witness = results.get(edge, (EdgeStatus.VALID.value, None))
            if status == EdgeStatus.VIOLATED.value:
                self._confirmed += 1
                confirmed_edges.append(edge)
                if self._emitter is not None:
                    self._emitter.emit(
                        type=EventType.cert_fail,
                        node=u,
                        level=level,
                        version=self._index.version(u),
                        payload={
                            "neighbor_id": v,
                            "reason": "relative_distance_violation",
                            "witness_id": witness if witness is not None else item.witness,
                            "t": t_curr,
                        },
                    )
            else:
                # Float64 prediction was not confirmed in float32 (near-tie/numerical discrepancy)
                self._false_alarms += 1
                eps = 1e-9 * max(1.0, abs(t_curr))
                if self._evaluate_and_schedule_edge(edge, t0=t_curr + eps, now=t_curr):
                    self._rescheduled += 1

        return confirmed_edges

    def on_trajectory_change(self, node: int, t: float = 0.0) -> None:
        """Update and reschedule certificates after node's trajectory or coordinates change.

        Parameters
        ----------
        node : int
            Node ID whose trajectory or position was modified.
        t : float, default=0.0
            Simulation timestamp when trajectory change took effect.
        """
        node_int = int(node)
        t_curr = float(t)
        node_lvl = self._index.node_level(node_int)

        affected: set[tuple[int, int, int]] = set()
        for lvl in range(node_lvl + 1):
            # 1. Outgoing edges of node
            for v in self._index.neighbors(node_int, lvl):
                affected.add((node_int, int(v), lvl))
            # 2. Outgoing edges of in-neighbors (where node is witness or target)
            for u in self._index.in_neighbors(node_int, lvl):
                for v in self._index.neighbors(u, lvl):
                    affected.add((int(u), int(v), lvl))

        for edge in affected:
            self._heap.invalidate_edge(edge)
            if self._evaluate_and_schedule_edge(edge, t0=t_curr, now=t_curr):
                self._rescheduled += 1

    def on_topology_change(self, u: int, level: int, t: float = 0.0) -> None:
        """Update certificates after topological mutation on (u, level).

        Parameters
        ----------
        u : int
            Source node ID whose neighbor list was modified.
        level : int
            Hierarchy layer level.
        t : float, default=0.0
            Simulation timestamp when topology change occurred.
        """
        u_int = int(u)
        lvl_int = int(level)
        t_curr = float(t)

        self._heap.invalidate_source(u_int, lvl_int)

        for v in self._index.neighbors(u_int, lvl_int):
            edge = (u_int, int(v), lvl_int)
            if self._evaluate_and_schedule_edge(edge, t0=t_curr, now=t_curr):
                self._rescheduled += 1

    def stats(self) -> dict[str, int]:
        """Return diagnostic counters for scheduling and confirmation.

        Returns
        -------
        dict[str, int]
            Dictionary with counts of scheduled, popped, confirmed, false_alarms,
            rescheduled, and fail_time_evals.
        """
        return {
            "scheduled": self._scheduled,
            "popped": self._popped,
            "confirmed": self._confirmed,
            "false_alarms": self._false_alarms,
            "rescheduled": self._rescheduled,
            "fail_time_evals": self._fail_time_evals,
        }
