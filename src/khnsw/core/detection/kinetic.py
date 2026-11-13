"""Kinetic certificate detector coordinating failure prediction and confirmation.

This module is strictly pure Python standard library and NumPy only.
Coordinates KineticEventHeap, CertificateStore, and continuous kinematic trajectories.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

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
        sync_vector: Callable[[int, NDArray[np.float64]], None],
        emitter: EventEmitter | None = None,
        horizon: float = float("inf"),
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
        sync_vector : Callable[[int, NDArray[np.float64]], None]
            Required callback to update a node's stored vector from float64 position
            without bumping index versions. The driver typically stores float32(pos).
        emitter : EventEmitter | None, default=None
            Optional event emitter for streaming trace events.
        horizon : float, default=infinity
            Lookahead time horizon duration. Only events within [t, t + horizon] are scheduled.
        """
        self._index = index
        self._store = store
        self._heap = heap
        self._trajectory_of = trajectory_of
        self._meter = meter
        self._sync_vector = sync_vector
        self._emitter = emitter
        self._horizon = float(horizon)

        # Time window state for window refresh
        self._window_start: float = 0.0
        self._window_end: float = float(horizon) if math.isfinite(horizon) else float("inf")

        # Diagnostic and accounting counters
        self._scheduled: int = 0
        self._popped: int = 0
        self._confirmed: int = 0
        self._false_alarms: int = 0
        self._rescheduled: int = 0
        self._fail_time_evals: int = 0
        self._window_refreshes: int = 0
        self._initial_evals: int = 0
        self._reschedule_evals: int = 0
        self._refresh_evals: int = 0

    def _sync_node(self, node_id: int, t: float) -> None:
        """Synchronize node vector to simulation timestamp t without bumping version."""
        a, b = self._trajectory_of(node_id)
        pos = a + b * float(t)
        self._sync_vector(node_id, pos)

    def sync_nodes(self, nodes: Iterable[int], t: float) -> None:
        """Synchronize stored vectors for the given nodes to simulation timestamp t.

        Applies the injected sync_vector callback without modifying index versions.

        Parameters
        ----------
        nodes : Iterable[int]
            Collection of node IDs to synchronize.
        t : float
            Simulation timestamp.
        """
        t_f = float(t)
        for u in nodes:
            self._sync_node(int(u), t_f)

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
        self._window_start = t0_f
        self._window_end = t0_f + self._horizon if math.isfinite(self._horizon) else float("inf")

        max_lvl = self._index.max_level()
        for level in range(max_lvl + 1):
            for u in self._index.ids():
                if self._index.node_level(u) < level:
                    continue
                for v in self._index.neighbors(u, level):
                    edge = (int(u), int(v), int(level))
                    before_evals = self._fail_time_evals
                    if self._evaluate_and_schedule_edge(edge, t0=t0_f, now=t0_f):
                        self._scheduled += 1
                    self._initial_evals += self._fail_time_evals - before_evals

    def _refresh_window(self) -> None:
        """Advance time window by horizon and reschedule all currently valid edges."""
        win_start = self._window_end
        win_end = win_start + self._horizon
        self._window_start = win_start
        self._window_end = win_end
        self._window_refreshes += 1

        max_lvl = self._index.max_level()
        for level in range(max_lvl + 1):
            for u in self._index.ids():
                if self._index.node_level(u) < level:
                    continue
                for v in self._index.neighbors(u, level):
                    edge = (int(u), int(v), int(level))
                    if self._store.status(edge) == EdgeStatus.VALID.value:
                        self._heap.invalidate_edge(edge)
                        before_evals = self._fail_time_evals
                        self._evaluate_and_schedule_edge(edge, t0=win_start, now=win_start)
                        self._refresh_evals += self._fail_time_evals - before_evals

    def _process_due(self, t_bound: float) -> list[tuple[int, int, int]]:
        """Pop due events up to t_bound and evaluate confirmation or false alarm."""
        due = self._heap.pop_due(t_bound)
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
                # Sync source and outgoing neighbors to current evaluation time in float32
                self._sync_node(u, t_bound)
                for w in nbrs:
                    self._sync_node(w, t_bound)
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
                            "t": t_bound,
                        },
                    )
            else:
                # Float64 prediction not confirmed in float32 (near-tie / discrepancy / transient)
                self._false_alarms += 1
                eps = 1e-9 * max(1.0, abs(t_bound))
                before_evals = self._fail_time_evals
                if self._evaluate_and_schedule_edge(edge, t0=t_bound + eps, now=t_bound):
                    self._rescheduled += 1
                self._reschedule_evals += self._fail_time_evals - before_evals

        return confirmed_edges

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
        confirmed_edges: list[tuple[int, int, int]] = []

        if math.isfinite(self._horizon):
            while t_curr > self._window_end:
                confirmed_edges.extend(self._process_due(self._window_end))
                self._refresh_window()

        confirmed_edges.extend(self._process_due(t_curr))
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
            before_evals = self._fail_time_evals
            if self._evaluate_and_schedule_edge(edge, t0=t_curr, now=t_curr):
                self._rescheduled += 1
            self._reschedule_evals += self._fail_time_evals - before_evals

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
            before_evals = self._fail_time_evals
            if self._evaluate_and_schedule_edge(edge, t0=t_curr, now=t_curr):
                self._rescheduled += 1
            self._reschedule_evals += self._fail_time_evals - before_evals

    def stats(self) -> dict[str, int]:
        """Return diagnostic counters for scheduling and confirmation.

        Returns
        -------
        dict[str, int]
            Dictionary with counts of scheduled, popped, confirmed, false_alarms,
            rescheduled, fail_time_evals, window_refreshes, initial_evals,
            reschedule_evals, and refresh_evals.
        """
        return {
            "scheduled": self._scheduled,
            "popped": self._popped,
            "confirmed": self._confirmed,
            "false_alarms": self._false_alarms,
            "rescheduled": self._rescheduled,
            "fail_time_evals": self._fail_time_evals,
            "window_refreshes": self._window_refreshes,
            "initial_evals": self._initial_evals,
            "reschedule_evals": self._reschedule_evals,
            "refresh_evals": self._refresh_evals,
        }
