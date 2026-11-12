"""Versioned kinetic event heap for HNSW graph topology maintenance.

This module is strictly pure Python standard library and NumPy only.
Manages future certificate invalidation events, enforces monotonic node-versioning
guards to discard stale repair tasks, and tracks active edge certificates.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable

from khnsw.core.events.schema import EventEmitter, EventType


@dataclass(frozen=True)
class DueEvent:
    """An edge invalidation event ready to be serviced at or before simulation time t.

    Attributes
    ----------
    t : float
        Simulation timestamp when the certificate violation occurs.
    edge : tuple[int, int, int]
        Directed edge tuple (u, v, level) where u is source, v is target, and level is layer.
    witness : int
        Node ID of the neighbor witness that invalidated the edge.
    """

    t: float
    edge: tuple[int, int, int]
    witness: int


class KineticEventHeap:
    """Priority queue tracking future kinetic certificate invalidations.

    Features:
    - Chronological event ordering with deterministic tie-breaking (t, u, v, level, push_counter).
    - Monotonic version guards: entries are verified against version_of(node) for source u,
      target v, and witness w. Outdated entries are discarded lazily as stale_skip events.
    - Generation-based edge certificate replacement: pushing a newer certificate for an edge
      immediately invalidates older heap entries for that same edge.
    - Automatic and manual compaction to eliminate stale heap entries in O(n) time.

    Complexity
    ----------
    push : O(log n)
    pop_due : amortized O(log n) per event
    peek_time : amortized O(log n)
    compact : O(n)
    memory : O(n)

    Note
    ----
    Heap entries do not by themselves detect a moved node; staleness detection relies
    entirely on version_of reflecting every coordinate move.
    """

    def __init__(
        self,
        version_of: Callable[[int], int],
        emitter: EventEmitter | None = None,
        compact_ratio: float = 0.5,
        compact_min: int = 1024,
    ) -> None:
        """Initialize kinetic event heap.

        Parameters
        ----------
        version_of : Callable[[int], int]
            Function returning the current monotonic version of a given node ID.
        emitter : EventEmitter | None, default=None
            Optional event emitter for streaming trace events (enqueue, pop, stale_skip).
        compact_ratio : float, default=0.5
            Fraction of known stale entries triggering automatic compaction during push.
        compact_min : int, default=1024
            Minimum physical heap size required before automatic compaction is considered.
        """
        self._version_of = version_of
        self._emitter = emitter
        self._compact_ratio = float(compact_ratio)
        self._compact_min = int(compact_min)

        # Heap entries: (t_fail, u, v, level, push_counter, witness, u_ver, v_ver, w_ver, edge_gen)
        self._heap: list[tuple[float, int, int, int, int, int, int, int, int, int]] = []
        self._push_counter: int = 0

        # Edge generation tracking: (u, v, level) -> current active generation
        self._edge_generations: dict[tuple[int, int, int], int] = {}

        # Set of active live edges currently scheduled in the heap
        self._live_edges: set[tuple[int, int, int]] = set()

        # Index mapping (u, level) -> set of live edges (u, v, level)
        self._source_to_edges: dict[tuple[int, int], set[tuple[int, int, int]]] = {}

    @property
    def raw_size(self) -> int:
        """Physical number of entries in the heap, including stale entries."""
        return len(self._heap)

    def __len__(self) -> int:
        """Number of active live certificates tracked in the heap.

        Entries known to be stale by edge generation or invalidation are excluded.
        Note that node-version staleness is only discovered lazily when popped or peeked.
        """
        return len(self._live_edges)

    def push(
        self,
        edge: tuple[int, int, int],
        t_fail: float,
        witness: int,
        now: float = 0.0,
        reason: str = "certificate_invalidated",
    ) -> None:
        """Schedule a certificate invalidation event for an edge.

        Replaces any earlier certificate for the same edge by bumping that edge's
        generation. Older heap entries for this edge become stale.

        Parameters
        ----------
        edge : tuple[int, int, int]
            Directed edge identifier (u, v, level).
        t_fail : float
            Predicted simulation failure timestamp (must be finite and >= now).
        witness : int
            Node ID of the neighbor witness causing the violation.
        now : float, default=0.0
            Current simulation timestamp.
        reason : str, default="certificate_invalidated"
            Descriptive reason string for enqueue trace event.

        Raises
        ------
        ValueError
            If t_fail is non-finite or t_fail < now.
        """
        t_f = float(t_fail)
        now_f = float(now)

        if not math.isfinite(t_f):
            msg = f"t_fail must be finite, got {t_fail}"
            raise ValueError(msg)
        if t_f < now_f:
            msg = f"t_fail ({t_f}) cannot be earlier than now ({now_f})"
            raise ValueError(msg)

        u, v, level = edge
        u_int = int(u)
        v_int = int(v)
        lvl_int = int(level)
        w_int = int(witness)
        edge_key = (u_int, v_int, lvl_int)

        # Bump generation so earlier certificates for this edge become stale
        gen = self._edge_generations.get(edge_key, 0) + 1
        self._edge_generations[edge_key] = gen
        self._live_edges.add(edge_key)

        # Update source -> edges index
        source_key = (u_int, lvl_int)
        if source_key not in self._source_to_edges:
            self._source_to_edges[source_key] = set()
        self._source_to_edges[source_key].add(edge_key)

        # Snapshot node versions at push time
        u_ver = int(self._version_of(u_int))
        v_ver = int(self._version_of(v_int))
        w_ver = int(self._version_of(w_int))

        self._push_counter += 1

        entry = (
            t_f,
            u_int,
            v_int,
            lvl_int,
            self._push_counter,
            w_int,
            u_ver,
            v_ver,
            w_ver,
            gen,
        )
        heapq.heappush(self._heap, entry)

        if self._emitter is not None:
            self._emitter.emit(
                type=EventType.enqueue,
                node=u_int,
                level=lvl_int,
                version=u_ver,
                payload={"priority": t_f, "reason": reason},
                t=now_f,
            )

        # Automatic compaction trigger
        if self.raw_size >= self._compact_min:
            known_stale = self.raw_size - len(self._live_edges)
            if known_stale / self.raw_size >= self._compact_ratio:
                self.compact()

    def invalidate_edge(self, edge: tuple[int, int, int]) -> None:
        """Invalidate all pending heap entries for an edge by bumping its generation."""
        u, v, level = edge
        edge_key = (int(u), int(v), int(level))
        if edge_key in self._edge_generations:
            self._edge_generations[edge_key] += 1
        self._live_edges.discard(edge_key)

        source_key = (int(u), int(level))
        if source_key in self._source_to_edges:
            self._source_to_edges[source_key].discard(edge_key)
            if not self._source_to_edges[source_key]:
                del self._source_to_edges[source_key]

    def invalidate_source(self, u: int, level: int) -> None:
        """Invalidate all pending certificates for all edges originating from (u, level)."""
        source_key = (int(u), int(level))
        edges = list(self._source_to_edges.get(source_key, set()))
        for edge in edges:
            self.invalidate_edge(edge)

    def _check_entry_validity(
        self,
        edge_key: tuple[int, int, int],
        gen: int,
        u: int,
        v: int,
        witness: int,
        u_ver: int,
        v_ver: int,
        w_ver: int,
    ) -> tuple[bool, int, int, int, str]:
        """Verify if a heap entry is active and up to date against current node versions."""
        # 1. Edge generation and liveness check
        current_gen = self._edge_generations.get(edge_key, 0)
        if gen != current_gen or edge_key not in self._live_edges:
            return False, u, u_ver, self._version_of(u), "stale_edge_generation"

        # 2. Source node u version check
        curr_u_ver = self._version_of(u)
        if curr_u_ver != u_ver:
            return False, u, u_ver, curr_u_ver, "stale_node_version"

        # 3. Target node v version check
        curr_v_ver = self._version_of(v)
        if curr_v_ver != v_ver:
            return False, v, v_ver, curr_v_ver, "stale_neighbor_version"

        # 4. Witness node version check
        curr_w_ver = self._version_of(witness)
        if curr_w_ver != w_ver:
            return False, witness, w_ver, curr_w_ver, "stale_witness_version"

        return True, u, u_ver, curr_u_ver, ""

    def pop_due(self, t_now: float) -> list[DueEvent]:
        """Pop all valid events with failure time t <= t_now in deterministic order.

        Stale entries are discarded and emit stale_skip events. Valid entries are
        removed from the live tracking map and emit pop events.

        Parameters
        ----------
        t_now : float
            Current simulation timestamp boundary (inclusive).

        Returns
        -------
        list[DueEvent]
            Chronologically sorted list of valid DueEvent objects due on or before t_now.
        """
        t_now_f = float(t_now)
        due_events: list[DueEvent] = []

        while self._heap and self._heap[0][0] <= t_now_f:
            entry = heapq.heappop(self._heap)
            (
                t_fail,
                u,
                v,
                level,
                _,
                witness,
                u_ver,
                v_ver,
                w_ver,
                gen,
            ) = entry
            edge_key = (u, v, level)

            is_valid, stale_node, stale_ver, curr_ver, stale_reason = self._check_entry_validity(
                edge_key, gen, u, v, witness, u_ver, v_ver, w_ver
            )

            if is_valid:
                # Remove from live tracking
                self._live_edges.discard(edge_key)
                source_key = (u, level)
                if source_key in self._source_to_edges:
                    self._source_to_edges[source_key].discard(edge_key)
                    if not self._source_to_edges[source_key]:
                        del self._source_to_edges[source_key]

                # Bump generation so duplicate older entries in heap are stale
                self._edge_generations[edge_key] = gen + 1

                if self._emitter is not None:
                    self._emitter.emit(
                        type=EventType.pop,
                        node=u,
                        level=level,
                        version=u_ver,
                        payload={"heap_size": len(self), "priority": t_fail},
                        t=t_now_f,
                    )

                due_events.append(DueEvent(t=t_fail, edge=edge_key, witness=witness))
            else:
                if self._emitter is not None:
                    self._emitter.emit(
                        type=EventType.stale_skip,
                        node=stale_node,
                        level=level,
                        version=stale_ver,
                        payload={"current_version": curr_ver, "reason": stale_reason},
                        t=t_now_f,
                    )

        return due_events

    def peek_time(self, now: float | None = None) -> float | None:
        """Discard stale entries at the top of the heap and return the earliest valid time.

        Parameters
        ----------
        now : float | None, default=None
            Timestamp used when emitting stale_skip events for discarded entries.
            Defaults to the failure timestamp of the popped stale entry if None.

        Returns
        -------
        float | None
            Earliest valid failure timestamp, or None if the heap is empty or contains
            only stale entries.
        """
        while self._heap:
            entry = self._heap[0]
            (
                t_fail,
                u,
                v,
                level,
                _,
                witness,
                u_ver,
                v_ver,
                w_ver,
                gen,
            ) = entry
            edge_key = (u, v, level)

            is_valid, stale_node, stale_ver, curr_ver, stale_reason = self._check_entry_validity(
                edge_key, gen, u, v, witness, u_ver, v_ver, w_ver
            )

            if is_valid:
                return t_fail

            # Discard stale entry from top
            heapq.heappop(self._heap)
            event_t = float(now) if now is not None else float(t_fail)
            if self._emitter is not None:
                self._emitter.emit(
                    type=EventType.stale_skip,
                    node=stale_node,
                    level=level,
                    version=stale_ver,
                    payload={"current_version": curr_ver, "reason": stale_reason},
                    t=event_t,
                )

        return None

    def compact(self) -> None:
        """Rebuild the heap dropping all stale entries, preserving order and determinism."""
        new_heap: list[tuple[float, int, int, int, int, int, int, int, int, int]] = []
        for entry in self._heap:
            _, u, v, level, _, witness, u_ver, v_ver, w_ver, gen = entry
            edge_key = (u, v, level)
            is_valid, _, _, _, _ = self._check_entry_validity(
                edge_key, gen, u, v, witness, u_ver, v_ver, w_ver
            )
            if is_valid:
                new_heap.append(entry)

        heapq.heapify(new_heap)
        self._heap = new_heap
