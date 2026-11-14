"""Topology repair engines executing maintenance under kinetic node updates."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from khnsw.core.detection.cost import CostMeter
from khnsw.core.events.schema import EventEmitter, EventSink
from khnsw.core.interfaces import MutableGraphIndex
from khnsw.core.repair.apply import apply_plan
from khnsw.core.repair.loop import repair_loop
from khnsw.core.repair.plan import mnru_plan, native_plan

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from khnsw.core.detection.certificate_store import CertificateStore
    from khnsw.core.detection.kinetic import KineticDetector


class KHNSWRepairEngine:
    """Certificate-driven iterative repair engine for K-HNSW graphs."""

    def __init__(
        self,
        store: CertificateStore | None = None,
        *,
        search_layer: Callable[..., Any] | None = None,
        detector: KineticDetector | None = None,
        time_source: Callable[[], float] | float | None = None,
        max_rounds: int = 10,
        ef: int | None = None,
        run_id: str = "khnsw_repair",
    ) -> None:
        self._store = store
        self._search_layer = search_layer
        self._detector = detector
        self._time_source = time_source
        self._max_rounds = max_rounds
        self._ef = ef
        self._run_id = run_id
        self._pending_moves: list[tuple[int, float]] = []
        self._moved_since_certified: set[int] = set()

    def notify_moved(self, node: int, t: float = 0.0) -> None:
        """Register a node that underwent spatial displacement at simulation timestamp t."""
        node_id = int(node)
        t_val = float(t)
        self._pending_moves.append((node_id, t_val))
        self._moved_since_certified.add(node_id)

    def on_move(
        self,
        index: MutableGraphIndex,
        node_id: int,
        old_vector: NDArray[np.float32],
        new_vector: NDArray[np.float32],
        sink: EventSink,
    ) -> None:
        """Handle single node movement notification fulfilling Repairer protocol."""
        t_curr = self._resolve_time()
        self.notify_moved(node_id, t=t_curr)
        if self._store is not None:
            meter = CostMeter()
            self._store.on_move(index, int(node_id), meter)
            self._moved_since_certified.discard(int(node_id))

    def _resolve_time(self) -> float:
        """Evaluate current simulation timestamp from configured time source."""
        if callable(self._time_source):
            return float(self._time_source())
        if self._time_source is not None:
            return float(self._time_source)
        if self._pending_moves:
            return self._pending_moves[-1][1]
        return 0.0

    def run_repairs(
        self,
        index: MutableGraphIndex,
        budget: float,
        sink: EventSink,
    ) -> dict[str, Any]:
        """Execute scheduled topological repairs within an allotted computational budget."""
        budget_cap = int(math.floor(budget))
        if budget_cap <= 0:
            return {
                "engine": "KHNSWRepairEngine",
                "rounds": 0,
                "edges_repaired": 0,
                "still_violated": self._collect_still_violated(index),
                "still_violated_count": len(self._collect_still_violated(index)),
                "unrepaired_nodes": [u for u, _ in self._pending_moves],
                "budget_used": {
                    "search": 0,
                    "planning": 0,
                    "store": 0,
                    "detector": 0,
                    "reverse_link": 0,
                },
                "budget_consumed": 0,
                "stopped_reason": "budget",
                "moved_in_drift": 0,
            }

        # Lazily instantiate and build CertificateStore if needed
        if self._store is None:
            if self._detector is not None:
                self._store = getattr(self._detector, "_store", None)
            if self._store is None:
                from khnsw.core.detection.certificate_store import CertificateStore

                self._store = CertificateStore()
                self._store.build(index, CostMeter())

        t_curr = self._resolve_time()

        # Check for confirmed detector failures
        detector_failures: list[tuple[int, int, int]] | None = None
        if self._detector is not None:
            detector_failures = self._detector.advance(t_curr)

        # In drift maintenance, extract moved nodes from confirmed failures:
        # candidate nodes are {u, v, witness} whose trajectory has a non-zero velocity.
        # Deduplicated per step.
        drift_moved_nodes: list[int] = []
        if self._detector is not None and detector_failures:
            traj_fn = getattr(self._detector, "trajectory_of", None)
            if traj_fn is None:
                traj_fn = getattr(self._detector, "_trajectory_of", None)
            if traj_fn is not None:
                candidates: set[int] = set()
                for edge in detector_failures:
                    u, v, _level = edge
                    candidates.add(int(u))
                    candidates.add(int(v))
                    witness = None
                    if self._store is not None:
                        witness = self._store.witness(edge)
                    if witness is not None:
                        candidates.add(int(witness))
                drift_set: set[int] = set()
                for nid in candidates:
                    try:
                        _, vel = traj_fn(nid)
                        if np.any(np.asarray(vel) != 0.0):
                            drift_set.add(int(nid))
                    except Exception:
                        pass
                drift_moved_nodes = sorted(drift_set)

        # Check if there is anything to do
        has_pending = bool(self._pending_moves)
        has_failures = bool(detector_failures)
        has_violated = bool(self._collect_still_violated(index))

        if not has_pending and not has_failures and not has_violated:
            return {
                "engine": "KHNSWRepairEngine",
                "rounds": 0,
                "edges_repaired": 0,
                "still_violated": [],
                "still_violated_count": 0,
                "unrepaired_nodes": [],
                "budget_used": {
                    "search": 0,
                    "planning": 0,
                    "store": 0,
                    "detector": 0,
                    "reverse_link": 0,
                },
                "budget_consumed": 0,
                "stopped_reason": "nothing_to_do",
                "moved_in_drift": 0,
            }

        emitter = EventEmitter(run_id=self._run_id, sink=sink)
        cumulative_budget: dict[str, int] = {
            "search": 0,
            "planning": 0,
            "store": 0,
            "detector": 0,
            "reverse_link": 0,
        }
        total_rounds = 0
        total_edges_repaired = 0
        stopped_reason = ""
        repaired_drift_count = 0
        unrepaired_drift_nodes: list[int] = []

        # Synchronize store for any moved nodes not yet certified
        for node_id in list(self._moved_since_certified):
            if sum(cumulative_budget.values()) >= budget_cap:
                stopped_reason = "budget"
                break
            meter_store = CostMeter()
            self._store.on_move(index, node_id, meter_store)
            cumulative_budget["store"] += meter_store.count
            self._moved_since_certified.discard(node_id)

        if stopped_reason != "budget":
            if self._pending_moves:
                moves_to_process = list(self._pending_moves)
                for item in moves_to_process:
                    node_id, t_val = item
                    rem_budget = budget_cap - sum(cumulative_budget.values())
                    if rem_budget <= 0:
                        stopped_reason = "budget"
                        break

                    res = repair_loop(
                        index=index,
                        store=self._store,
                        detector=self._detector,
                        moved_node=node_id,
                        failures=detector_failures,
                        t=t_val,
                        search_layer=self._search_layer,
                        budget=rem_budget,
                        max_rounds=self._max_rounds,
                        ef=self._ef,
                        emitter=emitter,
                    )
                    detector_failures = None
                    total_rounds += res.rounds
                    total_edges_repaired += res.edges_repaired
                    for k in cumulative_budget:
                        cumulative_budget[k] += res.budget_used.get(k, 0)

                    if res.stopped_reason == "budget":
                        stopped_reason = "budget"
                        break
                    elif res.stopped_reason == "max_rounds":
                        stopped_reason = "max_rounds"
                        self._pending_moves.remove(item)
                    else:
                        self._pending_moves.remove(item)
            elif drift_moved_nodes:
                moves_to_process = list(drift_moved_nodes)
                unrepaired_drift_nodes = list(drift_moved_nodes)
                for node_id in moves_to_process:
                    rem_budget = budget_cap - sum(cumulative_budget.values())
                    if rem_budget <= 0:
                        stopped_reason = "budget"
                        break

                    res = repair_loop(
                        index=index,
                        store=self._store,
                        detector=self._detector,
                        moved_node=node_id,
                        failures=detector_failures,
                        t=t_curr,
                        search_layer=self._search_layer,
                        budget=rem_budget,
                        max_rounds=self._max_rounds,
                        ef=self._ef,
                        emitter=emitter,
                    )
                    detector_failures = None
                    total_rounds += res.rounds
                    total_edges_repaired += res.edges_repaired
                    for k in cumulative_budget:
                        cumulative_budget[k] += res.budget_used.get(k, 0)

                    if res.stopped_reason == "budget":
                        stopped_reason = "budget"
                        break
                    elif res.stopped_reason == "max_rounds":
                        stopped_reason = "max_rounds"
                        unrepaired_drift_nodes.remove(node_id)
                        repaired_drift_count += 1
                    else:
                        unrepaired_drift_nodes.remove(node_id)
                        repaired_drift_count += 1

                if stopped_reason not in {"budget", "max_rounds"}:
                    still_v = self._collect_still_violated(index)
                    if still_v:
                        rem_budget = budget_cap - sum(cumulative_budget.values())
                        if rem_budget <= 0:
                            stopped_reason = "budget"
                        else:
                            res = repair_loop(
                                index=index,
                                store=self._store,
                                detector=self._detector,
                                failures=None,
                                t=t_curr,
                                search_layer=self._search_layer,
                                budget=rem_budget,
                                max_rounds=self._max_rounds,
                                ef=self._ef,
                                emitter=emitter,
                            )
                            total_rounds += res.rounds
                            total_edges_repaired += res.edges_repaired
                            for k in cumulative_budget:
                                cumulative_budget[k] += res.budget_used.get(k, 0)
                            stopped_reason = res.stopped_reason
            elif detector_failures or has_violated:
                rem_budget = budget_cap - sum(cumulative_budget.values())
                if rem_budget <= 0:
                    stopped_reason = "budget"
                else:
                    res = repair_loop(
                        index=index,
                        store=self._store,
                        detector=self._detector,
                        failures=detector_failures,
                        t=t_curr,
                        search_layer=self._search_layer,
                        budget=rem_budget,
                        max_rounds=self._max_rounds,
                        ef=self._ef,
                        emitter=emitter,
                    )
                    total_rounds += res.rounds
                    total_edges_repaired += res.edges_repaired
                    for k in cumulative_budget:
                        cumulative_budget[k] += res.budget_used.get(k, 0)
                    stopped_reason = res.stopped_reason

        if stopped_reason == "":
            stopped_reason = "converged"

        still_violated = self._collect_still_violated(index)

        return {
            "engine": "KHNSWRepairEngine",
            "rounds": total_rounds,
            "edges_repaired": total_edges_repaired,
            "still_violated": still_violated,
            "still_violated_count": len(still_violated),
            "unrepaired_nodes": [u for u, _ in self._pending_moves] + unrepaired_drift_nodes,
            "budget_used": cumulative_budget,
            "budget_consumed": sum(cumulative_budget.values()),
            "stopped_reason": stopped_reason,
            "moved_in_drift": repaired_drift_count,
        }

    def _collect_still_violated(self, index: MutableGraphIndex) -> list[tuple[int, int, int]]:
        """Collect all currently violated edges from the CertificateStore."""
        if self._store is None:
            return []
        violated: list[tuple[int, int, int]] = []
        for u in index.ids():
            for lvl in range(index.node_level(u) + 1):
                for v in index.neighbors(u, lvl):
                    edge = (u, v, lvl)
                    if self._store._edge_status.get(edge) == "VIOLATED":
                        violated.append(edge)
        return sorted(violated)


class HnswlibStyleEngine:
    """Emulation of hnswlib updatePoint repair engine."""

    def __init__(
        self,
        *,
        time_source: Callable[[], float] | float | None = None,
        ef: int | None = None,
        run_id: str = "hnswlib_repair",
    ) -> None:
        self._time_source = time_source
        self._ef = ef
        self._run_id = run_id
        self._pending_moves: list[int] = []

    def notify_moved(self, node: int, t: float = 0.0) -> None:
        """Register a moved node ID for update-style neighbor repair."""
        self._pending_moves.append(int(node))

    def on_move(
        self,
        index: MutableGraphIndex,
        node_id: int,
        old_vector: NDArray[np.float32],
        new_vector: NDArray[np.float32],
        sink: EventSink,
    ) -> None:
        """Handle node movement fulfilling Repairer protocol."""
        self.notify_moved(node_id)

    def run_repairs(
        self,
        index: MutableGraphIndex,
        budget: float,
        sink: EventSink,
    ) -> dict[str, Any]:
        """Execute update-style repairs for all accumulated moved nodes."""
        budget_cap = int(math.floor(budget))
        if budget_cap <= 0:
            return {
                "engine": "HnswlibStyleEngine",
                "rounds": 0,
                "edges_repaired": 0,
                "still_violated": [],
                "still_violated_count": 0,
                "unrepaired_nodes": list(self._pending_moves),
                "budget_used": {"search": 0, "planning": 0, "store": 0, "detector": 0},
                "budget_consumed": 0,
                "stopped_reason": "budget",
            }

        if not self._pending_moves:
            return {
                "engine": "HnswlibStyleEngine",
                "rounds": 0,
                "edges_repaired": 0,
                "still_violated": [],
                "still_violated_count": 0,
                "unrepaired_nodes": [],
                "budget_used": {"search": 0, "planning": 0, "store": 0, "detector": 0},
                "budget_consumed": 0,
                "stopped_reason": "nothing_to_do",
            }

        emitter = EventEmitter(run_id=self._run_id, sink=sink)
        meter = CostMeter()
        rounds = 0
        edges_repaired = 0
        stopped_reason = ""

        moves_to_process = list(self._pending_moves)
        for node_id in moves_to_process:
            if meter.count >= budget_cap:
                stopped_reason = "budget"
                break

            plan = native_plan(index, moved=node_id, ef=self._ef, meter=meter)
            # Safe halting: never apply mid-apply_plan if planning exceeded budget
            if meter.count > budget_cap:
                stopped_reason = "budget"
                break

            apply_plan(index, plan, emitter=emitter, store=None, meter=None)
            rounds += 1
            edges_repaired += len(plan.rewrites)
            self._pending_moves.remove(node_id)

        if stopped_reason == "":
            stopped_reason = "converged"

        return {
            "engine": "HnswlibStyleEngine",
            "rounds": rounds,
            "edges_repaired": edges_repaired,
            "still_violated": [],
            "still_violated_count": 0,
            "unrepaired_nodes": list(self._pending_moves),
            "budget_used": {
                "search": 0,
                "planning": meter.count,
                "store": 0,
                "detector": 0,
            },
            "budget_consumed": meter.count,
            "stopped_reason": stopped_reason,
        }


class MNRUStyleEngine:
    """Emulation of MN-RU (Mutual Neighbors Repair Update) repair engine."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        scope: str = "one_hop",
        time_source: Callable[[], float] | float | None = None,
        ef: int | None = None,
        run_id: str = "mnru_repair",
    ) -> None:
        self._alpha = float(alpha)
        self._scope = str(scope)
        self._time_source = time_source
        self._ef = ef
        self._run_id = run_id
        self._pending_moves: list[int] = []

    def notify_moved(self, node: int, t: float = 0.0) -> None:
        """Register a moved node ID for MN-RU mutual neighbor repair."""
        self._pending_moves.append(int(node))

    def on_move(
        self,
        index: MutableGraphIndex,
        node_id: int,
        old_vector: NDArray[np.float32],
        new_vector: NDArray[np.float32],
        sink: EventSink,
    ) -> None:
        """Handle node movement fulfilling Repairer protocol."""
        self.notify_moved(node_id)

    def run_repairs(
        self,
        index: MutableGraphIndex,
        budget: float,
        sink: EventSink,
    ) -> dict[str, Any]:
        """Execute MN-RU repairs for all accumulated moved nodes."""
        budget_cap = int(math.floor(budget))
        if budget_cap <= 0:
            return {
                "engine": "MNRUStyleEngine",
                "rounds": 0,
                "edges_repaired": 0,
                "still_violated": [],
                "still_violated_count": 0,
                "unrepaired_nodes": list(self._pending_moves),
                "budget_used": {"search": 0, "planning": 0, "store": 0, "detector": 0},
                "budget_consumed": 0,
                "stopped_reason": "budget",
            }

        if not self._pending_moves:
            return {
                "engine": "MNRUStyleEngine",
                "rounds": 0,
                "edges_repaired": 0,
                "still_violated": [],
                "still_violated_count": 0,
                "unrepaired_nodes": [],
                "budget_used": {"search": 0, "planning": 0, "store": 0, "detector": 0},
                "budget_consumed": 0,
                "stopped_reason": "nothing_to_do",
            }

        emitter = EventEmitter(run_id=self._run_id, sink=sink)
        meter = CostMeter()
        rounds = 0
        edges_repaired = 0
        stopped_reason = ""

        moves_to_process = list(self._pending_moves)
        for node_id in moves_to_process:
            if meter.count >= budget_cap:
                stopped_reason = "budget"
                break

            plan = mnru_plan(
                index,
                moved=node_id,
                alpha=self._alpha,
                scope=self._scope,
                ef=self._ef,
                meter=meter,
            )
            # Safe halting: never apply mid-apply_plan if planning exceeded budget
            if meter.count > budget_cap:
                stopped_reason = "budget"
                break

            apply_plan(index, plan, emitter=emitter, store=None, meter=None)
            rounds += 1
            edges_repaired += len(plan.rewrites)
            self._pending_moves.remove(node_id)

        if stopped_reason == "":
            stopped_reason = "converged"

        return {
            "engine": "MNRUStyleEngine",
            "rounds": rounds,
            "edges_repaired": edges_repaired,
            "still_violated": [],
            "still_violated_count": 0,
            "unrepaired_nodes": list(self._pending_moves),
            "budget_used": {
                "search": 0,
                "planning": meter.count,
                "store": 0,
                "detector": 0,
            },
            "budget_consumed": meter.count,
            "stopped_reason": stopped_reason,
        }
