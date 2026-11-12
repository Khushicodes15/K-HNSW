"""Event schema and sink interfaces for K-HNSW kinetic graph simulation traces.

This module is strictly pure Python standard library only.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

__all__ = [
    "Event",
    "EventEmitter",
    "EventSink",
    "EventType",
    "JsonlSink",
    "ListSink",
    "NullSink",
]


class EventType(str, Enum):
    """Enumeration of kinetic HNSW trace event types."""

    run_start = "run_start"
    move = "move"
    cert_fail = "cert_fail"
    enqueue = "enqueue"
    pop = "pop"
    stale_skip = "stale_skip"
    edge_add = "edge_add"
    edge_remove = "edge_remove"
    search = "search"
    metric = "metric"
    run_end = "run_end"


@dataclass(frozen=True)
class Event:
    """Immutable trace event recording graph state transitions, repairs, or metrics."""

    run_id: str
    seq: int
    t: float
    type: EventType
    node: int | None = None
    level: int | None = None
    version: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a JSON-serializable dictionary.

        Raises
        ------
        ValueError
            If payload contains NaN or infinite floating-point values.
        """
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "seq": self.seq,
            "t": float(self.t),
            "type": self.type.value if isinstance(self.type, EventType) else str(self.type),
            "node": self.node,
            "level": self.level,
            "version": self.version,
            "payload": self.payload,
        }
        # Validate that payload has no NaN or infinite values
        json.dumps(data, allow_nan=False)
        return data


@runtime_checkable
class EventSink(Protocol):
    """Protocol for receiving and persisting or routing kinetic simulation events."""

    def emit(self, event: Event) -> None:
        """Emit a trace event to the sink.

        Parameters
        ----------
        event : Event
            The kinetic trace event to emit.
        """
        ...


class NullSink:
    """Event sink that discards all emitted events."""

    def emit(self, event: Event) -> None:
        """Discard the given event."""
        pass


class ListSink:
    """In-memory event sink that appends emitted events to a list."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        """Append event to the internal in-memory list."""
        self.events.append(event)


class JsonlSink:
    """Event sink that keeps an open file handle and writes line-delimited JSON.

    Parameters
    ----------
    path : str | Path
        Target file path for appending trace events.
    truncate : bool, default=False
        If True, truncates any existing file. If False, raises FileExistsError
        when the target file already exists.
    """

    def __init__(self, path: str | Path, truncate: bool = False) -> None:
        self.path = Path(path)
        if self.path.exists() and not truncate:
            raise FileExistsError(
                f"JsonlSink target file already exists and truncate=False: {self.path}"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", encoding="utf-8")

    def emit(self, event: Event) -> None:
        """Write the serialized event as a line using allow_nan=False."""
        if self._file.closed:
            raise ValueError("I/O operation on closed JsonlSink.")
        line = json.dumps(event.to_dict(), allow_nan=False)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()

    def __enter__(self) -> JsonlSink:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class EventEmitter:
    """Event generator owning run_id, sequence counter, and simulation clock.

    Parameters
    ----------
    run_id : str
        Unique identifier for the simulation run.
    sink : EventSink
        Sink destination for emitted events.
    clock : Callable[[], float] | None, default=None
        Time provider returning current simulation time t. If None, uses time.time.
    """

    def __init__(
        self,
        run_id: str,
        sink: EventSink,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._run_id = run_id
        self._sink = sink
        self._clock = clock if clock is not None else time.time
        self._seq = 0

    @property
    def run_id(self) -> str:
        """Return the run identifier."""
        return self._run_id

    @property
    def seq(self) -> int:
        """Return current sequence index counter."""
        return self._seq

    @property
    def sink(self) -> EventSink:
        """Return target event sink."""
        return self._sink

    @property
    def clock(self) -> Callable[[], float]:
        """Return current clock callable."""
        return self._clock

    def emit(
        self,
        type: EventType,
        node: int | None = None,
        level: int | None = None,
        version: int | None = None,
        payload: dict[str, Any] | None = None,
        t: float | None = None,
    ) -> Event:
        """Construct, sequence, and emit an Event to the underlying sink."""
        timestamp = float(self._clock()) if t is None else float(t)
        event = Event(
            schema_version=1,
            run_id=self._run_id,
            seq=self._seq,
            t=timestamp,
            type=type,
            node=node,
            level=level,
            version=version,
            payload=payload if payload is not None else {},
        )
        self._seq += 1
        self._sink.emit(event)
        return event
