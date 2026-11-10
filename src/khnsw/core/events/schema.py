"""Event schema and sink interfaces for K-HNSW kinetic graph simulation traces.

This module is strictly pure Python standard library only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


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
        """Convert the event to a JSON-serializable dictionary."""
        return {
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
    """Event sink that writes line-delimited JSON objects to disk."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Event) -> None:
        """Append the serialized event as a line in the target JSONL file."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
