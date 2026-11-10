"""Smoke tests verifying package structure, interface protocols, and API health."""

from __future__ import annotations

from fastapi.testclient import TestClient

import khnsw
from khnsw.adapters.faiss_adapter import FaissAdapter
from khnsw.adapters.hnswlib_adapter import HnswlibAdapter
from khnsw.api.main import app
from khnsw.core.events.schema import (
    Event,
    EventEmitter,
    EventSink,
    EventType,
    JsonlSink,
    ListSink,
    NullSink,
)
from khnsw.core.interfaces import (
    ANNIndex,
    GraphIndex,
    LinearTrajectory,
    MovementModel,
    MutableGraphIndex,
    Repairer,
)


def test_package_metadata() -> None:
    """Verify package version and top-level namespace."""
    assert hasattr(khnsw, "__version__")
    assert khnsw.__version__ == "0.1.0"


def test_protocols_importable() -> None:
    """Verify that all core interfaces and event classes are importable protocols."""
    assert isinstance(ANNIndex, type)
    assert isinstance(GraphIndex, type)
    assert isinstance(MutableGraphIndex, type)
    assert isinstance(MovementModel, type)
    assert isinstance(LinearTrajectory, type)
    assert isinstance(Repairer, type)
    assert isinstance(EventSink, type)

    # Event types
    assert EventType.run_start.value == "run_start"
    assert EventType.move.value == "move"
    assert EventType.edge_add.value == "edge_add"
    assert EventType.run_end.value == "run_end"

    # Sinks and emitters
    assert issubclass(NullSink, EventSink)
    assert issubclass(ListSink, EventSink)
    assert issubclass(JsonlSink, EventSink)
    assert isinstance(EventEmitter, type)
    assert isinstance(Event, type)


def test_adapters_instantiable_stubs() -> None:
    """Verify external adapter stub classes exist with correct dim properties."""
    hnsw = HnswlibAdapter(dim=64)
    assert hnsw.dim == 64
    assert isinstance(hnsw, ANNIndex)

    faiss = FaissAdapter(dim=32, m=16)
    assert faiss.dim == 32
    assert isinstance(faiss, ANNIndex)


def test_fastapi_health_endpoint() -> None:
    """Verify that FastAPI /health endpoint returns HTTP 200 and healthy status."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
