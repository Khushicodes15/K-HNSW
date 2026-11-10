"""FastAPI entry point for K-HNSW service."""

from fastapi import FastAPI

app = FastAPI(
    title="K-HNSW API",
    description="Kinetic Hierarchical Navigable Small World Research Service",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Service health probe endpoint."""
    return {"status": "healthy"}
