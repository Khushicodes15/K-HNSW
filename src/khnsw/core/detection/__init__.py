"""Kinetic certificates and topology invalidation detection."""

from khnsw.core.detection.certificates import (
    dist_quadratic,
    edge_failure_time,
    first_violation_time,
    is_violated_static,
    quad_sub,
    real_roots,
)

__all__ = [
    "dist_quadratic",
    "edge_failure_time",
    "first_violation_time",
    "is_violated_static",
    "quad_sub",
    "real_roots",
]
