"""Kinetic certificates and topology invalidation detection."""

from khnsw.core.detection.affected_scan import (
    affected_after_move,
    affected_after_neighbor_change,
)
from khnsw.core.detection.certificate_store import (
    CertificateStore,
    EdgeStatus,
    Transition,
)
from khnsw.core.detection.certificates import (
    dist_quadratic,
    edge_failure_time,
    first_violation_time,
    is_violated_static,
    quad_sub,
    real_roots,
)
from khnsw.core.detection.cost import CostMeter

__all__ = [
    "CertificateStore",
    "CostMeter",
    "EdgeStatus",
    "Transition",
    "affected_after_move",
    "affected_after_neighbor_change",
    "dist_quadratic",
    "edge_failure_time",
    "first_violation_time",
    "is_violated_static",
    "quad_sub",
    "real_roots",
]
