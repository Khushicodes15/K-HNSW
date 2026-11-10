# Graph Repair (`src/khnsw/core/repair`)

## Purpose
Implements concrete repair routines that rewire degraded graph edges, replace broken certificates, enforce maximum degree constraints via heuristic pruning, and maintain navigability and connectivity under kinetic updates.

## Owner Role
**Member 3 (Update/Repair Engine - Repo Owner)**

## What Goes Here
- Implementations of the `Repairer` protocol:
  - Local 1-hop and 2-hop neighborhood exploration.
  - Hierarchy-aware repair prioritizing upper routing layers.
  - Heuristic edge pruning maintaining angular diversity.
- Symmetric edge reconciliation and reverse-edge maintenance.
- Invariant repair (e.g. restoring connectivity for zero in-degree nodes).
- Trace event emission (`edge_add`, `edge_remove`) via `EventSink`.

