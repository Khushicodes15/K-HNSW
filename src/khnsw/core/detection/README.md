# Detection & Kinetic Certificates (`src/khnsw/core/detection`)

## Purpose
Monitors kinetic validity of graph edges. When nodes drift, kinetic certificates verify whether existing edges still satisfy Navigable Small World and relative neighborhood heuristic conditions or have failed.

## Owner Role
**Member 3 (Update/Repair Engine - Repo Owner)**

## What Goes Here
- Kinetic certificate representations and evaluation functions.
- Relative distance condition verifiers (e.g. triangle inequality or angle criteria).
- Early invalidation triggers when a node's displacement exceeds safety radii.
- Invalidation event generation (`cert_fail`) forwarded to the repair scheduler.

