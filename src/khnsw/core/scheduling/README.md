# Scheduling & Event Heaps (`src/khnsw/core/scheduling`)

## Purpose
Maintains the repair priority queue. Tracks pending edge repairs, ranks them by urgency (layer level, certificate violation magnitude, degree deficits), and discards stale queue entries.

## Owner Role
**Member 3 (Update/Repair Engine - Repo Owner)**

## What Goes Here
- Priority queue data structure wrapping `heapq` with versioned node keys.
- Monotonic version counters per node to detect and skip stale updates (`stale_skip`).
- Priority scoring functions weighting upper hierarchy layers higher than lower layers.
- Budget accounting logic to limit repair work per time step.

