# Scheduling & Event Heaps (`src/khnsw/core/scheduling`)

## Purpose
Manages future kinetic certificate invalidations and coordinates topology repair scheduling. Tracks pending edge failures in chronological order, enforces monotonic node-version guards to filter obsolete work, and provides deterministic tie-breaking.

## Owner Role
**Member 3 (Update/Repair Engine - Repo Owner - Khushi Dhona)**

## Components

### 1. `KineticEventHeap` (`event_heap.py`)
A specialized priority queue wrapping standard library `heapq` that maintains future certificate failures:
- **`DueEvent`**: A frozen dataclass `(t: float, edge: tuple[int, int, int], witness: int)`.
- **Chronological Ordering**: Min-heap ordered by failure timestamp `t`, broken deterministically by `(u, v, level, push_counter)`.
- **Generation-Based Invalidation**: Each edge `(u, v, level)` maintains a monotonic generation counter. Re-pushing or explicitly invalidating an edge bumps its generation, immediately rendering earlier heap entries for that edge stale.

### 2. `NodeDisplacementTracker` (`tracker.py`)
Node-level kinematic displacement tracking and lazy reinsertion:
- **Anchor $p_i$ and Radius $r_i$**: For each node $i$, $p_i$ is its spatial vector upon insertion/reinsertion, and $r_i = \max_{j \in N_0(i)} \|p_i - x_j\|_2$ is the distance to its farthest level-0 neighbor.
- **Staleness Bound**: Node $i$ is stale at timestamp $t$ if and only if $\|x_i(t) - p_i\|_2 > \theta \cdot r_i$.
- **Targeted Check**: Only nodes with non-zero velocity are checked at each step ($1$ distance computation per moving node).
- **Maintenance Policy**: On staleness, reinserts node $i$ with fixed hierarchy level (with optional in-neighbor pre-repair), and updates anchor $p_i$ and radius $r_i$ to current topology.

### 3. `BudgetedPriorityTracker` (`tracker.py`)
Budget-constrained priority reinsertion:
- At each step, evaluates score $s_i = \|x_i(t) - p_i\|_2 / r_i$ for moving nodes.
- Reinserts at most the top $B$ nodes with highest departure scores.
- Resets anchor $p_i \leftarrow x_i(t)$ and recomputes $r_i$ for serviced nodes.

### 4. `MaintenancePolicy` (`policy.py`)
Unified public interface for downstream modules (Members 4 and 5):
- Pluggable policies: `"none"`, `"periodic"`, `"round_robin"`, `"lazy"`, `"budgeted_priority"`.
- Uniform entry point: `step(index, positions, t=0.0, moving_nodes=None) -> list[int]`.
- Enforces fixed hierarchy levels on reinsertion and meters costs under `CostMeter`.
- Contract-compliant: Operates cleanly without emitting uncontracted events (contract request noted).

### The Stale-Entry Rule
An entry for directed edge `(u, v, level)` with witness `w` is **VALID** if and only if:
1. Its recorded edge generation equals the active generation for `(u, v, level)`, AND
2. The edge is currently marked active in `live_edges`, AND
3. `version_of(u)` equals the recorded version of $u$ at push time, AND
4. `version_of(v)` equals the recorded version of $v$ at push time, AND
5. `version_of(w)` equals the recorded version of $w$ at push time.

If any of these conditions fails (e.g., $u$, $v$, or $w$ moves, incrementing its version counter; or the edge is re-pushed or severed), the entry is **STALE**. Stale entries are lazily discarded when popped or peeked, emitting a `stale_skip` trace event.

### Compaction Policy
Stale entries can be pruned in $O(n)$ time via `compact()`, which reconstructs the heap retaining only currently valid entries while strictly preserving original insertion sequence numbers and deterministic ordering. Automatic compaction triggers during `push` when `raw_size >= compact_min` and the known-stale ratio exceeds `compact_ratio`.

### Future Extension: Score-Ordered Repair Queue
Note: A separate score-ordered repair queue (ranking repair operations by hierarchy layer priority, degree deficit, and certificate violation magnitude within distance computation budgets) will use the same monotonic node-versioning pattern established here.

## Complexity
- **`push`**: $O(\log n)$
- **`pop_due`**: Amortized $O(\log n)$ per event
- **`peek_time`**: Amortized $O(\log n)$
- **`compact`**: $O(n)$
- **Memory**: $O(n)$
