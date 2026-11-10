# K-HNSW Kinetic Trace Event Specification (`docs/contracts/events.md`)

This document defines the semantics, field requirements, and payload contracts for all 11 event types emitted to `EventSink` by the K-HNSW kinetic graph simulation and repair engine.

## Base Envelope Contract
Every event adheres to the base envelope defined in [`event.schema.json`](event.schema.json):
- `schema_version` (integer): Always `1`.
- `run_id` (string): Unique identifier for the simulation or experiment run.
- `seq` (integer $\ge 0$): Monotonically increasing sequence number managed by `EventEmitter`.
- `t` (float $\ge 0.0$): Simulation or wall-clock timestamp.
- `type` (string enum): One of the 11 recognized event types.
- `node` (integer or null): Context-dependent node identifier.
- `level` (integer $\ge 0$ or null): Hierarchy layer level index.
- `version` (integer $\ge 0$ or null): Monotonic update counter of the target node.
- `payload` (object): Event-specific dictionary. All floating-point numbers must be finite (`allow_nan=False`). In test fixtures, payloads include `{"example": true}`.

---

## Per-Event-Type Contract Specifications

### 1. `run_start`
- **Purpose**: Marks the start of a simulation run and records initial parameters.
- **Envelope Fields**: `node = null`, `level = null`, `version = null`.
- **Payload Schema**:
  - `total_nodes` (integer, required): Number of initial indexed vectors.
  - `dim` (integer, optional): Vector dimensionality.
  - `dataset` (string, optional): Dataset name or distribution.

### 2. `move`
- **Purpose**: Spatial position update of an existing node in the index.
- **Envelope Fields**:
  - `node` (integer, required): ID of the node that moved.
  - `level` (integer, optional): Base layer (usually `0`).
  - `version` (integer, required): New version of the node (incremented by 1).
- **Payload Schema**:
  - `delta_norm` (number, required): Euclidean distance of the movement displacement.
  - `new_vector` (array of numbers, optional): Updated coordinates.

### 3. `cert_fail`
- **Purpose**: A kinetic certificate associated with an edge fails due to coordinate displacement.
- **Envelope Fields**:
  - `node` (integer, required): Source node ID whose certificate failed.
  - `level` (integer, required): Hierarchy layer of the invalidated connection.
  - `version` (integer, required): Current version of the moving node.
- **Payload Schema**:
  - `neighbor_id` (integer, required): Target node identifier of the invalidated edge.
  - `reason` (string, required): Invalidation reason (e.g. `"relative_distance_violation"`).

### 4. `enqueue`
- **Purpose**: Invalidation event or repair task pushed onto the priority queue.
- **Envelope Fields**:
  - `node` (integer, required): Node scheduled for repair.
  - `level` (integer, required): Hierarchy layer of the task.
  - `version` (integer, required): Node version at the time of enqueueing.
- **Payload Schema**:
  - `priority` (number, required): Calculated urgency score in the repair heap.
  - `reason` (string, optional): Reason for enqueueing.

### 5. `pop`
- **Purpose**: A repair task is popped from the priority heap to be serviced.
- **Envelope Fields**:
  - `node` (integer, required): Node to be repaired.
  - `level` (integer, required): Layer level of the task.
  - `version` (integer, required): Node version recorded when the task was enqueued.
- **Payload Schema**:
  - `heap_size` (integer, required): Remaining size of the priority queue after pop.
  - `priority` (number, optional): Popped priority score.

### 6. `stale_skip`
- **Purpose**: A popped repair task is discarded without work because `event.version < index.version(node)`.
- **Envelope Fields**:
  - `node` (integer, required): Target node ID.
  - `level` (integer, required): Layer level index.
  - `version` (integer, required): Stale version from the queue event.
- **Payload Schema**:
  - `current_version` (integer, required): Current active version of the node in the index.
  - `reason` (string, required): e.g. `"stale_event_version"`.

### 7. `edge_add`
- **Purpose**: Directed edge $u \to v$ established at a given hierarchy layer.
- **Envelope Fields**:
  - `node` (integer, required): Source node $u$.
  - `level` (integer, required): Hierarchy layer level.
  - `version` (integer, required): Current version of source node $u$.
- **Payload Schema**:
  - `target_id` (integer, required): Target node $v$.
  - `distance` (number, optional): Distance or squared distance between $u$ and $v$.

### 8. `edge_remove`
- **Purpose**: Directed edge $u \to v$ severed at a given hierarchy layer.
- **Envelope Fields**:
  - `node` (integer, required): Source node $u$.
  - `level` (integer, required): Hierarchy layer level.
  - `version` (integer, required): Current version of source node $u$.
- **Payload Schema**:
  - `target_id` (integer, required): Severed neighbor node $v$.
  - `reason` (string, required): e.g. `"heuristic_prune"`, `"drift_invalidated"`.

### 9. `search`
- **Purpose**: Search query execution and routing diagnostic.
- **Envelope Fields**: `node = null` (or query integer ID), `level = null`, `version = null`.
- **Payload Schema**:
  - `k` (integer, required): Target neighbor count.
  - `ef` (integer, required): Exploration queue budget.
  - `latency_ms` (number, optional): Elapsed search time in ms.
  - `distance_computations` (integer, optional): Number of pairwise distance evaluations.

### 10. `metric`
- **Purpose**: Periodic or milestone evaluation metric recorded during simulation.
- **Envelope Fields**: `node = null`, `level = integer or null`, `version = null`.
- **Payload Schema**:
  - `name` (string, required): Metric identifier (e.g. `"recall@10"`, `"reachability"`).
  - `value` (number, required): Numeric evaluation value.
  - `step` (integer, optional): Simulation iteration or step number.

### 11. `run_end`
- **Purpose**: Simulation or benchmark run concluded.
- **Envelope Fields**: `node = null`, `level = null`, `version = null`.
- **Payload Schema**:
  - `status` (string, required): e.g. `"completed"`, `"aborted"`.
  - `total_events` (integer, optional): Total count of events emitted.
