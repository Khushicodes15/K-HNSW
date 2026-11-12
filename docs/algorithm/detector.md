# Kinetic Detector Specification and Synchronization Model (`docs/algorithm/detector.md`)

This document specifies the operational design, scheduling mechanics, and vector synchronization contract for the `KineticDetector` engine in K-HNSW (`src/khnsw/core/detection/kinetic.py`).

---

## 1. Architectural Role and Separation of Concerns

The `KineticDetector` coordinates kinetic failure prediction, event scheduling, and real-vector validation across three core components:
1. **Index (`GraphIndex` / `MutableGraphIndex`)**: Holds indexed vectors and graph adjacency. Provides monotonic `version(node)` tracking.
2. **Event Heap (`KineticEventHeap`)**: Priority queue tracking predicted failure times $t_{\text{fail}}$ in `float64`. Discards stale entries lazily when `version_of(node)` changes or when an edge generation is superseded.
3. **Certificate Store (`CertificateStore`)**: Maintains ground-truth edge status (`VALID` vs `VIOLATED`) and evaluates the `float32` static predicate (`is_violated_static`) on demand via `certify_source`.

---

## 2. Synchronization Model (Sync Model)

### Core Invariant: Protection of Monotonic Node Versions
In K-HNSW, `index.version(node)` is a monotonic counter strictly reserved for **topological and kinetic trajectory mutations** (e.g., discrete repositioning or velocity changes).
- The priority heap records `(u_ver, v_ver, w_ver)` upon scheduling. If any node's version changes, all existing heap entries dependent on that node become stale and are skipped during popping.
- Therefore, **the detector MUST NOT call `index.move()` to advance routine simulation time**. Calling `index.move()` across all nodes at each step would increment all versions and wipe out all scheduled certificates in the heap.

### Vector Synchronization Mechanics
To perform the required `float32` confirmation re-check at simulation timestamp $t$:
1. **Who Synchronizes Coordinates**:
   - The detector accepts an optional callback: `sync_vector: Callable[[int, float], None] | None`.
   - If provided, `sync_vector(node_id, t)` updates the spatial vector of `node_id` in the underlying index to `float32(p(t))` without incrementing `version(node_id)`.
   - If not provided, the detector updates the node coordinates directly on the underlying storage (e.g. `index._store.set_vector` on `HNSWIndex`, or `index._vectors[node_id]` on `ReferenceGraph`), strictly bypassing `index.move()` so that `index.version(node_id)` remains unmodified.
   - The driver may also bulk-synchronize coordinates to $t$ before calling `detector.advance(t)`. The detector's localized synchronization is idempotent.
2. **Granularity of Synchronization**:
   - When an event for edge $(u, v, l)$ is due at timestamp $t$, the detector synchronizes source node $u$, target node $v$, and all sibling neighbors $w \in N_l(u)$ to timestamp $t$ immediately before invoking `store.certify_source(index, u, l, meter)`.
3. **Post-Evaluation State**:
   - Versions remain unchanged. Heap entries for other edges and nodes remain active and valid.

### Trajectory Changes vs Time Advancement
- **Time Advancement (`detector.advance(t)`)**:
  Continuous motion along fixed trajectories $p(t) = a + b \cdot t$. Vectors are updated in `float32` without bumping versions; heap entries remain valid; due events are popped, checked, and either confirmed (`VIOLATED`) or rescheduled.
- **Trajectory Mutation (`detector.on_trajectory_change(node, t)`)**:
  Occurs when the driver updates a node's physical path via `index.move(node, new_vec)`. This increments `index.version(node)`, naturally invalidating older heap entries. The detector then invalidates and reschedules all candidate edges involving `node` as source, target, or witness.

---

## 3. Event Scheduling and Lookahead Horizon

### Window Parameter: `horizon`
- In `KineticDetector`, `horizon: float` denotes the lookahead duration $\Delta t$ (defaulting to `float('inf')`).
- When scheduling from timestamp $t_0$, the absolute evaluation window end passed to `edge_failure_time` is $t_{\text{end}} = t_0 + \text{horizon}$.
- Only finite failure times satisfying $t_0 \le t_{\text{fail}} \le t_0 + \text{horizon}$ are pushed to the heap.
- Edges that are currently `VIOLATED` in `CertificateStore` receive **no certificate** (only `VALID` edges are scheduled).

### Witness Set Alignment
For every directed edge $u \to v$ at layer $l$, the witness set evaluated by `edge_failure_time` is identically:
$$W_l(u \to v) = N_l(u) \setminus \{v\}$$
This guarantees 100% structural agreement with `CertificateStore.certify_source`.

---

## 4. Re-check, Confirmation, and False Alarm Handling

When an event pops at time $t$ with predicted failure $t_{\text{fail}} \le t$:
1. **Re-check via `CertificateStore`**:
   Source $u$ and neighbors $N_l(u)$ are synchronized to $t$. The detector invokes `store.certify_source(index, u, level, meter)`.
2. **Confirmed Failure**:
   - If the store marks $(u, v, l)$ as `VIOLATED`:
   - Store records `VIOLATED` and canonical witness $\min(W_{\text{violating}})$.
   - Detector increments `confirmed` counter.
   - Detector emits `cert_fail` event:
     - `node = u`, `level = level`, `version = index.version(u)`
     - `payload = {"neighbor_id": v, "reason": "relative_distance_violation", "witness_id": witness, "t": t}`.
   - The edge is returned in `advance(t)`'s confirmed failures list.
3. **False Alarm (Numerical Boundary Disagreement)**:
   - If the store marks $(u, v, l)$ as `VALID`:
   - The predicted root in `float64` does not constitute a violation under `float32` static evaluation (e.g. near-tie asymmetry or rounding).
   - Detector increments `false_alarms` counter.
   - **Rescheduling**: The edge is NEVER silently dropped. The detector evaluates `edge_failure_time` over $W_l(u \to v)$ starting from $t_0 = t + \varepsilon$ (where $\varepsilon = 10^{-9} \cdot \max(1.0, |t|)$) to $t_{\text{end}} = t + \text{horizon}$. If a subsequent failure is found, it is pushed to the heap with `heap.push(edge, t_next, witness, now=t)`.

---

## 5. Affected Edge Invalidation Rules

- **`on_trajectory_change(node, t)`**:
  Called after `index.move(node, ...)`. Identifies all edges whose certificate depends on `node`:
  $$\text{Affected}(node) = \bigcup_{l=0}^{\text{node\_level}(node)} \left( \{(node, v, l) \mid v \in N_l(node)\} \cup \bigcup_{u \in \text{in\_neighbors}(node, l)} \{(u, v, l) \mid v \in N_l(u)\} \right)$$
  For each edge in $\text{Affected}(node)$:
  - Drops existing heap entry via `heap.invalidate_edge(edge)`.
  - If the edge is `VALID` in the store, recomputes `edge_failure_time` and schedules into the heap.
- **`on_topology_change(u, level, t)`**:
  Called after topological repairs (e.g. `apply_plan`).
  - Drops existing heap entries for $(u, level)$ via `heap.invalidate_source(u, level)`.
  - For each new neighbor $v \in N_l(u)$, if $(u, v, level)$ is `VALID`, schedules a fresh certificate.
