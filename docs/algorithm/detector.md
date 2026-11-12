# Kinetic Detector Specification and Synchronization Model (`docs/algorithm/detector.md`)

This document specifies the operational design, scheduling mechanics, window refresh protocol, and vector synchronization contract for the `KineticDetector` engine in K-HNSW (`src/khnsw/core/detection/kinetic.py`).

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
1. **Required Injected Callback**:
   - The detector constructor takes a **required** injected callable: `sync_vector: Callable[[int, NDArray[np.float64]], None]`.
   - The core code adheres strictly to core purity and never accesses private attributes (such as `_store` or `_vectors`) or uses reflection fallback chains.
   - `sync_vector(node_id, pos)` receives the unquantized `float64` coordinate array $p(t) = a + b \cdot t$. The driver is responsible for converting to `float32(pos)` and storing the coordinate in index storage without bumping `version(node_id)`.
2. **Public Helper `sync_nodes`**:
   - The detector exposes `sync_nodes(nodes: Iterable[int], t: float)` to synchronize arbitrary sets of nodes to simulation timestamp $t$ without modifying index versions. This method is used by repair loops to align coordinates before repair planning.
3. **Granularity of Synchronization**:
   - When an event for edge $(u, v, l)$ is due at timestamp $t$, the detector synchronizes source node $u$, target node $v$, and all sibling neighbors $w \in N_l(u)$ to timestamp $t$ immediately before invoking `store.certify_source(index, u, l, meter)`.
4. **Post-Evaluation State**:
   - Versions remain unchanged. Heap entries for other edges and nodes remain active and valid.

### Trajectory Changes vs Time Advancement
- **Time Advancement (`detector.advance(t)`)**:
  Continuous motion along fixed trajectories $p(t) = a + b \cdot t$. Vectors are updated in `float32` via `sync_vector` without bumping versions; heap entries remain valid; due events are popped, checked, and either confirmed (`VIOLATED`) or rescheduled.
- **Trajectory Mutation (`detector.on_trajectory_change(node, t)`)**:
  Occurs when the driver updates a node's physical path via `index.move(node, new_vec)`. This increments `index.version(node)`, naturally invalidating older heap entries. The detector then invalidates and reschedules all candidate edges involving `node` as source, target, or witness.

---

## 3. Event Scheduling, Lookahead Horizon, and Window Refresh

### Window Parameter: `horizon`
- In `KineticDetector`, `horizon: float` denotes the lookahead duration $\Delta t$ (defaulting to `float('inf')`).
- When scheduling from timestamp $t_0$, the initial evaluation window is $[t_0, t_0 + \text{horizon}]$.
- **Infinite Horizon Behavior and the $10^6$ Cap**:
  In `certificates.py`, `first_violation_time` deduplicates root partitions using a purely relative tolerance:
  $$\text{time\_scale} = \max(|t_0|, |t_{\text{end}}|), \quad \varepsilon = \text{tol} \cdot \text{time\_scale}$$
  If $t_{\text{end}} = \infty$, $\text{time\_scale} = \infty$, causing $\varepsilon = \infty$. As a consequence, all interior finite roots satisfy $(r - \tau) \le \infty$ and are discarded, leaving partition $[t_0, \infty)$ with midpoint $\infty$ (yielding `NaN`).
  To prevent this numerical degeneration when an unbounded lookahead is requested (`horizon = float('inf')`), `KineticDetector` safely clamps the evaluation window to a large finite upper bound:
  $$t_{\text{end}} = t_0 + 10^6$$
  This preserves the full numerical precision of the Citardauq quadratic solver and relative root partitioning while covering all practically reachable simulation horizons.
- Only finite failure times satisfying $t_0 \le t_{\text{fail}} \le t_{\text{end}}$ are pushed to the heap.
- Edges that are currently `VIOLATED` in `CertificateStore` receive **no certificate** (only `VALID` edges are scheduled).

### Window Refresh Protocol
When `advance(t)` is called with $t > W_{\text{end}}$ (the boundary of the current lookahead window):
1. **Window-Boundary Processing**:
   The detector first processes and verifies all events due up to $W_{\text{end}}$ via `_process_due(W_end)`.
2. **Window Advance**:
   The window advances to $[W_{\text{end}}, W_{\text{end}} + \text{horizon}]$, and `window_refreshes` is incremented.
3. **Reschedule of All Valid Edges**:
   For every edge in the graph that is currently `VALID`:
   - Any prior certificate in the heap is dropped via `heap.invalidate_edge(edge)`.
   - A new certificate is computed via `edge_failure_time` over the new window $[W_{\text{end}}, W_{\text{end}} + \text{horizon}]$ and pushed to the heap.
   - Extra evaluations performed during the refresh are added to `fail_time_evals` and tracked in `refresh_evals`.
   - This refresh rule holds universally, including for edges rescheduled mid-window by `on_trajectory_change`, `on_topology_change`, or false alarms.

### Witness Set Alignment
For every directed edge $u \to v$ at layer $l$, the witness set evaluated by `edge_failure_time` is identically:
$$W_l(u \to v) = N_l(u) \setminus \{v\}$$
This guarantees 100% structural agreement with `CertificateStore.certify_source`.

---

## 4. Re-check, Confirmation, False Alarms, and Sampled-Time Semantics

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
3. **False Alarms and Sampled-Time Semantics**:
   - If the store marks $(u, v, l)$ as `VALID`:
   - **Sampled-time Semantics**: If a violation began and ended strictly between observation timestamps (a transient violation), or if continuous float64 predicted a violation that evaluates to VALID in float32 at observation instant $t$, the edge is NOT confirmed as a failure at time $t$. Under sampled-time semantics, only persistent violations present at evaluation moments constitute failures.
   - Detector increments `false_alarms` counter.
   - **Rescheduling**: The edge is NEVER silently dropped. The detector evaluates `edge_failure_time` over $W_l(u \to v)$ starting strictly after $t$:
     $$t_0 = t + \varepsilon, \quad \varepsilon = 10^{-9} \cdot \max(1.0, |t|)$$
     If a subsequent failure is found in $(t_0, t_0 + \text{horizon}]$, it is pushed to the heap with `heap.push(edge, t_next, witness, now=t)`.
     The positive $\varepsilon$ offset strictly prevents zero-delay re-triggering of grazing or transient ties at timestamp $t$.

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
