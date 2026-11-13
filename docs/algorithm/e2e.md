# End-to-End Kinetic Graph Simulation & Event Stream Verification

## 1. Overview & Simulation Architecture

The end-to-end integration pipeline exercises the full K-HNSW kinetic graph system across all algorithmic layers:
1. **Hierarchical Graph Topology**: `HNSWIndex` ($n=500, M=8, d=8$).
2. **Ground-Truth Certificate Maintenance**: Incremental `CertificateStore`.
3. **Continuous Kinetic Failure Detection**: `KineticDetector` driven by `LinearDrift` constant-velocity kinematic trajectories and priority-scheduled via `KineticEventHeap`.
4. **Topology Maintenance**: `KHNSWRepairEngine` fulfilling the `Repairer` protocol (`run_repairs` with computational budget limits and trace instrumentation via `EventEmitter`/`EventSink`).

```
                    ┌──────────────────────────────┐
                    │      LinearDrift (float64)   │
                    └──────────────┬───────────────┘
                                   │ (offset_a, vel_b)
                                   ▼
 ┌──────────────────────┐  advance(t)   ┌───────────────────────┐
 │   KineticEventHeap   │ ◄───────────► │    KineticDetector    │
 └──────────────────────┘               └───────────┬───────────┘
                                                    │
                                           confirmed failures
                                                    │
                                                    ▼
 ┌──────────────────────┐  run_repairs  ┌───────────────────────┐
 │    HNSWIndex (M=8)   │ ◄──────────── │   KHNSWRepairEngine   │
 └──────────┬───────────┘               └───────────┬───────────┘
            │                                       │
            │ mutate                                │ emit trace
            ▼                                       ▼
 ┌──────────────────────┐               ┌───────────────────────┐
 │   CertificateStore   │               │   EventSink (JSONL)   │
 └──────────────────────┘               └───────────────────────┘
```

---

## 2. Three-Phase Kinetic Scenario

The simulation executes three consecutive continuous and discrete motion phases under fixed seeds:

### Phase 1: Continuous Linear Drift
- Time advances across steps $t \in \{0.05, 0.10, 0.15\}$.
- Points drift continuously with constant velocities ($p_i(t) = a_i + b_i \cdot t$).
- `KineticDetector.advance(t)` pops due certificate invalidation events from the `KineticEventHeap` and confirms true Delaunay empty-ellipse violations via float32 rechecking.
- `KHNSWRepairEngine.run_repairs` schedules localized graph rewrites, pruning invalid certificates and maintaining local Delaunay empty-ellipse invariants.

### Phase 2: Trajectory Modification under Drift
- At $t=0.20$, specific nodes ($\{10, 25, 50, 100, 200\}$) undergo trajectory changes (abrupt changes in velocity directions scaled by nearest-neighbor distance).
- Updated coordinates are applied via `index.move`, trajectories updated in `LinearDrift`, and `detector.on_trajectory_change` invalidates old certificates and schedules new failure events.
- Kinetic drift resumes through $t \in \{0.25, 0.30\}$, verifying that localized trajectory shifts integrate with the event heap.

### Phase 3: Discrete Perturbations (Large Displacements)
- At $t=0.30$, 5% of nodes ($25$ nodes) undergo large discrete displacements:
  - Half perturbed by $2 \times$ mean nearest-neighbor distance.
  - Half perturbed by $5 \times$ mean nearest-neighbor distance.
- `engine.notify_moved` schedules repair rounds via `repair_loop`.
- A parallel no-repair control graph (`ctrl_idx`) receives identical vector displacements without running topological repairs.

---

## 3. Systematic Invariant & Verification Audits

After every repair round and at simulation termination, the following assertions are enforced:

1. **Structural Invariants**:
   `validate_invariants(idx) == []` verifies layer hierarchy constraints, entry point validity, base-layer connectivity, lack of self-loops, and bidirectional edge consistency.
2. **Degree Bounds**:
   `check_degree_bounds(idx) == []` confirms out-degrees never exceed $M_0 = 16$ (level 0) or $M = 8$ (level $> 0$).
3. **From-Scratch Oracle Agreement**:
   A fresh `CertificateStore` built from scratch on the final graph state (`oracle_store.build(idx)`) is compared edge-by-edge against the incremental `store`. Mismatches must be zero (no missed failures, no false confirmations).
4. **Monotonic & Isolated Versions**:
   Node version numbers never decrease ($v_{\text{final}} \ge v_{\text{initial}}$), and increment only on nodes that were moved or underwent topological repair.
5. **Schema Conformance & Event Accounting**:
   Every emitted event validates against `docs/contracts/event.schema.json` and per-type schemas in `docs/contracts/schemas/events/*.schema.json`.
   The total number of emitted `cert_fail` events matches the confirmed failure count reported by `detector.stats()["confirmed"]`.
6. **Deterministic Topology Replay**:
   Starting from an initial adjacency snapshot, sequential application of emitted `edge_add` and `edge_remove` events reproduces the exact adjacency of the final index across all nodes and hierarchy levels.

---

## 4. Empirical Recall Comparison: Ours vs. No-Repair Control

Routing fidelity and recall@10 were evaluated on $n=500, M=8, d=8$ across 25 perturbed nodes ($5\%$ of the graph) and 100 global queries at $ef=64$, comparing the un-repaired control against K-HNSW before and after reciprocal reverse-link repair (averaged across seeds $\{42, 43, 44\}$):

| Displacement Scale | Metric | No-Repair Control | Ours (Without Reverse Links) | Ours (With Reverse Links) | Delete + Reinsert Upper Bound |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Moderate ($3.0\times$ NN)** | **Self-Findability** | 0.907 | 0.440 | **1.000** | 1.000 |
| | **In-Edge from True 10-NN** | 0.493 | 0.293 | **0.987** | 1.000 |
| | **Recall@10 (Moved Nodes)** | 0.977 | 0.919 | **0.999** | 1.000 |
| | **Recall@10 (Global 100)** | 0.997 | 0.998 | **1.000** | 0.998 |
| **Large ($12.0\times$ NN)** | **Self-Findability** | 0.493 | 0.160 | **1.000** | 1.000 |
| | **In-Edge from True 10-NN** | 0.173 | 0.240 | **1.000** | 0.987 |
| | **Recall@10 (Moved Nodes)** | 0.891 | 0.812 | **0.991** | 0.992 |
| | **Recall@10 (Global 100)** | 0.998 | 1.000 | **1.000** | 0.998 |
| **Extreme ($30.0\times$ NN)** | **Self-Findability** | 0.387 | 0.120 | **1.000** | 1.000 |
| | **In-Edge from True 10-NN** | 0.173 | 0.280 | **1.000** | 0.987 |
| | **Recall@10 (Moved Nodes)** | 0.847 | 0.769 | **0.988** | 0.995 |
| | **Recall@10 (Global 100)** | 0.998 | 1.000 | **1.000** | 0.998 |

> [!IMPORTANT]
> - **Moved Node Ingress**: Without reciprocal reverse links, rebuilding only out-links severely isolates moved nodes. Even though the moved node has outgoing edges, incoming edges from its origin neighborhood are dropped by certificate repair and no new in-edges are formed from the destination cluster. Consequently, queries targeting moved nodes lose routing ingress (self-findability drops to $0.120 - 0.160$ and recall drops to $0.769 - 0.812$, below the control).
> - **Restoration via Reverse Links**: Synthesizing reciprocal $w \to m$ edges directly restores routing ingress: self-findability reaches **$1.000$** (matching delete+reinsert), true 10-NN in-edge incidence reaches **$100\%$**, and recall@10 at moved nodes reaches **$0.991$** at $12\times$ and **$0.988$** at $30\times$, outperforming the control ($0.891$ and $0.847$).
> - **Global Reachability**: Across the 95% unperturbed nodes, the dense background graph retains global recall $\ge 0.997$ across all configurations.

---

## 5. Event Stream Reconstructibility Analysis: Frontend Capabilities & Gaps

The trace event stream emitted by K-HNSW (`events.jsonl`) provides high-resolution operational observability. However, when reconstructing graph state from the event stream alone, several critical data gaps exist:

### What the Event Stream CAN Reconstruct:
1. **Dynamic Topology Transitions**:
   - `edge_add`: Identifies source `node`, hierarchy `level`, and `payload.target_id`.
   - `edge_remove`: Identifies source `node`, hierarchy `level`, `payload.target_id`, and pruning `reason`.
   - Sequential application against a known baseline snapshot accurately tracks the complete edge adjacency.
2. **Certificate Failures & Diagnosis**:
   - `cert_fail`: Emits `node`, `level`, `payload.neighbor_id`, and `payload.witness_id` indicating exactly which neighbor violated the empty-ellipse condition.
3. **Execution Milestones**:
   - `run_start` and `run_end` convey dataset parameters, total nodes, run duration, and completion status.

### Reconstructibility Gaps (What the Event Stream CANNOT Reconstruct Alone):
1. **Initial Topology & Hierarchy Snapshot Gap**:
   - The event stream does not emit initial node insertions, per-node maximum level assignments, or initial construction edges.
   - *Impact*: A frontend cannot construct the graph from an empty state solely by replaying events; it **requires** `initial_snapshot.json`.
2. **Continuous Kinematic Coordinates Gap**:
   - During continuous kinetic drift, coordinates evolve smoothly ($p_i(t) = a_i + b_i \cdot t$). No coordinate update events are emitted for continuous drift steps.
   - *Impact*: A frontend cannot display live node coordinates during drift unless it has access to the kinematic trajectories or discrete position snapshots.
3. **Discrete Coordinate Updates in `index.move`**:
   - `index.move` updates the node vector in the underlying `GraphStore`. However, `KHNSWRepairEngine` and `apply_plan` only emit topological `edge_remove` and `edge_add` events; they do not emit `move` events.
   - *Impact*: Node position shifts are invisible in the event stream unless the outer driver explicitly emits `EventType.move`.
4. **Hierarchical Level Adjustments & Entry Points**:
   - The event stream lacks events denoting entry point reassignment (`set_entry_point`) or node promotion to higher layers.

