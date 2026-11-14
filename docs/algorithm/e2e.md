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
- `KHNSWRepairEngine.run_repairs` extracts drifting nodes from confirmed failures via velocity checking (`trajectory_of(u)[1] != 0`), executing both out-link rebuild and reciprocal reverse-link synthesis for moving nodes to maintain degree bounds and prevent in-degree collapse.
- Zero certificate violations achieved after each step represents adherence to local Delaunay topological invariants rather than an empirical claim of monotonically superior recall over un-repaired controls.

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
> - **Invariant Preservation vs Recall Metric**: Zero certificate violations indicates that all local Delaunay/witness conditions are strictly satisfied; it is an invariant guarantee, not a quality metric that guarantees superior recall under every transient perturbation. Drift-moved node handling preserves in-degrees and restores self-findability to 1.000 without claiming drift repair monotonically outperforms un-repaired controls across all speeds.

### 4.1 Kinetic Drift Protocol Findings (Phase 5k Settled & Vanilla Analysis)

> [!WARNING]
> **RADIAL-DRIFT PROTOCOL: does not generalise.**
> The claims below were generated under nominal `default_rng(seed)` + `LinearDrift(seed)` (`docs/data/reconcile_benchmark.json` and `docs/data/vanilla_arms_benchmark.json`). As audited in Phase 5n Item A, this protocol generates **pure radial drift** ($\cos(v, X_0) \equiv 1.0000$) due to NumPy `SeedSequence` collision. The finding that "all repair arms score strictly below no_repair at ef=16" holds strictly under radial drift, where search beams travel outward along coordinate rays, artificially preserving high recall. Under true isotropic drift (`indep_dir` / `RandomState`, see `docs/algorithm/repair.md` Section 8.9), no-repair degrades to $0.65$ and staggered reinsertion provides significant recall recovery ($0.924 - 0.935$).

1. **Settled Graph Performance Relative to No-Repair (Claim i)**:
   On the settled graph, all repair variants—including K-HNSW at $ef=16$ rebuild (drifting recall $0.8927$ at $10\times$, $0.8687$ at $30\times$), K-HNSW at $ef=64$ rebuild ($0.9007$ at $10\times$, $0.8833$ at $30\times$), in-place reinsertion ($0.8813$ at $10\times$, $0.8673$ at $30\times$), and oracle reinsertion arms ($0.8733 - 0.8873$ at $10\times$, $0.8520 - 0.8607$ at $30\times$)—score strictly below `no_repair` ($0.9487$ at $10\times$, $0.9060$ at $30\times$) at evaluation $ef=16$. At evaluation $ef=32$, repair arms perform within $0.01$ of `no_repair` ($0.9867$ at $10\times$, $0.9713$ at $30\times$). Rebuild does not restore recall above the un-repaired baseline at $ef=16$.

2. **Certificate Invariants vs. Search Recall (Claim ii)**:
   Achieving zero violated certificates confirms adherence to empty-ellipse Delaunay invariants; it does not imply superior search recall. In fact, un-repaired graphs retain topologically invalid shortcut edges that yield higher recall at lower search queue capacities ($ef=16$).

3. **Construction Settle Confound (Claim iii)**:
   Running `settle_hnsw` on a freshly constructed HNSWIndex at $t=0$ removes $32.1\%$ of level-0 edges (dropping level-0 edges from $5,131$ to $3,483$ and mean level-0 out-degree from $10.26$ to $6.97$, means across seeds 42–44). Consequently, the settled baseline is structurally distinct and significantly thinned compared to a standard vanilla HNSW graph.

4. **In-Neighbour Pre-Repair (Claim v)**:
   Re-selecting static in-neighbour lists using the standard heuristic prior to reinsertion improved in-place reinsertion recall by $\approx 0.04$ at $10\times$ (from $0.8813$ to $0.9207$) and $\approx 0.02$ at $30\times$ (from $0.8673$ to $0.8893$) at $ef=16$ on the settled graph. However, performance remained below the `no_repair` baseline ($0.9487$ at $10\times$, $0.9060$ at $30\times$).

5. **Vanilla Graph Benchmark & Decision Analysis (Claim iv)**:
   - On the vanilla graph (`settle=False`, output `docs/data/vanilla_arms_benchmark.json`), `no_repair` drifting recall@10 ($ef=16$) is **0.9693** at $10\times$ (cost 0) and **0.9400** at $30\times$ (cost 0), matching or exceeding the full rebuild ceiling (**0.9673** at $10\times$, cost 124,964; and **0.9393** at $30\times$, cost 124,496).
   - Only `no_repair` satisfies the criterion of reaching within $0.01$ of ceiling drifting recall at $<50\%$ of ceiling cost. All active repair arms (reinsert once, periodic reinsert, and K-HNSW engine) score below `no_repair` at $ef=16$ ($0.8767 - 0.9193$).
   - *Harness Discrepancy Note*: Two harnesses disagree on `no_repair` and `ceiling` recall for the same nominal setting ($n=500, d=8, M=8, 10\%\text{ drift}, T=0.30$): Phase 5j reported `no_repair`/`ceiling` at **0.8907**/**0.9933** at $10\times$, whereas Phase 5k reported **0.9693**/**0.9673**. In a controlled side-by-side test with identical inputs (Item A), both code paths produce identical outputs (`no_repair` 0.8720, `ceiling` 0.9960 on 5j inputs; `no_repair` 0.9740, `ceiling` 0.9600 on 5k inputs). However, their displacement curves across $\{1, 3, 9, 27\}\times$ mean 1-NN distance do not coincide (Item B: 5j collapses to $0.6520$ at $27\times$ vs 5k at $0.9087$). Headroom exists specifically in high-displacement and high-drift regimes as measured in `docs/data/harness_conflict_benchmark.json`.

6. **Scheduler/Ablation Switches (Claim vi)**:
   All prospective scheduler prioritization, batching, and candidate filtering switches not implemented in `src/khnsw` are marked **specified-not-built**.

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

