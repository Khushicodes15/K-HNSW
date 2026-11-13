# Topological Repair Policies and Pruning Heuristics

This document specifies the localized repair policies, candidate pool selection rules, and pruning heuristics for graph maintenance in K-HNSW.

---

## 1. Candidate Pruning Heuristic

Candidate neighbor selection uses an $\alpha$-relaxed Relative Neighborhood Graph (RNG) heuristic adapted from `hnswlib`'s `getNeighborsByHeuristic2`.

### 1.1 Mathematical Formulation ($\alpha$ on Squared Distances)

Given base node $u$, cap $M = \text{max\_degree}(l)$, relaxation factor $\alpha \ge 1.0$, and candidates sorted primarily by distance to $u$ ascending (with ties broken by candidate ID descending):

A candidate $e$ is considered in order. It is rejected if there exists some already-selected neighbor $r \in R$ such that:
$$\alpha \cdot D(r, e) < D(u, e)$$

> [!IMPORTANT]
> The relaxation factor $\alpha$ is applied directly to **squared Euclidean distances** $D(p, q) = \|p - q\|_2^2$:
> $$\alpha \cdot \|r - e\|_2^2 < \|u - e\|_2^2$$
> Consequently, an $\alpha = 1.1$ on squared distances corresponds to:
> $$\sqrt{1.1} \approx 1.048808848$$
> on true unsquared Euclidean distances.
> Strict inequality is required: ties where $\alpha \cdot D(r, e) = D(u, e)$ are **accepted** (not rejected).

### 1.2 Early Return Semantics

When `skip_if_below_cap=True` and the ranked candidate pool size is strictly less than $M$ (`len(ranked) < cap`), the heuristic returns the candidates directly without evaluating pairwise inter-candidate distances.

> [!WARNING]
> While `skip_if_below_cap=True` saves distance computations when degree is below capacity (as done in `hnswlib`), it can leave edges that statically violate the empty-ellipse certificate condition.
> In contrast, K-HNSW sets `skip_if_below_cap=False` during certificate repair, ensuring all certificate violations are actively pruned.

---

## 2. Repair Rules and Candidate Pools

### 2.1 Rule Summary

1. **`native`**: Emulates `hnswlib`'s `updatePoint` neighbor repair. Repaired set consists of all out-neighbors $L_1 = N_l(\text{moved})$. Pool for each $v \in L_1$ is $(\{\text{moved}\} \cup L_1 \cup \bigcup_{x \in L_1} N_l(x)) \setminus \{v\}$. Pruning with $\alpha = 1.0$, `skip_if_below_cap = True`.
2. **`mnru_beta`** (MN-RU $\beta$): Repaired set restricted to mutual neighbors $v \in N_l(\text{moved})$ such that $\text{moved} \in N_l(v)$. Pool for each $v$ is $(\{\text{moved}\} \cup N_l(\text{moved}) \cup N_l(v)) \setminus \{v\}$. Pruning with $\alpha = 1.0$, `skip_if_below_cap = True`.
3. **`mnru_gamma`** (MN-RU $\gamma$): Same repaired set and pool as `mnru_beta`, but pruned with relaxed parameter $\alpha = 1.1$ (on squared distance).
4. **`mnru_thn`** (MN-RU Two-Hop Neighbors): Scope expands to two-hop neighborhood $N_l(\text{moved}) \cup \bigcup_{x \in N_l(\text{moved})} N_l(x)$, but still filters for mutual/in-neighbors where $\text{moved} \in N_l(v)$. Pool is $(\{\text{moved}\} \cup N_l(\text{moved}) \cup N_l(v)) \setminus \{v\}$, pruned with $\alpha = 1.1$.
5. **`khnsw`**: Certificate-driven localized repair. Repaired nodes are violating edge sources $u$ identified by the certificate store. Pool for $u$ is $(N_l(u) \cup \bigcup_{v \in V} N_l(v)) \setminus \{u\}$ where $V$ are the violating edge targets from $u$. Pruning with $\alpha = 1.0$, `skip_if_below_cap = False`.

---

## 3. Node Class Repair Coverage

The table below indicates which node classes undergo repair when a node moves:

| Node Relationship to Moved Node | Native | MN-RU Beta ($\alpha=1.0$) | MN-RU Gamma ($\alpha=1.1$) | MN-RU THN ($\alpha=1.1$, 2-hop) | K-HNSW (when edge violated) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Out-neighbor** ($m \to v$, non-mutual) | **Repaired** | Skipped | Skipped | Skipped | Repaired if $m \to v$ violated |
| **Mutual neighbor** ($m \leftrightarrow v$) | **Repaired** | **Repaired** | **Repaired** | **Repaired** | Repaired if $v \to \cdot$ violated |
| **Two-hop mutual** ($v \in N_2(m), v \to m$) | Skipped | Skipped | Skipped | **Repaired** | Repaired if $v \to \cdot$ violated |
| **Non-mutual in-neighbor** ($v \to m, m \not\to v$) | Skipped | Skipped | Skipped | Skipped | **Repaired** (witness/endpoint violation) |
| **In-neighbor beyond 2 hops** | Skipped | Skipped | Skipped | Skipped | **Repaired** (via certificate invalidation) |

---

## 4. Architectural Notes

### 4.1 Pre-Write Snapshot Semantics
Planning functions (`native_plan`, `mnru_plan`, `khnsw_plan`) are strictly read-only. Candidate pools and existing neighbor lists are read from the initial index state before any rewrite is executed. This prevents order-dependent cascading side-effects during plan formulation.

### 4.2 Deferred Out-Link Rebuild
In `hnswlib`, rebuilding the out-going connections $N_l(\text{moved})$ of the moved node itself is carried out by `repairConnectionsForUpdate` using hierarchical beam search from the top layer. Because hierarchical beam search is part of routing and entry-point traversal, rebuilding $N_l(\text{moved})$ is deliberately deferred to the subsequent routing/search phase.


---

## 5. The K-HNSW Iterative Repair Loop

The repair loop (`repair_loop`) integrates kinetic detection, certificate maintenance, and localized rewiring into a robust convergence cycle.

### 5.1 Algorithmic Stages per Round

1. **Failure Ingestion**: Gathers confirmed certificate failure events from `detector.advance(t)`, pre-filtered `Transition` events (where `kind == "failure"`), or static `CertificateStore` violation status.
2. **Reverse-Index Scope Resolution**: Formulates the initial repair scope comprising all violating edge sources and their reverse in-neighbors (`index.in_neighbors(u, level)`). When a node `m` moves, `m` and all `index.in_neighbors(m, level)` are included in scope.
3. **Pre-Planning Coordinate Synchronization**: Calls `detector.sync_nodes(scope | candidate_pools, t)` before candidate ranking or distance computation, guaranteeing that candidate pools evaluate Euclidean metrics using up-to-date coordinates at simulation timestamp `t`.
4. **Single-Pass Unified Planning**:
   - Rebuilds the moved node's out-links across all levels using beam search (`search_layer`) starting from the entry point. This rebuild occurs **exactly once** per move in the initial pass.
   - Formulates localized rewires for other violating sources using `khnsw_plan`.
   - Combines rewrites into a single atomic `RepairPlan`.
5. **Atomic Application**: Executes `apply_plan` to rewire adjacency lists and emit deterministic trace events (`edge_remove` and `edge_add`).
6. **Certificate Store Synchronization**: Calls `store.on_neighbor_change(index, u, lvl, meter)` for every modified source, updating status and canonical witnesses.
7. **Kinetic Rescheduling**: Notifies `detector.on_topology_change(u, lvl, t=t)` with an explicit timestamp `t` to drop invalidated certificates and schedule new ones.
8. **Convergence and Halting**: Repeats until no VIOLATED edges remain in scope, the computational budget is exhausted, or `max_rounds` (default 10) is reached.

### 5.2 Mandatory Diversity Pruning (No Early-Return)

In baselines (`native_plan`, `mnru_plan`), when the ranked candidate pool is smaller than capacity (`len(ranked) < cap`), candidates are accepted immediately (`skip_if_below_cap = True`).

> [!IMPORTANT]
> K-HNSW strictly sets `skip_if_below_cap = False`.
> Even when the candidate pool contains fewer nodes than capacity, the $\alpha$-RNG diversity test is actively evaluated:
> $$\alpha \cdot D(r, e) < D(u, e)$$
> Any candidate occluded by an already-accepted neighbor is rejected.
> This ensures that all statically violated certificates are actively resolved and eliminated, differing from the baselines on purpose.

### 5.3 Computational Budget Accounting and Safe Halting

Repair budgets are denominated strictly in **pairwise distance computations** counted via `CostMeter`, partitioned across four distinct categories:
- `search`: Distance computations during `search_layer` beam search traversals for moved node out-links.
- `planning`: Distance computations during candidate pool ranking (`rank_pool`) and diversity pruning (`prune`).
- `store`: Distance computations during incremental certificate store updates (`store.on_neighbor_change` and `store.on_move`).
- `detector`: Distance computations during kinetic confirmation checks (`detector.advance`).

> [!NOTE]
> If the cumulative distance computations exceed the specified `budget`, the loop halts at a **safe round boundary** (never mid-`apply_plan`).
> Unrepaired edges remain `VIOLATED` and are returned in `RepairResult.still_violated`. They are **never dropped silently**.

### 5.4 Return Schema (`RepairResult`)

The repair loop returns a frozen dataclass `RepairResult`:
- `rounds`: Total integer rounds executed ($0 \le \text{rounds} \le \text{max\_rounds}$).
- `edges_repaired`: Count of unique violated edges successfully repaired.
- `still_violated`: Set of remaining violated edge keys `(u, v, level)`.
- `budget_used`: Dictionary `{"search": int, "planning": int, "store": int, "detector": int}`.
- `stopped_reason`: String flag indicating termination reason: `"converged"`, `"budget"`, or `"max_rounds"`.

---

## 6. Empirical Evaluation and Benchmark Results

The iterative repair loop and baseline emulations were evaluated using `scripts/measure_repair.py` on `HNSWIndex` across discrete movement perturbations and continuous linear trajectories.

### 6.1 Benchmark Definitions and Accounting Rules

- **Total Dist (Algorithm Cost)**: The sum of distance computations incurred by the algorithm during repair:
  $$\text{Total Dist} = \text{Search} + \text{Planning} + \text{Store} + \text{Detector}$$
  Baselines (`hnswlib-update` and `MN-RU`) do not maintain a `CertificateStore` or `KineticDetector`, paying only `Planning` cost (`Store = 0`, `Detector = 0`).
  Measurement overhead (from-scratch store validation and recall evaluation) is strictly excluded from `Total Dist`.
- **Violations**: The global count of directed edges $(u, v, l)$ failing the Delaunay/witness certificate condition in a fresh `CertificateStore` built from scratch across the entire graph at measurement time. The pre-move baseline is 0 across all configurations (settle pass settled).
- **No-Repair Control**: Evaluates the index after spatial displacement with zero repair applied.
- **Zero Violations as an Invariant (Not a Quality Metric)**:
  Achieving zero certificate violations confirms that every directed edge satisfies the local empty-ellipse Delaunay condition. Zero violations is a structural invariant and correctness contract of the graph, not a proxy metric for nearest-neighbor search recall. Repair does not claim to monotonically improve recall over an un-repaired control at individual snapshots; rather, it maintains bounded out-degrees, eliminates topologically invalid edges, and preserves certificate consistency across extended kinetic drift.

### 6.2 Empirical Measurements

> [!NOTE]
> The Discrete Moves benchmarks below incorporate reciprocal reverse-link synthesis (PR #17). Adding incoming links from destination neighbors rewrites those neighbors and re-evaluates their certificates, updating "Repaired" unique nodes from 0-1 to 5-8 (at $n=200$) and 27-46 (at $n=1000$) while maintaining 0 violations and 1.0000 recall. Section 8.6 provides empirical measurements for continuous drift under drift-moved node handling.

#### Discrete Moves ($n=200, M=8, d=2, \text{seed}=44$)

| Method | Recall@10 | Search | Planning | Store | Detector | Total Dist | Violations | Repaired |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scale 0.01 (disp=0.0016)** | | | | | | | | |
| no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| ours | 1.0000 | 362 | 502 | 388 | 0 | 1252 | 0 | 5 |
| our emulation of hnswlib-update-style | 1.0000 | 0 | 303 | 0 | 0 | 303 | 217 | 30 |
| our emulation of MN-RU-style (beta) | 1.0000 | 0 | 129 | 0 | 0 | 129 | 56 | 24 |
| **Scale 0.05 (disp=0.0081)** | | | | | | | | |
| no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| ours | 1.0000 | 362 | 505 | 388 | 0 | 1255 | 0 | 5 |
| our emulation of hnswlib-update-style | 1.0000 | 0 | 303 | 0 | 0 | 303 | 217 | 30 |
| our emulation of MN-RU-style (beta) | 1.0000 | 0 | 129 | 0 | 0 | 129 | 56 | 24 |
| **Scale 0.20 (disp=0.0325)** | | | | | | | | |
| no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| ours | 1.0000 | 367 | 631 | 503 | 0 | 1501 | 0 | 7 |
| our emulation of hnswlib-update-style | 1.0000 | 0 | 303 | 0 | 0 | 303 | 221 | 30 |
| our emulation of MN-RU-style (beta) | 1.0000 | 0 | 129 | 0 | 0 | 129 | 60 | 24 |

#### Discrete Moves ($n=200, M=8, d=8, \text{seed}=50$)

| Method | Recall@10 | Search | Planning | Store | Detector | Total Dist | Violations | Repaired |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scale 0.01 (disp=0.0167)** | | | | | | | | |
| no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ours | 1.0000 | 886 | 1895 | 3780 | 0 | 6561 | 0 | 8 |
| our emulation of hnswlib-update-style | 1.0000 | 0 | 10259 | 0 | 0 | 10259 | 7 | 19 |
| our emulation of MN-RU-style (beta) | 1.0000 | 0 | 1431 | 0 | 0 | 1431 | 154 | 29 |
| **Scale 0.05 (disp=0.0837)** | | | | | | | | |
| no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 9 | 0 |
| ours | 1.0000 | 885 | 2092 | 3982 | 0 | 6959 | 0 | 8 |
| our emulation of hnswlib-update-style | 1.0000 | 0 | 10328 | 0 | 0 | 10328 | 13 | 23 |
| our emulation of MN-RU-style (beta) | 1.0000 | 0 | 1434 | 0 | 0 | 1434 | 161 | 29 |
| **Scale 0.20 (disp=0.3348)** | | | | | | | | |
| no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 33 | 0 |
| ours | 1.0000 | 881 | 2874 | 4510 | 0 | 8265 | 0 | 8 |
| our emulation of hnswlib-update-style | 1.0000 | 0 | 10374 | 0 | 0 | 10374 | 27 | 30 |
| our emulation of MN-RU-style (beta) | 1.0000 | 0 | 1386 | 0 | 0 | 1386 | 171 | 33 |

#### Discrete Moves ($n=1000, M=8, d=2, \text{seed}=44$)

| Method | Recall@10 | Search | Planning | Store | Detector | Total Dist | Violations | Repaired |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scale 0.01 (disp=0.0008)** | | | | | | | | |
| no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| ours | 1.0000 | 2228 | 2616 | 2135 | 0 | 6979 | 0 | 28 |
| our emulation of hnswlib-update-style | 1.0000 | 0 | 5887 | 0 | 0 | 5887 | 911 | 137 |
| our emulation of MN-RU-style (beta) | 1.0000 | 0 | 860 | 0 | 0 | 860 | 381 | 128 |
| **Scale 0.05 (disp=0.0038)** | | | | | | | | |
| no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 15 | 0 |
| ours | 1.0000 | 2224 | 2760 | 2211 | 0 | 7195 | 0 | 27 |
| our emulation of hnswlib-update-style | 1.0000 | 0 | 5885 | 0 | 0 | 5885 | 917 | 142 |
| our emulation of MN-RU-style (beta) | 1.0000 | 0 | 860 | 0 | 0 | 860 | 390 | 128 |
| **Scale 0.20 (disp=0.0150)** | | | | | | | | |
| no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 42 | 0 |
| ours | 1.0000 | 2233 | 3343 | 2682 | 0 | 8258 | 0 | 34 |
| our emulation of hnswlib-update-style | 1.0000 | 0 | 5914 | 0 | 0 | 5914 | 929 | 152 |
| our emulation of MN-RU-style (beta) | 1.0000 | 0 | 860 | 0 | 0 | 860 | 407 | 128 |

#### Discrete Moves ($n=1000, M=8, d=8, \text{seed}=50$)

| Method | Recall@10 | Search | Planning | Store | Detector | Total Dist | Violations | Repaired |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scale 0.01 (disp=0.0139)** | | | | | | | | |
| no-repair control | 0.9990 | 0 | 0 | 0 | 0 | 0 | 7 | 0 |
| ours | 0.9990 | 6323 | 12037 | 28510 | 0 | 46908 | 0 | 45 |
| our emulation of hnswlib-update-style | 0.9990 | 0 | 83527 | 0 | 0 | 83527 | 25 | 115 |
| our emulation of MN-RU-style (beta) | 0.9990 | 0 | 15467 | 0 | 0 | 15467 | 677 | 126 |
| **Scale 0.05 (disp=0.0696)** | | | | | | | | |
| no-repair control | 0.9990 | 0 | 0 | 0 | 0 | 0 | 40 | 0 |
| ours | 0.9990 | 6279 | 13681 | 29393 | 0 | 49391 | 0 | 46 |
| our emulation of hnswlib-update-style | 0.9990 | 0 | 83069 | 0 | 0 | 83069 | 35 | 131 |
| our emulation of MN-RU-style (beta) | 0.9990 | 0 | 15265 | 0 | 0 | 15265 | 693 | 138 |
| **Scale 0.20 (disp=0.2784)** | | | | | | | | |
| no-repair control | 0.9990 | 0 | 0 | 0 | 0 | 0 | 164 | 0 |
| ours | 0.9990 | 6314 | 19678 | 33236 | 0 | 59228 | 0 | 46 |
| our emulation of hnswlib-update-style | 0.9990 | 0 | 81218 | 0 | 0 | 81218 | 96 | 180 |
| our emulation of MN-RU-style (beta) | 0.9990 | 0 | 14893 | 0 | 0 | 14893 | 765 | 177 |

#### Continuous LinearDrift ($\text{speed}=1.5, t=0.10$)

| Scenario Config | Method | Recall@10 | Search | Planning | Store | Detector | Total Dist | Violations | Repaired |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$n=200, d=2, \text{seed}=44$** | no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 10 | 0 |
| | ours | 1.0000 | 0 | 124 | 64 | 76 | 264 | 0 | 10 |
| | our emulation of hnswlib-update-style | 1.0000 | 0 | 1363 | 0 | 0 | 1363 | 134 | 30 |
| | our emulation of MN-RU-style (beta) | 1.0000 | 0 | 331 | 0 | 0 | 331 | 110 | 33 |
| **$n=200, d=8, \text{seed}=50$** | no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 27 | 0 |
| | ours | 1.0000 | 0 | 1442 | 784 | 965 | 3191 | 0 | 22 |
| | our emulation of hnswlib-update-style | 1.0000 | 0 | 31008 | 0 | 0 | 31008 | 26 | 54 |
| | our emulation of MN-RU-style (beta) | 1.0000 | 0 | 5678 | 0 | 0 | 5678 | 488 | 82 |
| **$n=1000, d=2, \text{seed}=44$** | no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 28 | 0 |
| | ours | 1.0000 | 0 | 442 | 190 | 248 | 880 | 0 | 28 |
| | our emulation of hnswlib-update-style | 1.0000 | 0 | 4748 | 0 | 0 | 4748 | 604 | 86 |
| | our emulation of MN-RU-style (beta) | 1.0000 | 0 | 687 | 0 | 0 | 687 | 351 | 82 |
| **$n=1000, d=8, \text{seed}=50$** | no-repair control | 0.9990 | 0 | 0 | 0 | 0 | 0 | 176 | 0 |
| | ours | 0.9990 | 0 | 12186 | 6828 | 7918 | 26932 | 0 | 152 |
| | our emulation of hnswlib-update-style | 1.0000 | 0 | 388788 | 0 | 0 | 388788 | 29 | 445 |
| | our emulation of MN-RU-style (beta) | 0.9990 | 0 | 65803 | 0 | 0 | 65803 | 1436 | 356 |

---

## 7. Topology Repair Engines (`Repairer` Protocol Implementation)

To decouple graph repair policies from simulation drivers and execution environments, maintenance algorithms are encapsulated within concrete implementations of the `Repairer` protocol (`src/khnsw/core/repair/engine.py`).

### 7.1 Engine Architecture

Three distinct repair engines implement the `Repairer` interface:
1. **`KHNSWRepairEngine`**:
   - Encapsulates the full K-HNSW iterative certificate-driven repair loop (`repair_loop`).
   - Supports optional injection of `CertificateStore`, `KineticDetector`, beam search `search_layer`, exploration factor `ef`, and a dynamic `time_source`.
   - Re-evaluates edge validity via relative neighborhood empty-ellipse tests with mandatory diversity pruning (`skip_if_below_cap = False`).
2. **`HnswlibStyleEngine`**:
   - Emulates `hnswlib`'s `updatePoint` repair policy.
   - Plans localized out-link replacement for all outgoing neighbors of moved nodes using `native_plan` with $\alpha = 1.0$ and early return (`skip_if_below_cap = True`).
   - Strictly isolated from `CertificateStore` (never touches certificates or detector).
3. **`MNRUStyleEngine`**:
   - Emulates Mutual Neighbors Repair Update (MN-RU) policies.
   - Restricts candidate scopes to mutual neighbors ($\text{moved} \leftrightarrow v$) via `mnru_plan`.
   - Configurable via `alpha` (e.g. 1.0 for MN-RU $\beta$ or 1.1 for MN-RU $\gamma$) and `scope` (`"one_hop"` or `"two_hop"`).
   - Strictly isolated from `CertificateStore`.

### 7.2 Movement Accumulation and Execution Cycle

- **`notify_moved(node: int, t: float = 0.0)`**:
  Records moved nodes into an internal worklist without immediate topological modification. Multiple moves between repair passes accumulate deterministically.
- **`on_move(index, node_id, old_vector, new_vector, sink)`**:
  Fulfills the strict call order of the `Repairer` contract (`read old vector -> index.move -> repairer.on_move`). Internally registers the node via `notify_moved` and synchronizes certificate status.
- **`run_repairs(index, budget, sink) -> dict[str, Any]`**:
  Executes repair passes for all accumulated moved nodes and/or confirmed detector failures. Upon completion, the worklist of pending moves is cleared. If called when no work is pending, immediately returns zero metrics with `stopped_reason = "nothing_to_do"`.
- **Drift-Moved Node Extraction**:
  During continuous kinetic drift where no discrete `notify_moved` calls occur, `KHNSWRepairEngine` inspects confirmed certificate failure edges from `detector.advance(t_curr)` and their canonical witnesses. It queries the detector's kinematic model via `trajectory_of(node_id)` and identifies nodes with non-zero velocity ($\vec{v} \ne \mathbf{0}$). For each unique drifting node, the engine executes `repair_loop(..., moved_node=node_id, t=t_curr)`, ensuring out-link beam search reconstruction (`_rebuild_moved_out_links`) and reciprocal reverse-link synthesis are performed at the new timestamp. Drifting nodes repaired in this pass are tracked in `result["moved_in_drift"]`.

### 7.3 Budget Accounting and Safe Halting

- **Budget Units**: Denominated strictly in floored pairwise distance computations (`int(math.floor(budget))`).
- **Safe Halting Guarantee**: Halting occurs cleanly at safe round/node boundaries and never mid-`apply_plan`. If budget is exhausted, any remaining unrepaired moved nodes and violated edges are retained and reported in `unrepaired_nodes` and `still_violated`, never silently discarded.

### 7.4 Return Payload Schema

The dictionary returned by `run_repairs` conforms to the standardized repair schema:
```python
{
    "engine": str,  # Engine class name
    "rounds": int,  # Total repair rounds executed
    "edges_repaired": int,  # Count of edges or rewrites repaired
    "still_violated": list[tuple],  # List of (u, v, level) edges still violated
    "still_violated_count": int,  # Length of still_violated
    "unrepaired_nodes": list[int],  # Node IDs that could not be repaired within budget
    "budget_used": dict[
        str, int
    ],  # {"search": int, "planning": int, "store": int, "detector": int, "reverse_link": int}
    "budget_consumed": int,  # Total distance computations consumed
    "stopped_reason": str,  # "converged", "budget", "max_rounds", or "nothing_to_do"
}
```

---

## 8. Reverse-Link Repair and Moved Node Findability

### 8.1 The Moved Node Isolation Problem

When points undergo large spatial displacement (e.g. $12\times$ or $30\times$ mean nearest-neighbor distance), an asymmetry emerges under naive unidirectional out-link rebuild:

1. **Out-links rebuilt**: The moved node $m$ has its outgoing connections $N_l(m)$ rebuilt towards its new spatial neighborhood using beam search (`search_layer`).
2. **Old in-links severed**: In-neighbors $u$ in $m$'s origin neighborhood experience certificate violations on edges $u \to m$ as $m$ moves away. Localized certificate repair drops $u \to m$.
3. **No reciprocal in-links formed**: Nodes $w$ in $m$'s new destination neighborhood are never prompted to add edges $w \to m$ because $w$ was not a source of any certificate violation.

This asymmetry leaves moved nodes almost completely devoid of incoming edges (mean in-degree collapsed to $0.49 - 0.52$). Consequently, search queries targeting the moved node fail to enter it from the background graph:

- **Self-Findability**: In $n=500, M=8, d=8$ under $12\times / 30\times$ displacement, search queries starting from the moved node's exact position located the node only $12\% - 16\%$ of the time (worse than the un-repaired control at $38.7\% - 49.3\%$, and far below delete+reinsert at $100\%$).
- **Recall at Moved Nodes**: Recall@10 at moved node coordinates dropped to $0.769 - 0.812$ (below control's $0.847 - 0.891$).

### 8.2 Algorithmic Formulation

To resolve isolation without requiring a global graph re-index, the K-HNSW repair loop executes reciprocal reverse-link synthesis directly following out-link beam search:

1. **Candidate Discovery**: For each hierarchy level $l \in [0, \text{node\_level}(m)]$, let $N_l(m)$ be the newly selected outgoing neighbors of $m$. Every node $w \in N_l(m)$ ($w \ne m$, $m \notin N_l(w)$) is evaluated as a reverse-link host.
2. **Pre-Prune Synchronization**: Before evaluating distances, `detector.sync_nodes(C, t)` updates coordinates for $w$, $N_l(w)$, and $m$ at simulation timestamp $t$.
3. **Capacity and Pruning**:
   - If $\deg_l(w) < \text{max\_degree}(l)$, $m$ is directly appended to $w$'s neighbor list:
     $$N_l(w) \leftarrow N_l(w) \cup \{m\}$$
   - If $\deg_l(w) == \text{max\_degree}(l)$, candidate pool $C_w = N_l(w) \cup \{m\}$ is ranked by Euclidean distance from $w$. The pool is pruned using the $\alpha$-RNG heuristic with $\text{base}=w$, $\text{cap}=\text{max\_degree}(l)$, $\alpha=1.0$, and mandatory diversity (`skip_if_below_cap=False`). If an existing edge is evicted, it is dropped cleanly.
4. **Atomic Application**: All reciprocal additions and evictions are packaged into `Rewrite(node=w, level=l, old=old_w, new=new_w)` within the round's `RepairPlan`.
5. **Certificate and Detector Synchronization**: For every rewritten neighbor $w$, `store.on_neighbor_change(index, w, l, meter)` recomputes edge certificates, and `detector.on_topology_change(w, l, t=t)` synchronizes the kinetic heap.
6. **Execution Frequency**: Executed **strictly once** per moved node in round 0. It is never repeated during subsequent convergence rounds.

### 8.3 Computational Budget Accounting

Pairwise Euclidean distance computations evaluated during reverse candidate ranking and heuristic pruning are tracked in `meter_reverse_link`. They are reported in `RepairResult.budget_used["reverse_link"]` and summed into `total_cost()`:

$$\text{Total Cost} = \text{Search} + \text{Planning} + \text{Store} + \text{Detector} + \text{Reverse Link}$$

When host nodes have spare capacity, adding reciprocal edges incurs zero additional distance evaluations.

### 8.4 Empirical Sweep Across Displacement Scales

The table below presents empirical measurements on `HNSWIndex` ($n=500, M=8, d=8$, 5% of nodes moved, 25 moves per run) across 5 displacement scales ($\{0.2\times, 1.0\times, 3.0\times, 12.0\times, 30.0\times\}$ mean nearest-neighbor distance) averaged across random seeds $\{42, 43, 44\}$:

| Scale | Variant | Mean In-Deg | Self-Find | True 10-NN In | Recall@10 (Moved) | Recall@10 (Global) | Total Cost |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.2x** | No-Repair Control | 6.52 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| | Ours (before reverse links) | 5.79 | 1.000 | 1.000 | 1.000 | 1.000 | 16,219 |
| | **Ours (with reverse links)** | **6.29** | **1.000** | **1.000** | **1.000** | **1.000** | **22,546** |
| | Hnswlib-style baseline | 6.36 | 1.000 | 1.000 | 1.000 | 1.000 | 26,020 |
| | MN-RU-style baseline | 6.59 | 1.000 | 1.000 | 1.000 | 1.000 | 4,761 |
| | Delete + Reinsert (Upper Bound) | 7.03 | 1.000 | 1.000 | 0.999 | 0.999 | 11,221 |
| **1.0x** | No-Repair Control | 6.52 | 1.000 | 0.987 | 0.999 | 1.000 | 0 |
| | Ours (before reverse links) | 2.99 | 0.947 | 0.880 | 0.988 | 0.997 | 22,159 |
| | **Ours (with reverse links)** | **5.04** | **1.000** | **1.000** | **1.000** | **1.000** | **29,432** |
| | Hnswlib-style baseline | 4.81 | 0.973 | 0.907 | 0.992 | 0.998 | 23,517 |
| | MN-RU-style baseline | 5.37 | 0.973 | 0.933 | 0.993 | 0.998 | 4,359 |
| | Delete + Reinsert (Upper Bound) | 6.23 | 0.987 | 0.987 | 0.999 | 1.000 | 11,076 |
| **3.0x** | No-Repair Control | 6.52 | 0.907 | 0.493 | 0.977 | 0.997 | 0 |
| | Ours (before reverse links) | 0.87 | 0.440 | 0.293 | 0.919 | 0.998 | 23,671 |
| | **Ours (with reverse links)** | **2.21** | **1.000** | **0.987** | **0.999** | **1.000** | **25,075** |
| | Hnswlib-style baseline | 3.28 | 0.693 | 0.480 | 0.949 | 0.998 | 21,522 |
| | MN-RU-style baseline | 4.55 | 0.747 | 0.467 | 0.957 | 0.998 | 4,215 |
| | Delete + Reinsert (Upper Bound) | 2.59 | 1.000 | 1.000 | 1.000 | 0.998 | 10,652 |
| **12.0x** | No-Repair Control | 6.52 | 0.493 | 0.173 | 0.891 | 0.998 | 0 |
| | Ours (before reverse links) | 0.52 | 0.160 | 0.240 | 0.812 | 1.000 | 23,905 |
| | **Ours (with reverse links)** | **1.85** | **1.000** | **1.000** | **0.991** | **1.000** | **24,413** |
| | Hnswlib-style baseline | 3.20 | 0.560 | 0.307 | 0.889 | 1.000 | 21,613 |
| | MN-RU-style baseline | 4.55 | 0.427 | 0.200 | 0.879 | 1.000 | 4,175 |
| | Delete + Reinsert (Upper Bound) | 1.93 | 1.000 | 0.987 | 0.992 | 0.998 | 10,258 |
| **30.0x** | No-Repair Control | 6.52 | 0.387 | 0.173 | 0.847 | 0.998 | 0 |
| | Ours (before reverse links) | 0.49 | 0.120 | 0.280 | 0.769 | 1.000 | 23,853 |
| | **Ours (with reverse links)** | **1.91** | **1.000** | **1.000** | **0.988** | **1.000** | **24,350** |
| | Hnswlib-style baseline | 3.15 | 0.467 | 0.293 | 0.849 | 1.000 | 21,489 |
| | MN-RU-style baseline | 4.55 | 0.320 | 0.200 | 0.828 | 1.000 | 4,175 |
| | Delete + Reinsert (Upper Bound) | 2.40 | 1.000 | 0.987 | 0.995 | 0.998 | 10,194 |

### 8.5 Empirical Insights and Findings (Discrete Moves)

1. **Elimination of Moved Node Blind Spots**:
   Under large displacements ($12\times$ and $30\times$ NN), reverse links increase self-findability from $12\% - 16\%$ to **$100.0\%$**, matching the full delete+reinsert upper bound.
2. **In-Neighbors from True Neighborhood**:
   $100\%$ of moved nodes establish at least one incoming edge from their true 10-nearest neighbors in their destination cluster (compared to $17.3\%$ in control and $24\% - 28\%$ before reverse links).
3. **Recall Superiority**:
   Recall@10 evaluated at moved nodes reaches **$0.991$** (at $12\times$) and **$0.988$** (at $30\times$), decisively outperforming the un-repaired control ($0.891$ and $0.847$), Hnswlib-style emulation ($0.889$ and $0.849$), and MN-RU-style emulation ($0.879$ and $0.828$).
4. **Delete + Reinsert Distance Accounting Note**:
   In the table above, the "Delete + Reinsert" upper bound cost reflects `index.add` distance computations tracked via `idx.distance_computations` (which meters `search_layer` evaluations during entry-point navigation). `index.delete` is purely topological (0 distance computations). When internal heuristic candidate evaluations inside `select_neighbors_heuristic` during `index.add` are also metered, total delete+reinsert work is ~2.0x–2.3x higher: 25,825 (0.2x), 24,768 (1.0x), 21,564 (3.0x), 21,675 (12.0x), and 22,013 (30.0x). Under either accounting, K-HNSW achieves comparable self-findability (100%) and recall without incurring full reinsertion churn.

### 8.6 Continuous Kinetic Drift with Drift-Moved Node Handling

Under continuous kinetic drift without explicit movement notifications, moving nodes cause certificate failures on incident edges. By identifying moving nodes via non-zero velocity filtering on confirmed failure candidates (`_trajectory_of(nid)`), `KHNSWRepairEngine` routes drifting nodes through out-link rebuild and reverse-link synthesis.

Evaluation on `HNSWIndex` ($n=500, M=8, d=8$, $T=0.30$ in 10 steps, 10% drifting nodes, seeds $\{42, 43, 44\}$):

| Speed | Variant | Drift Recall@10 ($ef=16$) | Drift Recall@10 ($ef=32$) | Global Recall@10 ($ef=16$) | Global Recall@10 ($ef=32$) | Self-Find ($ef=64$) | Mean In-Deg | Violations | Total Cost |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10x** | No-Repair Control | 0.9487 | 0.9867 | 0.9613 | 0.9933 | 1.000 | 6.66 | 419.3 | 0 |
| | Ours (before drift-moved fix) | 0.8880 | 0.9653 | 0.9667 | 0.9960 | 0.993 | 4.03 | **0.0** | 37,585 |
| | **Ours (after drift-moved fix)** | **0.9013** | **0.9827** | **0.9633** | **0.9950** | **1.000** | **6.37** | **0.0** | 201,997 |
| **30x** | No-Repair Control | 0.9060 | 0.9713 | 0.9613 | 0.9933 | 1.000 | 6.66 | 460.3 | 0 |
| | Ours (before drift-moved fix) | 0.8313 | 0.9253 | 0.9683 | 0.9963 | 0.960 | 3.23 | **0.0** | 34,048 |
| | **Ours (after drift-moved fix)** | **0.8840** | **0.9693** | **0.9667** | **0.9963** | **1.000** | **5.31** | **0.0** | 127,038 |
| **100x**| No-Repair Control | 0.8627 | 0.9520 | 0.9613 | 0.9933 | 1.000 | 6.66 | 477.3 | 0 |
| | Ours (before drift-moved fix) | 0.7733 | 0.8780 | 0.9703 | 0.9963 | 0.960 | 3.18 | **0.0** | 31,050 |
| | **Ours (after drift-moved fix)** | **0.8813** | **0.9627** | **0.9703** | **0.9963** | **1.000** | **4.89** | **0.0** | 85,713 |

Under 100% drifting nodes at $30\times$ speed:
- **No-Repair Control**: Drift Recall@10 $ef=16$: 0.9687 ($ef=32$: 0.9967), Violations: 712.3, In-Degree: 7.68.
- **Ours (after drift-moved fix)**: Drift Recall@10 $ef=16$: **0.9927** ($ef=32$: **0.9993**), Violations: **0.0**, Self-Find: **1.000**, In-Degree: 7.71.

> [!NOTE]
> Zero violations is an empty-ellipse topological invariant rather than a recall booster. While no-repair retains obsolete shortcut edges that yield high recall on static query sets, K-HNSW eliminates invalid certificates (achieving exactly 0 violations) while drift-moved rebuild restores self-findability to 1.000 and recovers in-degrees without leaving isolated nodes.



