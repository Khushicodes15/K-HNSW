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

Evaluation on `HNSWIndex` ($n=500, M=8, d=8$, $T=0.30$ in 10 steps, 10% drifting nodes, seeds $\{42, 43, 44\}$; benchmark script `scripts/measure_drift.py` and output `docs/data/drift_benchmark.json`):

| Speed | Variant | Drift Recall@10 ($ef=16$) | Drift Recall@10 ($ef=32$) | Global Recall@10 ($ef=16$) | Global Recall@10 ($ef=32$) | Self-Find ($ef=64$) | Mean In-Deg | Violations | Total Cost |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10x** | No-Repair Control | 0.9487 | 0.9867 | 0.9613 | 0.9933 | 1.000 | 6.88 | 419.3 | 0 |
| | Ours (before drift-moved fix) | 0.8880 | 0.9653 | 0.9667 | 0.9960 | 0.993 | 6.24 | **0.0** | 37,585 |
| | **Ours (after drift-moved fix)** | **0.9013** | **0.9827** | **0.9633** | **0.9950** | **1.000** | **6.42** | **0.0** | **91,753** |
| **30x** | No-Repair Control | 0.9060 | 0.9713 | 0.9613 | 0.9933 | 1.000 | 6.88 | 460.3 | 0 |
| | Ours (before drift-moved fix) | 0.8313 | 0.9253 | 0.9683 | 0.9963 | 0.960 | 6.19 | **0.0** | 34,048 |
| | **Ours (after drift-moved fix)** | **0.8840** | **0.9693** | **0.9667** | **0.9963** | **1.000** | **6.39** | **0.0** | **62,957** |
| **100x**| No-Repair Control | 0.8627 | 0.9520 | 0.9613 | 0.9933 | 1.000 | 6.88 | 477.3 | 0 |
| | Ours (before drift-moved fix) | 0.7733 | 0.8780 | 0.9703 | 0.9963 | 0.960 | 6.23 | **0.0** | 31,050 |
| | **Ours (after drift-moved fix)** | **0.8813** | **0.9627** | **0.9703** | **0.9963** | **1.000** | **6.39** | **0.0** | 85,713 |

Under 100% drifting nodes at $30\times$ speed:
- **No-Repair Control**: Drift Recall@10 $ef=16$: 0.9750 ($ef=32$: 0.9957), Global Recall $\mathcal{N}(0, I)$ $ef=16$: 0.9813 ($ef=32$: 0.9983), Violations: 712.3, In-Degree: 7.74.
- **Ours (after drift-moved fix)**: Drift Recall@10 $ef=16$: **0.9950** ($ef=32$: **1.0000**), Global Recall $\mathcal{N}(0, I)$ $ef=16$: 0.8447 ($ef=32$: 0.9517), Violations: **0.0**, Self-Find: **1.000**, In-Degree: 7.72.

> [!NOTE]
> Zero violations is an empty-ellipse topological invariant rather than a recall booster. While no-repair retains obsolete shortcut edges across void regions that yield high recall on static query sets, K-HNSW eliminates invalid certificates (achieving exactly 0 violations) while drift-moved rebuild restores self-findability to 1.000 and recovers in-degrees without leaving isolated nodes.

---

### 8.7 Settled Graph Benchmark Findings (Phase 5h–5k Claims)

Comprehensive reconciliation across `docs/data/reconcile_benchmark.json` and `docs/data/vanilla_arms_benchmark.json` established the following verified algorithmic findings on the settled graph:

1. **Recall Relative to No-Repair (Claim i)**:
   On the settled benchmark graph, all repair methods—including K-HNSW at $ef=16$ rebuild (drifting recall $0.8927$ at $10\times$, $0.8687$ at $30\times$), K-HNSW at $ef=64$ rebuild ($0.9007$ at $10\times$, $0.8833$ at $30\times$), in-place reinsertion ($0.8813$ at $10\times$, $0.8673$ at $30\times$), and oracle reinsertion arms ($0.8733 - 0.8873$ at $10\times$, $0.8520 - 0.8607$ at $30\times$)—score strictly below `no_repair` ($0.9487$ at $10\times$, $0.9060$ at $30\times$) at evaluation $ef=16$. At evaluation $ef=32$, repair arms perform within $0.01$ of `no_repair` ($0.9867$ at $10\times$, $0.9713$ at $30\times$). Rebuild does not restore recall above the un-repaired baseline at $ef=16$.

2. **Certificate Invariants vs. Search Recall (Claim ii)**:
   Achieving zero violated certificates confirms adherence to empty-ellipse Delaunay invariants; it does not imply superior search recall. In fact, un-repaired graphs retain topologically invalid shortcut edges that yield higher recall at lower search queue capacities ($ef=16$).

3. **Construction Settle Confound (Claim iii)**:
   Running `settle_hnsw` on a freshly constructed HNSWIndex at $t=0$ removes $32.1\%$ of level-0 edges (dropping level-0 edges from $5,131$ to $3,483$ and mean level-0 out-degree from $10.26$ to $6.97$, means across seeds 42–44). Consequently, the settled baseline is structurally distinct and significantly thinned compared to a standard vanilla HNSW graph.

4. **In-Neighbour Pre-Repair (Claim v)**:
   Re-selecting static in-neighbour lists using the standard heuristic prior to reinsertion improved in-place reinsertion recall by $\approx 0.04$ at $10\times$ (from $0.8813$ to $0.9207$) and $\approx 0.02$ at $30\times$ (from $0.8673$ to $0.8893$) at $ef=16$ on the settled graph. However, performance remained below the `no_repair` baseline ($0.9487$ at $10\times$, $0.9060$ at $30\times$).

5. **Scheduler and Ablation Switches (Claim vi)**:
   Prospective scheduler prioritization policies, batching mechanisms, and alternative candidate filtering switches not implemented in `src/khnsw` are cataloged as **specified-not-built**.

---

### 8.8 Vanilla Graph Benchmark Findings and Decision Analysis (Phase 5k)

Evaluating repair strategies on the un-settled vanilla graph (`settle=False`, output `docs/data/vanilla_arms_benchmark.json`) yielded the following findings:

#### 1. Harness Configuration & Discrepancy Note (Claim iv)
- Two harnesses disagree on `no_repair` and `ceiling` recall for the same nominal setting ($n=500, d=8, M=8, 10\%\text{ drift}, T=0.30$):
  - In Phase 5j (`scripts/measure_vanilla_headroom.py`), `no_repair` / `ceiling` drifting recall@10 ($ef=16$) was **0.8907** / **0.9933** at $10\times$ (mean across seeds 42–44).
  - In Phase 5k (`scripts/measure_vanilla_arms.py`), `no_repair` / `ceiling` drifting recall@10 ($ef=16$) was **0.9693** / **0.9673** at $10\times$ (mean across seeds 42–44).
- **Result of Item A Controlled Test**: When fed identical inputs ($X_0$, `drifting_nodes`, and `LinearDrift` generated once at seed 42), both code paths yield mathematically identical outputs (`no_repair` 0.8720, `ceiling` 0.9960 under the 5j inputs; `no_repair` 0.9740, `ceiling` 0.9600 under the 5k inputs). The protocols evaluate identical definitions of recall, ground truth, displacement ($3.0\times$ mean 1-NN at $10\times$), and ceiling level allocation.
- **Displacement Curves (Item B)**: Sweeping displacement multiples $\{1, 3, 9, 27\}\times$ mean 1-NN distance reveals that the two curves do not coincide: the 5j generator collapses sharply ($0.9833 \to 0.8907 \to 0.7413 \to 0.6520$) creating large headroom, whereas the 5k generator degrades much more slowly ($0.9853 \to 0.9693 \to 0.9400 \to 0.9087$). Neither curve is asserted as universally normative beyond the empirical sweep data in `docs/data/harness_conflict_benchmark.json`.
- Total drifting displacement corresponds to $3.0\times$ mean 1-NN distance at $10\times$ speed and $9.0\times$ mean 1-NN distance at $30\times$ speed ($T=0.30$).

#### 2. Decision Table on the Vanilla Graph (Claim iv continued)
Evaluated across seeds $\{42, 43, 44\}$ on the vanilla graph (eval $ef=16/32$, 10% drift):

| Speed | Arm | Drift Rec@10 ($ef=16$) | Drift Rec@10 ($ef=32$) | Static Rec@10 ($ef=16$) | Violated | Drift Cost | Setup Cost | Reaches $\ge \text{ceil}-0.01$ at $< 50\%$ Cost? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10x** | 1 no repair | 0.9693 | 0.9933 | 0.9890 | 2482.7 | 0 | 0 | **YES** |
| **10x** | 2 ceiling (rebuild all 500) | 0.9673 | 0.9953 | 0.9980 | 2165.3 | 124,964 | 0 | No |
| **10x** | 3 inplace reinsert once | 0.9120 | 0.9760 | 0.9890 | 1757.3 | 22,166 | 0 | No |
| **10x** | 4 inplace reinsert + prerepair once | 0.9127 | 0.9787 | 0.9887 | 679.7 | 34,646 | 0 | No |
| **10x** | 5a reinsert every step (no prerepair) | 0.9100 | 0.9760 | 0.9887 | 1731.0 | 221,283 | 0 | No |
| **10x** | 5b reinsert every step (+ prerepair) | 0.9187 | 0.9800 | 0.9867 | 432.7 | 253,831 | 0 | No |
| **10x** | 6a periodic k=2 (+ prerepair) | 0.9193 | 0.9827 | 0.9873 | 491.7 | 131,640 | 0 | No |
| **10x** | 6b periodic k=5 (+ prerepair) | 0.9180 | 0.9813 | 0.9877 | 605.7 | 58,326 | 0 | No |
| **10x** | 7a ours t0 repaired (ef=16) | 0.8927 | 0.9813 | 0.9767 | 0.0 | 91,753 | 80,061 | No |
| **10x** | 7a ours t0 repaired (ef=64) | 0.9013 | 0.9827 | 0.9767 | 0.0 | 201,997 | 80,061 | No |
| **10x** | 7b ours t0 exempted (ef=16) | 0.9033 | 0.9833 | 0.9893 | 1255.0 | 126,311 | 0 | No |
| **10x** | 7b ours t0 exempted (ef=64) | 0.8973 | 0.9800 | 0.9877 | 1017.3 | 241,781 | 0 | No |
| **30x** | 1 no repair | 0.9400 | 0.9867 | 0.9890 | 2506.3 | 0 | 0 | **YES** |
| **30x** | 2 ceiling (rebuild all 500) | 0.9393 | 0.9880 | 0.9987 | 2129.3 | 124,496 | 0 | No |
| **30x** | 3 inplace reinsert once | 0.9000 | 0.9773 | 0.9890 | 1746.3 | 22,028 | 0 | No |
| **30x** | 4 inplace reinsert + prerepair once | 0.8887 | 0.9700 | 0.9890 | 676.0 | 33,686 | 0 | No |
| **30x** | 5a reinsert every step (no prerepair) | 0.9007 | 0.9773 | 0.9887 | 1737.3 | 217,168 | 0 | No |
| **30x** | 5b reinsert every step (+ prerepair) | 0.8767 | 0.9620 | 0.9863 | 510.3 | 231,106 | 0 | No |
| **30x** | 6a periodic k=2 (+ prerepair) | 0.8787 | 0.9620 | 0.9877 | 576.7 | 120,517 | 0 | No |
| **30x** | 6b periodic k=5 (+ prerepair) | 0.8873 | 0.9673 | 0.9880 | 629.3 | 55,652 | 0 | No |
| **30x** | 7a ours t0 repaired (ef=16) | 0.8687 | 0.9633 | 0.9780 | 0.0 | 62,957 | 80,061 | No |
| **30x** | 7a ours t0 repaired (ef=64) | 0.8840 | 0.9693 | 0.9783 | 0.0 | 127,038 | 80,061 | No |
| **30x** | 7b ours t0 exempted (ef=16) | 0.9007 | 0.9740 | 0.9897 | 1298.3 | 94,427 | 0 | No |
| **30x** | 7b ours t0 exempted (ef=64) | 0.9120 | 0.9780 | 0.9883 | 1213.7 | 166,113 | 0 | No |

*Note*: Only `no_repair` satisfies the criterion of reaching within $0.01$ of the ceiling's drifting recall at less than half the ceiling's cost. All active repair arms score below `no_repair` at $ef=16$.

---

### 8.9 Node-Level Displacement Tracking & Lazy Reinsertion (Phase 5m & 5n)

#### 1. Staleness Rule and Kinematic Trigger Mechanics
The `NodeDisplacementTracker` component (`src/khnsw/core/scheduling/tracker.py`) replaces global continuous edge-certificate monitoring with localized node-level displacement checks:
- **Anchor and Neighborhood Radius**: For each node $i$, upon registration or reinsertion, the tracker stores anchor $p_i \in \mathbb{R}^d$ and level-0 neighborhood radius:
  $$r_i = \max_{j \in N_0(i)} \|p_i - x_j\|_2$$
- **Staleness Condition**: Evaluated exclusively for nodes with non-zero velocity at simulation timestamp $t$:
  $$\text{stale}(i, t) \iff \|x_i(t) - p_i\|_2 > \theta \cdot r_i$$
  Incurs exactly $1$ distance computation per moving node per step (`stale_check`).
- **Maintenance Action**: When triggered, node $i$ is reinserted via `delete` and `_insert_one` preserving its original hierarchy level. Following reinsertion, anchor $p_i \leftarrow x_i(t)$ and radius $r_i$ are refreshed from the newly formed level-0 neighborhood.

#### 2. Budgeted Priority Policy
The `BudgetedPriorityTracker` component (`src/khnsw/core/scheduling/tracker.py`) bounds maintenance expenditure by enforcing a fixed per-step reinsertion budget $B$:
- At each step $t$, compute score $s_i = \|x_i(t) - p_i\|_2 / r_i$ for all moving nodes (1 check per moving node).
- Rank moving nodes in descending order of $s_i$ and reinsert at most the top $B$ nodes.
- Reinserted nodes refresh $p_i \leftarrow x_i(t)$, resetting their score to $0$ for subsequent steps.

#### 3. Protocol Audit: Radial Drift Confound Resolution (Phase 5n Item A)
Auditing velocity alignments $\cos(v_i, X_0[i])$ across seeds $\{42, 43, 44\}$ reveals a fundamental protocol distinction:
- **`default_rng` Confound**: In `LinearDrift.__init__`, velocities are sampled via `rng = np.random.default_rng(np.random.SeedSequence([self._seed, 0]))`. When $X_0$ is drawn from `default_rng(seed)`, NumPy generates the identical PRNG variates for `normals` as for $X_0$. Consequently, $v_i \propto X_0[i]$, yielding **pure radial drift** ($\cos \equiv 1.0000$ for all nodes). Under radial drift, beam searches remain artificially aligned along rays, masking recall degradation even at extreme displacements.
- **`RandomState` and `indep_dir`**: Using distinct PRNG streams for $X_0$, drifting IDs, and velocity directions (`indep_dir`) yields true isotropic drift with mean cosine $\approx 0$. Drifting nodes depart from their local graph neighborhoods, causing `no_repair` recall to drop from $0.98$ to $0.65$.

| Protocol | Mean $\cos(v, X_0)$ | Min $\cos$ | Max $\cos$ | Mean $|\cos|$ | `no_repair` 1x | 3x | 9x | 27x |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomState** | -0.0125 | -0.7542 | +0.8580 | 0.2799 | 0.9833 | 0.8907 | 0.7413 | 0.6520 |
| **default_rng** *(radial)* | **+1.0000** | **+1.0000** | **+1.0000** | **1.0000** | **0.9853** | **0.9693** | **0.9400** | **0.9087** |
| **indep_dir** *(isotropic)* | **+0.0066** | **-0.8192** | **+0.8856** | **0.3191** | **0.9840** | **0.9013** | **0.7660** | **0.6540** |

#### 4. Controls and Staggering Ablation (Phase 5n Item B, $n=500, 30\%$ Drift, $8\times$, 10 Steps)
To determine whether adaptive policies gain from per-node departure tracking or simply from temporal staggering, reinsertion counts were matched to `lazy_theta_1.0` ($C=312$ reinsertions, $B \approx 31$/step) across seeds $\{42, 43, 44\}$:

| Protocol | Arm | Mean Drf@16 | Final Drf@16 | Mean St@16 | Total Cost | Total Reinserts |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **indep_dir** | `no_repair` | 0.7458 | 0.5740 | 0.9653 | 0 | 0 |
| | `periodic_k1` | 0.9413 | 0.9164 | 0.9797 | 701,763 | 1500 |
| | `periodic_k2` | 0.9345 | 0.9067 | 0.9715 | 346,411 | 750 |
| | `periodic_k5` | 0.9019 | 0.9044 | 0.9599 | 136,079 | 300 |
| | `lazy_theta_0.5` | 0.9345 | 0.9062 | 0.9738 | 232,581 | 545 |
| | `lazy_theta_1.0` | **0.9349** | 0.9191 | 0.9694 | 135,556 | 312 |
| | `lazy_theta_2.0` | 0.9044 | 0.9116 | 0.9655 | 68,549 | 152 |
| | `round_robin` | **0.9269** | 0.9151 | 0.9699 | 134,049 | 312 |
| | `fixed_distance` | 0.9293 | 0.9118 | 0.9686 | 190,108 | 450 |
| | `random` | 0.9126 | 0.8960 | 0.9686 | 137,460 | 312 |
| | `budgeted_priority` | **0.9280** | 0.9042 | 0.9700 | 136,336 | 312 |
| **RandomState** | `lazy_theta_1.0` | **0.9329** | 0.9053 | 0.9673 | 135,313 | 312 |
| | `round_robin` | **0.9236** | 0.8927 | 0.9653 | 133,810 | 312 |
| | `budgeted_priority` | **0.9250** | 0.9007 | 0.9674 | 135,828 | 312 |

**Pre-Registered Decision Rule Outcome**:
- In `indep_dir`: `lazy_theta_1.0` exceeds `round_robin` by $+0.0080$ ($< 0.01$); `budgeted_priority` exceeds by $+0.0010$ ($< 0.01$).
- In `RandomState`: `lazy_theta_1.0` exceeds `round_robin` by $+0.0093$ ($< 0.01$); `budgeted_priority` exceeds by $+0.0014$ ($< 0.01$).
- **Conclusion**: The pre-registered $\ge 0.01$ threshold is not met. **The performance gain of adaptive lazy reinsertion over periodic batching is predominantly attributable to temporal staggering** (distributing reinsertions continuously over time rather than executing batch repairs).
- **Per-Node Radius Effect**: Fixed-distance thresholding ($D = \text{median}(r_i)$) achieves $0.9293$ recall but requires $450$ reinsertions ($+44\%$), whereas adaptive radius $\theta=1.0$ achieves $0.9349$ with $312$ reinsertions. However, because both differ from blind `round_robin` ($0.9269$) by $< 0.01$, no claim is made that per-node radii provide a significant accuracy benefit over uniform staggering.

#### 5. Large-Scale Evaluation (Phase 5n Item C, $n=2000, 10\%$ Drift, $8\times$, 10 Steps, `indep_dir`)
Evaluated across seeds $\{42, 43, 44\}$ under the unconfounded `indep_dir` protocol:

| Arm | Mean Drf@16 | Final Drf@16 | Mean St@16 | Total Cost | Stale Check | Pre-Repair | Reinsert | Total Reinserts |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `no_repair` | 0.7456 | 0.5427 | 0.9757 | 0 | 0 | 0 | 0 | 0 |
| `periodic_k1` | 0.9113 | 0.8520 | 0.9754 | 1,724,902 | 0 | 294,200 | 1,430,701 | 2000 |
| `periodic_k5` | 0.8913 | 0.8595 | 0.9740 | 356,455 | 0 | 76,125 | 280,330 | 400 |
| `round_robin` | **0.9242** | 0.8825 | 0.9775 | 357,631 | 0 | 0 | 357,631 | 443 |
| `budgeted_priority` | **0.9248** | 0.8737 | 0.9789 | 360,704 | 2,000 | 0 | 358,704 | 443 |
| `lazy_theta_1.0` | **0.9280** | 0.8727 | 0.9787 | 358,641 | 2,000 | 0 | 356,641 | 443 |

#### 6. Decommissioned Repair Components: Measured Not Useful
Rigorous measurement across Phase 5 sweeps establishes that two proposed repair mechanisms provide negative value:
1. **Edge-Certificate Failure Detection**: Tracking kinetic Voronoi edge certificates via `KineticDetector` and `CertificateStore` adds $197,164$ ($n=500$) to $456,341$ ($n=2000$) distance calculations to the maintenance budget. In $n=2000$, `cert_triggered` achieves lower recall ($0.9089$) than simple staggered reinsertion ($0.9280$) at over $4\times$ the cost ($1,598,047$ vs $358,641$). Edge certificates are **measured not useful** for point drift maintenance.
2. **In-Neighbor Pre-Repair (`in_neighbour_pre_repair`)**: Heuristic pruning of static in-neighbors before node reinsertion adds $25,000 - 300,000$ distance evaluations without improving drifting recall (e.g., $0.9304$ with pre-repair vs $0.9329$ without in $n=500$; $0.8767$ vs $0.9007$ in $30\times$ continuous drift). In-neighbor pre-repair is **measured not useful**.




