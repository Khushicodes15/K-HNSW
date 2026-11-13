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
- **Note on Recall**: For small displacements in dense graphs, the no-repair control achieves recall $\approx 1.0000$ because minor coordinate perturbations do not immediately disrupt global routing connectivity. The primary purpose of repair is actively preserving the empty-ellipse Delaunay/certificate invariant to prevent structural degradation over extended trajectories.

### 6.2 Empirical Measurements

#### Discrete Moves ($n=200, M=8, d=2, \text{seed}=44$)

| Method | Recall@10 | Search | Planning | Store | Detector | Total Dist | Violations | Repaired |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scale 0.01 (disp=0.0016)** | | | | | | | | |
| no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| ours | 1.0000 | 362 | 411 | 292 | 0 | 1065 | 0 | 1 |
| our emulation of hnswlib-update-style | 1.0000 | 0 | 303 | 0 | 0 | 303 | 217 | 30 |
| our emulation of MN-RU-style (beta) | 1.0000 | 0 | 129 | 0 | 0 | 129 | 56 | 24 |
| **Scale 0.05 (disp=0.0081)** | | | | | | | | |
| no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| ours | 1.0000 | 362 | 414 | 292 | 0 | 1068 | 0 | 1 |
| our emulation of hnswlib-update-style | 1.0000 | 0 | 303 | 0 | 0 | 303 | 217 | 30 |
| our emulation of MN-RU-style (beta) | 1.0000 | 0 | 129 | 0 | 0 | 129 | 56 | 24 |
| **Scale 0.20 (disp=0.0325)** | | | | | | | | |
| no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| ours | 1.0000 | 367 | 485 | 344 | 0 | 1196 | 0 | 1 |
| our emulation of hnswlib-update-style | 1.0000 | 0 | 303 | 0 | 0 | 303 | 221 | 30 |
| our emulation of MN-RU-style (beta) | 1.0000 | 0 | 129 | 0 | 0 | 129 | 60 | 24 |

#### Discrete Moves ($n=200, M=8, d=8, \text{seed}=50$)

| Method | Recall@10 | Search | Planning | Store | Detector | Total Dist | Violations | Repaired |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scale 0.01 (disp=0.0167)** | | | | | | | | |
| no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ours | 1.0000 | 885 | 603 | 2396 | 0 | 3884 | 0 | 0 |
| our emulation of hnswlib-update-style | 1.0000 | 0 | 10259 | 0 | 0 | 10259 | 7 | 19 |
| our emulation of MN-RU-style (beta) | 1.0000 | 0 | 1431 | 0 | 0 | 1431 | 154 | 29 |
| **Scale 0.05 (disp=0.0837)** | | | | | | | | |
| no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 9 | 0 |
| ours | 1.0000 | 884 | 1049 | 2726 | 0 | 4659 | 0 | 1 |
| our emulation of hnswlib-update-style | 1.0000 | 0 | 10328 | 0 | 0 | 10328 | 13 | 23 |
| our emulation of MN-RU-style (beta) | 1.0000 | 0 | 1434 | 0 | 0 | 1434 | 161 | 29 |
| **Scale 0.20 (disp=0.3348)** | | | | | | | | |
| no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 33 | 0 |
| ours | 1.0000 | 881 | 1780 | 3128 | 0 | 5789 | 0 | 1 |
| our emulation of hnswlib-update-style | 1.0000 | 0 | 10374 | 0 | 0 | 10374 | 27 | 30 |
| our emulation of MN-RU-style (beta) | 1.0000 | 0 | 1386 | 0 | 0 | 1386 | 171 | 33 |

#### Discrete Moves ($n=1000, M=8, d=2, \text{seed}=44$)

| Method | Recall@10 | Search | Planning | Store | Detector | Total Dist | Violations | Repaired |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scale 0.01 (disp=0.0008)** | | | | | | | | |
| no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| ours | 1.0000 | 2228 | 2007 | 1551 | 0 | 5786 | 0 | 1 |
| our emulation of hnswlib-update-style | 1.0000 | 0 | 5887 | 0 | 0 | 5887 | 911 | 137 |
| our emulation of MN-RU-style (beta) | 1.0000 | 0 | 860 | 0 | 0 | 860 | 381 | 128 |
| **Scale 0.05 (disp=0.0038)** | | | | | | | | |
| no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 15 | 0 |
| ours | 1.0000 | 2223 | 2189 | 1659 | 0 | 6071 | 0 | 1 |
| our emulation of hnswlib-update-style | 1.0000 | 0 | 5885 | 0 | 0 | 5885 | 917 | 142 |
| our emulation of MN-RU-style (beta) | 1.0000 | 0 | 860 | 0 | 0 | 860 | 390 | 128 |
| **Scale 0.20 (disp=0.0150)** | | | | | | | | |
| no-repair control | 1.0000 | 0 | 0 | 0 | 0 | 0 | 42 | 0 |
| ours | 1.0000 | 2231 | 2580 | 1896 | 0 | 6707 | 0 | 1 |
| our emulation of hnswlib-update-style | 1.0000 | 0 | 5914 | 0 | 0 | 5914 | 929 | 152 |
| our emulation of MN-RU-style (beta) | 1.0000 | 0 | 860 | 0 | 0 | 860 | 407 | 128 |

#### Discrete Moves ($n=1000, M=8, d=8, \text{seed}=50$)

| Method | Recall@10 | Search | Planning | Store | Detector | Total Dist | Violations | Repaired |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scale 0.01 (disp=0.0139)** | | | | | | | | |
| no-repair control | 0.9990 | 0 | 0 | 0 | 0 | 0 | 7 | 0 |
| ours | 0.9990 | 6322 | 3940 | 19687 | 0 | 29949 | 0 | 1 |
| our emulation of hnswlib-update-style | 0.9990 | 0 | 83527 | 0 | 0 | 83527 | 25 | 115 |
| our emulation of MN-RU-style (beta) | 0.9990 | 0 | 15467 | 0 | 0 | 15467 | 677 | 126 |
| **Scale 0.05 (disp=0.0696)** | | | | | | | | |
| no-repair control | 0.9990 | 0 | 0 | 0 | 0 | 0 | 40 | 0 |
| ours | 0.9990 | 6277 | 5995 | 20988 | 0 | 33260 | 0 | 1 |
| our emulation of hnswlib-update-style | 0.9990 | 0 | 83069 | 0 | 0 | 83069 | 35 | 131 |
| our emulation of MN-RU-style (beta) | 0.9990 | 0 | 15265 | 0 | 0 | 15265 | 693 | 138 |
| **Scale 0.20 (disp=0.2784)** | | | | | | | | |
| no-repair control | 0.9990 | 0 | 0 | 0 | 0 | 0 | 164 | 0 |
| ours | 0.9990 | 6314 | 11702 | 23988 | 0 | 42004 | 0 | 1 |
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

