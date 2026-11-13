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
