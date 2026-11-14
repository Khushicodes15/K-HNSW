# Maintenance Policy Specification (`docs/algorithm/policies.md`)

> **Implementation**: [`src/khnsw/core/scheduling/policy.py`](../../src/khnsw/core/scheduling/policy.py)  
> **Backing Benchmark Data**: `docs/data/controls_benchmark.json`, `docs/data/missing_controls_benchmark.json`, `docs/data/lazy_reinsertion_benchmark.json`.

---

## 1. Overview & Architecture

The maintenance subsystem continuously regulates graph connectivity as node coordinates drift over time.
All policies conform to the `MaintenancePolicy` public API, implemented under Core Purity rules (standard library + NumPy only):

```python
class MaintenancePolicy:
    def __init__(
        self,
        policy: Literal["none", "periodic", "round_robin", "lazy", "budgeted_priority"] = "lazy",
        K: int = 1,
        B: int = 1,
        theta: float = 1.0,
        pre_repair: bool = False,
        seed: int = 42,
        meter: CostMeter | None = None,
        emitter: EventEmitter | None = None,
    ) -> None: ...
```

---

## 2. Policy Specifications

### Policy 1: `none` (No Maintenance Baseline)
* **Inputs**: Graph index $G$, position matrix $X(t)$.
* **Per-Step Procedure**: Updates vector coordinates in the underlying `GraphStore`. Performs no graph mutations.
* **Cost Categories**: $0$ distance computations across all categories.
* **Empirical Measurement**:
  - Under radial drift: Retained deceptively high recall ($>0.90$ at $10\times$).
  - Under isotropic drift (`indep_dir`, 8x displacement): Recall collapses from $0.98$ to **$0.6540$** ($ef=16$). Backed by `docs/data/controls_benchmark.json`.

---

### Policy 2: `periodic` (Bursty Batch Reinsertion)
* **Inputs**: Period $K \ge 1$, optional `pre_repair: bool`.
* **Per-Step Procedure**:
  - At step $s$, if $s \equiv 0 \pmod K$, reinserts **all** $|\mathcal{V}_{\text{drift}}|$ moving nodes.
  - If `pre_repair=True`, runs `in_neighbour_pre_repair` on the moving nodes before reinsertion.
  - Reinsertion preserves each node's assigned maximum level: `_reinsert_node(index, node_id, level=node_level)`.
* **Cost Categories**:
  - `stale_check`: $0$.
  - `reinsert`: Reinsertion routing down the hierarchy for all moving nodes.
  - `pre_repair`: Heuristic neighbor pruning cost (if enabled).
* **Parameters Tested**: $K \in \{1, 2, 5\}$.
* **Empirical Measurement**:
  - $K=1$ achieves high recall ($0.9224$ in $n=2000$) but requires an unsustainable $1,571,353$ distance calculations (`docs/data/missing_controls_benchmark.json`).
  - $K=5$ exhibits severe recall degradation between maintenance bursts, yielding mean drift recall of $0.8972$ at cost $316,203$.

---

### Policy 3: `round_robin` (Staggered Phase-Offset Reinsertion)
* **Inputs**: Per-step budget $B \ge 1$, optional `pre_repair: bool`.
* **Per-Step Procedure**:
  - Maintains a monotonic cyclic cursor: `cursor = 0`.
  - At step $s$, selects exactly $B$ nodes in cyclic order:
    $$\text{selected} = \{\mathcal{V}_{\text{drift}}[(\text{cursor} + j) \pmod{|\mathcal{V}_{\text{drift}}|}] \mid j \in [0, B-1]\}$$
    $$\text{cursor} \leftarrow \text{cursor} + B$$
  - Reinserts each selected node via `_reinsert_node` preserving hierarchy level.
  - Acts mathematically as **periodic reinsertion with per-node phase offsets**: Each node is reinserted once every $|\mathcal{V}_{\text{drift}}| / B$ steps, but maintenance load is distributed evenly across time.
* **Cost Categories**:
  - `stale_check`: $0$ (zero distance evaluations to select nodes).
  - `reinsert`: Exactly $B$ node reinsertions per step.
  - `pre_repair`: $0$ (unless `pre_repair=True`).
* **Empirical Measurement**:
  - In $n=2000$ (`docs/data/missing_controls_benchmark.json`), unassisted `round_robin` achieves **$0.9242$** mean drift recall at cost **$357,395$**.
  - Matches or exceeds $K=1$ periodic recall ($0.9224$) at **$22.7\%$ of its cost**, and decisively beats $K=5$ periodic repair ($0.8972$) at matched reinsertion volume ($+0.0270$ recall advantage).

---

### Policy 4: `lazy` (Displacement-Bounded Reinsertion)
* **Inputs**: Sensitivity threshold $\theta > 0$, optional `pre_repair: bool`.
* **Departure Bound Model**:
  - For each moving node $i$, establishes an anchor $\vec{p}_i \leftarrow \vec{x}_i(0)$ and a protective radius:
    $$r_i = \min_{j \in \mathcal{N}_0(i)} \|\vec{p}_i - \vec{x}_j(0)\|$$
  - At step $s$, evaluates displacement from anchor:
    $$d_i(t) = \|\vec{x}_i(t) - \vec{p}_i\|$$
  - A node is marked stale if:
    $$d_i(t) > \theta \cdot r_i$$
  - Triggered nodes are reinserted, and their anchors and radii are refreshed from their newly formed level-0 neighborhoods: $\vec{p}_i \leftarrow \vec{x}_i(t_s), r_i \leftarrow \min_{j \in \mathcal{N}_0^{\text{new}}(i)} \|\vec{x}_i(t_s) - \vec{x}_j(t_s)\|$.
* **Cost Categories**:
  - `stale_check`: Exactly $|\mathcal{V}_{\text{drift}}|$ distance computations per step.
  - `reinsert`: Cost of reinserting triggered nodes.
* **Parameters Tested**: $\theta \in \{0.25, 0.5, 1.0, 2.0\}$.
* **Pre-Registered Rule Outcome**:
  - In Phase 5n, the pre-registered decision rule specified that adaptive policies must beat `round_robin` by $\ge 0.01$ mean recall at matched count to be claimed superior.
  - Actual margins: $+0.0080$ (`indep_dir`, $n=500$), $+0.0091$ (`RandomState`, $n=500$), $+0.0042$ (`indep_dir`, $n=2000$).
  - **Result**: The pre-registered threshold was **not met**. The adaptive displacement trigger is **NOT claimed to beat round-robin**.

---

### Policy 5: `budgeted_priority` (Priority-Ranked Staggering)
* **Inputs**: Per-step budget $B \ge 1$.
* **Per-Step Procedure**:
  - Evaluates displacement ratio for all moving nodes:
    $$\text{score}_i = \frac{\|\vec{x}_i(t) - \vec{p}_i\|}{r_i}$$
  - Sorts moving nodes by $\text{score}_i$ descending.
  - Selects the top $B$ nodes with $\text{score}_i \ge 1.0$ (or largest score if none $\ge 1.0$).
  - Reinserts the selected $B$ nodes and resets their anchors and radii.
* **Cost Categories**:
  - `stale_check`: $|\mathcal{V}_{\text{drift}}|$ distance computations per step.
  - `reinsert`: Exactly $B$ node reinsertions per step.
* **Empirical Measurement**:
  - Achieves $0.9238$ recall in $n=500$ at cost $147,733$ (`docs/data/controls_benchmark.json`).
  - Did not beat cyclic round-robin ($\Delta = -0.0010$ vs $0.9248$), while incurring extra `stale_check` overhead.

---

## 3. Alternative Approaches

### The In-Neighbor Pre-Repair Option (`pre_repair`)
- **Mechanism**: Before reinserting node $u$, iterates over static in-neighbors $w \in \mathcal{N}_{\text{in}}(u)$ and prunes $u$ from $w$'s neighbor list if an alternative edge satisfies the heuristic diversity rule.
- **Empirical Status**: **Measured Not Useful**. In `docs/data/missing_controls_benchmark.json`, adding pre-repair degraded recall by $0.007 - 0.010$ while increasing maintenance cost by $+24\% - 25\%$.

### The Edge-Certificate Kinetic Engine (`KineticDetector`)
- **Mechanism**: Maintains kinetic Voronoi certificates on level-0 edges, tracking certificate failure events in a priority heap (`KineticEventHeap`).
- **Empirical Status**: **Measured Not Useful Under Isotropic Drift**. In 5l (`docs/data/harness_conflict_benchmark.json`), certificate repair scored lower recall than 1-pass reinsertion ($0.8549$ vs $0.9029$) at $5.7\times$ higher cost ($362\text{k}$ vs $63\text{k}$; unmatched in cost). In 5n, `cert_triggered` cost $1,598,047$ for $0.9089$ recall, compared to $357,395$ cost and $0.9242$ recall for `round_robin` ($4.5\times$ more expensive for worse recall).
