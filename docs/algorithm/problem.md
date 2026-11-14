# Problem Formulation & Evaluation Protocols (`docs/algorithm/problem.md`)

> **Context**: Dynamic metric graph maintenance under continuous coordinate motion.  
> **Committed Evidence Reference**: `docs/data/controls_benchmark.json`, `docs/data/missing_controls_benchmark.json`, `docs/data/lazy_reinsertion_benchmark.json`.

---

## 1. Problem Definition

Let $\mathcal{V} = \{0, 1, \dots, n-1\}$ be a set of $n$ indexed vectors in $\mathbb{R}^d$.
Each point $i \in \mathcal{V}$ possesses a continuous trajectory over time $t \in [0, T]$:
$$\vec{x}_i(t) = \vec{x}_i(0) + \vec{v}_i \cdot t$$
Points are partitioned into two disjoint subsets:
1. **Drifting Nodes ($\mathcal{V}_{\text{drift}}$)**: Nodes with non-zero velocities $\|\vec{v}_i\| > 0$. The drift fraction is $\delta = |\mathcal{V}_{\text{drift}}| / n$.
2. **Static Nodes ($\mathcal{V}_{\text{static}}$)**: Nodes remaining stationary: $\vec{v}_i = \mathbf{0}$, $\vec{x}_i(t) = \vec{x}_i(0)$.

A proximity graph index $G(t) = (\mathcal{V}, \mathcal{E}(t))$ is maintained across discrete evaluation timesteps $t_0 = 0 < t_1 < \dots < t_S = T$.
At each timestep $t_s$, positions are updated to $X(t_s) = [\vec{x}_0(t_s), \dots, \vec{x}_{n-1}(t_s)]^T$.
The system must maintain search accuracy across time while constrained by a per-step maintenance budget $B$.

---

## 2. Quality Metrics: Recall@k at Current Positions

Quality is evaluated as **Recall@k** ($k=10$) against exact Euclidean nearest-neighbor ground truth computed at the **current positions** $X(t_s)$.

### A. Exact Metric Definitions

1. **Drifting-Node Recall (`drift_rec`)**:
   - Queries are the current positions of the drifting nodes: $Q_{\text{drift}} = \{\vec{x}_i(t_s) \mid i \in \mathcal{V}_{\text{drift}}\}$.
   - Ground truth $\text{GT}_{\text{drift}}(i)$ is the exact top-$k$ nearest neighbors of $\vec{x}_i(t_s)$ across **all** $n$ vectors at their current positions $X(t_s)$.
   - Evaluates whether moving nodes can find their updated true neighbors and be routed to accurately.

2. **Static-Node Recall (`static_rec`)**:
   - Queries are the current positions of an unmoving subset: $Q_{\text{static}} = \{\vec{x}_j(t_s) \mid j \in \mathcal{V}_{\text{static\_eval}}\}$ (typically the first 100 static nodes).
   - Ground truth $\text{GT}_{\text{static}}(j)$ is the exact top-$k$ nearest neighbors of $\vec{x}_j(t_s)$ across all $n$ current vectors $X(t_s)$.
   - Evaluates whether drift in $\mathcal{V}_{\text{drift}}$ corrupts routing paths through the static backbone.

### B. Python Implementation (from `scripts/measure_protocol_and_controls.py`)

```python
def brute_force_knn(X: NDArray[Any], queries: NDArray[Any], k: int = 10) -> list[set[int]]:
    """Brute-force exact top-k nearest neighbors by squared Euclidean distance at CURRENT positions."""
    gt: list[set[int]] = []
    for q in queries:
        dists = np.sum((X - q) ** 2, axis=1)
        topk = np.argsort(dists)[:k]
        gt.append(set(int(x) for x in topk))
    return gt


def evaluate_recall(
    index: HNSWIndex, queries: NDArray[Any], ground_truth: list[set[int]], k: int = 10, ef: int = 16
) -> float:
    """Recall@k against precomputed ground truth sets."""
    hits = 0
    total = len(queries) * k
    for q, gt in zip(queries, ground_truth, strict=False):
        res_ids, _ = index.search(q, k=k, ef=ef)
        hits += len(set(int(x) for x in res_ids) & gt)
    return hits / total if total > 0 else 0.0
```

At step $s$, queries and ground truth are generated directly from the live position matrix `pos_step`:
```python
# scripts/measure_protocol_and_controls.py lines 381-389
q_d = pos_step[drift_nodes]
gt_d = brute_force_knn(pos_step, q_d, k=10)
q_s = pos_step[eval_static]
gt_s = brute_force_knn(pos_step, q_s, k=10)

rec16_d = evaluate_recall(index, q_d, gt_d, k=10, ef=16)
rec32_d = evaluate_recall(index, q_d, gt_d, k=10, ef=32)
rec16_s = evaluate_recall(index, q_s, gt_s, k=10, ef=16)
rec32_s = evaluate_recall(index, q_s, gt_s, k=10, ef=32)
```

---

## 3. Cost Accounting Model

Maintenance cost is measured strictly in **Euclidean distance computations**, tracked through `CostMeter` counters partitioned by operational category:

1. `stale_check`: Distance computations incurred checking whether a node has breached its departure bound ($\|\vec{x}_i(t) - \vec{p}_i\|$). Exactly $1$ distance computation per evaluated moving node per step.
2. `reinsert`: Distance computations incurred during node reinsertion (routing beam searches down the hierarchy and heuristic neighbor selection).
3. `pre_repair`: Distance computations incurred by optional in-neighbor heuristic pruning prior to reinsertion.
4. `detector`: Distance computations incurred evaluating kinetic edge certificates or geometric Voronoi failure predicates.

Total maintenance cost is:
$$\text{Cost}_{\text{total}} = C_{\text{stale\_check}} + C_{\text{reinsert}} + C_{\text{pre\_repair}} + C_{\text{detector}}$$

---

## 4. Movement Protocols & The Radial-Drift Hazard

### A. Displacement Parameterization
Motion velocity is calibrated in units of the graph's **mean 1-nearest-neighbor distance** at $t=0$:
$$\bar{d}_{\text{1-NN}} = \frac{1}{n} \sum_{i=1}^n \min_{j \ne i} \|\vec{x}_i(0) - \vec{x}_j(0)\|$$
For displacement multiplier $\lambda \in \{1\times, 3\times, 8\times, 27\times\}$ over duration $T = 0.30$, each moving node travels a total Euclidean distance of $D = \lambda \cdot \bar{d}_{\text{1-NN}}$, with speed $\|\vec{v}_i\| = D / T$.

### B. The Radial-Drift PRNG Collision Hazard

> [!WARNING] Evaluation Hazard: Pure Radial Drift
> In earlier phases (Phases 5g–5k), benchmark generators initialized both initial coordinates $X(0)$ and velocity directions $\vec{v}_i$ using NumPy's `default_rng(seed)` coupled with `LinearDrift(X0, seed=seed)`.
>
> In NumPy's `PCG64` implementation, `LinearDrift` instantiated its internal generator using `SeedSequence([seed, 0])`. Because this child stream drew numbers identical to the parent generator's stream, the drawn normal variates for velocity directions were proportional to the initial point coordinates:
> $$\vec{v}_i \propto \vec{x}_i(0) \implies \cos(\vec{v}_i, \vec{x}_i(0)) \equiv +1.0000 \quad \forall i$$
>
> **Consequence**: Under radial drift, points expand outward along rays from the origin. The angular relationships between vectors are preserved, and nearest-neighbor rank orderings experience minimal permutation. As a result, un-repaired HNSW graphs retain an artificially high recall ($>0.90$ at $10\times$ drift).
>
> **Protocol Rule**: All conclusions in this work must be derived from **isotropic drift protocols** where coordinate generation and velocity directions are strictly decoupled:
> 1. `indep_dir`: Coordinates from `default_rng(seed)`, moving IDs from `default_rng(seed+1)`, and velocity directions independently sampled from `default_rng(seed+2)`. Here, mean $|\cos(\vec{v}_i, \vec{x}_i(0))| \approx 0.0066$ (orthogonal).
> 2. `RandomState`: Legacy NumPy Mersenne Twister `RandomState(seed)`, which avoids the `SeedSequence` stream collision.
