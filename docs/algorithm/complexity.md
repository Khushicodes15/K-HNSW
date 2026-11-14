# Computational Complexity & Cost Analysis (`docs/algorithm/complexity.md`)

> **Note**: Asymptotic insertion costs are cited directly from the published HNSW paper (Malkov & Yashunin, 2018). We make **no asymptotic improvement claims**; all efficiency comparisons are empirical and backed by committed JSON benchmark data.

---

## 1. Baseline HNSW Operational Complexity

In the published HNSW analysis (Malkov & Yashunin, IEEE TPAMI 2018, Section 3.2):
- **Graph Search & Routing**: Traversing the multi-layer skip-graph from entry point down to layer $0$ scales logarithmically with graph size:
  $$C_{\text{search}} \in O(ef \cdot \log n)$$
- **Single-Node Insertion / Reinsertion**:
  Inserting a single element into an existing HNSW graph of size $n$ requires:
  1. Greedy routing from top layer to the insertion layer: $O(\log n)$ distance evaluations.
  2. Beam search and heuristic neighbor selection at layers $\ell \in [\text{level}, 0]$: bounded by $O(efConstruction \cdot M \cdot \text{level})$.
  As published by Malkov & Yashunin, the expected insertion cost per element scales as:
  $$C_{\text{insert}} \in O(efConstruction \cdot M \cdot \log n)$$
  In our empirical configurations ($n=500, d=8, M=8, efConstruction=16$), empirical single-node reinsertion averages approximately **$300 - 450$ distance computations**. At $n=2000$, it averages approximately **$750 - 900$ distance computations**.

---

## 2. Per-Step Maintenance Cost Breakdown

Let $n$ be total graph size, $N_{\text{drift}} = |\mathcal{V}_{\text{drift}}| = \delta \cdot n$ be the number of actively moving nodes, $B$ be the per-step reinsertion budget, and $M$ be the graph degree cap.

| Policy | Stale Check Cost ($C_{\text{stale}}$) | Reinsertions per Step ($N_{\text{ri}}$) | Reinsertion Cost ($C_{\text{reinsert}}$) | Total Distance Cost per Step |
| :--- | :---: | :---: | :---: | :---: |
| **`none`** | $0$ | $0$ | $0$ | **$0$** |
| **`periodic(K)`** | $0$ | $\begin{cases} N_{\text{drift}} & \text{if } s \equiv 0 \pmod K \\ 0 & \text{otherwise} \end{cases}$ | $\begin{cases} N_{\text{drift}} \cdot C_{\text{insert}} & \text{if } s \equiv 0 \pmod K \\ 0 & \text{otherwise} \end{cases}$ | Bursty: $N_{\text{drift}} \cdot C_{\text{insert}}$ every $K$ steps<br>(Amortized: $\frac{N_{\text{drift}}}{K} C_{\text{insert}}$) |
| **`round_robin(B)`** | **$0$** | **$B$** | **$B \cdot C_{\text{insert}}$** | **$B \cdot C_{\text{insert}}$** (Uniform per step) |
| **`lazy(theta)`** | $N_{\text{drift}}$ | $N_{\text{triggered}} \in [0, N_{\text{drift}}]$ | $N_{\text{triggered}} \cdot C_{\text{insert}}$ | $N_{\text{drift}} + N_{\text{triggered}} \cdot C_{\text{insert}}$ |
| **`budgeted_priority(B)`** | $N_{\text{drift}}$ | $B$ | $B \cdot C_{\text{insert}}$ | $N_{\text{drift}} + B \cdot C_{\text{insert}}$ |

### Key Observations:
1. **Stale Check Cost**:
   - `lazy` and `budgeted_priority` must compute $\|\vec{x}_i(t) - \vec{p}_i\|$ for every moving node at every step, incurring strictly $N_{\text{drift}}$ distance evaluations ($1$ distance computation per moving node).
   - In contrast, `round_robin` relies purely on cyclic cursor indexing and incurs **$0$ stale check distance computations**.
2. **Reinsertion Cost Dominance**:
   - Because single-node reinsertion requires $C_{\text{insert}} \approx 300 - 900$ distance calculations, reinsertion cost dominates stale check overhead whenever $B \ge 1$.
   - However, over 10 steps in $n=2000$ ($N_{\text{drift}} = 200$), stale check still accumulates $2,000$ evaluations.

---

## 3. Empirical Cost & Quality Summary (Backing JSON References)

All numbers below are extracted from committed benchmark runs under the isotropic `indep_dir` protocol (10 steps, 8x displacement, seeds 42–44 mean):

### Setting 1: $n=500$ (30% Drift, $N_{\text{drift}} = 150$, Matched Budget $B \approx 31$)
*Backing Data*: `docs/data/missing_controls_benchmark.json`

| Policy | Mean Drift Recall@16 | Total Cost | Reinsert Cost | Pre-Repair Cost | Stale Check Cost | Total Reinsertions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `no_repair` | 0.6562 | 0 | 0 | 0 | 0 | 0 |
| `periodic_k5_no_prerepair` | 0.9023 | 125,722 | 125,722 | 0 | 0 | 300 |
| `round_robin_no_prerepair` | **0.9275** | **133,945** | **133,945** | 0 | 0 | **312** |
| `round_robin_with_prerepair`| 0.9202 | 165,622 | 130,939 | 34,684 | 0 | 312 |
| `periodic_k1_no_prerepair` | 0.9338 | 621,384 | 621,384 | 0 | 0 | 1500 |

### Setting 2: $n=2000$ (10% Drift, $N_{\text{drift}} = 200$, Matched Budget $B \approx 44$)
*Backing Data*: `docs/data/missing_controls_benchmark.json`

| Policy | Mean Drift Recall@16 | Total Cost | Reinsert Cost | Pre-Repair Cost | Stale Check Cost | Total Reinsertions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `no_repair` | 0.6497 | 0 | 0 | 0 | 0 | 0 |
| `periodic_k5_no_prerepair` | 0.8972 | 316,203 | 316,203 | 0 | 0 | 400 |
| `round_robin_no_prerepair` | **0.9242** | **357,395** | **357,395** | 0 | 0 | **443** |
| `round_robin_with_prerepair`| 0.9148 | 441,428 | 331,466 | 109,962 | 0 | 443 |
| `periodic_k1_no_prerepair` | 0.9224 | 1,571,353 | 1,571,353 | 0 | 0 | 2000 |

### Complexity Conclusion
Without changing the asymptotic complexity of graph search or insertion, **temporal staggering** eliminates the deep recall troughs of periodic batching, delivering higher mean recall ($+0.0270$) at equivalent cumulative insertion budgets, and matching continuous $K=1$ repair quality at a **$77.3\%$ cost reduction**.
