# Computational Complexity & Cost Analysis (`docs/algorithm/complexity.md`)

> **Note**: Asymptotic insertion costs are cited directly from the published HNSW paper (Malkov & Yashunin, 2018). We make **no asymptotic improvement claims**; all efficiency comparisons are empirical and backed by committed JSON benchmark data.

---

## 1. Baseline HNSW Operational Complexity

In the foundational HNSW paper (Malkov & Yashunin, arXiv:1603.09320; the section number of the complexity discussion is **UNVERIFIED**, as the paper's body was not read in full):
- **Search & Routing**: Greedy routing through the layered graph, then beam search at layer 0. The paper states that the structure offers logarithmic complexity scaling and describes it as an extension of the skip-list structure (abstract/introduction).
- **Single-node insertion**: Coarse greedy positioning in upper layers, then candidate search (pool size $efConstruction$) and link selection (degree $M$) at the element's layers down to 0.
- **Construction cost**: The paper's claimed scaling of insertion/construction cost was **not verified** by us, and we do not quote it. We make no theoretical complexity claim; cost is measured empirically with software counters (`CostMeter`).

In our empirical configurations ($n=500, d=8, M=8, efConstruction=16$), single-node reinsertion averages approximately **$300 - 450$ distance computations**. At $n=2000$, it averages approximately **$750 - 900$ distance computations**.

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
Without changing the asymptotic complexity of search or insertion, temporal staggering (`round_robin`) scored higher mean drifting recall than bursty `periodic_k5` at similar, not identical, cost: +0.0252 at n=500 (0.9275 vs 0.9023; 133,945 vs 125,722 distance computations, +6.5%) and +0.0270 at n=2000 (0.9242 vs 0.8972; 357,395 vs 316,203, +13.0%).

Against `periodic_k1` (reinsertion every step), `round_robin` has slightly higher recall at n=2000 (0.9242 vs 0.9224) at 77.3% lower cost (357,395 vs 1,571,353). At n=500 it has lower recall (0.9275 vs 0.9338, -0.0063) at 78.4% lower cost (133,945 vs 621,384).

Source: `docs/data/missing_controls_benchmark.json`. These are internal results on synthetic data (3 seeds), not tested against external baselines.
