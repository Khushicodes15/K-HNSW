# Chapter 8: Dynamic Drift Maintenance Evaluation

**Author**: Saanvi Srivastava (Member 4, Benchmarks Lead)  
**Primary Dataset Backing**: [`docs/data/external_benchmarks.json`](../data/external_benchmarks.json)  
**Governing Contract**: Pre-Registered Decision Rule (§2, `docs/handoff/member4.md`)

---

## 8.1 Problem Setup, Isotropic Drift Generator, and Metrics

### 8.1.1 Problem Formulation & The Kinetic Setting
In streaming retrieval and dynamic vector databases, database items frequently undergo coordinate updates while search queries continue to arrive concurrently. When vectors move without graph structural adjustments, edge vectors become geometrically misaligned with the spatial topology, leading to severe recall collapse.

We evaluate graph maintenance on graphs initialized with standard HNSW structural parameters ($M=8$, $M_0=16$, $efConstruction=16$, Euclidean $L_2$ distance) across a discrete sequence of timestamps $t \in [0, t_{\text{final}}]$ divided into $T=10$ simulation steps.

### 8.1.2 Movement Protocol: Independent Isotropic Drift (`indep_dir`)
To avoid the **radial drift hazard** identified in project Phases 5g–5k (where coupling PRNG streams caused all velocity vectors to satisfy $\cos(\vec{v}_i, \vec{x}_i(0)) \equiv +1.000$, masking graph degradation), all experiments in this suite enforce three strictly decoupled pseudorandom number generator streams:
1. $\text{Stream}_1$ (`seed`): Draws initial Gaussian point coordinates $X_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$.
2. $\text{Stream}_2$ (`seed + 1`): Subsamples the moving node subset $\mathcal{M} \subset \{1, \dots, n\}$ of size $|\mathcal{M}| = \lceil \delta \cdot n \rceil$ ($\delta = 0.30$).
3. $\text{Stream}_3$ (`seed + 2`): Samples isotropic unit velocity directions $\hat{u}_i \in \mathbb{S}^{d-1}$ uniformly over the hypersphere.

Speed is calibrated against the empirical mean nearest-neighbor distance $\bar{d}_{\text{1-NN}}(X_0)$:
$$\|\vec{v}_i\| = \frac{\lambda \cdot \bar{d}_{\text{1-NN}}(X_0)}{t_{\text{final}}}$$
where $\lambda \in \{1\times, 3\times, 8\times, 27\times\}$ specifies the displacement multiplier.

### 8.1.3 Metrics & Accounting Rules
1. **Recall@$k$ against Instantaneous Ground Truth**:
   At every step $t$, brute-force exact $k$-nearest neighbors are calculated against the true coordinates $X(t)$. We record Recall@10 at beam width $ef=16$ and $ef=32$ on drifting queries.
2. **Computational Cost Partitioning**:
   - `stale_check`: Pairwise distances evaluated to test node displacement against threshold $\theta \cdot r_i$ (tracked via `CostMeter`).
   - `reinsert`: Algorithmic distance computations evaluated during fixed-level node reinsertion.
   - `setup_cost`: Construction distance computations, reported separately from dynamic maintenance.

---

## 8.2 Controlled Comparison on Synthetic Gaussian Data

We evaluate 7 maintenance policies across 5 independent random seeds ($\{42, 43, 44, 45, 46\}$) under $\delta = 30\%$ drifting nodes and $\lambda = 8\times$ displacement:

### Table 8.1: Performance on Synthetic Gaussian Data ($d=8$, 10 Steps, Mean over 5 Seeds)

| Size | Policy Arm | Parameters | Mean Drift Rec@16 | Final Drift Rec@16 | Mean Cost (Dists) | Reinsertions |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **$n=500$** | `no_repair` | — | $0.6843 \pm 0.009$ | $0.5026 \pm 0.018$ | $0$ | $0$ |
| | `periodic_k1` | $K=1$ | $0.7898 \pm 0.021$ | $0.6922 \pm 0.037$ | $131,426 \pm 1,327$ | $1500$ |
| | `periodic_k5` | $K=5$ | $0.7829 \pm 0.016$ | $0.6850 \pm 0.040$ | $26,576 \pm 233$ | $300$ |
| | `round_robin` | $B=30$ | $\mathbf{0.8000 \pm 0.014}$ | $\mathbf{0.7077 \pm 0.039}$ | $27,955 \pm 522$ | $300$ |
| | `lazy_theta_1.0` | $\theta=1.0$ | $0.8077 \pm 0.017$ | $0.7061 \pm 0.035$ | $64,242 \pm 959$ | $666$ |
| | `lazy_theta_0.5` | $\theta=0.5$ | $0.7997 \pm 0.022$ | $0.7072 \pm 0.039$ | $105,735 \pm 1,327$ | $1165$ |
| | `budgeted_priority` | $B=30$ | $0.7981 \pm 0.017$ | $0.7107 \pm 0.042$ | $30,958 \pm 386$ | $300$ |
| **$n=2000$** | `no_repair` | — | $0.5977 \pm 0.007$ | $0.4137 \pm 0.012$ | $0$ | $0$ |
| | `periodic_k1` | $K=1$ | $0.7259 \pm 0.009$ | $0.6277 \pm 0.018$ | $628,095 \pm 5,732$ | $6000$ |
| | `periodic_k5` | $K=5$ | $0.7078 \pm 0.010$ | $0.6261 \pm 0.018$ | $132,600 \pm 1,180$ | $1200$ |
| | `round_robin` | $B=120$ | $\mathbf{0.7222 \pm 0.009}$ | $\mathbf{0.6385 \pm 0.019}$ | $140,706 \pm 2,104$ | $1200$ |
| | `lazy_theta_1.0` | $\theta=1.0$ | $0.7381 \pm 0.010$ | $0.6341 \pm 0.019$ | $328,684 \pm 4,772$ | $2853$ |
| | `lazy_theta_0.5` | $\theta=0.5$ | $0.7303 \pm 0.008$ | $0.6391 \pm 0.017$ | $514,240 \pm 5,236$ | $4780$ |
| | `budgeted_priority` | $B=120$ | $0.7251 \pm 0.011$ | $0.6418 \pm 0.018$ | $153,605 \pm 1,842$ | $1200$ |

### 8.2.1 Analysis of Observations
1. **Unmanaged Drift Collapse**: In both $n=500$ and $n=2000$, `no_repair` experiences monotonic degradation, dropping to $50.2\%$ and $41.4\%$ final recall respectively.
2. **Temporal Staggering vs. Batch Periodic**:
   - At $n=500$, `round_robin` ($B=30$) achieves $+0.0171$ (+1.71 pp) higher mean recall than `periodic_k5` ($0.8000$ vs $0.7829$) at matched cumulative reinsertions ($300$).
   - At $n=2000$, `round_robin` ($B=120$) achieves $+0.0144$ (+1.44 pp) higher mean recall than `periodic_k5` ($0.7222$ vs $0.7078$) at matched cumulative reinsertions ($1200$).
3. **Evaluation Against Pre-Registered Decision Rule**:
   - The pre-registered decision rule strictly requires $\Delta \text{Recall} \ge +0.0200$ across $\ge 5$ seeds.
   - Because the measured margins ($+0.0171$ and $+0.0144$) fall short of the $+0.0200$ threshold, we **formally report that the pre-registered headline hypothesis was inconclusive / not supported** under the 5-seed strict criterion.

---

## 8.3 Ablation: Budget $B$ and Sensitivity Threshold $\theta$

1. **Threshold Sensitivity $\theta$**:
   Lowering $\theta$ from $1.0$ to $0.5$ in `lazy` roughly doubles reinsertion volume ($666 \to 1165$ at $n=500$) and cost ($64\text{k} \to 105\text{k}$), yet produces essentially flat recall ($0.8077 \to 0.7997$). Frequent re-routing in high-velocity regimes introduces edge churn without improving routing accuracy.
2. **Priority Ordering vs. Pure Staggering**:
   `budgeted_priority` ($0.7981$) performs similarly to simple cyclic `round_robin` ($0.8000$), but requires evaluating pairwise displacement checks every step ($+3\text{k}$ distance computations at $n=500$). Simple deterministic cyclic round-robin provides equivalent recall preservation with zero displacement overhead.
