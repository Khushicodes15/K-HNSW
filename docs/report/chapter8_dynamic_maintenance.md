# Chapter 8: Dynamic Drift Maintenance Evaluation

**Author**: Saanvi Srivastava (Member 4, Benchmarks Lead)  
**Primary Dataset Backing**: [`docs/data/external_benchmarks.json`](../data/external_benchmarks.json)  
**Governing Contract**: Pre-Registered Decision Rule (§2, `docs/handoff/member4.md`)

---

## 8.1 Problem Setup, Isotropic Drift Generator, and Metric Definitions

### 8.1.1 Problem Formulation & The Kinetic Setting
In streaming retrieval and dynamic vector databases, database items frequently undergo coordinate updates while search queries continue to arrive concurrently. When vectors move without graph structural adjustments under the independent isotropic drift motion model, edge vectors become geometrically misaligned with the spatial topology, leading to severe recall collapse.

We evaluate graph maintenance on graphs initialized with standard HNSW structural parameters ($M=8$, $M_0=16$, Euclidean $L_2$ distance) across a sequence of timestamps $t \in [0, t_{\text{final}}]$ divided into $T=10$ simulation steps.

### 8.1.2 Movement Protocol: Independent Isotropic Drift (`indep_dir`)
To avoid the **radial drift hazard** identified in project Phases 5g–5k (where coupling PRNG streams caused all velocity vectors to satisfy $\cos(\vec{v}_i, \vec{x}_i(0)) \equiv +1.000$, masking graph degradation), all experiments in this suite enforce three strictly decoupled pseudorandom number generator streams:
1. $\text{Stream}_1$ (`seed`): Draws initial Gaussian point coordinates $X_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$.
2. $\text{Stream}_2$ (`seed + 1`): Subsamples the moving node subset $\mathcal{M} \subset \{1, \dots, n\}$ of size $|\mathcal{M}| = \lceil \delta \cdot n \rceil$ ($\delta = 0.30$).
3. $\text{Stream}_3$ (`seed + 2`): Samples isotropic unit velocity directions $\hat{u}_i \in \mathbb{S}^{d-1}$ uniformly over the hypersphere.

Speed is calibrated against the empirical mean nearest-neighbor distance $\bar{d}_{\text{1-NN}}(X_0)$:
$$\|\vec{v}_i\| = \frac{\lambda \cdot \bar{d}_{\text{1-NN}}(X_0)}{t_{\text{final}}}$$
where $\lambda = 8.0\times$ specifies the displacement multiplier ($t_{\text{final}} = 0.30$).

### 8.1.3 Metrics & Accounting Rules
1. **Recall@$k$ against Instantaneous Ground Truth**:
   At every step $t$, brute-force exact $k$-nearest neighbors are calculated against the true coordinates $X(t)$. We record both **Mean Drifting Recall@16** (average over simulation steps $1 \dots 10$) and **Final Drifting Recall@16** (at the final step $t=10$) for $k=10$ at beam width $ef=16$ on drifting queries.
2. **Computational Cost Partitioning**:
   - `stale_check`: Pairwise distances evaluated to test node displacement against threshold $\theta \cdot r_i$ (tracked via `CostMeter`).
   - `reinsert`: Algorithmic distance computations evaluated during fixed-level node reinsertion (tracked via `index.distance_computations`).
   - `setup_cost`: Construction distance computations, reported separately from dynamic maintenance.

---

## 8.2 Controlled Comparison on Synthetic Gaussian Data

We evaluate 7 maintenance policies under two distinct graph construction regimes:
1. **Configuration A (`efConstruction = 16`, Seeds 42–46)**: Bounded construction beam width as tested during rapid evaluation sweeps.
2. **Configuration B (`efConstruction = 200`, Seeds 100–104)**: Production-grade construction beam width matching standard HNSW index construction.

---

### Table 8.1A: Synthetic Gaussian Drift (`efConstruction = 16`, Seeds 42–46)
*Parameters: $M=8, M_0=16, efConstruction=16, \delta=0.30, \lambda=8.0\times, k=10, ef=16, T=10$ steps*

| Size | Policy Arm | Parameters | Mean Drift Rec@16 | Final Drift Rec@16 | Mean Cost (Dists) | Reinsertions |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **n=500** | `no_repair` | — | 0.6843 ± 0.009 | 0.5026 ± 0.018 | 0 | 0 |
| | `periodic_k1` | K=1 | 0.7898 ± 0.021 | 0.6922 ± 0.037 | 131,426 ± 1,327 | 1,500 |
| | `periodic_k5` | K=5 | 0.7829 ± 0.016 | 0.6850 ± 0.040 | 26,576 ± 278 | 300 |
| | `round_robin` | B=30 | 0.8000 ± 0.017 | 0.6882 ± 0.032 | 27,955 ± 397 | 300 |
| | `lazy_theta_1.0` | θ=1.0 | 0.7974 ± 0.016 | 0.6840 ± 0.025 | 37,456 ± 511 | 389 |
| | `lazy_theta_0.5` | θ=0.5 | 0.7903 ± 0.020 | 0.6870 ± 0.024 | 65,407 ± 743 | 703 |
| | `budgeted_priority` | B=30 | 0.8011 ± 0.017 | 0.6752 ± 0.030 | 30,088 ± 368 | 300 |
| **n=2000** | `no_repair` | — | 0.5077 ± 0.016 | 0.2004 ± 0.021 | 0 | 0 |
| | `periodic_k1` | K=1 | 0.7183 ± 0.012 | 0.6000 ± 0.008 | 651,939 ± 5,159 | 6,000 |
| | `periodic_k5` | K=5 | 0.7078 ± 0.017 | 0.6120 ± 0.024 | 132,600 ± 1,530 | 1,200 |
| | `round_robin` | B=120 | 0.7222 ± 0.013 | 0.6026 ± 0.022 | 140,706 ± 1,301 | 1,200 |
| | `lazy_theta_1.0` | θ=1.0 | 0.7260 ± 0.013 | 0.5710 ± 0.022 | 191,483 ± 2,532 | 1,604 |
| | `lazy_theta_0.5` | θ=0.5 | 0.7253 ± 0.016 | 0.5910 ± 0.026 | 342,772 ± 5,206 | 2,952 |
| | `budgeted_priority` | B=120 | 0.7272 ± 0.017 | 0.5834 ± 0.038 | 150,237 ± 2,046 | 1,200 |

#### Paired Differences: Synthetic $n=500, d=8$ (`efConstruction = 16`, Seeds 42–46)

| Seed | `round_robin` Recall@16 | `periodic_k5` Recall@16 | Paired Difference (RR − Per) | Sign |
|:---:|:---:|:---:|:---:|:---:|
| **42** | 0.8032 | 0.7846 | **+0.0186** | + |
| **43** | 0.7951 | 0.7684 | **+0.0267** | + |
| **44** | 0.8149 | 0.8001 | **+0.0148** | + |
| **45** | 0.7696 | 0.7610 | **+0.0086** | + |
| **46** | 0.8171 | 0.8002 | **+0.0169** | + |

- **Mean Difference ($\Delta$)**: **+0.0171** (+1.71 pp)
- **Sample Std ($s$)**: 0.0066
- **95% Confidence Interval**: [+0.0090, +0.0253]
- **Positive Seeds**: **5 / 5**
- **Decision Rule Outcome ($\ge +0.0200$)**: Not Supported / Inconclusive (mean $+0.0171 < +0.0200$).

#### Paired Differences: Synthetic $n=2000, d=8$ (`efConstruction = 16`, Seeds 42–46)

| Seed | `round_robin` Recall@16 | `periodic_k5` Recall@16 | Paired Difference (RR − Per) | Sign |
|:---:|:---:|:---:|:---:|:---:|
| **42** | 0.7020 | 0.6876 | **+0.0144** | + |
| **43** | 0.7301 | 0.7291 | **+0.0010** | + |
| **44** | 0.7179 | 0.6993 | **+0.0186** | + |
| **45** | 0.7403 | 0.7263 | **+0.0140** | + |
| **46** | 0.7206 | 0.6965 | **+0.0241** | + |

- **Mean Difference ($\Delta$)**: **+0.0144** (+1.44 pp)
- **Sample Std ($s$)**: 0.0085
- **95% Confidence Interval**: [+0.0038, +0.0250]
- **Positive Seeds**: **5 / 5**
- **Decision Rule Outcome ($\ge +0.0200$)**: Not Supported / Inconclusive (mean $+0.0144 < +0.0200$).

---

### Table 8.1B: Synthetic Gaussian Drift (`efConstruction = 200`, Seeds 100–104)
*Parameters: $M=8, M_0=16, efConstruction=200, \delta=0.30, \lambda=8.0\times, k=10, ef=16, T=10$ steps*

| Size | Policy Arm | Parameters | Mean Drift Rec@16 | Final Drift Rec@16 | Mean Cost (Dists) | Reinsertions |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **n=500** | `no_repair` | — | 0.7484 ± 0.011 | 0.5740 ± 0.023 | 0 | 0 |
| | `periodic_k1` | K=1 | 0.9405 ± 0.010 | 0.9132 ± 0.018 | 626,105 ± 3,553 | 1,500 |
| | `periodic_k5` | K=5 | 0.9089 ± 0.012 | 0.9164 ± 0.015 | 126,927 ± 806 | 300 |
| | `round_robin` | B=30 | 0.9356 ± 0.008 | 0.9222 ± 0.016 | 129,963 ± 745 | 300 |
| | `lazy_theta_1.0` | θ=1.0 | 0.9381 ± 0.010 | 0.9156 ± 0.018 | 137,103 ± 1,418 | 313 |
| | `lazy_theta_0.5` | θ=0.5 | 0.9422 ± 0.010 | 0.9030 ± 0.014 | 235,211 ± 4,663 | 546 |
| | `budgeted_priority` | B=30 | 0.9344 ± 0.006 | 0.9106 ± 0.016 | 132,217 ± 978 | 300 |
| **n=2000** | `no_repair` | — | 0.5689 ± 0.012 | 0.2556 ± 0.016 | 0 | 0 |
| | `periodic_k1` | K=1 | 0.9384 ± 0.007 | 0.9342 ± 0.021 | 4,356,617 ± 23,796 | 6,000 |
| | `periodic_k5` | K=5 | 0.8807 ± 0.005 | 0.9472 ± 0.013 | 904,482 ± 5,295 | 1,200 |
| | `round_robin` | B=120 | 0.9152 ± 0.004 | 0.9496 ± 0.012 | 931,379 ± 3,979 | 1,200 |
| | `lazy_theta_1.0` | θ=1.0 | 0.9350 ± 0.004 | 0.9456 ± 0.012 | 996,633 ± 4,934 | 1,291 |
| | `lazy_theta_0.5` | θ=0.5 | 0.9434 ± 0.005 | 0.9372 ± 0.016 | 1,732,144 ± 9,031 | 2,293 |
| | `budgeted_priority` | B=120 | 0.9218 ± 0.003 | 0.9422 ± 0.007 | 939,868 ± 3,151 | 1,200 |

#### Paired Differences: Synthetic $n=500, d=8$ (`efConstruction = 200`, Seeds 100–104)

| Seed | `round_robin` Recall@16 | `periodic_k5` Recall@16 | Paired Difference (RR − Per) | Sign |
|:---:|:---:|:---:|:---:|:---:|
| **100** | 0.9326 | 0.9009 | **+0.0317** | + |
| **101** | 0.9334 | 0.9164 | **+0.0170** | + |
| **102** | 0.9283 | 0.9028 | **+0.0255** | + |
| **103** | 0.9510 | 0.9279 | **+0.0231** | + |
| **104** | 0.9325 | 0.8965 | **+0.0360** | + |

- **Mean Difference ($\Delta$)**: **+0.0267** (+2.67 pp)
- **Sample Std ($s$)**: 0.0074
- **95% Confidence Interval**: [+0.0175, +0.0359]
- **Positive Seeds**: **5 / 5**
- **Decision Rule Outcome ($\ge +0.0200$)**: **Supported (Win)** (margin $+0.0267 \ge +0.0200$).

#### Paired Differences: Synthetic $n=2000, d=8$ (`efConstruction = 200`, Seeds 100–104)

| Seed | `round_robin` Recall@16 | `periodic_k5` Recall@16 | Paired Difference (RR − Per) | Sign |
|:---:|:---:|:---:|:---:|:---:|
| **100** | 0.9213 | 0.8909 | **+0.0304** | + |
| **101** | 0.9162 | 0.8784 | **+0.0378** | + |
| **102** | 0.9157 | 0.8799 | **+0.0358** | + |
| **103** | 0.9144 | 0.8769 | **+0.0375** | + |
| **104** | 0.9082 | 0.8775 | **+0.0307** | + |

- **Mean Difference ($\Delta$)**: **+0.0344** (+3.44 pp)
- **Sample Std ($s$)**: 0.0036
- **95% Confidence Interval**: [+0.0299, +0.0390]
- **Positive Seeds**: **5 / 5**
- **Decision Rule Outcome ($\ge +0.0200$)**: **Supported (Win)** (margin $+0.0344 \ge +0.0200$).

---

## 8.3 Scientific Insights & Ablations

### 8.3.1 Impact of Construction Quality (`efConstruction = 16` vs `200`)
Under `efConstruction = 16`, distance computations per node reinsertion average $\approx 93.2$ ($27,955 / 300$), because candidate exploration in upper layers is constrained. Under `efConstruction = 200`, reinsertions evaluate $\approx 433.2$ distance computations ($129,963 / 300$), matching the project's internal reference engine. Higher construction quality provides higher overall recall ($93.6\%$ vs $80.0\%$) and widens the temporal staggering advantage over batch reinsertion ($+2.67\%$ at $n=500$; $+3.44\%$ at $n=2000$) under independent isotropic drift, successfully exceeding the pre-registered $+2.0\%$ decision threshold (noting that this staggering advantage is motion-model dependent and does not generalize across all movement regimes).

### 8.3.2 Ablation of Threshold $\theta$
Lowering $\theta$ from $1.0$ to $0.5$ in `lazy` roughly doubles reinsertion volume ($313 \to 546$ at $n=500$; $1291 \to 2293$ at $n=2000$) and almost doubles computational cost ($137\text{k} \to 235\text{k}$), while producing negligible recall gains ($0.9381 \to 0.9422$). Setting $\theta = 1.0$ provides the Pareto-optimal operational balance.
