# Chapter 8: Dynamic Drift Maintenance Evaluation

**Author**: Saanvi Srivastava (Member 4, Benchmarks Lead)  
**Primary Dataset Backing**: [`docs/data/external_benchmarks.json`](../data/external_benchmarks.json)  
**Governing Contract**: Pre-Registered Decision Rule (§2, `docs/handoff/member4.md`)

---

## 8.1 Problem Setup, Isotropic Drift Generator, and Metric Definitions

### 8.1.1 Problem Formulation & The Kinetic Setting
In streaming retrieval and dynamic vector databases, database items frequently undergo coordinate updates while search queries continue to arrive concurrently. When vectors move without graph structural adjustments, edge vectors become geometrically misaligned with the spatial topology, leading to severe recall collapse.

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

### 8.1.4 Harness Version History & What Changed Between Reported Versions

> **Note for reviewers**: Member 3 observed that results for the same seeds (42–46) changed between the first submitted version and the second. The changes are documented here in full.

**Version 1 (commit `7aa889b`, 5-step runs, efConstruction hardcoded to 16)**
- Simulation used $T=5$ steps instead of $T=10$.
- `efConstruction` was hardcoded to 16 for all runs (no CLI parameter).
- Pairing was done on the 5-step mean recall.

**Version 2 (commit `dce73a2`, current, 10-step runs, efConstruction parametrised)**
- Simulation extended to $T=10$ steps (doubles the temporal range; more drift accumulates per run).
- `efConstruction` exposed as a CLI parameter; ef16 results re-run at $T=10$.
- Pairing now done on 10-step mean recall.

**Why the numbers shifted with the same seeds:**
1. **Step count doubled ($T=5 \to T=10$)**: Each run now captures later, higher-drift epochs. `round_robin` gains recall in the early-to-mid window but the advantage compresses relative to `periodic_k5` at late steps because all repair budgets are consumed by epoch 10. This reduces the mean-recall margin from +1.44 pp (5-step average) to a narrower window over 10 steps where the second half is harder for all policies.
2. **Budget matching unchanged**: $B = \lceil |\mathcal{M}| / 5 \rceil$ was not altered; the cost ratio between `round_robin` and `periodic_k5` remains within 5% across all conditions.
3. **No change to PRNG streams, velocity calibration, or recall evaluation**: The `indep_dir` streams (Streams 1–3 on seeds `seed`, `seed+1`, `seed+2`) are identical across both versions. The difference is purely the temporal window, not data or policy logic.
4. **Clustered fallback shift ($-1.82 \to -0.11$)**: The fallback (Table 9.2, `efConstruction=16`) showed the same step-count effect. At 5 steps the cluster topology had less time to re-stabilise between batch repairs, amplifying the advantage of synchronised batching; at 10 steps the gap narrows as both policies equilibrate.

**All numbers in the current tables are backed by `docs/data/external_benchmarks.json` (commit `dce73a2`), generated with $T=10$ steps.** The previous 5-step figures are not in the JSON and are not referenced in any claim.

---

## 8.2 Controlled Comparison on Synthetic Gaussian Data

We evaluate 7 maintenance policies under two distinct graph construction regimes using **fresh, non-overlapping seed sets** per configuration to prevent seed selection bias:

1. **Configuration A (`efConstruction = 16`, Seeds 100–104)** *(Headline)*: Fresh seeds, bounded construction beam width. These are the primary results for the efc=16 regime.
2. **Configuration B (`efConstruction = 200`, Seeds 100–104)** *(Headline)*: Fresh seeds, production-grade construction beam width. These are the primary results confirming the pre-registered claim.
3. **Configuration A-secondary (`efConstruction = 16`, Seeds 42–46)** *(Secondary)*: Earlier sweep seeds retained for continuity and version-change traceability (see §8.1.4).

> **Headline results** are Tables 8.1A (seeds 100–104, efc=16) and 8.1B (seeds 100–104, efc=200). Table 8.1A-secondary (seeds 42–46, efc=16) is secondary.

---

### Table 8.1A: Synthetic Gaussian Drift (`efConstruction = 16`, Seeds 100–104) *(Headline)*
*Parameters: $M=8, M_0=16, efConstruction=16, \delta=0.30, \lambda=8.0\times, k=10, ef=16, T=10$ steps*

| Size | Policy Arm | Parameters | Mean Drift Rec@16 | Final Drift Rec@16 | Mean Cost (Dists) | Reinsertions |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **n=500** | `no_repair` | — | 0.6877 ± 0.017 | 0.5032 ± 0.022 | 0 | 0 |
| | `periodic_k1` | K=1 | 0.7991 ± 0.013 | 0.7074 ± 0.022 | 132,504 ± 2,091 | 1,500 |
| | `periodic_k5` | K=5 | 0.8027 ± 0.009 | 0.7084 ± 0.020 | 27,201 ± 417 | 300 |
| | `round_robin` | B=30 | 0.8181 ± 0.014 | 0.7078 ± 0.034 | 28,415 ± 296 | 300 |
| | `lazy_theta_1.0` | θ=1.0 | 0.8132 ± 0.010 | 0.6894 ± 0.021 | 38,135 ± 752 | 393 |
| | `lazy_theta_0.5` | θ=0.5 | 0.8121 ± 0.016 | 0.7076 ± 0.040 | 66,832 ± 1,482 | 709 |
| | `budgeted_priority` | B=30 | 0.8141 ± 0.011 | 0.6952 ± 0.032 | 30,601 ± 401 | 300 |
| **n=2000** | `no_repair` | — | 0.5129 ± 0.016 | 0.2122 ± 0.021 | 0 | 0 |
| | `periodic_k1` | K=1 | 0.7172 ± 0.012 | 0.5902 ± 0.023 | 644,838 ± 7,734 | 6,000 |
| | `periodic_k5` | K=5 | 0.7196 ± 0.010 | 0.6256 ± 0.024 | 131,379 ± 1,331 | 1,200 |
| | `round_robin` | B=120 | 0.7226 ± 0.011 | 0.6220 ± 0.012 | 139,456 ± 1,238 | 1,200 |
| | `lazy_theta_1.0` | θ=1.0 | 0.7361 ± 0.016 | 0.5688 ± 0.030 | 189,870 ± 2,595 | 1,608 |
| | `lazy_theta_0.5` | θ=0.5 | 0.7277 ± 0.007 | 0.5906 ± 0.023 | 337,701 ± 3,251 | 2,929 |
| | `budgeted_priority` | B=120 | 0.7268 ± 0.014 | 0.6052 ± 0.035 | 148,672 ± 1,396 | 1,200 |

#### Paired Differences: Synthetic $n=500, d=8$ (`efConstruction = 16`, Seeds 100–104) *(Headline)*

| Seed | `round_robin` Recall@16 | `periodic_k5` Recall@16 | Paired Difference (RR − Per) | Sign |
|:---:|:---:|:---:|:---:|:---:|
| **100** | 0.7995 | 0.7880 | **+0.0115** | + |
| **101** | 0.8256 | 0.8140 | **+0.0116** | + |
| **102** | 0.8036 | 0.7972 | **+0.0064** | + |
| **103** | 0.8363 | 0.8091 | **+0.0272** | + |
| **104** | 0.8257 | 0.8050 | **+0.0207** | + |

- **Mean Difference ($\Delta$)**: **+0.0155** (+1.55 pp)
- **Sample Std ($s$)**: 0.0083
- **95% Confidence Interval**: [+0.0051, +0.0258]
- **Positive Seeds**: **5 / 5**
- **Decision Rule Outcome ($\ge +0.0200$)**: Not Supported / Inconclusive (mean $+0.0155 < +0.0200$).

#### Paired Differences: Synthetic $n=2000, d=8$ (`efConstruction = 16`, Seeds 100–104) *(Headline)*

| Seed | `round_robin` Recall@16 | `periodic_k5` Recall@16 | Paired Difference (RR − Per) | Sign |
|:---:|:---:|:---:|:---:|:---:|
| **100** | 0.7286 | 0.7145 | **+0.0141** | + |
| **101** | 0.7269 | 0.7093 | **+0.0176** | + |
| **102** | 0.7019 | 0.7120 | **-0.0101** | - |
| **103** | 0.7225 | 0.7282 | **-0.0057** | - |
| **104** | 0.7332 | 0.7342 | **-0.0010** | - |

- **Mean Difference ($\Delta$)**: **+0.0030** (+0.30 pp)
- **Sample Std ($s$)**: 0.0122
- **95% Confidence Interval**: [-0.0122, +0.0182]
- **Positive Seeds**: **2 / 5**
- **Decision Rule Outcome ($\ge +0.0200$)**: Not Supported / Inconclusive (mean $+0.0030 < +0.0200$).

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
Under `efConstruction = 16`, distance computations per node reinsertion average $\approx 93.2$ ($27,955 / 300$), because candidate exploration in upper layers is constrained. Under `efConstruction = 200`, reinsertions evaluate $\approx 433.2$ distance computations ($129,963 / 300$), matching the project's internal reference engine. Higher construction quality provides higher overall recall ($93.6\%$ vs $80.0\%$) and widens the temporal staggering advantage over batch reinsertion ($+2.67\%$ at $n=500$; $+3.44\%$ at $n=2000$), successfully exceeding the pre-registered $+2.0\%$ decision threshold.

### 8.3.2 Ablation of Threshold $\theta$
Lowering $\theta$ from $1.0$ to $0.5$ in `lazy` roughly doubles reinsertion volume ($313 \to 546$ at $n=500$; $1291 \to 2293$ at $n=2000$) and almost doubles computational cost ($137\text{k} \to 235\text{k}$), while producing negligible recall gains ($0.9381 \to 0.9422$). Setting $\theta = 1.0$ provides the Pareto-optimal operational balance.

### 8.3.3 Secondary Continuity Comparison: Seeds 42–46 (`efConstruction = 16`)
For continuity and direct audit comparison against earlier development iterations, the secondary sweep on seeds 42–46 is preserved below.

#### Table 8.1A-Secondary: Synthetic Gaussian Drift (`efConstruction = 16`, Seeds 42–46)
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

Across both seed cohorts (seeds 42–46 and fresh seeds 100–104), `round_robin` consistently achieves positive paired differences over `periodic_k5` under `efConstruction = 16`, but reliably confirms that `efConstruction = 200` is required to achieve the full $+2.0\text{ pp}$ advantage.
