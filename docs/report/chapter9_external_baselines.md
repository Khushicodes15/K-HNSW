# Chapter 9: External Baselines & Scalability

**Author**: Saanvi Srivastava (Member 4, Benchmarks Lead)  
**Primary Dataset Backing**: [`docs/data/external_benchmarks.json`](../data/external_benchmarks.json), [`docs/data/sift_n2000_ef200.json`](../data/sift_n2000_ef200.json), [`docs/data/sift_n5000_ef200.json`](../data/sift_n5000_ef200.json)  
**Governing Contract**: Pre-Registered Decision Rule (§2, `docs/handoff/member4.md`)

---

## 9.1 Evaluation Under Clustered Embedding Drift (SIFT)

While synthetic Gaussian distributions provide controlled analytical baselines, visual descriptor embeddings exhibit non-negative feature values, sparse histogram patterns, and clustered local density.

### 9.1.1 Real Texmex SIFT-128 Dataset Evaluation
We evaluate maintenance policies on genuine SIFT visual descriptors from the standard Texmex corpus (`siftsmall_base.fvecs`, $d=128$, non-negative features).
- **Dataset File Integrity**: `siftsmall_base.fvecs` (10,000 vectors $\times 128$ dimensions, SHA-256: `1e90414a3254361aba48a0d58e461f7661fb135dabb3da985290e94de8f60fe0`).
- **Scale Calibration**: Velocity magnitude scaled by empirical 1-NN Euclidean distance ($\bar{d}_{\text{1-NN}} \approx 237.22$ at $n=2000$; $\bar{d}_{\text{1-NN}} \approx 205.43$ at $n=5000$).
- **Parameters**: $\delta=30\%$ moving nodes, $\lambda = 8.0\times$, $T=10$ simulation steps, `efConstruction=200`, evaluated over 5 fresh independent random seeds ($\{110, 111, 112, 113, 114\}$) across both $n=2000$ and $n=5000$.

---

### Table 9.1A: Real Texmex SIFT-128 Embeddings ($n=2000$, Seeds 110–114)
*Data Source: `siftsmall_base.fvecs` (SHA-256: `1e90414a3254361aba48a0d58e461f7661fb135dabb3da985290e94de8f60fe0`)*  
*Parameters: $M=8, M_0=16, efConstruction=200, \delta=0.30, \lambda=8.0\times, k=10, T=10$ steps, empirical $\bar{d}_{\text{1-NN}} = 237.22$*

| Policy Arm | Parameters | Mean Drift Rec@16 | Mean Drift Rec@32 | Final Drift Rec@16 | Mean Cost (Dists) | Reinsertions |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| `no_repair` | — | 0.7665 ± 0.005 | 0.8868 ± 0.007 | 0.6360 ± 0.020 | 0 | 0 |
| `periodic_k1` | K=1 | 0.7344 ± 0.006 | 0.8726 ± 0.007 | 0.6360 ± 0.027 | 4,167,194 ± 65,512 | 6,000 |
| `periodic_k5` | K=5 | 0.7595 ± 0.007 | 0.8849 ± 0.007 | 0.6384 ± 0.025 | 886,749 ± 13,926 | 1,200 |
| `round_robin` | B=120 | 0.7497 ± 0.006 | 0.8802 ± 0.006 | 0.6368 ± 0.029 | 866,796 ± 12,468 | 1,200 |
| `lazy_theta_1.0` | θ=1.0 | 0.7454 ± 0.006 | 0.8770 ± 0.007 | 0.6380 ± 0.025 | 973,158 ± 15,178 | 1,366 |
| `lazy_theta_0.5` | θ=0.5 | 0.7385 ± 0.006 | 0.8741 ± 0.006 | 0.6386 ± 0.028 | 1,530,865 ± 26,845 | 2,189 |
| `budgeted_priority` | B=120 | 0.7531 ± 0.006 | 0.8810 ± 0.006 | 0.6392 ± 0.028 | 876,785 ± 12,938 | 1,200 |

#### Paired Differences: Real Texmex SIFT-128 ($n=2000$, Seeds 110–114) — Recall@16

| Seed | `round_robin` Recall@16 | `periodic_k5` Recall@16 | Paired Difference (RR − Per) | Sign |
|:---:|:---:|:---:|:---:|:---:|
| **110** | 0.7466 | 0.7531 | **-0.0065** | - |
| **111** | 0.7532 | 0.7551 | **-0.0019** | - |
| **112** | 0.7597 | 0.7736 | **-0.0139** | - |
| **113** | 0.7419 | 0.7569 | **-0.0150** | - |
| **114** | 0.7469 | 0.7587 | **-0.0118** | - |

- **Mean Difference ($\Delta$)**: **-0.0098** (-0.98 pp)
- **Sample Std ($s$)**: 0.0055
- **95% Confidence Interval (t-stat)**: [-0.0167, -0.0030]
- **95% Bootstrap CI**: [-0.0139, -0.0054]
- **Positive Seeds**: **0 / 5** (`periodic_k5` superior on all 5 seeds)

#### Paired Differences: Real Texmex SIFT-128 ($n=2000$, Seeds 110–114) — Recall@32

| Seed | `round_robin` Recall@32 | `periodic_k5` Recall@32 | Paired Difference (RR − Per) | Sign |
|:---:|:---:|:---:|:---:|:---:|
| **110** | 0.8709 | 0.8722 | **-0.0013** | - |
| **111** | 0.8823 | 0.8828 | **-0.0005** | - |
| **112** | 0.8860 | 0.8933 | **-0.0073** | - |
| **113** | 0.8773 | 0.8854 | **-0.0081** | - |
| **114** | 0.8845 | 0.8909 | **-0.0064** | - |

- **Mean Difference ($\Delta$)**: **-0.0047** (-0.47 pp)
- **Sample Std ($s$)**: 0.0035
- **95% Confidence Interval (t-stat)**: [-0.0091, -0.0003]
- **95% Bootstrap CI**: [-0.0074, -0.0020]
- **Positive Seeds**: **0 / 5** (`periodic_k5` superior on all 5 seeds)

---

### Table 9.1B: Real Texmex SIFT-128 Embeddings ($n=5000$, Seeds 110–114)
*Data Source: `siftsmall_base.fvecs` (SHA-256: `1e90414a3254361aba48a0d58e461f7661fb135dabb3da985290e94de8f60fe0`)*  
*Parameters: $M=8, M_0=16, efConstruction=200, \delta=0.30, \lambda=8.0\times, k=10, T=10$ steps, empirical $\bar{d}_{\text{1-NN}} = 205.43$*

| Policy Arm | Parameters | Mean Drift Rec@16 | Mean Drift Rec@32 | Final Drift Rec@16 | Mean Cost (Dists) | Reinsertions |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| `no_repair` | — | 0.7245 ± 0.013 | 0.8402 ± 0.008 | 0.5494 ± 0.029 | 0 | 0 |
| `periodic_k1` | K=1 | 0.6891 ± 0.013 | 0.8219 ± 0.011 | 0.5668 ± 0.028 | 14,213,176 ± 43,810 | 15,000 |
| `periodic_k5` | K=5 | 0.7174 ± 0.014 | 0.8365 ± 0.011 | 0.5654 ± 0.031 | 3,042,554 ± 10,707 | 3,000 |
| `round_robin` | B=300 | 0.7130 ± 0.012 | 0.8342 ± 0.010 | 0.5660 ± 0.029 | 2,982,799 ± 8,984 | 3,000 |
| `lazy_theta_1.0` | θ=1.0 | 0.7044 ± 0.013 | 0.8281 ± 0.011 | 0.5660 ± 0.029 | 3,151,570 ± 43,456 | 3,209 |
| `lazy_theta_0.5` | θ=0.5 | 0.6937 ± 0.013 | 0.8238 ± 0.011 | 0.5648 ± 0.030 | 4,851,320 ± 75,239 | 5,067 |
| `budgeted_priority` | B=300 | 0.7097 ± 0.014 | 0.8310 ± 0.011 | 0.5650 ± 0.031 | 2,994,893 ± 7,037 | 3,000 |

#### Paired Differences: Real Texmex SIFT-128 ($n=5000$, Seeds 110–114) — Recall@16

| Seed | `round_robin` Recall@16 | `periodic_k5` Recall@16 | Paired Difference (RR − Per) | Sign |
|:---:|:---:|:---:|:---:|:---:|
| **110** | 0.7111 | 0.7207 | **-0.0096** | - |
| **111** | 0.7220 | 0.7304 | **-0.0084** | - |
| **112** | 0.7140 | 0.7133 | **+0.0007** | + |
| **113** | 0.7264 | 0.7310 | **-0.0046** | - |
| **114** | 0.6914 | 0.6917 | **-0.0003** | - |

- **Mean Difference ($\Delta$)**: **-0.0044** (-0.44 pp)
- **Sample Std ($s$)**: 0.0046
- **95% Confidence Interval (t-stat)**: [-0.0102, +0.0013]
- **95% Bootstrap CI**: [-0.0081, -0.0008]
- **Positive Seeds**: **1 / 5** (`periodic_k5` superior on 4 of 5 seeds)

#### Paired Differences: Real Texmex SIFT-128 ($n=5000$, Seeds 110–114) — Recall@32

| Seed | `round_robin` Recall@32 | `periodic_k5` Recall@32 | Paired Difference (RR − Per) | Sign |
|:---:|:---:|:---:|:---:|:---:|
| **110** | 0.8350 | 0.8398 | **-0.0048** | - |
| **111** | 0.8401 | 0.8442 | **-0.0041** | - |
| **112** | 0.8284 | 0.8286 | **-0.0002** | - |
| **113** | 0.8490 | 0.8498 | **-0.0008** | - |
| **114** | 0.8183 | 0.8200 | **-0.0017** | - |

- **Mean Difference ($\Delta$)**: **-0.0023** (-0.23 pp)
- **Sample Std ($s$)**: 0.0020
- **95% Confidence Interval (t-stat)**: [-0.0048, +0.0002]
- **95% Bootstrap CI**: [-0.0039, -0.0007]
- **Positive Seeds**: **0 / 5** (`periodic_k5` superior on all 5 seeds)

---

### 9.1.2 Comparison with SIFT-Like Synthetic Clustered Fallback
For environments where raw external binary datasets are unavailable, the testbed provides an offline multi-cluster synthetic generator (`generate_sift_like`). Below we report the SIFT-like synthetic clustered evaluation to demonstrate consistency with real embedding results.

### Table 9.2: SIFT-Like Synthetic Clustered (n=500, Seeds 42–46)
*Data Source: `SIFT-like synthetic`*  
*Parameters: $M=8, M_0=16, efConstruction=16, \delta=0.30, \lambda=8.0\times, k=10, ef=16, T=10$ steps*

| Policy Arm | Parameters | Mean Drift Rec@16 | Final Drift Rec@16 | Mean Cost (Dists) | Reinsertions |
|:---|:---|:---:|:---:|:---:|:---:|
| `no_repair` | — | 0.8176 ± 0.038 | 0.6962 ± 0.048 | 0 | 0 |
| `periodic_k1` | K=1 | 0.7758 ± 0.047 | 0.6820 ± 0.051 | 96,977 ± 2,168 | 1,500 |
| `periodic_k5` | K=5 | 0.8044 ± 0.041 | 0.6874 ± 0.051 | 20,735 ± 417 | 300 |
| `round_robin` | B=30 | 0.7862 ± 0.045 | 0.6752 ± 0.058 | 19,648 ± 380 | 300 |
| `lazy_theta_1.0` | θ=1.0 | 0.7997 ± 0.041 | 0.6754 ± 0.055 | 20,890 ± 607 | 287 |
| `lazy_theta_0.5` | θ=0.5 | 0.7871 ± 0.047 | 0.6754 ± 0.056 | 37,383 ± 493 | 550 |
| `budgeted_priority` | B=30 | 0.8037 ± 0.042 | 0.6732 ± 0.056 | 22,002 ± 261 | 300 |

#### Paired Differences: SIFT-Like Synthetic Clustered ($n=500$, Seeds 42–46)

| Seed | `round_robin` Recall@16 | `periodic_k5` Recall@16 | Paired Difference (RR − Per) | Sign |
|:---:|:---:|:---:|:---:|:---:|
| **42** | 0.7364 | 0.7527 | **-0.0163** | - |
| **43** | 0.7276 | 0.7571 | **-0.0295** | - |
| **44** | 0.8078 | 0.8258 | **-0.0180** | - |
| **45** | 0.8346 | 0.8424 | **-0.0078** | - |
| **46** | 0.8245 | 0.8440 | **-0.0195** | - |

- **Mean Difference ($\Delta$)**: **-0.0182** (-1.82 pp)
- **Sample Std ($s$)**: 0.0078
- **95% Confidence Interval (t-stat)**: [-0.0279, -0.0086]
- **95% Bootstrap CI**: [-0.0249, -0.0119]
- **Positive Seeds**: **0 / 5** (`periodic_k5` superior on all 5 seeds)

### 9.1.3 Empirical Insight on Clustered Descriptors & Motion Model Dependence
On real Texmex SIFT embeddings, the relative ranking of repair policies depends directly on the motion model:
- Under independent isotropic drift (Tables 9.1A, 9.1B), batch periodic reinsertion slightly outperforms or matches staggered round-robin (<!-- GEN-INLINE: sift_real_iso_ef200_n2000_margin_pp -->-0.98<!-- /GEN-INLINE --> pp at $n=2000$, <!-- GEN-INLINE: sift_real_iso_ef200_n5000_margin_pp -->-0.44<!-- /GEN-INLINE --> pp at $n=5000$). Similarly, on SIFT-like synthetic clustered vectors under isotropic drift (Table 9.2), RR minus periodic = <!-- GEN-INLINE: synth_sift_clustered_iso_ef16_n500_margin_pp -->-1.82<!-- /GEN-INLINE --> pp.
- Conversely, under linear interpolation drift (Section 9.4), staggered round-robin outperforms periodic reinsertion on trajectory-mean recall across $f=0.50$ (with jitter) to $f=1.00$ (with jitter):
  - At $ef=16$: +<!-- GEN-INLINE: interp_f050j_rr_k5_traj_diff_rec16_comm -->2.10<!-- /GEN-INLINE --> pp to +<!-- GEN-INLINE: interp_f100j_rr_k5_traj_diff_rec16_comm -->7.16<!-- /GEN-INLINE --> pp committed (+<!-- GEN-INLINE: interp_f050j_rr_k5_traj_diff_rec16_se -->2.26<!-- /GEN-INLINE --> pp to +<!-- GEN-INLINE: interp_f100j_rr_k5_traj_diff_rec16_se -->7.55<!-- /GEN-INLINE --> pp self-excluded).
  - At $ef=32$: +<!-- GEN-INLINE: interp_f050j_rr_k5_traj_diff_rec32_comm -->1.04<!-- /GEN-INLINE --> pp to +<!-- GEN-INLINE: interp_f100j_rr_k5_traj_diff_rec32_comm -->5.37<!-- /GEN-INLINE --> pp committed (+<!-- GEN-INLINE: interp_f050j_rr_k5_traj_diff_rec32_se -->1.22<!-- /GEN-INLINE --> pp to +<!-- GEN-INLINE: interp_f100j_rr_k5_traj_diff_rec32_se -->5.79<!-- /GEN-INLINE --> pp self-excluded).
  - At $f=0.25$ (no jitter), the trajectory-mean margin is near zero and statistically indistinguishable from zero across both metrics: committed Rec@16 diff = <!-- GEN-INLINE: interp_f025_rr_k5_traj_diff_rec16_comm -->+0.0019<!-- /GEN-INLINE --> (95% CI <!-- GEN-INLINE: interp_f025_rr_k5_traj_ci16_comm -->[-0.0023, +0.0060]<!-- /GEN-INLINE -->), Rec@32 diff = <!-- GEN-INLINE: interp_f025_rr_k5_traj_diff_rec32_comm -->+0.0018<!-- /GEN-INLINE -->; self-excluded Rec@16 diff = <!-- GEN-INLINE: interp_f025_rr_k5_traj_diff_rec16_se -->+0.0022<!-- /GEN-INLINE --> (95% CI <!-- GEN-INLINE: interp_f025_rr_k5_traj_ci16_se -->[-0.0015, +0.0059]<!-- /GEN-INLINE -->), Rec@32 diff = <!-- GEN-INLINE: interp_f025_rr_k5_traj_diff_rec32_se -->+0.0022<!-- /GEN-INLINE --> (both 95% CIs include 0).
  This interpolation finding is exploratory, and the underlying topological mechanism remains untested.

---

## 9.2 External C++ `hnswlib` Baseline Protocol & Status

The `benchmarks/run_hnswlib_baseline.py` harness implements the standard production C++ `hnswlib` update protocol:
$$\text{Update}(i, \vec{x}_{\text{new}}) = \text{mark\_deleted}(i) \;\to\; \text{add\_items}(\vec{x}_{\text{new}}, i, \text{replace\_deleted}=\text{True})$$

### Execution Environment Status
- **Standard Protocol Requirement**: Comparison against compiled C++ `hnswlib` requires native AVX/SSE acceleration to measure exact hardware-level update throughput and recall fidelity.
- **Reporting Status**: On developer hosts lacking pre-compiled C++ `hnswlib` binaries, the harness provides a dry-run emulation fallback. In accordance with AGENTS.md Rule 6 (*"No Fabricated Data or Benchmarks"*), emulation results and unverified wall-clock timings are excluded from this report. Production C++ benchmark numbers are designated for execution on the Linux CI testbed where `hnswlib` compiles natively.

---

## 9.3 Summary of Pre-Registered Decision Rule Outcomes Across All Suites

The pre-registered decision rule (`docs/handoff/member4.md` §2) was evaluated across all experimental suites:

$$\text{Decision Rule: WIN iff } \Delta \text{Recall} \ge +0.0200 \text{ at matched cost across } \ge 5 \text{ seeds}$$

| Evaluation Setting | Parameters | Measured Margin ($\Delta \text{Recall}$) | Cost Ratio ($C_{\text{RR}} / C_{\text{Per}}$) | Pre-Registered Verdict |
|:---|:---|:---:|:---:|:---:|
| **Synthetic $n=500, d=8$** | `efConstruction = 16` | $+0.0171$ | $1.052\times$ | **Not Supported / Inconclusive** |
| **Synthetic $n=2000, d=8$** | `efConstruction = 16` | $+0.0144$ | $1.061\times$ | **Not Supported / Inconclusive** |
| **Synthetic $n=500, d=8$** | `efConstruction = 200` | $\mathbf{+0.0267}$ | $1.024\times$ | **Supported (Win)** |
| **Synthetic $n=2000, d=8$** | `efConstruction = 200` | $\mathbf{+0.0344}$ | $1.030\times$ | **Supported (Win)** |
| **Real Texmex SIFT-128** | $n=2000, efConstruction=200$ | $-0.0098$ | $0.977\times$ | **Not Supported / Inconclusive** |
| **Real Texmex SIFT-128** | $n=5000, efConstruction=200$ | $-0.0044$ | $0.980\times$ | **Not Supported / Inconclusive** |
| **SIFT-Like Synthetic Clustered** | $n=500, efConstruction=16$ | $-0.0182$ | $0.948\times$ | **Not Supported / Inconclusive** |

### 9.3.1 Scientific Implications for Claims Register
1. **Claim 13 (Temporal Staggering Advantage)**:
   - On synthetic Gaussian data constructed with production beam width (`efConstruction = 200`), temporal staggering achieves a statistically significant win (+<!-- GEN-INLINE: synth_gaussian_iso_ef200_n500_margin_pct -->2.67%<!-- /GEN-INLINE --> at $n=500$ and +<!-- GEN-INLINE: synth_gaussian_iso_ef200_n2000_margin_pct -->3.44%<!-- /GEN-INLINE --> at $n=2000$) under independent isotropic drift, confirming the theoretical prediction across all 5 seeds.
   - Under constrained construction (`efConstruction = 16`), the advantage narrows to $+<!-- GEN-INLINE: synth_gaussian_iso_ef16_n2000_margin_pct -->1.44%<!-- /GEN-INLINE --> - +<!-- GEN-INLINE: synth_gaussian_iso_ef16_n500_margin_pct -->1.71%<!-- /GEN-INLINE -->$.
   - On real SIFT-128 embeddings, policy ordering depends on the motion model: under isotropic drift periodic reinsertion is slightly superior (<!-- GEN-INLINE: sift_real_iso_ef200_range_pct -->-0.4% to -1.0%<!-- /GEN-INLINE -->), whereas under linear interpolation drift staggered round-robin outperforms periodic reinsertion on trajectory-mean recall across $f=0.50$ (with jitter) to $f=1.00$ (with jitter) (+<!-- GEN-INLINE: interp_f050j_rr_k5_traj_diff_rec16_comm -->2.10<!-- /GEN-INLINE --> pp to +<!-- GEN-INLINE: interp_f100j_rr_k5_traj_diff_rec16_comm -->7.16<!-- /GEN-INLINE --> pp committed at $ef=16$, +<!-- GEN-INLINE: interp_f050j_rr_k5_traj_diff_rec32_comm -->1.04<!-- /GEN-INLINE --> pp to +<!-- GEN-INLINE: interp_f100j_rr_k5_traj_diff_rec32_comm -->5.37<!-- /GEN-INLINE --> pp committed at $ef=32$; +<!-- GEN-INLINE: interp_f050j_rr_k5_traj_diff_rec16_se -->2.26<!-- /GEN-INLINE --> pp to +<!-- GEN-INLINE: interp_f100j_rr_k5_traj_diff_rec16_se -->7.55<!-- /GEN-INLINE --> pp self-excluded at $ef=16$, +<!-- GEN-INLINE: interp_f050j_rr_k5_traj_diff_rec32_se -->1.22<!-- /GEN-INLINE --> pp to +<!-- GEN-INLINE: interp_f100j_rr_k5_traj_diff_rec32_se -->5.79<!-- /GEN-INLINE --> pp self-excluded at $ef=32$; at $f=0.25$ without jitter, committed Rec@16 diff is <!-- GEN-INLINE: interp_f025_rr_k5_traj_diff_rec16_comm -->+0.0019<!-- /GEN-INLINE --> with 95% CI <!-- GEN-INLINE: interp_f025_rr_k5_traj_ci16_comm -->[-0.0023, +0.0060]<!-- /GEN-INLINE -->, which includes 0; exploratory, mechanism untested).
2. **Claim 12 (Recall Collapse Under Motion)**:
   - Confirmed under unmanaged drift across tested configurations:
     - Synthetic Gaussian isotropic drift: final Rec@16 degrades to 0.5026 ($n=500$) and collapses to 0.2004 ($n=2000$) under `efConstruction=16`, and degrades to 0.5740 ($n=500$) and collapses to 0.2556 ($n=2000$) under `efConstruction=200` (`docs/data/external_benchmarks.json`).
     - Real Texmex SIFT isotropic drift: trajectory-mean Rec@16 is 0.7665 and final Rec@16 degrades to 0.6360 at $n=2000, efConstruction=200$ (`docs/data/external_benchmarks.json`, `docs/data/sift_n2000_ef200.json`); trajectory-mean Rec@16 is 0.7245 and final Rec@16 degrades to 0.5494 at $n=5000, efConstruction=200$ (`docs/data/sift_n5000_ef200.json`).
     - Real Texmex SIFT linear interpolation drift ($n=2000, efConstruction=200$, `docs/data/interpolation_benchmark.json`):
       - $f=0.25$ (no jitter): trajectory-mean Rec@16 is <!-- GEN-INLINE: interp_f025_nr_traj_rec16 -->0.9267<!-- /GEN-INLINE -->, final Rec@16 degrades to <!-- GEN-INLINE: interp_f025_nr_final_rec16 -->0.8262<!-- /GEN-INLINE -->.
       - $f=0.50$ (no jitter): trajectory-mean Rec@16 is <!-- GEN-INLINE: interp_f050_nr_traj_rec16 -->0.7324<!-- /GEN-INLINE -->, final Rec@16 collapses to <!-- GEN-INLINE: interp_f050_nr_final_rec16 -->0.3790<!-- /GEN-INLINE -->.
       - $f=0.50$ (with jitter): trajectory-mean Rec@16 is <!-- GEN-INLINE: interp_f050j_nr_traj_rec16 -->0.7413<!-- /GEN-INLINE -->, final Rec@16 collapses to <!-- GEN-INLINE: interp_f050j_nr_final_rec16 -->0.4080<!-- /GEN-INLINE -->.
       - $f=1.00$ (no jitter): trajectory-mean Rec@16 is <!-- GEN-INLINE: interp_f100_nr_traj_rec16 -->0.5571<!-- /GEN-INLINE -->, final Rec@16 degrades to <!-- GEN-INLINE: interp_f100_nr_final_rec16 -->0.6220<!-- /GEN-INLINE -->.
       - $f=1.00$ (with jitter): trajectory-mean Rec@16 is <!-- GEN-INLINE: interp_f100j_nr_traj_rec16 -->0.6245<!-- /GEN-INLINE -->, final Rec@16 degrades to <!-- GEN-INLINE: interp_f100j_nr_final_rec16 -->0.7518<!-- /GEN-INLINE -->.
       - Note: Final `no_repair` recall is not monotone in $f$, dropping from 0.8262 ($f=0.25$) to 0.3790 ($f=0.50$), before rebounding to 0.7518 ($f=1.00$ with jitter). The hypothesis that full displacement re-aligns drifting queries with dense destination clusters while intermediate $f=0.50$ displacement leaves nodes stranded in low-density inter-cluster space remains untested.


---

## 9.4 Interpolation Drift (Exploratory)

### 9.4.1 Exploratory Status, Protocol Scope, & Movement Model
To investigate whether motion geometry influences dynamic maintenance effectiveness, an interpolation drift benchmark was conducted on the Texmex SIFT-128 siftsmall subset ($n=2000, d=128$, `ef_construction=200`, seeds 110â€“114). In this setting:
- $\delta = 30\%$ of nodes ($600$ points) drift linearly toward randomly selected database target points over $T=10$ discrete steps ($t_{\text{final}} = 0.30, \Delta t = 0.03$):
  $$\vec{x}_u(t) = \vec{x}_u(0) + \frac{t}{t_{\text{final}}} \cdot f \cdot (\vec{x}_{\text{target}} - \vec{x}_u(0) + \vec{\eta})$$
- We evaluate displacement fractions $f \in \{0.25, 0.50, 1.00\}$ without jitter, and $f \in \{0.50, 1.00\}$ with small random target jitter $\vec{\eta}$.
- **Exploratory Status**: This protocol and choice of $f$ parameters were designed post-hoc after observing that batch periodic reinsertion matched or slightly outperformed temporal staggering under independent isotropic drift. Seeds 110â€“112 were previously examined during diagnostic exploration, and the $+2.0$ percentage point decision rule was not pre-registered for this movement regime. Accordingly, all results in this section are strictly **exploratory**, scoped exclusively to this linear interpolation movement model, and do not transfer to isotropic drift. Any proposed structural or routing mechanism remains untested.

### 9.4.2 Metric Definitions, Self-Exclusion, and Jitter Calibration
1. **Search Expansion & Beam Width**:
   In accordance with the benchmark accounting framework, Rec@16 and Rec@32 denote Recall@10 evaluated at search expansion beam widths $ef=16$ and $ef=32$, respectively ($k=10$ nearest neighbors).
2. **Committed vs. Self-Excluded Recall**:
   - The primary committed metric records the query vector's own node ID on both sides (ground truth and candidate retrieval).
   - To rule out self-retrieval inflation, a self-excluded check was executed concurrently: index search requests $k+1=11$ candidates at beam width $ef$, drops the query node's own ID by node ID (never by distance threshold), and retains the first $k=10$ items. Instantaneous ground truth is similarly computed as the 10 nearest non-self points (query node ID excluded).
   - Self-exclusion lowers absolute recall magnitudes across all arms, but preserves the sign, rank ordering, and statistical significance of all paired differences.
3. **Target Jitter Formulation & Norm Ratio**:
   In `benchmarks/run_interpolation_benchmark.py:107-114`, Gaussian jitter noise is sampled independently per coordinate:
   $$\vec{\eta} \sim \mathcal{N}\left(\mathbf{0}, (0.05 \cdot \bar{d}_{\text{1-NN}})^2 \mathbf{I}_d\right)$$
   Across $d=128$ dimensions, the expected Euclidean norm follows a scaled chi distribution with 128 degrees of freedom:
   $$\mathbb{E}[\|\vec{\eta}\|] \approx 0.05 \cdot \bar{d}_{\text{1-NN}} \cdot \sqrt{128} \approx 0.5657 \cdot \bar{d}_{\text{1-NN}}$$
   Empirical measurement across all 600 drifting nodes for seeds 110-114 yields a pooled mean $\|\vec{\eta}\| / \bar{d}_{\text{1-NN}} = $ <!-- GEN-INLINE: interp_jitter_pooled_ratio -->0.565827<!-- /GEN-INLINE --> (min <!-- GEN-INLINE: interp_jitter_min_ratio -->0.438147<!-- /GEN-INLINE -->, max <!-- GEN-INLINE: interp_jitter_max_ratio -->0.683792<!-- /GEN-INLINE -->), exactly <!-- GEN-INLINE: interp_jitter_scaled_factor -->11.316539<!-- /GEN-INLINE --> $\times 0.05$ ($\approx \sqrt{128} = 11.3137$).
4. **Duplicate Collision vs. Jittered Headline**:
   In $f=1.00$ without jitter, <!-- GEN-INLINE: interp_coloc_total_pct_range -->72-85%<!-- /GEN-INLINE --> of evaluated drifting queries arrive at duplicate collisions (evaluated in `benchmarks/diagnose_interpolation_colocation.py` across three mutually exclusive categories: <!-- GEN-INLINE: interp_coloc_sole_static_pct_range -->46-55%<!-- /GEN-INLINE --> co-located with a static node that is the query's target and no drifting node shares that target; <!-- GEN-INLINE: interp_coloc_shared_target_pct_range -->25-31%<!-- /GEN-INLINE --> whose target is shared by at least one other drifting node, resulting in collision with that node and, if static, the target; and <!-- GEN-INLINE: interp_coloc_other_pct_range -->0%<!-- /GEN-INLINE --> other; smallest nonzero distance $\approx <!-- GEN-INLINE: interp_f100_nojitter_min_dist_abs -->1.18e-05<!-- /GEN-INLINE -->$, ratio $\approx <!-- GEN-INLINE: interp_f100_nojitter_min_dist_frac -->5.04e-08<!-- /GEN-INLINE --> \times \bar{d}_{\text{1-NN}}$, consistent with float32 rounding (not verified per coordinate)). The hypothesis that 0-distance duplicate ties disrupt reinsertion edge pruning remains an untested hypothesis. Adding coordinate-level jitter completely eliminates duplicate collisions, maintaining a separation of at least <!-- GEN-INLINE: interp_f100j_min_dist_frac -->0.462<!-- /GEN-INLINE --> $\times \bar{d}_{\text{1-NN}}$ (minimum Euclidean distance <!-- GEN-INLINE: interp_f100j_min_dist_abs -->109.69<!-- /GEN-INLINE --> across all seeds). The jittered row ($f=1.00$ with jitter) represents the valid headline result for full-displacement interpolation.
5. **Rebuild Reference (Single Rebuild at Final Positions) & Cost Comparison**:
   `rebuild` represents a single fresh HNSW index construction on final positions ($t=0.30$) serving as a reference (single rebuild at final positions); trajectory-mean recall equals final-step recall by construction. No periodic-rebuild arm was evaluated. Dynamic maintenance repair cost spans <!-- GEN-INLINE: interp_repair_to_rebuild_cost_ratio_seed_range -->0.8706 to 1.0663<!-- /GEN-INLINE --> $\times$ a single rebuild (<!-- GEN-INLINE: interp_rebuild_cost_min -->919,728<!-- /GEN-INLINE --> to <!-- GEN-INLINE: interp_rebuild_cost_max -->1,084,601<!-- /GEN-INLINE --> distance computations across conditions), and rebuild reaches <!-- GEN-INLINE: interp_rebuild_final_rec16_se_range -->0.9762 to 0.9828<!-- /GEN-INLINE --> self-excluded final Rec@16 (<!-- GEN-INLINE: interp_rebuild_final_rec16_comm_range -->0.9790 to 0.9874<!-- /GEN-INLINE --> committed).
   From the paired difference intervals (Tables 9.4J and 9.4K), periodic_k5 at the final step is not distinguishable from rebuild across all low-to-intermediate displacement conditions ($f \le 0.50$): the 95% $t$-intervals include 0 on both metrics at $f=0.25$ (committed <!-- GEN-INLINE: interp_f025_k5_reb_final_ci_comm -->[-0.0002, +0.0086]<!-- /GEN-INLINE -->, self-excluded <!-- GEN-INLINE: interp_f025_k5_reb_final_ci_se -->[-0.0038, +0.0106]<!-- /GEN-INLINE -->), $f=0.50$ without jitter (committed <!-- GEN-INLINE: interp_f050_k5_reb_final_ci_comm -->[-0.0032, +0.0052]<!-- /GEN-INLINE -->, self-excluded <!-- GEN-INLINE: interp_f050_k5_reb_final_ci_se -->[-0.0005, +0.0077]<!-- /GEN-INLINE -->), and $f=0.50$ with jitter (committed <!-- GEN-INLINE: interp_f050j_k5_reb_final_ci_comm -->[-0.0071, +0.0111]<!-- /GEN-INLINE -->, self-excluded <!-- GEN-INLINE: interp_f050j_k5_reb_final_ci_se -->[-0.0038, +0.0102]<!-- /GEN-INLINE -->), at mean cost ratios of <!-- GEN-INLINE: interp_cost_ratio_per_reb_mean_min -->0.8901<!-- /GEN-INLINE --> to <!-- GEN-INLINE: interp_f025_k5_reb_cost_ratio -->0.9000<!-- /GEN-INLINE --> relative to rebuild (a <!-- GEN-INLINE: interp_f_le_050_k5_cost_reduction_pct_range -->10.01% to 10.99%<!-- /GEN-INLINE --> cost reduction). At $f=0.50$ (no jitter) under self-exclusion, periodic_k5 minus rebuild is positive on 5/5 seeds (Table 9.4K: difference +0.0036, 95% CI [-0.0005, +0.0077], positive seeds 5/5), yet the 95% $t$-interval includes 0. We note that periodic_k5's final-step value is measured right after its own batch reinsertion at step 10, whereas a single rebuild provides no index or recall at steps 1-9; this final-step evaluation therefore compares end states only. Only at high displacement ($f=1.00$) does periodic_k5 differ from rebuild, showing a final recall deficit whose 95% intervals strictly exclude 0 (committed <!-- GEN-INLINE: interp_f100_k5_reb_final_ci_comm -->[-0.0150, -0.0010]<!-- /GEN-INLINE --> unjittered, <!-- GEN-INLINE: interp_f100j_k5_reb_final_ci_comm -->[-0.0684, -0.0328]<!-- /GEN-INLINE --> jittered; self-excluded <!-- GEN-INLINE: interp_f100_k5_reb_final_ci_se -->[-0.0209, -0.0011]<!-- /GEN-INLINE --> unjittered, <!-- GEN-INLINE: interp_f100j_k5_reb_final_ci_se -->[-0.0794, -0.0414]<!-- /GEN-INLINE --> jittered) at cost ratios <!-- GEN-INLINE: interp_f100j_k5_reb_cost_ratio -->0.9881<!-- /GEN-INLINE --> to <!-- GEN-INLINE: interp_cost_ratio_per_reb_mean_max -->1.0293<!-- /GEN-INLINE -->.
   For round_robin, at $f=0.25$ (no jitter) its final-step recall reaches <!-- GEN-INLINE: interp_f025_rr_final_rec16_se -->0.9720<!-- /GEN-INLINE --> self-excluded (<!-- GEN-INLINE: interp_f025_rr_final_rec16_comm -->0.9778<!-- /GEN-INLINE --> committed) versus <!-- GEN-INLINE: interp_f025_reb_final_rec16_se -->0.9762<!-- /GEN-INLINE --> self-excluded (<!-- GEN-INLINE: interp_f025_reb_final_rec16_comm -->0.9816<!-- /GEN-INLINE --> committed) for rebuild at a cost ratio of <!-- GEN-INLINE: interp_f025_rr_reb_cost_ratio -->0.8755<!-- /GEN-INLINE --> (a <!-- GEN-INLINE: interp_f025_rr_reb_gap_se_pp -->0.42<!-- /GEN-INLINE --> pp self-excluded / <!-- GEN-INLINE: interp_f025_rr_reb_gap_comm_pp -->0.38<!-- /GEN-INLINE --> pp committed quality gap for a <!-- GEN-INLINE: interp_f025_rr_cost_reduction_pct -->12.45%<!-- /GEN-INLINE --> cost reduction); the paired difference 95% $t$-intervals include 0 on both metrics (committed <!-- GEN-INLINE: interp_f025_rr_reb_final_ci_comm -->[-0.0082, +0.0006]<!-- /GEN-INLINE -->, self-excluded <!-- GEN-INLINE: interp_f025_rr_reb_final_ci_se -->[-0.0107, +0.0023]<!-- /GEN-INLINE -->), so no statistically distinguishable recall deficit is detected at $f=0.25$. Across all higher-displacement conditions ($f \ge 0.50$), round_robin differs from rebuild with final Rec@16 deficits of <!-- GEN-INLINE: interp_f_ge_050_rr_reb_final_gap_se_range_pp -->3.44 to 6.02<!-- /GEN-INLINE --> pp self-excluded (<!-- GEN-INLINE: interp_f_ge_050_rr_reb_final_gap_comm_range_pp -->3.06 to 5.70<!-- /GEN-INLINE --> pp committed) whose 95% intervals strictly exclude 0, at cost ratios spanning <!-- GEN-INLINE: interp_f_ge_050_cost_ratio_rr_reb_mean_min -->0.8804<!-- /GEN-INLINE --> to <!-- GEN-INLINE: interp_f_ge_050_cost_ratio_rr_reb_mean_max -->1.0572<!-- /GEN-INLINE --> (per-seed: <!-- GEN-INLINE: interp_f_ge_050_cost_ratio_rr_reb_seed_min -->0.8709<!-- /GEN-INLINE --> to <!-- GEN-INLINE: interp_f_ge_050_cost_ratio_rr_reb_seed_max -->1.0663<!-- /GEN-INLINE -->).
   Measured per-reinsertion costs (Table 9.4L) show that across all five conditions, <!-- GEN-INLINE: interp_drifting_node_count -->600<!-- /GEN-INLINE --> nodes drift and each dynamic maintenance arm performs exactly <!-- GEN-INLINE: interp_dynamic_total_reinsertions -->1,200<!-- /GEN-INLINE --> reinsertions in total (periodic_k5 cost per reinsertion: <!-- GEN-INLINE: interp_k5_cost_per_reins_min -->753.19<!-- /GEN-INLINE --> to <!-- GEN-INLINE: interp_k5_cost_per_reins_max -->804.54<!-- /GEN-INLINE -->; round_robin cost per reinsertion: <!-- GEN-INLINE: interp_rr_cost_per_reins_min -->732.74<!-- /GEN-INLINE --> to <!-- GEN-INLINE: interp_rr_cost_per_reins_max -->810.29<!-- /GEN-INLINE -->), compared to a rebuild cost per node ($N=<!-- GEN-INLINE: interp_total_node_count -->2,000<!-- /GEN-INLINE -->$) of <!-- GEN-INLINE: interp_reb_cost_per_node_min -->459.86<!-- /GEN-INLINE --> to <!-- GEN-INLINE: interp_reb_cost_per_node_max -->542.30<!-- /GEN-INLINE -->; dynamic reinsertion thus costs <!-- GEN-INLINE: interp_cost_per_reins_to_reb_node_ratio_range -->1.4592 to 1.7620<!-- /GEN-INLINE --> $\times$ as much per node as inserting into a fresh index. The explanation that dynamic reinsertion costs more per node because it inserts into an already full 2,000-node graph is an untested hypothesis: dynamic reinsertion also includes unlinking and neighbour repair, and no separate distance computation subcategories (such as unlinking, search, or neighbour repair distance computations) are stored in the benchmark JSON (only aggregate total distance computation cost is recorded). A matched-cost comparison with a periodic-rebuild arm was not run.
   Across conditions, the mean cost ratios to a single rebuild range from <!-- GEN-INLINE: interp_cost_ratio_per_reb_mean_min -->0.8901<!-- /GEN-INLINE --> to <!-- GEN-INLINE: interp_cost_ratio_per_reb_mean_max -->1.0293<!-- /GEN-INLINE --> for periodic_k5 (per-seed: <!-- GEN-INLINE: interp_cost_ratio_per_reb_seed_min -->0.8807<!-- /GEN-INLINE --> to <!-- GEN-INLINE: interp_cost_ratio_per_reb_seed_max -->1.0382<!-- /GEN-INLINE -->) and from <!-- GEN-INLINE: interp_cost_ratio_rr_reb_mean_min -->0.8755<!-- /GEN-INLINE --> to <!-- GEN-INLINE: interp_cost_ratio_rr_reb_mean_max -->1.0572<!-- /GEN-INLINE --> for round_robin (per-seed: <!-- GEN-INLINE: interp_cost_ratio_rr_reb_seed_min -->0.8706<!-- /GEN-INLINE --> to <!-- GEN-INLINE: interp_cost_ratio_rr_reb_seed_max -->1.0663<!-- /GEN-INLINE -->). The round-robin to periodic cost ratio spans <!-- GEN-INLINE: interp_cost_ratio_rr_per_mean_min -->0.9729<!-- /GEN-INLINE --> to <!-- GEN-INLINE: interp_cost_ratio_rr_per_mean_max -->1.0421<!-- /GEN-INLINE --> (per-seed: <!-- GEN-INLINE: interp_cost_ratio_rr_per_seed_min -->0.9710<!-- /GEN-INLINE --> to <!-- GEN-INLINE: interp_cost_ratio_rr_per_seed_max -->1.0453<!-- /GEN-INLINE -->).

---

### Table 9.4A: Policy Arm Performance Summary Across 5 Interpolation Conditions
*Parameters: Real SIFT siftsmall ($n=2000, d=128, efConstruction=200, \delta=0.30, T=10$ steps, seeds 110â€“114). Metrics report self-excluded Recall@10 (mean Â± sample std over 5 seeds).*

<!-- BEGIN GENERATED: interpolation_by_arm_summary -->
| Condition | Policy Arm | Mean Drift Rec@16 | Final Drift Rec@16 | Mean Drift Rec@32 | Final Drift Rec@32 | Mean Static Rec@16 | Mean Cost (Dists) | Reinsertions |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| f=0.25 (no jitter) | `no_repair` | 0.9135 +/- 0.0046 | 0.7970 +/- 0.0037 | 0.9669 +/- 0.0041 | 0.9032 +/- 0.0119 | 0.9634 +/- 0.0025 | 0 +/- 0 | 0 |
| f=0.25 (no jitter) | `periodic_k5` | 0.9677 +/- 0.0057 | 0.9796 +/- 0.0082 | 0.9910 +/- 0.0028 | 0.9970 +/- 0.0007 | 0.9793 +/- 0.0013 | 903,829 +/- 10,108 | 1,200 |
| f=0.25 (no jitter) | `round_robin` | 0.9699 +/- 0.0028 | 0.9720 +/- 0.0088 | 0.9933 +/- 0.0017 | 0.9936 +/- 0.0034 | 0.9805 +/- 0.0011 | 879,283 +/- 8,286 | 1,200 |
| f=0.25 (no jitter) | `rebuild` | 0.9762 +/- 0.0039 | 0.9762 +/- 0.0039 | 0.9966 +/- 0.0009 | 0.9966 +/- 0.0009 | 0.9724 +/- 0.0045 | 1,004,310 +/- 4,753 | 0 |
| f=0.50 (no jitter) | `no_repair` | 0.7115 +/- 0.0057 | 0.3638 +/- 0.0242 | 0.8037 +/- 0.0067 | 0.4804 +/- 0.0317 | 0.8978 +/- 0.0049 | 0 +/- 0 | 0 |
| f=0.50 (no jitter) | `periodic_k5` | 0.9327 +/- 0.0057 | 0.9864 +/- 0.0011 | 0.9780 +/- 0.0025 | 0.9980 +/- 0.0012 | 0.9574 +/- 0.0022 | 965,449 +/- 7,597 | 1,200 |
| f=0.50 (no jitter) | `round_robin` | 0.9547 +/- 0.0030 | 0.9484 +/- 0.0076 | 0.9892 +/- 0.0018 | 0.9872 +/- 0.0062 | 0.9645 +/- 0.0015 | 954,903 +/- 9,649 | 1,200 |
| f=0.50 (no jitter) | `rebuild` | 0.9828 +/- 0.0038 | 0.9828 +/- 0.0038 | 0.9976 +/- 0.0009 | 0.9976 +/- 0.0009 | 0.9792 +/- 0.0068 | 1,084,601 +/- 6,315 | 0 |
| f=1.00 (no jitter) | `no_repair` | 0.5559 +/- 0.0089 | 0.6748 +/- 0.0148 | 0.6346 +/- 0.0078 | 0.7276 +/- 0.0197 | 0.7633 +/- 0.0072 | 0 +/- 0 | 0 |
| f=1.00 (no jitter) | `periodic_k5` | 0.8364 +/- 0.0053 | 0.9676 +/- 0.0023 | 0.9073 +/- 0.0045 | 0.9846 +/- 0.0023 | 0.9087 +/- 0.0046 | 946,662 +/- 10,756 | 1,200 |
| f=1.00 (no jitter) | `round_robin` | 0.9273 +/- 0.0021 | 0.9428 +/- 0.0082 | 0.9748 +/- 0.0008 | 0.9772 +/- 0.0044 | 0.9409 +/- 0.0021 | 972,344 +/- 11,693 | 1,200 |
| f=1.00 (no jitter) | `rebuild` | 0.9786 +/- 0.0060 | 0.9786 +/- 0.0060 | 0.9910 +/- 0.0040 | 0.9910 +/- 0.0040 | 0.9702 +/- 0.0057 | 919,728 +/- 5,761 | 0 |
| f=0.50 (with jitter) | `no_repair` | 0.7188 +/- 0.0045 | 0.3924 +/- 0.0178 | 0.8093 +/- 0.0080 | 0.5046 +/- 0.0229 | 0.9058 +/- 0.0032 | 0 +/- 0 | 0 |
| f=0.50 (with jitter) | `periodic_k5` | 0.9297 +/- 0.0040 | 0.9858 +/- 0.0024 | 0.9755 +/- 0.0030 | 0.9970 +/- 0.0014 | 0.9586 +/- 0.0021 | 963,928 +/- 10,275 | 1,200 |
| f=0.50 (with jitter) | `round_robin` | 0.9523 +/- 0.0032 | 0.9440 +/- 0.0084 | 0.9877 +/- 0.0010 | 0.9862 +/- 0.0044 | 0.9639 +/- 0.0020 | 952,055 +/- 8,061 | 1,200 |
| f=0.50 (with jitter) | `rebuild` | 0.9826 +/- 0.0048 | 0.9826 +/- 0.0048 | 0.9978 +/- 0.0018 | 0.9978 +/- 0.0018 | 0.9758 +/- 0.0064 | 1,081,415 +/- 5,395 | 0 |
| f=1.00 (with jitter) | `no_repair` | 0.6285 +/- 0.0111 | 0.8182 +/- 0.0164 | 0.7055 +/- 0.0081 | 0.8792 +/- 0.0086 | 0.8282 +/- 0.0073 | 0 +/- 0 | 0 |
| f=1.00 (with jitter) | `periodic_k5` | 0.8440 +/- 0.0036 | 0.9174 +/- 0.0126 | 0.9132 +/- 0.0031 | 0.9738 +/- 0.0042 | 0.9167 +/- 0.0038 | 928,088 +/- 10,119 | 1,200 |
| f=1.00 (with jitter) | `round_robin` | 0.9195 +/- 0.0038 | 0.9176 +/- 0.0050 | 0.9712 +/- 0.0024 | 0.9756 +/- 0.0030 | 0.9376 +/- 0.0047 | 967,185 +/- 11,338 | 1,200 |
| f=1.00 (with jitter) | `rebuild` | 0.9778 +/- 0.0056 | 0.9778 +/- 0.0056 | 0.9968 +/- 0.0016 | 0.9968 +/- 0.0016 | 0.9808 +/- 0.0047 | 939,222 +/- 4,269 | 0 |
<!-- END GENERATED: interpolation_by_arm_summary -->

---

### Table 9.4B: Paired Differences in Trajectory-Mean Drift Recall
*Paired differences ($arm_1 - arm_2$) evaluated on trajectory-mean drift recall ($k=10$).*

<!-- BEGIN GENERATED: interpolation_paired_differences_trajectory_mean -->
| Condition | Comparison | Traj Rec@16 Diff | 95% CI | Pos | Traj Rec@32 Diff | 95% CI | Pos |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| f=0.25 (no jitter) | periodic_k5 - no_repair | +0.0541 | [+0.0467, +0.0616] | 5/5 | +0.0241 | [+0.0189, +0.0294] | 5/5 |
| f=0.25 (no jitter) | round_robin - no_repair | +0.0563 | [+0.0506, +0.0621] | 5/5 | +0.0264 | [+0.0222, +0.0305] | 5/5 |
| f=0.25 (no jitter) | round_robin - periodic_k5 | +0.0022 | [-0.0015, +0.0059] | 3/5 | +0.0022 | [-0.0020, +0.0065] | 4/5 |
| f=0.50 (no jitter) | periodic_k5 - no_repair | +0.2212 | [+0.2074, +0.2350] | 5/5 | +0.1743 | [+0.1638, +0.1849] | 5/5 |
| f=0.50 (no jitter) | round_robin - no_repair | +0.2432 | [+0.2344, +0.2520] | 5/5 | +0.1856 | [+0.1760, +0.1951] | 5/5 |
| f=0.50 (no jitter) | round_robin - periodic_k5 | +0.0220 | [+0.0158, +0.0282] | 5/5 | +0.0112 | [+0.0097, +0.0127] | 5/5 |
| f=1.00 (no jitter) | periodic_k5 - no_repair | +0.2804 | [+0.2715, +0.2894] | 5/5 | +0.2727 | [+0.2640, +0.2814] | 5/5 |
| f=1.00 (no jitter) | round_robin - no_repair | +0.3714 | [+0.3615, +0.3812] | 5/5 | +0.3402 | [+0.3302, +0.3502] | 5/5 |
| f=1.00 (no jitter) | round_robin - periodic_k5 | +0.0910 | [+0.0834, +0.0985] | 5/5 | +0.0675 | [+0.0612, +0.0739] | 5/5 |
| f=0.50 (with jitter) | periodic_k5 - no_repair | +0.2109 | [+0.2013, +0.2205] | 5/5 | +0.1662 | [+0.1544, +0.1780] | 5/5 |
| f=0.50 (with jitter) | round_robin - no_repair | +0.2335 | [+0.2262, +0.2408] | 5/5 | +0.1784 | [+0.1676, +0.1893] | 5/5 |
| f=0.50 (with jitter) | round_robin - periodic_k5 | +0.0226 | [+0.0160, +0.0292] | 5/5 | +0.0122 | [+0.0081, +0.0163] | 5/5 |
| f=1.00 (with jitter) | periodic_k5 - no_repair | +0.2155 | [+0.2016, +0.2293] | 5/5 | +0.2078 | [+0.1979, +0.2176] | 5/5 |
| f=1.00 (with jitter) | round_robin - no_repair | +0.2910 | [+0.2787, +0.3033] | 5/5 | +0.2657 | [+0.2551, +0.2763] | 5/5 |
| f=1.00 (with jitter) | round_robin - periodic_k5 | +0.0755 | [+0.0680, +0.0830] | 5/5 | +0.0579 | [+0.0517, +0.0641] | 5/5 |
<!-- END GENERATED: interpolation_paired_differences_trajectory_mean -->

---

### Table 9.4C: Paired Differences in Final-Step Drift Recall (Timing-Dependent Metric)
*Paired differences ($arm_1 - arm_2$) evaluated at the final step ($t=10, t_{\text{final}}=0.30$). Note that periodic_k5 reinserts all nodes at step 10 immediately prior to final evaluation.*

<!-- BEGIN GENERATED: interpolation_paired_differences_final_step -->
| Condition | Comparison | Final Rec@16 Diff | 95% CI | Pos | Final Rec@32 Diff | 95% CI | Pos |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| f=0.25 (no jitter) | periodic_k5 - no_repair | +0.1826 | [+0.1708, +0.1944] | 5/5 | +0.0938 | [+0.0784, +0.1092] | 5/5 |
| f=0.25 (no jitter) | round_robin - no_repair | +0.1750 | [+0.1617, +0.1883] | 5/5 | +0.0904 | [+0.0715, +0.1093] | 5/5 |
| f=0.25 (no jitter) | round_robin - periodic_k5 | -0.0076 | [-0.0173, +0.0021] | 1/5 | -0.0034 | [-0.0070, +0.0002] | 0/5 |
| f=0.50 (no jitter) | periodic_k5 - no_repair | +0.6226 | [+0.5927, +0.6525] | 5/5 | +0.5176 | [+0.4777, +0.5575] | 5/5 |
| f=0.50 (no jitter) | round_robin - no_repair | +0.5846 | [+0.5503, +0.6189] | 5/5 | +0.5068 | [+0.4628, +0.5508] | 5/5 |
| f=0.50 (no jitter) | round_robin - periodic_k5 | -0.0380 | [-0.0474, -0.0286] | 0/5 | -0.0108 | [-0.0182, -0.0034] | 0/5 |
| f=1.00 (no jitter) | periodic_k5 - no_repair | +0.2928 | [+0.2748, +0.3108] | 5/5 | +0.2570 | [+0.2330, +0.2810] | 5/5 |
| f=1.00 (no jitter) | round_robin - no_repair | +0.2680 | [+0.2402, +0.2958] | 5/5 | +0.2496 | [+0.2221, +0.2771] | 5/5 |
| f=1.00 (no jitter) | round_robin - periodic_k5 | -0.0248 | [-0.0363, -0.0133] | 0/5 | -0.0074 | [-0.0120, -0.0028] | 0/5 |
| f=0.50 (with jitter) | periodic_k5 - no_repair | +0.5934 | [+0.5709, +0.6159] | 5/5 | +0.4924 | [+0.4630, +0.5218] | 5/5 |
| f=0.50 (with jitter) | round_robin - no_repair | +0.5516 | [+0.5231, +0.5801] | 5/5 | +0.4816 | [+0.4537, +0.5095] | 5/5 |
| f=0.50 (with jitter) | round_robin - periodic_k5 | -0.0418 | [-0.0521, -0.0315] | 0/5 | -0.0108 | [-0.0152, -0.0064] | 0/5 |
| f=1.00 (with jitter) | periodic_k5 - no_repair | +0.0992 | [+0.0741, +0.1243] | 5/5 | +0.0946 | [+0.0870, +0.1022] | 5/5 |
| f=1.00 (with jitter) | round_robin - no_repair | +0.0994 | [+0.0804, +0.1184] | 5/5 | +0.0964 | [+0.0883, +0.1045] | 5/5 |
| f=1.00 (with jitter) | round_robin - periodic_k5 | +0.0002 | [-0.0109, +0.0113] | 3/5 | +0.0018 | [-0.0037, +0.0073] | 2/5 |
<!-- END GENERATED: interpolation_paired_differences_final_step -->

---

### Table 9.4D: Self-Excluded vs. Committed Metric Side-by-Side Comparison & Self-Hit Rates
*Comparison between committed metric (self included on both sides) and self-excluded metric (self removed by node ID), alongside trajectory-mean self-hit rate at beam width ef=16.*

<!-- BEGIN GENERATED: interpolation_self_excluded_vs_committed -->
| Condition | Arm | Comm Traj Rec@16 | Self-Ex Traj Rec@16 | Comm Final Rec@16 | Self-Ex Final Rec@16 | Self-Hit ef=16 | Comm Traj Rec@32 | Self-Ex Traj Rec@32 | Comm Final Rec@32 | Self-Ex Final Rec@32 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| f=0.25 (no jitter) | `no_repair` | 0.9267 | 0.9135 | 0.8262 | 0.7970 | 0.9982 | 0.9723 | 0.9669 | 0.9182 | 0.9032 |
| f=0.25 (no jitter) | `periodic_k5` | 0.9738 | 0.9677 | 0.9858 | 0.9796 | 0.9980 | 0.9928 | 0.9910 | 0.9974 | 0.9970 |
| f=0.25 (no jitter) | `round_robin` | 0.9757 | 0.9699 | 0.9778 | 0.9720 | 0.9994 | 0.9946 | 0.9933 | 0.9944 | 0.9936 |
| f=0.25 (no jitter) | `rebuild` | 0.9816 | 0.9762 | 0.9816 | 0.9762 | 1.0000 | 0.9972 | 0.9966 | 0.9972 | 0.9966 |
| f=0.50 (no jitter) | `no_repair` | 0.7324 | 0.7115 | 0.3790 | 0.3638 | 0.8790 | 0.8184 | 0.8037 | 0.4980 | 0.4804 |
| f=0.50 (no jitter) | `periodic_k5` | 0.9426 | 0.9327 | 0.9884 | 0.9864 | 0.9940 | 0.9815 | 0.9780 | 0.9980 | 0.9980 |
| f=0.50 (no jitter) | `round_robin` | 0.9627 | 0.9547 | 0.9552 | 0.9484 | 0.9990 | 0.9911 | 0.9892 | 0.9888 | 0.9872 |
| f=0.50 (no jitter) | `rebuild` | 0.9874 | 0.9828 | 0.9874 | 0.9828 | 1.0000 | 0.9980 | 0.9976 | 0.9980 | 0.9976 |
| f=1.00 (no jitter) | `no_repair` | 0.5571 | 0.5559 | 0.6220 | 0.6748 | 0.5282 | 0.6331 | 0.6346 | 0.6714 | 0.7276 |
| f=1.00 (no jitter) | `periodic_k5` | 0.8512 | 0.8364 | 0.9710 | 0.9676 | 0.9416 | 0.9157 | 0.9073 | 0.9836 | 0.9846 |
| f=1.00 (no jitter) | `round_robin` | 0.9371 | 0.9273 | 0.9484 | 0.9428 | 0.9896 | 0.9780 | 0.9748 | 0.9768 | 0.9772 |
| f=1.00 (no jitter) | `rebuild` | 0.9790 | 0.9786 | 0.9790 | 0.9786 | 1.0000 | 0.9882 | 0.9910 | 0.9882 | 0.9910 |
| f=0.50 (with jitter) | `no_repair` | 0.7413 | 0.7188 | 0.4080 | 0.3924 | 0.8888 | 0.8253 | 0.8093 | 0.5208 | 0.5046 |
| f=0.50 (with jitter) | `periodic_k5` | 0.9397 | 0.9297 | 0.9888 | 0.9858 | 0.9922 | 0.9794 | 0.9755 | 0.9976 | 0.9970 |
| f=0.50 (with jitter) | `round_robin` | 0.9606 | 0.9523 | 0.9526 | 0.9440 | 0.9982 | 0.9898 | 0.9877 | 0.9886 | 0.9862 |
| f=0.50 (with jitter) | `rebuild` | 0.9868 | 0.9826 | 0.9868 | 0.9826 | 1.0000 | 0.9984 | 0.9978 | 0.9984 | 0.9978 |
| f=1.00 (with jitter) | `no_repair` | 0.6245 | 0.6285 | 0.7518 | 0.8182 | 0.5388 | 0.6978 | 0.7055 | 0.8090 | 0.8792 |
| f=1.00 (with jitter) | `periodic_k5` | 0.8597 | 0.8440 | 0.9326 | 0.9174 | 0.9380 | 0.9214 | 0.9132 | 0.9784 | 0.9738 |
| f=1.00 (with jitter) | `round_robin` | 0.9313 | 0.9195 | 0.9262 | 0.9176 | 0.9818 | 0.9752 | 0.9712 | 0.9768 | 0.9756 |
| f=1.00 (with jitter) | `rebuild` | 0.9832 | 0.9778 | 0.9832 | 0.9778 | 1.0000 | 0.9976 | 0.9968 | 0.9976 | 0.9968 |
<!-- END GENERATED: interpolation_self_excluded_vs_committed -->

---

### Table 9.4E: Diagnostic Verification: Duplicate Co-Location and Jitter Norms

<!-- BEGIN GENERATED: interpolation_colocation_and_jitter -->
#### Duplicate Co-Location at Final Step ($t=0.30$)
| Condition | Seed 110 | Seed 111 | Seed 112 | Seed 113 | Seed 114 | Mean Near Count / 100 | Smallest Nonzero Dist / mean_1nn |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| f=0.25 (no jitter) | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0.0/100 | 4.337087e-01 |
| f=0.50 (no jitter) | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0.0/100 | 5.560426e-01 |
| f=1.00 (no jitter) | 79/100 | 85/100 | 81/100 | 76/100 | 72/100 | 78.6/100 | 5.035033e-08 |
| f=0.50 (with jitter) | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0.0/100 | 6.392975e-01 |
| f=1.00 (with jitter) | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0.0/100 | 4.620345e-01 |

#### Measured Jitter Displacement Norm Relative to 1-NN Distance
| Seed | Mean ||jitter|| / mean_1nn | Min ||jitter|| / mean_1nn | Max ||jitter|| / mean_1nn | Mean Ratio / 0.05 |
|:---:|:---:|:---:|:---:|:---:|
| 110 | 0.566949 | 0.462035 | 0.661999 | 11.338984 |
| 111 | 0.564323 | 0.466693 | 0.669776 | 11.286457 |
| 112 | 0.566579 | 0.438147 | 0.676360 | 11.331571 |
| 113 | 0.567526 | 0.452450 | 0.678528 | 11.350521 |
| 114 | 0.563758 | 0.448721 | 0.683792 | 11.275164 |
| **Pooled** | **0.565827** | **0.438147** | **0.683792** | **11.316539** |
<!-- END GENERATED: interpolation_colocation_and_jitter -->

---

### Table 9.4F: Step-by-Step Recall Dynamics (Steps 8-10, $f=0.50$ with jitter)
*Drift Recall@16 at intermediate steps leading to step 10 evaluation for periodic_k5 vs. round_robin on condition f_0.50_jitter (seeds 110–114).*

<!-- BEGIN GENERATED: interpolation_step_dynamics_f050j -->
| Step | t | periodic_k5 Reinserted | periodic_k5 Rec@16 | round_robin Reinserted | round_robin Rec@16 | Paired Diff (RR - k5) | Pos Seeds |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Step 8 | t=0.24 | 0 | 0.9016 +/- 0.0132 | 120 | 0.9576 +/- 0.0064 | +0.0560 +/- 0.0152 | 5/5 |
| Step 9 | t=0.27 | 0 | 0.8274 +/- 0.0189 | 120 | 0.9506 +/- 0.0061 | +0.1232 +/- 0.0163 | 5/5 |
| Step 10 | t=0.30 | 600 | 0.9888 +/- 0.0036 | 120 | 0.9526 +/- 0.0102 | -0.0362 +/- 0.0099 | 0/5 |
<!-- END GENERATED: interpolation_step_dynamics_f050j -->

---

### Table 9.4G: Distance Computation Costs and Ratios Across Conditions
*Total distance computations and relative ratios for rebuild, periodic_k5, and round_robin across all 5 conditions (seeds 110–114).*

<!-- BEGIN GENERATED: interpolation_cost_and_ratios -->
| Condition | Policy Arm | Mean Cost (Dists) | Seed 110 Ratio | Seed 111 Ratio | Seed 112 Ratio | Seed 113 Ratio | Seed 114 Ratio | Mean Ratio |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| f=0.25 (no jitter) | `rebuild` | 1,004,310 +/- 4,753 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| f=0.25 (no jitter) | `periodic_k5` | 903,829 +/- 10,108 | 0.8936 | 0.8935 | 0.9019 | 0.9027 | 0.9080 | 0.8999 |
| f=0.25 (no jitter) | `round_robin` | 879,283 +/- 8,286 | 0.8717 | 0.8706 | 0.8758 | 0.8776 | 0.8817 | 0.8755 |
| f=0.50 (no jitter) | `rebuild` | 1,084,601 +/- 6,315 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| f=0.50 (no jitter) | `periodic_k5` | 965,449 +/- 7,597 | 0.8813 | 0.8901 | 0.8918 | 0.8925 | 0.8950 | 0.8901 |
| f=0.50 (no jitter) | `round_robin` | 954,903 +/- 9,649 | 0.8719 | 0.8765 | 0.8812 | 0.8834 | 0.8891 | 0.8804 |
| f=1.00 (no jitter) | `rebuild` | 919,728 +/- 5,761 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| f=1.00 (no jitter) | `periodic_k5` | 946,662 +/- 10,756 | 1.0262 | 1.0203 | 1.0322 | 1.0382 | 1.0294 | 1.0293 |
| f=1.00 (no jitter) | `round_robin` | 972,344 +/- 11,693 | 1.0496 | 1.0493 | 1.0617 | 1.0663 | 1.0591 | 1.0572 |
| f=0.50 (with jitter) | `rebuild` | 1,081,415 +/- 5,395 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| f=0.50 (with jitter) | `periodic_k5` | 963,928 +/- 10,275 | 0.8822 | 0.8807 | 0.8922 | 0.8980 | 0.9039 | 0.8914 |
| f=0.50 (with jitter) | `round_robin` | 952,055 +/- 8,061 | 0.8747 | 0.8709 | 0.8809 | 0.8820 | 0.8935 | 0.8804 |
| f=1.00 (with jitter) | `rebuild` | 939,222 +/- 4,269 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| f=1.00 (with jitter) | `periodic_k5` | 928,088 +/- 10,119 | 0.9835 | 0.9770 | 0.9971 | 0.9901 | 0.9929 | 0.9881 |
| f=1.00 (with jitter) | `round_robin` | 967,185 +/- 11,338 | 1.0210 | 1.0197 | 1.0362 | 1.0350 | 1.0369 | 1.0298 |
<!-- END GENERATED: interpolation_cost_and_ratios -->

---

### Table 9.4H: Trajectory-Mean and Final-Step Recall@16 and Cost Ratio Relative to Rebuild (Committed Metric)
*Committed Recall@10 at $ef=16$ and mean cost ratio relative to a single static rebuild across all 5 conditions. Rebuild is one evaluation at the final positions (its trajectory-mean equals its final value); round_robin and periodic_k5 trajectory-means average ten steps; the final-step columns are the like-for-like comparison with rebuild.*

<!-- BEGIN GENERATED: interpolation_cost_and_recall_comparison_committed -->
| Condition | Rebuild Rec@16 (Final) | Round-Robin Traj Rec@16 | Round-Robin Final Rec@16 | Round-Robin / Rebuild Cost | Periodic_k5 Traj Rec@16 | Periodic_k5 Final Rec@16 | Periodic_k5 / Rebuild Cost |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| f=0.25 (no jitter) | 0.9816 +/- 0.0030 | 0.9757 +/- 0.0024 | 0.9778 +/- 0.0055 | 0.8755 | 0.9738 +/- 0.0056 | 0.9858 +/- 0.0060 | 0.8999 |
| f=0.50 (no jitter) | 0.9874 +/- 0.0038 | 0.9627 +/- 0.0021 | 0.9552 +/- 0.0059 | 0.8804 | 0.9426 +/- 0.0057 | 0.9884 +/- 0.0015 | 0.8901 |
| f=1.00 (no jitter) | 0.9790 +/- 0.0029 | 0.9371 +/- 0.0021 | 0.9484 +/- 0.0048 | 1.0572 | 0.8512 +/- 0.0053 | 0.9710 +/- 0.0064 | 1.0293 |
| f=0.50 (with jitter) | 0.9868 +/- 0.0044 | 0.9606 +/- 0.0038 | 0.9526 +/- 0.0102 | 0.8804 | 0.9397 +/- 0.0034 | 0.9888 +/- 0.0036 | 0.8914 |
| f=1.00 (with jitter) | 0.9832 +/- 0.0068 | 0.9313 +/- 0.0040 | 0.9262 +/- 0.0049 | 1.0298 | 0.8597 +/- 0.0039 | 0.9326 +/- 0.0122 | 0.9881 |
<!-- END GENERATED: interpolation_cost_and_recall_comparison_committed -->

---

### Table 9.4I: Trajectory-Mean and Final-Step Recall@16 and Cost Ratio Relative to Rebuild (Self-Excluded Metric)
*Self-excluded Recall@10 at $ef=16$ and mean cost ratio relative to a single static rebuild across all 5 conditions. Rebuild is one evaluation at the final positions (its trajectory-mean equals its final value); round_robin and periodic_k5 trajectory-means average ten steps; the final-step columns are the like-for-like comparison with rebuild.*

<!-- BEGIN GENERATED: interpolation_cost_and_recall_comparison_self_excluded -->
| Condition | Rebuild Rec@16 (Final) | Round-Robin Traj Rec@16 | Round-Robin Final Rec@16 | Round-Robin / Rebuild Cost | Periodic_k5 Traj Rec@16 | Periodic_k5 Final Rec@16 | Periodic_k5 / Rebuild Cost |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| f=0.25 (no jitter) | 0.9762 +/- 0.0039 | 0.9699 +/- 0.0028 | 0.9720 +/- 0.0088 | 0.8755 | 0.9677 +/- 0.0057 | 0.9796 +/- 0.0082 | 0.8999 |
| f=0.50 (no jitter) | 0.9828 +/- 0.0038 | 0.9547 +/- 0.0030 | 0.9484 +/- 0.0076 | 0.8804 | 0.9327 +/- 0.0057 | 0.9864 +/- 0.0011 | 0.8901 |
| f=1.00 (no jitter) | 0.9786 +/- 0.0060 | 0.9273 +/- 0.0021 | 0.9428 +/- 0.0082 | 1.0572 | 0.8364 +/- 0.0053 | 0.9676 +/- 0.0023 | 1.0293 |
| f=0.50 (with jitter) | 0.9826 +/- 0.0048 | 0.9523 +/- 0.0032 | 0.9440 +/- 0.0084 | 0.8804 | 0.9297 +/- 0.0040 | 0.9858 +/- 0.0024 | 0.8914 |
| f=1.00 (with jitter) | 0.9778 +/- 0.0056 | 0.9195 +/- 0.0038 | 0.9176 +/- 0.0050 | 1.0298 | 0.8440 +/- 0.0036 | 0.9174 +/- 0.0126 | 0.9881 |
<!-- END GENERATED: interpolation_cost_and_recall_comparison_self_excluded -->

---

### Table 9.4J: Final-Step Paired Differences in Drift Recall (Committed Metric)
*Final-step drift Recall@10 at $ef=16$ paired differences across 5 seeds (mean difference, 95% $t$-interval, and positive seed count) comparing round_robin, periodic_k5, and rebuild.*

<!-- BEGIN GENERATED: interpolation_final_step_paired_differences_committed -->
| Condition | Comparison | Final Rec@16 Diff | 95% CI | Pos |
|:---|:---|:---:|:---:|:---:|
| f=0.25 (no jitter) | round_robin - rebuild | -0.0038 | [-0.0082, +0.0006] | 0/5 |
| f=0.25 (no jitter) | periodic_k5 - rebuild | +0.0042 | [-0.0002, +0.0086] | 4/5 |
| f=0.25 (no jitter) | round_robin - periodic_k5 | -0.0080 | [-0.0138, -0.0022] | 0/5 |
| f=0.50 (no jitter) | round_robin - rebuild | -0.0322 | [-0.0403, -0.0241] | 0/5 |
| f=0.50 (no jitter) | periodic_k5 - rebuild | +0.0010 | [-0.0032, +0.0052] | 2/5 |
| f=0.50 (no jitter) | round_robin - periodic_k5 | -0.0332 | [-0.0399, -0.0265] | 0/5 |
| f=1.00 (no jitter) | round_robin - rebuild | -0.0306 | [-0.0386, -0.0226] | 0/5 |
| f=1.00 (no jitter) | periodic_k5 - rebuild | -0.0080 | [-0.0150, -0.0010] | 1/5 |
| f=1.00 (no jitter) | round_robin - periodic_k5 | -0.0226 | [-0.0321, -0.0131] | 0/5 |
| f=0.50 (with jitter) | round_robin - rebuild | -0.0342 | [-0.0463, -0.0221] | 0/5 |
| f=0.50 (with jitter) | periodic_k5 - rebuild | +0.0020 | [-0.0071, +0.0111] | 3/5 |
| f=0.50 (with jitter) | round_robin - periodic_k5 | -0.0362 | [-0.0484, -0.0240] | 0/5 |
| f=1.00 (with jitter) | round_robin - rebuild | -0.0570 | [-0.0691, -0.0449] | 0/5 |
| f=1.00 (with jitter) | periodic_k5 - rebuild | -0.0506 | [-0.0684, -0.0328] | 0/5 |
| f=1.00 (with jitter) | round_robin - periodic_k5 | -0.0064 | [-0.0164, +0.0036] | 1/5 |
<!-- END GENERATED: interpolation_final_step_paired_differences_committed -->

---

### Table 9.4K: Final-Step Paired Differences in Drift Recall (Self-Excluded Metric)
*Final-step drift Recall@10 at $ef=16$ paired differences across 5 seeds (mean difference, 95% $t$-interval, and positive seed count) comparing round_robin, periodic_k5, and rebuild under self-exclusion.*

<!-- BEGIN GENERATED: interpolation_final_step_paired_differences_self_excluded -->
| Condition | Comparison | Final Rec@16 Diff | 95% CI | Pos |
|:---|:---|:---:|:---:|:---:|
| f=0.25 (no jitter) | round_robin - rebuild | -0.0042 | [-0.0107, +0.0023] | 0/5 |
| f=0.25 (no jitter) | periodic_k5 - rebuild | +0.0034 | [-0.0038, +0.0106] | 4/5 |
| f=0.25 (no jitter) | round_robin - periodic_k5 | -0.0076 | [-0.0173, +0.0021] | 1/5 |
| f=0.50 (no jitter) | round_robin - rebuild | -0.0344 | [-0.0449, -0.0239] | 0/5 |
| f=0.50 (no jitter) | periodic_k5 - rebuild | +0.0036 | [-0.0005, +0.0077] | 5/5 |
| f=0.50 (no jitter) | round_robin - periodic_k5 | -0.0380 | [-0.0474, -0.0286] | 0/5 |
| f=1.00 (no jitter) | round_robin - rebuild | -0.0358 | [-0.0458, -0.0258] | 0/5 |
| f=1.00 (no jitter) | periodic_k5 - rebuild | -0.0110 | [-0.0209, -0.0011] | 1/5 |
| f=1.00 (no jitter) | round_robin - periodic_k5 | -0.0248 | [-0.0363, -0.0133] | 0/5 |
| f=0.50 (with jitter) | round_robin - rebuild | -0.0386 | [-0.0497, -0.0275] | 0/5 |
| f=0.50 (with jitter) | periodic_k5 - rebuild | +0.0032 | [-0.0038, +0.0102] | 3/5 |
| f=0.50 (with jitter) | round_robin - periodic_k5 | -0.0418 | [-0.0521, -0.0315] | 0/5 |
| f=1.00 (with jitter) | round_robin - rebuild | -0.0602 | [-0.0717, -0.0487] | 0/5 |
| f=1.00 (with jitter) | periodic_k5 - rebuild | -0.0604 | [-0.0794, -0.0414] | 0/5 |
| f=1.00 (with jitter) | round_robin - periodic_k5 | +0.0002 | [-0.0109, +0.0113] | 3/5 |
<!-- END GENERATED: interpolation_final_step_paired_differences_self_excluded -->

---

### Table 9.4L: Per-Reinsertion Cost and Ratio to Rebuild Cost Per Node
*Distance computation costs per dynamic reinsertion (1,200 reinsertions over 10 steps across 600 drifting nodes) compared to fresh rebuild cost per node ($N=2000$).*

<!-- BEGIN GENERATED: interpolation_per_reinsertion_cost -->
| Condition | Rebuild Cost | Rebuild Cost / Node | Periodic_k5 Cost | Periodic_k5 Cost / Reins | Periodic_k5 Ratio | Round-Robin Cost | Round-Robin Cost / Reins | Round-Robin Ratio |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| f=0.25 (no jitter) | 1,004,310.2 | 502.16 | 903,829.4 | 753.19 | 1.4999 | 879,282.8 | 732.74 | 1.4592 |
| f=0.50 (no jitter) | 1,084,600.6 | 542.30 | 965,448.8 | 804.54 | 1.4836 | 954,903.2 | 795.75 | 1.4674 |
| f=1.00 (no jitter) | 919,727.8 | 459.86 | 946,661.8 | 788.88 | 1.7155 | 972,343.8 | 810.29 | 1.7620 |
| f=0.50 (with jitter) | 1,081,415.4 | 540.71 | 963,928.0 | 803.27 | 1.4856 | 952,055.4 | 793.38 | 1.4673 |
| f=1.00 (with jitter) | 939,221.8 | 469.61 | 928,088.0 | 773.41 | 1.6469 | 967,185.2 | 805.99 | 1.7163 |
<!-- END GENERATED: interpolation_per_reinsertion_cost -->

---

### 9.4.3 Analysis of Exploratory Observations
1. **Trajectory-Mean vs. Final-Step Timing Dependence**:
   - On trajectory-mean drift recall, temporal staggering (`round_robin`) outperforms batch periodic reinsertion across all intermediate and high displacement conditions: +<!-- GEN-INLINE: interp_f050_rr_k5_traj_diff_rec16_se_pp -->2.20<!-- /GEN-INLINE --> pp at $f=0.50$ (no jitter), +<!-- GEN-INLINE: interp_f050j_rr_k5_traj_diff_rec16_se_pp -->2.26<!-- /GEN-INLINE --> pp at $f=0.50$ (with jitter), and +<!-- GEN-INLINE: interp_f100j_rr_k5_traj_diff_rec16_se_pp -->7.55<!-- /GEN-INLINE --> pp at $f=1.00$ (with jitter) on self-excluded Rec@16 (all 5/5 positive seeds, 95% CIs strictly excluding 0).
   - At the final evaluation step, however, `periodic_k5` surpasses `round_robin` at $f=0.50$ (paired difference RR - k5 is <!-- GEN-INLINE: interp_f050j_rr_k5_final_diff_rec16_comm -->-0.0362<!-- /GEN-INLINE --> committed, <!-- GEN-INLINE: interp_f050j_rr_k5_final_diff_rec16_se -->-0.0418<!-- /GEN-INLINE --> self-excluded, 0/5 positive seeds). The explanation that this reversal occurs because `periodic_k5` performs a full batch reinsertion at step 10 immediately before final recall measurement, whereas `round_robin` has reinserted only <!-- GEN-INLINE: interp_rr_stagger_fraction_pct -->20%<!-- /GEN-INLINE --> of moving nodes since step 9, is consistent with scheduling cadence but remains an untested hypothesis.
2. **Search Beam Expansion ($ef=16$ vs. $ef=32$)**:
   Paired recall differences narrow at $ef=32$ relative to $ef=16$ (e.g., RR - k5 trajectory-mean margin decreases from +<!-- GEN-INLINE: interp_f050j_rr_k5_traj_diff_rec32_comm -->1.04<!-- /GEN-INLINE --> pp committed (+<!-- GEN-INLINE: interp_f050j_rr_k5_traj_diff_rec32_se_pp -->1.22<!-- /GEN-INLINE --> pp self-excluded) at $f=0.50$ jittered, and from +<!-- GEN-INLINE: interp_f100j_rr_k5_traj_diff_rec32_comm -->5.37<!-- /GEN-INLINE --> pp committed (+<!-- GEN-INLINE: interp_f100j_rr_k5_traj_diff_rec32_se_pp -->5.79<!-- /GEN-INLINE --> pp self-excluded) at $f=1.00$ jittered). The hypothesis that wider candidate exploration mitigates local edge degradation remains untested.
3. **Motion-Model Sensitivity**:
   These findings demonstrate that the empirical ranking between batch periodic and continuous staggered maintenance is highly sensitive to the motion regime: batch repair maintains parity or a slight edge under independent isotropic drift, while continuous staggering provides higher time-averaged recall under coherent interpolation drift.
