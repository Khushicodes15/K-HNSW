# Chapter 9: External Baselines & Scalability

**Author**: Saanvi Srivastava (Member 4, Benchmarks Lead)  
**Primary Dataset Backing**: [`docs/data/external_benchmarks.json`](../data/external_benchmarks.json)  
**Governing Contract**: Pre-Registered Decision Rule (§2, `docs/handoff/member4.md`)

---

## 9.1 Evaluation Under Real Embedding Drift (SIFT & GloVe)

While synthetic Gaussian distributions provide controlled analytical baselines, real embeddings exhibit anisotropic cluster structures, non-negative feature distributions, and non-uniform local density.

### 9.1.1 Experimental Configuration
- **Dataset**: SIFT128 visual descriptor vectors ($d=128$, non-negative features, clustered structure).
- **Scale Calibration**: Velocity magnitude scaled by empirical 1-NN Euclidean distance ($\bar{d}_{\text{1-NN}} \approx 133.45$).
- **Parameters**: $n=500$, $\delta=30\%$ moving nodes ($150$ nodes), $\lambda = 8\times$, 10 simulation steps, 5 independent seeds ($\{42, 43, 44, 45, 46\}$).

### Table 9.1: SIFT-128 Clustered Embedding Drift (10 Steps, 5 Seeds)

| Policy Arm | Parameters | Mean Drift Rec@16 | Final Drift Rec@16 | Total Cost (Dists) | Reinsertions |
|:---|:---|:---:|:---:|:---:|:---:|
| `no_repair` | — | $0.6976 \pm 0.006$ | $0.5186 \pm 0.015$ | $0$ | $0$ |
| `periodic_k1` | $K=1$ | $0.8122 \pm 0.018$ | $0.7258 \pm 0.032$ | $103,425 \pm 1,480$ | $1500$ |
| `periodic_k5` | $K=5$ | $\mathbf{0.8044 \pm 0.015}$ | $\mathbf{0.7180 \pm 0.035}$ | $20,735 \pm 280$ | $300$ |
| `round_robin` | $B=30$ | $0.7862 \pm 0.014$ | $0.6980 \pm 0.032$ | $19,648 \pm 410$ | $300$ |
| `lazy_theta_1.0` | $\theta=1.0$ | $0.8015 \pm 0.016$ | $0.7052 \pm 0.030$ | $48,650 \pm 820$ | $612$ |
| `budgeted_priority` | $B=30$ | $0.7910 \pm 0.015$ | $0.7044 \pm 0.033$ | $23,120 \pm 350$ | $300$ |

### 9.1.2 Empirical Finding on Real Clustered Embeddings
In contrast to low-dimensional isotropic Gaussian data, on clustered $d=128$ SIFT embeddings, **batch periodic reinsertion (`periodic_k5`) achieved higher recall than staggered round-robin** ($0.8044$ vs $0.7862$, margin $-0.0182$).
- *Geometric Explanation*: When points cluster closely in high dimensions, moving nodes frequently depart their local clusters simultaneously. Batch re-inserting the cluster members at synchronized intervals allows HNSW to discover mutual new neighbors within the displaced cluster, whereas staggered reinsertion places early-reinserted nodes into an intermediate topology where former cluster neighbors have drifted but not yet been re-indexed.

---

## 9.2 Comparison Against Production C++ `hnswlib` Baseline

We evaluate standard production `hnswlib` (Malkov & Yashunin 2018) under identical kinetic drift using its native replacement protocol:
$$\text{Update}(i, \vec{x}_{\text{new}}) = \text{mark\_deleted}(i) \;\to\; \text{add\_items}(\vec{x}_{\text{new}}, i, \text{replace\_deleted}=\text{True})$$

### Table 9.2: Production `hnswlib` Baseline Maintenance ($n=500, d=128$, 5 Seeds)

| Policy Strategy | Mean Drift Rec@16 | Total Reinsertions | Total Update Time (s) | Update Overhead |
|:---|:---:|:---:|:---:|:---:|
| `no_repair` | $0.7204 \pm 0.012$ | $0$ | $0.0000$ | Baseline |
| `periodic_k5` | $0.8913 \pm 0.015$ | $300$ | $0.0004$ | $0.04$ ms / step |
| `round_robin` | $0.9263 \pm 0.011$ | $300$ | $0.0005$ | $0.05$ ms / step |
| `periodic_k1` | $1.0000 \pm 0.000$ | $1500$ | $0.0023$ | $0.23$ ms / step |

### 9.2.1 Accounting Normalization & Findings
1. **Operational Parity**: Under matched reinsertion operations ($300$ node replacements), both `periodic_k5` and `round_robin` restore recall from $72.0\%$ back above $89\%$.
2. **Wall-Clock Feasibility**: In compiled C++ with native memory layouts, calling `mark_deleted + add_items` takes less than $0.1$ ms per simulation step, demonstrating that continuous dynamic maintenance is computationally practical in production systems.

---

## 9.3 Pre-Registered Decision Rule Outcomes & Discussion

### 9.3.1 Summary of Decision Rule Outcomes Across All Suites

The pre-registered decision rule (`docs/handoff/member4.md` §2) was applied across all experimental conditions:

$$\text{Decision Rule: WIN iff } \Delta \text{Recall} \ge +0.0200 \text{ at matched cost across } \ge 5 \text{ seeds}$$

| Evaluation Setting | Target Margin | Measured Margin ($\Delta \text{Recall}$) | Cost Ratio ($C_{\text{RR}} / C_{\text{Per}}$) | Pre-Registered Verdict |
|:---|:---:|:---:|:---:|:---:|
| **Synthetic $n=500, d=8$** | $\ge +0.0200$ | $+0.0171$ | $1.052\times$ | **Not Supported / Inconclusive** |
| **Synthetic $n=2000, d=8$** | $\ge +0.0200$ | $+0.0144$ | $1.061\times$ | **Not Supported / Inconclusive** |
| **Real SIFT-128 Clustered** | $\ge +0.0200$ | $-0.0182$ | $0.948\times$ | **Not Supported / Inconclusive** |

### 9.3.2 Scientific Implications & Publication Value
1. **Honest Reporting of Decision Rule**:
   Although temporal staggering consistently improves recall on synthetic Gaussian data ($+1.71\%$ at $n=500$ and $+1.44\%$ at $n=2000$), it fails to satisfy the pre-registered $+2.0\%$ margin across 5 seeds. Furthermore, under clustered high-dimensional data (SIFT), batch periodic reinsertion is superior.
2. **Impact on Core Claims**:
   - **Claim 13** (*supported on $n=500, 2000$ internal runs under seed-specific settings*) must be qualified: the staggering advantage shrinks below the pre-registered threshold when evaluated over a broader 5-seed battery and reverses on clustered manifolds.
   - **Claim 12** (*recall collapse under continuous motion without repair*) is strongly confirmed across all datasets (dropping to $41\% - 51\%$ in all configurations).
3. **Publication Value**:
   In top data engineering and management venues (SIGMOD/VLDB), rigorous negative results demonstrating that simple periodic batch replacement or lazy reinsertion suffices under real embedding distributions—and disproving narrow claims of staggering superiority—are valued as rigorous, honest empirical benchmark contributions.
