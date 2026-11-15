# Member 4 Handoff: Benchmark Evaluation Findings & Claims Update

**From**: Saanvi Srivastava (Member 4, Benchmarks Lead)  
**To**: Member 3 (Architecture Lead & Repo Owner)  
**Branch**: `member4/benchmarks`  
**Data Artifact**: `docs/data/external_benchmarks.json`  
**Report Chapters**: `docs/report/chapter8_dynamic_maintenance.md` & `docs/report/chapter9_external_baselines.md`  

---

### Executive Summary

Member 4 has completed the independent empirical benchmark evaluation suite as contracted in `docs/handoff/member4.md`. All experiments enforced the strict `indep_dir` decoupled-stream protocol, evaluated across $\ge 5$ independent random seeds ($\{42, 43, 44, 45, 46\}$), tested under synthetic Gaussian ($n \in \{500, 2000\}, d=8$) and real clustered embeddings (SIFT $d=128$), and benchmarked against the external C++ `hnswlib` baseline.

---

### Key Empirical Results

#### 1. Synthetic Gaussian Motion ($d=8, \lambda=8\times, \delta=30\%, 10$ steps, 5 seeds)
- **`no_repair`**: Experienced severe monotonic recall collapse (dropping to $50.3\%$ at $n=500$ and $41.4\%$ at $n=2000$).
- **`round_robin` vs. `periodic_k5`** (matched at $300$ reinsertions for $n=500$, $1200$ reinsertions for $n=2000$):
  - At $n=500$: `round_robin` recall $= 0.8000 \pm 0.014$ vs. `periodic_k5` recall $= 0.7829 \pm 0.016$ (**Margin: $+0.0171$ / $+1.71\%$**).
  - At $n=2000$: `round_robin` recall $= 0.7222 \pm 0.009$ vs. `periodic_k5` recall $= 0.7078 \pm 0.010$ (**Margin: $+0.0144$ / $+1.44\%$**).

#### 2. Real Clustered Embeddings (SIFT $d=128, \lambda=8\times, \delta=30\%, 10$ steps, 5 seeds)
- **`periodic_k5`**: Recall $= 0.8044 \pm 0.015$ (Cost $= 20,735$).
- **`round_robin`**: Recall $= 0.7862 \pm 0.014$ (Cost $= 19,648$).
- **Finding**: On high-dimensional clustered manifolds, **batch periodic reinsertion outperforms round-robin** (Margin: **$-0.0182$**). Synchronized cluster re-wiring avoids the intermediate topology disconnect of staggered reinsertion.

#### 3. Production C++ `hnswlib` Baseline Showdown
- Tested native `mark_deleted + replace_deleted` on $n=500, d=128$ over 5 seeds.
- Both periodic ($89.1\%$) and round-robin ($92.6\%$) dynamic replacements effectively restored recall from un-repaired baseline ($72.0\%$).
- Total update overhead was $< 0.1\text{ ms}$ per simulation step, demonstrating practical viability in production engines.

---

### Pre-Registered Decision Rule Outcome

Per `docs/handoff/member4.md` §2:
$$\text{Win if } \Delta \text{Recall} \ge +0.0200 \text{ at matched cost across } \ge 5 \text{ seeds}$$

| Setting | Measured Margin ($\Delta \text{Recall}$) | Threshold | Decision Rule Verdict |
|:---|:---:|:---:|:---:|
| **Synthetic $n=500, d=8$** | $+0.0171$ | $\ge +0.0200$ | **Not Supported / Inconclusive** |
| **Synthetic $n=2000, d=8$** | $+0.0144$ | $\ge +0.0200$ | **Not Supported / Inconclusive** |
| **Real SIFT-128 Clustered** | $-0.0182$ | $\ge +0.0200$ | **Not Supported / Inconclusive** |

---

### Action Items for Member 3 (Claims & Docs Updates)

1. **Update Claim 13 (`docs/claims.md`)**:
   - Currently: Marked `SUPPORTED` based on internal 3-seed runs ($+0.0252$ at $n=500$, $+0.0270$ at $n=2000$).
   - Update: Downgrade or qualify Claim 13 to **`PARTIALLY SUPPORTED / REFINED`**. Note that across a rigorous 5-seed battery, the margin narrows to $+0.0144 - +0.0171$ (missing the pre-registered $+0.02$ threshold), and actually reverses on clustered real embeddings (SIFT $d=128$, margin $-0.0182$).
2. **Reaffirm Claim 12 (`docs/claims.md`)**:
   - Strongly confirmed across all synthetic and real embedding runs: un-repaired graphs monotonically collapse to $41\% - 51\%$ recall under continuous drift.
3. **Review & Merge PR**:
   - Branch `member4/benchmarks` contains all 4 phases, 67/67 passing tests, full C++ baseline runners, raw JSON backing data, and Chapters 8 & 9.
