# Member 4: Phase-Wise Benchmark Evaluation Findings

**Author**: Saanvi Srivastava (Member 4, Benchmarks Lead)  
**Branch**: `member4/benchmarks`  
**Data Artifact**: [`docs/data/external_benchmarks.json`](../data/external_benchmarks.json)  
**Report Chapters**: [`docs/report/chapter8_dynamic_maintenance.md`](../report/chapter8_dynamic_maintenance.md) & [`docs/report/chapter9_external_baselines.md`](../report/chapter9_external_baselines.md)  

---

## Phase 1: Shared Toolkit & Dataset Pipeline
- **Deliverables**: `benchmarks/common.py`, `benchmarks/datasets.py`, `tests/unit/test_benchmark_common.py`, `tests/unit/test_benchmark_datasets.py`.
- **Infrastructure Verified**:
  1. **Strict PRNG Stream Decoupling**: Verified that initial point positions, moving node selection, and isotropic velocity directions use three isolated generators (`seed`, `seed+1`, `seed+2`), keeping mean $|\cos(\vec{v}, \vec{x})| < 0.15$ and eliminating the pure radial drift hazard.
  2. **Speed Scaling Calibration**: Implemented velocity calibration proportional to empirical mean 1-NN Euclidean distance ($\bar{d}_{\text{1-NN}}$).
  3. **Data Loaders**: Added binary SIFT `.fvecs`, GloVe `.txt`, and clustered synthetic fallbacks.
  4. **Test Status**: 47/47 tests passed.

---

## Phase 2: Synthetic Coordinate Drift Sweep
- **Deliverable**: `benchmarks/run_synthetic_sweep.py`, `tests/unit/test_synthetic_sweep.py`.
- **Experiment Setup**: Synthetic Gaussian point clouds ($n \in \{500, 2000\}$, $d=8$, 10 simulation steps, displacement $\lambda=8\times$, drift fraction $\delta=30\%$) evaluated across 5 independent seeds ($\{42, 43, 44, 45, 46\}$).
- **Key Findings**:
  1. **Unmanaged Drift Collapse (`no_repair`)**: Recall@16 collapsed monotonically from initial construction down to **$50.26\% \pm 1.8\%$** ($n=500$) and **$41.37\% \pm 1.2\%$** ($n=2000$).
  2. **Staggered (`round_robin`) vs. Batch (`periodic_k5`)**:
     - At $n=500$ ($300$ total reinsertions): `round_robin` achieved **$80.00\% \pm 1.4\%$** vs. `periodic_k5` at **$78.29\% \pm 1.6\%$** (**$+1.71\%$ advantage**).
     - At $n=2000$ ($1,200$ total reinsertions): `round_robin` achieved **$72.22\% \pm 0.9\%$** vs. `periodic_k5` at **$70.78\% \pm 1.0\%$** (**$+1.44\%$ advantage**).
  3. **Pre-Registered Decision Rule Outcome**:
     - Contract condition: Margin $\ge +0.0200$ (+2.00%) across $\ge 5$ seeds.
     - Verdict: **Not Supported / Inconclusive** (margins of $+1.71\%$ and $+1.44\%$ fell just shy of the $+2.0\%$ threshold).

---

## Phase 3: Real Embeddings & External `hnswlib` Baselines
- **Deliverables**: `benchmarks/run_real_embeddings.py`, `benchmarks/run_hnswlib_baseline.py`, `tests/unit/test_real_embeddings.py`, `tests/unit/test_hnswlib_baseline.py`.
- **Experiments**: Clustered SIFT-128 embeddings ($d=128$, non-negative features) and C++ `hnswlib` (`mark_deleted + replace_deleted`) across 5 seeds.
- **Key Findings**:
  1. **Real Clustered SIFT-128 Embeddings**:
     - `periodic_k5` (Batch): **$80.44\% \pm 1.5\%$** recall.
     - `round_robin` (Staggered): **$78.62\% \pm 1.4\%$** recall.
     - **Discovery**: On high-dimensional clustered data, **batch periodic reinsertion beat staggered reinsertion by $+1.82\%$**. Re-indexing tight cluster members together avoids leaving early-reinserted points stranded while their former cluster neighbors have moved but not yet been re-indexed.
  2. **Production C++ `hnswlib` Baseline Showdown**:
     - Both periodic ($89.13\%$) and round-robin ($92.63\%$) dynamic replacements effectively restored recall from the un-repaired baseline ($72.04\%$).
     - Native C++ replacement took **$< 0.1\text{ ms}$ per step**, confirming operational viability for live production systems.

---

## Phase 4: Data Consolidation & Report Chapters
- **Deliverables**:
  - `docs/data/external_benchmarks.json`: Raw empirical JSON results backing all experiment tables.
  - `docs/report/chapter8_dynamic_maintenance.md`: Chapter 8 (Dynamic Drift Maintenance Evaluation).
  - `docs/report/chapter9_external_baselines.md`: Chapter 9 (External Baselines & Scalability).
- **Core Claims Assessment**:
  1. **Claim 12 (Recall Collapse)**: **Strongly Confirmed** across all datasets without repair.
  2. **Claim 13 (Temporal Staggering Advantage)**: **Refined / Qualified**. Staggering provides a consistent $+1.4\%$ to $+1.7\%$ boost on uniform isotropic distributions, but narrows below the pre-registered $+2.0\%$ margin across 5 seeds and reverses on clustered real manifolds.
