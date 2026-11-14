# Empirical Scope & Methodological Limitations (`docs/limitations.md`)

> **Context**: Methodological self-assessment and technical threats to validity.  
> **Committed Evidence Reference**: `docs/data/controls_benchmark.json`, `docs/data/missing_controls_benchmark.json`, `docs/data/harness_conflict_benchmark.json`.

---

## 1. Experimental Limitations

### A. Synthetic Gaussian Data Distribution
All primary experiments evaluate coordinates drawn from isotropic standard normal distributions:
$$\vec{x}_i(0) \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$$
In real-world applications (e.g., dynamic recommender systems, continual representation learning, streaming document clustering), vectors reside on low-dimensional non-linear manifolds, exhibit clustered density variations, and display non-uniform local intrinsic dimensionality (LID). Behaviors on synthetic Gaussian spheres may not capture localized congestion or sparse boundary transitions present in real embeddings.

### B. Low Dimensionality ($d=8$)
Primary experiments were conducted in $d=8$ dimensions. While higher than planar computational geometry, real production embeddings operate in high dimensions:
$$d \in [128, 1536]$$
In extreme dimensions, distance concentration phenomena and hubness effects alter nearest-neighbor graph topologies. While our qualitative findings regarding staggering and the failure of planar-inspired Voronoi certificates are expected to hold (or intensify) in higher dimensions, exact recall numbers and cost ratios must be validated at full dimension.

### C. Scale ($n=500$ and $n=2000$)
Empirical benchmarks evaluate small-scale graphs ($n=500$ with $150$ moving nodes, and $n=2000$ with $200$ moving nodes). While $n=2000$ successfully confirmed that the staggering advantage over bursty periodic repair generalizes ($+0.0270$ recall advantage), enterprise vector indexes operate on $10^5$ to $10^9$ vectors. Multi-layer hierarchy effects play a more substantial role at $n \ge 10^6$.

### D. Sample Size & Random Seeds
Sweeps are evaluated across three deterministic random seeds:
$$\text{Seeds} \in \{42, 43, 44\}$$
While variance across these seeds was low (range $\le 0.015$ in recall), comprehensive confidence intervals and hypothesis testing across 10–20 random seeds will be required for archival publication.

### E. Execution Environment & Metrics
To satisfy repository Core Purity constraints (pure standard library and NumPy only for browser/Pyodide execution), the engine is written in pure Python.
- Consequently, performance is measured strictly via **hardware-independent operation counters** (distance evaluations, stale checks, reinsertions) rather than wall-clock QPS (queries per second) or latency percentiles under multi-threaded C++/SIMD architectures.

### F. Absence of External Baselines & Real Datasets (Member 4 Scope)
The present evaluation compares internal algorithmic arms (`no_repair`, `periodic`, `round_robin`, `lazy`, `budgeted_priority`, `cert_triggered`) on synthetic linear motion. Full integration with external C++ libraries (FAISS, hnswlib, DiskANN) and real-world drifting datasets (e.g., streaming text/image benchmarks) is explicitly designated as **Member 4's scope** and is excluded from Member 3's core engine deliverables.

---

## 2. Methodological Findings & Resolved Confounds

### A. The Radial-Drift Evaluation Artifact
In earlier project phases (Phases 5g–5k), benchmark generators coupled `default_rng(seed)` with `LinearDrift(X0, seed=seed)`.
- **The Artifact**: Due to NumPy's PRNG stream collision, velocity directions were perfectly aligned with initial coordinates:
  $$\cos(\vec{v}_i, \vec{x}_i(0)) \equiv +1.0000$$
- **Affected Results**: Tables and claims in `docs/data/drift_benchmark.json`, `docs/data/reconcile_benchmark.json`, and `docs/data/vanilla_arms_benchmark.json`.
- **Impact**: Radial motion expanded coordinates outward along rays from the origin, preserving angular relationships and masking un-repaired graph recall decay ($>0.90$ at $10\times$ displacement). Claims that "repair degrades recall compared to no-repair" were artifacts of this non-generalizable radial motion.
- **Resolution**: Phase 5n introduced the decoupled `indep_dir` protocol, where velocity directions are drawn from an independent PRNG stream ($\cos \approx 0.0066$). Under true isotropic drift, un-repaired recall decays severely to $\approx 0.65$, establishing a valid regime for maintenance evaluation.

### B. Resolution of the Pre-Repair Confound (Phase 6 Item B)
In Phase 5m, lazy reinsertion appeared to outperform periodic repair. However, the periodic arms in that script utilized `in_neighbour_pre_repair`, whereas lazy reinsertion did not.
- **Resolution**: Phase 6 Item B (`docs/data/missing_controls_benchmark.json`) rigorously disentangled this confound by executing a full 2x2 factorial sweep (periodic vs. round-robin, each with and without pre-repair).
- **Outcome**: The staggering advantage **unequivocally survived** without pre-repair (+0.0252 at $n=500$, +0.0270 at $n=2000$). Furthermore, adding pre-repair consistently degraded recall by $0.007 - 0.010$ while inflating maintenance cost by $+24\% - 25\%$.

### C. The Untested Pre-Repair Microscopic Hypothesis
While the empirical finding that pre-repair harms recall is definitively established, the intuitive explanation—*that pre-repair deletes obsolete edges prematurely before reinsertion connects replacing bi-directional links, causing momentary local routing sparsity*—remains an **untested hypothesis**. Detailed microscopic edge-connectivity tracing during the repair transition has not been formally conducted.
