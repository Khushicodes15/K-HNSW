# Literature Review & Related Work Matrix

> **Access Date for all primary sources**: 14 November 2026.  
> **Sourcing Standard**: Primary peer-reviewed conference/journal papers, official arXiv preprints, and official GitHub repositories. Any unverified claim is explicitly tagged `UNVERIFIED`.

---

## 1. Literature Matrix

| Work & Primary Citation | Problem Setting | Update Model | Maintenance Mechanism | Metrics | Datasets | Claimed Results | Relation to Our Work |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HNSW**<br>Malkov & Yashunin (2018)<br>[IEEE TPAMI](https://doi.org/10.1109/TPAMI.2018.2889473) / [arXiv:1603.09320](https://arxiv.org/abs/1603.09320) | Static vector search in metric spaces | Static batch construction; dynamic point insertions (`addPoint`) | Multi-layer skip-list graph. Greedy upper-layer routing; beam search at base layer $\ell=0$. Heuristic edge selection with diversity check. | Recall@k, QPS, distance evaluations, indexing time | SIFT1M, GIST1M, GloVe, MNIST | Outperforms NSW, Annoy, and FLANN by an order of magnitude at high recall. | Foundational index architecture. Our work builds directly on HNSW's layered topology, distance accounting, and heuristic pruning. |
| **hnswlib**<br>Malkov et al.<br>[GitHub: nmslib/hnswlib](https://github.com/nmslib/hnswlib) | Real-time vector search library | In-place label updates, point additions, and deletions | `mark_deleted(label)` soft-deletes nodes (skips during query output but preserves in routing). `replace_deleted=True` recycles slots for new items. | Query latency, recall, memory consumption | Standard ANN benchmarks | Avoids immediate graph restructuring by masking deleted labels. | Establishes the standard industry baseline for deletion/replacement in HNSW. Does not repair damaged routing paths or disconnected incoming links. |
| **MN-RU**<br>Xiao et al. (2024)<br>[arXiv:2407.07871](https://arxiv.org/abs/2407.07871) | Real-time dynamic updates in HNSW | High-frequency node replacements and deletions | **Mutual Neighbor Replaced Update**: Re-links 1-hop mutual neighbors of updated nodes using RNG heuristics to suppress "unreachable points" (nodes losing incoming links). | Recall degradation over update cycles, update time | SIFT1M, GIST1M, DEEP1M | Drastically reduces accumulation of unreachable nodes and sustains query recall during streaming updates. | Direct point of comparison for local structural repair. Our reverse-link repair and in-neighbor heuristics address the same ingress loss phenomenon. |
| **FreshDiskANN**<br>Singh et al. (2021)<br>[arXiv:2105.09613](https://arxiv.org/abs/2105.09613) | Streaming billion-scale vector search on SSD | Continuous insertions and deletions in streaming data | Decouples in-memory dynamic index (*FreshVamana*) from disk-resident graph (*DiskANN*); background *StreamingMerge* asynchronously rebuilds disk segments. | Recall@k, 99th percentile query latency, insertion throughput | DEEP1B, SIFT1B, SPACEV1B | Sustains >95% recall@10 at thousands of insertions/sec on a single commodity workstation. | Exemplifies tiered streaming maintenance (fast memory buffer + asynchronous rebuild). Our work focuses on continuous in-place coordinate drift rather than discrete streaming appends/deletes. |
| **Kinetic Data Structures (KDS)**<br>Basch, Guibas, Hershberger (1997/1999)<br>[SODA 1997 / J. Algorithms](https://doi.org/10.1006/jagm.1998.0988) | Computational geometry for continuously moving objects | Continuous trajectories parameterized by time: $\vec{x}_i(t)$ | Proof certificates (e.g. Voronoi/Delaunay conditions); priority event queue scheduling repair events when certificates expire. | Responsiveness, compactness, locality, efficiency | Synthetic continuous motion (2D/3D point clouds) | Optimal event processing matching combinatorial topological transitions for convex hulls, closest pairs, and Delaunay triangulations. | Theoretical inspiration for K-HNSW edge certificates (`KineticDetector`). However, our empirical evaluation proved Voronoi edge certificates are **not useful** in high-dimensional HNSW ($d \ge 8$). |
| **TPR-tree**<br>Šaltenis et al. (2000)<br>[ACM SIGMOD](https://doi.org/10.1145/342009.335427) | Spatio-temporal predictive queries on moving objects | Continuous linear trajectories $\vec{x}(t) = \vec{x}(t_0) + \vec{v}(t - t_0)$ | Time-Parameterized Bounding Boxes (V-MBRs) bounding coordinates and velocities; conservative bounding volume expansion. | Query I/O, node access frequency, update cost | Synthetic vehicle trajectory generators | Enables predictive interval range and nearest-neighbor queries without constant index restructuring. | Pioneers velocity-parameterized bounding. Differs because tree bounding volumes deteriorate rapidly in high dimensions ($d > 4$) due to the curse of dimensionality. |
| **Lazy Update R-trees & Safe Regions**<br>Kwon et al. (2002); Prabhakar et al. (2002)<br>[IEEE MDM](https://doi.org/10.1109/MDM.2002.994400) / [VLDB](https://www.vldb.org) | Location-based tracking of moving objects | Continuous or discrete spatial position shifts | **Safe Regions / Validity Bounds**: No update is triggered while an object remains within its pre-allocated bounding sphere/box. Structural update only when boundaries are breached. | Network message volume, database update throughput, query latency | Mobile sensor / GPS traces | Orders-of-magnitude reduction in update operations with zero loss of query accuracy. | Inspires our `NodeDisplacementTracker` ($\theta$-ball departure bound). However, unlike R-tree MBR containment, HNSW local routing topology degrades smoothly with drift rather than suffering a sharp validity boundary. |
| **Continual Embedding Learning & Drift**<br>Various (e.g. Su et al., 2024)<br>[arXiv:2404.16014](https://arxiv.org/abs/2404.16014) *(representative)* | Dynamic representation learning / RAG with evolving embedding models | Model retraining or concept drift shifting coordinate distributions | Query drift compensation (QDC), vector space translation layers, or scheduled background re-indexing. | Mean Average Precision, Recall@k, re-indexing overhead | MS MARCO, BEIR, dynamic text corpora | Bridges representation shifts without requiring instantaneous full-corpus vector re-embedding. | Highlights the macro-scale problem of embedding drift. Our work operates at the metric data structure level: given drifting coordinates, how to maintain index connectivity under tight compute budgets. |

---

## 2. Novelty Analysis

### A. Has Displacement-Bounded or Budgeted-Staggered Reinsertion Been Applied to Graph ANN?
1. **Displacement-Bounded Maintenance**:
   - Safe regions and displacement thresholds have a rich history in spatial databases (e.g., LUR-trees, safe-region query monitoring in GIS).
   - In graph-based ANN (HNSW, NSG, Vamana), existing dynamic work focuses almost exclusively on **discrete insertion/deletion streaming** (e.g., FreshDiskANN, MN-RU, streaming HNSW).
   - We investigated whether bounding displacement ($\|\vec{x}_i(t) - \vec{p}_i\| \le \theta \cdot r_i$) could effectively trigger lazy graph maintenance in HNSW.
   - **Empirical Finding**: While the displacement bound is intuitive, our pre-registered controlled experiments (`docs/data/controls_benchmark.json`) proved that the adaptive displacement trigger does **NOT** achieve a significant recall advantage over a simple cyclic round-robin policy at the same reinsertion budget ($\Delta < 0.01$).

2. **Budgeted-Staggered Reinsertion**:
   - Standard maintenance in graph indexes operates either:
     a. Synchronously per update (e.g., in-place delete + reinsert on every point move), which is computationally prohibitive under continuous drift.
     b. In periodic batch phases (e.g., every $K$ steps, batch reinsert all drifted points).
   - Our work investigates **systematic temporal staggering** (round-robin / periodic reinsertion with per-node phase offsets) under a strict per-step computational budget $B$.
   - **Novelty Status**: Staggered, phase-offset reinsertion has not previously been formalized or evaluated as a primary dynamic maintenance strategy for HNSW under continuous drift.

### B. What We Found vs. What We Could Not Verify
- **Verified**:
  - Un-repaired HNSW graphs collapse under isotropic continuous drift (recall drops from $>0.98$ to $\approx 0.65$ at 8x displacement).
  - Staggered round-robin maintenance outperforms bursty periodic reinsertion by $+0.025 - +0.027$ recall at identical cumulative insert budgets, matching full $K=1$ periodic repair at $22.7\%$ of the cost.
  - In-neighbor pre-repair and kinetic Voronoi edge certificates add significant computational overhead ($4\times$) and degrade recall.
- **Could Not Verify in Literature**:
  - We found no published baseline that evaluates continuous metric coordinate drift on proximity graphs using kinetic Voronoi certificates. The failure of edge certificates in high-dimensional graph ANN appears to be an unrecorded empirical result.

### C. What We CAN and CANNOT Claim as New
- **What We CAN Claim**:
  1. First rigorous formalization of continuous metric coordinate drift as a benchmark problem for graph-based ANN indexes.
  2. Proof and empirical demonstration of the **radial-drift evaluation artifact** in NumPy PRNG streams.
  3. Demonstration that temporal staggering (round-robin with phase offsets) resolves the bursty recall-decay problem of periodic maintenance.
  4. Evidence that kinetic edge certificates and heuristic pre-repair provide negative utility for high-dimensional drift maintenance.
- **What We CANNOT Claim**:
  1. We cannot claim that adaptive displacement-bounded lazy repair beats cyclic round-robin (the pre-registered 5n margin was not met: margins $+0.004$ to $+0.009$ were below the $\ge 0.01$ threshold).
  2. We cannot claim general asymptotic complexity improvements over standard HNSW insert/delete.
  3. We cannot claim empirical superiority on real-world high-dimensional dynamic embeddings until Member 4's benchmark suites are integrated.
