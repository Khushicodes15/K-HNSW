# Literature Review & Related Work Matrix

> **Sourcing standard**: Only citations whose titles, venues and identifiers were checked against a search result on 14 November 2026 are listed without a tag. Anything not re-checked is tagged `UNCHECKED`; anything that could not be confirmed is tagged `UNVERIFIED`. The "datasets" and "claimed results" columns were taken from the original M1 pass and have NOT been re-checked against the papers.

---

## 1. Literature Matrix

| Work & Primary Citation | Problem Setting | Update Model | Maintenance Mechanism | Metrics | Datasets | Claimed Results | Relation to Our Work |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HNSW**<br>Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs", IEEE TPAMI (2018 online; vol. 42, no. 4, 2020)<br>[DOI 10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473) / [arXiv:1603.09320](https://arxiv.org/abs/1603.09320) | Static vector search in metric spaces | Incremental insertion of points | Multi-layer proximity graph (described by the authors as an extension of the skip-list structure). Greedy routing in upper layers, beam search at layer 0, heuristic neighbour selection. | Recall@k, QPS, distance evaluations | UNCHECKED | Logarithmic complexity scaling (as stated in the paper's abstract/introduction). | Foundational index. Our engine follows its layered topology. |
| **hnswlib**<br>[GitHub: nmslib/hnswlib](https://github.com/nmslib/hnswlib) | Real-time vector search library | Label updates, additions, deletions | `mark_deleted` soft-delete; `replace_deleted` slot reuse. | UNCHECKED | UNCHECKED | UNCHECKED | Standard industry baseline for deletion/replacement. Member 4 must confirm exact behaviour from the repo before using it as a baseline. |
| **MN-RU**<br>Xiao et al., "Enhancing HNSW Index for Real-Time Updates: Addressing Unreachable Points and Performance Degradation" (2024)<br>[arXiv:2407.07871](https://arxiv.org/abs/2407.07871) | HNSW under many real-time deletions, insertions and updates | Deletions, insertions, replaced updates | Redesigned deletion/update strategies to reduce "unreachable points" created by updates. | UNCHECKED | UNCHECKED | The paper reports its methods outperform native HNSW update strategies. | Closest published repair method for HNSW updates. Our reverse-link repair addresses a related loss of incoming links. |
| **FreshDiskANN**<br>Singh et al., "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search" (2021)<br>[arXiv:2105.09613](https://arxiv.org/abs/2105.09613) | Streaming vector search at large scale | Continuous insertions and deletions | In-memory fresh index plus disk index with background merge. | UNCHECKED | UNCHECKED | UNCHECKED | Streaming insert/delete maintenance. Our setting is moving vectors, not appends/deletes. |
| **Kinetic Data Structures**<br>Basch, Guibas, Hershberger, "Data structures for mobile data", J. Algorithms 31 (1999) (SODA 1997)<br>[DOI 10.1006/jagm.1998.0988](https://doi.org/10.1006/jagm.1998.0988) (UNCHECKED) | Geometry of continuously moving objects | Continuous trajectories | Certificates with an event queue that triggers repair when a certificate fails. | UNCHECKED | UNCHECKED | UNCHECKED | Inspiration for our edge-certificate engine. We measured that engine as not useful (see `docs/claims.md`). |
| **TPR-tree**<br>Šaltenis, Jensen, Leutenegger, López, "Indexing the Positions of Continuously Moving Objects", SIGMOD 2000, pp. 331–342<br>[DOI 10.1145/342009.335427](https://doi.org/10.1145/342009.335427) | Index current and projected positions of moving objects (1D–3D) | Linear trajectories; dynamic updates | R*-tree-based index with time-parameterised bounding rectangles. | UNCHECKED | UNCHECKED | UNCHECKED | Velocity-parameterised index for low dimensions. Applicability to d ≥ 8 is not established here (UNVERIFIED). |
| **Velocity Constrained Indexing (VCI)**<br>Prabhakar, Xia, Kalashnikov, Aref, Hambrusch, "Query Indexing and Velocity Constrained Indexing: Scalable Techniques for Continuous Queries on Moving Objects", IEEE Trans. Computers 51(10), 1124–1140 (2002)<br>[DOI 10.1109/TC.2002.1039840](https://doi.org/10.1109/TC.2002.1039840) | Continuous queries over moving objects | Moving objects with a known maximum speed | VCI uses the maximum possible speed to delay expensive index updates. | UNCHECKED | UNCHECKED | Reported to outperform traditional approaches by almost two orders of magnitude (per the paper's abstract). | Closest moving-object analogue to a displacement-bounded trigger: delay index updates while a bound on movement holds. |
| **Lazy-Update R-tree**<br>Kwon et al., "Indexing the current positions of moving objects using the lazy update R-tree", IEEE MDM 2002 (UNCHECKED)<br>[DOI 10.1109/MDM.2002.994400](https://doi.org/10.1109/MDM.2002.994400) (UNCHECKED) | Current positions of moving objects | Frequent position updates | Defers R-tree structural updates for small movements. | UNCHECKED | UNCHECKED | UNCHECKED | Related to our displacement-bounded policy. |

*Removed in this revision*: a row citing Su et al. on embedding drift. The citation identifier could not be matched to that title, so it is UNVERIFIED and dropped.

---

## 2. Novelty Analysis & Search Provenance

### A. Searches run by the H3 agent (14 November 2026): results NOT recorded
The H3 agent ran these six web searches. The result lists it reported contained entries that do not exist (checked and removed), so **no result is recorded from that run**. This log claims nothing about what those searches returned.
1. `graph ANN index maintenance moving vectors`
2. `HNSW update continuous drift`
3. `periodic reinsertion HNSW staggered`
4. `round robin reinsertion approximate nearest neighbor index`
5. `displacement threshold lazy update proximity graph`
6. `dynamic ANN index vector updates embedding drift`

### B. Independent check (14 November 2026), search-result excerpts only
Related work surfaced by separate web searches. Only the search-result excerpts were read, not the full papers, so descriptions are limited to what the excerpts state:
- Xu, Zhao, Tan, Zhou, Li, *Proximity Graph Maintenance for Fast Online Nearest Neighbor Search* (2022), [arXiv:2206.10839](https://arxiv.org/pdf/2206.10839): incremental maintenance of proximity graphs under online insertion and deletion of vertices.
- Zhang, Wei, Engels, Shun, *CleanANN* (SPAA 2026), [PDF](https://jshun.csail.mit.edu/CleanANN.pdf): dynamic graph ANN index; excerpts discuss graph repair after deletes and distribution shift.
- *Deletion Control* (NeurIPS 2025 ML for Systems workshop), [arXiv:2512.06200](https://arxiv.org/pdf/2512.06200v1): evaluation of data-deletion methods for ANNS indexes.
- *MERIT: Efficient In-Place Deletion for Dynamic Graph-Based Approximate Nearest Neighbor Indexes*, [arXiv:2607.29173](https://arxiv.org/pdf/2607.29173): local repair after deletion.
- Sengupta, *Robust HNSW Vector Retrieval under Dynamic Product Updates via Local Rewiring* (CEUR-WS Vol-4250), [PDF](https://ceur-ws.org/Vol-4250/paper776.pdf): neighbourhood repair under catalogue updates.

In these excerpts the update model is insertions, deletions and replacements. We did not see continuous coordinate drift or staggered reinsertion mentioned. **This is excerpt-level only; everything beyond it is UNVERIFIED.**

### C. Earlier M1 sources
Identified through web search in Phase M1: HNSW paper, hnswlib repository, MN-RU, FreshDiskANN, Kinetic Data Structures, TPR-tree, lazy-update R-tree. The M1 log shows searches, not page fetches. Titles and venues for HNSW, MN-RU, TPR-tree and VCI were re-checked on 14 November 2026; the rest are `UNCHECKED`.

### D. Novelty assessment (scoped to what was checked)
1. **What we saw in the checked sources**
   - Dynamic graph-ANN work frames updates as insertions, deletions and replacements (B, C).
   - Moving-object databases use movement bounds to delay index updates (VCI), in low dimensions.
2. **What we can say**
   - In the sources checked, we did not find work that treats continuous coordinate drift as a graph-ANN maintenance problem.
   - In the sources checked, we did not find staggered (round-robin, per-node phase offset) reinsertion applied to graph ANN indexes.
   - These are statements about a small, partly excerpt-level search, not about the literature.
3. **What we cannot say**
   - That any of this has "not been done" outside the sources checked.
   - That adaptive displacement-bounded (lazy) repair beats round_robin: the pre-registered rule (margin ≥ 0.01) was not met (`docs/data/controls_benchmark.json`).
   - Any asymptotic improvement, or superiority on real data or against external baselines. That is Member 4's work.