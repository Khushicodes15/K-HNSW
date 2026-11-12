# K-HNSW Metric Definitions & Evaluation Protocol

This document establishes the official mathematical and procedural definitions of evaluation metrics used across all K-HNSW benchmarks and experiments.

---

## 1. Search Accuracy: Recall@k & Recall-vs-ef Trade-offs

### Recall@k Definition
$$\text{Recall@}k = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\text{approx\_topk}(q, k) \cap \text{exact\_topk}(q, k)|}{k}$$

- **Ground Truth**: Computed using exact brute-force linear scan using squared $\ell_2$ distance ($\|q - v\|_2^2$) over the current vector coordinates at evaluation time.
- **Tie Rule via Distance Tolerance**: When points are equidistant or near-equidistant to the query, an approximate neighbor candidate $v$ is credited as correct if:
  $$d^2(q, v) \le d^2(q, v_k^*) + \epsilon$$
  where $v_k^*$ is the $k$-th ground-truth neighbor and $\epsilon = 10^{-5}$ is the absolute squared distance tolerance.

### Recall-vs-ef Curve
Search accuracy must be evaluated across varied dynamic candidate list sizes $ef$:
$$ef \in [k, 2k, 4k, 8k, 16k]$$
Benchmark reports plot Recall@k against QPS or latency across the $ef$ sweep.

---

## 2. Query Latency, Throughput & Search Diagnostics

### Timing Environment & Logging Policy
- **Logging OFF During Latency Timing**: All kinetic event sinks and trace logging **must be completely disabled** (`NullSink` or logging flag false) during latency and QPS benchmark timing. Event serialization and disk I/O must never pollute query timing runs.
- **Warm-Up Protocol**: A minimum of 500 untimed warm-up queries must precede timed measurements.
- **Thread Count**: All query benchmarks must run with a pinned single thread (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) or an explicitly documented fixed thread pool.

### Latency Percentiles & Throughput
- **Query Latency**: Reported as $p50$, $p95$, and $p99$ in milliseconds ($\text{ms}$).
- **Queries Per Second (QPS)**:
  $$\text{QPS} = \frac{|Q|}{\text{total elapsed search time (seconds)}}$$

### Search Diagnostics
- **Distance Computations**: Mean, $p50$, and $p99$ pairwise vector distance evaluations executed per query during graph traversal.
- **Graph Hops**: Mean, $p50$, and $p99$ number of greedy graph navigation hops traversed per query (reported both for upper layers $l > 0$ and layer 0).

---

## 3. Kinetic Maintenance & Baseline Metrics

### Update Latency
Wall-clock and CPU time consumed by calling `move(node_id, new_vector)` to update coordinate tables and evaluate kinetic certificates. Reported as mean, $p50$, and $p99$ microseconds ($\mu\text{s}$) per moved vector.

### Repair Latency
Wall-clock and CPU time consumed executing `run_repairs(index, budget, sink)`. Reported per repair time step and per repaired node, along with the fraction of allocated budget consumed.

### Modified-Edge Count
The total number of edge mutations executed during graph maintenance:
$$\Delta E = E_{\text{added}} + E_{\text{removed}}$$
Recorded separately for layer 0 and higher layers ($l > 0$).

### Rebuild Time (Baseline Comparison)
Total wall-clock time required to reconstruct the HNSW index from scratch on the updated vector snapshot. Used as the gold standard baseline for measuring repair amortization versus periodic full re-indexing.

---

## 4. Graph Structural & Connectivity Metrics

### Layer-0 BFS Reachability (Primary Connectivity Metric)
The primary measure of graph structural health under kinetic drift:
$$\text{Reachability}_{\text{layer-0}} = \frac{|\text{BFS}_{L_0}(\text{entry\_point})|}{|V|}$$
Computed as the fraction of all indexed nodes in $V$ reachable via directed Breadth-First Search on layer 0 starting from `entry_point()` projected onto layer 0.

### Unreachable-Point Count
Count of nodes having an in-degree of 0 across all layers, **excluding the entry point** (which is an entry root and may naturally possess 0 incoming links at upper layers):
$$\text{Unreachable Count} = |\{v \in V \setminus \{\text{entry\_point}\} \mid \sum_{l=0}^{\text{max\_level}} |\text{in\_neighbors}(v, l)| = 0\}|$$

### Degree Distribution
- Mean, median, standard deviation, minimum, and maximum degrees across nodes.
- Distribution percentiles ($p50$, $p95$, $p99$) for out-degrees and in-degrees reported separately for layer 0 and upper layers.

### Memory Overhead
- Resident Set Size (RSS) delta in megabytes ($\text{MB}$) before and after index construction and kinetic updates.
- Index storage footprint: bytes per vector, including forward edges, reverse edges, and certificate tracking heaps.

---

## 5. Open Decisions (Owner: Member 1 - Mansi Bhattacharjee)
- [ ] **Query Distribution**: Determine benchmark protocol for query movement:
  - Option A: Stationary query set sampled from original data manifold.
  - Option B: Drifting query set tracking moving semantic cluster centroids.
  - Option C: Hybrid split (50% stationary historical queries, 50% drifting queries).
- [ ] **Query Batch Size**: Establish standard benchmark query stream configurations (sequential 1-query stream vs batch-64 evaluation).
- [ ] **Formal Trade-Off Representation**: Finalize curve specification plotting Recall@k against cumulative repair budget (distance computations).
