# K-HNSW Metric Definitions & Evaluation Protocol

This document establishes the official mathematical and procedural definitions of evaluation metrics used across all K-HNSW benchmarks and experiments.

## 1. Search Accuracy: Recall@k
$$\text{Recall@}k = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\text{approx\_topk}(q, k) \cap \text{exact\_topk}(q, k)|}{k}$$
- **Ground Truth**: Computed using exact brute-force linear scan ($\ell_2$ Euclidean distance or inner product) over the current vector coordinates at evaluation time.
- Ties in ground truth distance are broken deterministically by node identifier.
- Queries with fewer than $k$ results normalize by $k$.

## 2. Query Latency & Throughput
- **Warm-Up Protocol**: A minimum of 500 untimed warm-up queries must precede timed measurements.
- **Thread Count**: All query benchmarks must run with a pinned single thread (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) or an explicitly documented fixed thread pool.
- **Reporting Metrics**:
  - Per-query latency percentiles: $p50$, $p95$, and $p99$ in milliseconds ($\text{ms}$).
  - Queries Per Second (QPS):
    $$\text{QPS} = \frac{|Q|}{\text{total elapsed search time (seconds)}}$$

## 3. Kinetic & Maintenance Metrics
### Update Latency
The wall-clock and CPU time consumed by calling `move(node_id, new_vector)` to apply vector position changes and evaluate kinetic certificates. Reported as mean, $p50$, and $p99$ microseconds ($\mu\text{s}$) per moved vector.

### Repair Latency
The time consumed executing `run_repairs(index, budget, sink)`. Reported both per repair operation and per time step, along with the fraction of allocated budget consumed.

### Modified-Edge Count
The total number of edge mutations executed during graph maintenance:
$$\Delta E = E_{\text{added}} + E_{\text{removed}}$$
Recorded separately for layer 0 and higher layers ($l > 0$).

### Connectivity & Reachability
1. **Entry Point Reachability**: The percentage of total indexed nodes reachable via directed graph traversal starting from `entry_point()` at the top layer and traversing downward through layer 0.
2. **Unreachable-Point Count**: The exact count of nodes having an in-degree of 0 across all layers:
   $$\text{Unreachable Count} = |\{v \in V \mid \sum_{l=0}^{\text{max\_level}} |\text{in\_neighbors}(v, l)| = 0\}|$$

### Memory Overhead
- Resident Set Size (RSS) delta in megabytes ($\text{MB}$) before and after index construction and kinetic updates.
- Index storage footprint: bytes per vector, including forward edges, reverse edges, and certificate tracking heaps.

---

## Open Decisions (Owner: Member 1)
- [ ] Determine whether tie-breaking in approximate top-$k$ matches against ground-truth equidistant points should count toward recall or require identical distance tolerance $\epsilon \le 10^{-5}$.
- [ ] Establish standard benchmark query batch sizes (e.g., sequential 1-query stream vs batch-64 evaluation).
- [ ] Define the formal trade-off curve representation: Recall@k vs. cumulative repair compute budget.
- [ ] Finalize standard test trajectory distributions for kinetic drift benchmarks.

