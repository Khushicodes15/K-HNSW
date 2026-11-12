# Kinetic Certificate Store, Cost Meter, and Affected Edge Detection

## 1. Definitions and Edge Status Formulation

In K-HNSW, graph topology is certified at the granularity of directed edges within specific hierarchy layers.

- **Directed Edge Key**: An edge is identified by the 3-tuple:
  $$(u, v, l)$$
  representing a directed link from source node $u$ to target node $v$ at hierarchy layer $l$.
- **Outgoing Neighbor List**: $N_l(u)$ denotes the ordered list of active outgoing neighbors of node $u$ at layer $l$.
- **Static Invalidation Condition**: An edge $(u, v, l)$ has static status **`VIOLATED`** if and only if there exists at least one neighbor $w \in N_l(u) \setminus \{v\}$ satisfying the static certificate inequality:
  $$\text{is\_violated\_static}(D(u, v), D(u, w), D(v, w)) = \text{True}$$
  where $D(i, j) = \|p_i - p_j\|_2^2$ is the squared Euclidean distance between the current vector coordinates stored in the index.
  Otherwise, the edge status is **`VALID`**.
- **Canonical Witness**: If one or more neighbors $w \in N_l(u) \setminus \{v\}$ violate the edge, the canonical witness is defined deterministically as:
  $$\text{witness}(u, v, l) = \min \{ w \in N_l(u) \setminus \{v\} \mid \text{is\_violated\_static}(D(u, v), D(u, w), D(v, w)) \}$$
  If the edge is `VALID`, $\text{witness}(u, v, l) = \text{None}$.

---

## 2. Dependency Claim and Proof

### Claim
The validity status and witness of any directed edge $(u, v, l)$ depend strictly and exclusively on:
1. The spatial coordinates $\vec{p}_u$ of source node $u$.
2. The spatial coordinates $\vec{p}_v$ of target node $v$.
3. The spatial coordinates $\vec{p}_w$ of each candidate witness $w \in N_l(u) \setminus \{v\}$.
4. The set of neighbor identifiers comprising $N_l(u)$.

### Corollary 1: Affected Edges Following Coordinate Movement
When a single node $m$ updates its spatial position via `index.move(m, ...)` without mutating graph edges, only the coordinate vector $\vec{p}_m$ changes. By the dependency claim, an edge $(x, y, l)$ can change its status if and only if $m \in \{x, y\} \cup N_l(x)$.

Therefore, the set of all edges whose certificate status can change consists of:
- **(a) Outgoing edges of $m$**: Every directed edge $(m, x, l)$ for each level $l \le \text{node\_level}(m)$ (since $m = x$ is the source).
- **(b) Incoming edges to $m$**: Every directed edge $(u, m, l)$ for each $u \in \text{in\_neighbors}(m, l)$ (since $m = y$ is the target).
- **(c) Sibling out-edges of in-neighbors**: Every other out-edge $(u, v, l)$ originating from any in-neighbor $u \in \text{in\_neighbors}(m, l)$ (since $m \in N_l(u)$ acts as a candidate witness $w$).

Because $m \in N_l(u)$ for all in-neighbors $u \in \text{in\_neighbors}(m, l)$, classes (b) and (c) together comprise **all outgoing edges of all in-neighbors of $m$** across all layers where $m$ exists:
$$\text{Affected}(m) = \bigcup_{l=0}^{\text{node\_level}(m)} \left( \{(m, v, l) \mid v \in N_l(m)\} \cup \bigcup_{u \in \text{in\_neighbors}(m, l)} \{(u, v, l) \mid v \in N_l(u)\} \right)$$

### Corollary 2: Affected Edges Following Neighbor List Mutation
When graph topology is mutated via `set_neighbors`, `add_edge`, or `remove_edge` on node $u$ at layer $l$:
- The coordinate vectors remain unchanged.
- The neighbor set $N_l(u)$ is updated, while all other neighbor sets $N_{l'}(x)$ for $(x, l') \neq (u, l)$ remain invariant.
- Therefore, the only edges whose validity status can change are **the outgoing edges of $u$ at layer $l$**:
  $$\text{Affected}(u, l) = \{(u, v, l) \mid v \in N_l(u)\}$$

---

## 3. Legacy Baseline Rule and Transition Semantics

In fresh HNSW graphs constructed via standard heuristics (such as `hnswlib`), reverse connections added during `mutuallyConnectNewElement` do not always run the pruning heuristic when destination degree limits allow direct append. As a result:
- A newly built graph naturally contains edges that statically violate the certificate.
- **Baseline Invariance**: Edges that are already `VIOLATED` when first certified at graph construction are classified as **baseline violations**. They do NOT generate failure transitions or trigger repairs.
- **Kinetic Transition Classification**:
  - `VALID -> VIOLATED`: **Failure**. The edge has degraded due to motion and must be queued for kinetic repair.
  - `VIOLATED -> VALID`: **Healed**. Motion has resolved an earlier violation.
  - `VIOLATED -> VIOLATED`: **No event**. The edge was already violated; its witness may update, but no state change occurs.
  - `VALID -> VALID`: **No event**.
  - Newly added edges during rewiring: classified as **`appeared`** with their initial status; they never trigger spurious failure events.

### Invariance of Persistent Violations (VIOLATED -> VIOLATED)
When an edge $(u, v, l)$ transitions from `VIOLATED` to `VIOLATED`—even if the identity of the canonical witness changes from $w_1$ to $w_2$—no `Transition` event is emitted, and the certificate store silently updates its internal record for $(u, v, l)$ to track the updated canonical witness. Emitting no event is mathematically sound because the edge was already known to need repair, so no new repair need be scheduled; the kinetic event heap already holds or will hold a repair for it, and the store witness update is just local bookkeeping so when repair runs it sees the freshest witness.

---

## 4. Distance Metering and Exact Computation Formulas

Every distance calculation executed during detection passes through a dedicated `CostMeter`.

For an active source node $u$ at layer $l$ with out-degree $k = |N_l(u)|$:
1. **Source-to-neighbor distances**: $D(u, w)$ must be evaluated for each $w \in N_l(u)$:
   $$k \text{ distance computations}$$
2. **Neighbor-to-neighbor pairwise distances**: $D(v, w)$ must be evaluated for all unordered pairs $\{v, w\} \subseteq N_l(u)$ with $v \neq w$:
   $$\binom{k}{2} = \frac{k(k - 1)}{2} \text{ distance computations}$$

Total distance computations to certify all outgoing edges of source $(u, l)$ with degree $k$:
$$D(k) = k + \frac{k(k - 1)}{2} = \frac{k(k + 1)}{2}$$

- $k = 0 \implies D(0) = 0$
- $k = 1 \implies D(1) = 1$
- $k = 2 \implies D(2) = 3$
- $k = 3 \implies D(3) = 6$
- $k = 4 \implies D(4) = 10$
- $k = M \implies D(M) = \frac{M(M + 1)}{2}$

By precomputing and caching the $k$ distances from $u$ and the $\binom{k}{2}$ pairwise neighbor distances, `certify_source` achieves the theoretical lower bound with zero redundant distance calls.
