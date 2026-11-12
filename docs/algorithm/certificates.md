# Kinetic Geometric Certificates for HNSW Topology

## 1. Definitions and Invariant Formulation

In a Hierarchical Navigable Small World (HNSW) graph, the edge selection heuristic maintains a diverse set of neighbors to ensure logarithmic search routing and angular coverage. Let $p_i(t) \in \mathbb{R}^d$ denote the spatial position of node $i$ at simulation time $t \ge t_0$.

We define the squared Euclidean distance between two nodes $i$ and $j$ as:
$$D(i, j) = \|p_i - p_j\|_2^2$$

### Edge Violation by a Witness
For a directed edge $u \to v$ in layer $l$, another neighbor $w \in N_l(u) \setminus \{v\}$ acts as a **witness** that invalidates edge $u \to v$ if and only if:
$$D(u, w) \le D(u, v) \quad \text{AND} \quad D(v, w) < D(u, v)$$

- **Validity Condition**: The edge $u \to v$ is **valid** if and only if no witness $w \in N_l(u) \setminus \{v\}$ satisfies the violation condition.
- **Geometric Intuition**: If $w$ is closer to $u$ than $v$ is ($D(u, w) \le D(u, v)$), and $w$ is also closer to $v$ than $u$ is ($D(v, w) < D(u, v)$), then the edge $u \to v$ violates the relative neighborhood heuristic. Routing from $u$ towards $v$ is better served by stepping through $w$, making direct link $u \to v$ redundant and susceptible to long-range distortion.

---

## 2. Relationship to HNSW Heuristics and Proxy Role

### `[verify against hnswlib source]`
In the reference C++ implementation of `hnswlib` (`hnswalg.h` method `getNeighborsByHeuristic2`):
1. Candidate neighbors are sorted by ascending distance to the base node $u$.
2. Candidates are greedily accepted only if they are closer to $u$ than to any previously selected neighbor in the result set.
3. However, when bidirectional connections are added during graph assembly, reverse links may be appended without executing the full pruning heuristic on the target node's adjacency list. Consequently, **fresh static HNSW graphs can already contain edges that violate this condition**.

### The Certificate as a Kinetic Proxy
The certificate condition:
- Is a **heuristic proxy** for HNSW graph navigability, not a mathematical proof of reachability or greedy routing success.
- **K-HNSW reacts only to valid $\to$ violated transitions**. Existing static violations present at construction time do not trigger spurious repairs; repair mechanisms respond specifically to kinetic degradation caused by vector updates.

---

## 3. Kinetic Trajectories and Quadratic Distance Form

Under piecewise linear kinematic motion, each node $i$ moves along a linear trajectory:
$$p_i(t) = a_i + b_i t$$
where $a_i \in \mathbb{R}^d$ is the position offset and $b_i \in \mathbb{R}^d$ is the constant velocity vector.

For any pair of nodes $i, j$, the displacement vector $\Delta(t) = p_j(t) - p_i(t)$ is linear in $t$:
$$\Delta(t) = \Delta_a + \Delta_b t$$
where:
$$\Delta_a = a_j - a_i, \quad \Delta_b = b_j - b_i$$

The squared distance $D(i, j)(t)$ is a quadratic polynomial in $t$:
$$D(i, j)(t) = \|\Delta_a + \Delta_b t\|_2^2 = \alpha t^2 + \beta t + \gamma$$
where:
$$\alpha = \|\Delta_b\|_2^2 = \Delta_b \cdot \Delta_b$$
$$\beta = 2 (\Delta_a \cdot \Delta_b)$$
$$\gamma = \|\Delta_a\|_2^2 = \Delta_a \cdot \Delta_a$$

Because $\alpha = \|\Delta_b\|^2 \ge 0$ and $\gamma = \|\Delta_a\|^2 \ge 0$, $D(i, j)(t)$ is convex and non-negative for all $t \in \mathbb{R}$.

---

## 4. Invalidation Functions and Failure Time Computation

To evaluate whether witness $w$ violates edge $u \to v$, we define two quadratic scalar functions:
$$g_1(t) = D(u, w)(t) - D(u, v)(t)$$
$$g_2(t) = D(v, w)(t) - D(u, v)(t)$$

An edge is violated at time $t$ if and only if:
$$g_1(t) \le 0 \quad \text{AND} \quad g_2(t) < 0$$

### Failure Time Determination
The failure time $t^*(u, v, w)$ is the earliest timestamp $t \ge t_0$ within the planning horizon $[t_0, t_0 + H]$ at which the violation predicate becomes true:
1. **Immediate Failure**: If $g_1(t_0) \le 0$ and $g_2(t_0) < 0$, the edge is already violated at $t_0$; failure time is $t_0$.
2. **Roots of $g_1$ and $g_2$**: Compute all real roots of $g_1(t) = 0$ and $g_2(t) = 0$.
3. **Partition into Intervals**: Sort all roots falling strictly within $(t_0, t_0 + H]$:
   $$t_0 < \tau_1 < \tau_2 < \dots < \tau_m \le t_0 + H$$
4. **Interval Testing**: On each open interval $(\tau_k, \tau_{k+1})$, the signs of $g_1(t)$ and $g_2(t)$ are invariant. Test the predicate at midpoint $t_{\text{mid}} = \frac{\tau_k + \tau_{k+1}}{2}$:
   - If the open interval is violated, the violation starts at boundary $\tau_k$.
   - If the interval is not violated, evaluate the boundary root $\tau_{k+1}$.
5. **Edge Failure Time**: The edge failure time is the minimum over all current neighbors of $u$:
   $$t^*(u \to v) = \min_{w \in N_l(u) \setminus \{v\}} t^*(u, v, w)$$

---

## 5. Computational and Memory Complexity

### Continuous Motion Mode
- **Distance Coefficients**: Computing $(\alpha, \beta, \gamma)$ requires 2 vector subtractions and 3 dot products in $\mathbb{R}^d$: $O(d)$ operations.
- **Root Solving**: Roots of degree $\le 2$ polynomials are obtained in $O(1)$ operations via the quadratic formula.
- **Per-Edge Certificate**: For an edge with maximum out-degree $M$, comparing against all candidate witnesses requires $O(M d)$ operations.
- **Memory Footprint**: By storing only **one active certificate per directed edge** (recording the earliest-failing witness index $w^*$ and failure timestamp $t^*$), total certificate memory is strictly $O(E) = O(M |V|)$.

### Discrete Movement Updates
When a single node $x$ moves discretely to a new position, the set of affected edges that must be re-certified comprises:
1. **Out-edges of $x$**: Edges $x \to v$ for $v \in N_l(x)$ ($|N_l(x)| \le M$ edges).
2. **In-edges to $x$**: Edges $u \to x$ for $u \in \text{in}_l(x)$ ($|\text{in}_l(x)|$ edges).
3. **Witnessed edges**: Edges $u \to v$ where $u \in \text{in}_l(x)$ and $x \in N_l(u)$ acts as a witness ($|\text{in}_l(x)| \cdot M$ edges).

Total re-evaluation cost per moved node is:
$$O(M^2 + |\text{in}_l(x)| M) \text{ distance checks}$$

---

## 6. Numerical Policy and Degeneracy Handling

1. **Precision**: All certificate calculations (vector differences, dot products, polynomial subtractions, discriminants, and roots) are performed in **64-bit floating point (`float64`)**, even when stored node embeddings are `float32`.
2. **Numerically Stable Quadratic Formula**:
   To avoid catastrophic cancellation when $\beta \approx \sqrt{\beta^2 - 4\alpha\gamma}$, we use the Citardauq / Muller formulation:
   $$q = -\frac{1}{2} \left( \beta + \operatorname{sgn}(\beta) \sqrt{\beta^2 - 4\alpha\gamma} \right)$$
   $$t_1 = \frac{q}{\alpha}, \quad t_2 = \frac{\gamma}{q}$$
3. **Ties**: Distance ties are **never violations**. The condition $D(v, w) < D(u, v)$ requires strict inequality; if $D(v, w) \ge D(u, v) - \text{tol}$, the edge is not violated.
4. **Degenerate Cases**:
   - **Constant Polynomial** ($\alpha \approx 0, \beta \approx 0$): No finite roots; sign is constant for all $t$.
   - **Linear Degeneracy** ($\alpha \approx 0, |\beta| > \text{tol}$): Single root at $t = -\gamma / \beta$.
   - **Tangent Root** ($|\Delta| \le \text{tol}$): Single double root at $t = -\beta / (2\alpha)$.

---

## 7. Limitations and Research Scope

- **Proxy Approximation**: Edge invalidation certificates approximate HNSW routing degradation locally. A certificate failure indicates heuristic sub-optimality, but greedy routing may still navigate through alternative multi-hop paths.
- **Piecewise Linear Assumption**: Trajectories in continuous simulations are modeled as piecewise linear. Arbitrary non-linear kinematic curves require interval discretization or Taylor approximations.
- **Horizon Truncation**: Failures occurring past $t_0 + H$ are truncated to conserve priority queue size; un-repaired distant failures must be refreshed when the simulation advances past $t_0 + H$.
- **Cascading Invalidation**: A single moving node may trigger multiple certificate failures simultaneously; priority scheduling is essential to allocate bounded repair budgets efficiently.
