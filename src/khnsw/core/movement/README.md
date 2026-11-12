# Movement Models (`src/khnsw/core/movement`)

## Purpose
Simulates continuous or discrete spatial movement of embedded vector points over simulation time ($t \ge 0.0$), generating deterministic trajectory vectors for kinetic evaluations and HNSW graph topology degradation experiments.

## Owner Role
**Member 3 (Update/Repair Engine - Repo Owner - Khushi Dhona)**

## Implemented Movement Models

All models conform to the `MovementModel` protocol (`name`, `seed`, `positions(t) -> NDArray[np.float32]`, `changed_nodes(t_prev, t_curr) -> list[int]`).

1. **`LinearDrift` (`src/khnsw/core/movement/linear.py`)**:
   - Continuous linear motion: $p_i(t) = a_i + b_i \cdot t$.
   - $a_i = \text{base}_i$ (initial coordinate in float64).
   - $b_i = u_i \cdot (\text{speed} \cdot \text{scale})$, where $u_i$ is a deterministic random unit vector and $\text{scale} = \text{mean\_nn\_distance}(\text{base})$.
   - **Implements**: `LinearTrajectory` protocol (`trajectory(node_id) -> (a, b)` returning float64 arrays).

2. **`ClusterDrift` (`src/khnsw/core/movement/cluster.py`)**:
   - Semantic cluster motion with localized Brownian noise: $p_i(t) = a_i + b_i \cdot t$.
   - Cluster assignment: Deterministic seeded k-means-lite (10 Lloyd iterations with seeded initial centroids).
   - Velocity: $b_i = V_{\text{cluster}(i)} + v_{\text{noise}, i}$, where $V_k$ is the shared cluster velocity and $v_{\text{noise}, i}$ is node-specific noise.
   - **Implements**: `LinearTrajectory` protocol (`trajectory(node_id) -> (a, b)` returning float64 arrays).

3. **`DiscreteReplacement` (`src/khnsw/core/movement/discrete.py`)**:
   - Periodic discrete updates at integer steps $s = 1, 2, \dots$:
     $p(t) = \text{base} + \sum_{s=1}^{\lfloor t \rfloor} \Delta_s$.
   - At each step $s$, a deterministic subset of $\text{round}(\text{fraction} \times n)$ nodes (sampled via `rng_for(seed, s)`) is moved by exactly $\text{displacement} \times \text{scale}$ in a random unit direction.
   - Cumulative state is computed in strictly ordered steps and cached, ensuring that query evaluation order never affects numerical state.
   - **Does NOT implement**: `LinearTrajectory`.

4. **`TeleportAdversary` (`src/khnsw/core/movement/teleport.py`)**:
   - Adversarial non-local relocation benchmark stressing local graph repair.
   - At each integer step $s = 1, 2, \dots$, a deterministic subset of $\text{round}(\text{fraction} \times n)$ nodes is relocated to $\text{base}[j] + \text{jitter}$ for a randomly chosen other node $j \ne i$, where $\|\text{jitter}\|_2 \le 0.05 \times \text{scale}$.
   - **Does NOT implement**: `LinearTrajectory`.

## Utilities

- **`mean_nn_distance(X, sample=1000, seed=0)` (`src/khnsw/core/movement/base.py`)**:
  Exact brute-force Euclidean nearest-neighbor distance (L2, not squared) averaged across a deterministic sample of rows, excluding self. Returns `0.0` when $n < 2$.
- **`rng_for(seed, step)` (`src/khnsw/core/movement/base.py`)**:
  Generates an isolated NumPy `Generator` seeded with `SeedSequence([seed, step])` for order-independent deterministic per-step randomness.
