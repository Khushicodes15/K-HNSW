# GraphIndex Contract Specification

The index contracts specify the interfaces that any pure or adapted index implementations in K-HNSW must fulfill. The contracts are layered hierarchically:

```mermaid
classDiagram
    class ANNIndex {
        +int dim
        +add(ids, vectors)
        +move(node_id, new_vector)
        +search(query, k, ef) tuple
        +size() int
        +contains(node_id) bool
        +remove(node_id)
    }
    class GraphIndex {
        +neighbors(node_id, level) list
        +in_neighbors(node_id, level) list
        +node_level(node_id) int
        +max_level() int
        +entry_point() int
        +vector(node_id) ndarray
        +version(node_id) int
        +ids() list
        +max_degree(level) int
        +int distance_computations
        +snapshot() dict
    }
    class MutableGraphIndex {
        +set_neighbors(node_id, level, neighbors)
        +add_edge(u, v, level)
        +remove_edge(u, v, level)
    }
    ANNIndex <|-- GraphIndex
    GraphIndex <|-- MutableGraphIndex
```

## Architectural Model & Layer Hierarchy
The graph is organized into $L + 1$ discrete layers indexed from $l = 0$ (dense base layer containing all nodes) up to $l = L$ (sparse top layer containing long-range links).
An empty graph returns `max_level() == -1` and `entry_point() == None`.

```mermaid
flowchart TD
    subgraph Layer2 ["Layer 2 (Top Layer, Entry Point)"]
        N1_L2["Node 1"] --> N5_L2["Node 5"]
    end
    subgraph Layer1 ["Layer 1 (Intermediate)"]
        N1_L1["Node 1"] --> N3_L1["Node 3"]
        N3_L1 --> N5_L1["Node 5"]
    end
    subgraph Layer0 ["Layer 0 (Base Layer, All Nodes)"]
        N1_L0["Node 1"] --> N2_L0["Node 2"]
        N2_L0 --> N3_L0["Node 3"]
        N3_L0 --> N4_L0["Node 4"]
        N4_L0 --> N5_L0["Node 5"]
    end
```

## Distance Metric & Coordinate Contracts
- **Squared L2 Distance**: `search()` strictly computes and returns squared Euclidean distances ($\|q - v\|_2^2$).
- **Cosine Data**: Any dataset operating under cosine similarity or angular distance must be $\ell_2$-normalized prior to insertion and query (under unit normalization, $\|q - v\|_2^2 = 2 - 2 \langle q, v \rangle$).

## Lifecycle, Versioning, and Call Order
Under kinetic updates, components must follow a strict call sequence:
1. **Read Old Vector**: Query `v_old = index.vector(node_id)`.
2. **Move Node Coordinates**: Invoke `index.move(node_id, v_new)`. This updates the internal vector table and increments `version(node_id)` by 1.
3. **Notify Repairer**: Invoke `repairer.on_move(index, node_id, v_old, v_new, sink)` to trigger certificate evaluations and repair queuing.

## MutableGraphIndex Contract
`MutableGraphIndex` provides primitive mutation methods used by repair algorithms:
- `set_neighbors(node_id, level, neighbors)`: Atomically updates the forward neighbor list while updating reverse edge tables (`in_neighbors`) symmetrically for any added or severed links.
- `add_edge(u, v, level)`: Adds a directed edge $u \to v$ at layer $l$, ensuring $u \in \text{in\_neighbors}(v, l)$.
- `remove_edge(u, v, level)`: Removes directed edge $u \to v$ at layer $l$, ensuring $u \notin \text{in\_neighbors}(v, l)$.
- **Accounting**: Every mutation method must account for any pairwise distance calculations executed by incrementing `distance_computations`.

## Repair Budget & Invariants
- **Budget Unit**: Repair budgets allocated to `run_repairs(index, budget, sink)` are denominated strictly in **distance computations**.
- **Reachability Postcondition**: Reachability is **not** guaranteed to be preserved under arbitrary drift budgets; instead, reachability is **measured and reported** continuously as an evaluation metric.
