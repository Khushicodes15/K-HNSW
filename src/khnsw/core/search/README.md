# Search & Routing Engine (`src/khnsw/core/search`)

## Purpose
Provides layer-level beam exploration (`search_layer`) and relative-neighborhood diversity heuristic neighbor selection (`select_neighbors_heuristic`) for Hierarchical Navigable Small World graphs.

---

## Rules
- **Core Purity**: All code under `src/khnsw/core/` must strictly use the Python standard library and NumPy only. External libraries (`scipy`, `sklearn`, `torch`, `faiss`, etc.) are forbidden.
- **Pyodide-Compatible**: Code must run cleanly in browser WebAssembly environments (e.g. Pyodide). Avoid recursion in traversals.
- **Distance Metric**: Evaluates squared Euclidean distance ($\|q - v\|_2^2$). Cosine data must be $\ell_2$-normalized prior to insertion and query.

---

## Modules and Real Signatures

### 1. `search_layer.py`
Single-layer beam search implementing HNSW Algorithm 2 (Malkov & Yashunin, 2018).

#### Public Classes and Functions
```python
class SearchResult(NamedTuple):
    distance: float
    node_id: int

class DistanceCounter:
    __slots__ = ("value",)
    def __init__(self, value: int = 0) -> None: ...

def search_layer(
    store: GraphStore,
    query: NDArray[np.float32],
    entry_points: Sequence[int] | set[int] | int,
    ef: int,
    level: int,
    counter: DistanceCounter | None = None,
) -> list[SearchResult]: ...
```

#### Key Behaviors
- **Algorithm**: Implements HNSW Algorithm 2 using a candidate min-heap and a dynamic result max-heap of capacity bounded by `ef`.
- **Distance Metric**: Strictly computes squared Euclidean distance ($\|q - v\|_2^2$).
- **Distance Accounting (`DistanceCounter`)**: If an instance of `DistanceCounter` is passed to `counter`, its `.value` attribute is incremented by the exact count of pairwise vector distance evaluations executed during the search.
- **Result Ordering**: Returns a `list[SearchResult]` containing up to `ef` elements sorted in strictly ascending order of squared L2 distance (closest candidates first).

---

### 2. `heuristic.py`
Diverse-neighbor selection heuristic implementing HNSW Algorithm 4 (Malkov & Yashunin, 2018).

#### Public Functions
```python
def select_neighbors_heuristic(
    store: GraphStore,
    base: int | NDArray[np.float32],
    candidates: Sequence[int],
    m: int,
    *,
    keep_pruned_connections: bool = False,
    extend_candidates: bool = False,
    level: int = 0,
) -> list[int]: ...
```

#### Key Behaviors
- **Flexible Base Representation**: `base` can be either an existing integer node ID present in `store`, or a raw coordinate vector `NDArray[np.float32]` of shape `(store.dim,)`.
- **Diversity Condition (Algorithm 4)**: Evaluates candidates in ascending distance order to `base`. A candidate $e$ is accepted only if it is closer to `base` than to any previously accepted neighbor ($\|e - \text{base}\|_2^2 < \|e - n\|_2^2$ for all accepted $n$).
- **Self-Exclusion**: The base node itself is automatically excluded from the returned neighbor list.
- **Pruned Connections Top-Up (`keep_pruned_connections`)**: When `keep_pruned_connections=True`, any remaining capacity up to $m$ is topped up from discarded candidates in ascending distance order. Defaults to `False` to maintain parity with `hnswlib`.
- **Return Type & Order**: Returns `list[int]` containing up to $m$ selected neighbor node identifiers sorted in ascending order of distance to `base`.
