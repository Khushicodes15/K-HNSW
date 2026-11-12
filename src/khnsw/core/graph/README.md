# Core Graph Engine (`src/khnsw/core/graph`)

## Purpose
Provides foundational pure Python and NumPy Hierarchical Navigable Small World (HNSW) graph data structures, forward and reverse edge lists, exponential layer generation, entry point management, structural audit tools, flat-array serialization, and maintenance baselines.

---

## Rules
- **Core Purity**: All code under `src/khnsw/core/` must strictly use the Python standard library and NumPy only. External libraries (such as `scipy`, `sklearn`, `torch`, `faiss`, `requests`, or `pydantic`) are forbidden.
- **Pyodide-Compatible**: Code must run cleanly in browser WebAssembly environments (e.g. Pyodide). Avoid recursion in graph traversals to prevent call-stack exhaustion.
- **Distance Metric**: Search and heuristics compute squared Euclidean distance ($\|q - v\|_2^2$). Any cosine dataset must be $\ell_2$-normalized prior to insertion and query.

---

## Modules and Real Signatures

### 1. `store.py`
Low-level multi-layer graph container maintaining coupled forward and reverse adjacency sets.

#### Public Classes and Functions
```python
class GraphStore:
    def __init__(self, dim: int, m: int = 16, m0: int | None = None) -> None: ...
    @property
    def dim(self) -> int: ...
    @property
    def m(self) -> int: ...
    @property
    def m0(self) -> int: ...
    def max_m_for_level(self, level: int) -> int: ...
    def max_level(self) -> int: ...
    def entry_point(self) -> int | None: ...
    def set_entry_point(self, node_id: int | None) -> None: ...
    def node_level(self, node_id: int) -> int: ...
    def ids(self) -> list[int]: ...
    def vector(self, node_id: int) -> NDArray[np.float32]: ...
    def set_vector(self, node_id: int, vector: NDArray[np.float32]) -> None: ...
    def add_node(self, node_id: int, vector: NDArray[np.float32], level: int) -> None: ...
    def neighbors(self, node_id: int, level: int) -> list[int]: ...
    def in_neighbors(self, node_id: int, level: int) -> list[int]: ...
    def add_edge(self, u: int, v: int, level: int) -> None: ...
    def remove_edge(self, u: int, v: int, level: int) -> None: ...
    def remove_node(self, node_id: int) -> None: ...
    def set_neighbors(
        self, node_id: int, level: int, neighbors: Sequence[int] | set[int]
    ) -> None: ...
    def snapshot(self) -> dict[str, Any]: ...
```

#### Key Behaviors
- **Structural Invariants**:
  - *Layer Hierarchy*: A node assigned level $L$ exists across all layers $0 \le l \le L$.
  - *Degree Bounds*: Out-degree is strictly bounded by $M_0$ at level 0 (defaults to $2M$), and by $M$ at levels $l > 0$.
  - *Bidirectional Edge Consistency*: $u \in \text{in\_neighbors}(v, l) \iff v \in \text{neighbors}(u, l)$.
  - *No Self-Loops*: Directed edges $u \to u$ are rejected.
- **Sorted Adjacency**: `neighbors()` and `in_neighbors()` return sorted lists of integer node identifiers.
- **Read-Only Vectors**: Stored vectors have `flags.writeable = False` to guard against unintended coordinate mutation.
- **Node Removal (`remove_node`)**: Symmetrically removes all incident forward and reverse edges across all levels. If the deleted node was the entry point, reassigns it to an existing node at the new `max_level` (or `None` when empty).
- **Empty Graph State**: When no nodes are present, `max_level() == -1` and `entry_point() is None`.

---

### 2. `levels.py`
Hierarchical layer assignment using an exponential decay distribution.

#### Public Classes and Functions
```python
def compute_default_ml(m: int) -> float: ...
def assign_level(u: float, m_l: float) -> int: ...


class LevelGenerator:
    def __init__(
        self,
        m: int = 16,
        m_l: float | None = None,
        seed: int | np.random.Generator | None = None,
    ) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def m_l(self) -> float: ...
    def sample(self) -> int: ...
    def sample_many(self, size: int) -> Sequence[int]: ...
    def __call__(self) -> int: ...
```

#### Key Behaviors
- Uses decay formula $\text{level} = \lfloor -\ln(U) \cdot m_L \rfloor$ where $U \in (0, 1]$ and default $m_L = 1 / \ln(M)$.

---

### 3. `hnsw.py`
HNSW graph index wrapping `GraphStore` with Algorithm 1 insertion and Algorithm 5 k-NN search.

#### Public Classes and Functions
```python
class HNSWIndex:
    def __init__(
        self,
        dim: int,
        m: int = 16,
        ef_construction: int = 200,
        seed: int | None = None,
        m0: int | None = None,
        keep_pruned_connections: bool = False,
    ) -> None: ...

    @property
    def dim(self) -> int: ...
    @property
    def seed(self) -> int | None: ...
    @property
    def keep_pruned_connections(self) -> bool: ...

    def add(self, ids: NDArray[np.int64], vectors: NDArray[np.float32]) -> None: ...
    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        ef: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]: ...
    def move(self, node_id: int, new_vector: NDArray[np.float32]) -> None: ...
    def size(self) -> int: ...
    def contains(self, node_id: int) -> bool: ...
    def delete(self, node_id: int) -> None: ...
    def remove(self, node_id: int) -> None: ...

    # GraphIndex topology queries
    def neighbors(self, node_id: int, level: int) -> list[int]: ...
    def in_neighbors(self, node_id: int, level: int) -> list[int]: ...
    def node_level(self, node_id: int) -> int: ...
    def max_level(self) -> int: ...
    def entry_point(self) -> int | None: ...
    def vector(self, node_id: int) -> NDArray[np.float32]: ...
    def version(self, node_id: int) -> int: ...
    def ids(self) -> list[int]: ...
    def max_degree(self, level: int) -> int: ...
    @property
    def distance_computations(self) -> int: ...
    def snapshot(self) -> dict[str, Any]: ...

    # MutableGraphIndex topology mutations
    def set_neighbors(self, node_id: int, level: int, neighbors: list[int]) -> None: ...
    def add_edge(self, u: int, v: int, level: int) -> None: ...
    def remove_edge(self, u: int, v: int, level: int) -> None: ...
```

#### Key Behaviors
- **Constructor Defaults**:
  - `dim`: vector dimension (required).
  - `m = 16`: maximum out-degree on upper layers.
  - `ef_construction = 200`: candidate pool size during index building.
  - `seed = None`: random generator seed.
  - `m0 = None`: resolves to $2M$ if omitted.
  - `keep_pruned_connections = False`: defaults to `False` to maintain parity with `hnswlib` (which does not top up connections with pruned candidates).
- **Coordinate Drift (`move`)**: Updates vector coordinates and increments `version(node_id)` by 1. Does **not** perform topology rewiring or rebalancing.
- **Node Removal (`delete` / `remove`)**: Removes the node via `GraphStore.remove_node` and deletes its entry from version tracking. Does not repair severed neighbor connections.
- **Distance Accounting**: `distance_computations` accumulates distance calculations evaluated inside `search_layer`. It does not include internal comparisons inside `select_neighbors_heuristic`.

---

### 4. `audit.py`
Structural invariant verification and topological metrics computed using public query interfaces.

#### Public Functions
```python
def degree_stats(
    index: Any,
    ids: Iterable[int] | None = None,
) -> dict[int, dict[str, float | int]]: ...


def check_degree_bounds(
    index: Any,
    ids: Iterable[int] | None = None,
    m: int | None = None,
    m0: int | None = None,
) -> list[str]: ...


def zero_in_degree_count(
    index: Any,
    ids: Iterable[int] | None = None,
    level: int = 0,
) -> int: ...


def reachable_from_entry(
    index: Any,
    ids: Iterable[int] | None = None,
    level: int = 0,
) -> set[int]: ...


def connected_components(
    index: Any,
    ids: Iterable[int] | None = None,
    level: int = 0,
) -> list[set[int]]: ...


def validate_invariants(
    index: Any,
    ids: Iterable[int] | None = None,
    m: int | None = None,
    m0: int | None = None,
) -> list[str]: ...
```

#### Key Behaviors
- `degree_stats`: Per-level summary containing `min`, `max`, `mean` out-degree, and active node `count`.
- `check_degree_bounds`: Returns formatted violation strings for nodes exceeding $M_0$ (level 0) or $M$ (level $> 0$).
- `zero_in_degree_count`: Counts nodes with no incoming edges at `level`. Zero in-degree is reported as an empirical diagnostic metric, not a formal invariant violation.
- `reachable_from_entry`: Traverses forward edges from `entry_point()`. By the hierarchy invariant, the top-level entry point is present at all lower levels $l \le \text{max\_level}$.
- `connected_components`: Computes **weakly connected components** (treating edges as undirected) using an **iterative BFS queue** (avoiding recursion limits).
- `validate_invariants`: Full audit returning an empty list when all invariants hold (entry-point presence and level, layer hierarchy, degree bounds, no self-loops, forward/reverse symmetry).

---

### 5. `serialization.py`
Compact persistence using flat NumPy arrays in compressed NPZ format.

#### Public Functions
```python
def save(index: HNSWIndex, path: str | Path | BinaryIO) -> None: ...
def load(path: str | Path | BinaryIO) -> HNSWIndex: ...
```

#### Key Behaviors
- **Storage Layout**:
  - `params`: `[m, m0, dim, ef_construction, seed, keep_pruned_connections, format_version]` (`int64`, shape `(7,)`).
  - `ids`: node identifiers (`int64`, shape `(N,)`).
  - `vectors`: coordinates (`float32`, shape `(N, dim)`).
  - `levels`: assigned max level (`int64`, shape `(N,)`).
  - `entry_point`: top entry point scalar (`int64`, `-1` if empty).
  - `versions`: mutation version counters (`int64`, shape `(N,)`).
  - `edge_level_{l}_indptr` & `edge_level_{l}_indices`: CSR arrays for forward edges at layer $l$.
- **Security & Portability**: Uses `np.savez_compressed` and `np.load(..., allow_pickle=False)`. Accepts file paths (`str`, `Path`) or open binary streams (`BinaryIO`).
- **Load Validation**: Verifies format version, array dimensions, dtypes, ID uniqueness, degree bounds, and runs `validate_invariants`. Reverse edge sets are safely reconstructed from forward edges.

---

### 6. `baselines.py`
Baseline update strategies for kinetic comparison.

#### Public Functions
```python
def delete_and_reinsert(
    index: HNSWIndex,
    node_id: int,
    new_vector: NDArray[np.float32],
) -> None: ...


def rebuild(
    index: HNSWIndex,
    vectors_by_id: Mapping[int, NDArray[np.float32]] | None = None,
) -> HNSWIndex: ...
```

#### Key Behaviors
- `delete_and_reinsert`: Calls `index.delete(node_id)` (which removes incident edges without repairing remaining neighbors) and re-inserts the node at `new_vector` via `index.add()`.
- `rebuild`: Reconstructs a fresh `HNSWIndex` from all current vectors (applying any overrides in `vectors_by_id`) using the same parameters and random seed.

---

## Usage Example

```python
import tempfile
from pathlib import Path
import numpy as np

from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.graph import audit, serialization

# 1. Initialize index
dim, m, ef_construction = 4, 16, 50
index = HNSWIndex(dim=dim, m=m, ef_construction=ef_construction, seed=42)

# 2. Add vectors
vectors = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
ids = np.array([10, 20, 30, 40], dtype=np.int64)
index.add(ids, vectors)

# 3. Query approximate nearest neighbors
query = np.array([1.0, 0.1, 0.0, 0.0], dtype=np.float32)
labels, dists = index.search(query, k=2, ef=10)
print("Search labels:", labels)  # e.g. [10, 20]
print("Search squared L2 dists:", dists)  # e.g. [0.01, 1.81]

# 4. Verify graph invariants
violations = audit.validate_invariants(index)
assert len(violations) == 0, f"Invariant violations: {violations}"

# 5. Save and reload index
with tempfile.TemporaryDirectory() as tmpdir:
    save_path = Path(tmpdir) / "index.npz"
    serialization.save(index, save_path)
    loaded_index = serialization.load(save_path)

    loaded_labels, loaded_dists = loaded_index.search(query, k=2, ef=10)
    assert np.array_equal(labels, loaded_labels)
```
