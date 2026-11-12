"""Flat array graph serialization and deserialization for HNSWIndex.

Pure Python standard library and NumPy only (browser/Pyodide safe).
Uses np.savez_compressed with allow_pickle=False for secure, portable persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np

from khnsw.core.graph.audit import validate_invariants
from khnsw.core.graph.hnsw import HNSWIndex

FORMAT_VERSION = 1


def save(index: HNSWIndex, path: str | Path | BinaryIO) -> None:
    """Save an HNSWIndex to an NPZ file using flat NumPy arrays.

    Archive Layout
    --------------
    - ``params`` : int64 1D array of shape (7,)
        ``[m, m0, dim, ef_construction, seed, keep_pruned_connections, format_version]``
    - ``ids`` : int64 1D array of shape (N,)
        Indexed node identifiers in sorted order.
    - ``vectors`` : float32 2D array of shape (N, dim)
        Coordinate positions corresponding to ``ids``.
    - ``levels`` : int64 1D array of shape (N,)
        Assigned top hierarchy layer levels for each node.
    - ``entry_point`` : int64 scalar (value -1 if graph is empty).
    - ``versions`` : int64 1D array of shape (N,)
        Monotonic coordinate mutation version counters.
    - ``edge_level_{l}_indptr`` : int64 1D array of shape (N + 1,)
        CSR row pointer for outgoing edges at hierarchy layer :math:`l`.
    - ``edge_level_{l}_indices`` : int64 1D array of shape (E_l,)
        CSR column indices (neighbor node IDs) for layer :math:`l`.

    Parameters
    ----------
    index : HNSWIndex
        Source index instance to serialize.
    path : str | Path | BinaryIO
        Target file path or open binary write stream.
    """
    id_list = index.ids()
    n = len(id_list)
    dim = index.dim
    m = index.max_degree(1) if index.max_level() >= 1 else getattr(index._store, "m", 16)
    m0 = index.max_degree(0) if index.max_level() >= 0 else getattr(index._store, "m0", 32)
    ef_construction = getattr(index, "_ef_construction", 200)
    seed = getattr(index, "_seed", -1)
    if seed is None:
        seed = -1
    keep_pruned = int(getattr(index, "keep_pruned_connections", False))

    params = np.array(
        [m, m0, dim, ef_construction, seed, keep_pruned, FORMAT_VERSION],
        dtype=np.int64,
    )
    ids_arr = np.array(id_list, dtype=np.int64)

    if n > 0:
        vectors_arr = np.stack([index.vector(nid) for nid in id_list]).astype(np.float32)
        levels_arr = np.array([index.node_level(nid) for nid in id_list], dtype=np.int64)
        versions_arr = np.array([index.version(nid) for nid in id_list], dtype=np.int64)
    else:
        vectors_arr = np.empty((0, dim), dtype=np.float32)
        levels_arr = np.empty((0,), dtype=np.int64)
        versions_arr = np.empty((0,), dtype=np.int64)

    ep = index.entry_point()
    entry_point_arr = np.int64(ep if ep is not None else -1)

    arrays_to_save: dict[str, np.ndarray] = {
        "params": params,
        "ids": ids_arr,
        "vectors": vectors_arr,
        "levels": levels_arr,
        "entry_point": entry_point_arr,
        "versions": versions_arr,
    }

    max_lvl = index.max_level()
    if n > 0 and max_lvl >= 0:
        for lvl in range(max_lvl + 1):
            indptr = [0]
            indices: list[int] = []
            for nid in id_list:
                if index.node_level(nid) >= lvl:
                    nbrs = index.neighbors(nid, lvl)
                    indices.extend(nbrs)
                    indptr.append(indptr[-1] + len(nbrs))
                else:
                    indptr.append(indptr[-1])

            arrays_to_save[f"edge_level_{lvl}_indptr"] = np.array(indptr, dtype=np.int64)
            arrays_to_save[f"edge_level_{lvl}_indices"] = np.array(indices, dtype=np.int64)

    np.savez_compressed(path, **arrays_to_save)


def load(path: str | Path | BinaryIO) -> HNSWIndex:
    """Load and validate an HNSWIndex from an NPZ file.

    Guarantees:
        - Rebuilds reverse edges dynamically from validated forward edges.
        - Enforces strict structural invariants, degree bounds, and array schemas.
        - Validates invariants using ``audit.validate_invariants``.

    Parameters
    ----------
    path : str | Path | BinaryIO
        Source file path or open binary read stream.

    Returns
    -------
    HNSWIndex
        Fully reconstituted, validated index instance ready for queries and inserts.

    Raises
    ------
    ValueError
        If the file is corrupted, format version unsupported, array dimensions invalid,
        or topological invariants violated.
    """
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as e:
        raise ValueError(f"Failed to load NPZ archive (corrupted or invalid format): {e}") from e

    with data:
        files = set(data.files)
        required_keys = {"params", "ids", "vectors", "levels", "entry_point"}
        missing = required_keys - files
        if missing:
            raise ValueError(f"Archive missing required arrays: {sorted(missing)}")

        params = data["params"]
        if params.ndim != 1 or params.shape[0] < 7:
            raise ValueError(f"Invalid params array shape: expected (7,), got {params.shape}")
        if params.dtype != np.int64 and not np.issubdtype(params.dtype, np.integer):
            raise ValueError(f"Invalid params dtype: expected integer, got {params.dtype}")

        m = int(params[0])
        m0 = int(params[1])
        dim = int(params[2])
        ef_construction = int(params[3])
        seed_raw = int(params[4])
        seed = seed_raw if seed_raw >= 0 else None
        keep_pruned_connections = bool(params[5])
        format_version = int(params[6])

        if format_version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported format_version {format_version}, expected {FORMAT_VERSION}"
            )
        if dim <= 0:
            raise ValueError(f"Invalid dimension {dim}: must be positive")
        if m <= 1:
            raise ValueError(f"Invalid m {m}: must be > 1")
        if m0 <= 0:
            raise ValueError(f"Invalid m0 {m0}: must be positive")
        if ef_construction < 1:
            raise ValueError(f"Invalid ef_construction {ef_construction}: must be >= 1")

        ids_arr = data["ids"]
        vectors_arr = data["vectors"]
        levels_arr = data["levels"]
        entry_point_val = int(data["entry_point"])

        if ids_arr.ndim != 1:
            raise ValueError(f"ids must be 1D array, got {ids_arr.ndim}D")
        n = ids_arr.shape[0]

        if np.any(ids_arr < 0):
            raise ValueError("All node IDs in archive must be non-negative")
        if len(np.unique(ids_arr)) != n:
            raise ValueError("Node IDs in archive must be unique")

        if vectors_arr.ndim != 2 or vectors_arr.shape != (n, dim):
            raise ValueError(
                f"vectors array shape mismatch: expected ({n}, {dim}), got {vectors_arr.shape}"
            )
        if vectors_arr.dtype != np.float32:
            raise ValueError(f"vectors array dtype must be float32, got {vectors_arr.dtype}")

        if levels_arr.ndim != 1 or levels_arr.shape != (n,):
            raise ValueError(
                f"levels array shape mismatch: expected ({n},), got {levels_arr.shape}"
            )
        if n > 0 and np.any(levels_arr < 0):
            raise ValueError("Node levels in archive must be non-negative")

        id_set = set(ids_arr.tolist())
        id_to_level = {int(nid): int(lvl) for nid, lvl in zip(ids_arr, levels_arr, strict=True)}

        max_lvl = -1 if n == 0 else int(np.max(levels_arr))

        if n == 0:
            if entry_point_val != -1:
                raise ValueError(f"Empty index must have entry_point == -1, got {entry_point_val}")
        else:
            if entry_point_val == -1:
                raise ValueError("Non-empty index has invalid entry_point == -1")
            if entry_point_val not in id_set:
                raise ValueError(f"Entry point {entry_point_val} does not exist in node IDs")
            if id_to_level[entry_point_val] != max_lvl:
                raise ValueError(
                    f"Entry point {entry_point_val} level {id_to_level[entry_point_val]} "
                    f"!= max_level {max_lvl}"
                )

        # Validate edge arrays per level
        level_edges: dict[int, dict[int, list[int]]] = {}
        if n > 0 and max_lvl >= 0:
            for lvl in range(max_lvl + 1):
                indptr_key = f"edge_level_{lvl}_indptr"
                indices_key = f"edge_level_{lvl}_indices"
                if indptr_key not in files or indices_key not in files:
                    raise ValueError(f"Archive missing edge arrays for level {lvl}")

                indptr = data[indptr_key]
                indices = data[indices_key]

                if indptr.ndim != 1 or indptr.shape != (n + 1,):
                    raise ValueError(
                        f"{indptr_key} shape mismatch: expected ({n + 1},), got {indptr.shape}"
                    )
                if indices.ndim != 1:
                    raise ValueError(f"{indices_key} must be a 1D array")
                if indptr[0] != 0:
                    raise ValueError(f"{indptr_key}[0] must be 0, got {indptr[0]}")
                if np.any(np.diff(indptr) < 0):
                    raise ValueError(f"{indptr_key} must be monotonically non-decreasing")
                if indptr[-1] != indices.shape[0]:
                    raise ValueError(
                        f"{indptr_key}[-1] ({indptr[-1]}) != "
                        f"len({indices_key}) ({indices.shape[0]})"
                    )

                max_deg = m0 if lvl == 0 else m
                level_edges[lvl] = {}

                for i in range(n):
                    nid = int(ids_arr[i])
                    node_lvl = id_to_level[nid]
                    start = int(indptr[i])
                    end = int(indptr[i + 1])
                    deg = end - start

                    if node_lvl < lvl:
                        if deg != 0:
                            raise ValueError(
                                f"Node {nid} has level {node_lvl} < {lvl} but degree {deg} > 0"
                            )
                        continue

                    if deg > max_deg:
                        raise ValueError(
                            f"Node {nid} at level {lvl} has out-degree {deg} > max {max_deg}"
                        )

                    nbrs = indices[start:end].tolist()
                    nbr_set = set(nbrs)
                    if len(nbr_set) != deg:
                        raise ValueError(
                            f"Duplicate neighbors detected for node {nid} at level {lvl}"
                        )
                    if nid in nbr_set:
                        raise ValueError(f"Self-loop detected on node {nid} at level {lvl}")

                    for nbr in nbrs:
                        if nbr not in id_set:
                            raise ValueError(
                                f"Node {nid} at level {lvl} references non-existent neighbor {nbr}"
                            )
                        if id_to_level[nbr] < lvl:
                            raise ValueError(
                                f"Neighbor {nbr} maximum level {id_to_level[nbr]} < level {lvl}"
                            )

                    level_edges[lvl][nid] = nbrs

        # Construct HNSWIndex
        index = HNSWIndex(
            dim=dim,
            m=m,
            ef_construction=ef_construction,
            seed=seed,
            m0=m0,
            keep_pruned_connections=keep_pruned_connections,
        )
        if seed is not None:
            index._seed = seed

        # Restore versions if present
        versions_arr = data["versions"] if "versions" in files else None

        # Add all nodes
        for i in range(n):
            nid = int(ids_arr[i])
            vec = vectors_arr[i]
            lvl = int(levels_arr[i])
            index._store.add_node(nid, vec, lvl)
            if versions_arr is not None:
                index._versions[nid] = int(versions_arr[i])
            else:
                index._versions[nid] = 0

        # Wire forward edges and synchronously reconstruct reverse edges
        if n > 0 and max_lvl >= 0:
            for lvl in range(max_lvl + 1):
                for nid, nbrs in level_edges[lvl].items():
                    index._store.set_neighbors(nid, lvl, nbrs)

            index._store.set_entry_point(entry_point_val)

        # Enforce all structural invariants before returning
        violations = validate_invariants(index, m=m, m0=m0)
        if violations:
            raise ValueError(f"Loaded index violates invariants: {violations}")

        return index
