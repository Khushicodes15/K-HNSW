"""Baseline index maintenance strategies: delete-and-reinsert and full rebuild.

Pure Python standard library and NumPy only (browser/Pyodide safe).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from khnsw.core.graph.hnsw import HNSWIndex


def delete_and_reinsert(
    index: HNSWIndex,
    node_id: int,
    new_vector: NDArray[np.float32],
) -> None:
    """Update a node's coordinates by removing and re-inserting it.

    Deletes the node from the index (severing all its previous edges),
    and then re-inserts it with the new vector position using the index's
    standard Algorithm 1 insertion routine.

    Parameters
    ----------
    index : HNSWIndex
        Target HNSW graph index.
    node_id : int
        Identifier of the node to update.
    new_vector : NDArray[np.float32]
        New coordinate position for the node.
    """
    node_id = int(node_id)
    vec = np.asarray(new_vector, dtype=np.float32)

    # delete raises KeyError if node_id not in index
    index.delete(node_id)
    index.add(np.array([node_id], dtype=np.int64), vec.reshape(1, -1))


def rebuild(
    index: HNSWIndex,
    vectors_by_id: Mapping[int, NDArray[np.float32]] | None = None,
) -> HNSWIndex:
    """Construct a fresh HNSWIndex from current vectors, applying any updates.

    Builds an index from scratch using the same configuration parameters
    (dim, m, m0, ef_construction, seed, keep_pruned_connections).

    Parameters
    ----------
    index : HNSWIndex
        Source index providing configuration and existing vectors.
    vectors_by_id : Mapping[int, NDArray[np.float32]] | None, optional
        Optional mapping of node IDs to updated vector positions. For any
        ID present in vectors_by_id, the updated vector is used instead of
        the index's stored vector.

    Returns
    -------
    HNSWIndex
        Freshly constructed HNSW index containing all active nodes.
    """
    dim = index.dim
    m = getattr(index._store, "m", 16)
    m0 = getattr(index._store, "m0", 32)
    ef_construction = getattr(index, "_ef_construction", 200)
    seed = getattr(index, "seed", getattr(index, "_seed", None))
    keep_pruned = getattr(index, "keep_pruned_connections", False)

    fresh = HNSWIndex(
        dim=dim,
        m=m,
        ef_construction=ef_construction,
        seed=seed,
        m0=m0,
        keep_pruned_connections=keep_pruned,
    )

    ids = index.ids()
    if not ids:
        return fresh

    vec_dict: dict[int, NDArray[np.float32]] = {nid: index.vector(nid) for nid in ids}
    if vectors_by_id is not None:
        for nid, v in vectors_by_id.items():
            nid_int = int(nid)
            if nid_int in vec_dict:
                vec_dict[nid_int] = np.asarray(v, dtype=np.float32)

    sorted_ids = sorted(vec_dict.keys())
    ids_arr = np.array(sorted_ids, dtype=np.int64)
    vecs_arr = np.stack([vec_dict[nid] for nid in sorted_ids]).astype(np.float32)

    fresh.add(ids_arr, vecs_arr)
    return fresh
