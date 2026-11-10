"""External baseline adapters (hnswlib, faiss).

Heavy C++ library dependencies are allowed ONLY in this adapters package,
never in `khnsw.core`.
"""

from khnsw.adapters.faiss_adapter import FaissAdapter
from khnsw.adapters.hnswlib_adapter import HnswlibAdapter

__all__ = ["FaissAdapter", "HnswlibAdapter"]
