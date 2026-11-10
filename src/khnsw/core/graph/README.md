# Core Graph Engine (`src/khnsw/core/graph`)

## Purpose
Provides the foundational pure Python + NumPy Hierarchical Navigable Small World (HNSW) graph data structures, forward and reverse edge lists, layer assignment, entry point tracking, and structural audit tools.

## Owner Role
**Member 2 (Graph Engine)**

## What Goes Here
- Concrete pure implementation of `GraphIndex` using NumPy arrays and native containers.
- Forward adjacency tables and reverse-edge indices (`in_neighbors`) across all hierarchy layers.
- Layer generation distributions (exponential decay parameter $m_L$).
- Graph consistency audit functions (connectivity checks, degree bound verification, symmetry audits).
- Serialization and snapshot logic (`snapshot()`) for state replication.

