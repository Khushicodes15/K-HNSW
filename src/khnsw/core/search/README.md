# Search & Routing Engine (`src/khnsw/core/search`)

## Purpose
Implements multi-layer greedy routing and beam search (`search_layer`) over the hierarchical graph index, returning approximate top-k nearest neighbors.

## Owner Role
**Member 2 (Graph Engine)** (in collaboration with Member 1 algorithm specification)

## What Goes Here
- Pure greedy navigation across upper hierarchical layers ($l > 0$).
- Beam search with candidate queue size $ef$ at layer 0.
- Distance caching and metric evaluation counters.
- Search event tracing (`search`) emitted to `EventSink` for query replay visualization.

