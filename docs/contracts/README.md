# K-HNSW System Contracts & Schemas

This directory houses the formal interface contracts, metric definitions, and serialized data schemas that govern interactions across all modules of the K-HNSW monorepo (research core, benchmarks, backend API, and React frontend).

> [!IMPORTANT]
> The files located in `examples/` (such as `examples/sample_run.jsonl`) are purely synthetic schema test fixtures used to validate JSON Schema definitions and TypeScript schema parsers. They are **NOT** experimental results or empirical findings. No algorithm has produced these numbers.

## Document Directory
- [`graph_index.md`](file:///c:/Users/User/Downloads/khsnw/docs/contracts/graph_index.md): Prose specification of the `GraphIndex` protocol and invariants.
- [`metrics.md`](file:///c:/Users/User/Downloads/khsnw/docs/contracts/metrics.md): Mathematical definitions of evaluation metrics (Recall@k, latency percentiles, QPS, repair cost, connectivity).
- [`event.schema.json`](file:///c:/Users/User/Downloads/khsnw/docs/contracts/event.schema.json): Formal JSON Schema (draft 2020-12) for kinetic trace events emitted to `EventSink`.
- [`run_manifest.schema.json`](file:///c:/Users/User/Downloads/khsnw/docs/contracts/run_manifest.schema.json): Schema for reproducible benchmark run metadata.
- [`examples/sample_run.jsonl`](file:///c:/Users/User/Downloads/khsnw/docs/contracts/examples/sample_run.jsonl): Conforming synthetic event stream examples for contract test verification.

