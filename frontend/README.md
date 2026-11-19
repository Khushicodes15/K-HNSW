# K-HNSW Graph Studio

The React + TypeScript frontend provides seven interactive views:

- **Graph playground:** D3-powered force layout on Canvas, with layer filters, zoom, pan, node selection, and adjacency inspection.
- **Movement simulator:** discrete replacement, linear drift, cluster drift, and adversarial motion with playback and a displacement chart.
- **Violation view:** affected edges, certificate details, predicted failure times, and repair-needed nodes.
- **Priority view:** urgency ranking, illustrative recall-loss estimates, repair costs, and budget-fit scheduling.
- **Rewiring view:** side-by-side before/after topology with removed, added, and affected elements highlighted.
- **Baseline vs K-HNSW:** comparison table and chart for static HNSW, native update, delete/reinsert, full rebuild, and K-HNSW.
- **Live metrics:** event-by-event trace replay with recall, mean query latency, repair pops, edge changes, connectivity, and processing-time summaries.

The graph, movement, violation, priority, and rewiring views use deterministic local demo data in `src/data/playground.ts`; those values are illustrative. No baseline metrics are bundled because the repository does not contain a matching five-method benchmark with recall, latency, QPS, and memory measurements.

The baseline view accepts a JSON file with `schema_version: 1`, `run_id`, `dataset`, positive integer `k`, and a `results` array. Each result has a `method` (`static_hnsw`, `native_update`, `delete_reinsert`, `full_rebuild`, or `khnsw`) and optional `recallAtK` (fraction from 0 to 1), `p50LatencyMs`, `qps`, and `memoryMb` values. Upload one comparable run; absent fields remain blank.

The live-metrics view accepts JSONL events validated against the event schema in a Web Worker. It derives mean latency from `search.payload.latency_ms`, recall and connectivity from `metric` events, repair count from `pop` events, modified edges from `edge_add`/`edge_remove` events, QPS when `run_end.payload.elapsed_seconds` is present, and processing time from that same run-end field. Missing metrics are displayed as unavailable. Traces marked with `payload.example: true` are explicitly identified as synthetic.

## Development

```sh
npm install
npm run dev
```

## Checks

```sh
npm run lint
npm test
npm run build
```

The production build runs TypeScript project checks before Vite bundles the app.
