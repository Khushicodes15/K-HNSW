# K-HNSW Benchmarks Suite

Owned by **Member 4 (Benchmarks & Evaluation)**.

## Structure
- `datasets/`: Dataset loaders, synthetic generators, and schema adapters. Raw data files are gitignored.
- `baselines/`: Wrappers for comparison baselines (e.g. static HNSW rebuild, delete-and-reinsert via hnswlib/faiss).
- `experiments/`: Experiment execution runners, harness scripts, and parameter sweep orchestrators.
- `metrics/`: Metric computation implementations (recall@k, latency, QPS, modified edge counters, connectivity audits).
- `results/`: Committed run manifests and small aggregated metrics CSVs/JSONs. Raw vectors or large index dumps MUST NOT be committed.

## Guidelines
- Always record the exact library versions, git commit, seed, and config hash in the run manifest (`run_manifest.schema.json`).
- Ensure experiments can be run deterministically with fixed random seeds.
- Never commit fabricated or placeholder numbers as real results.

