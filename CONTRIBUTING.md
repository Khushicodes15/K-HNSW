# Contributing to K-HNSW

Welcome to the K-HNSW research project. Follow these guidelines to keep our codebase clean, reproducible, and contract-compliant.

## Branch Naming
All branches must adhere to the naming convention:
```
member<N>/<topic>
```
Examples:
- `member1/certificate-formulation`
- `member2/numpy-hnsw-index`
- `member3/event-heap-scheduling`
- `member4/groundtruth-harness`
- `member5/pyodide-worker`

## Pull Request Policy
1. **Target**: Pull requests target `main` only.
2. **Review Requirement**: At least 1 approving review from your designated review partner is required prior to merge.
   - **Review Pairings**:
     - Member 2 $\longleftrightarrow$ Member 3
     - Member 1 $\longleftrightarrow$ Member 4
     - Member 5 $\longleftrightarrow$ Member 3
3. **Green CI**: All GitHub Actions CI checks (`ci.yml`: Python 3.11, Python 3.12, Frontend build & test) must pass.
4. **Contract Changes**: Any modifications to `docs/contracts/` or `src/khnsw/core/interfaces.py` require explicit sign-off from the Repo Owner (Member 3).

## Quality Gates Before Pushing
Run the following checks locally:
```bash
# Python
ruff check .
ruff format --check .
pytest

# Frontend
cd frontend
npm run lint
npm test
npm run build
```

## Data and Benchmark Results Policy
- **No Large Files**: Never commit raw vector datasets, `.npz`, `.bin`, `.hdf5`, `.h5`, or serialized index binaries (`.index`).
- **Results Storage**: Commit only compact run manifests and summary metrics CSV files under `benchmarks/results/<run_id>/`:
  - `run_manifest.json` conforming to `docs/contracts/run_manifest.schema.json`.
  - `metrics.csv` containing per-iteration or summary metrics.
- Keep total PR diff sizes small.
