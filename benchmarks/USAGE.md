# Benchmark Runner Usage Reference

This guide documents execution commands for the benchmark runners in `benchmarks/`. Run all commands from the repository root.

> **Important**: Committed JSON files under `docs/data/` must not be overwritten because automated tests and report tables check them directly. All example commands below write to scratch paths (e.g. `out/scratch_*.json`).

## 1. Real Texmex SIFT-128 Evaluation: Isotropic Drift (Displacement 8.0)

Runner: `benchmarks.run_real_embeddings`

Evaluates isotropic drift on real Texmex SIFT-128 vectors with displacement multiplier `displacement_mult = 8.0` and drifting fraction `drift_frac = 0.3` (as recorded in `docs/data/sift_n2000_ef200.json`).

Committed JSON files and seeds used:
- `docs/data/sift_n2000_ef200.json`: evaluated with seeds `110 111 112 113 114` ($n=2000$, `ef_construction=200`, `displacement_mult=8.0`, `drift_frac=0.3`).
- `docs/data/sift_n5000_ef200.json`: evaluated with seeds `110 111 112 113 114` ($n=5000$, `ef_construction=200`, `displacement_mult=8.0`, `drift_frac=0.3`).
- `docs/data/external_benchmarks.json` (`real_sift_evaluation`): evaluated with seeds `110 111 112 113 114`.

Requires `docs/data/siftsmall/siftsmall_base.fvecs` (see `docs/data/siftsmall/DOWNLOAD.md`).

```bash
# Evaluate n=2000 on real SIFT isotropic drift (seeds 110-114, ef_construction=200)
python -m benchmarks.run_real_embeddings \
  --dataset sift \
  --n 2000 \
  --seeds 110 111 112 113 114 \
  --ef-construction 200 \
  --displacement 8.0 \
  --drift-frac 0.3 \
  --out out/scratch_sift_n2000_ef200.json

# Evaluate n=5000 on real SIFT isotropic drift (seeds 110-114, ef_construction=200)
python -m benchmarks.run_real_embeddings \
  --dataset sift \
  --n 5000 \
  --seeds 110 111 112 113 114 \
  --ef-construction 200 \
  --displacement 8.0 \
  --drift-frac 0.3 \
  --out out/scratch_sift_n5000_ef200.json
```

## 2. Interpolation Drift Benchmark (Exploratory)

Runner: `benchmarks.run_interpolation_benchmark`

Committed JSON file and seeds used:
- `docs/data/interpolation_benchmark.json`: evaluated with seeds `110 111 112 113 114` ($n=2000$, `ef_construction=200`). Results report exploratory interpolation drift across 5 conditions (`f_0.25`, `f_0.50`, `f_1.00`, `f_0.50_jitter`, `f_1.00_jitter`).

```bash
# Run exploratory interpolation drift benchmark (n=2000, seeds 110-114, ef_construction=200)
python -m benchmarks.run_interpolation_benchmark \
  --data-path docs/data/siftsmall/siftsmall_base.fvecs \
  --n 2000 \
  --seeds 110 111 112 113 114 \
  --ef-construction 200 \
  --output out/scratch_interpolation_benchmark.json
```

## 3. Self-Excluded Interpolation Drift Benchmark (Exploratory)

Runner: `benchmarks.run_interpolation_self_excluded`

Committed JSON file and seeds used:
- `docs/data/interpolation_self_excluded.json`: evaluated with seeds `110 111 112 113 114`. Re-evaluates interpolation drift under self-exclusion to eliminate static query self-matching.

```bash
# Run self-excluded interpolation drift benchmark
python -m benchmarks.run_interpolation_self_excluded \
  --data-path docs/data/siftsmall/siftsmall_base.fvecs \
  --committed-json docs/data/interpolation_benchmark.json \
  --output-path out/scratch_interpolation_self_excluded.json
```

## 4. Synthetic Drift Sweep

Runner: `benchmarks.run_synthetic_sweep`

Committed results are stored in `docs/data/external_benchmarks.json` under:
- `synthetic_drift_ef16`: evaluated with seeds `42 43 44 45 46` ($n=500$ and $n=2000$).
- `synthetic_drift_ef200`: evaluated with seeds `100 101 102 103 104` ($n=500$ and $n=2000$).

The exact multi-condition assembly script that compiled `docs/data/external_benchmarks.json` is not documented here. Individual sweep runs write to scratch paths:

```bash
# Synthetic drift sweep with efConstruction=16 (seeds 42-46)
python -m benchmarks.run_synthetic_sweep \
  --n 500 2000 \
  --seeds 42 43 44 45 46 \
  --ef-construction 16 \
  --out out/scratch_synthetic_sweep_ef16.json

# Synthetic drift sweep with efConstruction=200 (seeds 100-104)
python -m benchmarks.run_synthetic_sweep \
  --n 500 2000 \
  --seeds 100 101 102 103 104 \
  --ef-construction 200 \
  --out out/scratch_synthetic_sweep_ef200.json
```

## 5. hnswlib Baseline Note

Without native hnswlib installed, the runner prints a warning and writes no JSON output (see `benchmarks/run_hnswlib_baseline.py:359`).
