# benchmarks/ — Usage Reference

All benchmark runners are Python modules runnable with `-m`. Run from the
**repository root** so that relative paths resolve correctly.

```bash
# Install the package in editable mode first (once per environment):
pip install -e .
```

## 1. Synthetic Gaussian Drift Sweep

```bash
# ef_construction=16, seeds 100-104, n=500 and n=2000 (headline Table 8.1A)
python -m benchmarks.run_synthetic_sweep \
  --n 500 2000 --d 8 \
  --seeds 100 101 102 103 104 \
  --ef-construction 16 --steps 10 \
  --out docs/data/synthetic_sweep_ef16_seeds100.json

# ef_construction=200, seeds 100-104 (headline Table 8.1B)
python -m benchmarks.run_synthetic_sweep \
  --n 500 2000 --d 8 \
  --seeds 100 101 102 103 104 \
  --ef-construction 200 --steps 10 \
  --out docs/data/synthetic_sweep_ef200_seeds100.json
```

## 2. Real Texmex SIFT-128 Evaluation

> Download `siftsmall_base.fvecs` first — see `docs/data/siftsmall/DOWNLOAD.md`.

```bash
# Seeds 100-104, ef_construction=200 (headline Table 9.1)
python -m benchmarks.run_real_embeddings \
  --dataset sift --n 2000 \
  --seeds 100 101 102 103 104 \
  --ef-construction 200 --steps 10 \
  --out docs/data/sift_seeds100_ef200.json

# Seeds 42-46 (secondary / continuity run)
python -m benchmarks.run_real_embeddings \
  --dataset sift --n 2000 \
  --seeds 42 43 44 45 46 \
  --ef-construction 200 --steps 10 \
  --out docs/data/sift_seeds42_ef200.json

# Without the .fvecs file (synthetic fallback for local dev only):
python -m benchmarks.run_real_embeddings \
  --dataset sift --allow-fallback --n 500 --steps 5
```

## 3. hnswlib Baseline (Linux / CI only)

```bash
# Requires native hnswlib (pip install hnswlib on Linux).
# On Windows this will warn and exit without writing JSON.
python -m benchmarks.run_hnswlib_baseline \
  --n 2000 --seeds 42 43 44 45 46
```

## 4. Generate Report Tables from JSON

```bash
# Regenerates Markdown tables for Chapters 8 & 9 from external_benchmarks.json.
python -m benchmarks.generate_report_tables
```

## 5. Quick Smoke Tests

```bash
# Fast self-checks for all three runners (small n, few seeds):
python -m benchmarks.run_synthetic_sweep --smoke
python -m benchmarks.run_real_embeddings --smoke --allow-fallback
python -m benchmarks.run_hnswlib_baseline --smoke
```
