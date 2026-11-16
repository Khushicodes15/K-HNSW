# Downloading the Texmex SIFT-128 Dataset

The `*.fvecs` files are excluded from git (see `.gitignore: *.fvecs`) because
they are large binary files. You must download them manually before running
the real-SIFT benchmarks.

## Required File

| File | Size | Purpose |
|---|---|---|
| `siftsmall_base.fvecs` | ~3.6 MB | 10,000 SIFT-128 base vectors used by Chapter 9 |

## Download Command

```bash
# From the repo root:
mkdir -p docs/data/siftsmall
curl -L http://corpus-texmex.irisa.fr/sift.tar.gz \
  | tar xz -C docs/data/siftsmall --strip-components=1
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Force docs/data/siftsmall
Invoke-WebRequest http://corpus-texmex.irisa.fr/sift.tar.gz -OutFile sift.tar.gz
tar xz -C docs/data/siftsmall --strip-components=1 -f sift.tar.gz
Remove-Item sift.tar.gz
```

## Verify Checksum

After downloading, verify integrity before running benchmarks:

```bash
# Linux / macOS:
sha256sum docs/data/siftsmall/siftsmall_base.fvecs
# Expected: 1e90414a3254361aba48a0d58e461f7661fb135dabb3da985290e94de8f60fe0

# Windows (PowerShell):
(Get-FileHash docs/data/siftsmall/siftsmall_base.fvecs -Algorithm SHA256).Hash
# Expected: 1E90414A3254361ABA48A0D58E461F7661FB135DABB3DA985290E94DE8F60FE0
```

This checksum is also recorded in `docs/data/external_benchmarks.json`
under `real_sift_evaluation.sift_file_sha256`.

## What Happens Without the File

The benchmark runner (`benchmarks/run_real_embeddings.py`) will raise a
`FileNotFoundError` with download instructions rather than silently falling
back to synthetic data. To explicitly permit synthetic fallback (not for
official results), pass `--allow-fallback`.
