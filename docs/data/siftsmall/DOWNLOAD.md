# Downloading and Verifying the Texmex SIFT-128 siftsmall Dataset

## File Description
- **Dataset**: Texmex SIFT siftsmall base vectors
- **Dimensions**: 10,000 vectors $\times$ 128 dimensions
- **Format**: float32 `.fvecs` binary format
- **Size**: 5160000 bytes

## Download Source
- **URL as referenced in benchmarks/run_real_embeddings.py; verify it still resolves**:
  `ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz`

Extraction layout from the archive has not been independently verified; ensure the extracted base vector file is placed at `docs/data/siftsmall/siftsmall_base.fvecs`.

## Target File Location
The file must be placed at:
```
docs/data/siftsmall/siftsmall_base.fvecs
```

## Git Tracking
Binary `.fvecs` files are excluded from git tracking via `.gitignore` (rule `.gitignore:60:*.fvecs`) and are not committed to the repository.

## Checksum Verification

Verify the file integrity before running benchmarks:

### Linux / macOS (sha256sum)
```bash
sha256sum docs/data/siftsmall/siftsmall_base.fvecs
# Expected: 1e90414a3254361aba48a0d58e461f7661fb135dabb3da985290e94de8f60fe0
```

### Windows (PowerShell)
```powershell
(Get-FileHash docs/data/siftsmall/siftsmall_base.fvecs -Algorithm SHA256).Hash
# Expected: 1E90414A3254361ABA48A0D58E461F7661FB135DABB3DA985290E94DE8F60FE0
```

This SHA-256 digest is recorded in `docs/data/external_benchmarks.json` under key path `real_sift_evaluation.file_sha256`.

## Behavior When the File Is Missing
If `docs/data/siftsmall/siftsmall_base.fvecs` is not found, the benchmark runner raises a `FileNotFoundError` by default. Passing `--allow-fallback` generates synthetic clustered data labelled `"SIFT-like synthetic"`. Synthetic fallback data must never be reported as real SIFT.
