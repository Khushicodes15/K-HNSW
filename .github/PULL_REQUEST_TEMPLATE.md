## Description
<!-- Provide a brief description of the changes made in this pull request. -->

## Author & Reviewer
- **Author**: Member [1|2|3|4|5]
- **Reviewer**: Member [1|2|3|4|5] (Pairs: 2 <-> 3, 1 <-> 4, 5 <-> 3)

## Pre-Merge Quality Checklist
- [ ] **Tests Added / Updated**: Unit or property tests cover the changes.
- [ ] **Contracts Untouched or Approved**: No modifications to `docs/contracts/` or `interfaces.py` without explicit repo owner (Member 3) approval.
- [ ] **Core Purity Maintained**: Any code under `src/khnsw/core/` imports ONLY standard library, `numpy`, and internal `khnsw` modules (must run in Pyodide).
- [ ] **No Fabricated Numbers**: No fake benchmark numbers, mocked results presented as real data, or fabricated citations.
- [ ] **Lint and Formatting**: Passed `ruff check .` and `ruff format --check .`.
- [ ] **Pytest Passed**: All local tests pass cleanly (`pytest`).
- [ ] **Frontend Quality**: If frontend was touched, `npm run lint`, `npm test`, and `npm run build` pass.
- [ ] **No Secrets / Large Files**: No API keys, credentials, large binaries, or datasets committed.

