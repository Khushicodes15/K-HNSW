# Operational Guidelines for AI Coding Agents

Every AI assistant or autonomous agent working in this repository MUST strictly follow these rules:

1. **Read Contracts First**: Before drafting or modifying code, always read [`docs/contracts/README.md`](file:///docs/contracts/README.md), [`src/khnsw/core/interfaces.py`](file:///src/khnsw/core/interfaces.py), and relevant schemas.
2. **Core Purity Rule**: All code under `src/khnsw/core/` MUST be pure Python standard library and NumPy only. Never introduce external dependencies (e.g. `scipy`, `sklearn`, `torch`, `faiss`, `requests`, `pydantic`) into `src/khnsw/core/`. This code must run in the browser via Pyodide. Run `pytest tests/contracts/test_core_purity.py` to verify.
3. **One Small Task at a Time**: Work in focused, incremental iterations. Do not attempt multi-module refactorings in a single turn.
4. **Interface and Tests Before Implementation**: Write type-annotated interfaces and automated tests (`tests/unit/`, `tests/property/`) before writing algorithmic code.
5. **No Guessed APIs**: Never hallucinate library function arguments or signatures. Verify real installed package signatures (or documentation) before writing integration logic.
6. **No Fabricated Data or Benchmarks**: Never generate fake benchmark numbers, mocked QPS/recall statistics presented as empirical facts, synthetic screenshots, or fabricated paper citations.
7. **Protect Contracts**: Never alter `docs/contracts/` or `src/khnsw/core/interfaces.py` without explicit repo owner (Member 3) approval.
8. **No Skipping Tests**: Never mark failing tests as `xfail` or `@pytest.mark.skip` to achieve a passing status. If an issue arises, diagnose and resolve it or report the failure honestly.
9. **Keep Pull Requests Small**: Confine PRs to single discrete tasks from [`docs/TASKS.md`](file:///docs/TASKS.md).
10. **Verify All Checks**: Before declaring a task complete, verify that `ruff check .`, `ruff format --check .`, `pytest`, `npm run lint`, `npm test`, and `npm run build` pass cleanly.

