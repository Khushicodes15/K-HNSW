# ADR 0001: Monorepo Architecture for K-HNSW

## Status
Accepted

## Context
The K-HNSW research project spans mathematical algorithm specification, pure Python core implementation, high-throughput external benchmarking (C++/Python wrappers like hnswlib and FAISS), API service wrappers, and an interactive React + TypeScript visualization dashboard with Pyodide client-side execution.

Managing these components across disparate repositories would introduce synchronization overhead, fragmented versioning across schema changes, and high coordination costs for a 5-person team.

## Decision
Adopt a single unified monorepo containing:
1. Pure Python core (`src/khnsw/core/`) constrained to stdlib and NumPy for browser compatibility.
2. Adapter shims (`src/khnsw/adapters/`) isolating heavy C++ dependencies.
3. React + TypeScript frontend (`frontend/`) consuming shared JSON schema contracts.
4. Centralized contracts and documentation (`docs/contracts/`).
5. Unified CI workflows validating both Python and TypeScript components in every pull request.

## Consequences
### Positive
- Single source of truth for interfaces and event schemas.
- Atomic commits spanning core algorithm changes, contract updates, and frontend replay tooling.
- Streamlined CI and GitHub Pages deployment.
- Simplified onboarding and code review pairings.

### Negative / Trade-offs
- Root directory contains multiple toolchains (Python setuptools/pytest/ruff and Node.js npm/vite).
- Strict discipline required to prevent leaking heavy Python dependencies into the pure core package.

