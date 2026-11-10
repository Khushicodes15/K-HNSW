# K-HNSW: Kinetic Hierarchical Navigable Small World

> Research on repairing HNSW graph topology under continuous vector updates.

## Status
**Research prototype; algorithms are not yet implemented.**
This repository currently provides the monorepo architecture, interface contracts, event trace schemas, Pyodide sync tooling, containerization, and continuous integration workflows.

## What is NOT Claimed
We do **not** claim to have invented dynamic HNSW or dynamic Approximate Nearest Neighbor (ANN) search. This project investigates kinetic data structures, certificate-based validity tracking, and localized budget-constrained repair mechanisms for maintaining navigability and search recall in existing HNSW hierarchies subject to vector drift.

## License
There is currently **no LICENSE file** in this repository. Licensing terms will be decided by the project team at a later phase.

---

## Architectural Principles: Pure Core
The computational core located at [`src/khnsw/core/`](file:///src/khnsw/core/) is strictly **pure**:
- Imports are limited exclusively to Python's standard library, `numpy`, and internal `khnsw` modules.
- This constraint guarantees that the core graph, detection, and repair engine can be compiled and executed directly inside client web browsers using [Pyodide](https://pyodide.org/).
- Purity is validated on every commit by an AST-scanning contract test (`tests/contracts/test_core_purity.py`).
- External, heavy C++ libraries (such as `hnswlib` or `faiss-cpu`) are isolated entirely within [`src/khnsw/adapters/`](file:///src/khnsw/adapters/).

---

## Quick Start

### 1. Python Environment Setup
Requires Python $\ge 3.11$:
```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on Linux/macOS:
source .venv/bin/activate

# Install core in editable mode with development & API extras:
pip install -e ".[dev,api]"

# Optionally install benchmarking dependencies (faiss-cpu, hnswlib, scipy, pandas):
pip install -e ".[bench]"
```

### 2. Run Tests & Linters
```bash
# Check code formatting & linting with Ruff
ruff check .
ruff format --check .

# Run pytest test suite
pytest
```

### 3. Frontend Development
Requires Node.js $\ge 20$:
```bash
cd frontend
npm install
npm test
npm run lint
npm run build
npm run dev
```

### 4. Sync Core to Frontend (Pyodide)
To export the pure Python core package to `frontend/public/py/` for browser execution:
```bash
python scripts/sync_core_to_frontend.py
```
This populates `frontend/public/py/khnsw/` and generates `frontend/public/py/manifest.json`.

### 5. Docker & Docker Compose
Start both the FastAPI backend and the Vite frontend dev server in isolated containers:
```bash
docker compose up --build
```
- Backend API (Health): `http://localhost:8000/health`
- Frontend Dashboard: `http://localhost:5173/`

---

## Repository Map
```
.
├── .github/
│   ├── workflows/ci.yml       # Python (3.11, 3.12) & Frontend CI
│   ├── workflows/pages.yml    # GitHub Pages deployment pipeline
│   ├── CODEOWNERS             # Component ownership boundaries
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── ISSUE_TEMPLATE/        # Task and Bug issue templates
├── AGENTS.md                  # Strict operational rules for AI coding assistants
├── CONTRIBUTING.md            # Git workflows, code reviews, and PR policies
├── Dockerfile.backend         # Python 3.12 backend container
├── docker-compose.yml         # Backend + Frontend compose setup
├── pyproject.toml             # Python packaging configuration
├── benchmarks/                # Benchmark suites, baselines, and results
├── configs/                   # Experiment YAML configurations
├── data/                      # Dataset directory (contents gitignored)
├── docs/
│   ├── contracts/             # Protocols, metrics, and JSON schemas
│   ├── algorithm/             # Formal problem definition & certificate math
│   ├── literature/            # Literature review & paper notes
│   ├── report/                # Research reports and drafts
│   ├── adr/                   # Architecture Decision Records
│   ├── OWNERSHIP.md           # 5-member role matrix & review pairs
│   └── TASKS.md               # Task roadmaps and checklists
├── frontend/                  # React + TypeScript + Vite visualization app
├── scripts/                   # Tooling (sync_core_to_frontend.py)
├── src/khnsw/
│   ├── core/                  # Pure stdlib + NumPy algorithmic core
│   ├── adapters/              # External baseline wrappers (faiss, hnswlib)
│   └── api/                   # FastAPI microservice
└── tests/                     # Unit, property, parity, and contract tests
```

## Team & Ownership
- **Khushi Dhona** (Member 3 / Repo Owner): Update & Repair Engine (`src/khnsw/core/movement/`, `detection/`, `scheduling/`, `repair/`, `events/`)
- **Mansi Bhattacharjee** (Member 1): Research & Algorithm Spec (`docs/algorithm/`, `docs/literature/`, `docs/contracts/metrics.md`)
- **Arpanjot Kaur** (Member 2): Graph Engine & Baselines (`src/khnsw/core/graph/`, `src/khnsw/core/search/`, `src/khnsw/adapters/`)
- **Saanvi Srivastava** (Member 4): Benchmarks & Evaluation (`benchmarks/`, `configs/`, `tests/parity/`)
- **Ayushman Acharya** (Member 5): Frontend & System Integration (`frontend/`, `src/khnsw/api/`, `Dockerfile.backend`, `docker-compose.yml`, `.github/workflows/`)

See [`docs/OWNERSHIP.md`](docs/OWNERSHIP.md) for the full role and review pairing matrix.

## Documentation Links
- [Architecture Decision Record 0001: Monorepo](docs/adr/0001-monorepo.md)
- [Team Ownership & Code Review Pairing Matrix](docs/OWNERSHIP.md)
- [Work Checklists & Task Roadmaps](docs/TASKS.md)
- [System Contracts & Schemas](docs/contracts/README.md)
- [Evaluation Metrics & Protocols](docs/contracts/metrics.md)

