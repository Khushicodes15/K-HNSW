# K-HNSW: Kinetic Hierarchical Navigable Small World

> Research on maintaining Hierarchical Navigable Small World (HNSW) proximity graph topology under continuous vector drift and dynamic updates.

---

## 1. Project Status & Empirical Findings

K-HNSW investigates mechanisms to preserve search navigability and recall in HNSW indexes subject to continuous vector drift. For the complete inventory of empirical claims, formal verification statuses, and benchmark evidence, see [`docs/claims.md`](docs/claims.md).

### Supported Results
- **Temporal Staggering Advantage**: Unassisted `round_robin` reinsertion scores higher mean drifting recall than bursty `periodic_k5` at matched cumulative reinsertion counts: **+0.0252** at $n=500$ ($0.9275$ vs $0.9023$; $133,945$ vs $125,722$ distance computations) and **+0.0270** at $n=2000$ ($0.9242$ vs $0.8972$; $357,395$ vs $316,203$ distance computations). Furthermore, at $n=2000$, `round_robin` achieves slightly higher recall than step-by-step reinsertion `periodic_k1` ($0.9242$ vs $0.9224$) at **22.7% of its distance computation cost** ($357,395$ vs $1,571,353$, a $77.3\%$ savings). (Source: `docs/data/missing_controls_benchmark.json`).
- **Graph Recall Collapse Under Isotropic Drift**: Without maintenance (`no_repair`), 1-NN recall drops to $\approx 0.65$ under continuous isotropic drift ($0.6562$ at $n=500$ under 30% drift, $0.6497$ at $n=2000$ under 10% drift at $ef=16$).
- **In-Neighbor Pre-Repair Penalty**: Static in-neighbor pre-repair consistently degrades drifting recall (by $-0.0073$ at $n=500$ and $-0.0094$ at $n=2000$) while increasing distance computation costs by $+23.6\%$ to $+24.7\%$ (Source: `docs/data/missing_controls_benchmark.json`).
- **Initial Voronoi Filtering**: Voronoi edge pruning (`settle_hnsw`) removes $32.1\%$ of level-0 edges from freshly constructed indexes (mean level-0 edges drop from $5,131.3$ to $3,483.3$), but kinetic Voronoi edge certificates cost $\approx 4\times$ more distance evaluations than heuristic reinsertion without improving recall.

### Withdrawn Claims
- **Memory-Based Figures**: Early Phase 5g recall/cost numbers typed from unreconciled scratch/memory rather than script output were formally withdrawn (Claim 1 in `docs/claims.md`).
- **Drift Recall Ceiling**: The hypothesis that $\approx 0.90$ was an intrinsic ceiling imposed by drift itself was disproven; controlled full rebuild ceilings achieve $0.9671$ to $0.9938$ (Claim 2).
- **PRNG Collision Artifact**: Early claims that unassisted `no_repair` outperformed repair arms were traced to a PRNG stream collision in `LinearDrift` that induced pure radial drift ($\cos \equiv 1.0$), rather than realistic isotropic drift (Claims 8 and 10).

### Untested Hypotheses & Open Work
- **Microscopic Pre-Repair Transition**: The hypothesis that in-neighbor pre-repair hurts recall because it deletes obsolete edges before replacement bidirectional links are formed remains untested at the individual edge transition level.
- **Real High-Dimensional Embeddings**: Experiments have been conducted exclusively on synthetic Gaussian vectors ($d=8$ and $d=16$); evaluation on real-world text or vision embeddings is pending.
- **External Baselines**: Empirical comparisons have focused on internal controls (`round_robin`, `periodic_k5`, `periodic_k1`, `displacement_theta`); comparison against external dynamic graph libraries (e.g., FreshHNSW) has not yet been conducted.

---

## 2. Setup & Installation

The repository features a pure Python algorithmic core (`src/khnsw/core/`) requiring only Python standard library and NumPy, allowing in-browser execution via Pyodide.

### Python Environment
Requires Python $\ge 3.11$:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install core in editable mode with development & API extras:
pip install -e ".[dev,api]"
```

### Frontend Environment
Requires Node.js $\ge 20$:
```bash
cd frontend
npm install
```

---

## 3. Running Tests & Linters

### Python Linters and Contract Tests
```bash
# Code formatting and linting
ruff check .
ruff format --check .

# Contract purity test (enforces zero non-stdlib/numpy imports in core)
pytest tests/contracts/test_core_purity.py

# Full test suite
pytest
```

### Frontend Verification
```bash
cd frontend
npm run lint
npm test
npm run build
```

---

## 4. Diagnostic & Demonstration Scripts

### Production Policy Replay Events
Instruments `MaintenancePolicy` graph mutations into contracted `edge_remove` (`reason="policy_reinsert"`) and `edge_add` events, enabling deterministic graph state replay from an initial snapshot:
```bash
python scripts/diagnostic_replay_events.py
```

### Alternative Kinetic Edge-Certificate Engine
Demonstrates kinetic Voronoi certificate tracking and repair planning (`KineticDetector`, `CertificateStore`, `khnsw_plan`):
```bash
python scripts/demo_events.py
```
> [!NOTE]
> The kinetic certificate tracking model implemented in `scripts/demo_events.py` is an alternative experimental paradigm whose mathematical mechanics and event stream differ from the production `MaintenancePolicy` reinsertion engine.

---

## 5. Repository Architecture

```
.
├── src/khnsw/
│   ├── core/              # Pure Python stdlib + NumPy algorithmic core
│   │   ├── graph/         # HNSW graph data structure and neighbor heuristic
│   │   ├── movement/      # Linear and Brownian continuous drift models
│   │   ├── detection/     # Kinetic certificate store and event heap
│   │   ├── scheduling/    # MaintenancePolicy, displacement and priority trackers
│   │   ├── repair/        # Topological reinsertion and repair loops
│   │   └── events/        # Contracted trace schemas, event emitters, and sinks
│   ├── adapters/          # External ANN library adapters (faiss, hnswlib)
│   └── api/               # FastAPI backend service
├── tests/
│   ├── contracts/         # Core purity and JSON schema contract tests
│   ├── unit/              # Unit tests for core algorithms and policies
│   ├── integration/       # Multi-component repair and drift integration tests
│   └── parity/            # Ground truth and algorithmic parity tests
├── scripts/               # Diagnostic tooling, replay demonstrations, and Pyodide sync
├── docs/                  # Technical documentation, claims register, and benchmark data
└── frontend/              # React + TypeScript + Vite visualization dashboard
```

---

## 6. Documentation Index

- [`docs/claims.md`](docs/claims.md): Register of all empirical claims, verification statuses, and evidence files.
- [`docs/algorithm/problem.md`](docs/algorithm/problem.md): Formal formulation of continuous vector drift and index maintenance.
- [`docs/algorithm/policies.md`](docs/algorithm/policies.md): Reinsertion scheduling policies, ablation comparisons, and policy replay event contracts.
- [`docs/algorithm/repair.md`](docs/algorithm/repair.md): Topological repair mechanisms, in-neighbor pre-repair, and contract compliance audit.
- [`docs/limitations.md`](docs/limitations.md): Detailed architectural, empirical, and computational limitations.
- [`docs/contracts/`](docs/contracts/README.md): System contracts, event schemas, and evaluation metric specifications.
- [`docs/OWNERSHIP.md`](docs/OWNERSHIP.md): Team role matrix and code review pairings.
