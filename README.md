# K-HNSW: Kinetic Hierarchical Navigable Small World

> Research on maintaining Hierarchical Navigable Small World (HNSW) proximity graph topology under continuous vector drift and dynamic updates.

---

## 1. Project Status & Empirical Findings

K-HNSW investigates mechanisms to preserve search navigability and recall in HNSW indexes subject to continuous vector drift. The table below summarizes the formal status of all 16 claims registered in [`docs/claims.md`](docs/claims.md):

### Empirical Claims Summary (`docs/claims.md`)

| Status | Count | Claims |
| :--- | :---: | :--- |
| `supported` | 5 | Claims 11, 12, 13, 14, 15 |
| `unsupported` | 3 | Claims 2, 3, 7 |
| `withdrawn` | 6 | Claims 1, 4, 5, 6, 8, 9 |
| `radial-protocol-only` | 1 | Claim 10 |
| `untested hypothesis` | 1 | Claim 16 |

#### Supported Claims (5)
- **Claim 11 (`supported`)**: `settle_hnsw` removes 32.1% of level-0 edges (`docs/data/vanilla_headroom.json`). Running `settle_hnsw` on fresh `HNSWIndex` drops mean level-0 edges from $5,131.3$ to $3,483.3$ ($-32.12\%$) and mean out-degree from $10.26$ to $6.97$.
- **Claim 12 (`supported`)**: `no_repair` graph recall collapses under isotropic continuous drift (`docs/data/missing_controls_benchmark.json`, `docs/data/controls_benchmark.json`). At 8x displacement, unassisted drift recall collapses from $>0.98$ to $0.6562$ ($n=500$, 30% drift) and $0.6497$ ($n=2000$, 10% drift) at $ef=16$.
- **Claim 13 (`supported`)**: Temporal staggering (`round_robin`) outperforms bursty periodic repair (`docs/data/missing_controls_benchmark.json`). At matched cumulative reinsertion counts, unassisted `round_robin` beats unassisted `periodic_k5` by $+0.0252$ in $n=500$ ($0.9275$ vs $0.9023$; $133,945$ vs $125,722$ distance computations) and $+0.0270$ in $n=2000$ ($0.9242$ vs $0.8972$; $357,395$ vs $316,203$ distance computations). Against continuous step-by-step reinsertion (`periodic_k1`): at $n=500$, `periodic_k1` achieves higher recall ($0.9338$ vs $0.9275$, $+0.0063$) at $4.6\times$ the distance cost ($621,384$ vs $133,945$); at $n=2000$, `round_robin` achieves slightly higher recall ($0.9242$ vs $0.9224$) at $22.7\%$ of the cost of `periodic_k1` ($357,395$ vs $1,571,353$, a $77.3\%$ savings).
- **Claim 14 (`supported`)**: In-neighbor pre-repair degrades recall and inflates maintenance cost (`docs/data/missing_controls_benchmark.json`). Pre-repair consistently lowers recall by $0.0073$ in $n=500$ ($0.9202$ vs $0.9275$) and by $0.0094$ in $n=2000$ ($0.9148$ vs $0.9242$), while inflating distance costs by $+23.65\%$ in $n=500$ ($165,622.3$ vs $133,945.0$) and $+23.51\%$ in $n=2000$ ($441,428.3$ vs $357,395.0$).
- **Claim 15 (`supported`)**: Kinetic Voronoi edge certificates cost $\approx 4\times$ more for inferior recall (`docs/data/harness_conflict_benchmark.json`, `docs/data/lazy_reinsertion_benchmark.json`). In unified 5m benchmarks, `cert_triggered` achieves inferior recall ($0.9089$) at over $4.4\times$ higher cost ($1,598,047$ vs $357,629$) compared to `displacement_theta_1.0` ($0.9193$) at $n=2000$ (seed 42), and consumes $4.5\times$ more distance computations ($615,680$ vs $135,313$) for only $+0.0056$ recall at $n=500$ (seeds 42–44).

#### Unsupported Claims (3)
- **Claim 2 (`unsupported`)**: "The ~0.90 ceiling is a property of drift itself" (invalidated in Phase 5l; full rebuild ceilings in isotropic settings reach $0.9671 - 0.9938$; `docs/data/reconcile_benchmark.json`, `docs/data/harness_conflict_benchmark.json`).
- **Claim 3 (`unsupported`)**: "In-degree recovered to the settled level" (asserted without edge-level degree distribution verification; `docs/data/drift_benchmark.json`).
- **Claim 7 (`unsupported`)**: `fresh_insert_ref` is a valid reinsertion reference (invalid baseline: rebuilt the entire 500-node graph from scratch with unmetered static setup, structurally incomparable to single-node reinsertion; `docs/data/reconcile_benchmark.json`, `docs/data/vanilla_headroom.json`).

#### Withdrawn Claims (6)
- **Claim 1 (`withdrawn`)**: "5g static recall is 0.9750/0.9700/0.9730 with table costs 43k/91k/155k and violations 69/144/221" (`docs/data/drift_benchmark.json`; preliminary numbers typed from memory/unreconciled scratch; verified run gave mean static recall $0.9840$, total costs $201,997 / 126,827 / 85,713$, and violations $419 / 460 / 477$).
- **Claim 4 (`withdrawn`)**: "Identical routing paths" explanation of static recall resilience (speculative conjecture; search paths were not traced or instrumented).
- **Claim 5 (`withdrawn` - Wrong)**: "Miss classification class (d) = 0" (factually incorrect; committed JSON records miss counts of $d=41$ and $d=56$; `docs/data/drift_benchmark.json`).
- **Claim 6 (`withdrawn` - Deleted)**: Stale-edge audit metric (mathematically ill-defined for continuous velocity trajectories; fully deleted).
- **Claim 8 (`withdrawn`)**: "RNG confound" explains the 5j vs 5k recall conflict (`docs/data/harness_conflict_benchmark.json`, `docs/data/controls_benchmark.json`; retracted; root cause was a structural PRNG stream collision in `LinearDrift` generating pure radial drift under `default_rng`).
- **Claim 9 (`withdrawn`)**: Phase 5m "WIN" labels for lazy reinsertion over periodic repair (`docs/data/controls_benchmark.json`; retracted; margins $+0.0080, +0.0093, +0.0038$ over `round_robin` failed the pre-registered $\ge 0.01$ threshold).

#### Radial-Protocol-Only Claims (1)
- **Claim 10 (`radial-protocol-only`)**: All Phase 5g–5k claims comparing repair arms against `no_repair` (`docs/data/drift_benchmark.json`, `docs/data/reconcile_benchmark.json`, `docs/data/vanilla_arms_benchmark.json`; held strictly under radial drift $\cos \equiv 1.0$; under realistic isotropic drift, un-repaired recall decays to $0.65$).

#### Untested Hypothesis (1)
- **Claim 16 (`untested hypothesis`)**: "Pre-repair hurts because it deletes obsolete edges prematurely before reinsertion connects replacing bi-directional links" (empirical penalty confirmed by Claim 14, but microscopic topological transition mechanism has not been isolated experimentally).

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

### Policy Replay Export (Snapshots & Event Stream)
Exports initial graph snapshot (`initial_snapshot.json`), line-delimited trace events (`events.jsonl`), and final graph snapshot (`final_snapshot.json`) for downstream visualization:
```bash
python scripts/export_policy_replay.py --policy round_robin --steps 5 --n 200 --seed 42 --out-dir replay_output
```
```text
Exported replay_output\initial_snapshot.json
Exported replay_output\final_snapshot.json
Exported replay_output\events.jsonl (921 events)

Policy Replay Trace Event Summary:
-----------------------------------
Event Type           |      Count
-----------------------------------
edge_add             |        170
edge_remove          |        549
move                 |        200
run_end              |          1
run_start            |          1
-----------------------------------
Total Events         |        921
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
