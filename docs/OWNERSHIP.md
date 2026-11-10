# Team Ownership Matrix

| Member | Name | Role | Primary Owned Directories | Review Partner |
| :--- | :--- | :--- | :--- | :--- |
| **Member 1** | Mansi Bhattacharjee | Research / Algorithm Spec | `docs/algorithm/`, `docs/literature/`, `docs/contracts/metrics.md` | Member 4 |
| **Member 2** | Arpanjot Kaur | Graph Engine | `src/khnsw/core/graph/`, `src/khnsw/core/search/`, `src/khnsw/adapters/` | Member 3 |
| **Member 3** | Khushi Dhona | Update & Repair Engine *(Repo Owner)* | `src/khnsw/core/movement/`, `src/khnsw/core/detection/`, `src/khnsw/core/scheduling/`, `src/khnsw/core/repair/`, `src/khnsw/core/events/`, `docs/contracts/` | Member 2 |
| **Member 4** | Saanvi Srivastava | Benchmarks & Evaluation | `benchmarks/`, `configs/`, `tests/parity/` | Member 1 |
| **Member 5** | Ayushman Acharya | Frontend & Integration | `frontend/`, `src/khnsw/api/`, `Dockerfile.backend`, `docker-compose.yml`, `.github/workflows/` | Member 3 |

## Code Review Pairing Rules
- PRs from Member 2 (Arpanjot Kaur) are reviewed by Member 3 (Khushi Dhona), and vice versa.
- PRs from Member 1 (Mansi Bhattacharjee) are reviewed by Member 4 (Saanvi Srivastava), and vice versa.
- PRs from Member 5 (Ayushman Acharya) are reviewed by Member 3 (Khushi Dhona), and vice versa.
- Cross-cutting changes to `docs/contracts/` or `interfaces.py` require sign-off from the Repo Owner (Khushi Dhona).
