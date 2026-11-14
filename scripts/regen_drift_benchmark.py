"""Driver script to regenerate drift_benchmark.json with 1 process per seed in parallel."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
DATA_DIR = REPO_ROOT / "docs" / "data"


def main():
    seeds = [42, 43, 44]
    processes = []
    out_files = []

    print("Launching parallel regeneration across seeds 42, 43, 44...")
    for seed in seeds:
        out_file = DATA_DIR / f"regen_{seed}.json"
        out_files.append(out_file)
        cmd = [
            str(PYTHON_EXE),
            str(REPO_ROOT / "scripts" / "measure_drift.py"),
            "--seeds",
            str(seed),
            "--output",
            str(out_file),
        ]
        p = subprocess.Popen(cmd, cwd=REPO_ROOT)
        processes.append((seed, p))

    for seed, p in processes:
        p.wait()
        if p.returncode != 0:
            print(f"Error: Process for seed {seed} exited with code {p.returncode}")
            sys.exit(p.returncode)
        print(f"Seed {seed} finished successfully.")

    regen_path = DATA_DIR / "drift_benchmark_regen.json"
    merge_cmd = [
        str(PYTHON_EXE),
        str(REPO_ROOT / "scripts" / "measure_drift.py"),
        "--merge",
        *[str(f) for f in out_files],
        "--output",
        str(regen_path),
    ]
    subprocess.run(merge_cmd, check=True, cwd=REPO_ROOT)
    print(f"Regenerated benchmark merged into {regen_path}")

    # Clean up per-seed temporary files
    for f in out_files:
        if f.exists():
            f.unlink()


if __name__ == "__main__":
    main()
