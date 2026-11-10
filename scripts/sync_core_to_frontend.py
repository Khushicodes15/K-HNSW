#!/usr/bin/env python3
"""Sync pure K-HNSW core package into frontend public directory for Pyodide execution.

Copies `src/khnsw/__init__.py` and all files under `src/khnsw/core/**/*.py` into
`frontend/public/py/khnsw/`, then generates `frontend/public/py/manifest.json`.
"""

import json
import shutil
from pathlib import Path


def sync_core() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src_khnsw = repo_root / "src" / "khnsw"
    dest_dir = repo_root / "frontend" / "public" / "py" / "khnsw"

    # Reset destination directory
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    manifest_files: list[str] = []

    # Copy root __init__.py
    root_init = src_khnsw / "__init__.py"
    if root_init.exists():
        shutil.copy2(root_init, dest_dir / "__init__.py")
        manifest_files.append("khnsw/__init__.py")

    # Copy pure core directory
    core_src = src_khnsw / "core"
    core_dest = dest_dir / "core"

    for py_file in core_src.rglob("*.py"):
        rel_to_core = py_file.relative_to(core_src)
        target_path = core_dest / rel_to_core
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(py_file, target_path)

        manifest_rel = f"khnsw/core/{rel_to_core.as_posix()}"
        manifest_files.append(manifest_rel)

    manifest_files.sort()
    manifest_path = repo_root / "frontend" / "public" / "py" / "manifest.json"
    manifest_data = {
        "version": "0.1.0",
        "package": "khnsw.core",
        "files": manifest_files,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Synced {len(manifest_files)} Python core files to {dest_dir.as_posix()}")
    print(f"Generated manifest at {manifest_path.as_posix()}")


if __name__ == "__main__":
    sync_core()
