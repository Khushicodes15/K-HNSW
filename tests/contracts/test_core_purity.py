"""Purity enforcement test verifying src/khnsw/core uses only stdlib, numpy, and khnsw.

Enforces Pyodide browser safety with a strict denylist of OS-level or blocking stdlib modules.
"""

from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORE_DIR = REPO_ROOT / "src" / "khnsw" / "core"

DENIED_MODULES = {
    "multiprocessing",
    "subprocess",
    "socket",
    "sqlite3",
    "threading",
    "ctypes",
}

ALLOWED_TOP_LEVELS = (
    set(sys.stdlib_module_names) | {"__future__", "numpy", "khnsw"}
) - DENIED_MODULES


def find_external_imports(py_path: Path) -> list[str]:
    """Parse a python file into an AST and return all top-level modules imported."""
    with open(py_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(py_path))

    imported_top_levels: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_name = alias.name.split(".")[0]
                imported_top_levels.append(top_name)
        elif isinstance(node, ast.ImportFrom):
            # Relative imports (level > 0) are internal to khnsw
            if node.level > 0:
                continue
            if node.module:
                top_name = node.module.split(".")[0]
                imported_top_levels.append(top_name)

    return imported_top_levels


def is_module_forbidden(module_name: str) -> bool:
    """Return True if module is outside allowed set or explicitly in denylist."""
    if module_name in DENIED_MODULES:
        return True
    return module_name not in ALLOWED_TOP_LEVELS


def test_core_purity_ast_scan() -> None:
    """Ensure every Python file under src/khnsw/core imports ONLY allowed modules."""
    assert CORE_DIR.exists(), f"Core directory missing at {CORE_DIR}"
    py_files = list(CORE_DIR.rglob("*.py"))
    assert len(py_files) > 0, "No python files found under src/khnsw/core"

    violations: list[str] = []

    for py_file in py_files:
        imports = find_external_imports(py_file)
        forbidden = [mod for mod in imports if is_module_forbidden(mod)]
        if forbidden:
            rel_path = py_file.relative_to(REPO_ROOT).as_posix()
            violations.append(f"{rel_path}: imports forbidden modules {forbidden}")

    assert not violations, "Core purity violations detected:\n" + "\n".join(violations)


def test_scanner_catches_forbidden_imports() -> None:
    """Verify that the AST scanner catches both 3rd-party libs and denylisted stdlib modules."""
    forbidden_code = """
import numpy as np
from khnsw.core.events.schema import Event
import requests  # Disallowed external library
from scipy.spatial import distance  # Disallowed external library
import multiprocessing  # Denylisted stdlib
import subprocess  # Denylisted stdlib
import socket  # Denylisted stdlib
import sqlite3  # Denylisted stdlib
import threading  # Denylisted stdlib
import ctypes  # Denylisted stdlib
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(forbidden_code)
        tmp_path = Path(tmp.name)

    try:
        imports = find_external_imports(tmp_path)
        forbidden = set(mod for mod in imports if is_module_forbidden(mod))

        # Check 3rd party
        assert "requests" in forbidden
        assert "scipy" in forbidden
        assert "numpy" not in forbidden
        assert "khnsw" not in forbidden

        # Check each denylisted module
        for denied in DENIED_MODULES:
            assert denied in forbidden, f"Expected denylisted module {denied} to be caught"
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
