"""Purity enforcement test verifying src/khnsw/core uses only stdlib, numpy, and khnsw."""

from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORE_DIR = REPO_ROOT / "src" / "khnsw" / "core"

ALLOWED_TOP_LEVELS = set(sys.stdlib_module_names) | {
    "__future__",
    "numpy",
    "khnsw",
}


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


def test_core_purity_ast_scan() -> None:
    """Ensure every Python file under src/khnsw/core imports ONLY allowed modules."""
    assert CORE_DIR.exists(), f"Core directory missing at {CORE_DIR}"
    py_files = list(CORE_DIR.rglob("*.py"))
    assert len(py_files) > 0, "No python files found under src/khnsw/core"

    violations: list[str] = []

    for py_file in py_files:
        imports = find_external_imports(py_file)
        forbidden = [mod for mod in imports if mod not in ALLOWED_TOP_LEVELS]
        if forbidden:
            rel_path = py_file.relative_to(REPO_ROOT).as_posix()
            violations.append(f"{rel_path}: imports forbidden modules {forbidden}")

    assert not violations, "Core purity violations detected:\n" + "\n".join(violations)


def test_scanner_catches_forbidden_imports() -> None:
    """Verify that the AST import scanner properly identifies disallowed dependencies."""
    forbidden_code = """
import numpy as np
from khnsw.core.events.schema import Event
import requests  # Disallowed external library
from scipy.spatial import distance  # Disallowed external library
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(forbidden_code)
        tmp_path = Path(tmp.name)

    try:
        imports = find_external_imports(tmp_path)
        forbidden = [mod for mod in imports if mod not in ALLOWED_TOP_LEVELS]
        assert "requests" in forbidden
        assert "scipy" in forbidden
        assert "numpy" not in forbidden
        assert "khnsw" not in forbidden
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
