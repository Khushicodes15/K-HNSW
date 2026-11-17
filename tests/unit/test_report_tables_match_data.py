"""Test report tables and inline metrics match benchmark data artifacts exactly.

Enforces:
(a) Regenerates all blocks and inline values from JSON artifacts and asserts they equal
    the marked blocks and inline spans in the chapter files.
(b) Fails if the interpolation section contains a markdown table outside a generated block.
(c) Fails if "on-manifold" appears anywhere in docs/report or benchmarks/.
(d) Fails if the interpolation section lacks "exploratory" or "interpolation drift".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from benchmarks.emit_interpolation_tables import generate_all_inlines, generate_all_tables


def verify_tables_match(chapter_path: Path, self_ex_json: Path, comm_json: Path) -> None:
    """Verify that all generated table blocks in the chapter file match JSON artifacts."""
    tables = generate_all_tables(self_ex_json, comm_json)
    content = chapter_path.read_text(encoding="utf-8")

    for name, table_md in tables.items():
        pattern = (
            rf"<!-- BEGIN GENERATED: {name} -->\r?\n(.*?)(?:<!-- END GENERATED: {name} -->|\Z)"
        )
        match = re.search(pattern, content, flags=re.DOTALL)
        assert match is not None, (
            f"Block marker <!-- BEGIN GENERATED: {name} --> not found in {chapter_path.name}"
        )
        assert match.group(1).strip() == table_md.strip(), (
            f"Block '{name}' in {chapter_path.name} differs from generated data"
        )


def verify_inlines_match(chapter_path: Path, self_ex_json: Path, comm_json: Path) -> None:
    """Verify that all generated inline values in the chapter match JSON artifacts."""
    inlines = generate_all_inlines(self_ex_json, comm_json)
    content = chapter_path.read_text(encoding="utf-8")

    inline_pattern = re.compile(r"<!-- GEN-INLINE:\s*(\w+)\s*-->(.*?)<!-- /GEN-INLINE -->")
    matches = list(inline_pattern.finditer(content))
    assert len(matches) > 0, (
        f"No inline markers <!-- GEN-INLINE: ... --> found in {chapter_path.name}"
    )

    for match in matches:
        key = match.group(1)
        curr_val = match.group(2)
        assert key in inlines, f"Unknown inline key '{key}' found in {chapter_path.name}"
        assert curr_val == inlines[key], (
            f"Inline metric '{key}' in {chapter_path.name} is '{curr_val}', "
            f"expected '{inlines[key]}'"
        )


def verify_no_unmarked_tables_in_section(chapter_path: Path) -> None:
    """Verify that no markdown tables exist in interpolation section outside generated blocks."""
    content = chapter_path.read_text(encoding="utf-8")
    section_match = re.search(
        r"(^##\s+[^#\n]*Interpolation Drift\s*\(Exploratory\).*?)(?=\n##\s+|\Z)",
        content,
        flags=re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    assert section_match is not None, "Interpolation section not found in chapter file"
    section_text = section_match.group(1)

    # Strip all generated blocks
    stripped = re.sub(
        r"<!-- BEGIN GENERATED: \w+ -->.*?<!-- END GENERATED: \w+ -->",
        "",
        section_text,
        flags=re.DOTALL,
    )

    # Check for markdown table separator rows (|---| or |:---|)
    table_separator_pattern = re.compile(r"\|(?:\s*:?-+:?\s*\|)+")
    unmarked_tables = table_separator_pattern.findall(stripped)
    assert len(unmarked_tables) == 0, (
        f"Found {len(unmarked_tables)} unmarked markdown table(s) in interpolation section "
        "outside generated blocks!"
    )


def verify_no_on_manifold(report_dir: Path, benchmarks_dir: Path) -> None:
    """Verify that 'on-manifold' does not appear anywhere in docs/report or benchmarks/."""
    hits: list[str] = []
    # Check docs/report/*.md
    for md_file in report_dir.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if "on-manifold" in line.lower():
                hits.append(f"{md_file.name}:{line_no}: {line.strip()}")

    # Check benchmarks/**/* (.py and .md)
    for ext in ["*.py", "*.md"]:
        for f in benchmarks_dir.rglob(ext):
            text = f.read_text(encoding="utf-8", errors="ignore")
            for line_no, line in enumerate(text.splitlines(), start=1):
                if "on-manifold" in line.lower():
                    hits.append(f"{f.relative_to(benchmarks_dir)}:{line_no}: {line.strip()}")

    assert len(hits) == 0, (
        "Found forbidden phrase 'on-manifold' in docs/report or benchmarks/:\n" + "\n".join(hits)
    )


def verify_required_phrases(chapter_path: Path) -> None:
    """Verify that the interpolation section contains required terms."""
    content = chapter_path.read_text(encoding="utf-8")
    section_match = re.search(
        r"(^##\s+[^#\n]*Interpolation Drift\s*\(Exploratory\).*?)(?=\n##\s+|\Z)",
        content,
        flags=re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    assert section_match is not None, "Interpolation section not found in chapter file"
    section_text = section_match.group(1).lower()

    assert "exploratory" in section_text, "Interpolation section lacks 'exploratory'"
    assert "interpolation drift" in section_text, (
        "Interpolation section lacks 'interpolation drift'"
    )


# --- Test Functions ---


def test_report_tables_match_data() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    chapter_path = repo_root / "docs" / "report" / "chapter9_external_baselines.md"
    self_ex_json = repo_root / "docs" / "data" / "interpolation_self_excluded.json"
    comm_json = repo_root / "docs" / "data" / "interpolation_benchmark.json"

    if not self_ex_json.exists():
        pytest.skip("interpolation_self_excluded.json not yet generated")

    verify_tables_match(chapter_path, self_ex_json, comm_json)


def test_report_inlines_match_data() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    chapter_path = repo_root / "docs" / "report" / "chapter9_external_baselines.md"
    self_ex_json = repo_root / "docs" / "data" / "interpolation_self_excluded.json"
    comm_json = repo_root / "docs" / "data" / "interpolation_benchmark.json"

    if not self_ex_json.exists():
        pytest.skip("interpolation_self_excluded.json not yet generated")

    verify_inlines_match(chapter_path, self_ex_json, comm_json)


def test_no_tables_outside_generated_blocks() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    chapter_path = repo_root / "docs" / "report" / "chapter9_external_baselines.md"
    verify_no_unmarked_tables_in_section(chapter_path)


def test_no_forbidden_on_manifold_in_report_or_benchmarks() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    report_dir = repo_root / "docs" / "report"
    benchmarks_dir = repo_root / "benchmarks"
    verify_no_on_manifold(report_dir, benchmarks_dir)


def test_required_phrases_in_interpolation_section() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    chapter_path = repo_root / "docs" / "report" / "chapter9_external_baselines.md"
    verify_required_phrases(chapter_path)
