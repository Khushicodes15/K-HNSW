"""Unit tests for scripts/export_policy_replay.py.

Verifies:
(a) events.jsonl lines each validate against event.schema.json, with seq strictly increasing.
(b) Applying edge events to initial snapshot adjacency reproduces final snapshot adjacency.
(c) Running the script twice with the same seed produces byte-identical events.jsonl files.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from scripts.export_policy_replay import run_export

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_SCHEMA_PATH = REPO_ROOT / "docs" / "contracts" / "event.schema.json"


@pytest.fixture(scope="module")
def base_schema() -> dict:
    with open(BASE_SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_export_replay_contracts_and_determinism(tmp_path: Path, base_schema: dict) -> None:
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"

    config = {
        "policy_name": "round_robin",
        "steps": 4,
        "n": 40,
        "d": 4,
        "m": 4,
        "seed": 42,
        "drift_frac": 0.25,
        "displacement_mult": 5.0,
        "t_final": 0.20,
        "B": 3,
    }

    # Run 1
    res1 = run_export(out_dir=dir1, **config)
    events_file1 = res1["events"]
    assert events_file1.exists()

    # (a) Validate each event against schema and check seq strictly increasing
    events: list[dict] = []
    with open(events_file1, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            ev = json.loads(line)
            jsonschema.validate(instance=ev, schema=base_schema)
            assert ev["seq"] == idx, f"Sequence expected {idx}, got {ev['seq']}"
            events.append(ev)

    assert len(events) >= 10
    assert events[0]["type"] == "run_start"
    assert events[-1]["type"] == "run_end"

    # (b) Apply edge mutation events to initial snapshot adjacency and assert equals final
    init_snap = json.loads(Path(res1["initial_snapshot"]).read_text(encoding="utf-8"))
    final_snap = json.loads(Path(res1["final_snapshot"]).read_text(encoding="utf-8"))

    adj: dict[tuple[int, int], set[int]] = {}
    for u_str, lvls in init_snap["adjacency"].items():
        u = int(u_str)
        for lvl_str, nbrs in lvls.items():
            adj[(u, int(lvl_str))] = set(nbrs)

    for ev in events:
        etype = ev["type"]
        if etype == "edge_remove":
            u = ev["node"]
            lvl = ev["level"]
            target = ev["payload"]["target_id"]
            assert (u, lvl) in adj, f"Node ({u}, {lvl}) not found in adjacency"
            assert target in adj[(u, lvl)], f"Edge ({u}, {lvl}) -> {target} to remove missing"
            adj[(u, lvl)].remove(target)
        elif etype == "edge_add":
            u = ev["node"]
            lvl = ev["level"]
            target = ev["payload"]["target_id"]
            assert (u, lvl) in adj, f"Node ({u}, {lvl}) not found in adjacency"
            assert target not in adj[(u, lvl)], (
                f"Edge ({u}, {lvl}) -> {target} to add already exists"
            )
            adj[(u, lvl)].add(target)

    final_adj: dict[tuple[int, int], set[int]] = {}
    for u_str, lvls in final_snap["adjacency"].items():
        u = int(u_str)
        for lvl_str, nbrs in lvls.items():
            final_adj[(u, int(lvl_str))] = set(nbrs)

    assert adj == final_adj, "Replayed adjacency from events does not match final snapshot"

    # (c) Determinism: run 2 must produce byte-identical events.jsonl
    res2 = run_export(out_dir=dir2, **config)
    events_file2 = res2["events"]
    assert events_file1.read_bytes() == events_file2.read_bytes(), "Non-deterministic output"
