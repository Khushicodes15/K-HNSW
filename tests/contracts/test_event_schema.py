"""Validation tests for event and run manifest schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from khnsw.core.events.schema import Event, EventType

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVENT_SCHEMA_PATH = REPO_ROOT / "docs" / "contracts" / "event.schema.json"
MANIFEST_SCHEMA_PATH = REPO_ROOT / "docs" / "contracts" / "run_manifest.schema.json"
SAMPLE_RUN_PATH = REPO_ROOT / "docs" / "contracts" / "examples" / "sample_run.jsonl"


@pytest.fixture
def event_schema() -> dict[str, Any]:
    with open(EVENT_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def manifest_schema() -> dict[str, Any]:
    with open(MANIFEST_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_sample_run_jsonl_validates_against_schema(event_schema: dict[str, Any]) -> None:
    """Ensure every line in sample_run.jsonl validates against event.schema.json."""
    assert SAMPLE_RUN_PATH.exists(), f"Sample run file missing at {SAMPLE_RUN_PATH}"

    lines_checked = 0
    with open(SAMPLE_RUN_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            event_obj = json.loads(line_str)
            jsonschema.validate(instance=event_obj, schema=event_schema)
            assert event_obj.get("payload", {}).get("example") is True, (
                f"Line {line_num} payload must contain {{'example': true}}"
            )
            lines_checked += 1

    assert lines_checked >= 6, f"Expected at least 6 sample events, found {lines_checked}"


def test_event_dataclass_to_dict_validates_against_schema(event_schema: dict[str, Any]) -> None:
    """Ensure Event dataclass .to_dict() outputs validate against event.schema.json."""
    events = [
        Event(
            run_id="test-run-1",
            seq=0,
            t=0.0,
            type=EventType.run_start,
            payload={"dim": 64},
        ),
        Event(
            run_id="test-run-1",
            seq=1,
            t=0.05,
            type=EventType.move,
            node=10,
            level=1,
            version=2,
            payload={"delta": [0.1, -0.2]},
        ),
        Event(
            run_id="test-run-1",
            seq=2,
            t=0.1,
            type=EventType.edge_add,
            node=10,
            level=0,
            version=2,
            payload={"target": 15},
        ),
        Event(
            run_id="test-run-1",
            seq=3,
            t=1.0,
            type=EventType.run_end,
            payload={"success": True},
        ),
    ]

    for ev in events:
        data = ev.to_dict()
        jsonschema.validate(instance=data, schema=event_schema)


def test_minimal_run_manifest_validates_against_schema(manifest_schema: dict[str, Any]) -> None:
    """Ensure a minimal valid run manifest validates against run_manifest.schema.json."""
    sample_manifest = {
        "run_id": "exp-20261110-001",
        "created_utc": "2026-11-10T12:00:00Z",
        "git_commit": "abcdef0123456789abcdef0123456789abcdef01",
        "config_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "seed": 42,
        "dataset": "sift-128-euclidean",
        "library_versions": {
            "numpy": "1.26.4",
            "khnsw": "0.1.0",
        },
        "metrics_files": [
            "benchmarks/results/exp-20261110-001/metrics.csv",
        ],
        "notes": "Initial benchmark test on synthetic vector drift.",
    }

    jsonschema.validate(instance=sample_manifest, schema=manifest_schema)
