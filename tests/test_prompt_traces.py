"""Tests for prompt_traces.py — corrupt/legacy state tolerance.

Redirects DATA_DIR/TRACES_FILE to a temp dir so tests never touch real data.
"""

import json

import pytest

import prompt_traces
from prompt_tree import PromptTrace


@pytest.fixture
def isolated_traces(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_traces, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(prompt_traces, "TRACES_FILE", str(tmp_path / "prompt_traces.json"))
    return tmp_path


def _write(tmp_path, payload):
    (tmp_path / "prompt_traces.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def _trace(mid: str, ts: str = "2026-01-01T00:00:00Z") -> PromptTrace:
    return PromptTrace(
        message_id=mid,
        timestamp=ts,
        route=["chat/system/base"],
        variables={},
    )


def test_non_dict_top_level_degrades_to_empty(isolated_traces):
    _write(isolated_traces, ["not", "a", "dict"])
    assert prompt_traces.get_trace_count() == 0
    assert prompt_traces.get_recent_traces() == []
    assert prompt_traces.get_trace("anything") is None


def test_non_dict_entries_dropped_but_valid_kept(isolated_traces):
    _write(isolated_traces, {
        "good": {"message_id": "good", "timestamp": "2026-01-01T00:00:00Z"},
        "junk-str": "junk",
        "junk-num": 5,
        "junk-null": None,
    })
    assert prompt_traces.get_trace_count() == 1
    assert prompt_traces.get_trace("good") is not None
    assert prompt_traces.get_trace("junk-str") is None
    recent = prompt_traces.get_recent_traces()
    assert [t["message_id"] for t in recent] == ["good"]


def test_save_trace_recovers_from_corrupt_top_level(isolated_traces):
    _write(isolated_traces, "totally not json-shaped")
    prompt_traces.save_trace(_trace("m1"))
    stored = json.loads((isolated_traces / "prompt_traces.json").read_text("utf-8"))
    assert isinstance(stored, dict)
    assert prompt_traces.get_trace("m1")["message_id"] == "m1"


def test_save_trace_eviction_tolerates_corrupt_entries(isolated_traces, monkeypatch):
    monkeypatch.setattr(prompt_traces, "MAX_TRACES", 2)
    _write(isolated_traces, {
        "bad": "junk",
        "a": {"message_id": "a", "timestamp": "2026-01-01T00:00:00Z"},
        "b": {"message_id": "b", "timestamp": "2026-01-02T00:00:00Z"},
    })
    # Must not raise while sorting/evicting over the corrupt entry.
    prompt_traces.save_trace(_trace("c", "2026-01-03T00:00:00Z"))
    stored = json.loads((isolated_traces / "prompt_traces.json").read_text("utf-8"))
    assert "bad" not in stored
    assert "c" in stored and len(stored) <= 2
