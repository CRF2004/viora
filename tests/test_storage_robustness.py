"""Tests for storage.py corrupt/legacy tolerance (story cache + data hash).

Redirects story-cache paths to a temp dir so tests never touch real data.
"""

import json

import pytest

import storage


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(storage, "STORY_CACHE_FILE", str(tmp_path / "story_cache.json"))
    monkeypatch.setattr(storage, "STORY_CACHE_LOCK_FILE", str(tmp_path / "story_cache.json.lock"))
    return tmp_path


def _write(tmp_path, payload):
    (tmp_path / "story_cache.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def test_get_story_cache_non_dict_top_level(isolated_cache):
    _write(isolated_cache, [1, 2, 3])
    assert storage.get_story_cache("u1", "week") is None


def test_get_story_cache_non_dict_bucket(isolated_cache):
    _write(isolated_cache, {"u1": "junk", "u2": ["junk"]})
    assert storage.get_story_cache("u1", "week") is None
    assert storage.get_story_cache("u2", "week") is None


def test_get_story_cache_non_dict_period_value_dropped(isolated_cache):
    _write(isolated_cache, {"u1": {"week": "junk", "month": {"ok": 1}}})
    assert storage.get_story_cache("u1", "week") is None
    assert storage.get_story_cache("u1", "month") == {"ok": 1}


def test_set_story_cache_repairs_corrupt_bucket(isolated_cache):
    _write(isolated_cache, {"u1": "junk"})
    storage.set_story_cache("u1", "week", {"ok": True})
    assert storage.get_story_cache("u1", "week") == {"ok": True}
    stored = json.loads((isolated_cache / "story_cache.json").read_text("utf-8"))
    assert isinstance(stored["u1"], dict)


def test_set_story_cache_replaces_corrupt_period_value(isolated_cache):
    _write(isolated_cache, {"u1": {"week": "junk"}})
    storage.set_story_cache("u1", "week", {"ok": True})
    assert storage.get_story_cache("u1", "week") == {"ok": True}


def test_compute_data_hash_skips_corrupt_entries():
    messages = [
        {"timestamp": "2026-01-01T00:00:00Z"},
        "junk",
        {},
        {"no_timestamp": True},
        {"timestamp": "2026-01-02T00:00:00Z"},
    ]
    assert storage._compute_data_hash(messages) == "2:2026-01-02T00:00:00Z"


def test_compute_data_hash_wellformed_unchanged():
    messages = [{"timestamp": "2026-01-01T00:00:00Z"},
                {"timestamp": "2026-01-05T00:00:00Z"}]
    assert storage._compute_data_hash(messages) == "2:2026-01-05T00:00:00Z"
    assert storage._compute_data_hash([]) == "empty"
