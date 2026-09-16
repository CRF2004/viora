"""Tests for feedback.py — file-backed feedback CRUD with flock.

Redirects DATA_DIR/FEEDBACKS_FILE/LOCK_FILE to a temp dir so tests never
touch the real data/ directory.
"""

import json
import os

import pytest

import feedback


@pytest.fixture
def isolated_feedback(tmp_path, monkeypatch):
    """Point feedback persistence at a temp dir and reload module constants."""
    monkeypatch.setattr(feedback, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(feedback, "FEEDBACKS_FILE", str(tmp_path / "feedbacks.json"))
    monkeypatch.setattr(feedback, "LOCK_FILE", str(tmp_path / "feedbacks.json.lock"))
    return tmp_path


def test_create_and_list_feedback(isolated_feedback):
    fb = feedback.create_feedback("user-1", "bug", "按钮不响应")
    assert fb["user_id"] == "user-1"
    assert fb["type"] == "bug"
    assert fb["completed"] is False
    assert fb["id"]

    listed = feedback.list_feedbacks("user-1")
    assert len(listed) == 1
    assert listed[0]["id"] == fb["id"]
    # Other users see nothing
    assert feedback.list_feedbacks("user-2") == []


def test_list_sorted_by_created_desc(isolated_feedback):
    first = feedback.create_feedback("u", "idea", "a")
    second = feedback.create_feedback("u", "idea", "b")
    listed = feedback.list_feedbacks("u")
    # Newer one first
    assert listed[0]["id"] == second["id"]
    assert listed[1]["id"] == first["id"]


def test_update_owned_feedback(isolated_feedback):
    fb = feedback.create_feedback("u", "bug", "desc")
    updated = feedback.update_feedback(fb["id"], "u", completed=True, type="feature")
    assert updated is not None
    assert updated["completed"] is True
    assert updated["type"] == "feature"
    assert updated["updated_at"] >= fb["updated_at"]


def test_update_non_owner_returns_none(isolated_feedback):
    fb = feedback.create_feedback("u", "bug", "desc")
    assert feedback.update_feedback(fb["id"], "other-user", completed=True) is None
    # Original unchanged
    listed = feedback.list_feedbacks("u")
    assert listed[0]["completed"] is False


def test_delete_owned_feedback(isolated_feedback):
    fb = feedback.create_feedback("u", "bug", "desc")
    assert feedback.delete_feedback(fb["id"], "u") is True
    assert feedback.list_feedbacks("u") == []


def test_delete_non_owner_returns_false(isolated_feedback):
    fb = feedback.create_feedback("u", "bug", "desc")
    assert feedback.delete_feedback(fb["id"], "other-user") is False
    assert len(feedback.list_feedbacks("u")) == 1


def test_corrupt_json_auto_recovers(isolated_feedback):
    data_dir = isolated_feedback
    fb_path = data_dir / "feedbacks.json"
    fb_path.write_text("{corrupt json", encoding="utf-8")

    assert feedback._read_feedbacks_nolock() == []
    # Corrupt file preserved for forensics
    corrupts = list(data_dir.glob("feedbacks.json.corrupt.*"))
    assert len(corrupts) == 1
    assert corrupts[0].read_text(encoding="utf-8") == "{corrupt json"


def test_missing_file_returns_empty(isolated_feedback):
    assert feedback._read_feedbacks_nolock() == []


def test_non_list_json_resets_to_empty(isolated_feedback):
    fb_path = isolated_feedback / "feedbacks.json"
    fb_path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    assert feedback._read_feedbacks_nolock() == []


def _write_raw_feedbacks(isolated_feedback, entries):
    """Write *entries* verbatim to feedbacks.json (bypassing create_feedback)."""
    (isolated_feedback / "feedbacks.json").write_text(
        json.dumps(entries, ensure_ascii=False), encoding="utf-8"
    )


def _valid_fb(fb_id="ok-1", user="u", created_at="2026-01-01T00:00:00+00:00"):
    return {
        "id": fb_id,
        "user_id": user,
        "type": "bug",
        "description": "desc",
        "completed": False,
        "created_at": created_at,
        "updated_at": created_at,
    }


def test_list_skips_corrupt_non_dict_entries(isolated_feedback):
    """Non-dict entries (strings/numbers) in feedbacks.json must not 500 the list."""
    _write_raw_feedbacks(isolated_feedback, [
        _valid_fb("ok-1", "u"),
        "junk-string",      # corrupt → skipped, not .get() crash
        42,                 # corrupt → skipped
        None,               # corrupt → skipped
        _valid_fb("ok-2", "u", created_at="2026-01-02T00:00:00+00:00"),
    ])
    listed = feedback.list_feedbacks("u")
    assert [fb["id"] for fb in listed] == ["ok-2", "ok-1"]
    # Other user unaffected; junk never surfaces
    assert feedback.list_feedbacks("other") == []


def test_list_entry_missing_created_at_does_not_crash(isolated_feedback):
    """A corrupt entry without created_at sorts last instead of KeyError crash."""
    _write_raw_feedbacks(isolated_feedback, [
        _valid_fb("newer", "u", created_at="2026-01-02T00:00:00+00:00"),
        {"id": "older", "user_id": "u", "type": "bug", "description": "no-created"},
    ])
    listed = feedback.list_feedbacks("u")
    assert [fb["id"] for fb in listed] == ["newer", "older"]


def test_update_skips_corrupt_entries(isolated_feedback):
    """update_feedback skips non-dict entries and still updates the real one."""
    _write_raw_feedbacks(isolated_feedback, [
        _valid_fb("ok-1", "u"),
        "junk-string",
    ])
    updated = feedback.update_feedback("ok-1", "u", completed=True)
    assert updated is not None
    assert updated["completed"] is True
    # Corrupt entry preserved on disk, list still safe
    listed = feedback.list_feedbacks("u")
    assert len(listed) == 1
    assert listed[0]["completed"] is True


def test_delete_skips_corrupt_entries(isolated_feedback):
    """delete_feedback matches only dict entries; junk is left untouched."""
    _write_raw_feedbacks(isolated_feedback, [
        _valid_fb("ok-1", "u"),
        "junk-string",
    ])
    assert feedback.delete_feedback("ok-1", "u") is True
    listed = feedback.list_feedbacks("u")
    assert listed == []
