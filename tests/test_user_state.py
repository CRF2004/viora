"""Tests for user_state.py — corrupt/legacy stored state must not crash or leak.

Redirects DATA_DIR/USER_STATE_FILE to a temp dir so tests never touch the real
data/ directory.
"""

import json

import pytest

import user_state


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(user_state, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(user_state, "USER_STATE_FILE", str(tmp_path / "user_state.json"))
    return tmp_path


def _write_raw(tmp_path, payload):
    with open(tmp_path / "user_state.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


# --- read path -------------------------------------------------------------


def test_get_user_state_repairs_corrupt_fields(isolated_state):
    _write_raw(
        isolated_state,
        {
            "u": {
                "user_id": "u",
                "interaction_count": "abc",
                "avg_reply_length": {"x": 1},
                "sleep_baseline": "bad",
                "energy_baseline": [1, 2],
                "mood_baseline": None,
                "recent_mood_trend": {"weird": True},
                "user_preferences": "not-a-dict",
                "routine_summary": [1, 2, 3],
                "profile_notes": {"nested": 1},
            }
        },
    )
    state = user_state.get_user_state("u")
    assert state["interaction_count"] == 0
    assert state["avg_reply_length"] == 0.0
    assert state["sleep_baseline"] is None
    assert state["energy_baseline"] is None
    assert state["mood_baseline"] is None
    assert state["recent_mood_trend"] is None
    assert state["user_preferences"] == {}
    assert state["routine_summary"] == {}
    assert state["profile_notes"] == ""


def test_get_user_state_preserves_wellformed_values(isolated_state):
    _write_raw(
        isolated_state,
        {
            "u": {
                "interaction_count": 4,
                "avg_reply_length": 12.5,
                "sleep_baseline": 7.1,
                "recent_mood_trend": "up",
                "user_preferences": {"wake_up_hour": 7},
                "profile_notes": "喜欢跑步",
            }
        },
    )
    state = user_state.get_user_state("u")
    assert state["interaction_count"] == 4
    assert state["avg_reply_length"] == 12.5
    assert state["sleep_baseline"] == 7.1
    assert state["recent_mood_trend"] == "up"
    assert state["user_preferences"] == {"wake_up_hour": 7}
    assert state["profile_notes"] == "喜欢跑步"


def test_get_user_state_non_dict_entry_returns_default(isolated_state):
    _write_raw(isolated_state, {"u": ["not", "a", "dict"]})
    state = user_state.get_user_state("u", persona_id="coach")
    assert state["user_id"] == "u"
    assert state["persona_id"] == "coach"
    assert state["interaction_count"] == 0


# --- update path (crashed before sanitization) -----------------------------


def test_update_user_state_tolerates_corrupt_stored_state(isolated_state):
    _write_raw(
        isolated_state,
        {
            "u": {
                "user_id": "u",
                "interaction_count": "abc",
                "avg_reply_length": {"x": 1},
                "user_preferences": "nope",
            }
        },
    )
    # Previously raised ValueError from dict("nope") / int("abc").
    state = user_state.update_user_state(
        "u", message_text="hi", user_preferences={"last_reply_length": 2}
    )
    assert state["interaction_count"] == 1
    assert state["avg_reply_length"] == 2.0
    assert state["user_preferences"] == {"last_reply_length": 2}


def test_update_user_state_ignores_junk_numeric_extraction(isolated_state):
    _write_raw(
        isolated_state,
        {"u": {"user_id": "u", "sleep_baseline": "bad", "energy_baseline": {}}},
    )
    state = user_state.update_user_state(
        "u",
        extracted_data={
            "sleep": {"quality": "oops"},
            "energy": "junk",
            "mood": {"score": 7},
        },
    )
    # Junk measurements are ignored; valid mood still builds a baseline.
    assert state["sleep_baseline"] is None
    assert state["energy_baseline"] is None
    assert state["mood_baseline"] == 7.0
    assert state["recent_mood_trend"] == "stable"


def test_update_user_state_wellformed_ema_and_count(isolated_state):
    user_state.update_user_state("u", message_text="hello", extracted_data={"mood": 4})
    state = user_state.update_user_state(
        "u", message_text="world!!", extracted_data={"mood": 8}
    )
    assert state["interaction_count"] == 2
    assert state["avg_reply_length"] == round((5 + 7) / 2, 2)
    # EMA: 4*0.8 + 8*0.2 = 4.8
    assert state["mood_baseline"] == 4.8
    assert state["recent_mood_trend"] == "up"


def test_update_user_state_sanitizes_counts_from_legacy_strings(isolated_state):
    _write_raw(
        isolated_state,
        {"u": {"interaction_count": "3", "avg_reply_length": "not-a-number"}},
    )
    state = user_state.update_user_state("u", message_text="x")
    # Legacy string count is parsed, junk avg degrades to 0.0 then recomputed.
    assert state["interaction_count"] == 4
    assert state["avg_reply_length"] == 0.25


def test_sanitize_count_edge_cases():
    assert user_state._sanitize_count(-5) == 0
    assert user_state._sanitize_count(4.9) == 4
    assert user_state._sanitize_count(" 6 ") == 6
    assert user_state._sanitize_count("nan") == 0
    assert user_state._sanitize_count(None) == 0
    assert user_state._sanitize_count(float("nan")) == 0
