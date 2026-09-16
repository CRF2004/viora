"""
test_scheduler.py — Scheduler tests (proactive interaction scheduling).

Phase 3-B: Psycho-aware scheduling: should_send_proactive() considers
user's psychological state when deciding whether to send a proactive message.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from scheduler import (
    SchedulerState,
    should_send_proactive,
    _adjust_for_psycho_state,
    PSYCHO_PROACTIVE_ADJUSTMENT,
    CST,
    DAILY_PROACTIVE_LIMIT,
)


# =========================================================================
# _adjust_for_psycho_state
# =========================================================================

class TestPsychoAdjustment:
    """Verify the psycho state adjustment logic in isolation."""

    def test_none_psycho_state_passthrough(self):
        """No psycho state -> base decision unchanged."""
        base = {"should_send": True, "reason": "ok"}
        assert _adjust_for_psycho_state(base, None) == base

    def test_normal_psycho_state_passthrough(self):
        """Normal state -> base decision unchanged."""
        base = {"should_send": True, "reason": "ok"}
        assert _adjust_for_psycho_state(base, {"intervenable_state": "normal"}) == base

    def test_unknown_state_passthrough(self):
        """Unknown state -> base decision unchanged."""
        base = {"should_send": True, "reason": "ok"}
        assert _adjust_for_psycho_state(base, {"intervenable_state": "unknown_xyz"}) == base

    def test_willing_but_stuck_overrides_no(self):
        """willing_but_stuck: override soft 'no' to 'yes'."""
        base = {"should_send": False, "reason": "quiet_hours"}
        adjusted = _adjust_for_psycho_state(base, {"intervenable_state": "willing_but_stuck"})
        assert adjusted["should_send"] is True
        assert "psycho_" in adjusted["reason"]

    def test_low_mood_available_overrides_no(self):
        """low_mood_available: override soft 'no' to 'yes'."""
        base = {"should_send": False, "reason": "cooldown"}
        adjusted = _adjust_for_psycho_state(base, {"intervenable_state": "low_mood_available"})
        assert adjusted["should_send"] is True

    def test_avoidance_after_break_overrides_yes(self):
        """avoidance_after_break: override 'yes' to 'no'."""
        base = {"should_send": True, "reason": "ok"}
        adjusted = _adjust_for_psycho_state(base, {"intervenable_state": "avoidance_after_break"})
        assert adjusted["should_send"] is False
        assert "psycho_" in adjusted["reason"]

    def test_hard_limits_not_overridden(self):
        """Hard limits (silent_mode, daily_limit, inactive, not_responding) never overridden."""
        hard_reasons = ["silent_mode", "daily_limit_reached", "user_inactive", "user_not_responding"]
        for reason in hard_reasons:
            base = {"should_send": False, "reason": reason}
            adjusted = _adjust_for_psycho_state(base, {"intervenable_state": "willing_but_stuck"})
            assert adjusted["should_send"] is False
            assert adjusted["reason"] == reason

    def test_all_states_have_adjustments(self):
        """Every intervenable state has an adjustment entry."""
        from psycho_engine import INTERVENABLE_STATES
        for state in INTERVENABLE_STATES:
            assert state in PSYCHO_PROACTIVE_ADJUSTMENT, f"Missing: {state}"


# =========================================================================
# should_send_proactive -- psycho_state integration
# =========================================================================

def make_state(**kwargs):
    """Create a SchedulerState with overrides."""
    s = SchedulerState()
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


class TestShouldSendProactiveWithPsycho:
    """Verify should_send_proactive with psycho_state parameter."""

    # Prevent accounts module from interfering with test results
    @patch("accounts.compute_activity_score",
           return_value={"level": "active"})
    @patch("scheduler.datetime")
    def test_normal_no_override(self, mock_dt, mock_acct):
        """Normal state on a weekday morning: busy_hours -> no, no override."""
        mock_dt.now.return_value = datetime(2026, 7, 10, 10, 0, tzinfo=CST)  # Fri 10am
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw) if a else mock_dt.now()

        state = make_state(last_proactive_date="2026-07-09", proactive_count_today=0)
        decision = should_send_proactive(state, "u1",
                                          psycho_state={"intervenable_state": "normal"})
        assert decision["should_send"] is False  # busy hours
        assert decision["reason"] == "busy_hours"

    @patch("accounts.compute_activity_score",
           return_value={"level": "active"})
    @patch("scheduler.datetime")
    def test_willing_but_stuck_overrides_busy(self, mock_dt, mock_acct):
        """willing_but_stuck overrides busy_hours soft no."""
        mock_dt.now.return_value = datetime(2026, 7, 10, 10, 0, tzinfo=CST)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw) if a else mock_dt.now()

        state = make_state(last_proactive_date="2026-07-09", proactive_count_today=0)
        decision = should_send_proactive(state, "u1",
                                          psycho_state={"intervenable_state": "willing_but_stuck"})
        assert decision["should_send"] is True
        assert "psycho_" in decision["reason"]

    @patch("accounts.compute_activity_score",
           return_value={"level": "active"})
    @patch("scheduler.datetime")
    def test_avoidance_overrides_afternoon_ok(self, mock_dt, mock_acct):
        """avoidance_after_break overrides okay afternoon to no."""
        mock_dt.now.return_value = datetime(2026, 7, 11, 15, 0, tzinfo=CST)  # Sat 3pm
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw) if a else mock_dt.now()

        state = make_state(last_proactive_date="2026-07-09", proactive_count_today=0)
        decision = should_send_proactive(state, "u1",
                                          psycho_state={"intervenable_state": "avoidance_after_break"})
        assert decision["should_send"] is False
        assert "psycho_" in decision["reason"]

    @patch("accounts.compute_activity_score",
           return_value={"level": "active"})
    @patch("scheduler.datetime")
    def test_silent_mode_not_overridden(self, mock_dt, mock_acct):
        """Silent mode is a hard limit -- not overridden by positive psycho state."""
        mock_dt.now.return_value = datetime(2026, 7, 11, 15, 0, tzinfo=CST)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw) if a else mock_dt.now()
        # Make fromisoformat work on mock (for is_silent parsing)
        mock_dt.fromisoformat = datetime.fromisoformat

        # Use the mocked time for silent_until (matching scheduler.datetime)
        silent_until = "2026-07-12T15:00:00+08:00"
        state = make_state(silent_until=silent_until, proactive_count_today=0)
        decision = should_send_proactive(state, "u1",
                                          psycho_state={"intervenable_state": "social_motivated"})
        assert decision["should_send"] is False
        assert decision["reason"] == "silent_mode"

    @patch("accounts.compute_activity_score",
           return_value={"level": "active"})
    @patch("scheduler.datetime")
    def test_daily_limit_not_overridden(self, mock_dt, mock_acct):
        """Daily limit is a hard limit -- not overridden."""
        mock_dt.now.return_value = datetime(2026, 7, 11, 15, 0, tzinfo=CST)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw) if a else mock_dt.now()

        state = make_state(proactive_count_today=DAILY_PROACTIVE_LIMIT,
                           last_proactive_date="2026-07-11")
        decision = should_send_proactive(state, "u1",
                                          psycho_state={"intervenable_state": "willing_but_stuck"})
        assert decision["should_send"] is False
        assert decision["reason"] == "daily_limit_reached"

    @patch("accounts.compute_activity_score",
           return_value={"level": "active"})
    @patch("scheduler.datetime")
    def test_multiplier_map_comprehensive(self, mock_dt, mock_acct):
        """All psycho states with >1.0 multiplier override soft no."""
        mock_dt.now.return_value = datetime(2026, 7, 11, 15, 0, tzinfo=CST)  # Sat 3pm
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw) if a else mock_dt.now()

        for state_name, (mult, _reason) in PSYCHO_PROACTIVE_ADJUSTMENT.items():
            state = make_state(proactive_count_today=0)
            if mult > 1.0:
                decision = should_send_proactive(
                    state, "u1", psycho_state={"intervenable_state": state_name})
                assert decision["should_send"] is True, (
                    f"{state_name} (mult={mult}) should override soft no")
            elif mult < 1.0:
                decision = should_send_proactive(
                    state, "u1", psycho_state={"intervenable_state": state_name})
                assert decision["should_send"] is False, (
                    f"{state_name} (mult={mult}) should suppress yes")


# =========================================================================
# SchedulerState.load — corrupt / hand-edited persisted state
# =========================================================================


def _write_scheduler_state(tmp_path, payload):
    """Overwrite scheduler_state.json (default user) with a raw payload."""
    fp = Path(tmp_path) / "scheduler_state.json"
    fp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class TestSchedulerStateLoadCorrupt:
    """SchedulerState.load must tolerate corrupt/legacy scheduler_state.json."""

    @pytest.fixture
    def state_path(self, tmp_path, monkeypatch):
        import scheduler as scheduler_mod

        monkeypatch.setattr(
            scheduler_mod, "SCHEDULER_STATE_FILE",
            str(tmp_path / "scheduler_state.json"),
        )
        monkeypatch.setattr(scheduler_mod, "DATA_DIR", str(tmp_path))
        return tmp_path

    def test_load_tolerates_non_dict_top_level(self, state_path):
        """Valid JSON that is not a dict (list/int/str/null) degrades to defaults."""
        for bad in ([1, 2, 3], "junk", 42, None):
            _write_scheduler_state(state_path, bad)
            state = SchedulerState.load("default")
            assert state.unresponded_count == 0
            assert state.proactive_count_today == 0
            assert state.silent_until is None
            assert state.last_proactive_message is None

    def test_load_sanitizes_corrupt_field_types(self, state_path):
        """Corrupt field types degrade to safe defaults instead of crashing later."""
        _write_scheduler_state(state_path, {
            "unresponded_count": "abc",
            "proactive_count_today": None,
            "total_proactive_count": 2.7,
            "silent_until": {"bad": 1},
            "last_proactive_message": 5,
        })
        state = SchedulerState.load("default")
        assert state.unresponded_count == 0
        assert state.proactive_count_today == 0
        assert state.total_proactive_count == 2
        assert state.silent_until is None
        assert state.last_proactive_message is None
        # is_silent() must not raise on the sanitized fields.
        assert state.is_silent() is False


# =========================================================================
# _decide_proactive_type — corrupt/legacy message entries
# =========================================================================


class TestDecideProactiveTypeCorruptMessages:
    def test_silence_check_skips_non_dict_messages(self):
        """A trailing non-dict message doesn't crash the silence check."""
        from scheduler import _decide_proactive_type

        now = datetime(2026, 7, 6, 9, 0, tzinfo=CST)  # Mon 9am
        messages = [
            "legacy-junk-string",
            42,
            {"timestamp": "2026-07-01T08:00:00+08:00", "role": "user"},
        ]
        msg_type, extra = _decide_proactive_type("u1", now, messages, wake_up_hour=7)
        assert msg_type == "silence"
        assert extra["days_silent"] == 5

    def test_all_junk_messages_fall_through(self):
        """All-junk messages fall through to the general window without crashing."""
        from scheduler import _decide_proactive_type

        now = datetime(2026, 7, 6, 11, 0, tzinfo=CST)  # Mon 11am general window
        assert _decide_proactive_type("u1", now, ["junk", None, 42], wake_up_hour=7) == ("general", {})
