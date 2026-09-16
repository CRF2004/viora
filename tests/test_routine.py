"""Viora — Routine Module Tests

Tests for routine.py: RoutineModel, anomaly detection, and plan adherence checks.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from routine import RoutineModel, build_routine_model

CST = timezone(timedelta(hours=8))


def _make_msg(text: str, **extras) -> dict:
    """Create a minimal message dict."""
    return {
        "content": text,
        "extracted_data": extras,
        "timestamp": datetime.now(CST).isoformat(),
    }


# ── Test check_plan_adherence ───────────────────────────────────────────────


class TestCheckPlanAdherence:
    def test_empty_leaves(self):
        """Empty leaves returns empty list."""
        model = RoutineModel()
        assert model.check_plan_adherence([]) == []

    def test_never_completed_missed(self):
        """Leaf never completed is marked as 'missed'."""
        model = RoutineModel()
        leaves = [{
            "id": "l1", "label": "早餐全麦面包", "action_text": "早餐全麦面包替代白面包",
            "frequency": "每天", "time_slot": "早餐",
            "completed_count": 0, "last_completed": None, "status": "pending",
        }]
        results = model.check_plan_adherence(leaves)
        assert len(results) == 1
        assert results[0]["adherence"] == "missed"
        assert results[0]["leaf_id"] == "l1"

    def test_recently_completed_good(self):
        """Leaf completed today is 'good'."""
        model = RoutineModel()
        today_iso = datetime.now(CST).isoformat()
        leaves = [{
            "id": "l1", "label": "晨跑",
            "action_text": "晨跑30分钟", "frequency": "每天",
            "time_slot": "早晨", "completed_count": 5,
            "last_completed": today_iso, "status": "done",
        }]
        results = model.check_plan_adherence(leaves)
        assert results[0]["adherence"] == "good"
        assert results[0]["suggestion"] is None

    def test_missed_several_days(self):
        """Leaf not completed for several days is 'missed'."""
        model = RoutineModel()
        old_date = (datetime.now(CST) - timedelta(days=5)).isoformat()
        leaves = [{
            "id": "l2", "label": "冥想", "action_text": "冥想10分钟",
            "frequency": "每天", "time_slot": "睡前",
            "completed_count": 3, "last_completed": old_date, "status": "done",
        }]
        results = model.check_plan_adherence(leaves)
        assert results[0]["adherence"] == "missed"
        assert results[0]["days_missed"] >= 5

    def test_flagged_when_overdue(self):
        """Leaf overdue but not severely is 'flag'."""
        model = RoutineModel()
        old_date = (datetime.now(CST) - timedelta(days=2)).isoformat()
        leaves = [{
            "id": "l3", "label": "喝水", "action_text": "多喝水",
            "frequency": "每天", "time_slot": "全天",
            "completed_count": 10, "last_completed": old_date, "status": "done",
        }]
        results = model.check_plan_adherence(leaves)
        assert results[0]["adherence"] == "flag"

    def test_weekly_frequency(self):
        """Weekly frequency has longer tolerance."""
        model = RoutineModel()
        old_date = (datetime.now(CST) - timedelta(days=4)).isoformat()
        leaves = [{
            "id": "l4", "label": "游泳", "action_text": "游泳1小时",
            "frequency": "每周2次", "time_slot": "周末",
            "completed_count": 8, "last_completed": old_date, "status": "done",
        }]
        results = model.check_plan_adherence(leaves)
        # Expected gap for "每周" is 3 days, missed 4 days > 3 but < 6 (2*3)
        assert results[0]["adherence"] in ("flag",)

    def test_supports_reasoning_node_objects(self):
        """Works with ReasoningNode-like objects (not just dicts)."""
        model = RoutineModel()

        class FakeNode:
            id = "n1"
            label = "散步"
            action_text = "散步30分钟"
            frequency = "每天"
            time_slot = "傍晚"
            completed_count = 2
            last_completed = (datetime.now(CST) - timedelta(days=3)).isoformat()
            status = "done"

        results = model.check_plan_adherence([FakeNode()])
        assert len(results) == 1
        assert results[0]["leaf_id"] == "n1"

    def test_multiple_leaves_varied_adherence(self):
        """Multiple leaves with mixed adherence levels."""
        model = RoutineModel()
        now = datetime.now(CST)
        leaves = [
            {  # good — done today
                "id": "g", "label": "喝水",
                "frequency": "每天", "completed_count": 5,
                "last_completed": now.isoformat(),
            },
            {  # missed — never done
                "id": "m", "label": "跑步",
                "frequency": "隔天", "completed_count": 0,
                "last_completed": None,
            },
            {  # flag — overdue but not severely
                "id": "f", "label": "早睡",
                "frequency": "每天", "completed_count": 3,
                "last_completed": (now - timedelta(days=2)).isoformat(),
            },
        ]
        results = model.check_plan_adherence(leaves)
        result_map = {r["leaf_id"]: r["adherence"] for r in results}
        assert result_map["g"] == "good"
        assert result_map["m"] == "missed"
        assert result_map["f"] == "flag"


# ── Corrupt / malformed stored-state robustness ─────────────────────────────


def _msg_at(offset_days: int, **extracted) -> dict:
    """A message whose timestamp is *offset_days* from now."""
    ts = datetime.now(CST) + timedelta(days=offset_days)
    return {"content": "x", "extracted_data": extracted, "timestamp": ts.isoformat()}


class TestRoutineRobustness:
    """Messages are read from storage (LLM extraction persisted as JSON) and may
    be hand-edited / partially written. Out-of-schema extracted fields used to
    be stored as-is and crash the aggregation with TypeError/AttributeError
    (500 on /api/routine, /api/anomalies and the plan/chat flows).
    """

    # ── non-numeric numerics ────────────────────────────────────────────────

    def test_non_numeric_sleep_quality_does_not_crash(self):
        """A string sleep quality used to crash sum() in _build_routine_flags."""
        model = build_routine_model([
            _msg_at(0, sleep={"quality": "good"}),
            _msg_at(1, sleep={"quality": "bad"}),
        ])
        summary = model.get_routine_summary()
        assert "avg_sleep_quality" not in summary["patterns"]
        assert summary["data_points"]["sleep"] == 0

    def test_non_numeric_sleep_duration_does_not_crash(self):
        """A string sleep duration used to crash sum() in _build_routine_flags."""
        model = build_routine_model([
            _msg_at(0, sleep={"duration_hours": "7.5"}),
            _msg_at(1, sleep={"duration_hours": "8"}),
        ])
        assert "avg_sleep_duration" not in model.get_routine_summary()["patterns"]

    def test_non_numeric_exercise_duration_does_not_crash(self):
        """A string exercise duration used to crash sum() in _build_routine_flags."""
        model = build_routine_model([
            _msg_at(0, exercise={"type": "跑步", "duration_minutes": "30"}),
        ])
        patterns = model.get_routine_summary()["patterns"]
        assert patterns["primary_exercise_type"] == "跑步"
        assert "avg_exercise_duration" not in patterns

    def test_unhashable_exercise_type_does_not_crash(self):
        """A dict exercise type used to crash set() with TypeError."""
        model = build_routine_model([
            _msg_at(0, exercise={"type": {"a": 1}, "duration_minutes": 30}),
        ])
        patterns = model.get_routine_summary()["patterns"]
        assert "primary_exercise_type" not in patterns
        assert patterns["avg_exercise_duration"] == 30.0

    def test_numeric_junk_still_keeps_valid_siblings(self):
        """Junk on one message does not discard the well-formed siblings."""
        model = build_routine_model([
            _msg_at(0, sleep={"quality": 8, "duration_hours": 8.0}),
            _msg_at(1, sleep={"quality": "great"}),
            _msg_at(2, sleep={"quality": 6, "duration_hours": 6.0}),
        ])
        patterns = model.get_routine_summary()["patterns"]
        assert patterns["avg_sleep_quality"] == 7.0
        assert patterns["avg_sleep_duration"] == 7.0
        # The junk-only record carries no numeric signal and is dropped.
        assert patterns["sleep_data_points"] == 2

    # ── malformed message container / entries ───────────────────────────────

    @pytest.mark.parametrize("bad", [None, {}, "abc", 42])
    def test_non_list_message_container_degrades(self, bad):
        """A non-list container used to raise TypeError / AttributeError."""
        summary = build_routine_model(bad).get_routine_summary()
        assert summary["status"] == "data_insufficient"

    def test_non_dict_message_entries_skipped(self):
        """Non-dict entries used to crash msg.get() with AttributeError."""
        model = build_routine_model([
            "junk", 42, None,
            _msg_at(0, sleep={"quality": 7}),
            _msg_at(1, mood=6),
        ])
        summary = model.get_routine_summary()
        assert summary["status"] == "active"
        assert summary["data_points"]["total_messages"] == 2

    # ── well-formed behavior unchanged ──────────────────────────────────────

    def test_wellformed_summary_unchanged(self):
        """Well-formed input produces exactly the pre-hardening numbers."""
        model = build_routine_model([
            _msg_at(0, sleep={"quality": 7, "duration_hours": 7.5}, mood=6, energy=5,
                    exercise={"type": "跑步", "duration_minutes": 30}),
            _msg_at(1, sleep={"quality": 5, "duration_hours": 6.5}, mood=4, energy=3,
                    exercise={"type": "跑步", "duration_minutes": 50}),
        ])
        summary = model.get_routine_summary()
        assert summary["status"] == "active"
        assert summary["data_points"] == {
            "sleep": 2, "exercise": 2, "mood": 2, "energy": 2, "total_messages": 2,
        }
        patterns = summary["patterns"]
        assert patterns["avg_sleep_quality"] == 6.0
        assert patterns["avg_sleep_duration"] == 7.0
        assert patterns["sleep_data_points"] == 2
        assert patterns["exercise_count"] == 2
        assert patterns["primary_exercise_type"] == "跑步"
        assert patterns["avg_exercise_duration"] == 40.0
        assert patterns["avg_mood"] == 5.0
        assert patterns["mood_data_points"] == 2
        assert patterns["avg_energy"] == 4.0
        assert patterns["energy_data_points"] == 2
        assert patterns["interaction_count"] == 2
        assert isinstance(patterns["avg_interaction_hour"], float)
        assert summary["confidence"] == "low"

    def test_wellformed_detect_anomalies_unchanged(self):
        """Well-formed declining sleep quality still reports sleep_decline."""
        model = RoutineModel()
        model.sleep_times = [{"quality": q} for q in (9, 9, 9, 2, 1)]
        anomalies = model.detect_anomalies()
        assert [a["type"] for a in anomalies] == ["sleep_decline"]

    # ── detect_anomalies corrupt in-memory state ────────────────────────────

    def test_detect_anomalies_tolerates_non_numeric_quality(self):
        """Non-numeric stored quality used to crash sum() in detect_anomalies."""
        model = RoutineModel()
        model.sleep_times = [{"quality": "a"}, {"quality": "b"}, {"quality": "c"}]
        assert model.detect_anomalies() == []

    def test_detect_anomalies_tolerates_non_numeric_mood_score(self):
        """Non-numeric stored mood score used to crash sum() in detect_anomalies."""
        model = RoutineModel()
        model.mood_history = [{"score": "low"}, {"score": {"v": 1}}]
        assert model.detect_anomalies() == []

    # ── check_plan_adherence malformed leaves ───────────────────────────────

    def test_check_plan_adherence_non_str_frequency(self):
        """A numeric frequency used to crash `"每周" in frequency` with TypeError."""
        results = RoutineModel().check_plan_adherence([
            {"id": "l1", "label": "跑步", "frequency": 5, "last_completed": None},
        ])
        assert results[0]["frequency"] == ""
        assert results[0]["adherence"] == "missed"

    def test_check_plan_adherence_non_str_fields_sanitized(self):
        """Out-of-schema leaf fields used to leak into the API response verbatim."""
        results = RoutineModel().check_plan_adherence([
            {"id": ["a"], "label": {"x": 1}, "action_text": 7,
             "frequency": "每天", "time_slot": None, "last_completed": None},
        ])
        assert results[0]["leaf_id"] == ""
        assert results[0]["label"] == ""
        assert results[0]["action_text"] == ""
        assert results[0]["time_slot"] == ""
        assert results[0]["frequency"] == "每天"
        assert results[0]["adherence"] == "missed"
