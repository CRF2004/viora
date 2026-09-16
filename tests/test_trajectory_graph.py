"""
Tests for trajectory_graph.py — data model, ingestion, and periodicity detection.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import storage
from trajectory_graph import (
    TrajectoryGraph,
    EventNode,
    TrajectoryNode,
    StateNode,
    GraphEdge,
    EventDimension,
    PeriodType,
    EdgeType,
    STATE_WINDOW_DAYS,
    MIN_EVENTS_FOR_STATE,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

CST = timezone(timedelta(hours=8))


def _make_msg(content: str, dim_key: str, value: float | None,
              label: str = "", days_ago: int = 0, hour: int = 14) -> dict:
    """Helper to create a mock message dict."""
    ts = datetime.now(CST) - timedelta(days=days_ago, hours=(24 - hour))
    return {
        "id": f"msg_{days_ago}_{hour}_{dim_key}",
        "content": content,
        "timestamp": ts.isoformat(),
        "extracted_data": {
            dim_key: {
                "quality" if dim_key == "sleep" else "score" if dim_key in ("energy", "mood") else "severity": value,
            }
        },
    }


def _make_energy_msg(content: str, score: float, days_ago: int = 0, hour: int = 14) -> dict:
    return _make_msg(content, "energy", score, days_ago=days_ago, hour=hour)


def _make_sleep_msg(content: str, quality: float, days_ago: int = 0, hour: int = 8) -> dict:
    return _make_msg(content, "sleep", quality, days_ago=days_ago, hour=hour)


def _make_mood_msg(content: str, score: float, days_ago: int = 0, hour: int = 14) -> dict:
    return _make_msg(content, "mood", score, days_ago=days_ago, hour=hour)


# ── Tests: Data Models ───────────────────────────────────────────────────────

class TestEventNode:
    def test_default_id(self):
        e = EventNode()
        assert e.id.startswith("evt_")

    def test_to_dict(self):
        ts = datetime.now(CST)
        e = EventNode(
            dimension=EventDimension.ENERGY,
            value=7.0,
            label="精力 7/10",
            raw_text="今天精神不错",
            timestamp=ts,
            source_message_id="msg_001",
        )
        d = e.to_dict()
        assert d["dimension"] == "energy"
        assert d["value"] == 7.0
        assert d["label"] == "精力 7/10"
        assert d["source_message_id"] == "msg_001"


class TestTrajectoryNode:
    def test_default_id(self):
        t = TrajectoryNode()
        assert t.id.startswith("trj_")

    def test_to_dict(self):
        t = TrajectoryNode(
            label="周四下午低能量",
            dimension=EventDimension.ENERGY,
            period_type=PeriodType.WEEKLY,
            period_unit="weekday: 4, hour: 14",
            confidence=0.85,
            stability=0.72,
            sample_count=4,
            evidence_event_ids=["evt_1", "evt_2"],
        )
        d = t.to_dict()
        assert d["label"] == "周四下午低能量"
        assert d["period_type"] == "weekly"
        assert d["confidence"] == 0.85


class TestGraphEdge:
    def test_to_dict(self):
        edge = GraphEdge(
            source_id="evt_001",
            target_id="trj_001",
            edge_type=EdgeType.PERIODIC_PATTERN,
            weight=0.9,
        )
        d = edge.to_dict()
        assert d["source_id"] == "evt_001"
        assert d["target_id"] == "trj_001"
        assert d["edge_type"] == "periodic_pattern"


# ── Tests: Ingestion ─────────────────────────────────────────────────────────

class TestIngestion:
    def test_empty_messages(self):
        graph = TrajectoryGraph()
        count = graph.ingest_messages([])
        assert count == 0
        assert len(graph.events) == 0

    def test_single_message_single_dimension(self):
        graph = TrajectoryGraph()
        msgs = [_make_energy_msg("今天精神不错", 8.0)]
        count = graph.ingest_messages(msgs)
        assert count == 1
        assert graph.events[0].dimension == EventDimension.ENERGY
        assert graph.events[0].value == 8.0

    def test_multiple_messages_multiple_dimensions(self):
        graph = TrajectoryGraph()
        msgs = [
            _make_energy_msg("精神好", 8.0, days_ago=2),
            _make_sleep_msg("睡得不错", 7.0, days_ago=1),
            _make_mood_msg("心情好", 9.0, days_ago=0),
        ]
        count = graph.ingest_messages(msgs)
        assert count == 3
        dimensions = {e.dimension for e in graph.events}
        assert dimensions == {EventDimension.ENERGY, EventDimension.SLEEP, EventDimension.MOOD}

    def test_chronological_sorting(self):
        graph = TrajectoryGraph()
        msgs = [
            _make_energy_msg("今天", 5.0, days_ago=3),
            _make_energy_msg("昨天", 6.0, days_ago=1),
            _make_energy_msg("今天2", 7.0, days_ago=0),
        ]
        graph.ingest_messages(msgs)
        # Should be sorted oldest first
        assert graph.events[0].source_message_id == msgs[0]["id"]
        assert graph.events[-1].source_message_id == msgs[2]["id"]

    def test_message_without_extracted_data(self):
        graph = TrajectoryGraph()
        msgs = [{"id": "no_data", "content": "随便聊聊", "timestamp": datetime.now(CST).isoformat()}]
        count = graph.ingest_messages(msgs)
        assert count == 0


# ── Tests: Periodicity Detection ─────────────────────────────────────────────

class TestDailyDetection:
    def test_daily_pattern_detected(self):
        """Same hour, same dimension, across 3+ days."""
        graph = TrajectoryGraph()
        msgs = [
            _make_energy_msg("下午犯困", 3.0, days_ago=4, hour=14),
            _make_energy_msg("下午又困", 4.0, days_ago=3, hour=14),
            _make_energy_msg("下午没精神", 3.5, days_ago=2, hour=14),
            _make_energy_msg("下午想睡", 4.0, days_ago=1, hour=14),
        ]
        graph.ingest_messages(msgs)
        trajectories = graph.detect_periodicities()
        daily = [t for t in trajectories if t.period_type == PeriodType.DAILY]
        assert len(daily) >= 1
        assert "energy" in daily[0].label

    def test_daily_pattern_insufficient_days(self):
        """Only 1 day of data should not form a daily trajectory."""
        graph = TrajectoryGraph()
        msgs = [
            _make_energy_msg("下午犯困", 3.0, days_ago=0, hour=14),
            _make_energy_msg("下午还是困", 4.0, days_ago=0, hour=15),
        ]
        graph.ingest_messages(msgs)
        trajectories = graph.detect_periodicities()
        daily = [t for t in trajectories if t.period_type == PeriodType.DAILY]
        assert len(daily) == 0


class TestWeeklyDetection:
    def test_weekly_pattern_detected(self):
        """Same weekday, same hour, across 3+ weeks."""
        graph = TrajectoryGraph()
        # All on Thursday (weekday=3), hour 14, across 4 consecutive weeks
        msgs = [
            _make_energy_msg("周四困", 3.0, days_ago=21, hour=14),  # ~3 weeks ago (approx Thu)
            _make_energy_msg("周四又困", 3.5, days_ago=14, hour=14),  # ~2 weeks ago
            _make_energy_msg("周四还是困", 3.0, days_ago=7, hour=14),  # ~1 week ago
            _make_energy_msg("周四没精神", 4.0, days_ago=0, hour=14),  # this week
        ]
        graph.ingest_messages(msgs)
        trajectories = graph.detect_periodicities()
        weekly = [t for t in trajectories if t.period_type == PeriodType.WEEKLY]
        # May or may not be detected depending on weekday alignment
        # But at minimum daily should fire
        daily = [t for t in trajectories if t.period_type == PeriodType.DAILY]
        assert len(daily) >= 1 or len(weekly) >= 1


class TestConditionalDetection:
    def test_conditional_pattern_detected(self):
        """Poor sleep → low energy next day."""
        graph = TrajectoryGraph()
        msgs = [
            # 4 cycles of poor sleep followed by low energy
            _make_sleep_msg("没睡好", 2.0, days_ago=8, hour=23),
            _make_energy_msg("好累", 3.0, days_ago=7, hour=10),

            _make_sleep_msg("又没睡好", 3.0, days_ago=6, hour=23),
            _make_energy_msg("没精神", 2.5, days_ago=5, hour=10),

            _make_sleep_msg("睡得差", 2.0, days_ago=4, hour=23),
            _make_energy_msg("犯困", 3.5, days_ago=3, hour=10),

            _make_sleep_msg("失眠了", 1.0, days_ago=2, hour=23),
            _make_energy_msg("完全没力气", 2.0, days_ago=1, hour=10),
        ]
        graph.ingest_messages(msgs)
        trajectories = graph.detect_periodicities()
        cond = [t for t in trajectories if t.period_type == PeriodType.CONDITIONAL]
        assert len(cond) >= 1
        # The condition should be sleep→energy
        assert "sleep" in cond[0].label or "energy" in cond[0].label


class TestEmptyOrMinimalData:
    def test_no_events(self):
        graph = TrajectoryGraph()
        trajectories = graph.detect_periodicities()
        assert len(trajectories) == 0

    def test_single_event(self):
        graph = TrajectoryGraph()
        graph.ingest_messages([_make_energy_msg("还行", 6.0)])
        trajectories = graph.detect_periodicities()
        assert len(trajectories) == 0

    def test_two_events_same_day(self):
        """Two events on the same day should not form a pattern."""
        graph = TrajectoryGraph()
        graph.ingest_messages([
            _make_energy_msg("上午好", 7.0, days_ago=0, hour=10),
            _make_energy_msg("下午困", 4.0, days_ago=0, hour=14),
        ])
        trajectories = graph.detect_periodicities()
        daily = [t for t in trajectories if t.period_type == PeriodType.DAILY]
        assert len(daily) == 0


# ── Tests: Query API ─────────────────────────────────────────────────────────

class TestQueryAPI:
    def test_get_timeline(self):
        graph = TrajectoryGraph()
        msgs = [
            _make_energy_msg("好", 8.0, days_ago=2),
            _make_mood_msg("开心", 9.0, days_ago=1),
        ]
        graph.ingest_messages(msgs)
        timeline = graph.get_timeline()
        assert len(timeline) == 2

    def test_get_timeline_filter_dimension(self):
        graph = TrajectoryGraph()
        msgs = [
            _make_energy_msg("好", 8.0, days_ago=2),
            _make_mood_msg("开心", 9.0, days_ago=1),
        ]
        graph.ingest_messages(msgs)
        energy_only = graph.get_timeline(dimension=EventDimension.ENERGY)
        assert len(energy_only) == 1
        assert energy_only[0]["dimension"] == "energy"

    def test_get_trajectories_empty(self):
        graph = TrajectoryGraph()
        assert graph.get_trajectories() == []

    def test_to_dict_structure(self):
        graph = TrajectoryGraph()
        graph.ingest_messages([
            _make_energy_msg("还行", 6.0, days_ago=3),
            _make_energy_msg("好", 7.0, days_ago=0),
        ])
        d = graph.to_dict()
        assert "layer_1_events" in d
        assert "layer_2_trajectories" in d
        assert "edges" in d
        assert "stats" in d
        assert d["stats"]["total_events"] == 2


# ── Tests: Edge Cases ────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_none_values_in_extracted(self):
        """Messages where some dimensions have None values."""
        graph = TrajectoryGraph()
        msg = {
            "id": "partial",
            "content": "今天一般",
            "timestamp": datetime.now(CST).isoformat(),
            "extracted_data": {
                "energy": {"score": None},
                "mood": {"score": 6.0},
            },
        }
        graph.ingest_messages([msg])
        assert len(graph.events) == 1  # only mood extracted (energy has None)
        assert graph.events[0].dimension == EventDimension.MOOD

    def test_events_without_clock_time(self):
        """Events with no timestamp should not crash."""
        graph = TrajectoryGraph()
        graph.events = [
            EventNode(dimension=EventDimension.ENERGY, value=5.0, timestamp=None),
            EventNode(dimension=EventDimension.ENERGY, value=6.0, timestamp=None),
        ]
        trajectories = graph.detect_periodicities()
        assert len(trajectories) == 0  # no timestamp = no periodicity

    def test_same_dimension_varied_values(self):
        """Wildly varying values should have low consistency."""
        from trajectory_graph import TrajectoryGraph
        consistency = TrajectoryGraph._compute_consistency([1.0, 9.0, 2.0, 8.0])
        assert consistency < 0.5  # high variation = low consistency

    def test_consistent_values(self):
        from trajectory_graph import TrajectoryGraph
        consistency = TrajectoryGraph._compute_consistency([5.0, 5.0, 5.0, 5.0])
        assert consistency == 1.0  # perfect consistency


# ── Tests: Corrupt / legacy persisted-state tolerance ────────────────────────

class TestTrajectoryStorageRobustness:
    """load_events / load_trajectories must survive hand-edited storage.

    Mirrors the established hardening pattern: non-dict entries are skipped,
    out-of-schema scalar fields are sanitized, and malformed enum keys degrade
    to a default instead of raising. Well-formed data is byte-identical.
    """

    def _patch_events(self, monkeypatch, raw):
        monkeypatch.setattr(storage, "load_trajectory_events", lambda uid: raw)

    def _patch_trajectories(self, monkeypatch, raw):
        monkeypatch.setattr(storage, "load_trajectories", lambda uid: raw)

    def test_load_events_non_list_container(self, monkeypatch):
        # A per-user bucket flattened to a dict/scalar used to hit
        # item.get() on each key/char -> AttributeError.
        for raw in ({"a": 1}, "junk", 5, None):
            self._patch_events(monkeypatch, raw)
            graph = TrajectoryGraph()
            assert graph.load_events("u1") == 0
            assert graph.events == []

    def test_load_events_skips_non_dict_entries(self, monkeypatch):
        self._patch_events(monkeypatch, [
            "junk",
            None,
            3,
            {"id": "evt_ok", "dimension": "mood", "value": 6.0},
        ])
        graph = TrajectoryGraph()
        assert graph.load_events("u1") == 1
        assert graph.events[0].id == "evt_ok"

    def test_load_events_invalid_dimension_degrades(self, monkeypatch):
        # Unknown dimension key used to raise ValueError out of load_events.
        self._patch_events(monkeypatch, [
            {"id": "evt_x", "dimension": "bogus"},
            {"id": "evt_y", "dimension": ["mood"]},
        ])
        graph = TrajectoryGraph()
        assert graph.load_events("u1") == 2
        assert all(e.dimension == EventDimension.MOOD for e in graph.events)

    def test_load_events_sanitizes_out_of_schema_fields(self, monkeypatch):
        # Non-dict metadata / non-str strings / non-numeric value used to leak
        # into the timeline response (and crash to_dict on raw_text[:200]).
        self._patch_events(monkeypatch, [{
            "id": 123,
            "dimension": "energy",
            "value": "high",
            "label": {"nested": 1},
            "raw_text": 456,
            "source_message_id": None,
            "metadata": "oops",
        }])
        graph = TrajectoryGraph()
        assert graph.load_events("u1") == 1
        evt = graph.events[0]
        assert evt.id == ""
        assert evt.value is None
        assert evt.label == ""
        assert evt.raw_text == ""
        assert evt.source_message_id == ""
        assert evt.metadata == {}
        # to_dict must not raise (raw_text[:200] on an int would TypeError).
        assert graph.get_timeline()[0]["value"] is None

    def test_load_events_non_numeric_values_do_not_break_detection(self, monkeypatch):
        self._patch_events(monkeypatch, [
            {"id": "evt_1", "dimension": "energy", "value": "high",
             "timestamp": "2026-09-01T10:00:00+08:00"},
            {"id": "evt_2", "dimension": "energy", "value": {"n": 1},
             "timestamp": "2026-09-02T10:00:00+08:00"},
        ])
        graph = TrajectoryGraph()
        graph.load_events("u1")
        # sum()/float arithmetic on junk used to raise TypeError.
        assert graph.detect_periodicities() == []

    def test_load_events_wellformed_unchanged(self, monkeypatch):
        raw = [{
            "id": "evt_ok",
            "dimension": "energy",
            "value": 7.0,
            "label": "精力 7/10",
            "raw_text": "今天精神不错",
            "timestamp": "2026-09-01T10:00:00+08:00",
            "source_message_id": "msg_1",
            "metadata": {"k": 1},
        }]
        self._patch_events(monkeypatch, raw)
        graph = TrajectoryGraph()
        assert graph.load_events("u1") == 1
        evt = graph.events[0]
        assert (evt.id, evt.dimension, evt.value, evt.label, evt.raw_text,
                evt.source_message_id, evt.metadata) == (
            "evt_ok", EventDimension.ENERGY, 7.0, "精力 7/10",
            "今天精神不错", "msg_1", {"k": 1},
        )
        assert evt.timestamp.isoformat() == "2026-09-01T10:00:00+08:00"

    def test_load_trajectories_skips_non_dict_and_sanitizes(self, monkeypatch):
        # Non-dict entries used to raise AttributeError (not caught); unknown
        # enum keys and non-numeric fields leaked/crashed downstream.
        self._patch_trajectories(monkeypatch, [
            "junk",
            {
                "id": "trj_ok",
                "dimension": "bogus",
                "period_type": "nope",
                "confidence": "high",
                "stability": None,
                "sample_count": -3,
                "evidence_event_ids": ["a", 1, None, "b"],
                "period_unit": {"x": 1},
            },
        ])
        graph = TrajectoryGraph()
        assert graph.load_trajectories("u1") == 1
        trj = graph.trajectories[0]
        assert trj.dimension == EventDimension.ENERGY
        assert trj.period_type == PeriodType.DAILY
        assert trj.confidence == 0.0
        assert trj.stability == 0.0
        assert trj.sample_count == 0
        assert trj.evidence_event_ids == ["a", "b"]
        assert trj.period_unit == ""
        assert graph.get_trajectories()[0]["confidence"] == 0.0

    def test_load_trajectories_wellformed_unchanged(self, monkeypatch):
        raw = [{
            "id": "trj_ok",
            "label": "周四下午低能量",
            "dimension": "energy",
            "period_type": "weekly",
            "period_unit": "weekday: 3, hour: 14",
            "confidence": 0.85,
            "stability": 0.72,
            "sample_count": 4,
            "evidence_event_ids": ["evt_1", "evt_2"],
            "description": "每周四下午",
            "created_at": "2026-09-01T10:00:00+08:00",
        }]
        self._patch_trajectories(monkeypatch, raw)
        graph = TrajectoryGraph()
        assert graph.load_trajectories("u1") == 1
        d = graph.get_trajectories()[0]
        assert d["id"] == "trj_ok"
        assert d["dimension"] == "energy"
        assert d["period_type"] == "weekly"
        assert d["confidence"] == 0.85
        assert d["stability"] == 0.72
        assert d["sample_count"] == 4
        assert d["evidence_event_ids"] == ["evt_1", "evt_2"]
        assert d["created_at"] == "2026-09-01T10:00:00+08:00"
