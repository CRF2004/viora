"""Viora — Insights Module Tests

Regression tests for insights.py robustness against corrupt/legacy stored
message records, plus assertions that well-formed behaviour is unchanged.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from insights import compute_correlations, get_trend_insights


def _pair(energy, mood):
    return {"extracted_data": {"energy": energy, "mood": mood}}


WELL_FORMED = [_pair(1, 1), _pair(2, 2), _pair(3, 3), _pair(4, 4)]


# ── Well-formed behaviour is byte-identical ─────────────────────────────────


class TestWellFormedUnchanged:
    def test_correlation_output_unchanged(self):
        assert compute_correlations(WELL_FORMED) == [{
            "dimension1": "精力",
            "dimension2": "情绪",
            "correlation": 1.0,
            "n": 4,
            "description": "energy越好，mood也越好（强正向关联，r=1.00）",
        }]

    def test_trend_insights_output_unchanged(self):
        assert get_trend_insights(WELL_FORMED) == [
            {
                "type": "correlation",
                "text": "energy越好，mood也越好（强正向关联，r=1.00）",
                "confidence": "medium",
            },
            {
                "type": "trend",
                "text": "近期情绪趋势上升（1.0 → 3.0）",
                "confidence": "low",
            },
        ]

    def test_non_list_container_degrades_to_empty(self):
        for junk in (None, 42, "messages", {"a": 1}):
            assert compute_correlations(junk) == []
            assert get_trend_insights(junk) == []


# ── Corrupt / non-dict message entries ───────────────────────────────────────


class TestNonDictEntries:
    def test_compute_correlations_skips_junk_keeps_well_formed(self):
        # Previously: 'junk' -> AttributeError: 'str' object has no attribute 'get'
        msgs = ["junk", WELL_FORMED[0], None, WELL_FORMED[1],
                3.14, WELL_FORMED[2], ["nested"], WELL_FORMED[3]]
        result = compute_correlations(msgs)
        assert len(result) == 1
        assert result[0]["correlation"] == 1.0
        assert result[0]["n"] == 4

    def test_trend_insights_skips_junk_and_still_detects_trend(self):
        # Previously: non-dict entry -> AttributeError at m.get(...)
        msgs = ["junk", WELL_FORMED[0], None, WELL_FORMED[1],
                WELL_FORMED[2], {}, WELL_FORMED[3], 99]
        result = get_trend_insights(msgs)
        types = [i["type"] for i in result]
        assert types == ["correlation", "trend"]
        assert result[1]["text"] == "近期情绪趋势上升（1.5 → 3.5）"

    def test_all_junk_entries_return_empty(self):
        assert compute_correlations(["a", None, 1, []]) == []
        assert get_trend_insights(["a", None, 1, []]) == []


# ── Non-dict extracted_data ─────────────────────────────────────────────────


class TestExtractedDataShape:
    def test_compute_ignores_non_dict_extracted_data(self):
        # _extract_score_pairs already guarded this; keep it locked in.
        msgs = [{"extracted_data": "oops"}] + WELL_FORMED
        assert compute_correlations(msgs) == compute_correlations(WELL_FORMED)

    def test_trend_ignores_non_dict_extracted_data(self):
        # Previously: get_trend_insights crashed here while compute_correlations
        # did not -> AttributeError on the un-guarded .get("mood") chain.
        msgs = (
            [{"extracted_data": "oops"}, {"extracted_data": [1, 2]}]
            + WELL_FORMED
        )
        result = get_trend_insights(msgs)
        assert [i["type"] for i in result] == ["correlation", "trend"]
        assert result[1]["text"] == "近期情绪趋势上升（1.0 → 3.0）"

    def test_missing_extracted_data_is_ignored(self):
        assert compute_correlations([{}, {"content": "hi"}]) == []
        assert get_trend_insights([{}, {"content": "hi"}]) == []

    def test_non_numeric_mood_values_ignored(self):
        msgs = [
            _pair(1, 1), _pair(2, 2),
            {"extracted_data": {"mood": "good"}},
            {"extracted_data": {"mood": {"score": 3}}},
            _pair(3, 3), _pair(4, 5),
        ]
        result = get_trend_insights(msgs)
        trends = [i for i in result if i["type"] == "trend"]
        assert len(trends) == 1
        assert trends[0]["text"] == "近期情绪趋势上升（1.5 → 4.0）"
