"""Viora — Health Graph Tests

Tests for health_graph.py: correlation graph generation, dimension averages,
and deserialization / malformed-input robustness.

Covers the /api/health-graph path, which feeds stored messages (LLM extraction
persisted as JSON, possibly hand-edited / partially written) straight into the
graph builder. Before hardening, corrupt records crashed the endpoint with
AttributeError (non-dict messages) or TypeError (non-numeric scores, non-string
correlation labels / coefficients).
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from health_graph import (
    DIMENSION_NODES,
    generate_health_graph,
    _compute_dimension_avg,
    _find_dimension_id,
    _has_dimension,
)

CST = timezone(timedelta(hours=8))


def _msg(**extras) -> dict:
    """Create a minimal stored message dict."""
    return {
        "timestamp": datetime.now(CST).isoformat(),
        "extracted_data": extras,
    }


def _node(graph: dict, dim_id: str) -> dict:
    """Fetch a dimension node from the generated graph."""
    return next(n for n in graph["nodes"] if n["id"] == dim_id)


def _corr_edges(graph: dict) -> list[dict]:
    """Fetch the non-structural (correlation) relationships."""
    return [r for r in graph["relationships"] if r["type"] == "关联"]


# ── Well-formed behavior (must stay byte-identical) ─────────────────────────


class TestHealthGraphWellFormed:
    def test_well_formed_edge_unchanged(self):
        """A well-formed correlation renders the exact same edge."""
        graph = generate_health_graph(correlations=[{
            "dimension1": "睡眠", "dimension2": "精力",
            "correlation": 0.82, "n": 10, "description": "睡眠与精力强正相关",
        }])
        edges = _corr_edges(graph)
        assert edges == [{
            "source": "sleep",
            "target": "energy",
            "type": "关联",
            "strength": 0.82,
            "direction": "正向",
            "weight": 8,
            "color": "#e8a87c",
            "description": "睡眠与精力强正相关",
        }]

    def test_well_formed_negative_edge_unchanged(self):
        """Negative coefficients keep direction / color / weight semantics."""
        graph = generate_health_graph(correlations=[{
            "dimension1": "情绪", "dimension2": "睡眠",
            "correlation": -0.55, "description": "负相关",
        }])
        edges = _corr_edges(graph)
        assert len(edges) == 1
        assert edges[0]["source"] == "mood"
        assert edges[0]["target"] == "sleep"
        assert edges[0]["strength"] == 0.55
        assert edges[0]["direction"] == "负向"
        assert edges[0]["weight"] == 5
        assert edges[0]["color"] == "#a8b5a0"

    def test_well_formed_nodes_and_averages_unchanged(self):
        """Node averages / data_points are unchanged for well-formed messages."""
        messages = [
            _msg(sleep={"quality": 4, "duration_hours": 7.5}),
            _msg(sleep={"quality": 2, "duration_hours": 6.5}),
            _msg(energy={"score": 3}, mood={"score": 4}),
        ]
        graph = generate_health_graph(correlations=[], messages=messages)

        # center node + every dimension node
        assert len(graph["nodes"]) == len(DIMENSION_NODES) + 1
        assert graph["nodes"][0]["id"] == "center"

        assert _node(graph, "sleep")["properties"] == {"avg_score": 3.0, "data_points": 2}
        assert _node(graph, "energy")["properties"] == {"avg_score": 3.0, "data_points": 1}
        assert _node(graph, "mood")["properties"] == {"avg_score": 4.0, "data_points": 1}
        assert _node(graph, "exercise")["properties"] == {"avg_score": None, "data_points": 0}

    def test_center_edges_unchanged(self):
        """Every dimension is still connected to the center node."""
        graph = generate_health_graph(correlations=[])
        structural = [r for r in graph["relationships"] if r["type"] == "包含"]
        assert len(structural) == len(DIMENSION_NODES)
        assert {r["target"] for r in structural} == {d["id"] for d in DIMENSION_NODES}

    def test_compute_dimension_avg_legacy_int_format(self):
        """Legacy flat energy/mood ints are still averaged (0 included)."""
        messages = [_msg(energy=3), _msg(energy=0), _msg(energy=4)]
        assert _compute_dimension_avg(messages, "energy") == pytest.approx(2.3, abs=0.05)

    def test_find_dimension_id_well_formed(self):
        """Exact and fuzzy label matching is unchanged."""
        assert _find_dimension_id("睡眠") == "sleep"
        assert _find_dimension_id("睡眠质量") == "sleep"
        assert _find_dimension_id("情绪") == "mood"
        assert _find_dimension_id("不存在的维度") is None


# ── Malformed input / corrupt stored state ──────────────────────────────────


class TestHealthGraphRobustness:
    @pytest.mark.parametrize("bad_messages", [None, {"a": 1}, "abc", 7, 3.5])
    def test_non_list_messages_degrade_to_no_data(self, bad_messages):
        """A non-list messages container degrades instead of crashing."""
        graph = generate_health_graph(messages=bad_messages)
        assert _node(graph, "sleep")["properties"] == {"avg_score": None, "data_points": 0}
        assert _corr_edges(graph) == []

    def test_non_dict_message_entries_skipped(self):
        """Non-dict stored records are skipped, well-formed ones preserved."""
        messages = [
            _msg(sleep={"quality": 4}),
            "junk",
            None,
            5,
            ["nested"],
            _msg(sleep={"quality": 2}),
        ]
        graph = generate_health_graph(correlations=[], messages=messages)
        assert _node(graph, "sleep")["properties"] == {"avg_score": 3.0, "data_points": 2}

    @pytest.mark.parametrize("bad_quality", ["good", {"score": 4}, [3], None])
    def test_non_numeric_sleep_quality_ignored(self, bad_quality):
        """A corrupt sleep.quality no longer reaches sum() (was TypeError)."""
        messages = [_msg(sleep={"quality": bad_quality})]
        graph = generate_health_graph(correlations=[], messages=messages)
        assert _node(graph, "sleep")["properties"]["avg_score"] is None

    @pytest.mark.parametrize("bad_score", ["high", {"nested": 1}, [1, 2], "5"])
    def test_non_numeric_energy_mood_score_ignored(self, bad_score):
        """Corrupt energy/mood scores no longer reach sum() (was TypeError)."""
        messages = [_msg(energy={"score": bad_score}, mood={"score": bad_score})]
        graph = generate_health_graph(correlations=[], messages=messages)
        assert _node(graph, "energy")["properties"]["avg_score"] is None
        assert _node(graph, "mood")["properties"]["avg_score"] is None

    def test_corrupt_scores_do_not_poison_good_average(self):
        """Junk values are dropped; valid values still average correctly."""
        messages = [
            _msg(sleep={"quality": 4}),
            _msg(sleep={"quality": "good"}),
            _msg(sleep={"quality": 2}),
            _msg(sleep={"quality": {"x": 1}}),
        ]
        graph = generate_health_graph(correlations=[], messages=messages)
        assert _node(graph, "sleep")["properties"]["avg_score"] == 3.0

    def test_non_dict_extracted_data_skipped(self):
        """A non-dict extracted_data payload does not crash aggregation."""
        messages = [
            {"extracted_data": "broken"},
            {"extracted_data": [1, 2, 3]},
            {"extracted_data": None},
            _msg(sleep={"quality": 5}),
        ]
        graph = generate_health_graph(correlations=[], messages=messages)
        assert _node(graph, "sleep")["properties"]["avg_score"] == 5.0

    @pytest.mark.parametrize("bad_correlations", [None, "abc", {"a": 1}, 7])
    def test_non_list_correlations_degrade(self, bad_correlations):
        """A non-list correlations container yields no edges (was AttributeError)."""
        graph = generate_health_graph(correlations=bad_correlations)
        assert _corr_edges(graph) == []

    def test_non_dict_correlation_entries_skipped(self):
        """Junk entries in the correlations list are skipped, valid ones kept."""
        graph = generate_health_graph(correlations=[
            "junk",
            5,
            None,
            {"dimension1": "睡眠", "dimension2": "精力", "correlation": 0.5},
        ])
        edges = _corr_edges(graph)
        assert len(edges) == 1
        assert edges[0]["source"] == "sleep"
        assert edges[0]["target"] == "energy"

    @pytest.mark.parametrize("bad_strength", ["0.8", {"x": 1}, [0.5], None, True])
    def test_non_numeric_strength_skips_edge(self, bad_strength):
        """A non-numeric coefficient no longer raises in abs() (was TypeError)."""
        graph = generate_health_graph(correlations=[{
            "dimension1": "睡眠", "dimension2": "精力", "correlation": bad_strength,
        }])
        assert _corr_edges(graph) == []

    def test_missing_strength_keeps_legacy_zero_edge(self):
        """A missing coefficient still defaults to 0, as before."""
        graph = generate_health_graph(correlations=[{
            "dimension1": "睡眠", "dimension2": "精力",
        }])
        edges = _corr_edges(graph)
        assert len(edges) == 1
        assert edges[0]["strength"] == 0
        assert edges[0]["weight"] == 1

    @pytest.mark.parametrize("bad_label", [5, {"a": 1}, [1], None, b"sleep"])
    def test_non_string_correlation_labels_skipped(self, bad_label):
        """Non-string labels no longer raise in the `in` test (was TypeError)."""
        graph = generate_health_graph(correlations=[
            {"dimension1": bad_label, "dimension2": "精力", "correlation": 0.5},
            {"dimension1": "睡眠", "dimension2": bad_label, "correlation": 0.5},
        ])
        assert _corr_edges(graph) == []

    def test_empty_label_does_not_fabricate_edge(self):
        """An empty / missing label no longer fuzzy-matches the first dimension."""
        graph = generate_health_graph(correlations=[
            {"dimension1": "", "dimension2": "精力", "correlation": 0.6},
            {"dimension2": "精力", "correlation": 0.6},
        ])
        assert _corr_edges(graph) == []

    @pytest.mark.parametrize("bad_desc", [{"a": 1}, [1], 7, None])
    def test_non_string_description_sanitized(self, bad_desc):
        """A non-string description degrades to "" instead of leaking junk."""
        graph = generate_health_graph(correlations=[{
            "dimension1": "睡眠", "dimension2": "精力",
            "correlation": 0.5, "description": bad_desc,
        }])
        edges = _corr_edges(graph)
        assert len(edges) == 1
        assert edges[0]["description"] == ""

    def test_has_dimension_non_dict_message(self):
        """_has_dimension tolerates a non-dict stored record."""
        assert _has_dimension("junk", "sleep") is False
        assert _has_dimension(None, "sleep") is False
        assert _has_dimension(5, "sleep") is False
        assert _has_dimension(_msg(sleep={"quality": 3}), "sleep") is True

    def test_corrupt_messages_do_not_break_correlation_computation(self):
        """correlations=None with corrupt messages still builds a graph."""
        messages = [
            "junk",
            _msg(sleep={"quality": "good", "duration_hours": "7"}),
            _msg(energy={"score": "high"}),
            _msg(mood={"score": [1, 2]}),
        ]
        graph = generate_health_graph(messages=messages)
        assert len(graph["nodes"]) == len(DIMENSION_NODES) + 1
        assert _node(graph, "sleep")["properties"]["avg_score"] is None
