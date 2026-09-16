"""
Viora — PsychoEngine Test Suite (22+ test cases)

Validates the psychological profiling pipeline (Phase A):
1. PsychoEvent data model (creation, from_extracted, serialization)
2. Extraction normalization and validation
3. EMA computation helper
4. end-to-end LLM extraction (mocked) for 10+ typical psychological expressions

Phase A success criteria from plan.md §9.8:
  "用户进行运动相关对话时，能正确提取心理维度（意愿、疲劳、成就感等），
   准确率 > 70%"
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from psycho_engine import (
    PsychoEvent,
    PsychoState,
    PsychoTrait,
    _empty_psycho_extracted,
    _normalize_psycho_extracted,
    extract_psycho_data,
    classify_intervenable_state,
    recommend_intervention,
    compute_psycho_fingerprint,
    compute_ema,
    PSYCHO_DIMENSIONS,
    PSYCHO_PHASES,
)


# =========================================================================
# PsychoEvent Data Model
# =========================================================================

class TestPsychoEventDefault:
    """Default PsychoEvent creation."""

    def test_default_creation(self):
        event = PsychoEvent()
        assert event.id.startswith("psy_")
        assert event.phase == "idle"
        assert event.source_type == "chat_inferred"
        assert event.confidence == 0.7
        assert event.has_any_signal is False

    def test_default_all_dimensions_none(self):
        event = PsychoEvent()
        for dim in PSYCHO_DIMENSIONS:
            assert getattr(event, dim) is None, f"{dim} should be None"

    def test_default_user_id_empty(self):
        event = PsychoEvent()
        assert event.user_id == ""


class TestPsychoEventFromExtracted:
    """PsychoEvent.from_extracted() class method."""

    def test_all_dimensions(self):
        extracted = {
            "willingness": 3, "fatigue": 7, "stress": 6,
            "procrastination": 8, "achievement": 2, "frustration": 5,
            "guilt": 4, "next_confidence": 3, "avoidance": 6,
            "phase": "pre_exercise", "reason": "用户表达明显运动阻力",
        }
        event = PsychoEvent.from_extracted(extracted, user_id="u1")
        assert event.user_id == "u1"
        assert event.willingness == 3
        assert event.fatigue == 7
        assert event.stress == 6
        assert event.procrastination == 8
        assert event.achievement == 2
        assert event.frustration == 5
        assert event.guilt == 4
        assert event.next_confidence == 3
        assert event.avoidance == 6
        assert event.phase == "pre_exercise"
        assert event.reason == "用户表达明显运动阻力"
        assert event.has_any_signal is True

    def test_partial_dimensions(self):
        """Only expressed dimensions are populated; others remain None."""
        extracted = {"willingness": 8, "achievement": 9, "next_confidence": 8,
                     "phase": "post_exercise", "reason": "用户表达运动后积极感受"}
        event = PsychoEvent.from_extracted(extracted, user_id="u1")
        assert event.willingness == 8
        assert event.achievement == 9
        assert event.next_confidence == 8
        # All other dims should be None
        assert event.fatigue is None
        assert event.stress is None
        assert event.procrastination is None
        assert event.frustration is None
        assert event.guilt is None
        assert event.avoidance is None

    def test_null_phase_falls_back_to_idle(self):
        extracted = {"willingness": 5, "phase": None}
        event = PsychoEvent.from_extracted(extracted, user_id="u1")
        assert event.phase == "idle"

    def test_explicit_phase_override(self):
        extracted = {"willingness": 5}
        event = PsychoEvent.from_extracted(extracted, user_id="u1", phase="during_exercise")
        assert event.phase == "during_exercise"

    def test_source_message_id(self):
        extracted = {"achievement": 7}
        event = PsychoEvent.from_extracted(extracted, user_id="u1", message_id="msg_123")
        assert event.source_message_id == "msg_123"

    def test_confidence_scales_with_non_null_dims(self):
        """Confidence increases with the number of non-null dimensions."""
        low = PsychoEvent.from_extracted({"willingness": 5}, user_id="u1")
        high = PsychoEvent.from_extracted(
            {dim: 5 for dim in PSYCHO_DIMENSIONS}, user_id="u1"
        )
        assert low.confidence == 0.56  # 0.5 + 1 * 0.06
        assert high.confidence == 1.0  # capped at 1.0


class TestPsychoEventProperties:
    """PsychoEvent derived properties."""

    def test_has_any_signal_true(self):
        event = PsychoEvent(willingness=3)
        assert event.has_any_signal is True

    def test_has_any_signal_false(self):
        event = PsychoEvent()
        assert event.has_any_signal is False

    def test_to_dict_contains_all_fields(self):
        event = PsychoEvent(willingness=3, fatigue=7, phase="pre_exercise")
        d = event.to_dict()
        assert isinstance(d, dict)
        assert d["willingness"] == 3
        assert d["fatigue"] == 7
        assert d["phase"] == "pre_exercise"
        assert "id" in d
        assert "user_id" in d
        assert "timestamp" in d
        assert "confidence" in d


# =========================================================================
# Extraction Normalization
# =========================================================================

class TestEmptyPsychoExtracted:
    """_empty_psycho_extracted() helper."""

    def test_all_fields_none(self):
        result = _empty_psycho_extracted()
        for dim in PSYCHO_DIMENSIONS:
            assert result[dim] is None, f"{dim} should be None"
        assert result["phase"] is None
        assert result["reason"] is None

    def test_all_dimensions_present(self):
        result = _empty_psycho_extracted()
        for dim in PSYCHO_DIMENSIONS:
            assert dim in result
        assert "phase" in result
        assert "reason" in result


class TestNormalizePsychoExtracted:
    """_normalize_psycho_extracted() validation."""

    def test_valid_values_pass_through(self):
        raw = {dim: 5 for dim in PSYCHO_DIMENSIONS}
        raw["phase"] = "pre_exercise"
        raw["reason"] = "valid test"
        result = _normalize_psycho_extracted(raw)
        for dim in PSYCHO_DIMENSIONS:
            assert result[dim] == 5, f"{dim} should be 5"
        assert result["phase"] == "pre_exercise"
        assert result["reason"] == "valid test"

    def test_out_of_range_clamped(self):
        raw = {"willingness": -1, "fatigue": 15, "procrastination": 10}
        result = _normalize_psycho_extracted(raw)
        assert result["willingness"] is None  # < 0
        assert result["fatigue"] is None       # > 10
        assert result["procrastination"] == 10  # boundary OK

    def test_invalid_types_set_to_none(self):
        raw = {"willingness": "not_a_number", "fatigue": [1, 2, 3],
               "stress": {"value": 5}}
        result = _normalize_psycho_extracted(raw)
        assert result["willingness"] is None
        assert result["fatigue"] is None
        assert result["stress"] is None

    def test_float_values_preserved(self):
        raw = {"willingness": 3.5, "fatigue": 7.2}
        result = _normalize_psycho_extracted(raw)
        assert result["willingness"] == 3.5
        assert result["fatigue"] == 7.2

    def test_valid_phase_preserved(self):
        for phase in PSYCHO_PHASES:
            result = _normalize_psycho_extracted({"phase": phase})
            assert result["phase"] == phase, f"phase '{phase}' should be preserved"

    def test_invalid_phase_set_to_none(self):
        result = _normalize_psycho_extracted({"phase": "unknown_phase"})
        assert result["phase"] is None

    def test_unknown_keys_ignored(self):
        result = _normalize_psycho_extracted({"unknown_field": 123})
        assert "unknown_field" not in result

    def test_reason_string_preserved(self):
        """A string reason passes through unchanged."""
        result = _normalize_psycho_extracted({"reason": "用户表达运动后积极感受"})
        assert result["reason"] == "用户表达运动后积极感受"

    def test_reason_missing_defaults_to_empty_string(self):
        """Missing reason defaults to empty string (backward compatible)."""
        result = _normalize_psycho_extracted({})
        assert result["reason"] == ""

    def test_reason_null_preserved(self):
        """Explicit null reason stays null (matches _empty_psycho_extracted)."""
        result = _normalize_psycho_extracted({"reason": None})
        assert result["reason"] is None

    def test_reason_non_string_sanitized_to_none(self):
        """Structured/numeric junk in reason is sanitized, not leaked.

        reason is documented as str|None; a malformed LLM response putting a
        dict/list/number there must not leak out of the extraction schema
        (downstream PsychoEvent.reason is typed str).
        """
        for junk in ({"detail": "压力大"}, ["压力大"], 42, 3.5, True):
            result = _normalize_psycho_extracted({"reason": junk})
            assert result["reason"] is None, f"reason {junk!r} should be sanitized"


# =========================================================================
# EMA Computation
# =========================================================================

class TestComputeEMA:
    """Exponential moving average computation.

    Formula: result = alpha * existing + (1 - alpha) * new_value
    High alpha = more weight on existing value = smoother
    Low alpha = more weight on new value = faster adaptation
    """

    def test_first_value_returns_new_value(self):
        assert compute_ema(None, 5.0, 0.5) == 5.0

    def test_subsequent_update(self):
        result = compute_ema(8.0, 2.0, 0.5)
        assert result == pytest.approx(5.0)  # 0.5 * 8 + 0.5 * 2

    def test_alpha_one_maximum_smoothing(self):
        """alpha=1 means fully retain existing, ignore new value."""
        assert compute_ema(10.0, 3.0, 1.0) == 10.0  # 1.0 * 10 + 0.0 * 3

    def test_alpha_zero_no_smoothing(self):
        """alpha=0 means fully adopt new value, ignore existing."""
        assert compute_ema(7.0, 999.0, 0.0) == 999.0  # 0.0 * 7 + 1.0 * 999

    def test_high_alpha_slower_adaptation(self):
        """Higher alpha = more weight on existing (smoother)."""
        # alpha=0.75: 0.75*5 + 0.25*9 = 3.75 + 2.25 = 6.0
        assert compute_ema(5.0, 9.0, 0.75) == pytest.approx(6.0)

    def test_integer_inputs(self):
        result = compute_ema(8, 2, 0.5)
        assert result == pytest.approx(5.0)


# =========================================================================
# LLM Extraction — Mocked Integration Tests (10+ typical expressions)
# =========================================================================

class TestExtractPsychoData:
    """extract_psycho_data() with mocked LLM API."""

    def _mock_llm_response(self, content_dict):
        """Helper: create a mock requests.post response."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(content_dict)}}]
        }
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    # ── 1. 低运动意愿 + 高启动阻力 ──

    @patch("requests.post")
    def test_extract_low_willingness_high_procrastination(self, mock_post):
        """'今天一点都不想动' → low willingness, high procrastination."""
        mock_post.return_value = self._mock_llm_response({
            "willingness": 2, "procrastination": 8,
            "phase": "pre_exercise",
            "reason": "用户明确表达不想运动，启动阻力高",
        })
        result = extract_psycho_data("今天一点都不想动")
        assert result["willingness"] == 2
        assert result["procrastination"] == 8
        assert result["phase"] == "pre_exercise"

    # ── 2. 运动后积极反馈 ──

    @patch("requests.post")
    def test_extract_post_exercise_achievement(self, mock_post):
        """'跑完感觉还不错' → moderate achievement, next_confidence."""
        mock_post.return_value = self._mock_llm_response({
            "achievement": 7, "next_confidence": 7,
            "phase": "post_exercise",
            "reason": "用户表达运动后积极感受",
        })
        result = extract_psycho_data("跑完感觉还不错")
        assert result["achievement"] == 7
        assert result["next_confidence"] == 7
        assert result["phase"] == "post_exercise"

    # ── 3. 高强度疲劳 ──

    @patch("requests.post")
    def test_extract_high_fatigue(self, mock_post):
        """'太累了'（运动上下文）→ high fatigue."""
        mock_post.return_value = self._mock_llm_response({
            "fatigue": 8,
            "phase": "during_exercise",
            "reason": "用户表达高强度疲劳感",
        })
        result = extract_psycho_data("太累了")
        assert result["fatigue"] == 8
        assert result["phase"] == "during_exercise"

    # ── 4. 自责 + 回避倾向 ──

    @patch("requests.post")
    def test_extract_guilt_and_avoidance(self, mock_post):
        """'又没去，我真没用' → high guilt, high avoidance."""
        mock_post.return_value = self._mock_llm_response({
            "guilt": 9, "avoidance": 7,
            "phase": "idle",
            "reason": "用户因未坚持运动而自责，并有回避倾向",
        })
        result = extract_psycho_data("又没去，我真没用")
        assert result["guilt"] == 9
        assert result["avoidance"] == 7

    # ── 5. 高压力 ──

    @patch("requests.post")
    def test_extract_high_stress(self, mock_post):
        """'最近压力好大' → high stress."""
        mock_post.return_value = self._mock_llm_response({
            "stress": 8,
            "phase": "idle",
            "reason": "用户表达近期压力大",
        })
        result = extract_psycho_data("最近压力好大")
        assert result["stress"] == 8

    # ── 6. 运动后酸痛 ──

    @patch("requests.post")
    def test_extract_post_exercise_soreness(self, mock_post):
        """'昨天做完运动今天浑身酸痛' → high fatigue, post_exercise."""
        mock_post.return_value = self._mock_llm_response({
            "fatigue": 7,
            "phase": "next_day",
            "reason": "用户表达运动后第二天肌肉酸痛",
        })
        result = extract_psycho_data("昨天做完运动今天浑身酸痛")
        assert result["fatigue"] == 7
        assert result["phase"] == "next_day"

    # ── 7. 坚持的成就感 + 信心 ──

    @patch("requests.post")
    def test_extract_consistency_achievement(self, mock_post):
        """'坚持了三天，感觉还行' → moderate achievement + confidence."""
        mock_post.return_value = self._mock_llm_response({
            "achievement": 6, "next_confidence": 6,
            "phase": "post_exercise",
            "reason": "用户因连续运动表达成就感和信心",
        })
        result = extract_psycho_data("坚持了三天，感觉还行")
        assert result["achievement"] == 6
        assert result["next_confidence"] == 6

    # ── 8. 挫败感（运动效果不满意）──

    @patch("requests.post")
    def test_extract_frustration(self, mock_post):
        """'跑了一周一斤没瘦' → high frustration."""
        mock_post.return_value = self._mock_llm_response({
            "frustration": 8, "willingness": 4,
            "phase": "post_exercise",
            "reason": "用户对运动效果不满，挫败感明显",
        })
        result = extract_psycho_data("跑了一周一斤没瘦")
        assert result["frustration"] == 8
        assert result["willingness"] == 4

    # ── 9. 无心理信号（日常闲聊）──

    @patch("requests.post")
    def test_extract_no_signal_casual_chat(self, mock_post):
        """日常闲聊 → all null."""
        mock_post.return_value = self._mock_llm_response({
            "willingness": None, "fatigue": None, "stress": None,
            "procrastination": None, "achievement": None, "frustration": None,
            "guilt": None, "next_confidence": None, "avoidance": None,
            "phase": None, "reason": None,
        })
        result = extract_psycho_data("今天天气不错，吃了个好吃的")
        for dim in PSYCHO_DIMENSIONS:
            assert result[dim] is None, f"{dim} should be None for casual chat"

    # ── 10. API 错误降级 ──

    @patch("requests.post")
    def test_extract_api_failure_graceful_degradation(self, mock_post):
        """LLM API failure → gracefully return empty extraction (no crash)."""
        mock_post.side_effect = Exception("Connection timeout")
        result = extract_psycho_data("今天好累")
        for dim in PSYCHO_DIMENSIONS:
            assert result[dim] is None, f"{dim} should be None on API failure"
        assert result["phase"] is None

    # ── 11. 空消息 ──

    @patch("requests.post")
    def test_extract_empty_message(self, mock_post):
        """Empty/whitespace message → no crash."""
        mock_post.return_value = self._mock_llm_response({
            "willingness": None, "fatigue": None, "stress": None,
            "procrastination": None, "achievement": None, "frustration": None,
            "guilt": None, "next_confidence": None, "avoidance": None,
            "phase": None, "reason": None,
        })
        for msg in ["", "   ", "\n"]:
            result = extract_psycho_data(msg)
            assert isinstance(result, dict)

    # ── 12. LLM 返回非 JSON ──

    @patch("requests.post")
    def test_extract_non_json_response(self, mock_post):
        """LLM returns non-JSON content → graceful fallback to empty."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "I'm sorry, I can't do that."}}]
        }
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp
        result = extract_psycho_data("今天怎么样")
        for dim in PSYCHO_DIMENSIONS:
            assert result[dim] is None


# ══════════════════════════════════════════════════════════════════════════════
# Phase B — PsychoState Aggregation + State Classification Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizeDim:
    """Normalize 0-10 dimension values to 0-1."""

    def test_normal_ten_to_one(self):
        from psycho_engine import normalize_dim
        assert normalize_dim(5.0) == 0.5
        assert normalize_dim(0.0) == 0.0
        assert normalize_dim(10.0) == 1.0
        assert normalize_dim(3) == 0.3

    def test_none_defaults_to_mid(self):
        from psycho_engine import normalize_dim
        assert normalize_dim(None) == 0.5

    def test_none_with_custom_default(self):
        from psycho_engine import normalize_dim
        assert normalize_dim(None, default=0.0) == 0.0


class TestClassifyIntervenableState:
    """All 9 intervenable state classification rules."""

    def test_normal_when_all_mid(self):
        """All dimensions at 0.5 → normal."""
        ema = {d: 5.0 for d in PSYCHO_DIMENSIONS}
        assert classify_intervenable_state(ema) == "normal"

    def test_normal_when_all_none(self):
        """No data → normal."""
        ema = {d: None for d in PSYCHO_DIMENSIONS}
        assert classify_intervenable_state(ema) == "normal"

    def test_avoidance_after_break(self):
        """av > 0.7 AND g > 0.6 → avoidance_after_break (highest priority)."""
        ema = {"avoidance": 8, "guilt": 7, "willingness": 5, "fatigue": 5,
               "stress": 5, "procrastination": 5, "achievement": 5,
               "frustration": 5, "next_confidence": 5}
        assert classify_intervenable_state(ema) == "avoidance_after_break"

    def test_avoidance_priority_over_frustration(self):
        """Avoidance rule fires before frustration even if both match."""
        ema = {"avoidance": 8, "guilt": 7, "frustration": 7, "willingness": 5,
               "fatigue": 5, "stress": 5, "procrastination": 5, "achievement": 5,
               "next_confidence": 5}
        assert classify_intervenable_state(ema) == "avoidance_after_break"

    def test_post_exercise_frustrated(self):
        """fr > 0.6 → post_exercise_frustrated."""
        ema = {"frustration": 7, "willingness": 5, "fatigue": 5, "stress": 5,
               "procrastination": 5, "achievement": 5, "guilt": 5,
               "next_confidence": 5, "avoidance": 5}
        assert classify_intervenable_state(ema) == "post_exercise_frustrated"

    def test_high_achievement_fatigued(self):
        """a > 0.7 AND f > 0.6 → high_achievement_fatigued."""
        ema = {"achievement": 8, "fatigue": 7, "willingness": 5, "stress": 5,
               "procrastination": 5, "frustration": 5, "guilt": 5,
               "next_confidence": 5, "avoidance": 5}
        assert classify_intervenable_state(ema) == "high_achievement_fatigued"

    def test_willing_but_stuck(self):
        """w > 0.6 AND p > 0.5 → willing_but_stuck."""
        ema = {"willingness": 7, "procrastination": 6, "fatigue": 5, "stress": 5,
               "achievement": 5, "frustration": 5, "guilt": 5,
               "next_confidence": 5, "avoidance": 5}
        assert classify_intervenable_state(ema) == "willing_but_stuck"

    def test_low_mood_available(self):
        """w >= 0.4 AND s > 0.7 → low_mood_available."""
        ema = {"willingness": 5, "stress": 8, "fatigue": 5,
               "procrastination": 5, "achievement": 5, "frustration": 5,
               "guilt": 5, "next_confidence": 5, "avoidance": 5}
        assert classify_intervenable_state(ema) == "low_mood_available"

    def test_low_self_efficacy(self):
        """nc < 0.3 → low_self_efficacy."""
        ema = {"next_confidence": 2, "willingness": 5, "fatigue": 5, "stress": 5,
               "procrastination": 5, "achievement": 5, "frustration": 5,
               "guilt": 5, "avoidance": 5}
        assert classify_intervenable_state(ema) == "low_self_efficacy"

    def test_social_motivated(self):
        """a > 0.7 AND nc > 0.7 → social_motivated."""
        ema = {"achievement": 8, "next_confidence": 8, "willingness": 5,
               "fatigue": 5, "stress": 5, "procrastination": 5, "frustration": 5,
               "guilt": 5, "avoidance": 5}
        assert classify_intervenable_state(ema) == "social_motivated"

    def test_social_pressured(self):
        """av > 0.5 AND s > 0.6 → social_pressured."""
        ema = {"avoidance": 6, "stress": 7, "willingness": 5, "fatigue": 5,
               "procrastination": 5, "achievement": 5, "frustration": 5,
               "guilt": 5, "next_confidence": 5}
        assert classify_intervenable_state(ema) == "social_pressured"


class TestRecommendIntervention:
    """recommend_intervention() returns correct strategy per state."""

    def test_all_states_have_intervention(self):
        """Every intervenable state maps to an intervention."""
        from psycho_engine import INTERVENABLE_STATES, INTERVENTION_MAP
        for state in INTERVENABLE_STATES:
            result = recommend_intervention(state)
            assert result["intervenable_state"] == state
            assert "intervention_type" in result
            assert "intensity" in result
            assert "timing" in result
            assert "suggested_tone" in result

    def test_willing_but_stuck_intervention(self):
        result = recommend_intervention("willing_but_stuck")
        assert result["intervention_type"] == "lower_barrier"

    def test_avoidance_after_break_intervention(self):
        result = recommend_intervention("avoidance_after_break")
        assert result["intervention_type"] == "reset_soft"

    def test_normal_has_no_intervention(self):
        result = recommend_intervention("normal")
        assert result["intervention_type"] == "no_intervention"
        assert result["confidence"] == 0.5

    def test_non_normal_has_higher_confidence(self):
        result = recommend_intervention("low_self_efficacy")
        assert result["confidence"] == 0.7

    def test_unknown_state_falls_back_to_normal(self):
        result = recommend_intervention("non_existent_state")
        assert result["intervention_type"] == "no_intervention"

    def test_persona_hint_preserved(self):
        result = recommend_intervention("normal", persona_hint="fitness_coach")
        assert result["persona"] == "fitness_coach"


class TestPsychoStateModel:
    """PsychoState data model tests."""

    def test_fresh_creation(self):
        state = PsychoState.fresh("user_abc")
        assert state.user_id == "user_abc"
        assert state.intervenable_state == "normal"
        assert state.n_events_aggregated == 0
        assert state.ema_willingness is None

    def test_ema_dict_all_none(self):
        state = PsychoState.fresh("u1")
        ema = state.ema_dict()
        for dim in PSYCHO_DIMENSIONS:
            assert ema[dim] is None

    def test_update_from_event_first_time(self):
        """Single event populates EMA values."""
        state = PsychoState.fresh("u1")
        event = PsychoEvent(willingness=5, fatigue=7)
        state.update_from_event(event)
        assert state.ema_willingness == 5.0  # first value = raw
        assert state.ema_fatigue == 7.0
        assert state.n_events_aggregated == 1

    def test_update_from_event_ema_smoothing(self):
        """Second event applies EMA smoothing."""
        state = PsychoState.fresh("u1")
        state.update_from_event(PsychoEvent(willingness=5))
        state.update_from_event(PsychoEvent(willingness=3))
        # ema = 0.5 * 5 + 0.5 * 3 = 4.0
        assert state.ema_willingness == 4.0
        assert state.n_events_aggregated == 2

    def test_update_triggers_reclassification(self):
        """Updating with avoidance+guilt reclassifies state."""
        state = PsychoState.fresh("u1")
        state.update_from_event(PsychoEvent(avoidance=8, guilt=9))
        assert state.intervenable_state == "avoidance_after_break"

    def test_to_dict_and_from_dict_roundtrip(self):
        state = PsychoState.fresh("u1")
        state.update_from_event(PsychoEvent(willingness=7, fatigue=3))
        d = state.to_dict()
        restored = PsychoState.from_dict(d)
        assert restored.ema_willingness == state.ema_willingness
        assert restored.ema_fatigue == state.ema_fatigue
        assert restored.intervenable_state == state.intervenable_state
        assert restored.n_events_aggregated == state.n_events_aggregated

    def test_from_dict_ignores_unknown_keys(self):
        d = {"user_id": "u1", "ema_willingness": 5.0, "unknown_key": 123}
        state = PsychoState.from_dict(d)
        assert state.user_id == "u1"
        assert state.ema_willingness == 5.0
        assert not hasattr(state, "unknown_key")


# ══════════════════════════════════════════════════════════════════════════════
# Phase C — PsychoTrait + Fingerprint Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPsychoTrait:
    """PsychoTrait baseline model tests."""

    def test_default_trait_all_mid(self):
        from psycho_engine import PsychoTrait
        t = PsychoTrait(user_id="u1")
        assert t.user_id == "u1"
        for attr in ("autonomy_preference", "reward_sensitivity", "pressure_sensitivity",
                     "initial_self_efficacy", "exercise_identity", "social_exposure_pref",
                     "discomfort_baseline"):
            assert getattr(t, attr) == 0.5
        assert t.confidence == 0.1

    def test_to_dict(self):
        from psycho_engine import PsychoTrait
        t = PsychoTrait(user_id="u1")
        d = t.to_dict()
        assert d["user_id"] == "u1"
        assert d["autonomy_preference"] == 0.5


class TestComputePsychoFingerprint:
    """compute_psycho_fingerprint() tests."""

    def test_empty_events(self):
        fp = compute_psycho_fingerprint("u1", [])
        assert fp["n_events"] == 0
        assert fp["traits"]["user_id"] == "u1"
        assert fp["per_dim_stats"] == {}

    def test_single_event(self):
        events = [{"willingness": 5, "fatigue": 7, "stress": None,
                    "procrastination": None, "achievement": None,
                    "frustration": None, "guilt": None,
                    "next_confidence": None, "avoidance": None}]
        fp = compute_psycho_fingerprint("u1", events)
        assert fp["n_events"] == 1
        assert fp["has_signal_events"] == 1
        assert fp["per_dim_stats"]["willingness"]["count"] == 1
        assert fp["per_dim_stats"]["willingness"]["mean"] == 0.5
        assert fp["per_dim_stats"]["fatigue"]["count"] == 1
        assert fp["per_dim_stats"]["fatigue"]["mean"] == 0.7

    def test_multiple_events_stats(self):
        events = [
            {"willingness": 6, "fatigue": 4, "achievement": 7,
             "stress": None, "procrastination": None, "frustration": None,
             "guilt": None, "next_confidence": None, "avoidance": None},
            {"willingness": 4, "fatigue": 6, "achievement": 5,
             "stress": None, "procrastination": None, "frustration": None,
             "guilt": None, "next_confidence": None, "avoidance": None},
            {"willingness": 8, "fatigue": 3, "achievement": 9,
             "stress": None, "procrastination": None, "frustration": None,
             "guilt": None, "next_confidence": None, "avoidance": None},
        ]
        fp = compute_psycho_fingerprint("u1", events)
        assert fp["has_signal_events"] == 3
        w_mean = fp["per_dim_stats"]["willingness"]["mean"]
        assert 0.55 <= w_mean <= 0.65  # (6+4+8)/3/10 ≈ 0.6

    def test_trait_inferred_from_events(self):
        """High willingness + confidence → high autonomy + self-efficacy."""
        events = [{"willingness": 9, "next_confidence": 8, "avoidance": 2,
                    "frustration": 1, "stress": 2,
                    "fatigue": None, "procrastination": None, "achievement": None,
                    "guilt": None}]
        fp = compute_psycho_fingerprint("u1", events)
        traits = fp["traits"]
        assert traits["autonomy_preference"] >= 0.5
        assert traits["initial_self_efficacy"] >= 0.5
        assert traits["social_exposure_pref"] >= 0.4

    def test_per_dim_trend_stable(self):
        """Similar values → stable trend."""
        events = [{"willingness": 5, "fatigue": None, "stress": None,
                    "procrastination": None, "achievement": None,
                    "frustration": None, "guilt": None,
                    "next_confidence": None, "avoidance": None}] * 6
        fp = compute_psycho_fingerprint("u1", events)
        assert fp["per_dim_stats"]["willingness"]["trend"] == "stable"


# ══════════════════════════════════════════════════════════════════════════════
# Phase C — PsychoFingerprint Full Model Tests (μ, Σ, A, B, R)
# ══════════════════════════════════════════════════════════════════════════════

class TestFullFingerprintEmpty:
    """compute_full_fingerprint with empty/edge-case event lists."""

    def test_empty_events(self):
        from psycho_engine import compute_full_fingerprint
        fp = compute_full_fingerprint("u1", [])
        assert fp["user_id"] == "u1"
        assert fp["mu"] == {}
        assert fp["sigma"] == {}
        assert fp["ar_coefficients"] == {}
        assert fp["residuals"] == {}
        assert fp["n_events"] == 0
        assert fp["confidence"] == 0.1

    def test_single_event(self):
        from psycho_engine import compute_full_fingerprint
        events = [{"willingness": 7, "fatigue": 3, "stress": 4,
                    "procrastination": None, "achievement": None,
                    "frustration": None, "guilt": None,
                    "next_confidence": 6, "avoidance": 2}]
        fp = compute_full_fingerprint("u1", events)
        assert fp["n_events"] == 1
        assert fp["has_signal_events"] == 1
        # Single event → no cov/AR/residual
        assert fp["sigma"]["willingness"]["fatigue"] is None
        assert fp["ar_coefficients"]["willingness"]["coefficient"] is None
        assert fp["residuals"]["willingness"]["n"] == 0


class TestFullFingerprintMu:
    """μ (mean vector) computation."""

    def test_mu_basic(self):
        from psycho_engine import compute_full_fingerprint
        events = [{"willingness": 5, "fatigue": 3, "stress": 4}]
        fp = compute_full_fingerprint("u1", events)
        assert fp["mu"]["willingness"] == 5.0
        assert fp["mu"]["fatigue"] == 3.0
        assert fp["mu"]["stress"] == 4.0

    def test_mu_multiple_events(self):
        from psycho_engine import compute_full_fingerprint
        events = [
            {"willingness": 2, "fatigue": 8, "stress": 9},
            {"willingness": 4, "fatigue": 6, "stress": 7},
            {"willingness": 6, "fatigue": 4, "stress": 5},
        ]
        fp = compute_full_fingerprint("u1", events)
        assert fp["mu"]["willingness"] == 4.0  # (2+4+6)/3
        assert fp["mu"]["fatigue"] == 6.0       # (8+6+4)/3
        assert fp["mu"]["stress"] == 7.0        # (9+7+5)/3


class TestFullFingerprintSigma:
    """Σ (covariance matrix) computation."""

    def test_sigma_pairwise(self):
        from psycho_engine import compute_full_fingerprint
        # Positive correlation: willingness and confidence rise together
        events = [
            {"willingness": 2, "next_confidence": 2, "fatigue": 5},
            {"willingness": 5, "next_confidence": 5, "fatigue": 5},
            {"willingness": 8, "next_confidence": 8, "fatigue": 5},
        ]
        fp = compute_full_fingerprint("u1", events)
        # willingness-next_confidence should be positive cov
        assert fp["sigma"]["willingness"]["next_confidence"] > 0
        # willingness-fatigue should be ~0 (constant)
        assert abs(fp["sigma"]["willingness"]["fatigue"]) < 0.01

    def test_sigma_negative_corr(self):
        from psycho_engine import compute_full_fingerprint
        # Negative correlation: stress goes up, willingness goes down
        events = [
            {"willingness": 8, "stress": 2},
            {"willingness": 5, "stress": 5},
            {"willingness": 2, "stress": 8},
        ]
        fp = compute_full_fingerprint("u1", events)
        assert fp["sigma"]["willingness"]["stress"] < 0


class TestFullFingerprintAR:
    """A (AR(1) coefficients) computation."""

    def test_ar_autocorrelation(self):
        from psycho_engine import compute_full_fingerprint
        # Strong autocorrelation: willingness slowly changes
        events = [{"willingness": v} for v in [4, 5, 4, 5, 4, 5, 4, 5]]
        fp = compute_full_fingerprint("u1", events)
        ac = fp["ar_coefficients"]["willingness"]
        assert ac["coefficient"] is not None
        assert ac["n"] >= 6

    def test_ar_prediction_residuals(self):
        from psycho_engine import compute_full_fingerprint
        # Deterministic sequence → AR captures most variance
        events = [{"willingness": 6}] * 5
        fp = compute_full_fingerprint("u1", events)
        r = fp["residuals"]["willingness"]
        # Mean residual should be near 0 (constant value)
        if r["n"] > 0:
            assert abs(r["mean_residual"]) < 0.1


class TestFullFingerprintEmbedding:
    """B (behavioral embedding) computation."""

    def test_embedding_with_multiple_dims(self):
        from psycho_engine import compute_full_fingerprint
        events = [
            {"willingness": 7, "next_confidence": 7, "fatigue": 3, "stress": 2},
            {"willingness": 6, "next_confidence": 6, "fatigue": 4, "stress": 3},
            {"willingness": 8, "next_confidence": 8, "fatigue": 2, "stress": 2},
        ]
        fp = compute_full_fingerprint("u1", events)
        # With enough dimensions and events, should find at least one latent factor
        emb = fp["behavioral_embedding"]
        assert "latent_factors" in emb
        assert "explained_variance" in emb


class TestFullFingerprintEndToEnd:
    """End-to-end PsychoFingerprint validation."""

    def test_realistic_profile(self):
        from psycho_engine import compute_full_fingerprint
        # Simulate a "motivated but stressed" user
        events = []
        for i in range(10):
            events.append({
                "willingness": 6 + (i % 3 - 1),
                "fatigue": 7 - (i % 3),
                "stress": 8,
                "next_confidence": 5 + (i % 4),
                "avoidance": 3,
                "frustration": 2,
                "procrastination": None,
                "achievement": None,
                "guilt": None,
            })
        fp = compute_full_fingerprint("u1", events)
        # Core structure must be present
        assert len(fp["mu"]) == 9
        assert len(fp["sigma"]) == 9
        assert len(fp["ar_coefficients"]) == 9
        assert len(fp["residuals"]) == 9
        assert 0.1 <= fp["confidence"] <= 0.95
        assert fp["n_events"] == 10
        assert fp["has_signal_events"] == 10
        # Stress should have a clear mean
        assert fp["mu"]["stress"] == 8.0
        # Updated timestamp
        assert "updated_at" in fp

    def test_no_signal_events(self):
        from psycho_engine import compute_full_fingerprint
        # Events with all-None dimensions → no signal events
        events = [{"willingness": None, "fatigue": None, "stress": None}]
        fp = compute_full_fingerprint("u1", events)
        assert fp["has_signal_events"] == 0
        # Mu entries present but all None (no signal data)
        assert all(v is None for v in fp["mu"].values())


# ══════════════════════════════════════════════════════════════════════════════
# Phase C — Passive Inference Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPassiveInferenceEmpty:
    """infer_passive_psycho_signals with no data."""

    def test_no_inputs(self):
        from psycho_engine import infer_passive_psycho_signals
        result = infer_passive_psycho_signals()
        assert all(result.get(d) is None for d in
                   ["willingness", "fatigue", "stress", "procrastination",
                    "achievement", "frustration", "guilt", "next_confidence",
                    "avoidance"])
        assert result["phase"] is None

    def test_empty_dicts(self):
        from psycho_engine import infer_passive_psycho_signals
        result = infer_passive_psycho_signals(scheduler_state={},
                                               routine_anomalies=[],
                                               plan_context={})
        assert result["phase"] is None


class TestPassiveInferenceScheduler:
    """Signals from scheduler state."""

    def test_unresponded_proactives(self):
        from psycho_engine import infer_passive_psycho_signals
        result = infer_passive_psycho_signals(
            scheduler_state={"unresponded_count": 3})
        # 3 unresponded → guilt rises, willingness drops
        assert result["guilt"] is not None and result["guilt"] > 3.0
        assert result["avoidance"] is not None and result["avoidance"] > 2.0
        assert result["willingness"] is not None and result["willingness"] < 5.0
        assert result["phase"] == "passive"

    def test_no_unresponded(self):
        from psycho_engine import infer_passive_psycho_signals
        result = infer_passive_psycho_signals(
            scheduler_state={"unresponded_count": 0})
        # No unresponded → no guilt signal
        assert result["guilt"] is None

    def test_silent_mode(self):
        from psycho_engine import infer_passive_psycho_signals
        result = infer_passive_psycho_signals(
            scheduler_state={"silent_until": "2099-01-01T00:00:00",
                              "unresponded_count": 0})
        assert result["avoidance"] is not None and result["avoidance"] >= 5.0

    def test_silent_mode_schema_clean(self):
        """silent mode keeps the result schema-clean (no non-dimension stray keys).

        Regression guard: the output contract only allows PSYCHO_DIMENSIONS keys
        plus phase/reason; a stray key (e.g. the removed "social_pressured") would
        be dropped by downstream _normalize_psycho_extracted / PsychoEvent and
        only pollute the intermediate dict.
        """
        from psycho_engine import infer_passive_psycho_signals, PSYCHO_DIMENSIONS
        result = infer_passive_psycho_signals(
            scheduler_state={"silent_until": "2099-01-01T00:00:00",
                              "unresponded_count": 0})
        assert result["avoidance"] is not None and result["avoidance"] >= 5.0
        assert "social_pressured" not in result
        assert set(result.keys()) == set(PSYCHO_DIMENSIONS) | {"phase", "reason"}
        assert result["phase"] == "passive"


class TestPassiveInferenceRoutine:
    """Signals from routine anomalies."""

    def test_sleep_anomaly(self):
        from psycho_engine import infer_passive_psycho_signals
        result = infer_passive_psycho_signals(
            routine_anomalies=[{"dimension": "sleep", "severity": "high"}])
        assert result["fatigue"] is not None and result["fatigue"] >= 5.0

    def test_exercise_anomaly(self):
        from psycho_engine import infer_passive_psycho_signals
        result = infer_passive_psycho_signals(
            routine_anomalies=[{"dimension": "exercise", "severity": "medium"}])
        assert result["frustration"] is not None and result["frustration"] >= 2.0

    def test_mood_anomaly(self):
        from psycho_engine import infer_passive_psycho_signals
        result = infer_passive_psycho_signals(
            routine_anomalies=[{"dimension": "mood", "severity": "high"}])
        assert result["stress"] is not None and result["stress"] >= 5.0

    def test_multiple_anomalies(self):
        from psycho_engine import infer_passive_psycho_signals
        result = infer_passive_psycho_signals(
            routine_anomalies=[
                {"dimension": "sleep", "severity": "high"},
                {"dimension": "exercise", "severity": "low"},
            ])
        assert result["fatigue"] >= 5.0   # sleep anomaly
        assert result["frustration"] >= 1.0  # exercise anomaly


class TestPassiveInferencePlan:
    """Signals from plan behavior."""

    def test_plan_modifications(self):
        from psycho_engine import infer_passive_psycho_signals
        result = infer_passive_psycho_signals(
            plan_context={"modifications": 3, "completions": 0, "skips": 0})
        assert result["frustration"] is not None and result["frustration"] >= 3.0
        assert result["procrastination"] is not None

    def test_plan_completions(self):
        from psycho_engine import infer_passive_psycho_signals
        result = infer_passive_psycho_signals(
            plan_context={"modifications": 0, "completions": 3, "skips": 0})
        assert result["achievement"] is not None and result["achievement"] >= 3.0
        assert result["next_confidence"] is not None

    def test_plan_skips(self):
        from psycho_engine import infer_passive_psycho_signals
        result = infer_passive_psycho_signals(
            plan_context={"modifications": 0, "completions": 0, "skips": 3})
        assert result["guilt"] is not None and result["guilt"] >= 2.0
        assert result["procrastination"] is not None

    def test_mixed_plan_signals(self):
        from psycho_engine import infer_passive_psycho_signals
        # User completed some, modified some, skipped some
        result = infer_passive_psycho_signals(
            plan_context={"modifications": 2, "completions": 5, "skips": 1})
        # Achievement should dominate (5 completions)
        assert result["achievement"] >= 5.0
        # Low skips → low guilt
        assert result["guilt"] is None or result["guilt"] < 3.0
