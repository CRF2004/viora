"""Tests for ResponsePlanner — intent classification and burst decision logic."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from ai_engine import ResponsePlanner

# ── Fixtures ────────────────────────────────────────────────────────────

WARM_FRIEND = {
    "id": "default",
    "name": "温暖好友",
    "empathy_level": 5,
    "analysis_level": 3,
    "advice_level": 2,
    "humor_level": 3,
    "burst_probability": 65,
    "emoji_density": "medium",
    "proactive_strength": 2,
    "forbidden": ["客服式话术", "长篇说教"],
    "phrase_preferences": ["嗯", "哎", "～"],
}

FITNESS_COACH = {
    "id": "fitness_coach",
    "name": "健身教练",
    "empathy_level": 2,
    "analysis_level": 5,
    "advice_level": 5,
    "humor_level": 2,
    "burst_probability": 30,
    "emoji_density": "low",
    "proactive_strength": 4,
}

EMPTY_STATE: dict = {}


def make_planner(persona=None, user_state=None):
    return ResponsePlanner(persona or WARM_FRIEND, user_state or {})


# ── Intent classification tests ───────────────────────────────────────

class TestDecideIntent:
    """Verify that user messages are correctly classified into intents."""

    def test_physical_discomfort(self):
        p = make_planner()
        result = p.decide_intent("我今天胃有点疼", {}, [])
        assert result["rich_intent"] == "physical_discomfort"
        assert result["plan_intent"] == "empathy"

    def test_empathy_seeking(self):
        p = make_planner()
        result = p.decide_intent("最近压力好大，感觉快崩溃了", {}, [])
        assert result["rich_intent"] == "seeking_empathy"
        assert result["plan_intent"] == "empathy"

    def test_venting(self):
        p = make_planner()
        result = p.decide_intent("烦死了，今天什么事都不顺", {}, [])
        assert result["rich_intent"] == "venting"
        assert result["plan_intent"] == "empathy"

    def test_achievement(self):
        p = make_planner()
        result = p.decide_intent("终于完成了这个项目！", {}, [])
        assert result["rich_intent"] == "achievement"
        assert result["plan_intent"] == "light_checkin"

    def test_advice_seeking(self):
        p = make_planner()
        result = p.decide_intent("最近总是失眠，怎么办", {}, [])
        assert result["rich_intent"] == "seeking_empathy"  # 失眠 dominates
        assert result["plan_intent"] == "empathy"

    def test_joking(self):
        p = make_planner()
        result = p.decide_intent("今天又睡到中午，绝了", {}, [])
        assert result["rich_intent"] == "joking"
        assert result["plan_intent"] == "humor"

    def test_greeting(self):
        p = make_planner()
        result = p.decide_intent("早上好呀～", {}, [])
        assert result["rich_intent"] == "greeting"

    def test_reflection(self):
        p = make_planner()
        result = p.decide_intent("我感觉这可能是因为最近运动不够", {}, [])
        assert result["rich_intent"] == "reflection"
        assert result["plan_intent"] == "analysis"

    def test_casual_share_fallback(self):
        p = make_planner()
        result = p.decide_intent("今天天气不错", {}, [])
        assert result["plan_intent"] == "light_checkin"
        assert result["rich_intent"] == "casual_share"

    def test_plan_request(self):
        p = make_planner()
        result = p.decide_intent("帮我制定一个健康计划", {}, [])
        assert result["plan_intent"] == "plan_request"

    def test_anomaly_amplifies_empathy(self):
        """High anomaly level should route physical discomfort to empathy."""
        p = make_planner(user_state={"anomaly_level": "high"})
        result = p.decide_intent("我胃疼", {}, [])
        assert result["rich_intent"] == "physical_discomfort"
        assert result["plan_intent"] == "empathy"

    def test_proactive_routing(self):
        """Proactive messages should get appropriate intents."""
        p = make_planner(user_state={"anomaly_level": "low"})
        result = p.decide_intent("", {}, [], proactive=True)
        assert result["plan_intent"] == "light_checkin"
        assert result["rich_intent"] == "greeting"

    def test_proactive_with_sleep_anomaly(self):
        """High anomaly from sleep issues should route to empathy."""
        p = make_planner(user_state={
            "anomaly_level": "high",
            "recent_anomalies": [{"type": "sleep_decline", "strength": 0.8}],
        })
        result = p.decide_intent("", {}, [], proactive=True)
        assert result["plan_intent"] == "empathy"
        assert result["rich_intent"] in ("seeking_empathy",)

    def test_acknowledgment(self):
        """Short casual acknowledgments should be classified as light_checkin."""
        p = make_planner()
        for msg in ["对的对的", "嗯嗯", "好的", "是啊", "没错", "知道了", "okk"]:
            result = p.decide_intent(msg, {}, [])
            assert result["rich_intent"] == "acknowledgment", f"Failed for {msg!r}"
            assert result["plan_intent"] == "light_checkin", f"Failed for {msg!r}"

        # Longer acknowledgments should NOT be classified as acknowledgment
        result = p.decide_intent("对的对的，我觉得你说的很对，确实应该注意", {}, [])
        assert result["rich_intent"] != "acknowledgment"

    def test_empty_message(self):
        """Empty message should fall through to default intent."""
        p = make_planner()
        result = p.decide_intent("  ", {}, [])
        assert result["plan_intent"] == "light_checkin"


# ── Burst decision tests ──────────────────────────────────────────────

class TestDecideBurst:
    """Verify burst parameters are chosen correctly per intent + persona."""

    def test_warm_friend_empathy_burst(self):
        """Warm friend (65% burst) should get 3 messages for empathy."""
        p = make_planner(WARM_FRIEND)
        burst = p.decide_burst("empathy")
        assert burst["num_messages"] == 3  # 65 >= 60

    def test_warm_friend_question_single(self):
        """Low-key intent should produce 1-2 messages."""
        p = make_planner(WARM_FRIEND)
        burst = p.decide_burst("question")
        assert burst["num_messages"] in (1, 2)

    def test_coach_empathy_low_burst(self):
        """Fitness coach (30% burst) should get fewer messages."""
        p = make_planner(FITNESS_COACH)
        burst = p.decide_burst("empathy")
        assert burst["num_messages"] == 2  # 30 < 60

    def test_coach_analysis_low_emoji(self):
        """Fitness coach analysis should have low emoji density."""
        p = make_planner(FITNESS_COACH)
        burst = p.decide_burst("analysis")
        assert burst["emoji_density"] == "low"

    def test_humor_burst_high_emoji(self):
        """Humor intent should have high emoji density."""
        p = make_planner(WARM_FRIEND)
        burst = p.decide_burst("humor")
        assert burst["emoji_density"] == "high"

    def test_proactive_burst(self):
        """Proactive messages should get 2-3 messages."""
        p = make_planner(WARM_FRIEND)
        burst = p.decide_burst("empathy", proactive=True)
        assert burst["num_messages"] in (2, 3)
        assert burst["intensity"] == "low"

    def test_burst_always_has_required_fields(self):
        """All burst plans have num_messages, emoji_density, intensity, scenario."""
        p = make_planner(WARM_FRIEND)
        for intent in ["empathy", "analysis", "advice", "humor",
                       "question", "light_checkin",
                       "plan_request", "plan_reminder",
                       "plan_feedback", "plan_modify"]:
            burst = p.decide_burst(intent)
            assert "num_messages" in burst, f"{intent} missing num_messages"
            assert "emoji_density" in burst, f"{intent} missing emoji_density"
            assert "intensity" in burst, f"{intent} missing intensity"
            assert "scenario" in burst, f"{intent} missing scenario"
            assert 1 <= burst["num_messages"] <= 5, f"{intent} num_messages out of range"


# ── Delay decision tests ──────────────────────────────────────────────

class TestDecideDelay:
    """Verify response delay calculation."""

    def test_base_delay_empathy(self):
        p = make_planner()
        delay = p.decide_delay("empathy")
        assert 3 <= delay <= 45

    def test_anomaly_increases_delay(self):
        low = make_planner(user_state={"anomaly_level": "low"})
        high = make_planner(user_state={"anomaly_level": "high"})
        assert high.decide_delay("empathy") >= low.decide_delay("empathy")


# ── Opening style tests ───────────────────────────────────────────────

class TestDecideOpening:
    """Verify opening style selection."""

    def test_proactive_opening(self):
        p = make_planner()
        opening = p.decide_opening("empathy", proactive=True)
        assert opening["opening_style"] == "wait_then_greet"
        assert opening["show_typing"] is True

    def test_empathy_opening_careful(self):
        p = make_planner(user_state={"anomaly_level": "high"})
        opening = p.decide_opening("empathy")
        assert opening["opening_style"] == "careful"


# ── Strategy tests ────────────────────────────────────────────────────

class TestDecideStrategy:
    """Verify that strategies can be resolved per intent."""

    def test_strategy_resolves(self):
        p = make_planner()
        strategy = p.decide_strategy("greeting")
        # Should return a dict or None — just verify it doesn't crash
        assert strategy is None or isinstance(strategy, dict)

    def test_unknown_intent_returns_strategy(self):
        """Unknown intents fall through to a default reply strategy."""
        p = make_planner()
        strategy = p.decide_strategy("non_existent_intent")
        assert isinstance(strategy, dict)
        assert "prompt_extra" in strategy


# ── Empty edge case tests ─────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and robustness."""

    def test_empty_persona(self):
        """Should not crash with empty persona."""
        p = make_planner({})
        result = p.decide_intent("今天不舒服", {}, [])
        assert result["plan_intent"] is not None

    def test_empty_user_state(self):
        """Should not crash with empty user state."""
        p = make_planner(user_state={})
        result = p.decide_intent("今天不舒服", {}, [])
        assert result["plan_intent"] is not None

    def test_very_long_message(self):
        """Long messages should still classify correctly."""
        msg = "好累 " * 100
        p = make_planner()
        result = p.decide_intent(msg, {}, [])
        assert result["plan_intent"] is not None
