"""Viora — Interactive reasoning-tree info-gathering tests.

Covers ai_engine.ask_followup_question() and expand_reasoning_node(user_answer=...):
  - ask_followup_question returns (question, options) when the LLM says one is needed
  - ask_followup_question returns (None, None) when the LLM says expansion can proceed
  - ask_followup_question handles a null / missing followup_question field
  - ask_followup_question normalizes/caps the structured options array
  - expand_reasoning_node injects the user answer into the child prompt
"""

import sys
import json
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from ai_engine import ask_followup_question, expand_reasoning_node  # noqa: E402


NODE = {
    "id": "n1",
    "label": "饮食因素",
    "description": "饮食相关的健康影响因素",
    "depth": 1,
    "parent_id": "root",
}

USER_CONTEXT = {
    "concern": "脱发",
    "routine_summary": "作息不规律",
    "profile_notes": "程序员",
    "insights": "",
    "weather_context": "",
}


class TestAskFollowupQuestion:
    def test_returns_question_when_needed(self):
        """LLM returns a followup_question → function returns (question, options)."""
        llm_response = json.dumps({
            "followup_question": "这种情况持续多久了？",
            "options": ["已持续1个月", "半年以上", "不清楚"],
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            q, options = ask_followup_question(NODE, USER_CONTEXT)
        assert q == "这种情况持续多久了？"
        assert options == ["已持续1个月", "半年以上", "不清楚"]

    def test_returns_none_when_not_needed(self):
        """LLM returns followup_question=null → function returns (None, None)."""
        llm_response = json.dumps({"followup_question": None, "options": None})
        with patch("ai_engine._call_llm", return_value=llm_response):
            q, options = ask_followup_question(NODE, USER_CONTEXT)
        assert q is None
        assert options is None

    def test_returns_none_when_field_missing(self):
        """LLM returns an object without followup_question → (None, None)."""
        llm_response = json.dumps({"something_else": 1})
        with patch("ai_engine._call_llm", return_value=llm_response):
            q, options = ask_followup_question(NODE, USER_CONTEXT)
        assert q is None
        assert options is None

    def test_returns_none_on_llm_error(self):
        """LLM failure → (None, None) (graceful degradation, never blocks expansion)."""
        with patch("ai_engine._call_llm", side_effect=Exception("boom")):
            q, options = ask_followup_question(NODE, USER_CONTEXT)
        assert q is None
        assert options is None

    def test_options_normalized_and_capped(self):
        """options are stripped, empties dropped, and capped at 6 entries."""
        llm_response = json.dumps({
            "followup_question": "你是如何察觉的？",
            "options": [" 自己感觉 ", "   ", "体检发现", "家人提醒", "o1", "o2", "o3", "o4", "o5"],
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            q, options = ask_followup_question(NODE, USER_CONTEXT)
        assert q == "你是如何察觉的？"
        assert len(options) == 6
        assert options[0] == "自己感觉"
        assert "   " not in options

    def test_options_deduplicated_preserving_order(self):
        """options are deduplicated (LLM may emit repeated/near-identical choices)."""
        llm_response = json.dumps({
            "followup_question": "你察觉多久了？",
            "options": ["一个月", "一个月", "半年", " 一个月 ", "不清楚", "半年"],
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            q, options = ask_followup_question(NODE, USER_CONTEXT)
        assert q == "你察觉多久了？"
        assert options == ["一个月", "半年", "不清楚"]

    def test_legacy_question_only_response(self):
        """Backward compat: LLM returns only followup_question (no options) → options None."""
        llm_response = json.dumps({"followup_question": "是或否？"})
        with patch("ai_engine._call_llm", return_value=llm_response):
            q, options = ask_followup_question(NODE, USER_CONTEXT)
        assert q == "是或否？"
        assert options is None

    def test_options_skips_structured_junk(self):
        """LLM emitting structured objects/numbers as options is filtered cleanly."""
        llm_response = json.dumps({
            "followup_question": "你通常什么时候不舒服？",
            "options": [
                {"label": "早餐后"},   # dict → dropped, not str()-repr junk
                ["午餐后"],            # list → dropped
                " 晚餐后 ",           # kept (stripped)
                3,                     # scalar → "3"
                "   ",                 # empty → dropped
                None,                  # dropped
            ],
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            q, options = ask_followup_question(NODE, USER_CONTEXT)
        assert q == "你通常什么时候不舒服？"
        assert options == ["晚餐后", "3"]


class TestExpandUserAnswerInjection:
    def test_user_answer_is_injected_into_prompt(self):
        """expand_reasoning_node passes user_answer into the child expand prompt."""
        llm_response = json.dumps([{
            "label": "皮质醇升高",
            "description": "压力导致皮质醇水平上升",
            "emoji": "😰",
            "is_actionable": False,
        }])
        captured = {}

        def fake_call_llm(sys_prompt, prompt, time_ctx):
            captured["prompt"] = prompt
            return llm_response

        with patch("ai_engine._call_llm", side_effect=fake_call_llm):
            children = expand_reasoning_node(
                NODE,
                USER_CONTEXT,
                user_answer="A. 已持续半年以上",
            )

        assert len(children) == 1
        assert children[0]["label"] == "皮质醇升高"
        assert "A. 已持续半年以上" in captured["prompt"]

    def test_no_user_answer_uses_empty_section(self):
        """Without user_answer, the prompt's answer section is empty but renders."""
        llm_response = json.dumps([{
            "label": "睡眠不足",
            "description": "睡眠时长不足",
            "emoji": "🛌",
            "is_actionable": False,
        }])
        captured = {}

        def fake_call_llm(sys_prompt, prompt, time_ctx):
            captured["prompt"] = prompt
            return llm_response

        with patch("ai_engine._call_llm", side_effect=fake_call_llm):
            children = expand_reasoning_node(NODE, USER_CONTEXT)

        assert len(children) == 1
        # The "用户对追问的回答" section header must still be present.
        assert "用户对追问的回答" in captured["prompt"]

    def test_expandable_derived_from_actionable(self):
        """expandable is derived at the source: actionable→False, non-actionable→True.

        The expand prompts don't emit an "expandable" field, so the invariant
        (actionable leaves are terminal) is enforced in expand_reasoning_node
        itself — direct callers of expand_node() get a consistent value too.
        """
        llm_response = json.dumps([
            {"label": "早餐全麦面包", "description": "低GI方案", "emoji": "🍞",
             "is_actionable": True, "action_text": "早餐全麦面包替代白面包",
             "time_slot": "早餐", "frequency": "每天"},
            {"label": "睡眠不足", "description": "睡眠时长不足", "emoji": "🛌",
             "is_actionable": False},
        ])
        with patch("ai_engine._call_llm", return_value=llm_response):
            children = expand_reasoning_node(NODE, USER_CONTEXT)

        assert len(children) == 2
        actionable = next(c for c in children if c["is_actionable"])
        non_actionable = next(c for c in children if not c["is_actionable"])
        assert actionable["expandable"] is False
        assert non_actionable["expandable"] is True
