"""
ai_engine.py — LLM-powered health data extraction and response generation.

All LLM calls go through this module so the backend can be swapped later.
Currently uses the OpenAI-compatible chat API (supports Ollama, OpenAI, etc.).
"""

import json
import logging
import re
import time
import traceback
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from uuid import uuid4

import requests  # type: ignore

from config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL, LLM_TIMEOUT
from prompt_tree import (
    build_chat_system_prompt as _pt_system,
    build_chat_user_prompt as _pt_user,
    build_extract_prompt as _pt_extract,
    build_story_prompt as _pt_story,
    build_profile_extract_prompt as _pt_profile,
    build_plan_expand_root_prompt as _pt_expand_root,
    build_plan_expand_child_prompt as _pt_expand_child,
    build_plan_followup_prompt as _pt_followup,
    build_plan_reminder_extract_prompt as _pt_reminder,
    build_plan_feedback_prompt as _pt_feedback,
    PromptTrace,
    get_persona_overlay,
)
from prompt_traces import save_trace

logger = logging.getLogger(__name__)

# ── Internal helpers ───────────────────────────────────────────────────────

# China Standard Time (UTC+8)
CST = timezone(timedelta(hours=8))

def _current_hour() -> int:
    """Return the current hour (0-23) in China Standard Time."""
    return datetime.now(CST).hour
def _get_time_context() -> str:
    """
    Generate a time context string to inject into LLM prompts.
    Helps the AI be aware of the current time so it doesn't give
    nonsensical advice like 'go to bed at 11pm' when it's already 11:42pm.
    """
    now = datetime.now(CST)
    hour = now.hour
    weekday_names = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
    weekday = weekday_names[now.weekday()]

    # Determine period of day
    if 0 <= hour < 6:
        period = "凌晨"
        period_hint = "多数人处于睡眠中。"
    elif 6 <= hour < 9:
        period = "早晨"
        period_hint = "刚起床或即将起床的时间。"
    elif 9 <= hour < 12:
        period = "上午"
        period_hint = "工作时间。"
    elif 12 <= hour < 14:
        period = "中午"
        period_hint = "午休和午餐时间。"
    elif 14 <= hour < 18:
        period = "下午"
        period_hint = "工作日下午。"
    elif 18 <= hour < 21:
        period = "傍晚"
        period_hint = "晚饭和休闲时间。"
    elif 21 <= hour < 23:
        period = "晚上"
        period_hint = "晚间时段，尚未到入睡时间。"
    else:
        period = "深夜"
        period_hint = "多数人已入睡的深夜时段。"

    return (
        f"当前时间：{now.strftime('%Y-%m-%d')} {now.strftime('%H:%M')}（{weekday}）\n"
        f"时间段：{period}\n"
        f"时段提示：{period_hint}"
    )
def _call_llm(
    system_prompt: str,
    user_prompt: str,
    time_context: str = "",
    max_tokens: int = 2048,
    trace_route: Optional[list[str]] = None,
    trace_vars: Optional[dict] = None,
    trace_planner: Optional[dict] = None,
    trace_message_id: Optional[str] = None,
) -> str:
    """
    Call the LLM via OpenAI-compatible chat API.
    Returns the assistant message content.
    If time_context is provided, it is appended to the system prompt.

    If trace_route is provided, a PromptTrace is recorded after the call
    for debugging and prompt optimization.
    """
    _t_start = time.time()
    if time_context:
        system_prompt = system_prompt + "\n\n" + time_context
    url = f"{LLM_API_BASE.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }

    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            message = data["choices"][0]["message"]
            content = (message.get("content") or "").strip()
            if not content:
                content = (message.get("reasoning_content") or "").strip()
            if content:
                _latency = int((time.time() - _t_start) * 1000)
                # Record prompt trace if route provided
                if trace_route and trace_message_id:
                    try:
                        trace = PromptTrace(
                            message_id=trace_message_id,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            route=[r for r in trace_route if r],  # Filter empty
                            variables=trace_vars or {},
                            planner=trace_planner,
                            raw_response=content,
                            model=LLM_MODEL,
                            latency_ms=_latency,
                        )
                        save_trace(trace)
                    except Exception as _te:
                        logger.warning("Failed to save prompt trace: %s", _te)
                return content
            if attempt < max_attempts - 1:
                logger.warning(
                    "LLM returned empty content (attempt %d), retrying with +512 max_tokens",
                    attempt + 1,
                )
                payload["max_tokens"] = max_tokens + 512
                continue
            logger.error("LLM returned empty content after %d attempts", max_attempts)
            raise LLMError("LLM returned empty content")
        except requests.RequestException as e:
            if attempt < max_attempts - 1:
                logger.warning(
                    "LLM call failed (attempt %d): %s, retrying...",
                    attempt + 1, e,
                )
                continue
            logger.error("LLM call failed after %d attempts: %s", max_attempts, e)
            raise LLMError(f"LLM request failed after {max_attempts} attempts: {e}") from e
        except (KeyError, IndexError) as e:
            if attempt < max_attempts - 1:
                logger.warning(
                    "Unexpected LLM response (attempt %d): %s, retrying...",
                    attempt + 1, e,
                )
                continue
            logger.error("Unexpected LLM response after %d attempts: %s", max_attempts, e)
            raise LLMError(f"Unexpected LLM response after {max_attempts} attempts: {e}") from e
def _parse_json_response(text: str) -> dict | list:
    """
    Try to parse JSON from an LLM response, handling various formats:
    - Raw JSON (object or array)
    - Markdown code fences (```json ... ```)
    - Extra text before/after JSON
    - Leading/trailing whitespace
    - Trailing commas (common LLM issue)
    """
    import re

    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        # Find the last ``` to handle multi-line code blocks
        if "```" in text[3:]:
            text = text[3:]
            text = text[:text.rfind("```")].strip()
            # Remove language specifier if present (e.g., "json\n")
            if "\n" in text:
                first_line = text.split("\n")[0].strip()
                if first_line.isalpha():
                    text = text[len(first_line):].strip()
        else:
            text = text[3:].strip()

    # Try to find JSON object boundaries (for dict responses)
    obj_start = text.find("{")
    obj_end = text.rfind("}") + 1

    # Try to find JSON array boundaries (for list responses)
    arr_start = text.find("[")
    arr_end = text.rfind("]") + 1

    # Prefer object if both found and object comes first
    if obj_start >= 0 and obj_end > obj_start:
        if arr_start >= 0 and arr_start < obj_start and arr_end > arr_start:
            # Array starts before object — likely a JSON array response
            text = text[arr_start:arr_end]
        else:
            text = text[obj_start:obj_end]
    elif arr_start >= 0 and arr_end > arr_start:
        text = text[arr_start:arr_end]
    elif obj_start >= 0 and obj_end > obj_start:
        text = text[obj_start:obj_end]

    # Remove trailing commas before ] or } (common LLM JSON issue)
    text = re.sub(r',\s*([\]}])', r'\1', text)

    return json.loads(text)
# ── Public API ──────────────────────────────────────────────────────────────

class LLMError(Exception):
    """Raised when the LLM call fails."""
def extract_health_data(message: str) -> dict:
    """
    Extract structured health data from a user message using the LLM.

    Returns a dict matching the extracted_data schema:
    {
        "sleep": {"quality": int|None, "duration_hours": float|None},
        "energy": {"score": int|None, "reason": str|None},
        "mood": {"score": int|None, "reason": str|None},
        "exercise": {"type": str|None, "duration_minutes": int|None},
        "digestion": {"status": str|None, "severity": int|None},
        "skin": {"status": str|None},
        "pain": {"location": str|None, "severity": int|None},
        "diet": {"notes": str|None},
    }
    """
    prompt = _pt_extract(message)
    time_ctx = _get_time_context()
    result = _call_llm("You are a health data extraction assistant. Return only valid JSON.", prompt, time_ctx)
    logger.info("Raw extraction response: %s", repr(result[:500]))
    try:
        raw = _parse_json_response(result)
        return normalize_extracted(raw)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM extraction JSON, returning empty. Raw: %s", result[:200])
        # Return a template with all nulls so the caller always gets a valid structure
        return _empty_extracted()
def _normalize_burst_list(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        line = line.strip().strip('"').strip("'")
        if line:
            cleaned.append(line)
    return cleaned
def _split_proactive_fallback(text: str) -> list[str]:
    text = (text or "").strip().strip('"').strip("'").strip()
    if not text:
        return []

    segments = [seg.strip(" ，。！？；：,.!?") for seg in re.split(r"(?<=[。！？；!?])", text) if seg.strip()]
    if len(segments) <= 1:
        segments = [seg.strip(" ，。！？；：,.!?") for seg in re.split(r"[，；：,;]", text) if seg.strip()]

    cleaned: list[str] = []
    for seg in segments:
        if not seg:
            continue
        if len(seg) > 25:
            parts = [part.strip(" ，。！？；：,.!?") for part in re.split(r"[，；：,;]", seg) if part.strip()]
            cleaned.extend(parts or [seg])
        else:
            cleaned.append(seg)

    compact = [seg for seg in cleaned if seg]
    if len(compact) < 2:
        return [text]
    return compact[:3]

def _is_acknowledgment(text: str) -> bool:
    """Detect short casual acknowledgments that carry no new information.

    Examples: "对的对的", "嗯嗯", "好的", "是啊", "没错", "了解了", "知道了", "好滴", "okk"
    """
    text = text.strip().lower()
    if len(text) > 12:
        return False
    # Single-word or short-phrase acknowledgments
    ack_patterns = [
        "对的对的", "对对对", "对的", "对对",
        "嗯嗯", "嗯", "恩恩", "昂",
        "好的", "好滴", "好哒", "好的呢", "好呀", "好哦", "好",
        "是啊", "是的", "是的是的", "对呀",
        "没错", "没错没错", "也是",
        "了解了", "知道了", "懂了", "明白了", "收到", "get",
        "okk", "ok", "okay", "k",
        "行", "行吧", "可以", "可以的",
        "哈哈哈哈", "哈哈", "hhhh",
        "👍", "👌", "🙏", "❤️", "😊",
    ]
    # Exact match on stripped text
    if text in ack_patterns:
        return True
    # Also match patterns that are just repetitions of the same character
    # e.g. "对对对对", "嗯嗯嗯嗯"
    if len(text) >= 2 and len(set(text)) <= 2:
        # Check if it looks like a repetition pattern
        unique_chars = set(text)
        if unique_chars <= {"对", "嗯", "是", "好", "哈", "h", "a", "!", "？", "?", "~"}:
            return True
    return False

# ── Intent classification constants ──────────────────────────────────────

# Rich intent categories (for strategy selection)
RICH_INTENTS = [
    "venting", "seeking_empathy", "physical_discomfort",
    "seeking_advice", "casual_share", "acknowledgment",
    "achievement", "joking", "reflection", "greeting", "other",
]

# Map: rich_intent → plan_intent (for burst/delay/opening backward compat)
_RICH_TO_PLAN = {
    "venting": "empathy",
    "seeking_empathy": "empathy",
    "physical_discomfort": "empathy",
    "seeking_advice": "advice",
    "casual_share": "light_checkin",
    "acknowledgment": "light_checkin",
    "achievement": "light_checkin",
    "joking": "humor",
    "reflection": "analysis",
    "greeting": "light_checkin",
    "other": "question",
}
class ResponsePlanner:
    """Rule-based decision layer for chat burst planning and intent routing."""

    def __init__(self, persona: dict, user_state: Optional[dict] = None):
        self.persona = persona or {}
        self.user_state = user_state or {}

    def _routine_context(self) -> dict:
        return dict((self.user_state or {}).get("routine_summary") or {})

    def decide_delay(self, intent: str, proactive: bool = False) -> int:
        routine = self._routine_context()
        anomaly = (self.user_state or {}).get("anomaly_level", "low")
        active_hour = int(routine.get("active_hour", -1) or -1)
        current_hour = _current_hour()
        familiarity = int((self.user_state or {}).get("interaction_count", 0) or 0)

        if proactive:
            base = 18
        elif intent in {"empathy", "analysis"}:
            base = 12
        elif intent in {"advice", "humor"}:
            base = 8
        else:
            base = 5

        if anomaly in {"medium", "high"}:
            base += 4
        if familiarity < 4:
            base += 3
        if active_hour >= 0 and abs(current_hour - active_hour) >= 6:
            base += 4
        return max(3, min(base, 45))

    def decide_opening(self, intent: str, proactive: bool = False) -> dict:
        persona = self.persona
        anomaly = (self.user_state or {}).get("anomaly_level", "low")
        if proactive:
            return {"show_typing": True, "typing_delay": self.decide_delay(intent, proactive=True), "opening_style": "wait_then_greet"}
        if anomaly in {"medium", "high"} or intent == "empathy":
            return {"show_typing": True, "typing_delay": self.decide_delay(intent), "opening_style": "careful"}
        if persona.get("burst_probability", 50) < 45:
            return {"show_typing": True, "typing_delay": self.decide_delay(intent), "opening_style": "calm"}
        return {"show_typing": True, "typing_delay": self.decide_delay(intent), "opening_style": "natural"}

    # ── Rich intent classification ─────────────────────────────────────
    # Returns one of the RICH_INTENTS for fine-grained strategy selection.

    def _classify_rich_intent(self, message: str, extracted: dict,
                               context: list, proactive: bool = False) -> str:
        """Classify user message into a rich intent category."""
        text = (message or "").lower()
        anomaly = (self.user_state or {}).get("anomaly_level", "low")

        if proactive:
            return "greeting"

        # High anomaly → likely physical or emotional distress
        if anomaly in {"high", "medium"}:
            # Check if physical keywords dominate
            if any(k in text for k in ["疼", "痛", "难受", "胃", "恶心", "头晕", "感冒", "发烧"]):
                return "physical_discomfort"
            return "seeking_empathy"

        # ── Physical discomfort ─────────────────────────────────────
        if any(k in text for k in ["疼", "痛", "不舒服", "胃", "恶心",
                                     "头晕", "感冒", "发烧", "咳嗽", "过敏",
                                     "拉肚子", "胀气", "胸闷"]):
            return "physical_discomfort"

        # ── Seeking empathy (deep emotional distress) ───────────────
        if any(k in text for k in ["伤心", "难过", "低落", "崩溃", "抑郁",
                                     "焦虑", "压力大", "受不了", "心累",
                                     "没意思", "好难过", "好难受", "想哭", "孤独",
                                     "睡不着", "失眠", "没精神", "提不起劲"]):
            return "seeking_empathy"

        # ── Venting (frustration, minor complaint) ─────────────────
        if any(k in text for k in ["烦死了", "无语", "太坑了", "倒霉",
                                     "气死", "什么鬼", "真烦", "好烦",
                                     "受不了", "讨厌"]):
            return "venting"

        # ── Achievement / pride ────────────────────────────────────
        if any(k in text for k in ["成功", "完成了", "做到了", "终于",
                                     "坚持了", "打卡", "进步了", "突破了",
                                     "达成", "搞定了", "学会了"]):
            return "achievement"

        # ── Seeking advice ─────────────────────────────────────────
        if any(k in text for k in ["建议", "怎么办", "要不要", "该不该",
                                     "如何", "有没有办法", "怎么", "支个招",
                                     "推荐"]):
            return "seeking_advice"

        # ── Joking / playful ───────────────────────────────────────
        if any(k in text for k in ["哈哈", "笑死", "离谱", "搞笑", "整活",
                                     "绝了", "笑不活了"]):
            return "joking"

        # ── Acknowledgment / light affirmation ─────────────────────
        # Short casual responses that don't carry new information.
        # Must check BEFORE greeting/reflection since these are even
        # lower-signal.
        if _is_acknowledgment(text):
            return "acknowledgment"

        # ── Greeting ───────────────────────────────────────────────
        if any(k in text for k in ["嗨", "hello", "hi", "你好", "在吗",
                                     "早上好", "晚上好", "好久不见", "hey"]):
            return "greeting"

        # ── Reflection ─────────────────────────────────────────────
        if any(k in text for k in ["我发现", "我意识到", "反思", "想了想",
                                     "感觉", "可能是", "原因", "其实"]):
            return "reflection"

        # ── Plan-related intents (unchanged, mapped directly) ──────
        # These bypass the rich intent system and go straight to plan_intent.
        return None  # Signal to fall through to plan_intent detection

    def _map_to_plan_intent(self, rich_intent: str) -> str:
        return _RICH_TO_PLAN.get(rich_intent, "question")

    # ── Combined intent method ─────────────────────────────────────────
    # Returns dict with plan_intent (for burst/delay/opening) and
    # rich_intent (for strategy selection).

    def decide_intent(self, message: str, extracted: dict,
                       context: list, proactive: bool = False) -> dict:
        """Classify message intent.

        Returns:
            dict with:
              - plan_intent (str): legacy intent for burst/delay/opening
              - rich_intent (str): fine-grained intent for strategy selection
        """
        text = (message or "").lower()
        anomaly = (self.user_state or {}).get("anomaly_level", "low")

        # First try rich intent classification
        if not proactive:
            rich = self._classify_rich_intent(message, extracted, context, proactive)
            if rich is not None:
                plan = self._map_to_plan_intent(rich)
                return {"plan_intent": plan, "rich_intent": rich}

        # Proactive check-in: choose rich_intent based on anomaly context
        if proactive:
            # Anomaly-based proactive → pick appropriate rich_intent
            if anomaly in {"high", "medium"}:
                recent_anomalies = (self.user_state or {}).get("recent_anomalies", [])
                anomaly_text = " ".join(
                    " ".join(str(v) for v in a.values()) for a in recent_anomalies
                    if isinstance(a, dict)
                ).lower()
                if any(kw in anomaly_text for kw in ["胃痛", "pain", "sick", "ill", "cold",
                                                       "fever", "cough", "allergy", "digest"]):
                    rich_proactive = "physical_discomfort"
                elif any(kw in anomaly_text for kw in ["mood_decline", "情绪", "低落", "焦虑",
                                                         "抑郁", "stress", "anxiety"]):
                    rich_proactive = "seeking_empathy"
                elif any(kw in anomaly_text for kw in ["sleep_decline", "睡眠", "失眠", "熬夜",
                                                         "睡", "slee", "insomnia"]):
                    rich_proactive = "seeking_empathy"
                else:
                    # exercise_missed or unknown type → gentle check-in
                    rich_proactive = "greeting"
                plan_p = self._map_to_plan_intent(rich_proactive)
                return {"plan_intent": plan_p, "rich_intent": rich_proactive}

            # General proactive (no anomaly) → light check-in
            return {"plan_intent": "light_checkin", "rich_intent": "greeting"}

        # ── Plan-related intents ───────────────────────────────────────
        has_plan = bool((self.user_state or {}).get("plan_context"))
        if any(k in text for k in ["制定计划", "健康计划", "帮我规划", "调理方案", "做个计划",
                                     "定制计划", "养生计划", "制定方案", "调理身体"]):
            return {"plan_intent": "plan_request", "rich_intent": "other"}
        if has_plan:
            if any(k in text for k in ["改一下计划", "调整计划", "不想做", "太麻烦了",
                                         "换一个", "修改计划", "太难了", "做不到"]):
                return {"plan_intent": "plan_modify", "rich_intent": "other"}
            if any(k in text for k in ["跑了步", "试了你的建议", "完成了", "搞定了",
                                         "睡了", "吃了你", "按你说的"]):
                return {"plan_intent": "plan_feedback", "rich_intent": "other"}
        if any(k in text for k in ["提醒我", "记得提醒", "每天提醒", "定时提醒", "设个提醒",
                                     "提醒一下", "定个提醒"]):
            return {"plan_intent": "plan_reminder", "rich_intent": "other"}

        # Fallback for unmatched messages
        # Check for analysis/advice keywords that rich intent may have missed
        if any(k in text for k in ["为什么", "怎么", "分析", "原因", "趋势"]):
            return {"plan_intent": "analysis", "rich_intent": "reflection"}
        if any(k in text for k in ["建议", "怎么办", "要不要", "该不该"]):
            return {"plan_intent": "advice", "rich_intent": "seeking_advice"}

        return {"plan_intent": "light_checkin", "rich_intent": "casual_share"}

    # ── Strategy selection ─────────────────────────────────────────────

    def decide_strategy(self, rich_intent: str,
                         psycho_state: dict | None = None,
                         intervention: dict | None = None) -> dict | None:
        """Get persona-specific response strategy for the given intent.

        When psycho_state indicates a non-normal intervenable state,
        the strategy's prompt_extra is augmented with psycho-specific guidance
        so the LLM adjusts its tone and content according to the intervention.
        """
        from persona import get_strategy
        persona_id = self.persona.get("id", "default")
        strategy = get_strategy(persona_id, rich_intent)

        # ── Inject psycho intervention guidance into strategy ────────
        if strategy and isinstance(strategy, dict):
            psycho_guidance = _build_psycho_guidance(psycho_state, intervention)
            if psycho_guidance:
                existing_extra = strategy.get("prompt_extra", "")
                if existing_extra:
                    strategy["prompt_extra"] = existing_extra + "\n" + psycho_guidance
                else:
                    strategy["prompt_extra"] = psycho_guidance

        return strategy


    def decide_burst(self, intent: str, proactive: bool = False) -> dict:
        persona = self.persona
        burst_probability = int(persona.get("burst_probability", 50))
        emoji_density = persona.get("emoji_density", "medium")
        if proactive:
            return {"num_messages": 3 if burst_probability >= 50 else 2, "emoji_density": "high" if burst_probability >= 65 else emoji_density, "intensity": "low", "scenario": "\u4e3b\u52a8\u5173\u5fc3"}
        if intent == "empathy":
            return {"num_messages": 3 if burst_probability >= 60 else 2, "emoji_density": emoji_density, "intensity": "medium", "scenario": "\u5b89\u6170\u548c\u966a\u4f34"}
        if intent == "analysis":
            return {"num_messages": 2 if burst_probability >= 40 else 1, "emoji_density": "low", "intensity": "medium", "scenario": "\u7406\u6027\u5206\u6790"}
        if intent == "advice":
            return {"num_messages": 2, "emoji_density": "low", "intensity": "medium", "scenario": "\u7ed9\u51fa\u5efa\u8bae"}
        if intent == "humor":
            return {"num_messages": 2 if burst_probability >= 55 else 1, "emoji_density": "high", "intensity": "high", "scenario": "\u8f7b\u677e\u8c03\u4f83"}
        if intent == "plan_request":
            return {"num_messages": 2 if burst_probability >= 40 else 1, "emoji_density": "medium", "intensity": "medium", "scenario": "\u5065\u5eb7\u8ba1\u5212\u751f\u6210"}
        if intent == "plan_reminder":
            return {"num_messages": 1 if burst_probability < 60 else 2, "emoji_density": "low", "intensity": "low", "scenario": "\u63d0\u9192\u8bbe\u7f6e"}
        if intent == "plan_feedback":
            return {"num_messages": 2, "emoji_density": "medium", "intensity": "medium", "scenario": "\u8ba1\u5212\u6267\u884c\u53cd\u9988"}
        if intent == "plan_modify":
            return {"num_messages": 2 if burst_probability >= 50 else 1, "emoji_density": "medium", "intensity": "medium", "scenario": "\u8ba1\u5212\u52a8\u6001\u8c03\u6574"}
        if intent == "light_checkin":
            return {"num_messages": 1, "emoji_density": "low", "intensity": "low", "scenario": "\u81ea\u7136\u63a5\u8bdd"}
        return {"num_messages": 1 if burst_probability < 60 else 2, "emoji_density": emoji_density, "intensity": "low", "scenario": "\u81ea\u7136\u63a5\u8bdd"}


def _build_psycho_guidance(psycho_state: dict | None,
                            intervention: dict | None) -> str:
    """Build psycho-guidance text to inject into the response strategy.

    When the user is in a non-normal intervenable state, this returns
    LLF-friendly instructions for how to adjust tone and content.
    Returns empty string if no guidance is needed.
    """
    if not psycho_state and not intervention:
        return ""

    state_name = (psycho_state or {}).get("intervenable_state", "normal")
    inter = intervention or {}

    if state_name == "normal" or inter.get("intervention_type") in (None, "no_intervention"):
        return ""

    guidance_map = {
        "willing_but_stuck": (
            "- 用户有意愿但是启动阻力大，不要催\n"
            "- 降低运动门槛，给微运动入口——走5分钟就够\n"
            "- 不要说'加油'，要帮对方拆解到最小可执行动作"
        ),
        "low_mood_available": (
            "- 用户情绪偏低但有运动意愿\n"
            "- 推荐舒缓型运动（散步、拉伸），不要提强度\n"
            "- 语气温暖、不施压，像朋友轻声建议"
        ),
        "post_exercise_frustrated": (
            "- 用户刚运动完但有挫败感\n"
            "- 降低下次目标，避免羞辱反馈\n"
            "- 不要空洞地说'加油'，说'刚开始都这样'更有效"
        ),
        "avoidance_after_break": (
            "- 用户有回避倾向+自责，可能因为断签了\n"
            "- 【关键】绝不强调断签或提起'已经X天没动了'\n"
            "- 给重新开始的信号，像什么都没发生过\n"
            "- 语气要轻松、无压力"
        ),
        "high_achievement_fatigued": (
            "- 用户近期运动积极但明显疲劳\n"
            "- 建议休息日，避免过度训练\n"
            "- 可以说'今天歇一天吧'而不是'今天也加油'"
        ),
        "low_self_efficacy": (
            "- 用户自我效能偏低，需要极低目标\n"
            "- 给'完成就行'的极简任务\n"
            "- 多用肯定的语气——'你穿上了运动鞋就已经成功了'"
        ),
        "social_motivated": (
            "- 用户社交驱动力强\n"
            "- 可以给分享/组队/挑战入口\n"
            "- 语气可以兴奋一点"
        ),
        "social_pressured": (
            "- 用户对社交比较敏感/有压力\n"
            "- 不提排行榜，不与其他用户比较\n"
            "- 保持私密感，只关注用户自己的进步"
        ),
    }

    guidance = guidance_map.get(state_name)
    if guidance:
        return f"\n【心理状态建议】\n{guidance}"
    return ""


def _call_chat_generation(system_prompt: str, user_prompt: str, **trace_kwargs: Any) -> str:
    return _call_llm(system_prompt, user_prompt, **trace_kwargs) if trace_kwargs else _call_llm(system_prompt, user_prompt)
def generate_chat_response(message: str, extracted: dict, context: list, persona_id: str = None, user_id: str = None, user_state: Optional[dict] = None, insights: Optional[list] = None, proactive: bool = False, burst: bool = False, trace_message_id: str = None, psycho_state: dict | None = None) -> Any:
    """
    Generate a Persona-driven chat response.

    Returns either a string (legacy) or a list of strings when burst mode is enabled.
    If trace_message_id is provided, a PromptTrace is recorded for debugging.
    """
    from persona import get_active_persona
    from persona import PERSONA_PROFILES
    from weather import build_weather_prompt_context

    persona = get_active_persona(user_id)
    if persona_id and persona_id != persona["id"]:
        from persona import get_persona
        persona = get_persona(persona_id) or persona

    pid = persona.get("id", "default")
    time_ctx = _get_time_context()
    weather_ctx = build_weather_prompt_context(user_id)

    # ── Build trace route ────────────────────────────────────────────
    # Only the actual nodes rendered for the LLM call.
    # persona/time/weather are embedded inside chat/system/base via
    # {persona_overlay} and {extra_context} — they are NOT separate
    # render_chain nodes.
    trace_route = ["chat/system/base"]

    trace_vars = {
        "persona_overlay": get_persona_overlay(pid),
        "extra_context": ("\n\n" + time_ctx if time_ctx else "") + ("\n\n" + weather_ctx if weather_ctx else ""),
        # Metadata for debugging (not used by templates, preserved for trace inspection)
        "_persona_id": pid,
        "_has_time": bool(time_ctx),
        "_has_weather": bool(weather_ctx),
        "_message": message[:300],
    }

    # Build system prompt via tree
    system_prompt = _pt_system(pid, time_ctx, weather_ctx or "")

    planner = ResponsePlanner(persona, user_state=user_state)
    intent_result = planner.decide_intent(message, extracted, context, proactive=proactive)
    primary_intent = intent_result["plan_intent"]
    rich_intent = intent_result["rich_intent"]

    # ── Strategy (persona × intent specific) ─────────────────────────
    strategy = planner.decide_strategy(rich_intent, psycho_state=psycho_state)
    profile = PERSONA_PROFILES.get(persona.get("id", "default"), {})

    burst_plan = planner.decide_burst(primary_intent, proactive=proactive) if burst else {"num_messages": 1, "emoji_density": persona.get("emoji_density", "medium"), "intensity": "medium"}
    opening_plan = planner.decide_opening(primary_intent, proactive=proactive)
    planner_payload = {"primary_intent": primary_intent, "rich_intent": rich_intent, **burst_plan, "opening": opening_plan}

    if opening_plan.get("show_typing"):
        planner_payload["typing_delay_seconds"] = opening_plan.get("typing_delay", 5)

    if burst and burst_plan.get("num_messages", 1) > 1:
        trace_route.append("chat/user/burst")
        user_prompt = _pt_user(
            pid, message, extracted, context,
            user_state=user_state, planner=planner_payload,
            insights=insights, strategy=strategy, profile=profile,
            burst_mode=True,
        )
        # ── Build complete trace_vars for reconstruction ──────────────
        _full_trace_vars = _build_trace_vars(
            base_vars=trace_vars,
            planner_payload=planner_payload,
            primary_intent=primary_intent,
            persona_id=pid,
            user_state=user_state,
            insights=insights,
            strategy=strategy,
            profile=profile,
            message=message,
            extracted=extracted,
            context=context,
            burst_mode=True,
        )
        result = _call_chat_generation(
            system_prompt, user_prompt,
            trace_route=trace_route,
            trace_vars=_full_trace_vars,
            trace_planner=planner_payload,
            trace_message_id=trace_message_id,
        )
        if result.strip() == '__VIORA_SILENT__':
            return []
        raw_lines = [line for line in result.split("---")]
        lines = _normalize_burst_list(raw_lines)
        lines = [l for l in lines if l.strip() != '__VIORA_SILENT__']
        if not lines:
            return []
        if lines:
            if proactive and len(lines) == 1 and len(lines[0]) > 50 and "---" not in result:
                fallback_lines = _split_proactive_fallback(result)
                if len(fallback_lines) > 1:
                    final = fallback_lines[: burst_plan.get("num_messages", len(fallback_lines))]
                    _patch_trace_final(trace_message_id, final)
                    return final
            final = lines[: burst_plan.get("num_messages", len(lines))]
            _patch_trace_final(trace_message_id, final)
            return final

    trace_route.append("chat/user/default")
    user_prompt = _pt_user(
        pid, message, extracted, context,
        user_state=user_state, planner=planner_payload,
        insights=insights, strategy=strategy, profile=profile,
        burst_mode=False,
    )
    # ── Build complete trace_vars for reconstruction ──────────────
    _full_trace_vars = _build_trace_vars(
        base_vars=trace_vars,
        planner_payload=planner_payload,
        primary_intent=primary_intent,
        persona_id=pid,
        user_state=user_state,
        insights=insights,
        strategy=strategy,
        profile=profile,
        message=message,
        extracted=extracted,
        context=context,
        burst_mode=False,
    )
    result = _call_chat_generation(
        system_prompt, user_prompt,
        trace_route=trace_route,
        trace_vars=_full_trace_vars,
        trace_planner=planner_payload,
        trace_message_id=trace_message_id,
    )
    result = result.strip().strip('"').strip("'").strip()
    if result == '__VIORA_SILENT__':
        return ""
    if "---" in result:
        lines = _normalize_burst_list(result.split("---"))
        if lines:
            _patch_trace_final(trace_message_id, lines)
            return lines
    if proactive and len(result) > 50:
        fallback_lines = _split_proactive_fallback(result)
        if len(fallback_lines) > 1:
            _patch_trace_final(trace_message_id, fallback_lines)
            return fallback_lines
    _patch_trace_final(trace_message_id, result)
    return result
def _build_trace_vars(
    base_vars: dict,
    planner_payload: dict,
    primary_intent: str,
    persona_id: str,
    user_state: Optional[dict],
    insights: Optional[list],
    strategy: Optional[dict],
    profile: Optional[dict],
    message: str,
    extracted: dict,
    context: list,
    burst_mode: bool = False,
) -> dict:
    """Build complete trace_vars for prompt reconstruction.

    Must include every variable needed by the template nodes in trace_route,
    computed the same way as build_chat_user_prompt.
    """
    import json as _json

    user_state_copy = dict(user_state or {})

    # Plan context (same logic as build_chat_user_prompt)
    plan_context_block = ""
    if "plan_context" in user_state_copy:
        plan_text = user_state_copy.pop("plan_context")
        note = "注意：如果用户刚刚要求制定计划，你回复的重点是介绍这个计划、解释为什么这样安排，并鼓励用户开始执行。如果已有进行中的计划，回复时可以自然参考今日安排，但不要生硬地逐条念计划。"
        plan_context_block = f"【用户当前健康计划】\n{plan_text}\n\n{note}"

    brief_user_state = _json.dumps(user_state_copy, ensure_ascii=False)

    # Strategy block
    strategy_block = ""
    if strategy and strategy.get("prompt_extra"):
        strategy_block = f"\n【本次回复策略】\n{strategy['prompt_extra']}"

    # Profile / life experiences block
    profile_block = ""
    if profile and strategy and strategy.get("self_disclosure"):
        experiences = profile.get("life_experiences", {})
        if experiences:
            exp_lines = "\n".join(f"  · {k}：{v}" for k, v in experiences.items())
            profile_block = f"\n【你的个人经历（只分享相关的、不要生硬插入）】\n{exp_lines}\n\n注意：这些是你的真实经历，在对话中如果合适可以自然分享，但不要为了讲故事而讲故事。如果和当前话题不相关就不说。"

    # Personality traits block
    traits_block = ""
    if profile and strategy:
        traits = profile.get("personality", {})
        if traits:
            traits_block = f"\n【你的性格特质（据此调整语气）】\n共情倾向：{traits.get('empathy', 3)}/5  |  建议倾向：{traits.get('advice', 3)}/5\n自我表露：{traits.get('self_disclosure', 3)}/5  |  幽默感：{traits.get('humor', 3)}/5\n毒舌程度：{traits.get('teasing', 3)}/5  |  直接程度：{traits.get('directness', 3)}/5\n温暖程度：{traits.get('warmth', 3)}/5"

    if burst_mode:
        # Variables for chat/user/burst template
        persona_constraints = _json.dumps({
            "forbidden": [],
            "phrase_preferences": [],
        }, ensure_ascii=False)

        return {
            **base_vars,
            "scenario": planner_payload.get("scenario", "自然接话"),
            "primary_intent": primary_intent,
            "intensity": planner_payload.get("intensity", ""),
            "num_messages": str(planner_payload.get("num_messages", "")),
            "emoji_density": planner_payload.get("emoji_density", ""),
            "persona_constraints": persona_constraints,
            "plan_context_block": plan_context_block,
            "brief_user_state": brief_user_state,
            "insights": _json.dumps(insights or [], ensure_ascii=False),
            "strategy_block": strategy_block,
            "profile_block": profile_block,
            "traits_block": traits_block,
            "message": message,
            "extracted": _json.dumps(extracted or {}, ensure_ascii=False, indent=2),
            "context": _json.dumps(context or [], ensure_ascii=False, indent=2),
        }
    else:
        # Variables for chat/user/default template
        planner_block = _json.dumps(planner_payload or {}, ensure_ascii=False)
        insights_block = _json.dumps(insights or [], ensure_ascii=False)

        from prompt_tree import get_persona_chat_extra

        chat_prompt_extra = get_persona_chat_extra(persona_id)

        return {
            **base_vars,
            "message": message,
            "extracted": _json.dumps(extracted or {}, ensure_ascii=False, indent=2),
            "context": _json.dumps(context or [], ensure_ascii=False, indent=2),
            "brief_user_state": brief_user_state,
            "planner_block": planner_block,
            "insights_block": insights_block,
            "plan_context_block": plan_context_block,
            "strategy_block": strategy_block,
            "profile_block": profile_block,
            "traits_block": traits_block,
            "chat_prompt_extra": chat_prompt_extra,
        }


def _patch_trace_final(message_id: str, final_response: Any) -> None:
    """Update a previously-saved trace with the final processed response."""
    if not message_id:
        return
    try:
        from prompt_traces import get_trace, _write_all, _read_all
        traces = _read_all()
        if message_id in traces:
            traces[message_id]["final_response"] = (
                final_response if isinstance(final_response, str)
                else list(final_response)
            )
            _write_all(traces)
    except Exception:
        pass
def _parse_story_response(raw: str) -> dict:
    """Parse structured story response into title + sections.

    Expects format: === 标题 === ... === 数据速览 === ... === 本周故事 === ... === 小贴士 === ...

    Returns: dict with keys: title, stats, narrative, tips, _raw
    """
    parts = raw.split("===")
    result = {
        "title": "",
        "stats": "",
        "narrative": "",
        "tips": "",
        "_raw": raw,
    }

    # Map section labels to result keys (fuzzy match)
    section_map = {
        "\u6807\u9898": "title",
        "\u6570\u636e\u901f\u89c8": "stats",
        "\u672c\u5468\u6545\u4e8b": "narrative",
        "\u5c0f\u8d34\u58eb": "tips",
    }

    # Collect content between === markers
    sections = []
    current_label = None
    current_lines = []

    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        # Check if this part starts with a known section label
        matched = False
        for label, key in section_map.items():
            if part.startswith(label) or part.startswith(label.replace(" ", "")):
                if current_label is not None and current_lines:
                    sections.append((current_label, "\n".join(current_lines).strip()))
                current_label = key
                # Content after the label (strip the label itself)
                rest = part[len(label):].strip()
                current_lines = [rest] if rest else []
                matched = True
                break
        if not matched:
            if current_label is not None:
                current_lines.append(part)

    # Flush last section
    if current_label is not None and current_lines:
        sections.append((current_label, "\n".join(current_lines).strip()))

    for key, content in sections:
        if key in result:
            result[key] = content

    # Fallback: if parsing produced empty narrative, use the raw text
    if not result["narrative"] and raw.strip():
        # Try to extract anything useful from the raw text
        lines = [l.strip() for l in raw.split("\n") if l.strip() and not l.strip().startswith("===")]
        result["narrative"] = "\n".join(lines)

    return result
def generate_body_story(records: list, period: str, insights: list = None,
                        baselines: dict = None, persona: dict = None) -> str:
    """
    Generate a narrative "body story" from structured records.

    Args:
        records: List of message dicts with extracted_data.
        period: Time period string, e.g. "day", "week", "month".
        insights: Optional list of correlation/trend insight dicts.
        baselines: Optional dict of user health baselines (sleep, energy, mood, etc.).
        persona: Optional persona dict with voice/style info.

    Returns:
        A narrative string (structured with sections), or a plain fallback string.
    """
    if not records:
        return f"\u8fd9\u6bb5\u65f6\u95f4\uff08{period}\uff09\u8fd8\u6ca1\u6709\u8bb0\u5f55\u54e6\uff0c\u5f00\u59cb\u548c\u6211\u8bf4\u8bf4\u4f60\u7684\u8eab\u4f53\u611f\u53d7\u5427\uff5e"

    records_str = json.dumps(records, ensure_ascii=False, indent=2)

    # Format insights for prompt
    if insights:
        insight_lines = []
        for ins in insights:
            if 'text' in ins:
                insight_lines.append(f"- {ins['text']}\uff08\u7f6e\u4fe1\u5ea6\uff1a{ins.get('confidence', 'unknown')}\uff09")
            elif 'dimension1' in ins and 'description' in ins:
                insight_lines.append(f"- {ins['dimension1']} \u4e0e {ins['dimension2']}: {ins['description']}")
            elif 'description' in ins:
                insight_lines.append(f"- {ins['description']}")
        insights_str = "\n".join(insight_lines)
    else:
        insights_str = "\u6682\u65e0\u5173\u8054\u53d1\u73b0\uff08\u6570\u636e\u4e0d\u8db3\uff09"

    # Format baselines for prompt
    if baselines:
        bl_lines = []
        for dim, val in baselines.items():
            if val is not None:
                bl_lines.append(f"  {dim}: {val}")
        baselines_str = "\n".join(bl_lines) if bl_lines else "\u6682\u65e0\u57fa\u7ebf\u6570\u636e"
    else:
        baselines_str = "\u6682\u65e0\u57fa\u7ebf\u6570\u636e"

    # Format persona voice for prompt
    if persona:
        persona_voice_lines = [
            f"\u4eba\u8bbe: {persona.get('name', 'Viora')}",
            f"\u8bed\u6c14: {persona.get('tone', '\u6e29\u6696')}",
            f"\u98ce\u683c: {persona.get('voice', '\u4eb2\u5207\u53cb\u597d')}",
        ]
        focus = persona.get('focus', '')
        if focus:
            persona_voice_lines.append(f"\u5173\u6ce8\u9886\u57df: {focus}")
        persona_voice_str = "\n".join(persona_voice_lines)
    else:
        persona_voice_str = "\u8bed\u6c14: \u6e29\u6696\u5e7d\u9ed8\n\u98ce\u683c: \u4eb2\u5207\u53cb\u597d\u3001\u50cf\u670b\u53cb\u804a\u5929"

    user_prompt = _pt_story(
        period=period,
        records=records_str,
        insights=insights_str,
        baselines=baselines_str,
        persona_voice=persona_voice_str,
    )
    result = _call_llm(
        "\u4f60\u662f Viora\uff0c\u4e00\u4e2a\u6e29\u6696\u5e7d\u9ed8\u7684\u5065\u5eb7\u4f19\u4f34\u3002\u8bf7\u751f\u6210\u4e00\u6bb5\u6709\u8da3\u7684\u8eab\u4f53\u6545\u4e8b\u53d9\u8ff0\u3002",
        user_prompt,
        _get_time_context(),
    )
    return result.strip().strip('"').strip("'").strip()
def extract_profile_notes(recent_messages: list, current_notes: str = "") -> str:
    """
    Extract soft profile information from recent messages and merge with
    existing profile notes using the LLM.

    Args:
        recent_messages: List of recent message dicts from storage (each with
                         'content' and optionally 'ai_response' fields).
        current_notes: The existing profile_notes string (may be empty).

    Returns:
        Updated profile_notes string — a concise 1-4 sentence paragraph.
    """
    if not recent_messages:
        return current_notes

    # Format the last ~15 messages as a compact conversation transcript.
    # Storage messages have 'content' (user) and 'ai_response' (assistant).
    formatted_msgs = []
    for m in recent_messages[-15:]:
        content = (m.get("content") or "").strip()
        if content:
            formatted_msgs.append(f"[用户]: {content}")
        ai_resp = m.get("ai_response")
        if ai_resp:
            if isinstance(ai_resp, list):
                ai_resp = " ".join(str(p).strip() for p in ai_resp if str(p).strip())
            ai_resp = ai_resp.strip()
            if ai_resp and len(ai_resp) > 120:
                ai_resp = ai_resp[:120] + "..."
            if ai_resp:
                formatted_msgs.append(f"[助手]: {ai_resp}")

    if not formatted_msgs:
        return current_notes

    messages_str = "\n".join(formatted_msgs)
    prompt = _pt_profile(
        messages=messages_str,
        existing_notes=current_notes or "\uff08\u6682\u65e0\uff09",
    )

    try:
        result = _call_llm(
            "\u4f60\u662f\u4e00\u4e2a\u7b80\u6d01\u7684\u7528\u6237\u4fe1\u606f\u5f52\u7eb3\u52a9\u624b\u3002",
            prompt,
        )
        result = result.strip().strip('"').strip("'").strip()
        # Remove any residual markdown or commentary prefixes the LLM might add
        for prefix in ("\u7b14\u8bb0\uff1a", "\u66f4\u65b0\u540e\uff1a", "\u5408\u5e76\u540e\uff1a", "\u6458\u8981\uff1a"):
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
        return result
    except Exception:
        logger.warning("Profile notes extraction failed, returning current notes unchanged.")
        return current_notes
# ── Helpers ─────────────────────────────────────────────────────────────────

def _empty_extracted() -> dict:
    """Return the default structure with all fields set to None."""
    return {
        "sleep": {"quality": None, "duration_hours": None},
        "energy": {"score": None, "reason": None},
        "mood": {"score": None, "reason": None},
        "exercise": {"type": None, "duration_minutes": None},
        "digestion": {"status": None, "severity": None},
        "skin": {"status": None},
        "pain": {"location": None, "severity": None},
        "diet": {"notes": None},
    }
def normalize_extracted(raw: dict) -> dict:
    """
    Ensure the LLM extraction response matches the expected schema.

    The LLM may return flat nulls (e.g., "digestion": null) instead of
    nested objects. This normalizes every field to the expected structure.
    """
    template = _empty_extracted()
    result = {}

    for key, expected_val in template.items():
        raw_val = raw.get(key)
        if isinstance(expected_val, dict):
            # Handle backward compat: convert old int-format energy/mood to dict
            if key in ("energy", "mood") and isinstance(raw_val, (int, float)):
                raw_val = {"score": raw_val, "reason": None}
            # Should be a dict; if not, replace with template
            if isinstance(raw_val, dict):
                result[key] = {k: raw_val.get(k, v) for k, v in expected_val.items()}
            else:
                result[key] = dict(expected_val)
        else:
            result[key] = raw_val if raw_val is not None else expected_val

    return result

def _format_context_value(value: Any) -> str:
    """Format a context value (dict, list, str) into a concise string for prompt injection."""
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            if v is not None:
                parts.append(f"{k}: {v}")
        return "；".join(parts) if parts else "无"
    if isinstance(value, list):
        return "\n".join(str(item) for item in value) if value else "无"
    if isinstance(value, str):
        return value if value.strip() else "无"
    return str(value) if value is not None else "无"
# ── Expand Reasoning Node ──────────────────────────────────────
def expand_reasoning_node(
    node_dict: dict,
    user_context: dict,
    existing_nodes: list[dict] | None = None,
    path_from_root: list[dict] | None = None,
    user_answer: str | None = None,
) -> list[dict]:
    """
    Expand a reasoning tree node by calling the LLM to generate child nodes.

    For root nodes (depth=0 or parent_id=None), uses EXPAND_ROOT_PROMPT to
    generate the first layer of causal factors.

    For child nodes, uses EXPAND_CHILD_PROMPT to recursively decompose
    into more specific sub-factors.

    Args:
        node_dict: The node to expand. Expected keys:
            - label (str): Node label/name
            - description (str): Node description
            - id (str, optional): UUID of the node
            - parent_id (str, optional): UUID of parent node
            - depth (int, optional): Depth in the tree (default 0)
        user_context: Context dict for prompt formatting. Expected keys:
            - concern (str): User's main health concern
            - routine_summary (str or dict): User's routine summary
            - profile_notes (str): User profile notes
            - insights (str): Health insights
            - weather_context (str): Weather/environment info
        existing_nodes: List of existing nodes at the same level as the
            children being generated. Passed to the prompt so the LLM can
            avoid generating duplicate or overlapping nodes. Each dict
            should have at least "label" and optionally "description".
        user_answer: Optional answer to a previously-asked followup question
            (interactive info-gathering). When provided, it is injected into the
            child expansion prompt as additional context.

    Returns:
        List of child node dicts, each with: id, label, description, emoji,
        is_actionable, depth, parent_id, and optionally action_text, time_slot,
        frequency, duration. Returns empty list on any error.
    """
    node_id = node_dict.get("id")
    parent_id = node_dict.get("parent_id")
    node_depth = node_dict.get("depth", 0)
    is_root = (node_depth == 0 or parent_id is None)

    # Extract and format user context
    concern = user_context.get("concern", "")
    raw_routine = user_context.get("routine_summary", "无")
    profile_notes = user_context.get("profile_notes", "无")
    raw_insights = user_context.get("insights", "无")
    weather_context = user_context.get("weather_context", "无")

    # Format dict/list values into strings
    routine_summary = _format_context_value(raw_routine)
    insights = _format_context_value(raw_insights)

    time_ctx = _get_time_context()
    sys_prompt = "你是一个健康推理专家。只输出有效的JSON，不包含任何其他文字。"

    try:
        if is_root:
            prompt = _pt_expand_root(
                concern=concern,
                routine_summary=routine_summary,
                profile_notes=profile_notes,
                insights=insights,
                weather_context=weather_context,
            )
        else:
            parent_label = node_dict.get("label", "")
            parent_description = node_dict.get("description", "")

            # Format reasoning path from root to current node
            if path_from_root:
                path_lines = []
                for i, p in enumerate(path_from_root):
                    prefix = "├ " if i < len(path_from_root) - 1 else "└ "
                    path_lines.append(f'{prefix}{p.get("label", "")}：{p.get("description", "")}')
                reasoning_path_str = "\n".join(path_lines)
            else:
                reasoning_path_str = f"{parent_label}：{parent_description}"

            # Format existing sibling nodes — only labels for dedup
            if existing_nodes:
                sibling_labels = [sib.get("label", "") for sib in existing_nodes if sib.get("label")]
                existing_sibling_str = "、".join(sibling_labels) if sibling_labels else "无"
            else:
                existing_sibling_str = "无"

            prompt = _pt_expand_child(
                reasoning_path=reasoning_path_str,
                parent_label=parent_label,
                parent_description=parent_description,
                routine_summary=routine_summary,
                profile_notes=profile_notes,
                existing_sibling_nodes=existing_sibling_str,
                user_answer=user_answer or "",
            )

        response = _call_llm(sys_prompt, prompt, time_ctx)
        children = _parse_json_response(response)

        # Handle single object or wrapped response
        if isinstance(children, dict):
            # Check for wrapped format: {"children": [...]}
            if "children" in children and isinstance(children["children"], list):
                children = children["children"]
            else:
                children = [children]

        if not isinstance(children, list):
            logger.warning(
                "expand_reasoning_node: expected list, got %s",
                type(children).__name__,
            )
            return []

        # Validate and normalize each child node
        validated = []
        child_depth = node_depth + 1

        for child in children:
            if not isinstance(child, dict):
                continue

            child.setdefault("label", "未知因素")
            child.setdefault("description", "")
            child.setdefault("emoji", "🔍")
            child.setdefault("is_actionable", False)

            # Post-processing safety net: if LLM says is_actionable=true but
            # action_text is missing/empty or identical to the label (meaning
            # the LLM just copied the condition name), demote to non-actionable.
            if child.get("is_actionable"):
                action_text = child.get("action_text") or ""
                label = child.get("label", "")
                if not action_text.strip():
                    child["is_actionable"] = False
                elif action_text.strip() == label.strip():
                    # LLM just used the label as action_text (e.g.
                    # label="脾运化功能减弱" action_text="脾运化功能减弱")
                    # This is not a real executable action.
                    child["is_actionable"] = False
                    child["action_text"] = None

            # Post-processing: reject multi-entity labels (contains 与/和/及/、)
            label = child.get("label", "")
            if re.search(r"[与和及、]", label):
                logger.warning(
                    "Rejecting node with multi-entity label: '%s' (contains 与/和/及/、)",
                    label,
                )
                continue

            # Enforce the terminal-leaf invariant at the source: actionable
            # leaves are end-nodes (expandable=False), non-actionable nodes are
            # expandable. The expand prompts don't emit "expandable", so derive
            # it here instead of leaving it to callers to re-derive (app.py's
            # setdefault becomes a harmless no-op; direct callers of
            # expand_node() get a consistent value too).
            child["expandable"] = not child.get("is_actionable", False)

            child["id"] = str(uuid4())
            child["depth"] = child_depth
            child["parent_id"] = node_id

            validated.append(child)

        return validated

    except (ValueError, json.JSONDecodeError, LLMError, Exception) as e:
        logger.error(
            "expand_reasoning_node failed for node '%s': %s",
            node_dict.get("label", ""),
            e,
        )
        return []


def ask_followup_question(
    node_dict: dict,
    user_context: dict,
    path_from_root: list[dict] | None = None,
) -> tuple[str | None, list[str] | None]:
    """
    Decide whether a reasoning node needs a clarifying question before expanding.

    Interactive info-gathering (Phase 4 待完善方向): when the available context is
    insufficient to decide how to decompose a node, the LLM returns a concrete
    question (optionally with quick-choice options) to ask the user.

    Args:
        node_dict: The node to expand. Expected keys: label, description, id, parent_id, depth.
        user_context: Context dict (concern / routine_summary / profile_notes / insights / weather_context).
        path_from_root: Reasoning path from the root to the current node (for prompt context).

    Returns:
        (question, options) tuple. question is str or None if no clarification is
        needed; options is a list[str] or None (open-ended question / no choices).
    """
    node_label = node_dict.get("label", "")
    node_description = node_dict.get("description", "")
    raw_routine = user_context.get("routine_summary", "无")
    profile_notes = user_context.get("profile_notes", "无")
    routine_summary = _format_context_value(raw_routine)

    if path_from_root:
        path_lines = []
        for i, p in enumerate(path_from_root):
            prefix = "├ " if i < len(path_from_root) - 1 else "└ "
            path_lines.append(f'{prefix}{p.get("label", "")}：{p.get("description", "")}')
        reasoning_path_str = "\n".join(path_lines)
    else:
        reasoning_path_str = f"{node_label}：{node_description}"

    prompt = _pt_followup(
        node_label=node_label,
        node_description=node_description,
        reasoning_path=reasoning_path_str,
        routine_summary=routine_summary,
        profile_notes=profile_notes,
    )
    sys_prompt = "你是一个健康推理专家。只输出有效的JSON，不包含任何其他文字。"
    time_ctx = _get_time_context()

    try:
        response = _call_llm(sys_prompt, prompt, time_ctx)
        parsed = _parse_json_response(response)
        if not isinstance(parsed, dict):
            return None, None
        q = parsed.get("followup_question")
        if not (isinstance(q, str) and q.strip()):
            return None, None
        # Optional quick-choice options: normalize to a list of non-empty strings.
        # Deduplicate while preserving order (LLM may emit near-identical choices;
        # the prompt asks for mutually-exclusive options, so duplicates are noise).
        options = None
        raw_opts = parsed.get("options")
        if isinstance(raw_opts, list):
            # Keep only scalar entries (LLM occasionally emits structured objects
            # like {"label": "早餐"} — str() on a dict would leak a garbage repr
            # into the UI). Strip, drop empties, dedup preserving order, cap at 6.
            opts = list(dict.fromkeys(
                str(o).strip()
                for o in raw_opts
                if o is not None
                and not isinstance(o, (dict, list, tuple, set))
                and str(o).strip()
            ))
            if opts:
                options = opts[:6]  # cap at 6 options to avoid UI overflow
        return q.strip(), options
    except Exception as e:
        logger.warning("ask_followup_question failed for node '%s': %s", node_label, e)
        return None, None
def extract_reminder_from_message(message: str) -> dict:
    """
    Extract reminder info from a user message using LLM.
    Returns dict with keys: action, time_spec, frequency, days_of_week.
    """
    prompt = _pt_reminder(message=message)
    system_msg = "\u4f60\u662f\u4e00\u4e2a\u63d0\u9192\u4fe1\u606f\u63d0\u53d6\u52a9\u624b\u3002\u53ea\u8f93\u51faJSON\u3002"
    time_ctx = _get_time_context()

    try:
        result = _call_llm(system_msg, prompt, time_ctx)
        parsed = _parse_json_response(result)
        return {
            "action": parsed.get("action", ""),
            "time_spec": parsed.get("time_spec", "09:00"),
            "frequency": parsed.get("frequency", "\u6bcf\u5929"),
            "days_of_week": parsed.get("days_of_week"),
        }
    except Exception as e:
        logger.warning("Reminder extraction failed: %s", e)
        return {"action": message[:30], "time_spec": "09:00", "frequency": "\u6bcf\u5929", "days_of_week": None}
def extract_plan_feedback(message: str, plan_items: list[dict]) -> dict:
    """
    Extract plan execution feedback from user message.
    plan_items: list of dicts with item_id, time_slot, action fields.
    Returns dict with feedback_type, matched_item_id, suggested_change, needs_regenerate.
    """
    # Format plan items for the prompt
    items_text = "\n".join(
        f"  [{item['item_id']}] {item.get('time_slot', '')}: {item.get('action', '')}"
        for item in plan_items
    ) if plan_items else "\u6682\u65e0\u8ba1\u5212\u9879\u76ee"

    prompt = _pt_feedback(message=message, plan_items=items_text)
    system_msg = "\u4f60\u662f\u4e00\u4e2a\u8ba1\u5212\u53cd\u9988\u63d0\u53d6\u52a9\u624b\u3002\u53ea\u8f93\u51faJSON\u3002"

    try:
        result = _call_llm(system_msg, prompt)
        parsed = _parse_json_response(result)
        return {
            "feedback_type": parsed.get("feedback_type", "none"),
            "matched_item_id": parsed.get("matched_item_id"),
            "suggested_change": parsed.get("suggested_change"),
            "needs_regenerate": parsed.get("needs_regenerate", False),
        }
    except Exception as e:
        logger.warning("Plan feedback extraction failed: %s", e)
        return {"feedback_type": "none", "matched_item_id": None, "suggested_change": None, "needs_regenerate": False}
