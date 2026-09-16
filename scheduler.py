"""
scheduler.py — Proactive interaction scheduler for Viora Phase 2.

Implements the "主动智能" (proactive intelligence) concept:
Viora reaches out to the user at appropriate times based on
anomaly detection, routine patterns, and user availability.

Rules:
- Max 1 proactive message per day (configurable per persona)
- Not during user's busy hours (weekday mornings)
- Auto-decrease frequency if user doesn't respond
- Silent mode (user can toggle "don't bother me")
- Per-user state with activity-based frequency adjustment
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# China Standard Time
CST = timezone(timedelta(hours=8))

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Legacy global state file (for backward compatibility)
SCHEDULER_STATE_FILE = os.path.join(DATA_DIR, "scheduler_state.json")

# ── Scheduling config ────────────────────────────────────────────────────────

CHECK_INTERVAL = 3600  # 1 hour
PROACTIVE_COOLDOWN = 12
PROACTIVE_COOLDOWN_IDLE = 168
QUIET_HOURS_START = 23  # 11 PM
QUIET_HOURS_END = 7     # 7 AM
BUSY_HOURS_START = 9
BUSY_HOURS_END = 12
RESPONSE_BACKOFF_THRESHOLD = 2
DAILY_PROACTIVE_LIMIT = 1

# ── Psycho-aware proactive adjustment ──────────────────────────────────────

PSYCHO_PROACTIVE_ADJUSTMENT = {
    # state: (multiplier, override_reason)
    # multiplier > 1.0 = more likely to send (user receptive / needs nudge)
    # multiplier < 1.0 = less likely (need space / cooldown)
    "willing_but_stuck": (1.5, "user_needs_nudge"),
    "low_mood_available": (1.3, "user_receptive"),
    "low_self_efficacy": (1.2, "gentle_encouragement"),
    "social_motivated": (1.5, "social_engagement"),
    "social_pressured": (0.5, "reduce_social_pressure"),
    "avoidance_after_break": (0.3, "avoidance_sensitivity"),
    "post_exercise_frustrated": (0.6, "post_exercise_cooldown"),
    "high_achievement_fatigued": (0.4, "fatigue_respect"),
    "normal": (1.0, None),
}


def _adjust_for_psycho_state(base_decision: dict,
                              psycho_state: dict | None) -> dict:
    """Adjust proactive scheduling decision based on user's psychological state.

    Positive states (willing_but_stuck, low_mood_available, social_motivated):
      → more likely to nudge, even if base says no.
    Negative states (avoidance_after_break, social_pressured, fatigued):
      → less likely to disturb, override base yes to no.
    Hard limits (silent_mode, daily_limit) are never overridden.
    """
    if not psycho_state:
        return base_decision

    state_name = psycho_state.get("intervenable_state", "normal")
    adjustment = PSYCHO_PROACTIVE_ADJUSTMENT.get(state_name)
    if adjustment is None:
        return base_decision

    multiplier, override_reason = adjustment

    # Never override hard limits
    if not base_decision.get("should_send", False) and base_decision.get("reason") in (
        "silent_mode", "daily_limit_reached", "user_inactive", "user_not_responding",
    ):
        return base_decision

    # Positive states: override "no" → "yes" (for soft rejections)
    if multiplier > 1.0:
        if not base_decision.get("should_send", False):
            return {"should_send": True, "reason": f"psycho_{override_reason}"}
        # Already yes, keep it
        return base_decision

    # Negative states: override "yes" → "no"
    if multiplier < 1.0:
        if base_decision.get("should_send", False):
            return {"should_send": False, "reason": f"psycho_{override_reason}"}
        return base_decision

    return base_decision


def _state_file_for_user(user_id: str) -> str:
    """Get the state file path for a specific user."""
    if user_id == "default":
        return SCHEDULER_STATE_FILE
    return os.path.join(DATA_DIR, f"scheduler_state_{user_id}.json")


def _sanitize_optional_str(value):
    """Coerce scheduler-state optional-string fields to str | None.

    A hand-edited/legacy scheduler_state.json may store a number / dict / list
    where the schema expects an ISO timestamp or date string.  Such junk would
    crash datetime.fromisoformat() callers and leak into get_scheduler_status()
    responses — degrade it to None (treated as "absent" everywhere downstream).
    Well-formed data (str or None, which this module always writes) is untouched.
    """
    return value if (value is None or isinstance(value, str)) else None


def _sanitize_count(value, default: int = 0) -> int:
    """Coerce scheduler-state counter fields to a non-negative int.

    Corrupt counters (None / "abc" / 2.7 / dicts) crash the later `>=`
    comparisons and `+= 1` arithmetic in the scheduling logic with a TypeError.
    Legal data is always a non-negative int written by this module, so coercing
    junk to *default* (or truncating a float) does not change well-formed
    behavior — it only stops a corrupt file from breaking the daemon.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value >= 0 else default
    try:
        n = int(float(value))
        return n if n >= 0 else default
    except (ValueError, TypeError):
        return default


class SchedulerState:
    """Tracks scheduler state for proactive interactions (per user)."""

    def __init__(self):
        self.last_proactive_message: Optional[str] = None
        self.last_anomaly_check: Optional[str] = None
        self.unresponded_count: int = 0
        self.silent_until: Optional[str] = None
        self.proactive_count_today: int = 0
        self.last_proactive_date: Optional[str] = None
        self.total_proactive_count: int = 0

    def is_silent(self) -> bool:
        if not self.silent_until:
            return False
        try:
            until = datetime.fromisoformat(self.silent_until)
            if datetime.now(CST) < until:
                return True
            self.silent_until = None
        except (ValueError, TypeError):
            pass
        return False

    def reset_daily_count(self) -> None:
        today = datetime.now(CST).strftime("%Y-%m-%d")
        if self.last_proactive_date != today:
            self.proactive_count_today = 0
            self.last_proactive_date = today

    def save(self, user_id: str = "default") -> None:
        state_file = _state_file_for_user(user_id)
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, user_id: str = "default") -> "SchedulerState":
        state = cls()
        state_file = _state_file_for_user(user_id)
        try:
            if os.path.exists(state_file):
                with open(state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    state.__dict__.update(data)
                    # Sanitize field types so a hand-edited/corrupt state file
                    # degrades to defaults instead of crashing later (str counter
                    # compared to an int, dict fed to fromisoformat, etc.).
                    # Well-formed data written by this module is unaffected.
                    state.unresponded_count = _sanitize_count(state.unresponded_count)
                    state.proactive_count_today = _sanitize_count(state.proactive_count_today)
                    state.total_proactive_count = _sanitize_count(state.total_proactive_count)
                    state.silent_until = _sanitize_optional_str(state.silent_until)
                    state.last_proactive_message = _sanitize_optional_str(state.last_proactive_message)
                    state.last_proactive_date = _sanitize_optional_str(state.last_proactive_date)
                    state.last_anomaly_check = _sanitize_optional_str(state.last_anomaly_check)
                else:
                    logger.warning(
                        "Scheduler state for %s is not a dict (%s); starting fresh.",
                        user_id, type(data).__name__,
                    )
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load scheduler state for %s: %s", user_id, e)
        state.reset_daily_count()
        return state


def should_send_proactive(state: SchedulerState, user_id: str = "default",
                            wake_up_hour: int = None,
                            psycho_state: dict | None = None) -> dict:
    now = datetime.now(CST)

    # ── Hard limits (never overridden by psycho state) ──
    if state.is_silent():
        return {"should_send": False, "reason": "silent_mode"}

    state.reset_daily_count()
    if state.proactive_count_today >= DAILY_PROACTIVE_LIMIT:
        return {"should_send": False, "reason": "daily_limit_reached"}

    cooldown = PROACTIVE_COOLDOWN
    try:
        from accounts import compute_activity_score
        activity = compute_activity_score(user_id)
        if activity["level"] == "inactive":
            return {"should_send": False, "reason": "user_inactive"}
        elif activity["level"] == "idle":
            cooldown = PROACTIVE_COOLDOWN_IDLE
    except ImportError:
        pass

    if state.last_proactive_message:
        try:
            last_msg = datetime.fromisoformat(state.last_proactive_message)
            hours_since = (now - last_msg.replace(tzinfo=CST)).total_seconds() / 3600
            if hours_since < cooldown:
                return _adjust_for_psycho_state(
                    {"should_send": False, "reason": "cooldown"}, psycho_state)
        except (ValueError, TypeError):
            pass

    hour = now.hour
    # Use user's wake-up time for quiet hours end; default to 7
    quiet_end = wake_up_hour if wake_up_hour is not None else QUIET_HOURS_END
    if hour >= QUIET_HOURS_START or hour < quiet_end:
        return _adjust_for_psycho_state(
            {"should_send": False, "reason": "quiet_hours"}, psycho_state)

    if now.weekday() < 5:
        if BUSY_HOURS_START <= hour < BUSY_HOURS_END:
            return _adjust_for_psycho_state(
                {"should_send": False, "reason": "busy_hours"}, psycho_state)

    if state.unresponded_count >= RESPONSE_BACKOFF_THRESHOLD:
        return {"should_send": False, "reason": "user_not_responding"}

    return _adjust_for_psycho_state(
        {"should_send": True, "reason": "ok"}, psycho_state)


def enable_silent_mode(user_id: str = "default", duration_hours: float = 24) -> str:
    state = SchedulerState.load(user_id)
    until = datetime.now(CST) + timedelta(hours=duration_hours)
    state.silent_until = until.isoformat()
    state.save(user_id)
    logger.info("Silent mode enabled for %s until %s", user_id, until.strftime("%Y-%m-%d %H:%M"))
    return state.silent_until


def disable_silent_mode(user_id: str = "default") -> None:
    state = SchedulerState.load(user_id)
    state.silent_until = None
    state.save(user_id)
    logger.info("Silent mode disabled for %s", user_id)


def record_proactive_sent(user_id: str = "default") -> None:
    state = SchedulerState.load(user_id)
    state.last_proactive_message = datetime.now(CST).isoformat()
    state.proactive_count_today += 1
    state.total_proactive_count += 1
    if not state.last_proactive_date:
        state.last_proactive_date = datetime.now(CST).strftime("%Y-%m-%d")
    state.save(user_id)

    try:
        from accounts import record_proactive_received
        record_proactive_received(user_id)
    except ImportError:
        pass


def record_user_responded(user_id: str = "default") -> None:
    state = SchedulerState.load(user_id)
    state.unresponded_count = 0
    state.save(user_id)

    try:
        from accounts import record_proactive_reply
        record_proactive_reply(user_id)
    except ImportError:
        pass


def record_no_response(user_id: str = "default") -> None:
    state = SchedulerState.load(user_id)
    state.unresponded_count += 1
    state.save(user_id)


def delete_scheduler_state(user_id: str = "default") -> None:
    """Delete persisted scheduler state for a user."""
    state_file = _state_file_for_user(user_id)
    if os.path.exists(state_file):
        os.remove(state_file)


def get_scheduler_status(user_id: str = "default",
                          psycho_state: dict | None = None) -> dict:
    state = SchedulerState.load(user_id)
    decision = should_send_proactive(state, user_id, psycho_state=psycho_state)
    result = {
        "silent_mode": state.is_silent(),
        "silent_until": state.silent_until,
        "last_proactive": state.last_proactive_message,
        "unresponded_count": state.unresponded_count,
        "proactive_today": state.proactive_count_today,
        "total_count": state.total_proactive_count,
        "can_proactive": decision["should_send"],
        "reason": decision["reason"],
    }
    try:
        from accounts import compute_activity_score
        activity = compute_activity_score(user_id)
        result["activity"] = activity
    except ImportError:
        pass
    return result


def _decide_proactive_type(user_id: str, now: datetime, messages: list, wake_up_hour: int = 7) -> tuple:
    """
    Decide what kind of general proactive message to send when no anomalies exist.

    Priority: weather > morning > weekend > evening > silence > general
    Returns (type_str, extra_context_dict) or (None, {})

    wake_up_hour: User's typical wake-up hour (0-23). Defaults to 7.
                  Morning greeting window is wake_up_hour to wake_up_hour+2.
    """
    hour = now.hour

    # ── Weather check (most contextual, relevant any time) ──
    try:
        from weather import get_user_location, get_weather_context
        location = get_user_location(user_id)
        if location:
            weather_str = get_weather_context(location)
            if weather_str and _weather_notable(weather_str):
                return "weather", {
                    "weather": weather_str,
                    "city": location.get("city", ""),
                }
    except ImportError:
        pass

    # ── Morning greeting (uses user's wake-up time, default 7-9) ──
    morning_end = wake_up_hour + 2
    if wake_up_hour <= hour < morning_end:
        return "morning", {}

    # ── Weekend afternoon check-in ──
    weekday = now.weekday()
    if weekday >= 5 and 10 <= hour < 18:
        return "weekend", {}

    # ── Evening check-in (20:00-22:00) ──
    if 20 <= hour < 22:
        return "evening", {}

    # ── Silence check (>3 days since last message) ──
    if messages:
        try:
            # Find the newest well-formed message. A legacy/corrupt trailing
            # entry (bare string / number / null) must not crash .get("timestamp")
            # — skip non-dict entries and use the newest dict instead.
            last_msg = None
            for m in reversed(messages):
                if isinstance(m, dict):
                    last_msg = m
                    break
            if last_msg is not None:
                last_ts = last_msg.get("timestamp", "")
                last_dt = datetime.fromisoformat(last_ts)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=CST)
                days = (now - last_dt).days
                if days >= 3:
                    return "silence", {"days_silent": days}
        except (ValueError, TypeError, IndexError):
            pass

    # ── General state check-in (daytime hours, lowest priority) ──
    if 10 <= hour < 20:
        return "general", {}

    return None, {}


def _weather_notable(weather_str: str) -> bool:
    """Check if weather has notable conditions worth proactively mentioning."""
    notable_keywords = ["雨", "雪", "雷", "雾", "寒冷", "炎热", "降温", "高温"]
    return any(kw in weather_str for kw in notable_keywords)


def _build_proactive_message_prompt(msg_type: str, extra: dict) -> str:
    """Build the user-facing prompt for the LLM based on proactive type."""
    prompts = {
        "weather": "根据当前天气，给用户 1-3 条很短的主动关心消息。第一条只做轻轻的问候或提起天气感受，不要同条给建议也不要同条提问；如果需要提醒带伞/添衣/防暑，放到下一条；如果想问用户感受，再单独放一条。多条之间必须严格用 --- 分隔，每条控制在 4-25 个字。",
        "morning": "早上好。给用户 1-3 条很短的主动早安消息。注意：这是你主动发的消息，你并不知道用户此刻是否已经醒了、是否已经起床——所以绝对不要说「醒得真早」「醒啦」「起床了吗」这类假设用户状态的话，也不要说「今天起得…」这种评价。问候要适合一个可能刚醒、也可能还在睡的人看到。第一条只做温和早安问候，不要同时塞建议和问题；如果想提醒开窗、喝水或关心状态，拆到后续单独气泡。多条之间必须严格用 --- 分隔，每条控制在 4-25 个字。",
        "evening": "给用户 1-3 条很短的晚间主动关心消息。第一条只做轻松晚安/晚间问候；如果要问今天过得怎么样，单独放一条；不要一条里同时总结、建议、追问。多条之间必须严格用 --- 分隔，每条控制在 4-25 个字。",
        "weekend": "周末了，给用户 1-3 条很短的主动关心消息。第一条只做轻松周末问候；如果想问安排或提醒放松，拆到后续单独气泡。多条之间必须严格用 --- 分隔，每条控制在 4-25 个字。",
        "silence": f"用户已经有{extra.get('days_silent', 3)}天没说话了。给 1-3 条很短的惦记消息。第一条只表达想起对方/轻轻关心，不要同条追问太多；如果要问近况，单独放一条。不要显得催促。多条之间必须严格用 --- 分隔，每条控制在 4-25 个字。",
        "general": "随意地冒个泡，给用户 1-3 条很短的主动关心消息。第一条只做轻松问候；观察、建议、问题分别拆开，不要堆在同一条里。多条之间必须严格用 --- 分隔，每条控制在 4-25 个字。",
    }
    return prompts.get(msg_type, prompts["general"])


def generate_proactive_message(
    messages: list,
    persona_id: str = None,
    user_id: str = "default",
    account: dict = None,
    wake_up_hour: int = None,
) -> Optional[object]:
    """
    Generate a proactive caring message.

    Multi-tier strategy:
    1. Anomaly-based: if health anomalies detected → personalized caring about that anomaly
    2. General proactive: if no anomalies → time/weather/silence-based general caring

    wake_up_hour: User's typical wake-up hour. Defaults to 7 if not provided.
    """
    from routine import build_routine_model
    from ai_engine import generate_chat_response

    # ── Tier 1: Anomaly-based proactive ──
    model = build_routine_model(messages)
    anomalies = model.detect_anomalies()

    if anomalies:
        severity_order = {"high": 0, "medium": 1, "low": 2}
        anomalies.sort(key=lambda a: severity_order.get(a.get("severity", "low"), 2))
        top_anomaly = anomalies[0]

        try:
            return generate_chat_response(
                message=top_anomaly.get("description", "主动关心一下"),
                extracted={},
                context=messages[-5:],
                persona_id=persona_id,
                user_id=user_id,
                user_state={
                    "anomaly_level": top_anomaly.get("severity", "low"),
                    "recent_anomalies": anomalies[:3],
                    "proactive_mode": True,
                },
                insights=anomalies[:3],
                proactive=True,
                burst=True,
            )
        except Exception as e:
            logger.error("Failed to generate anomaly-based proactive message: %s", e)
            # Fall through to general proactive

    # ── Tier 2: General proactive (no anomaly data needed) ──
    now = datetime.now(CST)
    wh = wake_up_hour if wake_up_hour is not None else 7
    msg_type, extra = _decide_proactive_type(user_id, now, messages, wake_up_hour=wh)

    if msg_type is None:
        return None

    prompt = _build_proactive_message_prompt(msg_type, extra)
    logger.info(
        "General proactive → type=%s user=%s extra=%s",
        msg_type, user_id, extra,
    )

    user_state = {
        "anomaly_level": "low",
        "proactive_mode": True,
        "proactive_type": msg_type,
    }
    user_state.update(extra)

    try:
        return generate_chat_response(
            message=prompt,
            extracted={},
            context=messages[-5:] if messages else [],
            persona_id=persona_id,
            user_id=user_id,
            user_state=user_state,
            proactive=True,
            burst=True,
        )
    except Exception as e:
        logger.error("Failed to generate general proactive message: %s", e)
        return None
