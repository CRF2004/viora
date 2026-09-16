"""
user_state.py — lightweight persistent user state for Viora.

Tracks a compact summary of each user's long-term conversational and health state
so the response planner can make more consistent decisions without loading the
entire history every time.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Optional

from config import DATA_DIR

logger = logging.getLogger(__name__)

USER_STATE_FILE = os.path.join(DATA_DIR, "user_state.json")


def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(USER_STATE_FILE):
        with open(USER_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_all() -> dict:
    _ensure_data_dir()
    try:
        with open(USER_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, FileNotFoundError):
        logger.warning("Corrupt user state file, starting fresh.")
        return {}


def _write_all(states: dict) -> None:
    _ensure_data_dir()
    with open(USER_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(states, f, ensure_ascii=False, indent=2)


def _sanitize_count(value: Any, default: int = 0) -> int:
    """Coerce a stored count to a non-negative int, falling back on junk."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        if value != value:  # NaN
            return default
        return max(0, int(value))
    if isinstance(value, str):
        try:
            return max(0, int(float(value.strip())))
        except (ValueError, TypeError):
            return default
    return default


def _sanitize_optional_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Coerce a stored numeric optional to float, falling back on junk/NaN."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        value = float(value)
    elif isinstance(value, str):
        try:
            value = float(value.strip())
        except (ValueError, TypeError):
            return default
    else:
        return default
    if value != value:  # NaN
        return default
    return value


def _sanitize_optional_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _sanitize_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


_BASELINE_KEYS = ("sleep_baseline", "energy_baseline", "mood_baseline")
_TREND_KEYS = ("recent_mood_trend", "recent_energy_trend", "recent_sleep_trend")


def _sanitize_state(raw: dict) -> dict:
    """Return a copy of a stored user state with out-of-schema values repaired.

    Hand-edited / partially written user_state.json entries can carry wrong
    types (a count as a string, a baseline as a dict, a preferences container
    flattened to a scalar). Those values would either crash the update path
    (int()/float()/dict()) or leak into API responses and the prompt. Repair
    them here so callers always see the declared schema.
    """
    state = dict(raw)
    if "interaction_count" in state:
        state["interaction_count"] = _sanitize_count(state["interaction_count"])
    if "avg_reply_length" in state:
        state["avg_reply_length"] = _sanitize_optional_float(
            state["avg_reply_length"], 0.0
        )
    for key in _BASELINE_KEYS:
        if key in state:
            state[key] = _sanitize_optional_float(state[key], None)
    for key in _TREND_KEYS:
        if key in state:
            state[key] = _sanitize_optional_str(state[key])
    if "user_preferences" in state:
        state["user_preferences"] = _sanitize_dict(state["user_preferences"])
    if "routine_summary" in state:
        state["routine_summary"] = _sanitize_dict(state["routine_summary"])
    if "profile_notes" in state:
        state["profile_notes"] = _sanitize_optional_str(state["profile_notes"]) or ""
    return state


@dataclass
class UserState:
    user_id: str
    persona_id: str = "default"
    display_name: Optional[str] = None
    anomaly_level: str = "low"
    sleep_baseline: Optional[float] = None
    energy_baseline: Optional[float] = None
    mood_baseline: Optional[float] = None
    recent_mood_trend: Optional[str] = None
    recent_energy_trend: Optional[str] = None
    recent_sleep_trend: Optional[str] = None
    interaction_count: int = 0
    avg_reply_length: float = 0.0
    last_message_at: Optional[str] = None
    last_updated_at: str = field(default_factory=_now_iso)
    user_preferences: dict[str, Any] = field(default_factory=dict)
    routine_summary: dict[str, Any] = field(default_factory=dict)
    profile_notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def get_user_state(user_id: str, persona_id: Optional[str] = None) -> dict:
    states = _read_all()
    raw = states.get(user_id)
    if isinstance(raw, dict):
        raw = _sanitize_state(raw)
        raw.setdefault("user_id", user_id)
        raw.setdefault("profile_notes", "")
        if persona_id and not raw.get("persona_id"):
            raw["persona_id"] = persona_id
        return raw
    state = UserState(user_id=user_id, persona_id=persona_id or "default")
    return state.to_dict()


def update_user_state(
    user_id: str,
    *,
    persona_id: Optional[str] = None,
    display_name: Optional[str] = None,
    extracted_data: Optional[dict] = None,
    message_text: Optional[str] = None,
    routine_summary: Optional[dict] = None,
    anomaly_level: Optional[str] = None,
    user_preferences: Optional[dict] = None,
    profile_notes: Optional[str] = None,
) -> dict:
    states = _read_all()
    state = get_user_state(user_id, persona_id=persona_id)

    if persona_id:
        state["persona_id"] = persona_id
    if display_name is not None:
        state["display_name"] = display_name
    if anomaly_level:
        state["anomaly_level"] = anomaly_level
    if routine_summary is not None:
        state["routine_summary"] = routine_summary
    if user_preferences:
        merged = dict(state.get("user_preferences", {}))
        merged.update(user_preferences)
        state["user_preferences"] = merged
    if profile_notes is not None:
        state["profile_notes"] = profile_notes

    if extracted_data and isinstance(extracted_data, dict):
        sleep = extracted_data.get("sleep") if isinstance(extracted_data.get("sleep"), dict) else {}
        energy_raw = extracted_data.get("energy")
        mood_raw = extracted_data.get("mood")

        # Handle both old (int) and new (dict with "score" key) formats
        energy = energy_raw if isinstance(energy_raw, (int, float)) else (energy_raw.get("score") if isinstance(energy_raw, dict) else None)
        mood = mood_raw if isinstance(mood_raw, (int, float)) else (mood_raw.get("score") if isinstance(mood_raw, dict) else None)

        state["sleep_baseline"] = _update_baseline(state.get("sleep_baseline"), sleep.get("quality"))
        state["energy_baseline"] = _update_baseline(state.get("energy_baseline"), energy)
        state["mood_baseline"] = _update_baseline(state.get("mood_baseline"), mood)

        state["recent_sleep_trend"] = _trend_label(state.get("sleep_baseline"), sleep.get("quality"))
        state["recent_energy_trend"] = _trend_label(state.get("energy_baseline"), energy)
        state["recent_mood_trend"] = _trend_label(state.get("mood_baseline"), mood)

    if message_text:
        state["interaction_count"] = int(state.get("interaction_count", 0)) + 1
        current_avg = float(state.get("avg_reply_length", 0.0) or 0.0)
        count = state["interaction_count"]
        reply_len = len(message_text.strip())
        state["avg_reply_length"] = round(((current_avg * (count - 1)) + reply_len) / count, 2)
        state["last_message_at"] = _now_iso()

    state["last_updated_at"] = _now_iso()
    states[user_id] = state
    _write_all(states)
    return state


def delete_user_state(user_id: str) -> None:
    states = _read_all()
    if user_id in states:
        del states[user_id]
        _write_all(states)


def _update_baseline(previous: Optional[float], current: Optional[float]) -> Optional[float]:
    previous = _sanitize_optional_float(previous, None)
    current = _sanitize_optional_float(current, None)
    if current is None:
        return previous
    if previous is None:
        return round(current, 2)
    return round(previous * 0.8 + current * 0.2, 2)


def _trend_label(baseline: Optional[float], current: Optional[float]) -> Optional[str]:
    baseline = _sanitize_optional_float(baseline, None)
    current = _sanitize_optional_float(current, None)
    if baseline is None or current is None:
        return None
    delta = float(current) - float(baseline)
    if delta > 0.6:
        return "up"
    if delta < -0.6:
        return "down"
    return "stable"


def extract_wake_up_hour_from_profile(profile_notes: str) -> Optional[int]:
    """
    Extract wake-up time (hour 0-23) from user profile notes using regex.

    Looks for patterns like:
    - "早上7点起" / "七点起床" / "8点醒"
    - "早上七八点起" / "6-7点起床"
    - "常熬夜到凌晨两三点" (implies late wake-up)
    - "每天五点就醒了"

    Returns the wake-up hour as an integer (0-23), or None if not found.
    When multiple mentions exist, returns the LAST (most recent) one.
    Hours < 6 are assumed to be PM for wake-up (treated as +12).
    """
    if not profile_notes:
        return None

    # Chinese digit → integer mapping
    cn_nums = {
        "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        "十一": 11, "十二": 12,
    }

    # Match "X点" followed by a wake-up verb. No prefix anchoring — just
    # extract every mention of a wake-up time regardless of surrounding text.
    patterns: list[tuple[str, int]] = [
        # Digit form: "7点起", "8点半醒", "9点左右起床"
        (r"(\d{1,2})\s*点\s*(?:多|半|左右|钟)?\s*(?:起|醒|起床|起来|醒了|就醒)", 1),
        # Chinese single digit: "七点起", "八点醒"
        (r"([一二两三四五六七八九])\s*点\s*(?:多|半|左右|钟)?\s*(?:起|醒|起床|起来|醒了|就醒)", 1),
        # Chinese two-digit: "十点起", "十一点醒", "十二点起床"
        (r"(十[一二]?)\s*点\s*(?:多|半|左右|钟)?\s*(?:起|醒|起床|起来|醒了|就醒)", 1),
        # "每天X点醒" (sometimes "醒" alone without "起")
        (r"每天\s*(\d{1,2})\s*点\s*(?:就\s*)?醒", 1),
        (r"每天\s*([一二两三四五六七八九十])\s*点\s*(?:就\s*)?醒", 1),
    ]

    matches: list[tuple[int, int, int]] = []  # (start_pos, end_pos, hour)

    for pattern, group_idx in patterns:
        for m in re.finditer(pattern, profile_notes):
            hour_str = m.group(group_idx)
            if hour_str.isdigit():
                hour = int(hour_str)
            else:
                hour = cn_nums.get(hour_str)
            if hour is None:
                continue
            if 0 <= hour <= 23:
                matches.append((m.start(), m.end(), hour))

    if not matches:
        return None

    # Deduplicate overlapping matches (e.g., "十一点" vs "一点"):
    # When match A contains/overlaps match B, keep the longer match.
    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))  # earliest first, longest first
    deduped: list[tuple[int, int, int]] = []
    for m in matches:
        # Skip if this match is fully contained within an already-accepted match
        if any(d[0] <= m[0] and m[1] <= d[1] for d in deduped):
            continue
        # Skip if this match overlaps with a shorter already-accepted match
        # (longer match replaces shorter overlapping one)
        overlapping = [i for i, d in enumerate(deduped) if m[0] < d[1] and d[0] < m[1]]
        if overlapping:
            # Keep the longer match; if same length, keep later one
            for i in reversed(overlapping):
                d = deduped[i]
                if (m[1] - m[0]) >= (d[1] - d[0]):
                    deduped.pop(i)
                else:
                    break  # existing is longer, skip this match
            else:
                deduped.append(m)
        else:
            deduped.append(m)

    if not deduped:
        return None

    # Return the LAST (most recent) mention by position
    deduped.sort(key=lambda x: x[0])
    return deduped[-1][2]
