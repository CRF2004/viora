"""
routine.py — Routine learning engine for Viora Phase 2.

Analyzes message history to build a model of user daily patterns:
- Sleep/wake times
- Exercise patterns (type, frequency, time of day)
- Meal patterns
- Energy/mood baselines
- Anomaly detection for proactive triggers
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# China Standard Time
CST = timezone(timedelta(hours=8))


# ── Deserialization / malformed-input hardening ──────────────────────────────
#
# Messages come from storage (LLM extraction persisted as JSON) and may be
# hand-edited / partially written. extracted_data fields documented as numeric
# could then hold a string ("7.5", "good") or a dict, which used to be stored
# as-is and crash the arithmetic in _build_routine_flags / detect_anomalies
# with TypeError (500 on /api/routine, /api/anomalies and the plan/chat flows).
# The helpers below follow the repo's established _sanitize_* pattern: junk
# degrades gracefully, well-formed values are returned unchanged.


def _message_dicts(messages) -> list[dict]:
    """Return only well-formed dict message entries.

    Tolerates a non-list container (None/dict/str/int) by degrading to an
    empty list, and skips non-dict entries so a single corrupt stored record
    cannot crash the whole routine-learning pass.
    """
    if not isinstance(messages, list):
        return []
    return [m for m in messages if isinstance(m, dict)]


def _sanitize_number(value):
    """Coerce a persisted numeric field to int/float, else None.

    Corrupt sleep.quality / sleep.duration_hours / exercise.duration_minutes
    values are dropped instead of being summed; bools are rejected (they are
    not measurements) and well-formed ints/floats (incl. None) are returned
    unchanged, so numeric output stays byte-identical.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _sanitize_str(value) -> str:
    """Coerce a persisted string field to str, else "".

    An exercise type of dict/list is unhashable and crashed
    ``set(exercise_types)`` in _build_routine_flags; non-str ids / labels /
    frequencies leak out-of-schema values into check_plan_adherence results.
    Well-formed strings are returned unchanged.
    """
    return value if isinstance(value, str) else ""


class RoutineModel:
    """Learns and tracks user daily patterns from message history."""

    def __init__(self):
        self.sleep_times: list[dict] = []       # {time_str, quality, hours}
        self.exercise_records: list[dict] = []  # {type, duration, time_of_day}
        self.mood_history: list[dict] = []      # {score, time_of_day}
        self.energy_history: list[dict] = []    # {score, time_of_day}
        self.digestion_history: list[dict] = [] # {status, severity}
        self.diet_notes: list[dict] = []        # {notes, time_of_day}
        self.interaction_times: list[datetime] = []  # when user messages
        self.routine_flags: dict = {}           # established patterns

    def learn_from_messages(self, messages: list[dict]) -> None:
        """Process all messages and extract routine patterns."""
        for msg in _message_dicts(messages):
            self._extract_from_message(msg)
        self._build_routine_flags()

    def _extract_from_message(self, msg: dict) -> None:
        """Extract routine data from a single message."""
        ts = self._parse_timestamp(msg.get("timestamp", ""))
        if not ts:
            return

        extracted = msg.get("extracted_data", {})
        if not isinstance(extracted, dict):
            return

        time_of_day = self._time_of_day(ts.hour)

        # Sleep data
        sleep = extracted.get("sleep")
        if isinstance(sleep, dict):
            quality = _sanitize_number(sleep.get("quality"))
            duration_hours = _sanitize_number(sleep.get("duration_hours"))
            if quality is not None or duration_hours is not None:
                self.sleep_times.append({
                    "time": ts.astimezone(CST).strftime("%H:%M"),
                    "quality": quality,
                    "duration_hours": duration_hours,
                    "day_of_week": ts.weekday(),
                })

        # Exercise
        exercise = extracted.get("exercise")
        if isinstance(exercise, dict):
            exercise_type = _sanitize_str(exercise.get("type"))
            duration_minutes = _sanitize_number(exercise.get("duration_minutes"))
            if exercise_type or duration_minutes:
                self.exercise_records.append({
                    "type": exercise_type,
                    "duration_minutes": duration_minutes,
                    "time_of_day": time_of_day,
                    "day_of_week": ts.weekday(),
                })

        # Mood
        mood = extracted.get("mood")
        if mood is not None and isinstance(mood, (int, float)):
            self.mood_history.append({"score": int(mood), "time_of_day": time_of_day, "ts": ts.isoformat()})

        # Energy
        energy = extracted.get("energy")
        if energy is not None and isinstance(energy, (int, float)):
            self.energy_history.append({"score": int(energy), "time_of_day": time_of_day, "ts": ts.isoformat()})

        # Digestion
        digestion = extracted.get("digestion")
        if isinstance(digestion, dict) and digestion.get("status"):
            self.digestion_history.append({
                "status": digestion["status"],
                "severity": digestion.get("severity"),
            })

        # Diet
        diet = extracted.get("diet")
        if isinstance(diet, dict) and diet.get("notes"):
            self.diet_notes.append({"notes": diet["notes"], "time_of_day": time_of_day})

        # Track interaction time
        self.interaction_times.append(ts)

    def _parse_timestamp(self, ts_str: str) -> Optional[datetime]:
        """Parse ISO timestamp string."""
        try:
            return datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            return None

    def _time_of_day(self, hour: int) -> str:
        """Categorize hour into time-of-day bucket."""
        if 0 <= hour < 6:
            return "凌晨"
        elif 6 <= hour < 9:
            return "早晨"
        elif 9 <= hour < 12:
            return "上午"
        elif 12 <= hour < 14:
            return "中午"
        elif 14 <= hour < 18:
            return "下午"
        elif 18 <= hour < 21:
            return "傍晚"
        else:
            return "深夜"

    def _build_routine_flags(self) -> None:
        """Derive established patterns from accumulated data."""
        flags = {}

        # Sleep pattern (need at least 2 data points)
        if len(self.sleep_times) >= 2:
            qualities = [q for q in
                         (_sanitize_number(s.get("quality")) for s in self.sleep_times) if q]
            durations = [d for d in
                         (_sanitize_number(s.get("duration_hours")) for s in self.sleep_times) if d]
            if qualities:
                flags["avg_sleep_quality"] = round(sum(qualities) / len(qualities), 1)
            if durations:
                flags["avg_sleep_duration"] = round(sum(durations) / len(durations), 1)
            flags["sleep_data_points"] = len(self.sleep_times)

        # Exercise pattern
        if len(self.exercise_records) >= 1:
            flags["exercise_count"] = len(self.exercise_records)
            exercise_types = [t for t in
                              (_sanitize_str(r.get("type")) for r in self.exercise_records) if t]
            if exercise_types:
                most_common = max(set(exercise_types), key=exercise_types.count)
                flags["primary_exercise_type"] = most_common
            durations = [d for d in
                         (_sanitize_number(r.get("duration_minutes")) for r in self.exercise_records) if d]
            if durations:
                flags["avg_exercise_duration"] = round(sum(durations) / len(durations), 0)

        # Mood baseline
        if len(self.mood_history) >= 2:
            scores = [m["score"] for m in self.mood_history]
            flags["avg_mood"] = round(sum(scores) / len(scores), 1)
            flags["mood_data_points"] = len(scores)

        # Energy baseline
        if len(self.energy_history) >= 2:
            scores = [e["score"] for e in self.energy_history]
            flags["avg_energy"] = round(sum(scores) / len(scores), 1)
            flags["energy_data_points"] = len(scores)

        # Interaction pattern (when does the user usually message)
        if len(self.interaction_times) >= 2:
            hours = [t.hour for t in self.interaction_times]
            flags["avg_interaction_hour"] = round(sum(hours) / len(hours), 0)
            flags["interaction_count"] = len(hours)

        self.routine_flags = flags

    def get_routine_summary(self) -> dict:
        """Return a human-readable summary of learned patterns."""
        if not self.routine_flags:
            return {"status": "data_insufficient", "message": "还需要更多交互数据才能建立作息模型"}

        return {
            "status": "active",
            "data_points": {
                "sleep": len(self.sleep_times),
                "exercise": len(self.exercise_records),
                "mood": len(self.mood_history),
                "energy": len(self.energy_history),
                "total_messages": len(self.interaction_times),
            },
            "patterns": self.routine_flags,
            "confidence": self._confidence_level(),
        }

    def _confidence_level(self) -> str:
        """Estimate confidence in the routine model."""
        total = sum(len(v) for v in [
            self.sleep_times, self.exercise_records,
            self.mood_history, self.energy_history,
        ])
        if total >= 20:
            return "high"
        elif total >= 10:
            return "medium"
        elif total >= 5:
            return "low"
        return "very_low"

    def detect_anomalies(self) -> list[dict]:
        """Detect anomalies in recent behavior compared to established patterns."""
        anomalies = []
        now = datetime.now(CST)

        # Check sleep quality anomaly
        if len(self.sleep_times) >= 2:
            qualities = [q for q in
                         (_sanitize_number(s.get("quality")) for s in self.sleep_times[-5:]) if q]
            if qualities:
                recent_avg = sum(qualities[-3:]) / min(3, len(qualities))
                overall_avg = sum(qualities) / len(qualities)
                if recent_avg < overall_avg - 1.0:
                    anomalies.append({
                        "type": "sleep_decline",
                        "severity": "medium",
                        "description": f"最近睡眠质量下降了（{recent_avg:.1f} vs 平均 {overall_avg:.1f}）",
                        "suggested_action": "sleep_inquiry",
                    })

        # Check mood anomaly
        if len(self.mood_history) >= 2:
            scores = [s for s in
                      (_sanitize_number(m.get("score")) for m in self.mood_history[-5:])
                      if s is not None]
            if scores:
                recent_avg = sum(scores[-3:]) / min(3, len(scores))
                overall_avg = sum(scores) / len(scores)
                if recent_avg < overall_avg - 1.0:
                    anomalies.append({
                        "type": "mood_decline",
                        "severity": "high",
                        "description": f"最近情绪偏低（{recent_avg:.1f} vs 平均 {overall_avg:.1f}）",
                        "suggested_action": "mood_check",
                    })

        # Check exercise gap
        if len(self.exercise_records) >= 3:
            # Get last exercise date
            recent_exercise = self.exercise_records[-3:]
            has_exercise_today = any(
                r.get("day_of_week") == now.weekday() for r in self.exercise_records[-5:]
            )
            if not has_exercise_today and self.routine_flags.get("primary_exercise_type"):
                # Check if user usually exercises on this day
                day_records = [r for r in self.exercise_records if r.get("day_of_week") == now.weekday()]
                if len(day_records) >= 2:
                    hour = now.hour
                    # If it's past the usual exercise time
                    exercise_times = [r.get("time_of_day") for r in day_records]
                    if hour >= 20 and exercise_times:
                        anomalies.append({
                            "type": "exercise_missed",
                            "severity": "low",
                            "description": f"今天还没有运动记录（平时{''.join(exercise_times)}都有运动）",
                            "suggested_action": "gentle_reminder",
                        })

        return anomalies

    def check_plan_adherence(self, leaves: list) -> list[dict]:
        """Check adherence to a health plan's actionable items.

        Compares each leaf's completion record against its expected frequency
        to identify neglected items that may need adjustment.

        Args:
            leaves: List of ReasoningNode objects (or dict-like objects) with
                    keys: id, label, action_text, frequency, time_slot,
                    completed_count, last_completed, status.

        Returns:
            List of dicts:
                leaf_id, label, action_text, frequency, time_slot,
                adherence ("good"|"flag"|"missed"), days_missed (int),
                suggestion (str | None)
        """
        if not leaves:
            return []

        now = datetime.now(CST)
        results = []

        for leaf in leaves:
            leaf_id: str = ""
            label: str = ""
            action_text: str = ""
            frequency: str = ""
            time_slot: str = ""
            completed_count: int = 0
            last_completed_str: str | None = None

            if isinstance(leaf, dict):
                leaf_id = _sanitize_str(leaf.get("id"))
                label = _sanitize_str(leaf.get("label"))
                action_text = _sanitize_str(leaf.get("action_text")) or label
                frequency = _sanitize_str(leaf.get("frequency"))
                time_slot = _sanitize_str(leaf.get("time_slot"))
                completed_count = leaf.get("completed_count", 0)
                last_completed_str = leaf.get("last_completed")
            else:
                leaf_id = _sanitize_str(getattr(leaf, "id", ""))
                label = _sanitize_str(getattr(leaf, "label", ""))
                action_text = _sanitize_str(getattr(leaf, "action_text", None)) or label
                frequency = _sanitize_str(getattr(leaf, "frequency", ""))
                time_slot = _sanitize_str(getattr(leaf, "time_slot", ""))
                completed_count = getattr(leaf, "completed_count", 0)
                last_completed_str = getattr(leaf, "last_completed", None)

            # Parse last completed date
            last_date = None
            if last_completed_str:
                try:
                    dt = datetime.fromisoformat(last_completed_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=CST)
                    last_date = dt.astimezone(CST).date()
                except (ValueError, TypeError):
                    pass

            today = now.date()
            days_missed = 0
            if last_date:
                days_missed = (today - last_date).days

            # Determine expected frequency in days
            if frequency == "每天":
                expected_gap_days = 1
            elif frequency and "每周" in frequency:
                expected_gap_days = 3
            elif frequency and "隔天" in frequency:
                expected_gap_days = 2
            elif frequency and "每月" in frequency:
                expected_gap_days = 14
            else:
                expected_gap_days = 3

            # Determine adherence level
            if not last_date:
                adherence = "missed"
                suggestion = f"还没有开始「{action_text}」，是不是这个计划不太适合？"
            elif days_missed > expected_gap_days * 2:
                adherence = "missed"
                suggestion = f"已经{days_missed}天没执行「{action_text}」了，需要调整频率或换一种方式吗？"
            elif days_missed > expected_gap_days:
                adherence = "flag"
                suggestion = f"最近{days_missed}天没按计划{action_text}，是不是太忙了？"
            else:
                adherence = "good"
                suggestion = None

            results.append({
                "leaf_id": leaf_id,
                "label": label,
                "action_text": action_text,
                "frequency": frequency,
                "time_slot": time_slot,
                "adherence": adherence,
                "days_missed": days_missed if last_date else max(days_missed, 1),
                "suggestion": suggestion,
            })

        return results


def build_routine_model(messages: list[dict]) -> RoutineModel:
    """Convenience function: build a RoutineModel from message list."""
    model = RoutineModel()
    model.learn_from_messages(messages)
    return model
