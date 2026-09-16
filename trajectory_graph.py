"""
trajectory_graph.py — User trajectory modeling for Viora.

Two-layer architecture (per user design decision on 2026-06-25):
  Layer 1 — Raw event timeline: chronological events from chat extraction
  Layer 2 — Periodic trajectory patterns: extracted via daily/weekly/conditional detection

  StateNode is used internally as an aggregation intermediate (3-day window),
  but is NOT exposed as an independent layer.

Usage:
    graph = TrajectoryGraph()
    graph.ingest_messages(messages)     # Layer 1: extract events
    graph.detect_periodicities()        # Layer 2: find trajectory patterns
    graph.get_timeline()               # returns Layer 1 events
    graph.get_trajectories()           # returns Layer 2 patterns
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# China Standard Time (UTC+8)
CST = timezone(timedelta(hours=8))

# ── Constants ────────────────────────────────────────────────────────────────

STATE_WINDOW_DAYS = 3          # aggregation window for internal state
MIN_EVENTS_FOR_STATE = 2       # min events to form a state (user confirmed: 2-3)
MIN_TRAJECTORY_CONFIDENCE = 0.5  # minimum confidence to emit a trajectory

# ── Enums ────────────────────────────────────────────────────────────────────

class EventDimension(str, Enum):
    SLEEP = "sleep"
    ENERGY = "energy"
    MOOD = "mood"
    EXERCISE = "exercise"
    DIGESTION = "digestion"
    SKIN = "skin"
    PAIN = "pain"
    DIET = "diet"

class PeriodType(str, Enum):
    DAILY = "daily"         # same time-of-day across multiple days
    WEEKLY = "weekly"       # same weekday + time-of-day pattern
    CONDITIONAL = "conditional"  # event A → event B within a window

class EdgeType(str, Enum):
    OCCURS_AT = "occurs_at"           # event → timestamp (implicit)
    AGGREGATES_TO = "aggregates_to"   # events → internal state
    PERIODIC_PATTERN = "periodic_pattern"  # events → trajectory
    CONDITIONAL = "conditional"       # event type A precedes event type B

# ── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class EventNode:
    """Layer 1 — A single raw event extracted from a chat message."""
    id: str = field(default_factory=lambda: f"evt_{uuid4().hex[:12]}")
    dimension: EventDimension = EventDimension.MOOD
    value: float | None = None          # normalized score 0-10 or None
    label: str = ""                     # human-readable summary
    raw_text: str = ""                  # original message snippet
    timestamp: datetime | None = None   # when the event occurred
    source_message_id: str = ""         # backlink to original chat msg
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dimension": self.dimension.value,
            "value": self.value,
            "label": self.label,
            "raw_text": self.raw_text[:200],
            "timestamp": self.timestamp.astimezone(CST).isoformat() if self.timestamp else None,
            "source_message_id": self.source_message_id,
            "metadata": self.metadata,
        }


@dataclass
class StateNode:
    """Internal aggregation — 3-day windowed state (not exposed in UI)."""
    id: str = field(default_factory=lambda: f"stt_{uuid4().hex[:12]}")
    dimension: EventDimension = EventDimension.MOOD
    mean_value: float | None = None
    trend: str = "stable"               # "improving" | "declining" | "stable"
    event_count: int = 0
    window_start: datetime | None = None
    window_end: datetime | None = None
    event_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dimension": self.dimension.value,
            "mean_value": self.mean_value,
            "trend": self.trend,
            "event_count": self.event_count,
            "window_start": self.window_start.astimezone(CST).isoformat() if self.window_start else None,
            "window_end": self.window_end.astimezone(CST).isoformat() if self.window_end else None,
            "event_ids": self.event_ids,
        }


@dataclass
class TrajectoryNode:
    """Layer 2 — A periodic trajectory pattern detected from event timeline."""
    id: str = field(default_factory=lambda: f"trj_{uuid4().hex[:12]}")
    label: str = ""                     # "周四下午低能量"
    dimension: EventDimension = EventDimension.ENERGY
    period_type: PeriodType = PeriodType.WEEKLY
    period_unit: str = ""               # "hour: 14-17" or "weekday: 4"
    confidence: float = 0.0             # 0-1, how reliable this pattern is
    stability: float = 0.0             # 0-1, how consistently it repeats
    sample_count: int = 0               # how many occurrences observed
    evidence_event_ids: list[str] = field(default_factory=list)
    internal_state_ids: list[str] = field(default_factory=list)
    description: str = ""               # natural language summary
    created_at: datetime = field(default_factory=lambda: datetime.now(CST))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "dimension": self.dimension.value,
            "period_type": self.period_type.value,
            "period_unit": self.period_unit,
            "confidence": round(self.confidence, 3),
            "stability": round(self.stability, 3),
            "sample_count": self.sample_count,
            "evidence_event_ids": self.evidence_event_ids,
            "description": self.description,
            "created_at": self.created_at.astimezone(CST).isoformat(),
        }


@dataclass
class GraphEdge:
    """Relationship between two nodes (event→state→trajectory)."""
    id: str = field(default_factory=lambda: f"edg_{uuid4().hex[:12]}")
    source_id: str = ""
    target_id: str = ""
    edge_type: EdgeType = EdgeType.AGGREGATES_TO
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": round(self.weight, 3),
            "metadata": self.metadata,
        }


# ── Extraction Helpers ───────────────────────────────────────────────────────

def _parse_iso(s: str | None) -> datetime | None:
    """Safely parse an ISO timestamp string to a datetime."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _sanitize_str(value: Any) -> str:
    """Coerce a persisted string field to str, else "".

    Hand-edited / partially written trajectory JSON can store a number / dict
    where the schema expects a string. Such junk leaks into get_timeline()
    responses and crashes ``raw_text[:200]`` in EventNode.to_dict with
    TypeError. Well-formed strings are returned unchanged.
    """
    return value if isinstance(value, str) else ""


def _sanitize_optional_number(value: Any) -> float | int | None:
    """Coerce a persisted numeric field to int/float, else None.

    Corrupt entries may hold a non-numeric score ("high", a dict). Passing it
    through would leak out-of-schema data into the timeline response and crash
    the arithmetic in _make_state / _compute_consistency with TypeError.
    Bools are rejected (they are not scores); well-formed ints/floats (incl.
    None) are returned unchanged.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _sanitize_count(value: Any, default: int = 0) -> int:
    """Coerce a persisted counter to a non-negative int, else *default*."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value >= 0 else default
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


def _sanitize_dict(value: Any) -> dict:
    """Return *value* if it is a dict, else a fresh empty dict."""
    return value if isinstance(value, dict) else {}


def _sanitize_str_list(value: Any) -> list[str]:
    """Return the str items of a persisted list, dropping junk / non-lists."""
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str)]


def _sanitize_dimension(value: Any,
                        default: EventDimension = EventDimension.MOOD) -> EventDimension:
    """Coerce a persisted dimension key to EventDimension, else *default*.

    A corrupt/legacy file may hold an unknown key; ``EventDimension(...)``
    raises ValueError, which load_events previously let escape (500 on the
    trajectory API). Unknown / missing / non-string keys degrade to *default*.
    """
    if isinstance(value, EventDimension):
        return value
    if isinstance(value, str):
        try:
            return EventDimension(value)
        except ValueError:
            return default
    return default


def _sanitize_period_type(value: Any,
                          default: PeriodType = PeriodType.DAILY) -> PeriodType:
    """Coerce a persisted period_type key to PeriodType, else *default*."""
    if isinstance(value, PeriodType):
        return value
    if isinstance(value, str):
        try:
            return PeriodType(value)
        except ValueError:
            return default
    return default


def _extract_dimension_value(extracted: dict, dim: EventDimension) -> tuple[float | None, str]:
    """
    Extract a normalized numeric value and label from the extracted_data dict
    for a given dimension.
    """
    data = extracted.get(dim.value, {})
    if not isinstance(data, dict):
        return None, ""

    dim_label_map = {
        EventDimension.SLEEP: lambda d: (d.get("quality"), f"睡眠质量 {d.get('quality', '?')}/10"),
        EventDimension.ENERGY: lambda d: (d.get("score"), f"精力 {d.get('score', '?')}/10"),
        EventDimension.MOOD: lambda d: (d.get("score"), f"情绪 {d.get('score', '?')}/10"),
        EventDimension.EXERCISE: lambda d: (d.get("duration_minutes"), f"运动 {d.get('duration_minutes', '?')}分钟"),
        EventDimension.DIGESTION: lambda d: (_severity_to_score(d.get("severity")), f"肠胃: {d.get('status', '?')}"),
        EventDimension.SKIN: lambda d: (None, f"皮肤: {d.get('status', '?')}"),
        EventDimension.PAIN: lambda d: (_severity_to_score(d.get("severity")), f"疼痛 ({d.get('location', '?')})"),
        EventDimension.DIET: lambda d: (None, f"饮食: {d.get('notes', '?')[:50]}"),
    }

    handler = dim_label_map.get(dim)
    if handler:
        return handler(data)
    return None, ""


def _severity_to_score(severity: int | None) -> float | None:
    """Convert severity (1-5) to 0-10 score (inverted: high severity = low score)."""
    if severity is None:
        return None
    return max(0.0, 10.0 - severity * 2.0)


def _get_hour_key(dt: datetime) -> str:
    """Return hour-of-day string key like '14'."""
    cst = dt.astimezone(CST)
    return str(cst.hour)


def _get_weekday_key(dt: datetime) -> str:
    """Return weekday key 0=Mon .. 6=Sun in CST."""
    cst = dt.astimezone(CST)
    return str(cst.weekday())


def _get_date_key(dt: datetime) -> str:
    """Return date string YYYY-MM-DD in CST."""
    return dt.astimezone(CST).strftime("%Y-%m-%d")


def _get_week_key(dt: datetime) -> str:
    """Return ISO week string YYYY-Www in CST."""
    iso = dt.astimezone(CST).isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


# ── Main Graph ───────────────────────────────────────────────────────────────

class TrajectoryGraph:
    """
    Two-layer user trajectory model.

    Layer 1 (events): ingest_messages() → extracts EventNodes chronologically.
    Layer 2 (trajectories): detect_periodicities() → finds repeating patterns.

    Internal states are computed on-the-fly during periodicity detection
    but are NOT exposed via the public API.
    """

    def __init__(self):
        self.events: list[EventNode] = []           # Layer 1
        self.trajectories: list[TrajectoryNode] = []  # Layer 2
        self.edges: list[GraphEdge] = []             # edges across layers
        self._states: list[StateNode] = []           # internal, not exposed

    # ── Layer 1: Ingestion ─────────────────────────────────────────────────

    def ingest_messages(self, messages: list[dict]) -> int:
        """
        Process a list of message dicts (from storage.list_messages) and
        extract EventNodes. Returns the number of events created.
        """
        count = 0
        for msg in messages:
            extracted = msg.get("extracted_data", {})
            if not isinstance(extracted, dict) or not extracted:
                continue

            ts = _parse_iso(msg.get("timestamp"))
            msg_id = msg.get("id", "")
            content = msg.get("content", "")

            for dim in EventDimension:
                # Skip dimensions not present in the extracted data
                dim_data = extracted.get(dim.value, {})
                if not isinstance(dim_data, dict) or all(v is None for v in dim_data.values()):
                    continue
                val, label = _extract_dimension_value(extracted, dim)

                event = EventNode(
                    dimension=dim,
                    value=val,
                    label=label,
                    raw_text=content[:200],
                    timestamp=ts or datetime.now(CST),
                    source_message_id=msg_id,
                )
                self.events.append(event)
                count += 1

        # Sort events chronologically
        self.events.sort(key=lambda e: e.timestamp or datetime.min)
        logger.info("Ingested %d events from %d messages", count, len(messages))
        return count

    # ── Layer 2: Periodicity Detection ──────────────────────────────────────

    def detect_periodicities(self) -> list[TrajectoryNode]:
        """
        Run all periodicity detectors and return discovered trajectories.
        Clears any previously detected trajectories.
        """
        self.trajectories.clear()
        self._states.clear()
        self.edges.clear()

        if len(self.events) < MIN_EVENTS_FOR_STATE:
            logger.info("Too few events (%d) for periodicity detection", len(self.events))
            return []

        # Build internal states (3-day windows per dimension)
        self._build_internal_states()

        # Run detectors
        self._detect_daily_patterns()
        self._detect_weekly_patterns()
        self._detect_conditional_patterns()

        logger.info("Detected %d trajectory patterns", len(self.trajectories))
        return self.trajectories

    def _build_internal_states(self) -> None:
        """Aggregate events into 3-day internal StateNodes per dimension."""
        # Group events by dimension, then by 3-day window
        by_dim: dict[str, list[EventNode]] = defaultdict(list)
        for evt in self.events:
            by_dim[evt.dimension.value].append(evt)

        for dim_key, dim_events in by_dim.items():
            dim = EventDimension(dim_key)
            if len(dim_events) < MIN_EVENTS_FOR_STATE:
                continue

            # Sort by timestamp and slide a 3-day window
            dim_events.sort(key=lambda e: e.timestamp or datetime.min)
            window_start = dim_events[0].timestamp
            window_events: list[EventNode] = []

            for evt in dim_events:
                if evt.timestamp is None:
                    continue
                if window_start is None:
                    window_start = evt.timestamp

                days_diff = (evt.timestamp - window_start).total_seconds() / 86400
                if days_diff <= STATE_WINDOW_DAYS:
                    window_events.append(evt)
                else:
                    # Flush current window
                    if len(window_events) >= MIN_EVENTS_FOR_STATE:
                        state = self._make_state(dim, window_events)
                        self._states.append(state)
                    # Start new window
                    window_start = evt.timestamp
                    window_events = [evt]

            # Flush last window
            if len(window_events) >= MIN_EVENTS_FOR_STATE:
                state = self._make_state(dim, window_events)
                self._states.append(state)

    def _make_state(self, dim: EventDimension, events: list[EventNode]) -> StateNode:
        """Create a StateNode from a list of same-dimension events."""
        values = [e.value for e in events if e.value is not None]
        mean_val = sum(values) / len(values) if values else None

        # Compute simple trend (compare first half vs second half)
        trend = "stable"
        if len(values) >= 4:
            mid = len(values) // 2
            first_half = sum(values[:mid]) / mid
            second_half = sum(values[mid:]) / (len(values) - mid)
            diff = second_half - first_half
            if diff > 0.5:
                trend = "improving"
            elif diff < -0.5:
                trend = "declining"

        timestamps = [e.timestamp for e in events if e.timestamp]
        return StateNode(
            dimension=dim,
            mean_value=mean_val,
            trend=trend,
            event_count=len(events),
            window_start=min(timestamps) if timestamps else None,
            window_end=max(timestamps) if timestamps else None,
            event_ids=[e.id for e in events],
        )

    def _detect_daily_patterns(self) -> None:
        """
        Detect daily patterns: same hour-of-day across multiple days
        with consistent dimension values.
        """
        # Group events by (dimension, hour_key)
        buckets: dict[tuple[str, str], list[EventNode]] = defaultdict(list)
        for evt in self.events:
            if evt.timestamp is None:
                continue
            key = (evt.dimension.value, _get_hour_key(evt.timestamp))
            buckets[key].append(evt)

        for (dim_key, hour), events in buckets.items():
            if len(events) < 2:  # need at least 2 occurrences
                continue

            # Count unique days
            unique_days = set()
            for evt in events:
                if evt.timestamp:
                    unique_days.add(_get_date_key(evt.timestamp))

            if len(unique_days) < 2:  # need at least 2 different days
                continue

            dim = EventDimension(dim_key)
            values = [e.value for e in events if e.value is not None]
            consistency = self._compute_consistency(values)
            confidence = min(1.0, consistency * (len(unique_days) / 5.0))

            if confidence < MIN_TRAJECTORY_CONFIDENCE:
                continue

            hour_label = f"{int(hour):02d}:00-{int(hour):02d}:59"
            weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

            trj = TrajectoryNode(
                label=f"每日{hour_label} {dim.value}趋势",
                dimension=dim,
                period_type=PeriodType.DAILY,
                period_unit=f"hour: {hour}",
                confidence=confidence,
                stability=consistency,
                sample_count=len(unique_days),
                evidence_event_ids=[e.id for e in events],
                description=f"每天{hour_label}左右, {dim.value} 出现 {len(unique_days)} 次, 一致度 {consistency:.1%}",
            )
            self.trajectories.append(trj)
            self._link_events_to_trajectory(events, trj)

    def _detect_weekly_patterns(self) -> None:
        """
        Detect weekly patterns: same weekday + hour across multiple weeks.
        """
        buckets: dict[tuple[str, str, str], list[EventNode]] = defaultdict(list)
        for evt in self.events:
            if evt.timestamp is None:
                continue
            key = (evt.dimension.value, _get_weekday_key(evt.timestamp), _get_hour_key(evt.timestamp))
            buckets[key].append(evt)

        for (dim_key, weekday, hour), events in buckets.items():
            if len(events) < 2:
                continue

            # Count unique weeks
            unique_weeks = set()
            for evt in events:
                if evt.timestamp:
                    unique_weeks.add(_get_week_key(evt.timestamp))

            if len(unique_weeks) < 2:  # need at least 2 different weeks
                continue

            dim = EventDimension(dim_key)
            values = [e.value for e in events if e.value is not None]
            consistency = self._compute_consistency(values)
            confidence = min(1.0, consistency * (len(unique_weeks) / 4.0))

            if confidence < MIN_TRAJECTORY_CONFIDENCE:
                continue

            weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            wd_name = weekday_names[int(weekday)]
            hour_label = f"{int(hour):02d}:00"

            trj = TrajectoryNode(
                label=f"每周{wd_name}{hour_label} {dim.value}",
                dimension=dim,
                period_type=PeriodType.WEEKLY,
                period_unit=f"weekday: {weekday}, hour: {hour}",
                confidence=confidence,
                stability=consistency,
                sample_count=len(unique_weeks),
                evidence_event_ids=[e.id for e in events],
                description=f"每{wd_name}{hour_label}左右出现, 持续 {len(unique_weeks)} 周, 一致度 {consistency:.1%}",
            )
            self.trajectories.append(trj)
            self._link_events_to_trajectory(events, trj)

    def _detect_conditional_patterns(self) -> None:
        """
        Detect conditional patterns: event A (e.g., poor sleep) is frequently
        followed by event B (e.g., low energy) within a time window.
        """
        if len(self.events) < 4:
            return

        # Sort all events by time
        sorted_events = sorted(self.events, key=lambda e: e.timestamp or datetime.min)

        # Look for pairs (A, B) where A is NOT same dimension as B
        # and B occurs within 24 hours of A
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        pair_hours: dict[tuple[str, str], list[float]] = defaultdict(list)

        for i, event_a in enumerate(sorted_events):
            if event_a.timestamp is None:
                continue
            for j in range(i + 1, len(sorted_events)):
                event_b = sorted_events[j]
                if event_b.timestamp is None:
                    continue
                if event_b.dimension == event_a.dimension:
                    continue  # skip same-dimension pairs

                hours_diff = (event_b.timestamp - event_a.timestamp).total_seconds() / 3600
                if 0 < hours_diff <= 24:
                    pair = (event_a.dimension.value, event_b.dimension.value)
                    pair_counts[pair] += 1
                    pair_hours[pair].append(hours_diff)

        for (dim_a, dim_b), count in pair_counts.items():
            if count < 2:  # need at least 2 occurrences
                continue

            # Confidence based on: count / total_events * consistency of lag
            avg_lag = sum(pair_hours[(dim_a, dim_b)]) / count
            confidence = min(1.0, count / max(len(self.events) * 0.1, 1))

            if confidence < MIN_TRAJECTORY_CONFIDENCE:
                continue

            trj = TrajectoryNode(
                label=f"{dim_a} → {dim_b} (条件模式)",
                dimension=EventDimension(dim_b),
                period_type=PeriodType.CONDITIONAL,
                period_unit=f"lag_hours: {avg_lag:.1f}",
                confidence=confidence,
                stability=min(1.0, count / 5.0),
                sample_count=count,
                description=f"{dim_a} 之后平均 {avg_lag:.1f} 小时内出现 {dim_b} (观测 {count} 次)",
            )
            self.trajectories.append(trj)

            # Link: find the specific event pairs as evidence
            sorted_events = sorted(self.events, key=lambda e: e.timestamp or datetime.min)
            for i, event_a in enumerate(sorted_events):
                if event_a.timestamp is None or event_a.dimension.value != dim_a:
                    continue
                for j in range(i + 1, len(sorted_events)):
                    event_b = sorted_events[j]
                    if event_b.timestamp is None or event_b.dimension.value != dim_b:
                        continue
                    hours = (event_b.timestamp - event_a.timestamp).total_seconds() / 3600
                    if 0 < hours <= 24:
                        trj.evidence_event_ids.append(event_a.id)
                        trj.evidence_event_ids.append(event_b.id)
                        break  # one pair per A event

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_consistency(values: list[float | None]) -> float:
        """
        Compute how consistent the values are (1.0 = perfectly consistent,
        0.0 = wildly varying). Uses inverse of normalized std-dev.
        """
        clean = [v for v in values if v is not None]
        if len(clean) < 2:
            return 0.5

        mean = sum(clean) / len(clean)
        if mean == 0:
            return 0.5

        variance = sum((v - mean) ** 2 for v in clean) / len(clean)
        cv = (variance ** 0.5) / abs(mean)  # coefficient of variation
        return max(0.0, min(1.0, 1.0 - cv))

    def _link_events_to_trajectory(self, events: list[EventNode], trj: TrajectoryNode) -> None:
        """Create edges between events and their trajectory."""
        for evt in events:
            edge = GraphEdge(
                source_id=evt.id,
                target_id=trj.id,
                edge_type=EdgeType.PERIODIC_PATTERN,
            )
            self.edges.append(edge)

    # ── Persistence ────────────────────────────────────────────────────────

    def load_events(self, user_id: str) -> int:
        """
        Load persisted EventNodes from storage for a given user.
        Returns the number of events loaded.
        """
        from storage import load_trajectory_events
        raw = load_trajectory_events(user_id)
        self.events.clear()
        if not isinstance(raw, list):
            logger.warning(
                "Malformed trajectory events container for user %s; treating as empty",
                user_id,
            )
            raw = []
        for item in raw:
            if not isinstance(item, dict):
                logger.warning(
                    "Skipping non-dict trajectory event entry for user %s", user_id
                )
                continue
            ts = _parse_iso(item.get("timestamp"))
            evt = EventNode(
                id=_sanitize_str(item.get("id")),
                dimension=_sanitize_dimension(item.get("dimension")),
                value=_sanitize_optional_number(item.get("value")),
                label=_sanitize_str(item.get("label")),
                raw_text=_sanitize_str(item.get("raw_text")),
                timestamp=ts,
                source_message_id=_sanitize_str(item.get("source_message_id")),
                metadata=_sanitize_dict(item.get("metadata")),
            )
            self.events.append(evt)
        self.events.sort(key=lambda e: e.timestamp or datetime.min)
        logger.info("Loaded %d trajectory events for user %s", len(self.events), user_id)
        return len(self.events)

    def save_events(self, user_id: str) -> None:
        """Persist current EventNodes to storage for a given user."""
        from storage import replace_trajectory_events
        raw = [e.to_dict() for e in self.events]
        replace_trajectory_events(user_id, raw)
        logger.info("Saved %d trajectory events for user %s", len(raw), user_id)

    def save_trajectories(self, user_id: str) -> None:
        """Persist detected trajectory patterns to storage."""
        from storage import save_trajectories as _save_trj
        raw = self.get_trajectories()
        _save_trj(user_id, raw)
        logger.info("Saved %d trajectory patterns for user %s", len(raw), user_id)

    def load_trajectories(self, user_id: str) -> int:
        """Load persisted trajectory patterns from storage."""
        from storage import load_trajectories as _load_trj
        raw = _load_trj(user_id)
        self.trajectories.clear()
        if not isinstance(raw, list):
            logger.warning(
                "Malformed trajectory patterns container for user %s; treating as empty",
                user_id,
            )
            raw = []
        for item in raw:
            if not isinstance(item, dict):
                logger.warning(
                    "Skipping non-dict trajectory pattern entry for user %s", user_id
                )
                continue
            confidence = _sanitize_optional_number(item.get("confidence"))
            stability = _sanitize_optional_number(item.get("stability"))
            trj = TrajectoryNode(
                id=_sanitize_str(item.get("id")),
                label=_sanitize_str(item.get("label")),
                dimension=_sanitize_dimension(item.get("dimension"), EventDimension.ENERGY),
                period_type=_sanitize_period_type(item.get("period_type")),
                period_unit=_sanitize_str(item.get("period_unit")),
                confidence=confidence if confidence is not None else 0.0,
                stability=stability if stability is not None else 0.0,
                sample_count=_sanitize_count(item.get("sample_count")),
                evidence_event_ids=_sanitize_str_list(item.get("evidence_event_ids")),
                description=_sanitize_str(item.get("description")),
                created_at=_parse_iso(item.get("created_at")) or datetime.now(CST),
            )
            self.trajectories.append(trj)
        logger.info("Loaded %d trajectory patterns for user %s", len(self.trajectories), user_id)
        return len(self.trajectories)

    def incremental_ingest(self, messages: list[dict], user_id: str) -> dict:
        """
        Incremental pipeline: load existing events → ingest new messages →
        detect patterns → persist everything.

        Skips messages that already have corresponding events (by source_message_id).

        Returns summary dict with counts of new events and detected trajectories.
        """
        # Load existing events
        self.load_events(user_id)

        # Load existing trajectories (to preserve them in case detection fails)
        self.load_trajectories(user_id)

        # Track which message IDs we already have events for
        existing_msg_ids = set()
        for evt in self.events:
            if evt.source_message_id:
                existing_msg_ids.add(evt.source_message_id)

        # Filter to only new messages
        new_messages = [m for m in messages if m.get("id", "") not in existing_msg_ids]

        if not new_messages:
            # No new messages — update trajectory cache in storage and return empty
            self.save_events(user_id)
            self.save_trajectories(user_id)
            return {"new_events": 0, "total_events": len(self.events), "new_trajectories": 0}

        # Ingest only new messages
        new_event_count = self.ingest_messages(new_messages)

        # Re-detect periodicities on full event set
        new_trajectories = self.detect_periodicities()

        # Persist everything
        self.save_events(user_id)
        self.save_trajectories(user_id)

        logger.info(
            "Incremental ingest: %d new events → %d total events → %d trajectory patterns",
            new_event_count, len(self.events), len(new_trajectories),
        )
        return {
            "new_events": new_event_count,
            "total_events": len(self.events),
            "new_trajectories": len(new_trajectories),
        }

    # ── Query API ──────────────────────────────────────────────────────────

    def get_timeline(self, dimension: EventDimension | None = None,
                     since: datetime | None = None) -> list[dict]:
        """
        Return Layer 1 timeline events as dicts.
        Optionally filter by dimension or start time.
        """
        result = self.events
        if dimension:
            result = [e for e in result if e.dimension == dimension]
        if since:
            result = [e for e in result if e.timestamp and e.timestamp >= since]
        return [e.to_dict() for e in result]

    def get_trajectories(self, period_type: PeriodType | None = None,
                         dimension: EventDimension | None = None) -> list[dict]:
        """
        Return Layer 2 trajectory patterns as dicts.
        Optionally filter by period type or dimension.
        """
        result = self.trajectories
        if period_type:
            result = [t for t in result if t.period_type == period_type]
        if dimension:
            result = [t for t in result if t.dimension == dimension]
        return [t.to_dict() for t in result]

    def get_edges(self, edge_type: EdgeType | None = None) -> list[dict]:
        """Return all graph edges."""
        if edge_type:
            return [e.to_dict() for e in self.edges if e.edge_type == edge_type]
        return [e.to_dict() for e in self.edges]

    def to_dict(self) -> dict:
        """Export the full graph as a serializable dict."""
        return {
            "layer_1_events": [e.to_dict() for e in self.events],
            "layer_2_trajectories": [t.to_dict() for t in self.trajectories],
            "edges": [e.to_dict() for e in self.edges],
            "stats": {
                "total_events": len(self.events),
                "total_trajectories": len(self.trajectories),
                "total_edges": len(self.edges),
            },
        }
