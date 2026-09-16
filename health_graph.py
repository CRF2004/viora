"""
health_graph.py — Health dimension correlation graph for Viora.

Generates a graph representation where:
- Nodes are health dimensions (睡眠, 精力, 情绪, 运动, 肠胃, etc.)
- Edges represent correlation strength between dimensions
- Edge thickness/color reflects correlation coefficient

Also generates a personal health graph from user's message history.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Dimension definitions ───────────────────────────────────────────────────

DIMENSION_NODES = [
    {"id": "sleep", "label": "睡眠", "icon": "🌙", "color": "#9b8ec4"},
    {"id": "energy", "label": "精力", "icon": "⚡", "color": "#d4b86a"},
    {"id": "mood", "label": "情绪", "icon": "😊", "color": "#e8a87c"},
    {"id": "exercise", "label": "运动", "icon": "🏃", "color": "#a8b5a0"},
    {"id": "digestion", "label": "肠胃", "icon": "🍽️", "color": "#d4a5a5"},
    {"id": "skin", "label": "皮肤", "icon": "✨", "color": "#7eb5a6"},
    {"id": "pain", "label": "疼痛", "icon": "⚠️", "color": "#c9c0b6"},
    {"id": "diet", "label": "饮食", "icon": "🥗", "color": "#a8b5a0"},
]

DIMENSION_MAP = {d["id"]: d for d in DIMENSION_NODES}


# ── Deserialization / malformed-input hardening ──────────────────────────────
#
# /api/health-graph passes storage messages (LLM extraction persisted as JSON,
# possibly hand-edited or partially written) straight through, plus the
# correlations computed from them. A corrupt record used to crash the whole
# endpoint: non-dict messages / entries hit ``m.get`` (AttributeError), a
# non-numeric sleep quality or energy/mood score hit ``sum()`` (TypeError) and
# a non-string correlation label / coefficient hit ``in`` / ``abs()``
# (TypeError). The helpers below follow the repo's established _sanitize_*
# pattern: junk degrades gracefully, well-formed values are returned unchanged.


def _sanitize_number(value) -> Optional[float | int]:
    """Coerce a persisted numeric field to int/float, else None.

    Corrupt sleep.quality / energy.score / mood.score values ("good", a dict)
    were appended to the score list verbatim and crashed
    ``sum(scores)`` in _compute_dimension_avg with TypeError. Bools are rejected
    (they are not measurements); well-formed ints/floats (including 0) are
    returned unchanged so numeric output stays byte-identical.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _sanitize_str(value) -> str:
    """Coerce a persisted string field to str, else "".

    A non-string correlation label crashed ``label in dim_info["label"]`` in
    _find_dimension_id with TypeError, and a non-string description leaked an
    out-of-schema dict/list into the /api/health-graph response. Well-formed
    strings are returned unchanged.
    """
    return value if isinstance(value, str) else ""


def _message_dicts(messages) -> list[dict]:
    """Return only well-formed dict message entries.

    Tolerates a non-list container (None/dict/str/int) by degrading to an
    empty list, and skips non-dict entries so a single corrupt stored record
    cannot crash the whole graph build.
    """
    if not isinstance(messages, list):
        return []
    return [m for m in messages if isinstance(m, dict)]


def _correlation_dicts(correlations) -> list[dict]:
    """Return only well-formed dict correlation entries (non-list -> [])."""
    if not isinstance(correlations, list):
        return []
    return [c for c in correlations if isinstance(c, dict)]


def generate_health_graph(correlations: list = None, messages: list = None) -> dict:
    """
    Generate a health correlation graph.

    Args:
        correlations: List of {dimension1, dimension2, correlation, n} dicts
        messages: Raw messages (used to compute correlations if not provided)

    Returns:
        Graph data: {nodes: [...], relationships: [...]}
    """
    from insights import compute_correlations

    # Drop corrupt stored records / entries up-front so every downstream
    # aggregation (scores, data_points) only ever sees dict messages.
    messages = _message_dicts(messages)

    # Compute correlations if not provided
    if correlations is None and messages:
        correlations = compute_correlations(messages)

    correlations = _correlation_dicts(correlations)

    # Create nodes
    nodes = []
    for dim in DIMENSION_NODES:
        node = dict(dim)
        # Add average score from messages
        if messages:
            avg = _compute_dimension_avg(messages, dim["id"])
            node["properties"] = {"avg_score": avg, "data_points": len([m for m in messages if _has_dimension(m, dim["id"])])}
        else:
            node["properties"] = {"avg_score": None, "data_points": 0}
        nodes.append(node)

    # Create center node
    center_node = {
        "id": "center",
        "label": "你",
        "icon": "🧑",
        "color": "#e8a87c",
        "properties": {"name": "你的健康"},
    }
    nodes.insert(0, center_node)

    # Create edges from correlations
    relationships = []
    for corr in correlations:
        # Bools / strings / dicts are not coefficients: degrade to None and skip
        # the edge rather than emitting a bogus 0-strength "负向" relationship.
        strength = _sanitize_number(corr.get("correlation", 0))
        if strength is None:
            continue

        d1 = _sanitize_str(corr.get("dimension1"))
        d2 = _sanitize_str(corr.get("dimension2"))

        # Find matching dimension IDs
        d1_id = _find_dimension_id(d1)
        d2_id = _find_dimension_id(d2)

        if d1_id and d2_id and d1_id != d2_id:
            abs_corr = abs(strength)
            relationships.append({
                "source": d1_id,
                "target": d2_id,
                "type": "关联",
                "strength": round(abs_corr, 2),
                "direction": "正向" if strength > 0 else "负向",
                "weight": max(1, int(abs_corr * 10)),
                "color": _correlation_color(strength),
                "description": _sanitize_str(corr.get("description")),
            })

    # Always connect all dimensions to center
    for dim in DIMENSION_NODES:
        relationships.append({
            "source": "center",
            "target": dim["id"],
            "type": "包含",
            "strength": 0,
            "direction": "",
            "weight": 1,
            "color": "#c9c0b6",
            "description": "",
        })

    return {"nodes": nodes, "relationships": relationships}


def _find_dimension_id(label: str) -> Optional[str]:
    """Find dimension ID from Chinese label.

    Non-string labels used to crash ``label in dim_info["label"]`` with
    TypeError and an empty label used to fuzzy-match the first dimension
    (``"" in "睡眠"`` is True), fabricating an edge. Both now return None.
    """
    if not isinstance(label, str) or not label:
        return None
    for dim_id, dim_info in DIMENSION_MAP.items():
        if dim_info["label"] == label:
            return dim_id
    # Fuzzy match
    for dim_id, dim_info in DIMENSION_MAP.items():
        if dim_info["label"] in label or label in dim_info["label"]:
            return dim_id
    return None


def _correlation_color(corr: float) -> str:
    """Return color based on correlation strength and direction."""
    abs_corr = abs(corr)
    if abs_corr >= 0.7:
        return "#e8a87c"  # Strong positive: peach
    elif abs_corr >= 0.5:
        return "#a8b5a0"  # Medium: sage green
    elif corr < 0:
        return "#d4a5a5"  # Negative: rose
    else:
        return "#c9c0b6"  # Weak: grey


def _compute_dimension_avg(messages: list, dim_id: str) -> Optional[float]:
    """Compute average score for a dimension."""
    scores = []
    for m in _message_dicts(messages):
        ed = m.get("extracted_data", {})
        if not isinstance(ed, dict):
            continue

        if dim_id == "sleep":
            sleep = ed.get("sleep", {})
            if isinstance(sleep, dict):
                # Non-numeric quality ("good", a dict) used to reach sum().
                q = _sanitize_number(sleep.get("quality"))
                if q is not None:
                    scores.append(q)
        elif dim_id in ("energy", "mood"):
            val = ed.get(dim_id)
            # Handle new dict format: {"score": 3, "reason": "..."}
            if isinstance(val, dict):
                s = _sanitize_number(val.get("score"))
                if s is not None:
                    scores.append(s)
            # Handle old int format (backward compat): energy: 3
            elif isinstance(val, (int, float)):
                s = _sanitize_number(val)
                if s is not None:
                    scores.append(s)

    return round(sum(scores) / len(scores), 1) if scores else None


# Maps dimension IDs to their sub-fields for dict-type dimensions.
# Mirrors the DIM_FIELDS mapping in app.py's /api/health-graph/points endpoint.
_DIM_FIELDS = {
    "sleep": ["quality", "duration_hours"],
    "energy": ["score"],
    "mood": ["score"],
    "exercise": ["type", "duration_minutes"],
    "digestion": ["status", "severity"],
    "skin": ["status"],
    "pain": ["location", "severity"],
    "diet": ["notes"],
}


def _has_dimension(message: dict, dim_id: str) -> bool:
    """Check if a message has data for a dimension.

    Mirrors the filtering logic in app.py's /api/health-graph/points endpoint
    so that the data_points count matches actual point records returned.
    """
    if not isinstance(message, dict):
        return False

    ed = message.get("extracted_data", {})
    if not isinstance(ed, dict):
        return False

    dim_data = ed.get(dim_id)
    if dim_data is None:
        return False

    fields = _DIM_FIELDS.get(dim_id, [])

    if fields:
        # Dict-type dimensions (including new energy/mood format): at least one sub-field must be non-None
        if isinstance(dim_data, dict):
            for field in fields:
                if dim_data.get(field) is not None:
                    return True
        # Backward compat: old int-format energy/mood
        if isinstance(dim_data, (int, float)) and dim_data is not None:
            return True
    else:
        # Numeric dimensions: value must be a non-None number
        if isinstance(dim_data, (int, float)) and dim_data is not None:
            return True

    return False
