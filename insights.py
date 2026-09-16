"""
insights.py — Correlation and trend analysis for Viora Phase 2.

Computes simple statistical relationships between health dimensions
to power "body story" correlation insights.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _message_dicts(messages) -> list[dict]:
    """Return only well-formed dict message entries.

    Tolerates a non-list container (None/dict/str/int) by degrading to an
    empty list, and skips non-dict entries so a single corrupt stored record
    cannot crash the whole correlation/trend pass.
    """
    if not isinstance(messages, list):
        return []
    return [m for m in messages if isinstance(m, dict)]


def _extracted_data(msg: dict) -> dict:
    """Return a message's extracted_data if it is a dict, else an empty dict."""
    ed = msg.get("extracted_data")
    return ed if isinstance(ed, dict) else {}


def compute_correlations(messages: list[dict]) -> list[dict]:
    """
    Compute pairwise correlations between health dimensions.

    Returns list of {dimension1, dimension2, correlation, description}.
    Uses simple Pearson correlation on co-occurring data points.
    """
    # Extract paired scores from messages
    pairs = _extract_score_pairs(messages)

    dimensions = [
        ("sleep_quality", "sleep", "quality"),
        ("sleep_duration", "sleep", "duration_hours"),
        ("energy", "energy", None),
        ("mood", "mood", None),
    ]

    results = []
    for i in range(len(dimensions)):
        for j in range(i + 1, len(dimensions)):
            d1_name, d1_key, d1_sub = dimensions[i]
            d2_name, d2_key, d2_sub = dimensions[j]

            scores1 = [p[i] for p in pairs]
            scores2 = [p[j] for p in pairs]

            if len(scores1) < 3:
                continue

            corr = _pearson(scores1, scores2)
            if corr is not None and abs(corr) >= 0.3:
                desc = _describe_correlation(d1_name, d2_name, corr)
                results.append({
                    "dimension1": _dim_label(d1_name),
                    "dimension2": _dim_label(d2_name),
                    "correlation": round(corr, 2),
                    "n": len(scores1),
                    "description": desc,
                })

    # Sort by absolute correlation
    results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return results


def _extract_score_pairs(messages: list[dict]) -> list[tuple]:
    """Extract (sleep_quality, sleep_duration, energy, mood) tuples."""
    pairs = []
    for msg in _message_dicts(messages):
        ed = _extracted_data(msg)

        sq = _safe_num(ed.get("sleep"), "quality") if isinstance(ed.get("sleep"), dict) else None
        sd = _safe_num(ed.get("sleep"), "duration_hours") if isinstance(ed.get("sleep"), dict) else None
        en = _safe_num(ed.get("energy"))
        mo = _safe_num(ed.get("mood"))

        # Only keep pairs where we have at least 2 values
        values = [v for v in [sq, sd, en, mo] if v is not None]
        if len(values) >= 2:
            pairs.append((sq, sd, en, mo))

    return pairs


def _pearson(x: list[float], y: list[float]) -> Optional[float]:
    """Compute Pearson correlation coefficient."""
    # Filter out None values from both lists (paired removal)
    paired = [(xi, yi) for xi, yi in zip(x, y) if xi is not None and yi is not None]
    if len(paired) < 3:
        return None

    x = [p[0] for p in paired]
    y = [p[1] for p in paired]
    n = len(x)

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = (sum((xi - mean_x) ** 2 for xi in x)) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y)) ** 0.5

    if std_x == 0 or std_y == 0:
        return None

    return cov / (std_x * std_y)


def _safe_num(val, subkey: str = None):
    """Extract a numeric value safely."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict) and subkey:
        v = val.get(subkey)
        if isinstance(v, (int, float)):
            return float(v)
    return None


_DIM_LABELS = {
    "sleep_quality": "睡眠质量",
    "sleep_duration": "睡眠时长",
    "energy": "精力",
    "mood": "情绪",
}


def _dim_label(key: str) -> str:
    return _DIM_LABELS.get(key, key)


def _describe_correlation(d1: str, d2: str, corr: float) -> str:
    """Generate a human-readable description of the correlation."""
    dir_str = "正向" if corr > 0 else "负向"
    strength = "强" if abs(corr) >= 0.7 else "中等" if abs(corr) >= 0.5 else "弱"

    if corr > 0:
        return f"{d1}越好，{d2}也越好（{strength}{dir_str}关联，r={corr:.2f}）"
    else:
        return f"{d1}越差，{d2}也越差（{strength}{dir_str}关联，r={corr:.2f}）"


def get_trend_insights(messages: list[dict]) -> list[dict]:
    """
    Generate AI-friendly trend descriptions from message history.

    Returns list of insight dicts that can be passed to the LLM story prompt.
    """
    insights = []
    correlations = compute_correlations(messages)
    msgs = _message_dicts(messages)

    for corr in correlations:
        insights.append({
            "type": "correlation",
            "text": corr["description"],
            "confidence": "medium" if corr["n"] < 5 else "high",
        })

    # Recent trend analysis
    if len(msgs) >= 2:
        recent = msgs[-3:]
        earlier = msgs[:-3] if len(msgs) > 3 else []

        recent_mood = [_safe_num(_extracted_data(m).get("mood")) for m in recent]
        recent_mood = [m for m in recent_mood if m is not None]

        if recent_mood and earlier:
            earlier_mood = [_safe_num(_extracted_data(m).get("mood")) for m in earlier]
            earlier_mood = [m for m in earlier_mood if m is not None]

            if recent_mood and earlier_mood:
                r_avg = sum(recent_mood) / len(recent_mood)
                e_avg = sum(earlier_mood) / len(earlier_mood)
                diff = r_avg - e_avg
                if abs(diff) >= 0.5:
                    direction = "上升" if diff > 0 else "下降"
                    insights.append({
                        "type": "trend",
                        "text": f"近期情绪趋势{direction}（{e_avg:.1f} → {r_avg:.1f}）",
                        "confidence": "low",
                    })

    return insights
