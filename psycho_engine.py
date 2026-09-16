"""
psycho_engine.py — Psychological profiling engine for Viora.

Phase A: PsychoEvent data model + LLM extraction from chat messages.  ✅
Phase B: PsychoState aggregation + intervenable state classification. ✅
Phase C: PsychoTrait baseline + PsychoFingerprint + ResponsePlanner integration.

Usage:
    from psycho_engine import extract_psycho_data, PsychoEvent, PsychoState
    from psycho_engine import classify_intervenable_state, recommend_intervention

    # Extract psychological dimensions from a message
    psycho_data = extract_psycho_data("今天一点都不想动，好烦")

    # Store as a PsychoEvent
    event = PsychoEvent.from_extracted(psycho_data, user_id="u1", msg_id="m1")
    store = PsychoEventStore()
    store.append(user_id, event)

    # Aggregate into PsychoState
    state = PsychoState.from_events(existing_state, new_event)
    intervenable = classify_intervenable_state(state.compute_ema())
    intervention = recommend_intervention(intervenable)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from uuid import uuid4

from config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL, LLM_TIMEOUT

logger = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))

# ── Constants ──────────────────────────────────────────────────────────────────

PSYCHO_DIMENSIONS = [
    "willingness",      # 运动意愿 0-10
    "fatigue",          # 疲劳感 0-10
    "stress",           # 压力 0-10
    "procrastination",  # 启动阻力 0-10
    "achievement",      # 成就感 0-10
    "frustration",      # 挫败感 0-10
    "guilt",            # 自责 0-10
    "next_confidence",  # 下次信心 0-10
    "avoidance",        # 回避倾向 0-10
]

PSYCHO_PHASES = [
    "idle",               # 日常闲聊
    "pre_exercise",       # 运动前讨论
    "during_exercise",    # 运动中
    "post_exercise",      # 运动刚结束
    "next_day",           # 运动后第二天
]


# ── Extraction Prompt ──────────────────────────────────────────────────────────

PSYCHO_EXTRACT_PROMPT = """从用户的对话中提取心理状态指标。

【提取规则】
1. 只在用户明确表达时提取对应维度，不确定的字段设为 null
2. 所有评分 0-10（0=完全没有，10=极端强烈）
3. 关注与运动/健康/日常行为相关的心理信号，不泛化提取

【评分示例】
- "今天一点都不想动" → willingness=2, procrastination=8
- "跑完感觉还不错" → achievement=7, next_confidence=7
- "太累了"（运动上下文）→ fatigue=8
- "又没去，我真没用" → guilt=9, avoidance=7
- "最近压力好大" → stress=8
- "昨天做完运动今天浑身酸痛" → fatigue=7
- "坚持了三天，感觉还行" → achievement=6, next_confidence=6
- 用户闲聊、没有心理信号 → 所有字段 null

【输出 JSON】
{
  "willingness": <int 0-10 或 null>,
  "fatigue": <int 0-10 或 null>,
  "stress": <int 0-10 或 null>,
  "procrastination": <int 0-10 或 null>,
  "achievement": <int 0-10 或 null>,
  "frustration": <int 0-10 或 null>,
  "guilt": <int 0-10 或 null>,
  "next_confidence": <int 0-10 或 null>,
  "avoidance": <int 0-10 或 null>,
  "phase": "idle" | "pre_exercise" | "during_exercise" | "post_exercise" | "next_day" | null,
  "reason": <str, 简要说明提取依据, 或 null>
}
"""


# ── Data Models ────────────────────────────────────────────────────────────────

@dataclass
class PsychoEvent:
    """单次心理事件，类似 EMA 的一次采样

    从对话中自然提取的心理状态快照。每个聊天消息最多产生一个 PsychoEvent，
    包含所有能被 LLM 可靠提取的维度。
    """
    id: str = field(default_factory=lambda: f"psy_{uuid4().hex[:12]}")
    user_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(CST).isoformat())
    phase: str = "idle"

    # 心理状态快照（0-10, null=未表达）
    willingness: float | None = None
    fatigue: float | None = None
    stress: float | None = None
    procrastination: float | None = None
    achievement: float | None = None
    frustration: float | None = None
    guilt: float | None = None
    next_confidence: float | None = None
    avoidance: float | None = None

    source_type: str = "chat_inferred"
    source_message_id: str = ""
    confidence: float = 0.7    # LLM extraction confidence baseline
    reason: str = ""

    @classmethod
    def from_extracted(cls, extracted: dict, user_id: str,
                       message_id: str = "", phase: str | None = None) -> PsychoEvent:
        """Create a PsychoEvent from LLM-extracted dict."""
        # Count non-null dimensions as a rough confidence signal
        non_null = sum(1 for dim in PSYCHO_DIMENSIONS if extracted.get(dim) is not None)
        confidence = min(1.0, 0.5 + non_null * 0.06)

        return cls(
            user_id=user_id,
            timestamp=datetime.now(CST).isoformat(),
            phase=phase or extracted.get("phase") or "idle",
            willingness=extracted.get("willingness"),
            fatigue=extracted.get("fatigue"),
            stress=extracted.get("stress"),
            procrastination=extracted.get("procrastination"),
            achievement=extracted.get("achievement"),
            frustration=extracted.get("frustration"),
            guilt=extracted.get("guilt"),
            next_confidence=extracted.get("next_confidence"),
            avoidance=extracted.get("avoidance"),
            source_type="chat_inferred",
            source_message_id=message_id,
            confidence=confidence,
            reason=extracted.get("reason", ""),
        )

    def to_dict(self) -> dict:
        result = asdict(self)
        return result

    @property
    def has_any_signal(self) -> bool:
        """Check if this event contains any non-null psychological signal."""
        return any(
            getattr(self, dim) is not None
            for dim in PSYCHO_DIMENSIONS
        )


# ── LLM Extraction ─────────────────────────────────────────────────────────────

def extract_psycho_data(message: str) -> dict:
    """
    Extract psychological state indicators from a user message via LLM.

    Returns a dict of {dimension: int|None} following PSYCHO_EXTRACT_PROMPT schema.
    """
    import requests
    from ai_engine import _parse_json_response, _get_time_context

    time_ctx = _get_time_context()
    system_msg = "你是一个心理状态提取助手。从对话中提取用户的心理信号，只返回 JSON。"
    user_msg = f"当前时间上下文：{time_ctx}\n\n用户消息：{message}\n\n{PSYCHO_EXTRACT_PROMPT}"

    url = f"{LLM_API_BASE.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens": 512,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"].get("content", "").strip()
        if not content:
            logger.warning("Empty psycho extraction response")
            return _empty_psycho_extracted()

        raw = _parse_json_response(content)
        if isinstance(raw, dict):
            return _normalize_psycho_extracted(raw)
        logger.warning("Psycho extraction returned non-dict: %s", type(raw))
        return _empty_psycho_extracted()
    except Exception as e:
        logger.warning("Psycho extraction failed (non-critical): %s", e)
        return _empty_psycho_extracted()


def _empty_psycho_extracted() -> dict:
    """Return an empty psycho extraction result."""
    return {dim: None for dim in PSYCHO_DIMENSIONS} | {"phase": None, "reason": None}


def _normalize_psycho_extracted(raw: dict) -> dict:
    """Normalize and validate psycho extraction output."""
    result = _empty_psycho_extracted()
    for dim in PSYCHO_DIMENSIONS:
        val = raw.get(dim)
        if val is not None:
            try:
                val = float(val)
                if 0 <= val <= 10:
                    result[dim] = val
            except (ValueError, TypeError):
                pass
    phase = raw.get("phase")
    if phase in PSYCHO_PHASES:
        result["phase"] = phase
    # reason is documented as str|None. A malformed LLM response could put a
    # dict/list/number here — sanitize to None instead of leaking non-string
    # junk out of the extraction schema (downstream PsychoEvent.reason is str).
    reason = raw.get("reason", "")
    result["reason"] = reason if isinstance(reason, str) else None
    return result


# ── In-Memory Aggregation Helpers (Phase B prep) ──────────────────────────────

def compute_ema(existing: float | None, new_value: float, alpha: float) -> float:
    """Exponential moving average update."""
    if existing is None:
        return new_value
    return alpha * existing + (1 - alpha) * new_value


EMA_ALPHAS = {
    "willingness": 0.5,
    "fatigue": 0.6,
    "stress": 0.6,
    "procrastination": 0.5,
    "achievement": 0.7,
    "frustration": 0.5,
    "guilt": 0.5,
    "next_confidence": 0.6,
    "avoidance": 0.75,
}


# ══════════════════════════════════════════════════════════════════════════════
# Phase B — PsychoState Aggregation + Intervenable State Classification
# ══════════════════════════════════════════════════════════════════════════════

# ── Intervenable states ───────────────────────────────────────────────────────

INTERVENABLE_STATES = [
    "willing_but_stuck",          # 高意愿+高启动阻力
    "low_mood_available",         # 意愿正常+低情绪
    "low_self_efficacy",          # 低自我效能
    "post_exercise_frustrated",   # 运动后挫败
    "avoidance_after_break",      # 断签后回避+自责
    "high_achievement_fatigued",  # 高成就+高疲劳
    "social_motivated",           # 社交驱动型
    "social_pressured",           # 社交压力敏感
    "normal",                     # 无显著偏离
]

INTERVENTION_MAP = {
    "willing_but_stuck": {
        "intervention_type": "lower_barrier",
        "intensity": "medium",
        "timing": "now",
        "suggested_tone": "casual",
        "description": "降低启动门槛，给微运动入口",
        "example": "不用跑步，就下楼走5分钟～",
    },
    "low_mood_available": {
        "intervention_type": "gentle_suggest",
        "intensity": "light",
        "timing": "now",
        "suggested_tone": "warm",
        "description": "推荐舒缓型运动，不提强度",
        "example": "今天天气不错，出去透透气就好",
    },
    "low_self_efficacy": {
        "intervention_type": "micro_goal",
        "intensity": "light",
        "timing": "now",
        "suggested_tone": "encouraging",
        "description": "极低目标 + 完成反馈",
        "example": "今天就做一件事：穿上运动鞋。做到了就算赢😏",
    },
    "post_exercise_frustrated": {
        "intervention_type": "normalize",
        "intensity": "light",
        "timing": "soon",
        "suggested_tone": "understanding",
        "description": "降低下次目标，避免羞辱反馈",
        "example": "刚开始都这样，慢慢来不用急",
    },
    "avoidance_after_break": {
        "intervention_type": "reset_soft",
        "intensity": "medium",
        "timing": "soon",
        "suggested_tone": "casual",
        "description": "不强调断签，给重新开始的信号",
        "example": "好久不见～想从哪开始都行",
    },
    "high_achievement_fatigued": {
        "intervention_type": "rest_reminder",
        "intensity": "light",
        "timing": "now",
        "suggested_tone": "caring",
        "description": "建议休息日，避免过度训练",
        "example": "最近挺拼的，今天歇一天吧",
    },
    "social_motivated": {
        "intervention_type": "social_boost",
        "intensity": "medium",
        "timing": "soon",
        "suggested_tone": "excited",
        "description": "给分享/组队/挑战入口",
        "example": "要不要把你的记录发出去？",
    },
    "social_pressured": {
        "intervention_type": "private_default",
        "intensity": "low",
        "timing": "never",
        "suggested_tone": "neutral",
        "description": "不提排行榜，不用比较语气",
        "example": None,
    },
    "normal": {
        "intervention_type": "no_intervention",
        "intensity": "none",
        "timing": "never",
        "suggested_tone": "normal",
        "description": "常规聊天节奏，不额外干预",
        "example": None,
    },
}


# ── PsychoState Data Model ─────────────────────────────────────────────────────

@dataclass
class PsychoState:
    """聚合后的短期心理状态（3-7 天滑动窗口）

    通过 EMA 指数衰减聚合最近 PsychoEvent 序列，生成当前心理基线。
    """
    id: str = field(default_factory=lambda: f"psy_state_{uuid4().hex[:12]}")
    user_id: str = ""

    # EMA 聚合值（0-10, null=数据不足）
    ema_willingness: float | None = None
    ema_fatigue: float | None = None
    ema_stress: float | None = None
    ema_procrastination: float | None = None
    ema_achievement: float | None = None
    ema_frustration: float | None = None
    ema_guilt: float | None = None
    ema_next_confidence: float | None = None
    ema_avoidance: float | None = None

    # 趋势方向
    willingness_trend: str = "stable"   # "improving" | "declining" | "stable"
    fatigue_trend: str = "stable"
    avoidance_trend: str = "stable"

    # 分类结果
    intervenable_state: str = "normal"
    last_updated: str = field(default_factory=lambda: datetime.now(CST).isoformat())
    n_events_aggregated: int = 0

    def ema_dict(self) -> dict:
        """Return EMA dimensions as a flat dict for classification."""
        return {
            "willingness": self.ema_willingness,
            "fatigue": self.ema_fatigue,
            "stress": self.ema_stress,
            "procrastination": self.ema_procrastination,
            "achievement": self.ema_achievement,
            "frustration": self.ema_frustration,
            "guilt": self.ema_guilt,
            "next_confidence": self.ema_next_confidence,
            "avoidance": self.ema_avoidance,
        }

    def update_from_event(self, event: PsychoEvent) -> PsychoState:
        """Update this state with a new PsychoEvent using EMA."""
        ema = self.ema_dict()

        for dim in PSYCHO_DIMENSIONS:
            new_val = getattr(event, dim)
            if new_val is not None:
                existing = ema.get(dim)
                alpha = EMA_ALPHAS.get(dim, 0.5)
                updated = compute_ema(existing, new_val, alpha)
                setattr(self, f"ema_{dim}", updated)

        self.n_events_aggregated += 1
        self.last_updated = datetime.now(CST).isoformat()

        # Reclassify after update
        self.intervenable_state = classify_intervenable_state(self.ema_dict())
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PsychoState:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def fresh(cls, user_id: str) -> PsychoState:
        """Create a fresh empty PsychoState for a new user."""
        return cls(user_id=user_id)


# ── State Classifier ──────────────────────────────────────────────────────────

def normalize_dim(val: float | None, default: float = 0.5) -> float:
    """Normalize a 0-10 dimension to 0-1 for rule matching."""
    if val is None:
        return default
    return val / 10.0


def classify_intervenable_state(ema: dict) -> str:
    """Classify the user's current intervenable state from EMA-aggregated dimensions.

    Rule priority order (first match wins):
      1. avoidance_after_break: avoidance > 0.7 AND guilt > 0.6
      2. post_exercise_frustrated: frustration > 0.6
      3. high_achievement_fatigued: achievement > 0.7 AND fatigue > 0.6
      4. willing_but_stuck: willingness > 0.6 AND procrastination > 0.5
      5. low_mood_available: willingness >= 0.4 AND stress > 0.7
      6. low_self_efficacy: next_confidence < 0.3
      7. social_motivated: achievement > 0.7 AND next_confidence > 0.7
      8. social_pressured: avoidance > 0.5 AND stress > 0.6
      9. normal: no significant deviation

    Returns one of INTERVENABLE_STATES.
    """
    w = normalize_dim(ema.get("willingness"))
    f = normalize_dim(ema.get("fatigue"))
    s = normalize_dim(ema.get("stress"))
    p = normalize_dim(ema.get("procrastination"))
    a = normalize_dim(ema.get("achievement"))
    fr = normalize_dim(ema.get("frustration"))
    g = normalize_dim(ema.get("guilt"))
    nc = normalize_dim(ema.get("next_confidence"))
    av = normalize_dim(ema.get("avoidance"))

    # 1. Avoidance after break — highest urgency
    if av > 0.7 and g > 0.6:
        return "avoidance_after_break"

    # 2. Post-exercise frustration
    if fr > 0.6:
        return "post_exercise_frustrated"

    # 3. High achievement + high fatigue (over-training risk)
    if a > 0.7 and f > 0.6:
        return "high_achievement_fatigued"

    # 4. Willing but stuck
    if w > 0.6 and p > 0.5:
        return "willing_but_stuck"

    # 5. Low mood but available
    if w >= 0.4 and s > 0.7:
        return "low_mood_available"

    # 6. Low self-efficacy
    if nc < 0.3:
        return "low_self_efficacy"

    # 7. Social motivated
    if a > 0.7 and nc > 0.7:
        return "social_motivated"

    # 8. Social pressured
    if av > 0.5 and s > 0.6:
        return "social_pressured"

    return "normal"


# ── Intervention Recommender ──────────────────────────────────────────────────

def recommend_intervention(intervenable_state: str,
                            persona_hint: str = "warm_friend") -> dict:
    """Recommend an intervention strategy for the given intervenable state.

    Returns a dict with intervention_type, intensity, timing, suggested_tone,
    description, and example message.
    """
    base = INTERVENTION_MAP.get(intervenable_state, INTERVENTION_MAP["normal"])
    result = dict(base)
    result["intervenable_state"] = intervenable_state
    result["confidence"] = 0.5 if intervenable_state == "normal" else 0.7
    result["persona"] = persona_hint
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Phase C — PsychoTrait Baseline + PsychoFingerprint
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PsychoTrait:
    """长期稳定的心理画像参数，从累积心理事件中推断。

    不同于 PsychoState（短期，EMA 聚合），PsychoTrait 是长期稳定的基线，
    反映用户的固有心理特征（自主性偏好、奖励敏感性、压力敏感性等）。
    初始各参数默认为 0.5（中性），随事件积累逐步校准。
    """
    id: str = field(default_factory=lambda: f"psy_trait_{uuid4().hex[:12]}")
    user_id: str = ""

    # 心理基线维度（0-1）
    autonomy_preference: float = 0.5         # 自主性偏好
    reward_sensitivity: float = 0.5          # 奖励敏感性
    pressure_sensitivity: float = 0.5        # 压力敏感性
    initial_self_efficacy: float = 0.5       # 初始自我效能
    exercise_identity: float = 0.5           # 运动身份认同
    social_exposure_pref: float = 0.5        # 社交暴露偏好
    discomfort_baseline: float = 0.5         # 身体不适基线

    confidence: float = 0.1                  # 整体置信度（0-1）
    sample_count: int = 0
    updated_at: str = field(default_factory=lambda: datetime.now(CST).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


def compute_psycho_fingerprint(user_id: str,
                                events: list[dict]) -> dict:
    """Compute a user's psycho fingerprint from accumulated events.

    Returns a dict with:
      - traits: PsychoTrait baseline (simple heuristic-based inference)
      - per_dim_stats: {dim: {mean, std, count, min, max, trend}}
      - n_events: total event count
      - has_signal_events: events with at least one non-null dimension
    """
    if not events:
        return {
            "traits": PsychoTrait(user_id=user_id).to_dict(),
            "per_dim_stats": {},
            "n_events": 0,
            "has_signal_events": 0,
        }

    # Filter to events that have at least one signal
    signal_events = [e for e in events if any(
        e.get(d) is not None for d in PSYCHO_DIMENSIONS
    )]

    # Per-dimension statistics
    per_dim_stats = {}
    for dim in PSYCHO_DIMENSIONS:
        values = [e[dim] for e in signal_events if e.get(dim) is not None]
        if not values:
            per_dim_stats[dim] = {"count": 0, "mean": None}
            continue

        import statistics
        n = len(values)
        mean_val = round(statistics.mean(values), 4)
        std_val = round(statistics.stdev(values), 4) if n > 1 else 0.0

        # Simple trend: compare last 1/3 vs first 1/3
        third = max(1, n // 3)
        early = statistics.mean(values[:third])
        late = statistics.mean(values[-third:])
        if late > early + 0.3:
            trend = "improving"
        elif late < early - 0.3:
            trend = "declining"
        else:
            trend = "stable"

        per_dim_stats[dim] = {
            "count": n,
            "mean": round(mean_val / 10.0, 4) if mean_val else None,  # normalize to 0-1
            "std": round(std_val / 10.0, 4),
            "min": round(min(values) / 10.0, 4),
            "max": round(max(values) / 10.0, 4),
            "trend": trend,
        }

    # Heuristic PsychoTrait inference from aggregated stats
    w = per_dim_stats.get("willingness", {}).get("mean") or 0.5
    nc = per_dim_stats.get("next_confidence", {}).get("mean") or 0.5
    av = per_dim_stats.get("avoidance", {}).get("mean") or 0.5
    fr = per_dim_stats.get("frustration", {}).get("mean") or 0.5
    st = per_dim_stats.get("stress", {}).get("mean") or 0.5

    trait = PsychoTrait(
        user_id=user_id,
        autonomy_preference=round(max(0, min(1, 0.3 + w)), 2),
        reward_sensitivity=round(max(0, min(1, 0.3 + nc * 0.7)), 2),
        pressure_sensitivity=round(max(0, min(1, 0.2 + st * 0.8)), 2),
        initial_self_efficacy=round(max(0, min(1, 0.3 + nc * 0.7)), 2),
        exercise_identity=round(max(0, min(1, 0.2 + w * 0.8)), 2),
        social_exposure_pref=round(max(0, min(1, 0.5 - av * 0.5)), 2),
        discomfort_baseline=round(max(0, min(1, 0.3 + fr * 0.5)), 2),
        confidence=round(min(0.9, 0.1 + len(signal_events) * 0.02), 2),
        sample_count=len(signal_events),
    )

    return {
        "traits": trait.to_dict(),
        "per_dim_stats": per_dim_stats,
        "n_events": len(events),
        "has_signal_events": len(signal_events),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Phase C — PsychoFingerprint Full Model (μ, Σ, A, B, R)
# ══════════════════════════════════════════════════════════════════════════════


def _compute_mu(signal_events: list[dict]) -> dict:
    """Compute mean vector (μ) — per-dimension mean across all signal events."""
    mu = {}
    for dim in PSYCHO_DIMENSIONS:
        values = [e[dim] for e in signal_events if e.get(dim) is not None]
        if values:
            mu[dim] = round(sum(values) / len(values), 4)
        else:
            mu[dim] = None
    return mu


def _compute_sigma(signal_events: list[dict]) -> dict:
    """Compute covariance matrix (Σ) — pairwise covariances between dimensions.

    Returns {dim1: {dim2: cov, ...}, ...} where cov is the sample covariance.
    Only includes events where both dimensions are non-null.
    """
    sigma = {}
    for d1 in PSYCHO_DIMENSIONS:
        sigma[d1] = {}
        for d2 in PSYCHO_DIMENSIONS:
            # Get events where both dimensions are present
            pairs = [(e[d1], e[d2]) for e in signal_events
                     if e.get(d1) is not None and e.get(d2) is not None]
            if len(pairs) < 2:
                sigma[d1][d2] = None
                continue

            n = len(pairs)
            mean1 = sum(p[0] for p in pairs) / n
            mean2 = sum(p[1] for p in pairs) / n
            cov = sum((p[0] - mean1) * (p[1] - mean2) for p in pairs) / (n - 1)
            sigma[d1][d2] = round(cov / 100.0, 4)  # normalize to [-1, 1] scale
    return sigma


def _compute_ar1_coefficients(signal_events: list[dict]) -> dict:
    """Compute AR(1) coefficients (A) — lag-1 autocorrelation per dimension.

    For each dimension, fits: x_t = A * x_{t-1} + ε
    Returns {dim: {coefficient: float, r_squared: float, n: int}}
    Coefficient near 0 = no temporal dependency (highly variable).
    Coefficient near 1 = strong autocorrelation (stable trait).
    """
    a_coeffs = {}
    for dim in PSYCHO_DIMENSIONS:
        values = [e[dim] for e in signal_events if e.get(dim) is not None]
        if len(values) < 3:
            a_coeffs[dim] = {"coefficient": None, "raw_coefficient": None, "r_squared": None, "n": len(values)}
            continue

        x_prev = values[:-1]
        x_curr = values[1:]
        n = len(x_prev)

        # Simple linear regression: x_curr = A * x_prev
        mean_prev = sum(x_prev) / n
        mean_curr = sum(x_curr) / n
        num = sum((x_prev[i] - mean_prev) * (x_curr[i] - mean_curr) for i in range(n))
        den = sum((x_prev[i] - mean_prev) ** 2 for i in range(n))

        if den == 0:
            a_coeffs[dim] = {"coefficient": 0.0, "raw_coefficient": 0.0, "r_squared": 0.0, "n": n}
            continue

        coeff = num / den
        # R-squared: variance explained by the model
        predicted = [mean_prev + coeff * (x_prev[i] - mean_prev) for i in range(n)]
        ss_res = sum((x_curr[i] - predicted[i]) ** 2 for i in range(n))
        ss_tot = sum((x_curr[i] - mean_curr) ** 2 for i in range(n))
        r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        a_coeffs[dim] = {
            "coefficient": round(coeff / 10.0, 4),  # normalized for 0-1 scale
            "raw_coefficient": round(coeff, 4),      # original 0-10 scale
            "r_squared": round(max(0, min(1, r_sq)), 4),
            "n": n,
        }
    return a_coeffs


def _compute_behavioral_embedding(signal_events: list[dict]) -> dict:
    """Compute behavioral embedding (B) — latent behavioral patterns.

    Simplified approach: uses the pairwise correlation structure to identify
    latent factors. Returns:
      - latent_factors: list of {factor_N: [loading_1,...,loading_9]} — the top
        2 eigenvectors of the correlation matrix (simplified SVD).
      - explained_variance: fraction of total variance explained by each factor.
    """
    if len(signal_events) < 2:
        return {"latent_factors": [], "explained_variance": [], "n_events": 0}

    n_dims = len(PSYCHO_DIMENSIONS)
    # Build correlation matrix
    corr = {}
    for i, d1 in enumerate(PSYCHO_DIMENSIONS):
        row = {}
        for j, d2 in enumerate(PSYCHO_DIMENSIONS):
            pairs = [(e[d1], e[d2]) for e in signal_events
                     if e.get(d1) is not None and e.get(d2) is not None]
            if len(pairs) < 2:
                row[d2] = 0.0
                continue
            n = len(pairs)
            m1 = sum(p[0] for p in pairs) / n
            m2 = sum(p[1] for p in pairs) / n
            num = sum((p[0] - m1) * (p[1] - m2) for p in pairs)
            s1 = (sum((p[0] - m1) ** 2 for p in pairs) / (n - 1)) ** 0.5
            s2 = (sum((p[1] - m2) ** 2 for p in pairs) / (n - 1)) ** 0.5
            row[d2] = round(num / ((n - 1) * s1 * s2), 4) if s1 > 0 and s2 > 0 else 0.0
        corr[d1] = row

    # Simplified eigen-decomposition via power iteration for top 2 factors
    # Build the correlation matrix as a flat list of lists
    dims = PSYCHO_DIMENSIONS
    mat = [[corr[d1][d2] for d2 in dims] for d1 in dims]

    def power_iteration(matrix, n_iter=20):
        n = len(matrix)
        v = [1.0 / n ** 0.5] * n  # initial guess: uniform
        for _ in range(n_iter):
            # Multiply: v = M @ v
            v_new = [sum(matrix[i][j] * v[j] for j in range(n)) for i in range(n)]
            norm = sum(x ** 2 for x in v_new) ** 0.5
            if norm == 0:
                break
            v = [x / norm for x in v_new]
        # Rayleigh quotient: eigenvalue ≈ v^T M v
        ray = sum(v[i] * sum(mat[i][j] * v[j] for j in range(n)) for i in range(n))
        return v, max(0, ray / n)  # normalized eigenvalue

    vec1, ev1 = power_iteration(mat)
    # Deflate: M' = M - λ₁ v₁ v₁ᵀ
    mat2 = [[mat[i][j] - ev1 * n_dims * vec1[i] * vec1[j] for j in range(n_dims)]
            for i in range(n_dims)]
    vec2, ev2 = power_iteration(mat2)

    # Only keep factors with meaningful eigenvalues
    factors = []
    ev_ratios = []
    total_ev = ev1 + ev2
    if ev1 > 0.1:
        factors.append({f"factor_1": vec1})
        ev_ratios.append(round(ev1 / total_ev, 4) if total_ev > 0 else 0.5)
    if ev2 > 0.1:
        factors.append({f"factor_2": vec2})
        ev_ratios.append(round(ev2 / total_ev, 4) if total_ev > 0 else 0.3)

    return {
        "latent_factors": factors,
        "explained_variance": ev_ratios,
        "n_events": len(signal_events),
    }


def _compute_residuals(signal_events: list[dict],
                       ar_coefficients: dict) -> dict:
    """Compute residuals (R) — prediction errors for AR(1) model per dimension.

    Returns {dim: {mean_residual, std_residual, last_residual, n}}
    Mean residual near 0 = AR model fits well.
    Large std_residual = dimension is noisy / unpredictable.
    """
    residuals = {}
    for dim in PSYCHO_DIMENSIONS:
        values = [e[dim] for e in signal_events if e.get(dim) is not None]
        if len(values) < 2:
            residuals[dim] = {"mean_residual": None, "std_residual": None,
                              "last_residual": None, "n": 0}
            continue

        coeff_info = ar_coefficients.get(dim, {})
        raw_coeff = coeff_info.get("raw_coefficient") or 0.0

        # Mean-centered AR(1): x_t - μ = A * (x_{t-1} - μ) + ε
        x_prev = values[:-1]
        x_curr = values[1:]
        n = len(x_prev)
        mu = sum(values) / len(values)

        predicted = [mu + raw_coeff * (x_prev[i] - mu) for i in range(n)]
        resids = [x_curr[i] - predicted[i] for i in range(n)]
        mean_r = sum(resids) / n
        std_r = (sum((r - mean_r) ** 2 for r in resids) / max(1, n - 1)) ** 0.5

        residuals[dim] = {
            "mean_residual": round(mean_r / 10.0, 4),
            "std_residual": round(std_r / 10.0, 4),
            "last_residual": round(resids[-1] / 10.0, 4) if resids else None,
            "n": n,
        }
    return residuals


def compute_full_fingerprint(user_id: str,
                              events: list[dict]) -> dict:
    """Compute the complete PsychoFingerprint model (μ, Σ, A, B, R).

    Returns:
      - user_id: str
      - mu: {dim: mean} — mean vector
      - sigma: {dim: {dim: cov}} — covariance matrix
      - ar_coefficients: {dim: {coefficient, r_squared, n}} — AR(1) model
      - behavioral_embedding: {latent_factors, explained_variance, n_events}
      - residuals: {dim: {mean_residual, std_residual, last_residual, n}}
      - n_events: total event count
      - has_signal_events: events with at least one signal dimension
      - confidence: 0-1 (based on sample size)
      - updated_at: ISO timestamp
    """
    if not events:
        return {
            "user_id": user_id,
            "mu": {},
            "sigma": {},
            "ar_coefficients": {},
            "behavioral_embedding": {"latent_factors": [], "explained_variance": [],
                                     "n_events": 0},
            "residuals": {},
            "n_events": 0,
            "has_signal_events": 0,
            "confidence": 0.1,
            "updated_at": datetime.now(CST).isoformat(),
        }

    signal_events = [e for e in events if any(
        e.get(d) is not None for d in PSYCHO_DIMENSIONS
    )]

    mu = _compute_mu(signal_events)
    sigma = _compute_sigma(signal_events)
    ar_coeffs = _compute_ar1_coefficients(signal_events)
    embedding = _compute_behavioral_embedding(signal_events)
    residuals = _compute_residuals(signal_events, ar_coeffs)

    # Confidence scales with sample size: 0.1 + 0.02 * n, capped at 0.95
    confidence = round(min(0.95, 0.1 + len(signal_events) * 0.02), 2)

    return {
        "user_id": user_id,
        "mu": mu,
        "sigma": sigma,
        "ar_coefficients": ar_coeffs,
        "behavioral_embedding": embedding,
        "residuals": residuals,
        "n_events": len(events),
        "has_signal_events": len(signal_events),
        "confidence": confidence,
        "updated_at": datetime.now(CST).isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Phase C — Passive Inference Signal Ingestion
# ══════════════════════════════════════════════════════════════════════════════


def infer_passive_psycho_signals(
    scheduler_state: dict | None = None,
    routine_anomalies: list[dict] | None = None,
    plan_context: dict | None = None,
) -> dict:
    """Infer psychological signals from passive behavioral data sources.

    Produces a dict with PSYCHO_DIMENSIONS keys (plus phase/reason) that can
    be fed directly into the PsychoEvent pipeline, just like LLM extraction.

    Data sources:
      - scheduler_state: {unresponded_count, last_proactive_message, silent_until}
        → guilt / avoidance / willingness
      - routine_anomalies: [{dimension, severity, description}, ...]
        → fatigue / stress / frustration / achievement
      - plan_context: {modifications, completions, skips}
        → frustration / achievement / guilt / procrastination

    Returns: {dim: float | None, ..., "phase": str, "reason": str}
      Compatible with _normalize_psycho_extracted() output format.
    """
    result = _empty_psycho_extracted()
    reasons = []

    # ── Scheduler signals ──────────────────────────────────────────────────
    if scheduler_state:
        unresponded = scheduler_state.get("unresponded_count", 0)
        if unresponded >= 2:
            # User hasn't responded to proactives → guilt + avoidance
            guilt = min(8.0, 3.0 + unresponded * 1.5)
            result["guilt"] = guilt
            result["avoidance"] = min(7.0, 2.0 + unresponded * 1.0)
            result["willingness"] = max(2.0, 5.0 - unresponded * 0.8)
            reasons.append(f"unresponded_proactives={unresponded}")

        if scheduler_state.get("silent_until"):
            # User explicitly went silent → avoidance
            result["avoidance"] = max(result["avoidance"] or 3.0, 6.0)
            reasons.append("silent_mode_enabled")

        last_proactive = scheduler_state.get("last_proactive_message")
        if last_proactive:
            try:
                from datetime import datetime, timezone
                last_ts = datetime.fromisoformat(last_proactive)
                # Deprecated call removed: use timezone-aware comparison
                _ = last_ts.astimezone(CST)
            except (ValueError, TypeError, ImportError):
                pass

    # ── Routine anomaly signals ────────────────────────────────────────────
    if routine_anomalies:
        for anomaly in routine_anomalies:
            dim = (anomaly or {}).get("dimension", "")
            sev = (anomaly or {}).get("severity", "medium")
            score = {"high": 7.0, "medium": 5.0, "low": 3.0}.get(sev, 5.0)

            if "sleep" in dim:
                result["fatigue"] = max(result["fatigue"] or 0.0, score)
                reasons.append(f"sleep_anomaly_{sev}")
            elif "exercise" in dim or "workout" in dim or "运动" in dim:
                result["frustration"] = max(result["frustration"] or 0.0, score * 0.7)
                reasons.append(f"exercise_anomaly_{sev}")
            elif "mood" in dim or "情绪" in dim or "stress" in dim or "压力" in dim:
                result["stress"] = max(result["stress"] or 0.0, score)
                reasons.append(f"mood_anomaly_{sev}")
            elif "energy" in dim or "精力" in dim:
                result["fatigue"] = max(result["fatigue"] or 0.0, score * 0.8)
                reasons.append(f"energy_anomaly_{sev}")
            else:
                # Generic anomaly → stress
                result["stress"] = max(result["stress"] or 0.0, score * 0.5)
                reasons.append(f"anomaly_{sev}")

    # ── Plan behavioral signals ────────────────────────────────────────────
    if plan_context:
        mods = plan_context.get("modifications", 0)
        completions = plan_context.get("completions", 0)
        skips = plan_context.get("skips", 0)

        if mods >= 2:
            # Frequent plan changes → frustration
            fr = min(7.0, 2.0 + mods * 1.0)
            result["frustration"] = max(result["frustration"] or 0.0, fr)
            result["procrastination"] = max(result["procrastination"] or 0.0, min(6.0, mods * 0.8))
            reasons.append(f"plan_modifications={mods}")

        if completions >= 1:
            # Plan completions → achievement + next_confidence
            ac = min(8.0, 3.0 + completions * 0.5)
            result["achievement"] = max(result["achievement"] or 0.0, ac)
            nc = min(7.0, 4.0 + completions * 0.3)
            result["next_confidence"] = max(result["next_confidence"] or 0.0, nc)
            reasons.append(f"plan_completions={completions}")

        if skips >= 2:
            # Repeated skips → guilt + procrastination
            g = min(6.0, 2.0 + skips * 0.8)
            result["guilt"] = max(result["guilt"] or 0.0, g)
            result["procrastination"] = max(result["procrastination"] or 0.0, min(7.0, 3.0 + skips * 0.5))
            reasons.append(f"plan_skips={skips}")

    has_signal = any(result.get(d) is not None for d in PSYCHO_DIMENSIONS)
    if has_signal:
        result["phase"] = "passive"
        result["reason"] = "; ".join(reasons) if reasons else "passive_inference"
    else:
        result["phase"] = None
        result["reason"] = ""

    return result
