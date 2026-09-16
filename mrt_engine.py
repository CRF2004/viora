"""
mrt_engine.py — Micro-Randomized Trial (MRT) Validation Framework for Viora Phase D.

MRT provides causal evidence for intervention effectiveness at the individual level.
At each decision point, the system randomly selects an intervention strategy from
the available options. Proximal outcomes (user's next-state change) are measured
to estimate which interventions work best for which users in which contexts.

References:
  - Klasnja et al. (2015). "Micro-randomized trials: An experimental design for
    developing just-in-time adaptive interventions." Health Psychology.
  - Liao et al. (2020). "N-of-1 trials in mobile health."

Usage:
    from mrt_engine import MRTrialManager, randomize_intervention, record_outcome

    m = MRTrialManager.load("user_123")
    chosen = m.randomize("low_mood_available", available=["gentle_suggest",
                                                          "lower_barrier"])
    # ... deliver intervention, observe response ...
    m.record_outcome(chosen, willingness_delta=+2, responded=True)
    effect = m.estimate_effect(chosen)
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional

from config import DATA_DIR

logger = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))

MRT_DATA_FILE = os.path.join(DATA_DIR, "mrt_data.json")


# ── Intervention catalog (maps intervention_type to its hypothesis) ──────────

INTERVENTION_CATALOG = {
    "lower_barrier": {
        "target_dim": "procrastination",
        "hypothesis": "降低启动门槛（微运动）→ 减少启动阻力",
        "expected_effect": "negative",  # reduce procrastination
    },
    "gentle_suggest": {
        "target_dim": "willingness",
        "hypothesis": "轻声建议舒缓运动 → 维持/提升意愿",
        "expected_effect": "positive",
    },
    "positive_framing": {
        "target_dim": "self_efficacy",
        "hypothesis": "积极重构（'刚开始都这样'）→ 提升自我效能",
        "expected_effect": "positive",
    },
    "rest_signal": {
        "target_dim": "fatigue",
        "hypothesis": "建议休息 → 降低疲劳感",
        "expected_effect": "positive",  # rest is good
    },
    "social_boost": {
        "target_dim": "social_motivation",
        "hypothesis": "分享/组队入口 → 激发社交驱动力",
        "expected_effect": "positive",
    },
    "silent_support": {
        "target_dim": "avoidance",
        "hypothesis": "不提断签、轻松回归 → 降低回避",
        "expected_effect": "negative",
    },
}

# ── MRT config ──────────────────────────────────────────────────────────────

# Randomization weights can be uniform or adaptive
DEFAULT_RAND_WEIGHTS = "uniform"  # equal probability per available intervention

# Proximal outcome window (minutes after intervention)
PROXIMAL_WINDOW_MINUTES = 120


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class MRTDecision:
    """A single micro-randomized decision point.

    Records which intervention was chosen, the context at decision time,
    and the proximal outcome observed afterward.
    """
    timestamp: str
    user_id: str
    intervenable_state: str
    chosen_intervention: str
    available_interventions: list[str]
    randomization_weights: dict[str, float] | None = None

    # Proximal outcome (filled after observation window)
    proximal_responded: bool | None = None
    proximal_willingness_delta: float | None = None
    proximal_fatigue_delta: float | None = None
    proximal_avoidance_delta: float | None = None
    proximal_outcome_ts: str | None = None

    # Metadata
    prior_psycho_state: dict | None = None
    hour_of_day: int | None = None
    day_of_week: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MRTDecision":
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})


class MRTrialManager:
    """Per-user MRT manager.

    Maintains the history of micro-randomized decisions and outcomes
    for a single user, and provides effect estimation.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.decisions: list[MRTDecision] = []

    def randomize(self, current_state: str,
                  available: list[str] | None = None,
                  weights: dict[str, float] | None = None,
                  prior_state: dict | None = None) -> str:
        """Randomly select an intervention at a decision point.

        Args:
            current_state: The user's current intervenable_state.
            available: List of intervention types to randomize over.
                       If None, uses all interventions from the catalog.
            weights: Per-intervention randomization weights. If None, uniform.
            prior_state: The user's prior psycho_state dict (for metadata).

        Returns:
            The chosen intervention type string.
        """
        if available is None:
            available = list(INTERVENTION_CATALOG.keys())

        if weights is None:
            weights = {iv: 1.0 / len(available) for iv in available}

        # Ensure weights sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        choices, probs = zip(*weights.items())
        chosen = random.choices(choices, weights=probs, k=1)[0]

        now = datetime.now(CST)
        decision = MRTDecision(
            timestamp=now.isoformat(),
            user_id=self.user_id,
            intervenable_state=current_state,
            chosen_intervention=chosen,
            available_interventions=available,
            randomization_weights=weights,
            prior_psycho_state=prior_state,
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
        )
        self.decisions.append(decision)
        return chosen

    def record_outcome(self, intervention_type: str,
                       responded: bool = False,
                       willingness_delta: float | None = None,
                       fatigue_delta: float | None = None,
                       avoidance_delta: float | None = None) -> None:
        """Record the proximal outcome for the most recent decision of the
        given intervention_type.

        Args:
            intervention_type: The intervention that was delivered.
            responded: Whether the user responded / engaged.
            willingness_delta: Change in willingness after intervention.
            fatigue_delta: Change in fatigue after intervention.
            avoidance_delta: Change in avoidance after intervention.
        """
        # Find the most recent unmatched decision of this type
        for d in reversed(self.decisions):
            if d.chosen_intervention == intervention_type and d.proximal_outcome_ts is None:
                d.proximal_responded = responded
                d.proximal_willingness_delta = willingness_delta
                d.proximal_fatigue_delta = fatigue_delta
                d.proximal_avoidance_delta = avoidance_delta
                d.proximal_outcome_ts = datetime.now(CST).isoformat()
                return
        logger.warning("No unmatched decision found for intervention: %s",
                       intervention_type)

    def estimate_effect(self, intervention_type: str) -> dict:
        """Estimate the proximal effect of a given intervention type.

        Compares outcomes when this intervention was chosen vs other interventions.
        Uses simple mean comparison (in practice, use regression adjustment).

        Returns:
            dict with: n_trials, mean_response_rate, mean_willingness_diff,
                       comparison_rate (other interventions), effect_size
        """
        this_iv = [d for d in self.decisions
                   if d.chosen_intervention == intervention_type
                   and d.proximal_outcome_ts is not None]
        other_iv = [d for d in self.decisions
                    if d.chosen_intervention != intervention_type
                    and d.proximal_outcome_ts is not None]

        if not this_iv:
            return {"n_trials": 0, "error": "no_data"}

        # Response rate
        this_rate = sum(1 for d in this_iv if d.proximal_responded) / len(this_iv)
        other_rate = (sum(1 for d in other_iv if d.proximal_responded) / len(other_iv)
                      if other_iv else None)

        # Willingness delta
        this_w_deltas = [d.proximal_willingness_delta for d in this_iv
                         if d.proximal_willingness_delta is not None]
        other_w_deltas = [d.proximal_willingness_delta for d in other_iv
                          if d.proximal_willingness_delta is not None]

        def mean(xs):
            return sum(xs) / len(xs) if xs else None

        def _safe_round(v, digits=4):
            return round(v, digits) if v is not None else None

        return {
            "n_trials": len(this_iv),
            "n_other": len(other_iv) if other_iv else 0,
            "mean_response_rate": round(this_rate, 4),
            "comparison_response_rate": round(other_rate, 4) if other_rate is not None else None,
            "mean_willingness_delta": _safe_round(mean(this_w_deltas)),
            "mean_fatigue_delta": _safe_round(mean(
                [d.proximal_fatigue_delta for d in this_iv
                 if d.proximal_fatigue_delta is not None])),
            "mean_avoidance_delta": _safe_round(mean(
                [d.proximal_avoidance_delta for d in this_iv
                 if d.proximal_avoidance_delta is not None])),
        }

    def n_of_1_analysis(self, min_trials: int = 3) -> dict:
        """Run N-of-1 analysis to find the optimal intervention for this user.

        Args:
            min_trials: Minimum number of trials per intervention to include.

        Returns:
            dict with: rankings (list of intervention_type sorted by effect),
                       best_intervention, total_decisions, n_with_outcome
        """
        effects = {}
        for iv_name in INTERVENTION_CATALOG:
            effect = self.estimate_effect(iv_name)
            if effect["n_trials"] >= min_trials:
                effects[iv_name] = effect

        # Rank by mean_willingness_delta (primary proximal outcome)
        ranked = sorted(
            effects.items(),
            key=lambda x: (x[1].get("mean_response_rate", 0) or 0),
            reverse=True,
        )

        completed = [d for d in self.decisions if d.proximal_outcome_ts is not None]
        return {
            "rankings": [
                {"intervention": iv, **eff}
                for iv, eff in ranked
            ],
            "best_intervention": ranked[0][0] if ranked else None,
            "total_decisions": len(self.decisions),
            "n_with_outcome": len(completed),
            "user_id": self.user_id,
        }

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "decisions": [d.to_dict() for d in self.decisions],
        }

    def save(self) -> None:
        """Persist to disk."""
        os.makedirs(os.path.dirname(MRT_DATA_FILE), exist_ok=True)
        store = _load_mrt_store()
        store[self.user_id] = self.to_dict()
        temp = MRT_DATA_FILE + ".tmp"
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)
        os.replace(temp, MRT_DATA_FILE)
        logger.info("MRT data saved for user %s (%d decisions)",
                    self.user_id, len(self.decisions))

    @classmethod
    def load(cls, user_id: str) -> "MRTrialManager":
        """Load from disk or create fresh.

        Legacy/corrupt records are tolerated: a non-dict user record yields a
        fresh manager, non-list decisions are ignored, and malformed decision
        entries are skipped instead of crashing the route.
        """
        store = _load_mrt_store()
        record = store.get(user_id)
        if isinstance(record, dict):
            m = cls(user_id)
            decisions = record.get("decisions", [])
            if isinstance(decisions, list):
                for d in decisions:
                    if not isinstance(d, dict):
                        continue
                    try:
                        m.decisions.append(MRTDecision.from_dict(d))
                    except (TypeError, ValueError):
                        logger.warning(
                            "Skipping malformed MRT decision for user %s", user_id
                        )
            return m
        return cls(user_id)


def _load_mrt_store() -> dict:
    """Load the global MRT store JSON.

    A non-dict top level (legacy/corrupt file) degrades to an empty store.
    """
    try:
        if os.path.exists(MRT_DATA_FILE):
            with open(MRT_DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to load MRT data: %s", e)
    return {}
