"""
scheduled_stories.py — Weekly body story pre-generator for Viora.

Runs every Sunday early morning (via story_scheduler_daemon.py).  For each
user, checks whether the period (week/month) has ≥5 messages with health
data.  If yes, generates via LLM and caches.  If not, caches a lightweight
placeholder so the frontend can show a "not enough data" template.

Safe for gunicorn multi-worker: all writes go through storage.py file locks.
"""

import logging
import sys
import os
import time
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import storage
from ai_engine import generate_body_story, LLMError
from insights import compute_correlations, get_trend_insights
from accounts import list_accounts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scheduled_stories")

# Minimum messages with extracted health data required to trigger LLM generation
MIN_MESSAGES = 5

# Period labels for the fallback template
PERIOD_LABELS = {"week": "本周", "month": "本月"}

FALLBACK_TEMPLATE = """{period_label}你还没有足够的健康记录呢～

目前只收集到 {count} 条有效数据，需要至少 {min_count} 条才能生成完整的身体故事。

继续和我说说你的睡眠、运动、饮食、情绪吧，我会在每周日凌晨自动为你生成～

小声说：哪怕只是随口聊几句"今天好累"或"昨晚没睡好"，我都能从中提取有用的信息哦 😊"""


def _count_health_messages(records: list) -> int:
    """Count messages that contain actual extracted health data (not just chat)."""
    count = 0
    for r in records:
        ed = r.get("extracted_data", {})
        if not isinstance(ed, dict):
            continue
        # Count if any dimension has data
        has_data = False
        sleep = ed.get("sleep", {})
        if isinstance(sleep, dict):
            if sleep.get("quality") is not None or sleep.get("duration_hours") is not None:
                has_data = True
        if ed.get("energy") is not None:
            has_data = True
        if ed.get("mood") is not None:
            has_data = True
        exercise = ed.get("exercise", {})
        if isinstance(exercise, dict) and exercise.get("type") is not None:
            has_data = True
        digestion = ed.get("digestion", {})
        if isinstance(digestion, dict) and digestion.get("status") is not None:
            has_data = True
        if has_data:
            count += 1
    return count


def generate_stories_for_user(user_id: str) -> dict:
    """Generate stories for all periods for a single user.

    Returns a dict of {period: "cached"|"generated"|"insufficient"|"empty"|"error"}.
    """
    results = {}
    for period in ("week", "month"):
        try:
            records = storage.list_messages(limit=200, period=period, user_id=user_id)

            if not records:
                results[period] = "empty"
                continue

            data_hash = storage._compute_data_hash(records)
            cached = storage.get_story_cache(user_id, period)

            # If hash unchanged, skip (already cached, including insufficient sentinels)
            if cached and cached.get("data_hash") == data_hash:
                results[period] = "cached"
                continue

            health_count = _count_health_messages(records)
            logger.info("  [%s] %s: %d records, %d with health data",
                        user_id, period, len(records), health_count)

            # ── Not enough data → cache a sentinel pointing to fallback template ──
            if health_count < MIN_MESSAGES:
                label = PERIOD_LABELS.get(period, period)
                fallback_narrative = FALLBACK_TEMPLATE.format(
                    period_label=label,
                    count=health_count,
                    min_count=MIN_MESSAGES,
                )
                result = {
                    "data_hash": data_hash,
                    "narrative": fallback_narrative,
                    "data_points": [],
                    "insights": [],
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "_fallback": True,  # sentinel: frontend/api can detect this
                    "_health_count": health_count,
                }
                storage.set_story_cache(user_id, period, result)
                results[period] = "insufficient"
                continue

            # ── Enough data → generate via LLM ──
            logger.info("  [%s] %s: generating story via LLM", user_id, period)

            correlations = compute_correlations(records)
            trend_insights = get_trend_insights(records)
            all_insights = correlations + trend_insights

            narrative = generate_body_story(records, period, all_insights)

            data_points = []
            for r in records:
                data_points.append({
                    "timestamp": r["timestamp"],
                    "content": r["content"],
                    "extracted": r.get("extracted_data", {}),
                })

            result = {
                "data_hash": data_hash,
                "narrative": narrative,
                "data_points": data_points,
                "insights": all_insights,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "_fallback": False,
            }
            storage.set_story_cache(user_id, period, result)
            results[period] = "generated"

        except LLMError as e:
            logger.error("  [%s] %s: LLM error: %s", user_id, period, e)
            results[period] = "error"
        except Exception:
            logger.exception("  [%s] %s: unexpected error", user_id, period)
            results[period] = "error"

    return results


def main() -> None:
    start = time.monotonic()
    now_str = datetime.now(timezone.utc).isoformat(timespec="seconds")

    logger.info("=== Weekly story pre-generation: %s ===", now_str)

    # Process ALL users who have ever sent messages (not just recently active)
    accounts = list_accounts()
    users_to_process: list[dict] = []

    for acc in accounts:
        if acc.get("is_deleted"):
            continue
        if acc.get("total_messages", 0) == 0:
            continue
        users_to_process.append(acc)

    logger.info("  %d total accounts, %d with messages", len(accounts), len(users_to_process))

    if not users_to_process:
        logger.info("  No users to process — exiting")
        elapsed = time.monotonic() - start
        logger.info("=== Run complete (%.1fs) ===", elapsed)
        return

    total = {"cached": 0, "generated": 0, "insufficient": 0, "empty": 0, "error": 0}
    for acc in users_to_process:
        uid = acc["user_id"]
        name = acc.get("display_name", uid)
        logger.info("  Processing user: %s (%s), %d messages", name, uid, acc.get("total_messages", 0))
        try:
            results = generate_stories_for_user(uid)
            for status in results.values():
                total[status] = total.get(status, 0) + 1
        except Exception:
            logger.exception("  Failed to process user %s", uid)
            total["error"] += 1

    elapsed = time.monotonic() - start
    logger.info(
        "=== Run complete (%.1fs) | generated=%d cached=%d insufficient=%d empty=%d error=%d ===",
        elapsed, total["generated"], total["cached"], total["insufficient"], total["empty"], total["error"],
    )


if __name__ == "__main__":
    main()
