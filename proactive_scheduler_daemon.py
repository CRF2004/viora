"""
proactive_scheduler_daemon.py — Long-running daemon that triggers proactive
caring messages for Viora.

Managed by PM2.  Wakes every CHECK_INTERVAL minutes, iterates over all active
users, checks whether each should receive a proactive message, and generates +
stores one if conditions are met.

Rules enforced by scheduler.py:
- Max 1 proactive message per day per user
- Not during quiet hours (23:00-7:00)
- Not during busy hours (9:00-12:00 on weekdays)
- 12h cooldown after last proactive (168h for idle users)
- Auto-backoff if user doesn't respond 2+ times in a row
- Silent mode support
"""

import time
import sys
import os
import logging
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import storage
from accounts import list_accounts, compute_activity_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("proactive-scheduler")

CST = timezone(timedelta(hours=8))

# Wake up every CHECK_INTERVAL minutes to inspect all users
# (matches CHECK_INTERVAL concept in scheduler.py)
CHECK_INTERVAL_MINUTES = 60

# Don't even run the loop during deep quiet hours — save CPU, avoid API noise.
# End at 5 AM (was 6) to accommodate users who wake up early (5-6 AM).
# Per-user quiet hours still gate individual message delivery.
DEEP_SLEEP_START = 23  # 11 PM
DEEP_SLEEP_END = 5     # 5 AM

# Skip accounts that haven't logged in for N days (dormant).
# They'll be picked up again automatically the next time they log in.
DORMANT_DAYS_THRESHOLD = 3


def _now_cst_str() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")


def _in_deep_sleep() -> bool:
    """Skip the entire loop during deep night — nobody should be bothered."""
    hour = datetime.now(CST).hour
    return hour >= DEEP_SLEEP_START or hour < DEEP_SLEEP_END


def _seconds_until_next_check() -> float:
    return CHECK_INTERVAL_MINUTES * 60


def run_proactive_cycle() -> dict:
    """
    One full proactive check cycle for all active users.

    Returns a summary dict with per-status counts.
    """
    from scheduler import (
        should_send_proactive,
        generate_proactive_message,
        record_proactive_sent,
        SchedulerState,
    )

    accounts = list_accounts()
    summary = {
        "checked": 0,
        "sent": 0,
        "skipped_inactive": 0,
        "skipped_dormant": 0,
        "skipped_rules": 0,
        "skipped_no_anomalies": 0,
        "error": 0,
        "reasons": {},
    }

    deadline = datetime.now(CST) - timedelta(days=DORMANT_DAYS_THRESHOLD)

    for acc in accounts:
        user_id = acc.get("user_id", "")
        if not user_id or acc.get("is_deleted"):
            continue

        total_msgs = acc.get("total_messages", 0)
        if total_msgs == 0:
            continue

        # ── Dormant check: skip if hasn't logged in for N days ──
        # Resets automatically when they log in again (last_login_at updates).
        last_login = acc.get("last_login_at")
        if last_login:
            try:
                login_dt = datetime.fromisoformat(last_login)
                if login_dt.replace(tzinfo=None) < deadline.replace(tzinfo=None):
                    summary["skipped_dormant"] += 1
                    continue
            except (ValueError, TypeError):
                pass  # unparseable timestamp → give benefit of doubt, check anyway

        summary["checked"] += 1

        try:
            # ── Step 0: Load user state for wake-up time ──
            wake_up_hour = None
            try:
                from user_state import get_user_state, extract_wake_up_hour_from_profile
                us = get_user_state(user_id)
                # First check explicit user_preferences
                wake_up_hour = us.get("user_preferences", {}).get("wake_up_hour")
                # If not set, try extracting from profile_notes
                if wake_up_hour is None:
                    profile = us.get("profile_notes", "")
                    wake_up_hour = extract_wake_up_hour_from_profile(profile)
            except Exception:
                pass

            # ── Step 1: Check scheduling rules ──
            state = SchedulerState.load(user_id)
            try:
                from storage import load_psycho_state
                psycho_state = load_psycho_state(user_id)
            except (ImportError, Exception):
                psycho_state = None
            decision = should_send_proactive(state, user_id, wake_up_hour=wake_up_hour, psycho_state=psycho_state)

            if not decision["should_send"]:
                reason = decision.get("reason", "unknown")
                summary["reasons"][reason] = summary["reasons"].get(reason, 0) + 1
                if reason == "user_inactive":
                    summary["skipped_inactive"] += 1
                else:
                    summary["skipped_rules"] += 1
                continue

            # ── Step 2: Get recent messages ──
            messages = storage.list_messages(limit=200, user_id=user_id)
            if not messages:
                summary["skipped_no_anomalies"] += 1
                continue

            # ── Step 3: Generate proactive message (multi-tier: anomaly → general) ──
            persona_id = acc.get("persona_id")
            msg = generate_proactive_message(
                messages,
                persona_id=persona_id,
                user_id=user_id,
                account=acc,
                wake_up_hour=wake_up_hour,
            )

            if msg is None:
                summary["skipped_no_anomalies"] += 1
                continue

            # ── Step 4: Store + record ──
            storage.create_message(
                content="",
                extracted_data={},
                user_id=user_id,
                ai_response=msg,
                is_proactive=True,
            )
            record_proactive_sent(user_id)

            display = acc.get("display_name", user_id)
            logger.info(
                "Proactive message sent → %s (%s) | persona=%s",
                display, user_id, persona_id or "default",
            )

            summary["sent"] += 1

        except Exception:
            logger.exception("Error processing user %s during proactive cycle", user_id)
            summary["error"] += 1

        # ── Step 5: Check plan reminders ──
        try:
            _check_plan_reminders_for_user(user_id)
        except Exception:
            logger.exception("Error checking plan reminders for user %s", user_id)
            summary["error"] += 1

    return summary


def _check_plan_reminders_for_user(user_id: str) -> int:
    """
    Check and fire due plan reminders for a user.
    Returns the number of reminders sent.
    """
    now = datetime.now(CST)
    current_hour = now.hour
    current_minute = now.minute
    current_weekday = now.weekday() + 1  # Mon=1, Sun=7

    try:
        from health_plan import list_reminders, get_active_plan
        reminders = list_reminders(user_id, active_only=True)
    except Exception:
        return 0

    sent = 0
    for reminder in reminders:
        if not reminder.is_active:
            continue

        # Parse time_spec (e.g., "15:00")
        try:
            parts = reminder.time_spec.split(":")
            target_hour = int(parts[0])
            target_minute = int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            continue

        # Check time (±5 minute window)
        time_diff = (current_hour * 60 + current_minute) - (target_hour * 60 + target_minute)
        if abs(time_diff) > 5:
            continue

        # Check day of week
        if reminder.days_of_week and current_weekday not in reminder.days_of_week:
            continue

        # Adaptive: check user state before sending
        if reminder.adaptive:
            try:
                from user_state import get_user_state
                state = get_user_state(user_id)
                anomaly = state.get("anomaly_level", "low")
                # Skip if user is in a bad state
                if anomaly == "high":
                    logger.info("Skipping reminder %s for %s due to high anomaly", reminder.id[:8], user_id)
                    continue
                # Skip if silent mode (from scheduler)
                try:
                    from scheduler import SchedulerState
                    sched = SchedulerState.load(user_id)
                    if sched.get("silent_until"):
                        from datetime import datetime as _dt
                        silent_until = _dt.fromisoformat(sched["silent_until"])
                        if silent_until > now:
                            continue
                except Exception:
                    pass
            except Exception:
                pass

        # Check if item already completed today
        plan = get_active_plan(user_id)
        if plan and reminder.plan_id == plan.id:
            from health_plan import get_today_items
            today_items = get_today_items(plan)
            action_lower = reminder.message_template.lower()
            already_done = any(
                item.status == "done" and item.action.lower() in action_lower
                for item in today_items
            )
            if already_done:
                logger.info("Skipping reminder %s — item already done today", reminder.id[:8])
                continue

        # Fire reminder: store as proactive message
        try:
            storage.create_message(
                content="",
                extracted_data={},
                user_id=user_id,
                ai_response=[reminder.message_template],
                is_proactive=True,
            )
            logger.info(
                "Plan reminder sent → %s | reminder=%s time=%s",
                user_id, reminder.id[:8], reminder.time_spec,
            )
            sent += 1
        except Exception:
            logger.exception("Failed to store plan reminder for user %s", user_id)

    return sent


def main_loop() -> None:
    """Main daemon loop.  Runs until killed by PM2."""
    logger.info(
        "[proactive-scheduler] Started at %s CST | check_interval=%dm | deep_sleep=%02d:00-%02d:00",
        _now_cst_str(), CHECK_INTERVAL_MINUTES, DEEP_SLEEP_START, DEEP_SLEEP_END,
    )

    while True:
        # ── Deep sleep: skip the whole loop ──
        if _in_deep_sleep():
            # Wake up at DEEP_SLEEP_END
            now = datetime.now(CST)
            wake = now.replace(hour=DEEP_SLEEP_END, minute=0, second=0, microsecond=0)
            if now.hour >= DEEP_SLEEP_START:
                # Past midnight → next morning
                wake += timedelta(days=1)
            wait = (wake - now).total_seconds()
            hours = wait / 3600
            logger.info("[proactive-scheduler] Deep sleep (%02d:00-%02d:00) — waking in %.1fh",
                        DEEP_SLEEP_START, DEEP_SLEEP_END, hours)
            time.sleep(wait)
            continue

        # ── Run proactive cycle ──
        logger.info("[proactive-scheduler] === Cycle start at %s CST ===", _now_cst_str())
        start = time.monotonic()
        summary = run_proactive_cycle()
        elapsed = time.monotonic() - start

        logger.info(
            "[proactive-scheduler] === Cycle done (%.1fs) | "
            "checked=%d sent=%d dormant=%d inactive=%d rules=%d no_anomaly=%d err=%d ===",
            elapsed,
            summary["checked"],
            summary["sent"],
            summary["skipped_dormant"],
            summary["skipped_inactive"],
            summary["skipped_rules"],
            summary["skipped_no_anomalies"],
            summary["error"],
        )

        # ── Wait for next cycle ──
        wait = _seconds_until_next_check()
        logger.info("[proactive-scheduler] Next cycle in %dm", int(wait / 60))
        time.sleep(wait)


if __name__ == "__main__":
    main_loop()
