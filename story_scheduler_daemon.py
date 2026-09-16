"""
story_scheduler_daemon.py — Long-running daemon that triggers body story
generation every Sunday at 00:17 (Beijing time).

Managed by PM2.  Sleeps through the week, wakes on Sunday to run the
pre-generation script, then goes back to sleep.
"""

import time
import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scheduled_stories import main

CST = timezone(timedelta(hours=8))
# Run at 00:17 on Sunday (Beijing time)
TARGET_HOUR = 0
TARGET_MINUTE = 17
TARGET_WEEKDAY = 6  # Sunday (Monday=0)

# Check interval — how often to wake up and ask "is it Sunday 00:17 yet?"
CHECK_INTERVAL = 300  # 5 minutes


def _seconds_until_next_run() -> float:
    """Return seconds until the next Sunday 00:17 CST."""
    now = datetime.now(CST)
    # Build the target datetime for this week's Sunday 00:17
    days_until_sunday = (TARGET_WEEKDAY - now.weekday()) % 7
    if days_until_sunday == 0:
        # It's Sunday — are we before or after 00:17?
        target_today = now.replace(hour=TARGET_HOUR, minute=TARGET_MINUTE, second=0, microsecond=0)
        if now < target_today:
            # Before 00:17 → run later today
            return (target_today - now).total_seconds()
        else:
            # After 00:17 → run next Sunday
            days_until_sunday = 7
    # Build next Sunday 00:17
    next_sunday = now.replace(hour=TARGET_HOUR, minute=TARGET_MINUTE, second=0, microsecond=0)
    next_sunday += timedelta(days=days_until_sunday)
    delta = (next_sunday - now).total_seconds()
    return max(delta, 0)


def _now_cst_str() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    print(f"[story-scheduler] Started at {_now_cst_str()} CST", flush=True)
    print(f"[story-scheduler] Will run every Sunday at {TARGET_HOUR:02d}:{TARGET_MINUTE:02d} CST", flush=True)

    while True:
        wait = _seconds_until_next_run()
        hours = wait / 3600
        print(f"[story-scheduler] Next run in {hours:.1f}h (Sunday {TARGET_HOUR:02d}:{TARGET_MINUTE:02d} CST)",
              flush=True)
        # Sleep in chunks so we don't miss by oversleeping
        while wait > 0:
            chunk = min(wait, CHECK_INTERVAL)
            time.sleep(chunk)
            wait -= chunk

        # It's Sunday 00:17! Run the generation.
        print(f"[story-scheduler] === Triggering weekly generation at {_now_cst_str()} CST ===", flush=True)
        try:
            main()
        except Exception:
            import traceback
            traceback.print_exc()
        print(f"[story-scheduler] === Weekly run finished at {_now_cst_str()} CST ===", flush=True)

        # Small cooldown to avoid re-triggering in the same minute
        time.sleep(120)
