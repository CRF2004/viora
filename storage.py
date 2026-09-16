"""
storage.py — JSON file-based message storage for Viora.

Provides a simple CRUD layer over a JSON file (messages.json) that mirrors
messages stored per account:
  id, user_id, content, timestamp, extracted_data (JSON), raw_text

Thread/process safety: uses fcntl.flock (Linux) for cross-process
synchronisation (gunicorn multi-worker).  Writes are atomic (temp file +
fsync + os.replace).  Corrupt JSON is preserved for forensics instead of
being silently discarded.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Any

# China Standard Time (UTC+8) — used for day/week/month period boundaries
CST = timezone(timedelta(hours=8))

from config import DATA_DIR, MESSAGES_FILE

logger = logging.getLogger(__name__)

LOCK_FILE = MESSAGES_FILE + ".lock"
STORY_CACHE_FILE = os.path.join(DATA_DIR, "story_cache.json")
STORY_CACHE_LOCK_FILE = STORY_CACHE_FILE + ".lock"


# ── File locking (cross-process, gunicorn-safe) ────────────────────────────

LOCK_TIMEOUT = 5.0       # max seconds to wait for a lock
LOCK_RETRY_INTERVAL = 0.05  # seconds between non-blocking attempts


class LockError(RuntimeError):
    """Raised when a file lock cannot be acquired within the timeout."""


def _acquire_lock(shared: bool = False):
    """Acquire a cross-process flock on LOCK_FILE (non-blocking with retry).

    Returns the open file descriptor.  The caller MUST call _release_lock().

    Uses fcntl.flock with LOCK_NB so gunicorn worker timeouts do not kill the
    process while it waits — retries every LOCK_RETRY_INTERVAL until
    LOCK_TIMEOUT elapses, then raises LockError.
    """
    import fcntl
    fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o666)
    op_nb = (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB
    deadline = time.monotonic() + LOCK_TIMEOUT
    while True:
        try:
            fcntl.flock(fd, op_nb)
            return fd
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise LockError(
                    f"Could not acquire {'shared' if shared else 'exclusive'} "
                    f"lock on {LOCK_FILE} within {LOCK_TIMEOUT}s"
                )
            time.sleep(LOCK_RETRY_INTERVAL)


def _release_lock(fd: int) -> None:
    """Release a previously acquired flock and close the file descriptor."""
    import fcntl
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except Exception:
        pass
    finally:
        try:
            os.close(fd)
        except Exception:
            pass


# ── File management ────────────────────────────────────────────────────────

def _ensure_data_dir() -> None:
    """Create the data directory if it does not exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def _read_all_nolock() -> list:
    """Read all messages from the JSON file.  Caller must hold the appropriate lock."""
    _ensure_data_dir()
    try:
        with open(MESSAGES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("messages.json is not a list, resetting.")
            return []
        return data
    except json.JSONDecodeError:
        corrupt_path = MESSAGES_FILE + ".corrupt." + str(int(time.time()))
        try:
            with open(MESSAGES_FILE, "rb") as src:
                raw_bytes = src.read()
            with open(corrupt_path, "wb") as dst:
                dst.write(raw_bytes)
            logger.error(
                "Corrupt JSON in %s (%d bytes) — preserved at %s",
                MESSAGES_FILE, len(raw_bytes), corrupt_path,
            )
        except Exception as copy_err:
            logger.error(
                "Failed to save corrupt-file copy from %s: %s",
                MESSAGES_FILE, copy_err,
            )
        raise
    except FileNotFoundError:
        logger.warning("Messages file not found (first run?), starting fresh.")
        return []


def _read_all() -> list:
    """Read all messages under a shared (read) lock.  Safe for read-only consumers."""
    fd = _acquire_lock(shared=True)
    try:
        return _read_all_nolock()
    finally:
        _release_lock(fd)


def _write_all_nolock(messages: list) -> None:
    """Atomically replace messages.json.  Caller must hold the exclusive lock."""
    _ensure_data_dir()
    tmp_path = MESSAGES_FILE + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, MESSAGES_FILE)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _write_all(messages: list) -> None:
    """Atomically replace messages.json under an exclusive lock."""
    fd = _acquire_lock(shared=False)
    try:
        _write_all_nolock(messages)
    finally:
        _release_lock(fd)


# ── CRUD operations ────────────────────────────────────────────────────────

def create_message(
    content: str,
    extracted_data: dict,
    user_id: str,
    raw_text: Optional[str] = None,
    ai_response: Optional[Any] = None,
    is_proactive: bool = False,
    message_id: Optional[str] = None,
) -> dict:
    """Create and persist a new message record (atomic, locked)."""
    msg = {
        "id": message_id or str(uuid.uuid4()),
        "user_id": user_id,
        "content": content,
        "ai_response": ai_response,
        "is_proactive": is_proactive,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "extracted_data": extracted_data,
        "raw_text": raw_text or content,
    }

    fd = _acquire_lock(shared=False)   # exclusive lock for read-modify-write
    try:
        messages = _read_all_nolock()
        messages.append(msg)
        _write_all_nolock(messages)
    finally:
        _release_lock(fd)

    logger.info("Saved message %s", msg["id"])

    # Update account activity
    if not is_proactive:
        try:
            from accounts import record_user_message
            record_user_message(user_id)
        except ImportError:
            pass

    return msg


def list_messages(
    limit: int = 20,
    offset: int = 0,
    period: Optional[str] = None,
    user_id: Optional[str] = None,
) -> list:
    """List messages with pagination, optional period filter and user filter."""
    messages = _read_all()

    if user_id:
        messages = [m for m in messages if m.get("user_id") == user_id]

    if period:
        now_cst = datetime.now(CST)
        cutoffs = {
            "day": now_cst.replace(hour=0, minute=0, second=0, microsecond=0),
            "week": now_cst.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=now_cst.weekday()),
            "month": now_cst.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
        }
        cutoff = cutoffs.get(period)
        if cutoff:
            # Both cutoff (CST) and message timestamp (UTC) are tz-aware; comparison works directly
            messages = [m for m in messages if datetime.fromisoformat(m["timestamp"]) >= cutoff]

    messages.sort(key=lambda m: m["timestamp"], reverse=True)
    return messages[offset : offset + limit]


def get_message(message_id: str) -> Optional[dict]:
    """Get a single message by ID."""
    messages = _read_all()
    for m in messages:
        if m["id"] == message_id:
            return m
    return None


def delete_message(message_id: str) -> bool:
    """Delete a message by ID (atomic, locked).  Returns True if found and deleted."""
    fd = _acquire_lock(shared=False)   # exclusive lock for read-modify-write
    try:
        messages = _read_all_nolock()
        new_messages = [m for m in messages if m["id"] != message_id]
        if len(new_messages) == len(messages):
            return False
        _write_all_nolock(new_messages)
        return True
    finally:
        _release_lock(fd)


def delete_messages_for_user(user_id: str) -> int:
    """Delete all messages for a given user_id (atomic, locked)."""
    fd = _acquire_lock(shared=False)   # exclusive lock for read-modify-write
    try:
        messages = _read_all_nolock()
        new_messages = [m for m in messages if m.get("user_id") != user_id]
        deleted = len(messages) - len(new_messages)
        if deleted:
            _write_all_nolock(new_messages)
        return deleted
    finally:
        _release_lock(fd)


# ── Story cache ─────────────────────────────────────────────────────────────


def _read_story_cache_nolock() -> dict:
    """Read the story cache file (caller must hold lock).

    Legacy/corrupt caches are tolerated: a non-dict top level (or a non-dict
    user bucket / period value) is dropped so callers can safely .get() into
    the nested {user_id: {period: result_dict}} structure.
    """
    try:
        with open(STORY_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        uid: {period: val for period, val in bucket.items()
              if isinstance(val, dict)}
        for uid, bucket in data.items()
        if isinstance(bucket, dict)
    }


def _write_story_cache_nolock(cache: dict) -> None:
    """Atomically write the story cache file (caller must hold lock)."""
    _ensure_data_dir()
    tmp_path = STORY_CACHE_FILE + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, STORY_CACHE_FILE)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _story_cache_lock(shared: bool = False):
    """Acquire lock on story cache file using the standard retry pattern."""
    import fcntl
    fd = os.open(STORY_CACHE_LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o666)
    op_nb = (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB
    deadline = time.monotonic() + LOCK_TIMEOUT
    while True:
        try:
            fcntl.flock(fd, op_nb)
            return fd
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise LockError(
                    f"Could not acquire story-cache lock within {LOCK_TIMEOUT}s"
                )
            time.sleep(LOCK_RETRY_INTERVAL)


def _release_story_cache_lock(fd: int) -> None:
    """Release story cache lock."""
    import fcntl
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except Exception:
        pass
    finally:
        try:
            os.close(fd)
        except Exception:
            pass


def get_story_cache(user_id: str, period: str) -> Optional[dict]:
    """Get cached story for (user_id, period). Returns None if not cached."""
    fd = _story_cache_lock(shared=True)
    try:
        cache = _read_story_cache_nolock()
        user_cache = cache.get(user_id)
        if not isinstance(user_cache, dict):
            return None
        return user_cache.get(period)
    finally:
        _release_story_cache_lock(fd)


def set_story_cache(user_id: str, period: str, result: dict) -> None:
    """Cache a story result for (user_id, period)."""
    fd = _story_cache_lock(shared=False)
    try:
        cache = _read_story_cache_nolock()
        user_cache = cache.get(user_id)
        if not isinstance(user_cache, dict):
            user_cache = {}
            cache[user_id] = user_cache
        user_cache[period] = result
        _write_story_cache_nolock(cache)
    finally:
        _release_story_cache_lock(fd)


def clear_story_cache(user_id: str) -> None:
    """Clear all cached stories for a user (e.g. after new message)."""
    fd = _story_cache_lock(shared=False)
    try:
        cache = _read_story_cache_nolock()
        if user_id in cache:
            del cache[user_id]
            _write_story_cache_nolock(cache)
    finally:
        _release_story_cache_lock(fd)


def _compute_data_hash(messages: list) -> str:
    """Compute a hash of message count + max timestamp to detect changes.

    Non-dict or timestamp-less entries are ignored so a legacy/corrupt
    messages list cannot crash story generation.
    """
    timestamps = [
        m.get("timestamp") for m in messages
        if isinstance(m, dict) and m.get("timestamp") is not None
    ]
    if not timestamps:
        return "empty"
    return f"{len(timestamps)}:{max(timestamps)}"


# ── Helpers ────────────────────────────────────────────────────────────────

def _iso_week_start(dt: datetime) -> datetime:
    """Return the Monday of the current ISO week at midnight UTC."""
    days_since_monday = dt.weekday()  # Monday = 0
    monday = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    from datetime import timedelta
    return monday - timedelta(days=days_since_monday)


# ── Trajectory graph storage ───────────────────────────────────────────────
# Persists trajectory_graph data (events + trajectories) per user.
# Uses the same flock + atomic-write pattern as message storage.

from config import TRAJECTORY_EVENTS_FILE, TRAJECTORIES_FILE

TRAJECTORY_EVENTS_LOCK = TRAJECTORY_EVENTS_FILE + ".lock"
TRAJECTORIES_LOCK = TRAJECTORIES_FILE + ".lock"


def _read_json_nolock(filepath: str) -> dict:
    """Read a JSON object from file. Returns empty dict on missing/corrupt."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_json_nolock(filepath: str, data: dict) -> None:
    """Atomically write a JSON object to file."""
    _ensure_data_dir()
    tmp_path = filepath + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, filepath)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _user_json_lock(filepath: str, shared: bool = False):
    """Acquire exclusive or shared lock on a JSON file's lock file."""
    import fcntl
    lock_path = filepath + ".lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    op_nb = (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB
    deadline = time.monotonic() + LOCK_TIMEOUT
    while True:
        try:
            fcntl.flock(fd, op_nb)
            return fd
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise LockError(f"Could not acquire lock on {lock_path} within {LOCK_TIMEOUT}s")
            time.sleep(LOCK_RETRY_INTERVAL)


def _user_json_unlock(fd: int) -> None:
    """Release a lock acquired by _user_json_lock."""
    import fcntl
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except Exception:
        pass
    finally:
        try:
            os.close(fd)
        except Exception:
            pass


# ── Trajectory events (per-user) ───────────────────────────────────────────

def load_trajectory_events(user_id: str) -> list[dict]:
    """Load serialized EventNodes for a given user."""
    fd = _user_json_lock(TRAJECTORY_EVENTS_FILE, shared=True)
    try:
        store = _read_json_nolock(TRAJECTORY_EVENTS_FILE)
        return store.get(user_id, [])
    finally:
        _user_json_unlock(fd)


def append_trajectory_events(user_id: str, new_events: list[dict]) -> list[dict]:
    """Append new events for a user and return the full event list."""
    fd = _user_json_lock(TRAJECTORY_EVENTS_FILE, shared=False)
    try:
        store = _read_json_nolock(TRAJECTORY_EVENTS_FILE)
        existing = store.get(user_id, [])
        # Avoid duplicates by event id
        existing_ids = {e["id"] for e in existing}
        for evt in new_events:
            if evt["id"] not in existing_ids:
                existing.append(evt)
                existing_ids.add(evt["id"])
        store[user_id] = existing
        _write_json_nolock(TRAJECTORY_EVENTS_FILE, store)
        return existing
    finally:
        _user_json_unlock(fd)


def replace_trajectory_events(user_id: str, events: list[dict]) -> None:
    """Replace all events for a user (used on full rebuild)."""
    fd = _user_json_lock(TRAJECTORY_EVENTS_FILE, shared=False)
    try:
        store = _read_json_nolock(TRAJECTORY_EVENTS_FILE)
        store[user_id] = events
        _write_json_nolock(TRAJECTORY_EVENTS_FILE, store)
    finally:
        _user_json_unlock(fd)


# ── Trajectory patterns (per-user) ─────────────────────────────────────────

def load_trajectories(user_id: str) -> list[dict]:
    """Load detected trajectory patterns for a given user."""
    fd = _user_json_lock(TRAJECTORIES_FILE, shared=True)
    try:
        store = _read_json_nolock(TRAJECTORIES_FILE)
        return store.get(user_id, [])
    finally:
        _user_json_unlock(fd)


def save_trajectories(user_id: str, trajectories: list[dict]) -> None:
    """Replace all detected trajectory patterns for a user."""
    fd = _user_json_lock(TRAJECTORIES_FILE, shared=False)
    try:
        store = _read_json_nolock(TRAJECTORIES_FILE)
        store[user_id] = trajectories
        _write_json_nolock(TRAJECTORIES_FILE, store)
    finally:
        _user_json_unlock(fd)


# ── Psycho event storage (per-user) ────────────────────────────────────────
# Psychological events (PsychoEvent) extracted from chat messages.
# Each entry: {user_id: [PsychoEvent.to_dict(), ...]}

from config import PSYCHO_EVENTS_FILE

PSYCHO_EVENTS_LOCK = PSYCHO_EVENTS_FILE + ".lock"


def load_psycho_events(user_id: str) -> list[dict]:
    """Load psycho event list for a user."""
    fd = _user_json_lock(PSYCHO_EVENTS_FILE, shared=True)
    try:
        store = _read_json_nolock(PSYCHO_EVENTS_FILE)
        return store.get(user_id, [])
    finally:
        _user_json_unlock(fd)


def append_psycho_event(user_id: str, event: dict) -> list[dict]:
    """Append a single psycho event for a user and return full list."""
    fd = _user_json_lock(PSYCHO_EVENTS_FILE, shared=False)
    try:
        store = _read_json_nolock(PSYCHO_EVENTS_FILE)
        events = store.get(user_id, [])
        # Avoid duplicate by id
        if not any(e.get("id") == event.get("id") for e in events):
            events.append(event)
        store[user_id] = events
        _write_json_nolock(PSYCHO_EVENTS_FILE, store)
        return events
    finally:
        _user_json_unlock(fd)


# ── Psycho state storage ───────────────────────────────────────────────────
# Persistent aggregated PsychoState per user.
from config import PSYCHO_STATES_FILE, PSYCHO_FINGERPRINTS_FILE

PSYCHO_STATES_LOCK = PSYCHO_STATES_FILE + ".lock"
PSYCHO_FINGERPRINTS_LOCK = PSYCHO_FINGERPRINTS_FILE + ".lock"


def load_psycho_state(user_id: str) -> dict | None:
    """Load the current aggregated PsychoState for a user. Returns None if none exists."""
    fd = _user_json_lock(PSYCHO_STATES_FILE, shared=True)
    try:
        store = _read_json_nolock(PSYCHO_STATES_FILE)
        return store.get(user_id)
    finally:
        _user_json_unlock(fd)


def save_psycho_state(user_id: str, state_dict: dict) -> None:
    """Save/replace the current aggregated PsychoState for a user."""
    fd = _user_json_lock(PSYCHO_STATES_FILE, shared=False)
    try:
        store = _read_json_nolock(PSYCHO_STATES_FILE)
        store[user_id] = state_dict
        _write_json_nolock(PSYCHO_STATES_FILE, store)
    finally:
        _user_json_unlock(fd)


# ── Psycho fingerprint storage ────────────────────────────────────────────
from config import PSYCHO_FINGERPRINTS_FILE

PSYCHO_FINGERPRINTS_LOCK = PSYCHO_FINGERPRINTS_FILE + ".lock"


def load_psycho_fingerprint(user_id: str) -> dict | None:
    """Load the cached psycho fingerprint for a user."""
    fd = _user_json_lock(PSYCHO_FINGERPRINTS_FILE, shared=True)
    try:
        store = _read_json_nolock(PSYCHO_FINGERPRINTS_FILE)
        return store.get(user_id)
    finally:
        _user_json_unlock(fd)


def save_psycho_fingerprint(user_id: str, fp: dict) -> None:
    """Cache the psycho fingerprint for a user."""
    fd = _user_json_lock(PSYCHO_FINGERPRINTS_FILE, shared=False)
    try:
        store = _read_json_nolock(PSYCHO_FINGERPRINTS_FILE)
        store[user_id] = fp
        _write_json_nolock(PSYCHO_FINGERPRINTS_FILE, store)
    finally:
        _user_json_unlock(fd)
