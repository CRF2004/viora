import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from config import DATA_DIR

logger = logging.getLogger(__name__)

FEEDBACKS_FILE = os.path.join(DATA_DIR, "feedbacks.json")
LOCK_FILE = FEEDBACKS_FILE + ".lock"

LOCK_TIMEOUT = 5.0
LOCK_RETRY_INTERVAL = 0.05

class LockError(RuntimeError):
    pass

def _acquire_feedback_lock(shared: bool = False):
    """Acquire flock on LOCK_FILE with non-blocking retry. Same pattern as storage.py."""
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
                raise LockError(f"Could not acquire lock on {LOCK_FILE}")
            time.sleep(LOCK_RETRY_INTERVAL)

def _release_feedback_lock(fd: int) -> None:
    """Release flock and close fd. Same pattern as storage.py."""
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

def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

def _is_feedback_dict(entry) -> bool:
    """Return True if *entry* looks like a feedback dict (not structured junk).

    Guards against legacy/corrupt entries (bare strings / numbers / nulls that
    can leak into a hand-edited or partially-written feedbacks.json): callers
    that iterate the list skip non-dict entries instead of crashing on .get().
    """
    return isinstance(entry, dict)


def _read_feedbacks_nolock() -> list:
    """Read all feedbacks from JSON. Caller must hold lock."""
    _ensure_data_dir()
    try:
        with open(FEEDBACKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("feedbacks.json is not a list, resetting.")
            return []
        return data
    except json.JSONDecodeError:
        # Preserve corrupt file for forensics, then auto-recover with empty list
        corrupt_path = FEEDBACKS_FILE + ".corrupt." + str(int(time.time()))
        try:
            with open(FEEDBACKS_FILE, "rb") as src:
                raw_bytes = src.read()
            with open(corrupt_path, "wb") as dst:
                dst.write(raw_bytes)
            logger.error("Corrupt JSON in %s (%d bytes) — preserved at %s, auto-recovering with empty list",
                        FEEDBACKS_FILE, len(raw_bytes), corrupt_path)
        except Exception as copy_err:
            logger.error("Failed to save corrupt-file copy: %s", copy_err)
        # Return empty list so the API doesn't 500; corrupt data is preserved for recovery
        return []
    except FileNotFoundError:
        logger.warning("Feedbacks file not found (first run?), starting fresh.")
        return []

def _write_feedbacks_nolock(feedbacks: list) -> None:
    """Atomically write feedbacks.json. Caller must hold lock."""
    _ensure_data_dir()
    tmp_path = FEEDBACKS_FILE + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(feedbacks, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, FEEDBACKS_FILE)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

def create_feedback(user_id: str, type: str, description: str) -> dict:
    """Create a new feedback item."""
    now = datetime.now(timezone.utc).isoformat()
    fb = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "type": type,
        "description": description,
        "completed": False,
        "created_at": now,
        "updated_at": now,
    }
    fd = _acquire_feedback_lock(shared=False)
    try:
        feedbacks = _read_feedbacks_nolock()
        feedbacks.append(fb)
        _write_feedbacks_nolock(feedbacks)
    finally:
        _release_feedback_lock(fd)
    logger.info("Created feedback %s for user %s", fb["id"], user_id)
    return fb

def list_feedbacks(user_id: str) -> list:
    """List all feedbacks for a user, sorted by created_at descending."""
    fd = _acquire_feedback_lock(shared=True)
    try:
        feedbacks = _read_feedbacks_nolock()
    finally:
        _release_feedback_lock(fd)
    user_feedbacks = [fb for fb in feedbacks
                      if _is_feedback_dict(fb) and fb.get("user_id") == user_id]
    # Sort defensively: a corrupt entry missing "created_at" must not crash the
    # list endpoint (fall back to "" so it sorts last in descending order).
    user_feedbacks.sort(key=lambda fb: fb.get("created_at", ""), reverse=True)
    return user_feedbacks

def update_feedback(feedback_id: str, user_id: str, **updates) -> dict | None:
    """Update a feedback item. Only the owner can update."""
    fd = _acquire_feedback_lock(shared=False)
    try:
        feedbacks = _read_feedbacks_nolock()
        for fb in feedbacks:
            if not _is_feedback_dict(fb):
                continue  # corrupt non-dict entry: can't own a feedback id
            if fb["id"] == feedback_id and fb["user_id"] == user_id:
                if "type" in updates:
                    fb["type"] = updates["type"]
                if "description" in updates:
                    fb["description"] = updates["description"]
                if "completed" in updates:
                    fb["completed"] = bool(updates["completed"])
                fb["updated_at"] = datetime.now(timezone.utc).isoformat()
                _write_feedbacks_nolock(feedbacks)
                logger.info("Updated feedback %s", feedback_id)
                return dict(fb)
        return None
    finally:
        _release_feedback_lock(fd)

def delete_feedback(feedback_id: str, user_id: str) -> bool:
    """Delete a feedback item. Only the owner can delete."""
    fd = _acquire_feedback_lock(shared=False)
    try:
        feedbacks = _read_feedbacks_nolock()
        new_feedbacks = [fb for fb in feedbacks
                        if not (_is_feedback_dict(fb)
                                and fb["id"] == feedback_id and fb["user_id"] == user_id)]
        deleted = len(new_feedbacks) < len(feedbacks)
        if deleted:
            _write_feedbacks_nolock(new_feedbacks)
            logger.info("Deleted feedback %s", feedback_id)
        return deleted
    finally:
        _release_feedback_lock(fd)
