"""
accounts.py — Account management and activity tracking for Viora.

Manages user accounts with activity scoring that drives proactive
scheduler frequency. Supports:
- Registration / login with password hashes
- CRUD for accounts
- Activity score computation (login frequency + reply rate + response time)
- Auto-frequency adjustment based on activity level
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from werkzeug.security import check_password_hash, generate_password_hash

logger = logging.getLogger(__name__)

# ── Storage ──────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ACCOUNTS_FILE = os.path.join(DATA_DIR, "accounts.json")


def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ACCOUNTS_FILE):
        with open(ACCOUNTS_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)


def _read_all() -> dict:
    """Read all accounts from JSON file. Returns {user_id: account_dict}."""
    _ensure_data_dir()
    try:
        with open(ACCOUNTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def _write_all(accounts: dict) -> None:
    _ensure_data_dir()
    with open(ACCOUNTS_FILE, "w", encoding="utf-8") as f:
        json.dump(accounts, f, ensure_ascii=False, indent=2)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_account(
    user_id: str,
    display_name: str,
    username: Optional[str] = None,
    password_hash: Optional[str] = None,
) -> dict:
    now = _now_iso()
    return {
        "user_id": user_id,
        "username": username or user_id,
        "password_hash": password_hash,
        "display_name": display_name,
        "persona_id": "default",
        "created_at": now,
        "last_login_at": None,
        "last_active_at": now,
        "total_messages": 0,
        "proactive_received": 0,
        "proactive_replied": 0,
        "silent_until": None,
        "notification_enabled": True,
        "is_deleted": False,
    }


def _public_account(account: dict) -> dict:
    """Remove sensitive fields before sending to clients."""
    public = dict(account)
    public.pop("password_hash", None)
    return public


# ── CRUD ─────────────────────────────────────────────────────────────────────

def get_or_create_default() -> dict:
    """Get or create the implicit 'default' account. Always exists."""
    account = get_account("default")
    if account is None:
        account = create_account(user_id="default", display_name="默认用户", username="default")
    return account


def find_account_by_username(username: str) -> Optional[dict]:
    """Find an account by username."""
    if not username:
        return None
    username = username.strip()
    accounts = _read_all()
    for account in accounts.values():
        if account.get("is_deleted"):
            continue
        if account.get("username") == username:
            return account
    return None


def create_account(
    user_id: Optional[str] = None,
    display_name: str = "用户",
    username: Optional[str] = None,
    password: Optional[str] = None,
    password_hash: Optional[str] = None,
) -> dict:
    """Create a new account.

    If username is provided and already exists, raises ValueError.
    If user_id is None, generates one.
    """
    accounts = _read_all()

    if username:
        existing = find_account_by_username(username)
        if existing:
            raise ValueError("username_exists")

    if user_id is None:
        user_id = f"user_{uuid.uuid4().hex[:8]}"

    if user_id in accounts:
        return accounts[user_id]

    if password_hash is None and password is not None:
        password_hash = generate_password_hash(password)

    account = _default_account(user_id, display_name, username=username, password_hash=password_hash)
    accounts[user_id] = account
    _write_all(accounts)
    logger.info("Created account: %s (%s)", user_id, display_name)
    return account


def register_account(username: str, password: str, display_name: str = "用户") -> dict:
    """Register a new account using username/password."""
    username = (username or "").strip()
    if not username:
        raise ValueError("username_required")
    if not password:
        raise ValueError("password_required")

    return create_account(
        display_name=display_name or username,
        username=username,
        password=password,
    )


def authenticate(username: str, password: str) -> Optional[dict]:
    """Validate username/password and return the matching account."""
    account = find_account_by_username(username)
    if not account:
        return None
    password_hash = account.get("password_hash")
    if not password_hash:
        return None
    if not check_password_hash(password_hash, password or ""):
        return None
    return account


def mark_login(user_id: str) -> Optional[dict]:
    """Update login timestamps after successful authentication."""
    accounts = _read_all()
    if user_id not in accounts:
        return None
    accounts[user_id]["last_login_at"] = _now_iso()
    accounts[user_id]["last_active_at"] = _now_iso()
    _write_all(accounts)
    return accounts[user_id]


def get_account(user_id: str) -> Optional[dict]:
    """Get an account by user_id."""
    if not user_id:
        return None
    accounts = _read_all()
    return accounts.get(user_id)


def update_account(user_id: str, **kwargs) -> Optional[dict]:
    """Update account fields. Does NOT bump last_active_at (only real interactions do)."""
    accounts = _read_all()
    if user_id not in accounts:
        return None

    allowed = {
        "display_name",
        "persona_id",
        "silent_until",
        "notification_enabled",
        "last_active_at",
        "last_login_at",
        "total_messages",
        "proactive_received",
        "proactive_replied",
        "location",
        "location_city",
        "location_lat",
        "location_lon",
        "location_updated_at",
    }
    for key, val in kwargs.items():
        if key in allowed:
            accounts[user_id][key] = val

    # Do NOT auto-bump last_active_at here.  Only genuine user interactions
    # (mark_login, record_user_message, record_proactive_reply) should update it.
    # Otherwise compute_activity_score always returns days_since_active ≈ 0
    # because any account field change (persona, location, etc.) resets the clock.
    _write_all(accounts)
    return accounts[user_id]


def list_accounts() -> list[dict]:
    """Return all accounts as a list."""
    accounts = _read_all()
    return [acc for acc in accounts.values() if not acc.get("is_deleted")]


def public_account(account: Optional[dict]) -> Optional[dict]:
    """Return a sanitized account dict for API responses."""
    if not account:
        return None
    return _public_account(account)


def delete_account(user_id: str) -> bool:
    """Delete an account. Cannot delete 'default'."""
    if user_id == "default":
        return False
    accounts = _read_all()
    if user_id in accounts:
        del accounts[user_id]
        _write_all(accounts)
        return True
    return False


# ── Activity tracking ────────────────────────────────────────────────────────

def record_user_message(user_id: str) -> None:
    """Call when user sends a message. Updates counters and last_active_at."""
    accounts = _read_all()
    if user_id not in accounts:
        accounts[user_id] = _default_account(user_id, "用户")
    accounts[user_id]["total_messages"] = accounts[user_id].get("total_messages", 0) + 1
    accounts[user_id]["last_active_at"] = _now_iso()
    _write_all(accounts)


def record_proactive_received(user_id: str) -> None:
    """Call when a proactive message is sent to this user."""
    accounts = _read_all()
    if user_id not in accounts:
        accounts[user_id] = _default_account(user_id, "用户")
    accounts[user_id]["proactive_received"] = accounts[user_id].get("proactive_received", 0) + 1
    _write_all(accounts)


def record_proactive_reply(user_id: str) -> None:
    """Call when user replies to a proactive message."""
    accounts = _read_all()
    if user_id not in accounts:
        accounts[user_id] = _default_account(user_id, "用户")
    accounts[user_id]["proactive_replied"] = accounts[user_id].get("proactive_replied", 0) + 1
    accounts[user_id]["last_active_at"] = _now_iso()
    _write_all(accounts)


def compute_activity_score(user_id: str) -> dict:
    """
    Compute activity score for a user.

    Returns:
        {
            "score": float 0-1,
            "level": "active" | "idle" | "inactive",
            "reply_rate": float 0-1,
            "days_since_active": int,
            "total_messages": int,
            "proactive_received": int,
            "proactive_replied": int,
        }
    """
    account = get_account(user_id)
    if account is None:
        return {
            "score": 0,
            "level": "inactive",
            "reply_rate": 0,
            "days_since_active": 999,
            "total_messages": 0,
            "proactive_received": 0,
            "proactive_replied": 0,
        }

    try:
        last_active = datetime.fromisoformat(account.get("last_active_at", ""))
        if last_active.tzinfo is None:
            last_active = last_active.replace(tzinfo=timezone.utc)
        days_since = (datetime.now(timezone.utc) - last_active).total_seconds() / 86400
    except (ValueError, TypeError):
        days_since = 999
    days_since = int(days_since)

    received = account.get("proactive_received", 0)
    replied = account.get("proactive_replied", 0)
    reply_rate = replied / max(received, 1)

    level = "active"
    if days_since > 14:
        level = "inactive"
    elif days_since > 7:
        level = "idle"

    if received >= 3 and replied == 0 and level == "active":
        level = "idle"

    recency_score = max(0, 1 - days_since / 14)
    reply_score = reply_rate
    msg_score = min(account.get("total_messages", 0) / 50, 1)
    score = round(recency_score * 0.4 + reply_score * 0.3 + msg_score * 0.3, 2)

    return {
        "score": score,
        "level": level,
        "reply_rate": round(reply_rate, 2),
        "days_since_active": days_since,
        "total_messages": account.get("total_messages", 0),
        "proactive_received": received,
        "proactive_replied": replied,
    }
