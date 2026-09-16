"""
health_plan.py — Data model and storage layer for the Health Plan module.

Provides dataclass models (HealthPlan, ReasoningNode, PlanReminder)
and a full CRUD layer over two JSON files (health_plans.json, plan_reminders.json).

Thread/process safety: uses fcntl.flock (Linux) for cross-process
synchronisation (gunicorn multi-worker).  Writes are atomic (temp file +
fsync + os.replace).  Corrupt JSON is preserved for forensics instead of
being silently discarded.

Follows the exact same patterns as storage.py.
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from config import DATA_DIR

logger = logging.getLogger(__name__)

# ── File paths ───────────────────────────────────────────────────────────────

HEALTH_PLANS_FILE = os.path.join(DATA_DIR, "health_plans.json")
PLAN_REMINDERS_FILE = os.path.join(DATA_DIR, "plan_reminders.json")

HEALTH_PLANS_LOCK_FILE = HEALTH_PLANS_FILE + ".lock"
PLAN_REMINDERS_LOCK_FILE = PLAN_REMINDERS_FILE + ".lock"

# ── Time slot sort order ─────────────────────────────────────────────────────

TIME_SLOT_ORDER = {
    "早晨起床": 0,
    "早餐": 1,
    "上午": 2,
    "午餐": 3,
    "下午": 4,
    "傍晚": 5,
    "晚餐": 6,
    "睡前": 7,
    "深夜": 8,
    "": 99,
}

# ── Data models ──────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class ReasoningNode:
    """A node in the health plan reasoning tree."""

    id: str
    label: str                          # "高升糖饮食" / "早餐全麦面包替代白面包"
    description: str                    # 机制说明
    emoji: str                          # 🧬 🥗 🏃 😰 🍽️
    depth: int                          # 根节点=0
    parent_id: str | None = None
    children: list["ReasoningNode"] = field(default_factory=list)
    expanded: bool = False              # 用户是否已展开
    expandable: bool = True             # False=已到叶子不能再展开

    # 叶子节点专属
    is_actionable: bool = False         # True=叶子节点，有具体建议
    action_text: str | None = None      # "早餐全麦面包替代白面包"
    time_slot: str | None = None        # "早餐" | "午餐" | "傍晚" | "睡前"
    frequency: str | None = None        # "每天" | "每周3次"
    duration: str | None = None         # "15分钟" | None
    reasoning: str | None = None        # 科学依据

    # 状态跟踪
    status: str = "pending"             # "pending" | "done" | "skipped" | "modified"
    completed_count: int = 0
    last_completed: str | None = None
    created_at: str = ""

    # 交互式信息搜集 (Phase 4 待完善方向)
    followup_question: str | None = None   # 待用户回答的追问；None=无需追问
    followup_options: list[str] | None = None  # 选择题选项（可与追问搭配，快速作答）


@dataclass
class HealthPlan:
    """A complete health plan for a user."""

    id: str  # uuid4
    user_id: str
    concern: str  # "最近脱发好严重"
    title: str  # "🌿 头发养护计划"
    goal: str  # 目标描述
    status: str = "active"  # "active" | "archived" | "completed"
    created_at: str = ""  # ISO timestamp
    updated_at: str = ""  # ISO timestamp
    context_snapshot: dict = field(default_factory=dict)
    reasoning_tree: ReasoningNode | None = None
    version: int = 1


@dataclass
class PlanReminder:
    """A scheduled reminder associated with a health plan."""

    id: str  # uuid4
    plan_id: str
    user_id: str
    trigger_type: str = "time"
    time_spec: str = ""  # "15:00"
    days_of_week: list[int] | None = None  # [1,3,5] or None=daily
    message_template: str = ""
    is_active: bool = True
    last_triggered: str | None = None
    adaptive: bool = True
    created_at: str = ""


# ── File locking (cross-process, gunicorn-safe) ────────────────────────────

LOCK_TIMEOUT = 5.0       # max seconds to wait for a lock
LOCK_RETRY_INTERVAL = 0.05  # seconds between non-blocking attempts


class LockError(RuntimeError):
    """Raised when a file lock cannot be acquired within the timeout."""


def _acquire_lock(lock_path: str, shared: bool = False):
    """Acquire a cross-process flock on *lock_path* (non-blocking with retry).

    Returns the open file descriptor.  The caller MUST call _release_lock().

    Uses fcntl.flock with LOCK_NB so gunicorn worker timeouts do not kill the
    process while it waits — retries every LOCK_RETRY_INTERVAL until
    LOCK_TIMEOUT elapses, then raises LockError.
    """
    import fcntl

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
                raise LockError(
                    f"Could not acquire {'shared' if shared else 'exclusive'} "
                    f"lock on {lock_path} within {LOCK_TIMEOUT}s"
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


def _read_json_nolock(file_path: str) -> dict:
    """Read a JSON dict from *file_path*.  Caller must hold the appropriate lock.

    Returns an empty dict on file-not-found or non-dict content.
    Preserves corrupt JSON to a *.corrupt.TIMESTAMP file before raising.
    """
    _ensure_data_dir()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("%s is not a dict, resetting.", file_path)
            return {}
        return data
    except json.JSONDecodeError:
        corrupt_path = file_path + ".corrupt." + str(int(time.time()))
        try:
            with open(file_path, "rb") as src:
                raw_bytes = src.read()
            with open(corrupt_path, "wb") as dst:
                dst.write(raw_bytes)
            logger.error(
                "Corrupt JSON in %s (%d bytes) — preserved at %s",
                file_path, len(raw_bytes), corrupt_path,
            )
        except Exception as copy_err:
            logger.error(
                "Failed to save corrupt-file copy from %s: %s",
                file_path, copy_err,
            )
        raise
    except FileNotFoundError:
        logger.warning("%s not found (first run?), starting fresh.", file_path)
        return {}


def _write_json_nolock(file_path: str, data: dict) -> None:
    """Atomically replace *file_path* with *data*.  Caller must hold exclusive lock."""
    _ensure_data_dir()
    tmp_path = file_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


# ── Serialisation helpers ──────────────────────────────────────────────────


def _node_to_dict(node: ReasoningNode) -> dict:
    """递归将 ReasoningNode 转为 dict。"""
    return {
        "id": node.id,
        "label": node.label,
        "description": node.description,
        "emoji": node.emoji,
        "depth": node.depth,
        "parent_id": node.parent_id,
        "children": [_node_to_dict(c) for c in node.children],
        "expanded": node.expanded,
        "expandable": node.expandable,
        "is_actionable": node.is_actionable,
        "action_text": node.action_text,
        "time_slot": node.time_slot,
        "frequency": node.frequency,
        "duration": node.duration,
        "reasoning": node.reasoning,
        "status": node.status,
        "completed_count": node.completed_count,
        "last_completed": node.last_completed,
        "created_at": node.created_at,
        "followup_question": node.followup_question,
        "followup_options": node.followup_options,
    }


def _normalize_followup_options(raw) -> list[str] | None:
    """Coerce a serialized ``followup_options`` value into a clean ``list[str] | None``.

    Guards against legacy/corrupt data: the field is documented as ``list[str]``,
    but a malformed JSON file (or an older writer) could store a bare string, a
    list containing structured junk (dicts/lists), or empty entries. The
    ``ReasoningNode`` dataclass type hint is not enforced at runtime, so a raw
    passthrough could leak non-list values into serialization / API responses
    and trip up front-end rendering (e.g. JS ``Array.isArray`` checks).

    Normalizing here keeps well-formed data byte-for-byte identical while
    making bad data degrade gracefully to ``None`` / a clean string list.
    """
    if not isinstance(raw, list):
        return None
    cleaned: list[str] = []
    for item in raw:
        if item is None or isinstance(item, (dict, list, tuple, set)):
            continue  # structured junk (e.g. {"label": "..."}) or null → drop
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned or None


def _dict_to_node(d: dict) -> ReasoningNode:
    """递归将 dict 转为 ReasoningNode。"""
    raw_children = d.get("children", [])
    if not isinstance(raw_children, list):
        # Legacy/corrupt data may store children as null or a dict — degrade
        # to an empty list instead of crashing deserialization (a TypeError /
        # iterating dict keys as node dicts would otherwise leak out of
        # _dict_to_plan / get_plan). Well-formed data is unaffected.
        logger.warning(
            "_dict_to_node: children is %s, resetting to []",
            type(raw_children).__name__,
        )
        raw_children = []
    children = [_dict_to_node(c) for c in raw_children if isinstance(c, dict)]
    return ReasoningNode(
        id=d.get("id", str(uuid.uuid4())),
        label=d.get("label", ""),
        description=d.get("description", ""),
        emoji=d.get("emoji", ""),
        depth=d.get("depth", 0),
        parent_id=d.get("parent_id"),
        children=children,
        expanded=d.get("expanded", False),
        expandable=d.get("expandable", True),
        is_actionable=d.get("is_actionable", False),
        action_text=d.get("action_text"),
        time_slot=d.get("time_slot"),
        frequency=d.get("frequency"),
        duration=d.get("duration"),
        reasoning=d.get("reasoning"),
        status=d.get("status", "pending"),
        completed_count=d.get("completed_count", 0),
        last_completed=d.get("last_completed"),
        created_at=d.get("created_at", ""),
        followup_question=d.get("followup_question"),
        followup_options=_normalize_followup_options(d.get("followup_options")),
    )


def _plan_to_dict(plan: HealthPlan) -> dict:
    """Convert a HealthPlan dataclass to a JSON-serialisable dict."""
    result = {
        "id": plan.id,
        "user_id": plan.user_id,
        "concern": plan.concern,
        "title": plan.title,
        "goal": plan.goal,
        "status": plan.status,
        "created_at": plan.created_at,
        "updated_at": plan.updated_at,
        "context_snapshot": plan.context_snapshot,
        "version": plan.version,
    }
    if plan.reasoning_tree is not None:
        result["reasoning_tree"] = _node_to_dict(plan.reasoning_tree)
    else:
        result["reasoning_tree"] = None
    return result


def _dict_to_plan(d: dict) -> HealthPlan:
    """Convert a raw dict back to a HealthPlan.

    Handles legacy data: if categories exist but reasoning_tree is empty,
    attempts to convert the old format into the new tree structure.
    """
    raw_tree = d.get("reasoning_tree")
    reasoning_tree = None
    if isinstance(raw_tree, dict):
        reasoning_tree = _dict_to_node(raw_tree)

    # Legacy compatibility: if no reasoning_tree but categories exist, convert.
    # Guard categories to a list — a truthy-but-corrupt categories (dict, or a
    # list containing non-dicts) used to crash _convert_legacy_categories_to_tree
    # during get_plan/list_plans; degrade to no tree (None) instead.
    raw_categories = d.get("categories")
    if reasoning_tree is None and isinstance(raw_categories, list):
        reasoning_tree = _convert_legacy_categories_to_tree(raw_categories)

    raw_ctx = d.get("context_snapshot", {})
    if raw_ctx is not None and not isinstance(raw_ctx, dict):
        # Legacy/corrupt data may store context_snapshot as a non-dict (string /
        # list / number); later callers (e.g. the regenerate flow) call .get()
        # on it and would otherwise 500. Degrade to the documented {} default.
        logger.warning(
            "_dict_to_plan: context_snapshot is %s, resetting to {}",
            type(raw_ctx).__name__,
        )
        raw_ctx = {}

    return HealthPlan(
        id=d["id"],
        user_id=d.get("user_id", ""),
        concern=d.get("concern", ""),
        title=d.get("title", ""),
        goal=d.get("goal", ""),
        status=d.get("status", "active"),
        created_at=d.get("created_at", ""),
        updated_at=d.get("updated_at", ""),
        context_snapshot=raw_ctx,
        reasoning_tree=reasoning_tree,
        version=d.get("version", 1),
    )


def _convert_legacy_categories_to_tree(categories: list[dict]) -> ReasoningNode:
    """转换遗留格式的 PlanCategory/PlanItem 数据为 ReasoningNode 树结构。

    与 children 反序列化加固同理：遗留/损坏 categories（dict、含非 dict 条目、
    items 非 list 或含非 dict 项）旧实现会在 get_plan/list_plans 时对非 dict 调
    .get() 而抛 AttributeError（500）。现跳过损坏条目、损坏 items 视为空，
    非 list categories 返回空根树；良构数据行为完全不变。
    """
    root = ReasoningNode(
        id="root",
        label="健康计划",
        description="健康计划",
        emoji="🧑‍🦲",
        depth=0,
        expandable=True,
        expanded=True,
        created_at="",
    )
    if not isinstance(categories, list):
        logger.warning(
            "_convert_legacy_categories_to_tree: categories is %s, returning empty root",
            type(categories).__name__,
        )
        return root
    for cat in categories:
        if not isinstance(cat, dict):
            continue  # corrupt entry → skip instead of crashing on cat.get()
        cat_name = cat.get("name", "未分类")
        cat_icon = cat.get("icon", "📋")
        cat_node = ReasoningNode(
            id=str(uuid.uuid4()),
            label=cat_name,
            description=cat_name,
            emoji=cat_icon,
            depth=1,
            parent_id=root.id,
            expandable=True,
            expanded=True,
            created_at="",
        )
        raw_items = cat.get("items", [])
        if not isinstance(raw_items, list):
            raw_items = []  # corrupt items → treat as no items
        for item in raw_items:
            if not isinstance(item, dict):
                continue  # corrupt item → skip
            leaf = ReasoningNode(
                id=item.get("item_id", str(uuid.uuid4())),
                label=item.get("action", ""),
                description=item.get("reasoning", ""),
                emoji="✅",
                depth=2,
                parent_id=cat_node.id,
                expandable=False,
                expanded=False,
                is_actionable=True,
                action_text=item.get("action", ""),
                time_slot=item.get("time_slot"),
                frequency=item.get("frequency"),
                duration=item.get("duration"),
                reasoning=item.get("reasoning"),
                status=item.get("status", "pending"),
                completed_count=item.get("completed_count", 0),
                last_completed=item.get("last_completed"),
                created_at="",
            )
            cat_node.children.append(leaf)
        root.children.append(cat_node)
    return root


def _is_reminder_dict(entry) -> bool:
    """Return True if *entry* is a well-formed reminder dict.

    Guards plan_reminders.json against legacy/corrupt entries — bare strings /
    numbers / nulls, or dicts missing a usable "id" — that can leak into a
    hand-edited or partially-written file.  Callers skip junk instead of
    crashing on .get()/indexing (same strategy as _find_node_in_dict_tree and
    feedback._is_feedback_dict).  All reminders this module writes carry a
    non-empty uuid4 string id, so well-formed data always passes.
    """
    return isinstance(entry, dict) and isinstance(entry.get("id"), str) and bool(entry.get("id"))


def _reminders_container(data: dict, user_id: str) -> list:
    """Read the reminder list for *user_id* from parsed plan_reminders.json.

    The file maps user_id → list[dict].  If a legacy/hand-edited file stores a
    non-list under the key (dict / str / int / null), iterating or appending to
    it would raise AttributeError/TypeError — treat a corrupt container as an
    empty list instead.  A well-formed list is returned untouched (same object),
    so in-place append in create_reminder still persists through data[user_id].
    """
    value = data.get(user_id, [])
    return value if isinstance(value, list) else []


def _reminder_to_dict(reminder: PlanReminder) -> dict:
    """Convert a PlanReminder dataclass to a JSON-serialisable dict."""
    return asdict(reminder)


def _dict_to_reminder(d: dict) -> PlanReminder:
    """Convert a raw dict back to a PlanReminder."""
    return PlanReminder(
        id=d["id"],
        plan_id=d.get("plan_id", ""),
        user_id=d.get("user_id", ""),
        trigger_type=d.get("trigger_type", "time"),
        time_spec=d.get("time_spec", ""),
        days_of_week=d.get("days_of_week"),
        message_template=d.get("message_template", ""),
        is_active=d.get("is_active", True),
        last_triggered=d.get("last_triggered"),
        adaptive=d.get("adaptive", True),
        created_at=d.get("created_at", ""),
    )


def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ── Tree traversal helpers ────────────────────────────────────────────────


def find_node_in_tree(node_id: str, root: ReasoningNode) -> ReasoningNode | None:
    """递归在推理树中查找指定 ID 的节点。"""
    if root.id == node_id:
        return root
    for child in root.children:
        result = find_node_in_tree(node_id, child)
        if result is not None:
            return result
    return None


def build_path_to_node(node_id: str, root: ReasoningNode) -> list[dict]:
    """从根节点到指定节点构建推理路径（节点 label + description 列表）。
    返回从根到目标节点的有序列表。"""
    path = []

    def _dfs(current, target_id, current_path):
        current_path.append({"label": current.label, "description": current.description})
        if current.id == target_id:
            return True
        for child in current.children:
            if _dfs(child, target_id, current_path):
                return True
        current_path.pop()
        return False

    _dfs(root, node_id, path)
    return path


def _find_node_in_dict_tree(node_id: str, node_dict: dict) -> dict | None:
    """递归在 dict 格式的推理树中查找指定 ID 的节点（用于直接操作序列化数据）。

    与 _dict_to_node 的 children 清洗保持一致：update_leaf_status / expand_node /
    set_followup_question 直接操作从文件读出的原始 dict 树（不经 _dict_to_node
    归一化），若遗留/损坏 JSON 含 null/dict 形态的 children、非 dict 条目或非
    dict 的整棵子树，旧实现会对非 dict 调 .get()/迭代 dict 键而抛
    AttributeError/TypeError（500）。现对非 dict 子树返回 None、对非 dict 子条目
    跳过、对非 list children 不再下探，使损坏数据优雅地"未找到"。良构数据行为
    完全不变。
    """
    if not isinstance(node_dict, dict):
        return None
    if node_dict.get("id") == node_id:
        return node_dict
    raw_children = node_dict.get("children", [])
    if not isinstance(raw_children, list):
        return None
    for child in raw_children:
        if not isinstance(child, dict):
            continue
        result = _find_node_in_dict_tree(node_id, child)
        if result is not None:
            return result
    return None


# ── Plan CRUD ──────────────────────────────────────────────────────────────


def create_plan(
    user_id: str,
    concern: str,
    title: str,
    goal: str,
    reasoning_tree: ReasoningNode | None = None,
    context_snapshot: dict = None,
) -> HealthPlan:
    """Create and persist a new health plan.

    *reasoning_tree* is an optional ReasoningNode root node. If not provided,
    a default root node is created automatically using the concern text.

    Returns the newly created HealthPlan.
    """
    if reasoning_tree is None:
        reasoning_tree = ReasoningNode(
            id="root",
            label=concern,
            description=concern,
            emoji="🧑‍🦲",
            depth=0,
            expandable=True,
        )

    plan = HealthPlan(
        id=str(uuid.uuid4()),
        user_id=user_id,
        concern=concern,
        title=title,
        goal=goal,
        status="active",
        created_at=_now_iso(),
        updated_at=_now_iso(),
        context_snapshot=context_snapshot or {},
        reasoning_tree=reasoning_tree,
        version=1,
    )

    fd = _acquire_lock(HEALTH_PLANS_LOCK_FILE, shared=False)
    try:
        data = _read_json_nolock(HEALTH_PLANS_FILE)
        user_plans = data.setdefault(user_id, [])
        user_plans.append(_plan_to_dict(plan))
        _write_json_nolock(HEALTH_PLANS_FILE, data)
    finally:
        _release_lock(fd)

    logger.info("Created health plan %s for user %s (version %d)", plan.id, user_id, plan.version)
    return plan


def get_active_plan(user_id: str) -> HealthPlan | None:
    """Get the current active plan for a user.  Returns None if no active plan."""
    fd = _acquire_lock(HEALTH_PLANS_LOCK_FILE, shared=True)
    try:
        data = _read_json_nolock(HEALTH_PLANS_FILE)
        user_plans = data.get(user_id, [])
    finally:
        _release_lock(fd)

    for plan_dict in user_plans:
        if plan_dict.get("status") == "active":
            return _dict_to_plan(plan_dict)
    return None


def get_plan(plan_id: str, user_id: str) -> HealthPlan | None:
    """Get a specific plan by ID and user_id."""
    fd = _acquire_lock(HEALTH_PLANS_LOCK_FILE, shared=True)
    try:
        data = _read_json_nolock(HEALTH_PLANS_FILE)
        user_plans = data.get(user_id, [])
    finally:
        _release_lock(fd)

    for plan_dict in user_plans:
        if plan_dict.get("id") == plan_id:
            return _dict_to_plan(plan_dict)
    return None


def list_plans(user_id: str, status: str = None) -> list[HealthPlan]:
    """List plans for a user, optionally filtered by status.

    Results are sorted by created_at descending (newest first).
    """
    fd = _acquire_lock(HEALTH_PLANS_LOCK_FILE, shared=True)
    try:
        data = _read_json_nolock(HEALTH_PLANS_FILE)
        user_plans = data.get(user_id, [])
    finally:
        _release_lock(fd)

    if status:
        user_plans = [p for p in user_plans if p.get("status") == status]

    user_plans.sort(key=lambda p: p.get("created_at", ""), reverse=True)
    return [_dict_to_plan(p) for p in user_plans]


def update_plan_status(plan_id: str, user_id: str, status: str) -> HealthPlan | None:
    """Update plan status (active/archived/completed).  Returns updated plan or None."""
    fd = _acquire_lock(HEALTH_PLANS_LOCK_FILE, shared=False)
    try:
        data = _read_json_nolock(HEALTH_PLANS_FILE)
        user_plans = data.get(user_id, [])
        found = None
        for plan_dict in user_plans:
            if plan_dict.get("id") == plan_id:
                plan_dict["status"] = status
                plan_dict["updated_at"] = _now_iso()
                found = plan_dict
                break
        if found is None:
            return None
        _write_json_nolock(HEALTH_PLANS_FILE, data)
    finally:
        _release_lock(fd)

    logger.info("Updated plan %s status to %s for user %s", plan_id, status, user_id)
    return _dict_to_plan(found)


def rename_plan(plan_id: str, user_id: str, new_title: str) -> HealthPlan | None:
    """Rename a plan. Returns updated plan or None if not found."""
    fd = _acquire_lock(HEALTH_PLANS_LOCK_FILE, shared=False)
    try:
        data = _read_json_nolock(HEALTH_PLANS_FILE)
        user_plans = data.get(user_id, [])
        found = None
        for plan_dict in user_plans:
            if plan_dict.get("id") == plan_id:
                plan_dict["title"] = new_title
                plan_dict["updated_at"] = datetime.now(timezone.utc).isoformat()
                found = plan_dict
                break
        if not found:
            return None
        data[user_id] = user_plans
        _write_json_nolock(HEALTH_PLANS_FILE, data)
    finally:
        _release_lock(fd)

    logger.info("Renamed plan %s to '%s' for user %s", plan_id, new_title, user_id)
    return _dict_to_plan(found)


def delete_plan(plan_id: str, user_id: str) -> bool:
    """Delete a plan and its reminders.  Returns True if the plan was found and deleted."""
    # Remove the plan
    fd_plan = _acquire_lock(HEALTH_PLANS_LOCK_FILE, shared=False)
    removed_plan = False
    try:
        data = _read_json_nolock(HEALTH_PLANS_FILE)
        user_plans = data.get(user_id, [])
        new_plans = [p for p in user_plans if p.get("id") != plan_id]
        if len(new_plans) == len(user_plans):
            return False
        removed_plan = True
        data[user_id] = new_plans
        _write_json_nolock(HEALTH_PLANS_FILE, data)
    finally:
        _release_lock(fd_plan)

    # Remove associated reminders
    if removed_plan:
        fd_rem = _acquire_lock(PLAN_REMINDERS_LOCK_FILE, shared=False)
        try:
            rem_data = _read_json_nolock(PLAN_REMINDERS_FILE)
            user_reminders = _reminders_container(rem_data, user_id)
            rem_data[user_id] = [r for r in user_reminders
                                 if not (_is_reminder_dict(r) and r.get("plan_id") == plan_id)]
            _write_json_nolock(PLAN_REMINDERS_FILE, rem_data)
        finally:
            _release_lock(fd_rem)

    logger.info("Deleted plan %s for user %s", plan_id, user_id)
    return True


def update_leaf_status(plan_id: str, user_id: str, leaf_id: str, status: str) -> HealthPlan | None:
    """
    更新推理树中某个叶子节点的 status。
    如果 status="done"，递增 completed_count，更新 last_completed；
    若从 "done" 改回其它状态（撤销完成），同步递减 completed_count 并清空 last_completed，
    避免“叶子已非 done 但计数/时间仍停留”的不一致。
    返回更新后的 HealthPlan。
    """
    fd = _acquire_lock(HEALTH_PLANS_LOCK_FILE, shared=False)
    try:
        data = _read_json_nolock(HEALTH_PLANS_FILE)
        user_plans = data.get(user_id, [])
        found = None
        node_found = False
        for plan_dict in user_plans:
            if plan_dict.get("id") != plan_id:
                continue
            found = plan_dict
            tree = plan_dict.get("reasoning_tree")
            if tree is None:
                break
            node = _find_node_in_dict_tree(leaf_id, tree)
            if node is not None:
                old_status = node.get("status")
                node["status"] = status
                if status == "done":
                    node["completed_count"] = node.get("completed_count", 0) + 1
                    node["last_completed"] = _now_iso()
                elif old_status == "done":
                    # 撤销完成：叶子不再处于 done 状态，计数与时间戳同步回退（不重复扣减）。
                    node["completed_count"] = max(0, node.get("completed_count", 0) - 1)
                    node["last_completed"] = None
                plan_dict["updated_at"] = _now_iso()
                node_found = True
            break

        # 未找到计划或目标节点不存在时返回 None，使调用方（adopt / update-item API）
        # 能区分"未找到"与"成功更新"，避免静默无操作。
        if found is None or not node_found:
            return None
        _write_json_nolock(HEALTH_PLANS_FILE, data)
    finally:
        _release_lock(fd)

    logger.info("Updated leaf %s status to %s in plan %s", leaf_id, status, plan_id)
    return _dict_to_plan(found)


def expand_node(
    plan_id: str, user_id: str, node_id: str, children_data: list[dict]
) -> HealthPlan | None:
    """
    向推理树的指定节点添加子节点。
    children_data 是由 ai_engine.expand_node() 返回的子节点字典列表。
    每个 dict 中包含: label, description, emoji, is_actionable, expandable,
                    action_text, time_slot, frequency, duration, reasoning
                    （后5个仅 actionable 时有）
    会为每个子节点生成 id，设置 depth 和 parent_id。
    设置父节点的 expanded=True。
    保存后返回更新后的 HealthPlan。

    幂等性：若目标节点已 expanded=True，则不做任何修改（防止重复展开产生重复子节点）。
    该守卫与 /api/plan/<id>/reasoning/expand 的行为保持一致。
    """
    fd = _acquire_lock(HEALTH_PLANS_LOCK_FILE, shared=False)
    try:
        data = _read_json_nolock(HEALTH_PLANS_FILE)
        user_plans = data.get(user_id, [])
        found = None
        did_expand = False
        for plan_dict in user_plans:
            if plan_dict.get("id") != plan_id:
                continue
            found = plan_dict
            tree = plan_dict.get("reasoning_tree")
            if tree is None:
                break
            parent_node = _find_node_in_dict_tree(node_id, tree)
            if parent_node is None:
                break

            # 幂等守卫：已展开的节点不重复展开
            if parent_node.get("expanded"):
                break

            parent_depth = parent_node.get("depth", 0)
            new_children = []
            for cd in children_data:
                child = {
                    "id": str(uuid.uuid4()),
                    "label": cd.get("label", ""),
                    "description": cd.get("description", ""),
                    "emoji": cd.get("emoji", ""),
                    "depth": parent_depth + 1,
                    "parent_id": node_id,
                    "children": [],
                    "expanded": False,
                    "expandable": cd.get("expandable", True),
                    "is_actionable": cd.get("is_actionable", False),
                    "action_text": cd.get("action_text"),
                    "time_slot": cd.get("time_slot"),
                    "frequency": cd.get("frequency"),
                    "duration": cd.get("duration"),
                    "reasoning": cd.get("reasoning"),
                    "status": "pending",
                    "completed_count": 0,
                    "last_completed": None,
                    "created_at": _now_iso(),
                    "followup_question": cd.get("followup_question"),
                    "followup_options": _normalize_followup_options(cd.get("followup_options")),
                }
                new_children.append(child)

            # No-op guard: an empty expansion (transient LLM failure returning
            # no children) must not mark the node expanded — otherwise the node
            # is permanently locked out of future expansion. Treat it as no-op
            # and let the caller retry.
            if not new_children:
                break

            parent_node.setdefault("children", []).extend(new_children)
            parent_node["expanded"] = True
            # The node's pending followup question is resolved once it expands.
            parent_node["followup_question"] = None
            parent_node["followup_options"] = None
            plan_dict["updated_at"] = _now_iso()
            did_expand = True
            break

        if found is None:
            return None
        if did_expand:
            _write_json_nolock(HEALTH_PLANS_FILE, data)
    finally:
        _release_lock(fd)

    logger.info(
        "Expanded node %s in plan %s with %d children",
        node_id, plan_id, len(children_data) if did_expand else 0,
    )
    return _dict_to_plan(found)


def set_followup_question(
    plan_id: str,
    user_id: str,
    node_id: str,
    question: str | None,
    options: list[str] | None = None,
) -> HealthPlan | None:
    """Set (or clear, when question is None) a pending followup question on a node.

    Interactive info-gathering: when the expand flow determines more info is
    needed, the question is persisted on the node (optionally with quick-choice
    options). Options are normalized via _normalize_followup_options so malformed
    input (structured junk / non-list) degrades to None instead of leaking into
    serialization / API responses, consistent with _dict_to_node and expand_node.
    Once the user answers and the node expands, the question is cleared (see
    expand_node). Returns the updated plan, or None if the plan/node is not found.
    """
    fd = _acquire_lock(HEALTH_PLANS_LOCK_FILE, shared=False)
    try:
        data = _read_json_nolock(HEALTH_PLANS_FILE)
        user_plans = data.get(user_id, [])
        found = None
        node_found = False
        for plan_dict in user_plans:
            if plan_dict.get("id") != plan_id:
                continue
            found = plan_dict
            tree = plan_dict.get("reasoning_tree")
            if tree is None:
                break
            node = _find_node_in_dict_tree(node_id, tree)
            if node is None:
                break
            node["followup_question"] = question
            node["followup_options"] = _normalize_followup_options(options)
            plan_dict["updated_at"] = _now_iso()
            node_found = True
            break
        if found is None or not node_found:
            return None
        _write_json_nolock(HEALTH_PLANS_FILE, data)
    finally:
        _release_lock(fd)
    return _dict_to_plan(found)


def collect_actionable_leaves(plan: HealthPlan) -> list[ReasoningNode]:
    """
    递归遍历 reasoning_tree，收集所有 is_actionable=True 的叶子节点。
    按 time_slot 排序（早晨→上午→中午→下午→傍晚→睡前→深夜）。
    返回 ReasoningNode 列表。
    """
    if plan.reasoning_tree is None:
        return []

    leaves: list[ReasoningNode] = []

    def _collect(node: ReasoningNode) -> None:
        if node.is_actionable:
            leaves.append(node)
        for child in node.children:
            _collect(child)

    _collect(plan.reasoning_tree)
    leaves.sort(key=lambda n: TIME_SLOT_ORDER.get(n.time_slot or "", 99))
    return leaves


def adopt_leaf(plan_id: str, user_id: str, leaf_id: str) -> HealthPlan | None:
    """
    用户"采纳"一条建议时的快捷操作（等价于设置 status="pending"）。
    返回更新后的 HealthPlan。
    """
    return update_leaf_status(plan_id, user_id, leaf_id, "pending")


def list_new_actionable_leaves(plan: HealthPlan, known_ids: set[str]) -> list[dict]:
    """
    返回 *plan* 中 id 不在 *known_ids* 集合内的可采纳叶子（序列化为精简 dict）。

    用于 expand API：展开（含追问答完后）后识别"本次新增的可采纳叶子"，
    供前端提供一键全部采纳（追问后自动采纳体验）。精简字段仅包含前端
    横幅展示 + 逐个采纳所需的最小信息。
    """
    if plan is None or plan.reasoning_tree is None:
        return []
    return [
        {
            "id": n.id,
            "label": n.label,
            "emoji": n.emoji,
            "time_slot": n.time_slot,
            "frequency": n.frequency,
            "duration": n.duration,
        }
        for n in collect_actionable_leaves(plan)
        if n.id not in known_ids
    ]


def regenerate_plan(
    plan_id: str,
    user_id: str,
    reasoning_tree: ReasoningNode,
    new_context: dict = None,
) -> HealthPlan:
    """Create a new version of an existing plan.

    Archives the old plan (marks as "archived"), then creates a new plan with
    version = old_version + 1.  The new plan carries the same concern and
    title as the original.

    *reasoning_tree* is the new ReasoningNode root node for the regenerated plan.

    Returns the newly created HealthPlan.
    """
    fd = _acquire_lock(HEALTH_PLANS_LOCK_FILE, shared=False)
    try:
        data = _read_json_nolock(HEALTH_PLANS_FILE)
        user_plans = data.get(user_id, [])

        # Find old plan to derive title/concern/version
        old_dict = None
        for plan_dict in user_plans:
            if plan_dict.get("id") == plan_id:
                old_dict = plan_dict
                plan_dict["status"] = "archived"
                plan_dict["updated_at"] = _now_iso()
                break

        if old_dict is None:
            logger.warning("Plan %s not found for regeneration", plan_id)
            raise ValueError(f"Plan {plan_id} not found")

        new_version = old_dict.get("version", 1) + 1
        new_plan = HealthPlan(
            id=str(uuid.uuid4()),
            user_id=user_id,
            concern=old_dict.get("concern", ""),
            title=old_dict.get("title", ""),
            goal=old_dict.get("goal", ""),
            status="active",
            created_at=_now_iso(),
            updated_at=_now_iso(),
            context_snapshot=new_context or old_dict.get("context_snapshot", {}),
            reasoning_tree=reasoning_tree,
            version=new_version,
        )

        user_plans.append(_plan_to_dict(new_plan))
        _write_json_nolock(HEALTH_PLANS_FILE, data)
    finally:
        _release_lock(fd)

    logger.info(
        "Regenerated plan %s -> %s for user %s (v%d -> v%d)",
        plan_id, new_plan.id, user_id, new_version - 1, new_version,
    )
    return new_plan


# ── Reminder CRUD (unchanged) ──────────────────────────────────────────────


def create_reminder(
    user_id: str,
    plan_id: str,
    time_spec: str,
    message_template: str,
    days_of_week: list[int] = None,
    adaptive: bool = True,
) -> PlanReminder:
    """Create a new reminder."""
    reminder = PlanReminder(
        id=str(uuid.uuid4()),
        plan_id=plan_id,
        user_id=user_id,
        trigger_type="time",
        time_spec=time_spec,
        days_of_week=days_of_week,
        message_template=message_template,
        is_active=True,
        last_triggered=None,
        adaptive=adaptive,
        created_at=_now_iso(),
    )

    fd = _acquire_lock(PLAN_REMINDERS_LOCK_FILE, shared=False)
    try:
        data = _read_json_nolock(PLAN_REMINDERS_FILE)
        # Guard against a corrupt per-user container (e.g. a dict/str where a
        # list is expected): setdefault would return the corrupt value and
        # .append() would crash.  Degrade to a fresh list instead.
        user_reminders = _reminders_container(data, user_id)
        user_reminders.append(_reminder_to_dict(reminder))
        data[user_id] = user_reminders
        _write_json_nolock(PLAN_REMINDERS_FILE, data)
    finally:
        _release_lock(fd)

    logger.info("Created reminder %s for plan %s (user %s)", reminder.id, plan_id, user_id)
    return reminder


def list_reminders(user_id: str, active_only: bool = True) -> list[PlanReminder]:
    """List reminders for a user.

    If *active_only* is True (default), only active reminders are returned.
    """
    fd = _acquire_lock(PLAN_REMINDERS_LOCK_FILE, shared=True)
    try:
        data = _read_json_nolock(PLAN_REMINDERS_FILE)
        user_reminders = _reminders_container(data, user_id)
    finally:
        _release_lock(fd)

    # Skip corrupt entries (bare strings / numbers / nulls / dicts without an
    # id) instead of crashing in _dict_to_reminder; well-formed data unchanged.
    reminders = [_dict_to_reminder(r) for r in user_reminders if _is_reminder_dict(r)]
    if active_only:
        reminders = [r for r in reminders if r.is_active]
    return reminders


def delete_reminder(reminder_id: str, user_id: str) -> bool:
    """Delete a reminder.  Returns True if the reminder was found and deleted."""
    fd = _acquire_lock(PLAN_REMINDERS_LOCK_FILE, shared=False)
    try:
        data = _read_json_nolock(PLAN_REMINDERS_FILE)
        user_reminders = _reminders_container(data, user_id)
        new_reminders = [r for r in user_reminders
                         if not (_is_reminder_dict(r) and r.get("id") == reminder_id)]
        if len(new_reminders) == len(user_reminders):
            return False
        data[user_id] = new_reminders
        _write_json_nolock(PLAN_REMINDERS_FILE, data)
        return True
    finally:
        _release_lock(fd)
