"""Viora — Health Plan Module Tests

Tests for health_plan.py: data models, CRUD, tree operations, serialization,
reminder management, and legacy format conversion.
"""

import sys
import json
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from health_plan import (
    ReasoningNode,
    HealthPlan,
    PlanReminder,
    create_plan,
    get_active_plan,
    get_plan,
    list_plans,
    update_plan_status,
    rename_plan,
    delete_plan,
    expand_node,
    update_leaf_status,
    collect_actionable_leaves,
    adopt_leaf,
    regenerate_plan,
    find_node_in_tree,
    build_path_to_node,
    create_reminder,
    list_reminders,
    delete_reminder,
    _node_to_dict,
    _dict_to_node,
    _plan_to_dict,
    _dict_to_plan,
    _convert_legacy_categories_to_tree,
    HEALTH_PLANS_FILE,
    PLAN_REMINDERS_FILE,
    HEALTH_PLANS_LOCK_FILE,
    PLAN_REMINDERS_LOCK_FILE,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_leaf(**overrides) -> ReasoningNode:
    """Create an actionable leaf node with sensible defaults."""
    kwargs = dict(
        id=str(uuid.uuid4()),
        label="全麦面包替代白面包",
        description="低GI替代方案",
        emoji="🍞",
        depth=2,
        parent_id="parent",
        is_actionable=True,
        action_text="早餐用全麦面包替代白面包",
        time_slot="早餐",
        frequency="每天",
        reasoning="低GI减少胰岛素波动",
    )
    kwargs.update(overrides)
    return ReasoningNode(**kwargs)


def _seed_plan_tree(tmp_path, tree_dict):
    """Create a plan, then overwrite its stored reasoning_tree with a raw dict.

    Used to simulate legacy/corrupt persisted JSON (children as dicts, non-dict
    entries, etc.) that the write-path functions (update_leaf_status / expand_node
    / set_followup_question) read back directly from the file.
    """
    plan = create_plan("u1", "脱发", "P1", "改善")
    fp = Path(tmp_path) / "health_plans.json"
    data = json.loads(fp.read_text(encoding="utf-8"))
    for pd in data["u1"]:
        if pd["id"] == plan.id:
            pd["reasoning_tree"] = tree_dict
    fp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return plan.id


def _seed_reminders_file(tmp_path, data):
    """Overwrite plan_reminders.json with a raw payload.

    Used to simulate legacy/corrupt persisted reminder data (non-list containers
    under a user, junk entries inside a list) for the reminder CRUD functions.
    """
    fp = Path(tmp_path) / "plan_reminders.json"
    fp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _corrupt_sibling_tree():
    """Root whose children mix a valid leaf with corrupt entries."""
    return {
        "id": "root", "label": "r", "description": "", "emoji": "🧑",
        "depth": 0, "expanded": True, "expandable": False,
        "children": [
            {"id": "leaf1", "label": "早餐全麦", "description": "低GI",
             "emoji": "🍞", "depth": 1, "is_actionable": True,
             "status": "pending", "completed_count": 0,
             "last_completed": None, "children": []},
            # corrupt: "children" is a dict (not a list) — a node hides inside.
            {"id": "bad1", "label": "损坏", "description": "",
             "emoji": "💥", "depth": 1, "is_actionable": False,
             "status": "pending",
             "children": {"inner": {"id": "inner1", "label": "x", "children": []}}},
            "pure-junk-string",
        ],
    }


def _make_mid_node(**overrides) -> ReasoningNode:
    """Create a non-actionable intermediate node."""
    kwargs = dict(
        id=str(uuid.uuid4()),
        label="饮食因素",
        description="饮食相关的健康影响因素",
        emoji="🥗",
        depth=1,
        parent_id="root",
        is_actionable=False,
    )
    kwargs.update(overrides)
    return ReasoningNode(**kwargs)


# ── Fixtures: patch file paths to temp dirs ────────────────────────────────


@pytest.fixture
def hp_patch(tmp_path, monkeypatch):
    """Patch health_plan module-level file/ lock paths to use tmp_path."""
    import health_plan as hp

    monkeypatch.setattr(hp, "HEALTH_PLANS_FILE", str(tmp_path / "health_plans.json"))
    monkeypatch.setattr(hp, "HEALTH_PLANS_LOCK_FILE", str(tmp_path / "health_plans.json.lock"))
    monkeypatch.setattr(hp, "PLAN_REMINDERS_FILE", str(tmp_path / "plan_reminders.json"))
    monkeypatch.setattr(hp, "PLAN_REMINDERS_LOCK_FILE", str(tmp_path / "plan_reminders.json.lock"))
    monkeypatch.setattr(hp, "DATA_DIR", str(tmp_path))
    yield tmp_path


# ── Test Data Models ────────────────────────────────────────────────────────


class TestReasoningNode:
    def test_minimal_node(self):
        """Node can be created with only required fields."""
        node = ReasoningNode(id="n1", label="测试", description="desc", emoji="🧪", depth=0)
        assert node.id == "n1"
        assert node.label == "测试"
        assert node.depth == 0
        assert node.children == []
        assert not node.is_actionable
        assert node.status == "pending"

    def test_actionable_node(self):
        """Actionable node has execution metadata."""
        node = _make_leaf()
        assert node.is_actionable
        assert node.action_text == "早餐用全麦面包替代白面包"
        assert node.time_slot == "早餐"
        assert node.frequency == "每天"
        assert node.reasoning is not None

    def test_node_parent_child(self):
        """Node can hold children."""
        parent = _make_mid_node()
        child = _make_leaf(parent_id=parent.id)
        parent.children.append(child)
        assert len(parent.children) == 1
        assert parent.children[0].id == child.id
        assert child.parent_id == parent.id


class TestHealthPlan:
    def test_minimal_plan(self):
        """Plan can be created with required fields."""
        plan = HealthPlan(
            id="plan1",
            user_id="user_001",
            concern="脱发严重",
            title="头发养护计划",
            goal="改善脱发",
        )
        assert plan.id == "plan1"
        assert plan.status == "active"
        assert plan.version == 1
        assert plan.reasoning_tree is None

    def test_plan_with_tree(self):
        """Plan can hold a reasoning tree."""
        root = ReasoningNode(id="root", label="脱发", description="主诉", emoji="🧑‍🦲", depth=0)
        plan = HealthPlan(
            id="plan2",
            user_id="user_001",
            concern="脱发严重",
            title="头发养护计划",
            goal="改善脱发",
            reasoning_tree=root,
        )
        assert plan.reasoning_tree is not None
        assert plan.reasoning_tree.label == "脱发"


class TestPlanReminder:
    def test_minimal_reminder(self):
        """Reminder can be created with required fields."""
        rem = PlanReminder(id="rem1", plan_id="plan1", user_id="user_001")
        assert rem.trigger_type == "time"
        assert rem.is_active

    def test_reminder_with_time(self):
        """Reminder can have time specification."""
        rem = PlanReminder(
            id="rem1", plan_id="plan1", user_id="user_001",
            time_spec="15:00", days_of_week=[1, 3, 5],
        )
        assert rem.time_spec == "15:00"
        assert rem.days_of_week == [1, 3, 5]


# ── Test Serialization ─────────────────────────────────────────────────────


class TestNodeSerialization:
    def test_node_to_dict_roundtrip(self):
        """ReasoningNode → dict → ReasoningNode preserves data."""
        original = _make_leaf()
        d = _node_to_dict(original)
        restored = _dict_to_node(d)
        assert restored.id == original.id
        assert restored.label == original.label
        assert restored.description == original.description
        assert restored.is_actionable == original.is_actionable
        assert restored.action_text == original.action_text
        assert restored.time_slot == original.time_slot
        assert restored.frequency == original.frequency

    def test_followup_question_roundtrip(self):
        """followup_question survives node serialization roundtrip."""
        original = _make_mid_node(followup_question="A. 持续1个月 B. 持续半年 C. 超过1年")
        d = _node_to_dict(original)
        assert d["followup_question"] == original.followup_question
        restored = _dict_to_node(d)
        assert restored.followup_question == original.followup_question

    def test_followup_options_roundtrip(self):
        """followup_options survives node serialization roundtrip."""
        original = _make_mid_node(
            followup_question="这种情况持续多久了？",
            followup_options=["已持续1个月", "半年以上", "不清楚"],
        )
        d = _node_to_dict(original)
        assert d["followup_options"] == ["已持续1个月", "半年以上", "不清楚"]
        restored = _dict_to_node(d)
        assert restored.followup_options == ["已持续1个月", "半年以上", "不清楚"]

    def test_followup_question_default_none(self):
        """Node without followup_question serializes to null."""
        node = _make_mid_node()
        assert node.followup_question is None
        assert node.followup_options is None
        d = _node_to_dict(node)
        assert d["followup_question"] is None
        assert d["followup_options"] is None

    def test_followup_options_normalized_on_load(self):
        """Legacy/corrupt followup_options values are coerced to list[str] | None.

        The field is documented as list[str], but a malformed JSON file (or an
        older writer) could store a bare string, a list with structured junk
        (dicts/lists), numbers, or empty entries. Loading must degrade gracefully
        instead of leaking non-list values into serialization / API responses.
        """
        # Non-list (bare string) → None
        assert _dict_to_node({"id": "x", "label": "l", "followup_options": "早餐,午餐"}).followup_options is None
        # List with structured junk / numbers / empties → clean string list
        node = _dict_to_node({
            "id": "x",
            "label": "l",
            "followup_options": [
                {"label": "早餐"},        # dict → dropped
                ["午餐"],                  # list → dropped
                " 自己感觉 ",             # kept (stripped)
                "   ",                    # empty → dropped
                3,                        # scalar → "3"
                None,                     # dropped
            ],
        })
        assert node.followup_options == ["自己感觉", "3"]
        # Empty/whitespace-only list → None
        assert _dict_to_node({"id": "x", "label": "l", "followup_options": []}).followup_options is None
        assert _dict_to_node({"id": "x", "label": "l", "followup_options": ["  "]}) .followup_options is None

    def test_node_to_dict_with_children(self):
        """dict representation includes children recursively."""
        parent = _make_mid_node()
        child = _make_leaf(parent_id=parent.id)
        parent.children.append(child)
        d = _node_to_dict(parent)
        assert len(d["children"]) == 1
        assert d["children"][0]["id"] == child.id
        assert d["children"][0]["is_actionable"]

    def test_dict_to_node_restores_children(self):
        """Restoring from dict recovers child nodes."""
        parent = _make_mid_node()
        child = _make_leaf(parent_id=parent.id)
        parent.children.append(child)
        d = _node_to_dict(parent)
        restored = _dict_to_node(d)
        assert len(restored.children) == 1
        assert restored.children[0].label == child.label

    def test_dict_to_node_null_children_degrade_to_empty(self):
        """Legacy/corrupt 'children: null' must not crash deserialization."""
        node = _dict_to_node({"id": "x", "label": "l", "children": None})
        assert node.children == []

    def test_dict_to_node_dict_children_degrade_to_empty(self):
        """Legacy/corrupt 'children: {..}' (a dict) must not iterate keys as nodes."""
        node = _dict_to_node({"id": "x", "label": "l", "children": {"child": "not-a-node"}})
        assert node.children == []

    def test_dict_to_node_skips_non_dict_children(self):
        """Children entries that aren't dicts are dropped, valid ones are kept.

        A malformed JSON file could hold a bare string / number alongside real
        node dicts; deserializing must not raise AttributeError on .get().
        """
        node = _dict_to_node({
            "id": "x",
            "label": "l",
            "children": [
                {"id": "c1", "label": "真实子节点", "description": "", "emoji": "", "depth": 1},
                "junk-string",          # non-dict → dropped
                42,                     # non-dict → dropped
                None,                   # non-dict → dropped
            ],
        })
        assert len(node.children) == 1
        assert node.children[0].id == "c1"
        assert node.children[0].label == "真实子节点"


class TestPlanSerialization:
    def test_plan_to_dict_roundtrip(self):
        """HealthPlan → dict → HealthPlan preserves data."""
        root = ReasoningNode(id="root", label="脱发", description="主诉", emoji="🧑‍🦲", depth=0)
        plan = HealthPlan(
            id="plan_x", user_id="u1", concern="脱发", title="头发养护",
            goal="改善脱发", reasoning_tree=root,
        )
        d = _plan_to_dict(plan)
        restored = _dict_to_plan(d)
        assert restored.id == plan.id
        assert restored.concern == plan.concern
        assert restored.reasoning_tree is not None
        assert restored.reasoning_tree.label == "脱发"

    def test_plan_to_dict_no_tree(self):
        """Plan without reasoning tree serializes correctly."""
        plan = HealthPlan(id="plan_y", user_id="u1", concern="失眠", title="睡眠计划", goal="改善睡眠")
        d = _plan_to_dict(plan)
        assert d["reasoning_tree"] is None
        restored = _dict_to_plan(d)
        assert restored.reasoning_tree is None

    def test_dict_to_plan_non_dict_reasoning_tree_degrades_to_none(self):
        """Corrupt 'reasoning_tree' (string/list) → None instead of a crash.

        Legacy/corrupt data could store the tree as a non-dict value; loading
        must degrade gracefully (falling back to legacy categories if present)
        rather than raising AttributeError in _dict_to_node.
        """
        plan = _dict_to_plan({
            "id": "p1", "user_id": "u1", "concern": "失眠", "title": "睡眠计划",
            "goal": "改善睡眠", "reasoning_tree": "corrupt-not-a-dict",
        })
        assert plan.reasoning_tree is None


# ── Test Plan CRUD ─────────────────────────────────────────────────────────


class TestCreatePlan:
    def test_create_basic_plan(self, hp_patch):
        """Create a basic health plan."""
        plan = create_plan(
            user_id="user_001",
            concern="脱发严重",
            title="🌿 头发养护计划",
            goal="改善脱发状况",
        )
        assert plan.id is not None
        assert plan.status == "active"
        assert plan.version == 1
        assert plan.reasoning_tree is not None
        assert plan.reasoning_tree.label == "脱发严重"

    def test_create_with_tree(self, hp_patch):
        """Create a plan with a predefined reasoning tree."""
        child = _make_leaf()
        root = ReasoningNode(
            id="root", label="脱发", description="主诉", emoji="🧑‍🦲",
            depth=0, children=[child],
        )
        plan = create_plan(
            user_id="user_001",
            concern="脱发",
            title="头发养护",
            goal="改善脱发",
            reasoning_tree=root,
        )
        assert plan.reasoning_tree.id == "root"
        assert len(plan.reasoning_tree.children) == 1

    def test_create_multiple_users(self, hp_patch):
        """Plans for different users are isolated."""
        p1 = create_plan("u1", "脱发", "P1", "改善")
        p2 = create_plan("u2", "失眠", "P2", "改善")
        plans_u1 = list_plans("u1")
        plans_u2 = list_plans("u2")
        assert len(plans_u1) == 1
        assert len(plans_u2) == 1
        assert plans_u1[0].id == p1.id
        assert plans_u2[0].id == p2.id


class TestGetPlan:
    def test_get_active_plan_returns_most_recent(self, hp_patch):
        """get_active_plan returns the active plan."""
        p1 = create_plan("u1", "脱发", "脱发计划", "改善")
        active = get_active_plan("u1")
        assert active is not None
        assert active.id == p1.id

    def test_get_active_plan_no_plan(self, hp_patch):
        """get_active_plan returns None when no plan exists."""
        assert get_active_plan("nobody") is None

    def test_get_plan_by_id(self, hp_patch):
        """get_plan returns a specific plan by ID."""
        p = create_plan("u1", "脱发", "脱发计划", "改善")
        found = get_plan(p.id, "u1")
        assert found is not None
        assert found.id == p.id

    def test_get_plan_wrong_user(self, hp_patch):
        """get_plan returns None for wrong user."""
        p = create_plan("u1", "脱发", "脱发计划", "改善")
        found = get_plan(p.id, "u2")
        assert found is None

    def test_list_plans_empty(self, hp_patch):
        """list_plans returns empty list for new user."""
        assert list_plans("nobody") == []

    def test_list_plans_multiple(self, hp_patch):
        """list_plans returns all plans for a user."""
        create_plan("u1", "脱发", "P1", "改善")
        create_plan("u1", "失眠", "P2", "改善")
        plans = list_plans("u1")
        assert len(plans) == 2

    def test_list_plans_filter_by_status(self, hp_patch):
        """list_plans can filter by status."""
        p = create_plan("u1", "脱发", "P1", "改善")
        update_plan_status(p.id, "u1", "archived")
        create_plan("u1", "失眠", "P2", "改善")
        active = list_plans("u1", status="active")
        archived = list_plans("u1", status="archived")
        assert len(active) == 1
        assert len(archived) == 1


class TestUpdatePlan:
    def test_update_status(self, hp_patch):
        """Update plan status."""
        p = create_plan("u1", "脱发", "P1", "改善")
        updated = update_plan_status(p.id, "u1", "completed")
        assert updated.status == "completed"
        # Verify persistence
        reloaded = get_plan(p.id, "u1")
        assert reloaded.status == "completed"

    def test_update_status_not_found(self, hp_patch):
        """Update status on nonexistent plan returns None."""
        result = update_plan_status("nonexistent", "u1", "active")
        assert result is None

    def test_rename_plan(self, hp_patch):
        """Rename a plan."""
        p = create_plan("u1", "脱发", "旧标题", "改善")
        renamed = rename_plan(p.id, "u1", "新标题")
        assert renamed.title == "新标题"
        reloaded = get_plan(p.id, "u1")
        assert reloaded.title == "新标题"

    def test_delete_plan(self, hp_patch):
        """Delete a plan."""
        p = create_plan("u1", "脱发", "P1", "改善")
        assert delete_plan(p.id, "u1")
        assert get_plan(p.id, "u1") is None

    def test_delete_plan_not_found(self, hp_patch):
        """Delete nonexistent plan returns False."""
        assert not delete_plan("nonexistent", "u1")


# ── Test Tree Operations ────────────────────────────────────────────────────


class TestFindNodeInTree:
    def test_find_root(self):
        """Find root node by ID."""
        root = ReasoningNode(id="root", label="脱发", description="主诉", emoji="🧑‍🦲", depth=0)
        found = find_node_in_tree("root", root)
        assert found is not None
        assert found.id == "root"

    def test_find_child(self):
        """Find a child node by ID."""
        child = _make_leaf(id="leaf1")
        root = ReasoningNode(id="root", label="脱发", description="主诉", emoji="🧑‍🦲", depth=0, children=[child])
        found = find_node_in_tree("leaf1", root)
        assert found is not None
        assert found.id == "leaf1"

    def test_find_nested(self):
        """Find a deeply nested node."""
        leaf = _make_leaf(id="deep_leaf")
        mid = _make_mid_node(children=[leaf])
        root = ReasoningNode(id="root", label="脱发", description="主诉", emoji="🧑‍🦲", depth=0, children=[mid])
        found = find_node_in_tree("deep_leaf", root)
        assert found is not None
        assert found.id == "deep_leaf"

    def test_find_nonexistent(self):
        """Find returns None for nonexistent ID."""
        root = ReasoningNode(id="root", label="脱发", description="主诉", emoji="🧑‍🦲", depth=0)
        assert find_node_in_tree("ghost", root) is None


class TestBuildPathToNode:
    def test_path_to_root(self):
        """Path from root to root is a single entry."""
        root = ReasoningNode(id="root", label="脱发", description="主诉", emoji="🧑‍🦲", depth=0)
        path = build_path_to_node("root", root)
        assert len(path) == 1
        assert path[0]["label"] == "脱发"

    def test_path_to_child(self):
        """Path includes all ancestors."""
        child = _make_leaf(id="leaf1")
        root = ReasoningNode(id="root", label="脱发", description="主诉", emoji="🧑‍🦲", depth=0, children=[child])
        path = build_path_to_node("leaf1", root)
        assert len(path) == 2
        assert path[0]["label"] == "脱发"
        assert path[1]["label"] == child.label

    def test_path_to_nonexistent(self):
        """Path returns empty for nonexistent node."""
        root = ReasoningNode(id="root", label="脱发", description="主诉", emoji="🧑‍🦲", depth=0)
        assert build_path_to_node("ghost", root) == []


class TestExpandNode:
    def test_expand_adds_children(self, hp_patch):
        """expand_node adds children to a node and sets expanded=True."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        children_data = [
            {"label": "激素因素", "description": "雄激素影响", "emoji": "🧬",
             "is_actionable": False, "expandable": True},
        ]
        updated = expand_node(plan.id, "u1", plan.reasoning_tree.id, children_data)
        assert updated is not None
        root = updated.reasoning_tree
        assert root.expanded
        assert len(root.children) == 1
        assert root.children[0].label == "激素因素"

    def test_expand_with_actionable(self, hp_patch):
        """expand_node handles actionable children."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        children_data = [
            {"label": "早餐全麦面包", "description": "低GI方案", "emoji": "🍞",
             "is_actionable": True, "expandable": False,
             "action_text": "早餐全麦面包替代白面包", "time_slot": "早餐",
             "frequency": "每天", "reasoning": "低GI减少胰岛素波动"},
        ]
        updated = expand_node(plan.id, "u1", plan.reasoning_tree.id, children_data)
        child = updated.reasoning_tree.children[0]
        assert child.is_actionable
        assert child.action_text == "早餐全麦面包替代白面包"
        assert child.time_slot == "早餐"
        assert child.frequency == "每天"

    def test_expand_nonexistent_node(self, hp_patch):
        """expand_node returns plan unchanged for nonexistent node."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        result = expand_node(plan.id, "u1", "nonexistent", [])
        assert result is not None
        assert result.id == plan.id
        # Tree should be unchanged (no children added)
        assert len(result.reasoning_tree.children) == 0

    def test_expand_twice_does_not_duplicate_children(self, hp_patch):
        """Double-expanding the same node is idempotent (no duplicate children)."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        children_data = [
            {"label": "激素因素", "description": "雄激素影响", "emoji": "🧬",
             "is_actionable": False, "expandable": True},
        ]
        first = expand_node(plan.id, "u1", plan.reasoning_tree.id, children_data)
        assert len(first.reasoning_tree.children) == 1
        first_child_id = first.reasoning_tree.children[0].id
        assert first.reasoning_tree.expanded

        second = expand_node(plan.id, "u1", plan.reasoning_tree.id, children_data)
        assert second is not None
        assert len(second.reasoning_tree.children) == 1
        # The original child is preserved and no new sibling was added.
        assert second.reasoning_tree.children[0].id == first_child_id

        # Reload from disk: guard is persisted, so still no duplicates.
        plan = get_plan(plan.id, "u1")
        assert len(plan.reasoning_tree.children) == 1

    def test_expand_with_empty_children_is_noop(self, hp_patch):
        """expand_node with empty children_data is a no-op (doesn't lock the node)."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        root_id = plan.reasoning_tree.id
        assert plan.reasoning_tree.expanded is False

        result = expand_node(plan.id, "u1", root_id, [])
        assert result is not None
        assert result.reasoning_tree.expanded is False
        assert len(result.reasoning_tree.children) == 0

        # The node must still be expandable afterwards (not permanently locked).
        children_data = [
            {"label": "激素因素", "description": "雄激素影响", "emoji": "🧬",
             "is_actionable": False, "expandable": True},
        ]
        updated = expand_node(plan.id, "u1", root_id, children_data)
        assert updated.reasoning_tree.expanded is True
        assert len(updated.reasoning_tree.children) == 1

    def test_list_new_actionable_leaves_diffs_by_known_ids(self, hp_patch):
        """list_new_actionable_leaves returns only leaves NOT in the known set.

        Mirrors the expand API flow: capture known_ids before expansion, then
        after expansion only the newly created actionable leaves are returned
        (used by the frontend "追问后一键全部采纳" banner).
        """
        from health_plan import collect_actionable_leaves, list_new_actionable_leaves
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        before_ids = {n.id for n in collect_actionable_leaves(plan)}
        # No actionable leaves yet → nothing new
        assert list_new_actionable_leaves(plan, before_ids) == []

        children_data = [
            {"label": "早餐全麦面包", "description": "低GI方案", "emoji": "🍞",
             "is_actionable": True, "expandable": False,
             "action_text": "早餐全麦面包替代白面包", "time_slot": "早餐",
             "frequency": "每天", "reasoning": "低GI减少胰岛素波动"},
            {"label": "激素因素", "description": "继续拆解", "emoji": "🧬",
             "is_actionable": False, "expandable": True},
        ]
        updated = expand_node(plan.id, "u1", plan.reasoning_tree.id, children_data)

        new_leaves = list_new_actionable_leaves(updated, before_ids)
        assert len(new_leaves) == 1
        assert new_leaves[0]["label"] == "早餐全麦面包"
        # 精简字段含前端横幅展示 + 逐个采纳所需的最小信息
        assert set(new_leaves[0]) >= {"id", "label", "emoji", "time_slot", "frequency", "duration"}
        # 已在上一次"known set"中的叶子不会再被标记为新增
        assert list_new_actionable_leaves(updated, before_ids | {new_leaves[0]["id"]}) == []

    def test_expand_clears_pending_followup(self, hp_patch):
        """expanding a node clears its pending followup_question (question answered)."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        root_id = plan.reasoning_tree.id
        from health_plan import set_followup_question
        set_followup_question(plan.id, "u1", root_id, "A. 持续1个月 B. 半年以上")
        pending = get_plan(plan.id, "u1")
        assert pending.reasoning_tree.followup_question == "A. 持续1个月 B. 半年以上"

        children_data = [
            {"label": "激素因素", "description": "雄激素影响", "emoji": "🧬",
             "is_actionable": False, "expandable": True},
        ]
        updated = expand_node(plan.id, "u1", root_id, children_data)
        assert updated.reasoning_tree.followup_question is None

    def test_expand_clears_followup_options(self, hp_patch):
        """expanding a node clears pending followup_options along with the question."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        root_id = plan.reasoning_tree.id
        from health_plan import set_followup_question
        set_followup_question(plan.id, "u1", root_id, "多久了？", options=["1个月", "半年"])
        pending = get_plan(plan.id, "u1")
        assert pending.reasoning_tree.followup_options == ["1个月", "半年"]

        children_data = [
            {"label": "激素因素", "description": "雄激素影响", "emoji": "🧬",
             "is_actionable": False, "expandable": True},
        ]
        updated = expand_node(plan.id, "u1", root_id, children_data)
        assert updated.reasoning_tree.followup_question is None
        assert updated.reasoning_tree.followup_options is None

    def test_expand_carries_followup_on_children(self, hp_patch):
        """child nodes preserve followup_question passed via children_data."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        children_data = [
            {"label": "激素因素", "description": "雄激素影响", "emoji": "🧬",
             "is_actionable": False, "expandable": True,
             "followup_question": "A. 已查过激素 B. 未查过",
             "followup_options": ["已查过激素", "未查过"]},
        ]
        updated = expand_node(plan.id, "u1", plan.reasoning_tree.id, children_data)
        child = updated.reasoning_tree.children[0]
        assert child.followup_question == "A. 已查过激素 B. 未查过"
        assert child.followup_options == ["已查过激素", "未查过"]


class TestListNewActionableLeaves:
    def test_returns_only_newly_created_leaves(self, hp_patch):
        """known_ids 外的可采纳叶子才会被列出（用于展开后的新增识别）。"""
        from health_plan import list_new_actionable_leaves
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        plan = expand_node(plan.id, "u1", plan.reasoning_tree.id, [
            {"label": "早餐全麦", "description": "低GI", "emoji": "🍞",
             "is_actionable": True, "expandable": False},
            {"label": "激素因素", "description": "雄激素影响", "emoji": "🧬",
             "is_actionable": False, "expandable": True},
        ])
        new_leaf = plan.reasoning_tree.children[0]

        # 模拟展开前已知叶子为空 → 新增叶子应包含早餐全麦
        result = list_new_actionable_leaves(plan, set())
        assert len(result) == 1
        assert result[0]["id"] == new_leaf.id
        assert result[0]["label"] == "早餐全麦"
        assert result[0]["emoji"] == "🍞"
        assert "children" not in result[0]  # 精简 dict，不含树结构

        # 已知叶子集合包含该叶子 → 不再算新增
        assert list_new_actionable_leaves(plan, {new_leaf.id}) == []

    def test_known_ids_ignores_non_actionable(self, hp_patch):
        """非可采纳子节点（如展开中间节点）不会被当作新增叶子列出。"""
        from health_plan import list_new_actionable_leaves
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        plan = expand_node(plan.id, "u1", plan.reasoning_tree.id, [
            {"label": "激素因素", "description": "雄激素影响", "emoji": "🧬",
             "is_actionable": False, "expandable": True},
        ])
        assert list_new_actionable_leaves(plan, set()) == []


class TestSetFollowupQuestion:
    def test_set_and_get(self, hp_patch):
        """set_followup_question persists the question on a node."""
        plan = create_plan("u1", "失眠", "睡眠计划", "改善睡眠")
        from health_plan import set_followup_question
        updated = set_followup_question(plan.id, "u1", plan.reasoning_tree.id, "A. 入睡困难 B. 早醒 C. 两者都有")
        assert updated.reasoning_tree.followup_question == "A. 入睡困难 B. 早醒 C. 两者都有"
        # Reloaded from disk
        reloaded = get_plan(plan.id, "u1")
        assert reloaded.reasoning_tree.followup_question == "A. 入睡困难 B. 早醒 C. 两者都有"

    def test_set_and_get_with_options(self, hp_patch):
        """set_followup_question persists quick-choice options alongside the question."""
        plan = create_plan("u1", "失眠", "睡眠计划", "改善睡眠")
        from health_plan import set_followup_question
        updated = set_followup_question(
            plan.id, "u1", plan.reasoning_tree.id, "入睡困难还是早醒？",
            options=["入睡困难", "早醒", "两者都有"],
        )
        assert updated.reasoning_tree.followup_question == "入睡困难还是早醒？"
        assert updated.reasoning_tree.followup_options == ["入睡困难", "早醒", "两者都有"]
        reloaded = get_plan(plan.id, "u1")
        assert reloaded.reasoning_tree.followup_options == ["入睡困难", "早醒", "两者都有"]

    def test_set_normalizes_options(self, hp_patch):
        """set_followup_question coerces malformed options via _normalize_followup_options."""
        plan = create_plan("u1", "失眠", "睡眠计划", "改善睡眠")
        from health_plan import set_followup_question
        updated = set_followup_question(
            plan.id, "u1", plan.reasoning_tree.id, "饮食上有什么偏好？",
            options=[" 少油少盐 ", None, {"label": "垃圾"}, ["嵌套"], "  ", "清淡"],
        )
        assert updated.reasoning_tree.followup_question == "饮食上有什么偏好？"
        # Structured junk / null / blank dropped; scalars stripped in order
        assert updated.reasoning_tree.followup_options == ["少油少盐", "清淡"]
        reloaded = get_plan(plan.id, "u1")
        assert reloaded.reasoning_tree.followup_options == ["少油少盐", "清淡"]

    def test_clear_sets_none(self, hp_patch):
        """set_followup_question(..., None) clears the question."""
        plan = create_plan("u1", "失眠", "睡眠计划", "改善睡眠")
        from health_plan import set_followup_question
        set_followup_question(plan.id, "u1", plan.reasoning_tree.id, "是否需要确认？")
        set_followup_question(plan.id, "u1", plan.reasoning_tree.id, None)
        reloaded = get_plan(plan.id, "u1")
        assert reloaded.reasoning_tree.followup_question is None
        assert reloaded.reasoning_tree.followup_options is None

    def test_set_nonexistent_node(self, hp_patch):
        """set_followup_question on a nonexistent node returns None."""
        plan = create_plan("u1", "失眠", "睡眠计划", "改善睡眠")
        from health_plan import set_followup_question
        result = set_followup_question(plan.id, "u1", "nonexistent", "问？")
        assert result is None


class TestUpdateLeafStatus:
    def test_mark_done(self, hp_patch):
        """Mark a leaf as done increments count."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        children = [{"label": "早餐全麦", "description": "低GI", "emoji": "🍞",
                      "is_actionable": True, "expandable": False,
                      "action_text": "全麦面包", "time_slot": "早餐", "frequency": "每天"}]
        plan = expand_node(plan.id, "u1", plan.reasoning_tree.id, children)
        leaf_id = plan.reasoning_tree.children[0].id
        updated = update_leaf_status(plan.id, "u1", leaf_id, "done")
        leaf = find_node_in_tree(leaf_id, updated.reasoning_tree)
        assert leaf.status == "done"
        assert leaf.completed_count == 1
        assert leaf.last_completed is not None

    def test_mark_done_twice(self, hp_patch):
        """Marking done twice increments count to 2."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        children = [{"label": "早餐全麦", "description": "低GI", "emoji": "🍞",
                      "is_actionable": True, "expandable": False}]
        plan = expand_node(plan.id, "u1", plan.reasoning_tree.id, children)
        leaf_id = plan.reasoning_tree.children[0].id
        update_leaf_status(plan.id, "u1", leaf_id, "done")
        update_leaf_status(plan.id, "u1", leaf_id, "done")
        plan = get_plan(plan.id, "u1")
        leaf = find_node_in_tree(leaf_id, plan.reasoning_tree)
        assert leaf.completed_count == 2

    def test_missing_leaf_returns_none(self, hp_patch):
        """Updating a leaf that doesn't exist returns None (no silent no-op)."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        plan = expand_node(plan.id, "u1", plan.reasoning_tree.id, [
            {"label": "早餐全麦", "description": "低GI", "emoji": "🍞",
             "is_actionable": True, "expandable": False},
        ])
        result = update_leaf_status(plan.id, "u1", "nonexistent-leaf-id", "pending")
        assert result is None

    def test_missing_plan_returns_none(self, hp_patch):
        """Updating a plan that doesn't exist returns None."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        plan = expand_node(plan.id, "u1", plan.reasoning_tree.id, [
            {"label": "早餐全麦", "description": "低GI", "emoji": "🍞",
             "is_actionable": True, "expandable": False},
        ])
        result = update_leaf_status("nonexistent-plan-id", "u1", plan.reasoning_tree.children[0].id, "pending")
        assert result is None

    def test_un_done_clears_last_completed_and_decrements_count(self, hp_patch):
        """Un-done (done → pending) clears last_completed and decrements completed_count.

        A leaf that is no longer "done" must not keep a stale completion timestamp
        nor an over-counted completed_count.
        """
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        plan = expand_node(plan.id, "u1", plan.reasoning_tree.id, [
            {"label": "早餐全麦", "description": "低GI", "emoji": "🍞",
             "is_actionable": True, "expandable": False},
        ])
        leaf_id = plan.reasoning_tree.children[0].id

        done = update_leaf_status(plan.id, "u1", leaf_id, "done")
        leaf = find_node_in_tree(leaf_id, done.reasoning_tree)
        assert leaf.completed_count == 1
        assert leaf.last_completed is not None

        undone = update_leaf_status(plan.id, "u1", leaf_id, "pending")
        leaf = find_node_in_tree(leaf_id, undone.reasoning_tree)
        assert leaf.status == "pending"
        assert leaf.completed_count == 0
        assert leaf.last_completed is None

    def test_skip_after_done_also_un_does(self, hp_patch):
        """Skipping a previously done leaf also un-does it (count/timestamp revert)."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        plan = expand_node(plan.id, "u1", plan.reasoning_tree.id, [
            {"label": "早餐全麦", "description": "低GI", "emoji": "🍞",
             "is_actionable": True, "expandable": False},
        ])
        leaf_id = plan.reasoning_tree.children[0].id

        update_leaf_status(plan.id, "u1", leaf_id, "done")
        skipped = update_leaf_status(plan.id, "u1", leaf_id, "skipped")
        leaf = find_node_in_tree(leaf_id, skipped.reasoning_tree)
        assert leaf.status == "skipped"
        assert leaf.completed_count == 0
        assert leaf.last_completed is None

    def test_undone_then_redone_increments_once(self, hp_patch):
        """done → pending → done counts exactly one completion (no over-count)."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        plan = expand_node(plan.id, "u1", plan.reasoning_tree.id, [
            {"label": "早餐全麦", "description": "低GI", "emoji": "🍞",
             "is_actionable": True, "expandable": False},
        ])
        leaf_id = plan.reasoning_tree.children[0].id

        update_leaf_status(plan.id, "u1", leaf_id, "done")
        update_leaf_status(plan.id, "u1", leaf_id, "pending")
        redone = update_leaf_status(plan.id, "u1", leaf_id, "done")
        leaf = find_node_in_tree(leaf_id, redone.reasoning_tree)
        assert leaf.status == "done"
        assert leaf.completed_count == 1
        assert leaf.last_completed is not None

    def test_mark_pending_again_is_noop(self, hp_patch):
        """Setting a non-done leaf to pending again does not touch counters."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        plan = expand_node(plan.id, "u1", plan.reasoning_tree.id, [
            {"label": "早餐全麦", "description": "低GI", "emoji": "🍞",
             "is_actionable": True, "expandable": False},
        ])
        leaf_id = plan.reasoning_tree.children[0].id

        result = update_leaf_status(plan.id, "u1", leaf_id, "pending")
        leaf = find_node_in_tree(leaf_id, result.reasoning_tree)
        assert leaf.status == "pending"
        assert leaf.completed_count == 0
        assert leaf.last_completed is None


class TestCollectActionableLeaves:
    def test_empty_tree(self, hp_patch):
        """Empty tree yields no leaves."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        leaves = collect_actionable_leaves(plan)
        assert leaves == []

    def test_collects_actionable(self, hp_patch):
        """Collect only actionable nodes."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        plan = expand_node(plan.id, "u1", plan.reasoning_tree.id, [
            {"label": "激素因素", "description": "雄激素", "emoji": "🧬",
             "is_actionable": False, "expandable": True},
        ])
        hormone_node_id = plan.reasoning_tree.children[0].id
        plan = expand_node(plan.id, "u1", hormone_node_id, [
            {"label": "早餐全麦", "description": "低GI", "emoji": "🍞",
             "is_actionable": True, "expandable": False,
             "action_text": "全麦面包", "time_slot": "早餐", "frequency": "每天"},
        ])
        leaves = collect_actionable_leaves(plan)
        assert len(leaves) == 1
        assert leaves[0].label == "早餐全麦"

    def test_sorted_by_time_slot(self, hp_patch):
        """Leaves are sorted by time slot order."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        plan = expand_node(plan.id, "u1", plan.reasoning_tree.id, [
            {"label": "饮食", "description": "饮食", "emoji": "🥗",
             "is_actionable": False, "expandable": True},
        ])
        diet_node_id = plan.reasoning_tree.children[0].id
        plan = expand_node(plan.id, "u1", diet_node_id, [
            {"label": "睡前牛奶", "description": "助眠", "emoji": "🥛",
             "is_actionable": True, "expandable": False,
             "action_text": "睡前喝温牛奶", "time_slot": "睡前", "frequency": "每天"},
            {"label": "早餐燕麦", "description": "早餐", "emoji": "🥣",
             "is_actionable": True, "expandable": False,
             "action_text": "早餐吃燕麦", "time_slot": "早餐", "frequency": "每天"},
            {"label": "下午水果", "description": "加餐", "emoji": "🍎",
             "is_actionable": True, "expandable": False,
             "action_text": "下午吃水果", "time_slot": "下午", "frequency": "每天"},
        ])
        leaves = collect_actionable_leaves(plan)
        time_slots = [l.time_slot for l in leaves]
        # Expected order: 早餐, 下午, 睡前
        assert time_slots == ["早餐", "下午", "睡前"]


class TestAdoptLeaf:
    def test_adopt_sets_pending(self, hp_patch):
        """Adopt sets leaf status to pending."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        plan = expand_node(plan.id, "u1", plan.reasoning_tree.id, [
            {"label": "早餐全麦", "description": "低GI", "emoji": "🍞",
             "is_actionable": True, "expandable": False},
        ])
        leaf_id = plan.reasoning_tree.children[0].id
        updated = adopt_leaf(plan.id, "u1", leaf_id)
        leaf = find_node_in_tree(leaf_id, updated.reasoning_tree)
        assert leaf.status == "pending"

    def test_adopt_missing_leaf_returns_none(self, hp_patch):
        """Adopting a non-existent leaf returns None (so the API can 404)."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        plan = expand_node(plan.id, "u1", plan.reasoning_tree.id, [
            {"label": "早餐全麦", "description": "低GI", "emoji": "🍞",
             "is_actionable": True, "expandable": False},
        ])
        assert adopt_leaf(plan.id, "u1", "nonexistent-leaf-id") is None


class TestRegeneratePlan:
    def test_regenerate_archives_old_and_creates_new(self, hp_patch):
        """Regenerate archives the old plan and creates a new version."""
        plan = create_plan("u1", "脱发", "头发养护", "改善")
        old_id = plan.id
        new_tree = ReasoningNode(
            id="new_root", label="脱发v2", description="v2", emoji="🧑‍🦲", depth=0,
        )
        new_plan = regenerate_plan(plan.id, "u1", new_tree)
        assert new_plan.id != old_id
        assert new_plan.version == 2
        assert new_plan.reasoning_tree.id == "new_root"
        # Old plan should be archived
        old_plan = get_plan(old_id, "u1")
        assert old_plan.status == "archived"


# ── Test Reminder CRUD ─────────────────────────────────────────────────────


class TestReminderCRUD:
    def test_create_reminder(self, hp_patch):
        """Create a basic reminder."""
        rem = create_reminder(
            user_id="u1", plan_id="plan1",
            time_spec="15:00", message_template="该跑步了",
        )
        assert rem.id is not None
        assert rem.plan_id == "plan1"
        assert rem.time_spec == "15:00"
        assert rem.is_active

    def test_list_reminders(self, hp_patch):
        """List reminders for a user."""
        create_reminder("u1", "plan1", "15:00", "跑步")
        create_reminder("u1", "plan1", "21:00", "睡觉")
        rems = list_reminders("u1")
        assert len(rems) == 2

    def test_list_reminders_empty(self, hp_patch):
        """List returns empty for user with no reminders."""
        assert list_reminders("nobody") == []

    def test_delete_reminder(self, hp_patch):
        """Delete a reminder."""
        rem = create_reminder("u1", "plan1", "15:00", "跑步")
        assert delete_reminder(rem.id, "u1")
        assert list_reminders("u1") == []

    def test_delete_reminder_not_found(self, hp_patch):
        """Delete nonexistent reminder returns False."""
        assert not delete_reminder("ghost", "u1")

    def test_list_skips_junk_entries(self, hp_patch):
        """list_reminders skips non-dict / id-less entries instead of crashing."""
        _seed_reminders_file(hp_patch, {
            "u1": [
                {"id": "r1", "plan_id": "p1", "user_id": "u1", "time_spec": "15:00",
                 "message_template": "跑步", "is_active": True, "adaptive": True},
                "pure-junk-string",
                42,
                None,
                {"id": "r2", "plan_id": "p1", "user_id": "u1", "time_spec": "21:00",
                 "message_template": "睡觉", "is_active": False, "adaptive": True},
                {"plan_id": "p1", "time_spec": "07:00"},  # dict without an id → junk
            ],
        })
        # Default active_only=True: only the active well-formed reminder shows.
        assert [r.id for r in list_reminders("u1")] == ["r1"]
        assert [r.id for r in list_reminders("u1", active_only=False)] == ["r1", "r2"]

    def test_list_tolerates_corrupt_container(self, hp_patch):
        """A non-list value stored under the user id degrades to an empty list."""
        _seed_reminders_file(hp_patch, {"u1": "not-a-list"})
        assert list_reminders("u1") == []
        _seed_reminders_file(hp_patch, {"u1": {"oops": 1}})
        assert list_reminders("u1") == []
        _seed_reminders_file(hp_patch, {"u1": None})
        assert list_reminders("u1") == []

    def test_delete_skips_junk_and_still_deletes_real(self, hp_patch):
        """delete_reminder ignores junk entries and still deletes the real one."""
        _seed_reminders_file(hp_patch, {
            "u1": [
                {"id": "keep", "plan_id": "p1", "user_id": "u1", "time_spec": "15:00",
                 "message_template": "跑步", "is_active": True, "adaptive": True},
                "junk-string",
            ],
        })
        assert delete_reminder("keep", "u1") is True
        assert list_reminders("u1") == []
        # Deleting a junk id is a no-op, not a crash.
        assert delete_reminder("junk-string", "u1") is False

    def test_create_repairs_corrupt_container(self, hp_patch):
        """create_reminder replaces a corrupt non-list container with a fresh list."""
        _seed_reminders_file(hp_patch, {"u1": {"oops": 1}})
        rem = create_reminder("u1", "plan1", "15:00", "跑步")
        reminders = list_reminders("u1")
        assert [r.id for r in reminders] == [rem.id]
        assert reminders[0].message_template == "跑步"

    def test_delete_plan_reminder_cleanup_tolerates_junk(self, hp_patch):
        """delete_plan removes matching reminders while skipping junk entries."""
        plan = create_plan("u1", "脱发", "P1", "改善")
        _seed_reminders_file(hp_patch, {
            "u1": [
                {"id": "rm1", "plan_id": plan.id, "user_id": "u1", "time_spec": "15:00",
                 "message_template": "跑步", "is_active": True, "adaptive": True},
                "junk-string",
                {"id": "other", "plan_id": "some-other-plan", "user_id": "u1",
                 "time_spec": "21:00", "message_template": "睡觉",
                 "is_active": True, "adaptive": True},
            ],
            "u2": ["junk-for-other-user"],
        })
        assert delete_plan(plan.id, "u1") is True
        reminders = list_reminders("u1", active_only=False)
        assert [r.id for r in reminders] == ["other"]


# ── Test Legacy Conversion ──────────────────────────────────────────────────


class TestLegacyConversion:
    def test_convert_empty(self):
        """Convert empty categories list."""
        root = _convert_legacy_categories_to_tree([])
        assert root.label == "健康计划"
        assert root.children == []

    def test_convert_single_category(self):
        """Convert a single category with items."""
        categories = [
            {
                "name": "饮食调整",
                "icon": "🥗",
                "items": [
                    {"item_id": "i1", "action": "早餐全麦面包",
                     "time_slot": "早餐", "frequency": "每天",
                     "reasoning": "低GI食物"},
                ],
            }
        ]
        root = _convert_legacy_categories_to_tree(categories)
        assert len(root.children) == 1
        cat = root.children[0]
        assert cat.label == "饮食调整"
        assert cat.emoji == "🥗"
        assert len(cat.children) == 1
        leaf = cat.children[0]
        assert leaf.is_actionable
        assert leaf.action_text == "早餐全麦面包"
        assert leaf.time_slot == "早餐"

    def test_convert_multi_category(self):
        """Convert multiple categories."""
        categories = [
            {"name": "饮食", "icon": "🥗", "items": [{"item_id": "i1", "action": "多吃蔬菜"}]},
            {"name": "运动", "icon": "🏃", "items": [{"item_id": "i2", "action": "每天快走"}]},
        ]
        root = _convert_legacy_categories_to_tree(categories)
        assert len(root.children) == 2
        assert root.children[0].label == "饮食"
        assert root.children[1].label == "运动"


# ── Test Edge Cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_lock_file_created(self, hp_patch):
        """Lock files are created during operations."""
        create_plan("u1", "脱发", "P1", "改善")
        lock_path = Path(hp_patch / "health_plans.json.lock")
        assert lock_path.exists()

    def test_data_file_created(self, hp_patch):
        """Data file is created after plan creation."""
        create_plan("u1", "脱发", "P1", "改善")
        data_path = Path(hp_patch / "health_plans.json")
        assert data_path.exists()
        with open(data_path) as f:
            data = json.load(f)
        assert "u1" in data
        assert len(data["u1"]) == 1

    def test_create_plan_with_context_snapshot(self, hp_patch):
        """Context snapshot is persisted."""
        snapshot = {"persona_id": "default", "routine_summary": {"sleep": "23:00-07:00"}}
        plan = create_plan("u1", "脱发", "P1", "改善", context_snapshot=snapshot)
        assert plan.context_snapshot["persona_id"] == "default"
        reloaded = get_plan(plan.id, "u1")
        assert reloaded.context_snapshot["persona_id"] == "default"

    def test_plan_has_correct_defaults(self):
        """HealthPlan has correct default values."""
        plan = HealthPlan(id="p1", user_id="u1", concern="c", title="t", goal="g")
        assert plan.status == "active"
        assert plan.version == 1
        assert plan.reasoning_tree is None
        assert plan.context_snapshot == {}

    def test_reminder_has_correct_defaults(self):
        """PlanReminder has correct default values."""
        rem = PlanReminder(id="r1", plan_id="p1", user_id="u1")
        assert rem.trigger_type == "time"
        assert rem.time_spec == ""
        assert rem.is_active
        assert rem.adaptive

    def test_delete_plan_also_deletes_reminders(self, hp_patch):
        """Deleting a plan also removes associated reminders."""
        plan = create_plan("u1", "脱发", "P1", "改善")
        create_reminder("u1", plan.id, "15:00", "测试提醒")
        assert len(list_reminders("u1")) == 1
        delete_plan(plan.id, "u1")
        assert len(list_reminders("u1")) == 0


# ── Test corrupt-tree write paths (update/expand/set operate on raw dict) ────


class TestCorruptTreeWritePaths:
    """update_leaf_status / expand_node / set_followup_question read the stored
    raw dict tree directly (bypassing _dict_to_node normalization), so they must
    tolerate the same legacy/corrupt children shapes that _dict_to_node now does.
    """

    def test_update_leaf_status_tolerates_corrupt_siblings(self, hp_patch):
        plan_id = _seed_plan_tree(hp_patch, _corrupt_sibling_tree())
        updated = update_leaf_status(plan_id, "u1", "leaf1", "done")
        assert updated is not None
        leaf = find_node_in_tree("leaf1", updated.reasoning_tree)
        assert leaf.status == "done"
        assert leaf.completed_count == 1

        # A node hidden inside a corrupt dict-shaped children subtree is
        # unreachable: degrade to not-found (None) rather than crashing.
        assert update_leaf_status(plan_id, "u1", "inner1", "done") is None

    def test_set_followup_question_tolerates_corrupt_siblings(self, hp_patch):
        from health_plan import set_followup_question
        plan_id = _seed_plan_tree(hp_patch, _corrupt_sibling_tree())
        updated = set_followup_question(
            plan_id, "u1", "leaf1", "多久了？", options=["1个月", "半年"],
        )
        assert updated is not None
        leaf = find_node_in_tree("leaf1", updated.reasoning_tree)
        assert leaf.followup_question == "多久了？"
        assert leaf.followup_options == ["1个月", "半年"]

    def test_expand_node_tolerates_corrupt_siblings(self, hp_patch):
        # Root is expanded and holds a corrupt sibling before a valid unexpanded
        # mid node; expand_node must skip the corrupt branch and still reach mid1.
        tree = {
            "id": "root", "label": "r", "description": "", "emoji": "🧑",
            "depth": 0, "expanded": True, "expandable": False,
            "children": [
                {"id": "bad1", "label": "损坏", "description": "",
                 "emoji": "💥", "depth": 1, "is_actionable": False,
                 "expanded": False,
                 "children": {"inner": {"id": "inner1", "children": []}}},
                {"id": "mid1", "label": "激素因素", "description": "雄激素",
                 "emoji": "🧬", "depth": 1, "is_actionable": False,
                 "expanded": False, "children": []},
            ],
        }
        plan_id = _seed_plan_tree(hp_patch, tree)
        updated = expand_node(plan_id, "u1", "mid1", [
            {"label": "早餐全麦", "description": "低GI", "emoji": "🍞",
             "is_actionable": True, "expandable": False,
             "action_text": "全麦", "time_slot": "早餐", "frequency": "每天"},
        ])
        assert updated is not None
        mid = find_node_in_tree("mid1", updated.reasoning_tree)
        assert mid is not None
        assert mid.expanded is True
        assert len(mid.children) == 1
        assert mid.children[0].is_actionable


class TestDictToPlanDeserializationGuards:
    """Legacy/corrupt persisted values that previously crashed get_plan/list_plans
    via _dict_to_plan now degrade gracefully."""

    def test_corrupt_categories_dict_degrades_to_no_tree(self, hp_patch):
        plan = create_plan("u1", "脱发", "P1", "改善")
        fp = Path(hp_patch) / "health_plans.json"
        data = json.loads(fp.read_text(encoding="utf-8"))
        for pd in data["u1"]:
            if pd["id"] == plan.id:
                pd["reasoning_tree"] = None
                # truthy-but-corrupt: categories is a dict, not a list of dicts
                pd["categories"] = {"饮食": {"name": "饮食"}}
        fp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        loaded = get_plan(plan.id, "u1")
        assert loaded is not None
        assert loaded.reasoning_tree is None

    def test_corrupt_categories_list_skips_junk(self, hp_patch):
        # categories is a list mixing junk with valid entries; valid entries
        # survive, corrupt categories/items entries are skipped.
        plan = create_plan("u1", "脱发", "P1", "改善")
        fp = Path(hp_patch) / "health_plans.json"
        data = json.loads(fp.read_text(encoding="utf-8"))
        for pd in data["u1"]:
            if pd["id"] == plan.id:
                pd["reasoning_tree"] = None
                pd["categories"] = [
                    "junk-string",
                    {"name": "饮食", "icon": "🥗", "items": [
                        {"item_id": "i1", "action": "多吃蔬菜"},
                        42,
                    ]},
                    {"name": "运动", "icon": "🏃", "items": "not-a-list"},
                    {"name": "睡眠", "icon": "🛌"},
                ]
        fp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        loaded = get_plan(plan.id, "u1")
        assert loaded.reasoning_tree is not None
        labels = [c.label for c in loaded.reasoning_tree.children]
        assert labels == ["饮食", "运动", "睡眠"]
        diet = loaded.reasoning_tree.children[0]
        assert [leaf.label for leaf in diet.children] == ["多吃蔬菜"]

    def test_non_dict_context_snapshot_degrades_to_empty(self):
        plan = _dict_to_plan({
            "id": "p1", "user_id": "u1", "concern": "c", "title": "t",
            "goal": "g", "context_snapshot": "corrupt-not-a-dict",
        })
        assert plan.context_snapshot == {}

        plan_list = _dict_to_plan({
            "id": "p2", "user_id": "u1", "concern": "c", "title": "t",
            "goal": "g", "context_snapshot": [1, 2],
        })
        assert plan_list.context_snapshot == {}

    def test_context_snapshot_none_and_missing_unchanged(self):
        # Missing key keeps the {} default; explicit null stays None (unchanged).
        assert _dict_to_plan({
            "id": "p1", "user_id": "u1", "concern": "c", "title": "t", "goal": "g",
        }).context_snapshot == {}
        assert _dict_to_plan({
            "id": "p2", "user_id": "u1", "concern": "c", "title": "t", "goal": "g",
            "context_snapshot": None,
        }).context_snapshot is None

    def test_convert_non_list_categories_returns_empty_root(self):
        root = _convert_legacy_categories_to_tree({"饮食": [{"action": "x"}]})
        assert root.children == []
        assert _convert_legacy_categories_to_tree("junk").children == []

    def test_convert_skips_corrupt_categories_and_items(self):
        categories = [
            "junk-string",
            {"name": "饮食", "icon": "🥗", "items": [
                {"item_id": "i1", "action": "多吃蔬菜"},
                42,  # corrupt item → skipped
            ]},
            {"name": "运动", "icon": "🏃", "items": "not-a-list"},  # → no items
            {"name": "睡眠", "icon": "🛌"},  # valid, no items
        ]
        root = _convert_legacy_categories_to_tree(categories)
        assert [c.label for c in root.children] == ["饮食", "运动", "睡眠"]
        diet = root.children[0]
        assert [leaf.label for leaf in diet.children] == ["多吃蔬菜"]
