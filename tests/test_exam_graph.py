"""Viora — Physical Exam Graph Tests

Tests for exam_graph.py: build, save, load, and validate exam graph generation.
"""

import sys
import json
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from exam_graph import (
    build_exam_graph,
    save_exam_graph,
    load_exam_graph,
    ExamGraphError,
)


# ── Sample Exam Data Fixtures ─────────────────────────────────────────────

@pytest.fixture
def minimal_exam_data():
    """Minimal valid exam data."""
    return {
        "个人信息": {
            "姓名": "张三",
            "年龄": 30,
            "性别": "男",
            "体检号": "PE2025001",
            "参检日期": "2025-01-15",
        },
        "血液检查": {
            "检查日期": "2025-01-15",
            "检查医生": "李医生",
            "白细胞计数": {
                "结果": "5.0",
                "单位": "10^9/L",
                "参考值": "4.0-10.0",
            },
            "红细胞计数": {
                "结果": "4.5",
                "单位": "10^12/L",
                "参考值": "4.0-5.5",
            },
        },
        "影像检查": {
            "检查所见": "肺部正常",
            "小结": "无异常",
        },
    }


@pytest.fixture
def complex_exam_data():
    """Exam data with nested structures and abnormal results."""
    return {
        "体检信息": {
            "姓名": "李四",
            "年龄": 45,
            "性别": "女",
            "体检号": "PE2025030",
            "参检日期": "2025-03-20",
        },
        "检验结果": {
            "血常规": {
                "检查日期": "2025-03-20",
                "检查医生": "王医生",
                "项目": [
                    {"项目": "白细胞计数", "结果": "3.5", "单位": "10^9/L",
                     "参考值": "4.0-10.0", "异常": "偏低"},
                    {"项目": "红细胞计数", "结果": "4.2", "单位": "10^12/L",
                     "参考值": "4.0-5.5", "异常": "正常"},
                ],
                "结果提示": "白细胞偏低",
            },
            "肝功能": {
                "检查日期": "2025-03-20",
                "谷丙转氨酶": {"结果": "45", "单位": "U/L", "参考值": "0-40"},
                "谷草转氨酶": {"结果": "35", "单位": "U/L", "参考值": "0-40"},
            },
        },
    }


@pytest.fixture
def exam_with_wrapper():
    """Exam data wrapped in top-level container keys."""
    return {
        "体检项目": {
            "个人信息": {
                "姓名": "王五",
                "年龄": 28,
                "性别": "男",
                "参检日期": "2025-05-01",
            },
            "一般检查": {
                "身高": {"结果": "175", "单位": "cm"},
                "体重": {"结果": "70", "单位": "kg"},
            },
        },
    }


# ── Build Exam Graph Tests ────────────────────────────────────────────────

class TestBuildExamGraph:
    """Test build_exam_graph() with various input formats."""

    def test_build_minimal_exam(self, minimal_exam_data):
        """Build graph from minimal exam data."""
        graph = build_exam_graph(minimal_exam_data)
        assert "nodes" in graph
        assert "relationships" in graph
        # Should have at least: patient, exam, blood check category,
        # 2 blood results, imaging category, 1 imaging result
        assert len(graph["nodes"]) >= 7
        assert len(graph["relationships"]) >= 5

    def test_build_complex_exam(self, complex_exam_data):
        """Build graph from complex nested exam data."""
        graph = build_exam_graph(complex_exam_data)
        assert len(graph["nodes"]) > 0
        assert len(graph["relationships"]) > 0
        # Verify node labels present
        labels = set()
        for n in graph["nodes"]:
            lbl = n.get("label", [])
            if isinstance(lbl, list):
                labels.update(lbl)
            elif isinstance(lbl, str):
                labels.add(lbl)
        assert "受检人" in labels or "个人" in labels
        assert "体检" in labels

    def test_build_with_wrapper(self, exam_with_wrapper):
        """Build graph from wrapped exam data (容器键自动展开)."""
        graph = build_exam_graph(exam_with_wrapper)
        assert len(graph["nodes"]) >= 4  # patient, exam, general check, height, weight
        # Check that personal info was extracted
        props_list = [n.get("properties", {}) for n in graph["nodes"]]
        has_patient = any("个人" in str(p.get("name", "")) for p in props_list)
        assert has_patient

    def test_patient_node_exists(self, minimal_exam_data):
        """Patient node is created with correct label."""
        graph = build_exam_graph(minimal_exam_data)
        patient_nodes = [
            n for n in graph["nodes"]
            if "受检人" in (n.get("label", []) if isinstance(n.get("label"), list) else [n.get("label", "")])
        ]
        assert len(patient_nodes) == 1
        assert "个人" in str(patient_nodes[0].get("properties", {}))

    def test_exam_node_exists(self, minimal_exam_data):
        """Exam event node is created."""
        graph = build_exam_graph(minimal_exam_data)
        exam_nodes = [
            n for n in graph["nodes"]
            if (isinstance(n.get("label"), list) and "事件" in n.get("label"))
        ]
        assert len(exam_nodes) >= 1

    def test_personal_info_extracted(self, minimal_exam_data):
        """Personal info fields are extracted to exam node, not left in data."""
        graph = build_exam_graph(minimal_exam_data)
        # Find exam node
        exam_nodes = [n for n in graph["nodes"] if isinstance(n.get("label"), list) and "事件" in n["label"]]
        exam_props = exam_nodes[0].get("properties", {})
        assert exam_props.get("name") == "张三"
        assert exam_props.get("age") == 30

    def test_relationships_have_valid_refs(self, minimal_exam_data):
        """All relationship source/target IDs exist in nodes."""
        graph = build_exam_graph(minimal_exam_data)
        node_ids = {n["id"] for n in graph["nodes"]}
        for rel in graph["relationships"]:
            assert rel["source"] in node_ids, f"Source not found: {rel['source']}"
            assert rel["target"] in node_ids, f"Target not found: {rel['target']}"

    def test_abnormal_results_preserved(self, complex_exam_data):
        """Abnormal flags are preserved in result nodes."""
        graph = build_exam_graph(complex_exam_data)
        has_abnormal = False
        for n in graph["nodes"]:
            props = n.get("properties", {})
            if props.get("abnormal") == "偏低":
                has_abnormal = True
                break
        assert has_abnormal, "Should preserve abnormal='偏低' flag"

    def test_all_nodes_have_ids(self, minimal_exam_data):
        """Every node has a unique ID."""
        graph = build_exam_graph(minimal_exam_data)
        ids = [n["id"] for n in graph["nodes"]]
        assert len(ids) == len(set(ids))
        for n in graph["nodes"]:
            assert isinstance(n["id"], str)
            assert len(n["id"]) > 0

    def test_graph_is_json_serializable(self, minimal_exam_data):
        """Graph output is JSON-serializable."""
        graph = build_exam_graph(minimal_exam_data)
        serialized = json.dumps(graph, ensure_ascii=False)
        roundtripped = json.loads(serialized)
        assert len(roundtripped["nodes"]) == len(graph["nodes"])


class TestExamGraphErrors:
    """Test error handling in build_exam_graph()."""

    def test_empty_data_raises(self):
        """Empty dict raises ExamGraphError."""
        with pytest.raises(ExamGraphError):
            build_exam_graph({})

    def test_none_data_raises(self):
        """None raises ExamGraphError."""
        with pytest.raises(ExamGraphError):
            build_exam_graph(None)

    def test_non_dict_raises(self):
        """Non-dict input raises ExamGraphError."""
        with pytest.raises(ExamGraphError):
            build_exam_graph([])


class TestExamGraphSaveLoad:
    """Test save_exam_graph() and load_exam_graph()."""

    def test_save_and_load(self, minimal_exam_data, tmp_path, monkeypatch):
        """Save and load an exam graph round-trip."""
        import exam_graph as eg
        monkeypatch.setattr(eg, "PRESETS_DIR", tmp_path)

        graph = build_exam_graph(minimal_exam_data)
        filename = save_exam_graph(graph, user_id="test_user")
        assert filename == "exam_test_user.json"
        assert (tmp_path / filename).exists()

        loaded = load_exam_graph(user_id="test_user")
        assert loaded is not None
        assert len(loaded["nodes"]) == len(graph["nodes"])
        assert len(loaded["relationships"]) == len(graph["relationships"])

    def test_load_nonexistent(self, tmp_path, monkeypatch):
        """Loading non-existent exam returns None."""
        import exam_graph as eg
        monkeypatch.setattr(eg, "PRESETS_DIR", tmp_path)
        assert load_exam_graph(user_id="no_such_user") is None

    def test_load_corrupt_json_returns_none(self, tmp_path, monkeypatch):
        """A partially written / corrupt exam file must not raise."""
        import exam_graph as eg
        monkeypatch.setattr(eg, "PRESETS_DIR", tmp_path)
        (tmp_path / "exam_bad.json").write_text("{not valid json", encoding="utf-8")
        assert load_exam_graph(user_id="bad") is None

    def test_load_non_dict_returns_none(self, tmp_path, monkeypatch):
        """A valid-JSON non-dict file is not a graph and yields None."""
        import exam_graph as eg
        monkeypatch.setattr(eg, "PRESETS_DIR", tmp_path)
        (tmp_path / "exam_bad.json").write_text("[1, 2, 3]", encoding="utf-8")
        assert load_exam_graph(user_id="bad") is None

    def test_multiple_users(self, minimal_exam_data, tmp_path, monkeypatch):
        """Different users get separate exam graph files."""
        import exam_graph as eg
        monkeypatch.setattr(eg, "PRESETS_DIR", tmp_path)

        build_exam_graph(minimal_exam_data)
        save_exam_graph(build_exam_graph(minimal_exam_data), user_id="user_a")
        save_exam_graph(build_exam_graph(minimal_exam_data), user_id="user_b")

        assert (tmp_path / "exam_user_a.json").exists()
        assert (tmp_path / "exam_user_b.json").exists()


class TestExamGraphLabels:
    """Test that graph nodes use standard label formats."""

    def test_nodes_have_label_field(self, minimal_exam_data):
        """All nodes have a 'label' field."""
        graph = build_exam_graph(minimal_exam_data)
        for n in graph["nodes"]:
            assert "label" in n, f"Node missing label: {n.get('id')}"

    def test_relationships_have_type_field(self, minimal_exam_data):
        """All relationships have a 'type' field."""
        graph = build_exam_graph(minimal_exam_data)
        for rel in graph["relationships"]:
            assert "type" in rel
            assert rel["type"] in ("经历了", "包含")

    def test_nodes_have_properties(self, minimal_exam_data):
        """All nodes have 'properties' dict."""
        graph = build_exam_graph(minimal_exam_data)
        for n in graph["nodes"]:
            assert "properties" in n
            assert isinstance(n["properties"], dict)
