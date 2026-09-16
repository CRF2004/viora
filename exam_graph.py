"""
exam_graph.py — Physical examination graph builder for Viora.

Parses a physical exam JSON report and builds a graph skeleton that
serves as the foundation for the health correlation graph.

Integrates the logic from:
- health_examination_graph_build(json).py (exam file parsing)
- json_func.py (check item → graph node conversion)

Usage:
    from exam_graph import build_exam_graph, ExamGraphError
    graph = build_exam_graph(exam_json_data, user_index=1)
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PRESETS_DIR = Path(__file__).resolve().parent / "presets_graph"


class ExamGraphError(Exception):
    """Raised when exam graph building fails."""


# ── Check Item → Graph Node Conversion ────────────────────────────────────

def _create_check_item_graph(exam_node: dict, data: dict) -> dict:
    """Convert raw exam check items into graph nodes and relationships.

    This is the core parsing engine from json_func.py, refactored for
    importability. Handles nested exam categories, sub-checks, results,
    doctors, dates, and abnormal flags.
    """
    graph = {"nodes": [], "relationships": []}

    exam_node_id = str(uuid.uuid4())
    graph["nodes"].append({
        "id": exam_node_id,
        "label": ["体检", "体检大类"],
        "properties": exam_node,
    })

    for category_name, category_data in data.items():
        main_props = {"name": category_name}
        contain_complex = False

        if isinstance(category_data, dict):
            # Extract category-level metadata
            _extract_category_meta(category_data, main_props)
            contain_complex = bool(
                main_props.get("hint") or main_props.get("check_date") or main_props.get("doctor")
            )

            # Flatten nested wrappers
            _flatten_nested_wrappers(category_data)

        # Create category node
        main_cat_node = {"name": category_name, **main_props}
        main_cat_id = str(uuid.uuid4())
        graph["nodes"].append({
            "id": main_cat_id,
            "label": ["体检", "体检大类"],
            "properties": main_cat_node,
        })
        graph["relationships"].append({
            "source": exam_node_id,
            "target": main_cat_id,
            "type": "包含",
        })

        # Process sub-items
        _process_category_items(graph, category_data, category_name,
                                main_cat_id, contain_complex)

    return graph


def _extract_category_meta(category_data: dict, main_props: dict):
    """Extract hint, date, doctor from category-level metadata.

    IMPORTANT: Extracted keys are removed from category_data to prevent
    premature loop termination in _process_dict_items.
    """
    if "结果提示" in category_data:
        hint = category_data["结果提示"]
        if isinstance(hint, dict):
            main_props["hint"] = hint.get("结果")
            category_data["结果提示"] = hint.get("结果")
        else:
            main_props["hint"] = hint
        del category_data["结果提示"]

    for date_key in ("检查时间", "检查日期", "总检时间"):
        if date_key in category_data:
            val = category_data[date_key]
            main_props["check_date"] = val.get("结果") if isinstance(val, dict) else val
            del category_data[date_key]
            break

    if "检查医生" in category_data:
        doc = category_data["检查医生"]
        main_props["doctor"] = doc.get("结果") if isinstance(doc, dict) else doc
        del category_data["检查医生"]

    if "小结" in category_data:
        s = category_data["小结"]
        main_props["hint"] = s.get("结果") if isinstance(s, dict) else s
        del category_data["小结"]


def _flatten_nested_wrappers(category_data: dict):
    """Flatten deeply nested result wrappers."""
    for key in ("检查所见", "诊断提示"):
        if key in category_data and isinstance(category_data[key], dict):
            if "结果" in category_data[key]:
                category_data[key] = category_data[key]["结果"]


def _process_category_items(graph: dict, category_data, category_name: str,
                            parent_id: str, contain_complex: bool):
    """Process sub-items within a category."""
    if isinstance(category_data, dict):
        _process_dict_items(graph, category_data, category_name, parent_id, contain_complex)
    elif isinstance(category_data, list):
        _process_list_items(graph, category_data, category_name, parent_id)


def _process_dict_items(graph: dict, data: dict, category_name: str,
                        parent_id: str, contain_complex: bool):
    """Process dict-typed category items."""
    for sub_key, sub_value in data.items():
        # Preprocess nested wrappers
        if isinstance(sub_value, dict):
            _preprocess_sub_value(sub_value)

        if isinstance(sub_value, dict):
            _add_dict_sub_item(graph, sub_value, sub_key, category_name, parent_id)
        elif isinstance(sub_value, list):
            _add_list_sub_items(graph, sub_value, sub_key, category_name, parent_id)
        elif contain_complex and not _has_complex_fields(data):
            _add_simple_result(graph, category_name, data, parent_id)
            break
        elif not _is_data_field(data, sub_key):
            _add_kv_result(graph, sub_key, sub_value, parent_id)
        else:
            _add_standard_result(graph, sub_key, data, parent_id)
            break


def _preprocess_sub_value(sub_value: dict):
    """Preprocess nested wrappers in sub-values."""
    if "小结" in sub_value and isinstance(sub_value["小结"], dict):
        if "结果提示" in sub_value["小结"]:
            sub_value["诊断提示"] = sub_value["小结"]["结果提示"]
            del sub_value["小结"]
        elif "结果" in sub_value["小结"]:
            sub_value["小结"] = sub_value["小结"]["结果"]
    if "检查所见" in sub_value and isinstance(sub_value["检查所见"], dict):
        sub_value = sub_value["检查所见"]


def _has_complex_fields(data: dict) -> bool:
    """Check if data has complex result fields."""
    return bool(data.get("结果") or data.get("检查所见") or data.get("检查医生"))


def _is_data_field(data: dict, key: str) -> bool:
    """Check if key is a data field (vs structural key)."""
    return key in ("结果", "检查所见", "检查医生", "检查日期", "单位",
                   "参考值", "异常", "参考区间", "参考范围", "诊断提示",
                   "结果提示", "建议", "小结", "项目", "项目名称")


def _add_dict_sub_item(graph: dict, sub_value: dict, sub_key: str,
                       category_name: str, parent_id: str):
    """Add a dict-typed sub-item to the graph."""
    for i, (sub_sub_key, sub_sub_value) in enumerate(sub_value.items()):
        if isinstance(sub_sub_value, dict):
            # Create sub-check node for first dict entry with non-project key
            if i == 0 and sub_key not in ("项目名称", "项目"):
                sub_node = {"name": sub_key}
                sub_id = str(uuid.uuid4())
                graph["nodes"].append({
                    "id": sub_id, "label": ["体检", "子检查"],
                    "properties": sub_node,
                })
                graph["relationships"].append({
                    "source": parent_id, "target": sub_id, "type": "包含",
                })
            else:
                sub_id = parent_id

            result_node = _build_result_node(sub_sub_key, sub_sub_value)
            result_id = str(uuid.uuid4())
            graph["nodes"].append({
                "id": result_id, "label": ["体检", "子检查项目结果"],
                "properties": result_node,
            })
            graph["relationships"].append({
                "source": sub_id, "target": result_id, "type": "包含",
            })
            return

    # Fallback: add as simple KV
    _add_standard_result(graph, sub_key, sub_value, parent_id)


def _add_list_sub_items(graph: dict, sub_value: list, sub_key: str,
                        category_name: str, parent_id: str):
    """Add list-typed sub-items to the graph."""
    for list_item in sub_value:
        if isinstance(list_item, dict):
            result_node = _build_result_node(
                list_item.get("项目", sub_key), list_item
            )
            result_id = str(uuid.uuid4())
            graph["nodes"].append({
                "id": result_id, "label": ["体检", "子检查项目结果"],
                "properties": result_node,
            })
            graph["relationships"].append({
                "source": parent_id, "target": result_id, "type": "包含",
            })
        elif isinstance(list_item, list):
            graph["nodes"].append({
                "id": str(uuid.uuid4()), "label": ["体检", "检查结果"],
                "properties": {"name": sub_key, "value": ",".join(str(x) for x in sub_value)},
            })
            break
        else:
            graph["nodes"].append({
                "id": str(uuid.uuid4()), "label": ["体检", "检查结果"],
                "properties": {"name": category_name, "value": ",".join(str(x) for x in sub_value)},
            })
            break


def _add_simple_result(graph: dict, category_name: str, data: dict, parent_id: str):
    """Add a simple flat result."""
    result = data.get("检查所见") or data.get("结果")
    if result is not None:
        node = {"name": category_name, "result": result,
                "hint": data.get("诊断提示") or data.get("结果提示") or data.get("建议") or data.get("小结")}
        result_id = str(uuid.uuid4())
        graph["nodes"].append({
            "id": result_id, "label": ["体检", "检查结果"],
            "properties": node,
        })
        graph["relationships"].append({
            "source": parent_id, "target": result_id, "type": "包含",
        })


def _add_kv_result(graph: dict, key: str, value, parent_id: str):
    """Add a simple key-value result."""
    result_id = str(uuid.uuid4())
    graph["nodes"].append({
        "id": result_id, "label": ["体检", "子检查项目结果"],
        "properties": {"name": key, "result": value},
    })
    graph["relationships"].append({
        "source": parent_id, "target": result_id, "type": "包含",
    })


def _add_standard_result(graph: dict, sub_key: str, data: dict, parent_id: str):
    """Add a standard formatted result node."""
    result_node = _build_result_node(
        data.get("项目") or data.get("项目名称") or sub_key, data
    )
    # Remove dict-typed properties (JSON-safe)
    result_node = {k: v for k, v in result_node.items() if not isinstance(v, dict)}
    result_id = str(uuid.uuid4())
    graph["nodes"].append({
        "id": result_id, "label": ["体检", "检查结果"],
        "properties": result_node,
    })
    graph["relationships"].append({
        "source": parent_id, "target": result_id, "type": "包含",
    })


def _build_result_node(name: str, data: dict) -> dict:
    """Build a standardized result node from raw data."""
    return {
        "name": str(name),
        "result": (
            data.get("结果") or data.get("检查所见")
            or data.get("检查结果") or data.get("result")
        ),
        "unit": data.get("单位") or data.get("unit"),
        "reference": data.get("参考值") or data.get("参考区间") or data.get("参考范围"),
        "abnormal": data.get("异常") or data.get("abnormal"),
        "hint": data.get("诊断提示") or data.get("结果提示") or data.get("建议"),
        "conclusion": data.get("小结"),
        "date": data.get("检查日期"),
        "doctor": data.get("检查医生"),
    }


# ── Physical Exam → Full Graph ────────────────────────────────────────────

# Fields that contain personal information (extracted to exam node, removed from data)
_PERSON_INFO_FIELDS = ["体检信息", "个人信息"]

# Top-level wrapper keys that should be unwrapped
_TOP_LEVEL_WRAPPERS = [
    "体检项目", "体检结论与健康指导", "检验结果",
    "检查综述", "健康体检结果",
]


def build_exam_graph(
    exam_data: dict,
    user_index: int = 1,
) -> dict:
    """Build a full graph from physical exam JSON data.

    Args:
        exam_data: Parsed JSON dict from a physical exam report.
        user_index: User index for labeling (default 1).

    Returns:
        Graph dict with {"nodes": [...], "relationships": [...]}

    Raises:
        ExamGraphError: If the exam data is empty or unparseable.
    """
    if not exam_data or not isinstance(exam_data, dict):
        raise ExamGraphError("体检数据为空或格式不正确")

    data = dict(exam_data)  # shallow copy to avoid mutating input
    graph = {"nodes": [], "relationships": []}

    # ── Unwrap top-level container keys ──
    for wrapper in _TOP_LEVEL_WRAPPERS:
        if wrapper in data:
            data = data[wrapper]
    if "本次体检总结" in data and "检验结果" in data:
        data = data["检验结果"]

    # ── Remove summary if present ──
    data.pop("小结", None)

    # ── Create patient node ──
    patient_node = {"name": f"个人{user_index}"}
    patient_id = str(uuid.uuid4())
    graph["nodes"].append({
        "id": patient_id,
        "label": ["受检人", "个人"],
        "properties": patient_node,
    })

    # ── Create physical exam node ──
    exam_node = {
        "time": data.get("参检日期") or data.get("检查日期") or "",
        "type": "体检报告",
    }

    # Extract personal info and remove from data
    for field in _PERSON_INFO_FIELDS:
        if field in data:
            info = data[field]
            if isinstance(info, dict):
                exam_node.update({
                    "name": info.get("姓名"),
                    "check_id": info.get("体检号"),
                    "age": info.get("年龄"),
                    "date": info.get("参检日期") or info.get("检查日期"),
                    "gender": info.get("性别"),
                    "phone": info.get("联系电话"),
                    "identity": info.get("证件号码"),
                })
            del data[field]

    exam_id = str(uuid.uuid4())
    graph["nodes"].append({
        "id": exam_id,
        "label": ["体检", "事件"],
        "properties": exam_node,
    })
    graph["relationships"].append({
        "source": patient_id,
        "target": exam_id,
        "type": "经历了",
    })

    # ── Process check items ──
    check_graph = _create_check_item_graph(exam_node, data)
    graph["nodes"].extend(check_graph["nodes"])
    graph["relationships"].extend(check_graph["relationships"])

    return graph


def save_exam_graph(graph: dict, user_id: str = "default") -> str:
    """Save an exam graph to the presets directory.

    Args:
        graph: Graph dict from build_exam_graph().
        user_id: User identifier for filename.

    Returns:
        Filename of the saved graph.
    """
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"exam_{user_id}.json"
    filepath = PRESETS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    logger.info("Exam graph saved: %s (%d nodes, %d edges)",
                filename, len(graph["nodes"]), len(graph["relationships"]))
    return filename


def load_exam_graph(user_id: str = "default") -> Optional[dict]:
    """Load a previously saved exam graph.

    Returns None if no exam graph exists for this user, or if the stored file
    is corrupt / not a graph object (legacy or partially written data must not
    crash the exam route).
    """
    filepath = PRESETS_DIR / f"exam_{user_id}.json"
    if not filepath.exists():
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load exam graph %s: %s", filepath, e)
        return None
    return data if isinstance(data, dict) else None
