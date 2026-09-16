import json
import uuid

def create_check_item_json(exam_node, data):
    graph_json = {
        "nodes": [],
        "relationships": []
    }
    
    # Add exam node to nodes list
    exam_node_id = str(uuid.uuid4())
    graph_json["nodes"].append({
        "id": exam_node_id,
        "label": "体检大类",
        "properties": exam_node
    })

    for category_name, category_data in data.items():
        main_props = {"name": category_name}
        contain_complex = False

        # Handle category-level properties
        if isinstance(category_data, dict):
            if "结果提示" in category_data:
                if not isinstance(category_data["结果提示"], dict):
                    main_props["hint"] = category_data["结果提示"]
                    contain_complex = True
                elif "结果" in category_data["结果提示"]:
                    main_props["hint"] = category_data["结果提示"]["结果"]
                    category_data["结果提示"] = category_data["结果提示"]["结果"]
                    contain_complex = True
            if ("检查时间" in category_data) or ("检查日期" in category_data) or ("总检时间" in category_data):
                if isinstance(category_data.get("检查日期"), str):
                    main_props["check_date"] = category_data.get("检查时间", None) or category_data.get("检查日期") or category_data.get("总检时间")
                elif isinstance(category_data.get("检查日期"), dict):
                    main_props["check_date"] = category_data["检查日期"].get("结果", None)
                contain_complex = True
            if "检查医生" in category_data:
                if isinstance(category_data["检查医生"], str):
                    main_props["doctor"] = category_data["检查医生"] or None
                elif isinstance(category_data["检查医生"], dict):
                    main_props["doctor"] = category_data["检查医生"]["结果"] or None
                contain_complex = True
                del category_data["检查医生"]
            if "小结" in category_data:
                if isinstance(category_data["小结"], dict):
                    main_props["hint"] = category_data["小结"].get("结果")
                elif isinstance(category_data["小结"], str):
                    main_props["hint"] = category_data["小结"]
                del category_data["小结"]
            if ("检查所见" in category_data) and (isinstance(category_data.get("检查所见"), dict)) and ("结果" in category_data.get("检查所见")):
                category_data["检查所见"] = category_data["检查所见"]["结果"]
            if ("诊断提示" in category_data) and (isinstance(category_data.get("诊断提示"), dict)) and ("结果" in category_data.get("诊断提示")):
                category_data["诊断提示"] = category_data["诊断提示"]["结果"]

        # Create main category node
        main_category_node = {"name": category_name, **main_props}
        main_category_node_id = str(uuid.uuid4())
        graph_json["nodes"].append({
            "id": main_category_node_id,
            "label": "体检大类",
            "properties": main_category_node
        })
        graph_json["relationships"].append({
            "source": exam_node_id,
            "target": main_category_node_id,
            "type": "包含"
        })

        if isinstance(category_data, dict):
            for sub_key, sub_value in category_data.items():
                if not isinstance(sub_value, str):
                    if "小结" in sub_value and isinstance(sub_value["小结"], dict):
                        if "结果提示" in sub_value["小结"]:
                            sub_value["诊断提示"] = sub_value["小结"]["结果提示"]
                            del sub_value["小结"]
                        elif "结果" in sub_value["小结"]:
                            sub_value["小结"] = sub_value["小结"]["结果"]
                    if ("检查所见" in sub_value) and (isinstance(sub_value["检查所见"], dict)):
                        sub_value = sub_value["检查所见"]

                if isinstance(sub_value, dict):
                    for i, (sub_sub_key, sub_sub_value) in enumerate(sub_value.items()):
                        if isinstance(sub_sub_value, dict):
                            if (i == 0) and (sub_key != '项目名称') and (sub_key != '项目'):
                                sub_check_node = {"name": sub_key}
                                sub_check_node_id = str(uuid.uuid4())
                                graph_json["nodes"].append({
                                    "id": sub_check_node_id,
                                    "label": "子检查",
                                    "properties": sub_check_node
                                })
                                graph_json["relationships"].append({
                                    "source": main_category_node_id,
                                    "target": sub_check_node_id,
                                    "type": "包含"
                                })

                            sub_check_result_node = {
                                "name": sub_sub_key,
                                "result": sub_sub_value.get("结果") or sub_sub_value.get("检查所见") or sub_sub_value.get("检查结果") or sub_sub_value.get("result") or None,
                                "unit": sub_sub_value.get("单位", None) or sub_sub_value.get("unit") or None,
                                "reference": sub_sub_value.get("参考值", None) or sub_sub_value.get("参考区间") or sub_sub_value.get("参考范围") or None,
                                "abnormal": sub_sub_value.get("异常", None) or sub_sub_value.get("abnormal") or None,
                                "hint": sub_sub_value.get("诊断提示") or sub_sub_value.get("结果提示") or sub_sub_value.get("建议") or None,
                                "conclusion": sub_sub_value.get("小结", None) or None,
                                "date": sub_sub_value.get("检查日期", None) or None,
                                "doctor": sub_sub_value.get("检查医生", None) or None
                            }
                            sub_check_result_node_id = str(uuid.uuid4())
                            graph_json["nodes"].append({
                                "id": sub_check_result_node_id,
                                "label": "子检查项目结果",
                                "properties": sub_check_result_node
                            })
                            if (sub_key != '项目名称') and (sub_key != '项目'):
                                graph_json["relationships"].append({
                                    "source": sub_check_node_id,
                                    "target": sub_check_result_node_id,
                                    "type": "包含"
                                })
                            else:
                                graph_json["relationships"].append({
                                    "source": main_category_node_id,
                                    "target": sub_check_result_node_id,
                                    "type": "包含"
                                })
                        elif isinstance(sub_sub_value, list):
                            check_result_node = {
                                "name": sub_key,
                                "value": ",".join(sub_sub_value)
                            }
                            check_result_node_id = str(uuid.uuid4())
                            graph_json["nodes"].append({
                                "id": check_result_node_id,
                                "label": "检查结果",
                                "properties": check_result_node
                            })
                            graph_json["relationships"].append({
                                "source": main_category_node_id,
                                "target": check_result_node_id,
                                "type": "包含"
                            })
                        else:
                            sub_check_result_node = {
                                "name": sub_key,
                                "result": sub_value.get("结果") or sub_value.get("检查所见") or sub_value.get("result") or None,
                                "unit": sub_value.get("单位", None) or sub_value.get("unit") or None,
                                "reference": sub_value.get("参考值", None) or sub_value.get("参考区间") or sub_value.get("参考范围") or None,
                                "abnormal": sub_value.get("异常", None) or sub_value.get("abnormal") or None,
                                "conclusion": sub_value.get("小结", None) or None,
                                "hint": sub_value.get("诊断提示") or sub_value.get("结果提示") or sub_value.get("建议") or None,
                                "date": sub_value.get("检查日期", None) or None,
                                "doctor": sub_value.get("检查医生", None) or None
                            }
                            sub_check_result_node_id = str(uuid.uuid4())
                            graph_json["nodes"].append({
                                "id": sub_check_result_node_id,
                                "label": "子检查项目结果",
                                "properties": sub_check_result_node
                            })
                            graph_json["relationships"].append({
                                "source": main_category_node_id,
                                "target": sub_check_result_node_id,
                                "type": "包含"
                            })
                            break
                elif isinstance(sub_value, list):
                    for list_item in sub_value:
                        if isinstance(list_item, dict):
                            sub_check_result_node = {
                                "name": list_item.get("项目", sub_key),
                                "result": list_item.get("结果") or list_item.get("检查所见") or list_item.get("result") or None,
                                "unit": list_item.get("单位", None) or list_item.get("unit") or None,
                                "reference": list_item.get("参考值", None) or list_item.get("参考区间") or list_item.get("参考范围") or None,
                                "abnormal": list_item.get("异常", None) or list_item.get("abnormal") or None,
                                "conclusion": list_item.get("小结", None) or None,
                                "hint": list_item.get("诊断提示") or list_item.get("结果提示") or list_item.get("建议") or None,
                                "date": list_item.get("检查日期", None) or None,
                                "doctor": list_item.get("检查医生", None) or None
                            }
                            sub_check_result_node_id = str(uuid.uuid4())
                            graph_json["nodes"].append({
                                "id": sub_check_result_node_id,
                                "label": "子检查项目结果",
                                "properties": sub_check_result_node
                            })
                            graph_json["relationships"].append({
                                "source": main_category_node_id,
                                "target": sub_check_result_node_id,
                                "type": "包含"
                            })
                        elif isinstance(list_item, list):
                            check_result_node = {
                                "name": sub_key,
                                "value": ",".join(sub_value)
                            }
                            check_result_node_id = str(uuid.uuid4())
                            graph_json["nodes"].append({
                                "id": check_result_node_id,
                                "label": "检查结果",
                                "properties": check_result_node
                            })
                            graph_json["relationships"].append({
                                "source": main_category_node_id,
                                "target": check_result_node_id,
                                "type": "包含"
                            })
                            break
                        else:
                            check_result_node = {
                                "name": category_name,
                                "value": ",".join(sub_value)
                            }
                            check_result_node_id = str(uuid.uuid4())
                            graph_json["nodes"].append({
                                "id": check_result_node_id,
                                "label": "检查结果",
                                "properties": check_result_node
                            })
                            graph_json["relationships"].append({
                                "source": main_category_node_id,
                                "target": check_result_node_id,
                                "type": "包含"
                            })
                            break
                else:
                    if contain_complex:
                        result = category_data.get("检查所见", None) or category_data.get("结果") or None
                        if result is not None:
                            check_result_node = {
                                "name": category_name,
                                "result": result,
                                "hint": category_data.get("诊断提示", None) or category_data.get("结果提示") or category_data.get("建议") or category_data.get("小结")
                            }
                            check_result_node_id = str(uuid.uuid4())
                            graph_json["nodes"].append({
                                "id": check_result_node_id,
                                "label": "检查结果",
                                "properties": check_result_node
                            })
                            graph_json["relationships"].append({
                                "source": main_category_node_id,
                                "target": check_result_node_id,
                                "type": "包含"
                            })
                            break
                    elif not (category_data.get("结果") or category_data.get("检查所见") or category_data.get("检查医生")):
                        sub_check_result_node = {
                            "name": sub_key,
                            "result": sub_value
                        }
                        sub_check_result_node_id = str(uuid.uuid4())
                        graph_json["nodes"].append({
                            "id": sub_check_result_node_id,
                            "label": "子检查项目结果",
                            "properties": sub_check_result_node
                        })
                        graph_json["relationships"].append({
                            "source": main_category_node_id,
                            "target": sub_check_result_node_id,
                            "type": "包含"
                        })
                    else:
                        check_result_node = {
                            "name": category_data.get("项目") or category_name,
                            "result": category_data.get("结果") or category_data.get("检查所见") or category_data.get("result") or None,
                            "unit": category_data.get("单位", None) or category_data.get("unit") or None,
                            "reference": category_data.get("参考值", None) or category_data.get("参考区间") or category_data.get("参考范围") or None,
                            "abnormal": category_data.get("异常", None) or category_data.get("abnormal") or None,
                            "conclusion": category_data.get("小结", None) or None,
                            "hint": category_data.get("诊断提示") or category_data.get("结果提示") or category_data.get("建议") or None,
                            "date": category_data.get("检查日期", None) or None,
                            "doctor": category_data.get("检查医生", None) or None
                        }
                        # Remove any dictionary-type properties to prevent JSON serialization issues
                        for key, value in dict(check_result_node).items():
                            if isinstance(value, dict):
                                del check_result_node[key]
                        check_result_node_id = str(uuid.uuid4())
                        graph_json["nodes"].append({
                            "id": check_result_node_id,
                            "label": "检查结果",
                            "properties": check_result_node
                        })
                        graph_json["relationships"].append({
                            "source": main_category_node_id,
                            "target": check_result_node_id,
                            "type": "包含"
                        })
                        break
        elif isinstance(category_data, list):
            for list_item in category_data:
                if isinstance(list_item, dict):
                    sub_check_result_node = {
                        "name": list_item.get("项目", None),
                        "result": list_item.get("结果") or list_item.get("检查所见") or list_item.get("result") or None,
                        "unit": list_item.get("单位", None) or list_item.get("unit") or None,
                        "reference": list_item.get("参考值", None) or list_item.get("参考区间") or list_item.get("参考范围") or None,
                        "abnormal": list_item.get("异常", None) or list_item.get("abnormal") or None,
                        "conclusion": list_item.get("小结", None) or None,
                        "hint": list_item.get("诊断提示") or list_item.get("结果提示") or list_item.get("建议") or None,
                        "date": list_item.get("检查日期", None) or None,
                        "doctor": list_item.get("检查医生", None) or None
                    }
                    sub_check_result_node_id = str(uuid.uuid4())
                    graph_json["nodes"].append({
                        "id": sub_check_result_node_id,
                        "label": "子检查项目结果",
                        "properties": sub_check_result_node
                    })
                    graph_json["relationships"].append({
                        "source": main_category_node_id,
                        "target": sub_check_result_node_id,
                        "type": "包含"
                    })
                elif isinstance(list_item, list):
                    check_result_node = {
                        "name": sub_key,
                        "value": ",".join(sub_value)
                    }
                    check_result_node_id = str(uuid.uuid4())
                    graph_json["nodes"].append({
                        "id": check_result_node_id,
                        "label": "检查结果",
                        "properties": check_result_node
                    })
                    graph_json["relationships"].append({
                        "source": main_category_node_id,
                        "target": check_result_node_id,
                        "type": "包含"
                    })
                    break
                else:
                    check_result_node = {
                        "value": ",".join(category_data)
                    }
                    check_result_node_id = str(uuid.uuid4())
                    graph_json["nodes"].append({
                        "id": check_result_node_id,
                        "label": "检查结果",
                        "properties": check_result_node
                    })
                    graph_json["relationships"].append({
                        "source": main_category_node_id,
                        "target": check_result_node_id,
                        "type": "包含"
                    })
                    break

    return json.dumps(graph_json, ensure_ascii=False, indent=2)

# if __name__ == "__main__":
#     exam_node = {"name": "体检报告"}
#     data = {
#         "血液检查": {
#             "检查日期": "2023-10-01",
#             "检查医生": "张医生",
#             "血常规": {
#                 "白细胞计数": {"结果": "5.0", "单位": "10^9/L", "参考值": "4.0-10.0"},
#                 "红细胞计数": {"结果": "4.5", "单位": "10^12/L", "参考值": "4.0-5.5"}
#             },
#             "结果提示": "正常"
#         },
#         "影像检查": {
#             "检查所见": {"结果": "肺部正常"},
#             "小结": "无异常"
#         }
#     }
#     json_output = create_check_item_json(exam_node, data)
#     print(json_output)