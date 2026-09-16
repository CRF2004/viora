import json
from json_func import create_check_item_json
from tqdm import tqdm
import uuid

def build_json_graph(index, path):
    full_graph_json = {
        "nodes": [],
        "relationships": []
    }
    index = 1
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Preprocess data to extract relevant section
    if data.get("体检项目"):
        data = data["体检项目"]
    if data.get("体检结论与健康指导"):
        data = data["体检结论与健康指导"]
    if data.get("本次体检总结") and data.get("检验结果"):
        data = data["检验结果"]
    if data.get("检查综述"):
        data = data["检查综述"]
    if data.get("健康体检结果"):
        data = data["健康体检结果"]
    if data.get("小结"):
        del data["小结"]

    # Create patient node
    patient_node = {"name": f"个人{index}"}
    patient_node_id = str(uuid.uuid4())
    full_graph_json["nodes"].append({
        "id": patient_node_id,
        "label": "受检人",
        "properties": patient_node
    })

    # Create physical exam node
    physical_exam_node = {
        "time": "2025-5-10",
        "json_link": path
    }
    PERSON_INFO_FIELDS = ["体检信息", "个人信息"]
    for field in PERSON_INFO_FIELDS:
        if data.get(field):
            d = data[field]
            physical_exam_node.update({
                "name": d.get("姓名", None),
                "check_id": d.get("体检号", None),
                "age": d.get("年龄", None),
                "date": d.get("参检日期", None) or d.get("参检日期"),
                "gender": d.get("性别", None),
                "phone": d.get("联系电话", None),
                "identity": d.get("证件号码", None),
            })
            del data[field]
            
    physical_exam_node_id = str(uuid.uuid4())
    full_graph_json["nodes"].append({
        "id": physical_exam_node_id,
        "label": "体检",
        "properties": physical_exam_node
    })

    # Create relationship between patient and physical exam
    full_graph_json["relationships"].append({
        "source": patient_node_id,
        "target": physical_exam_node_id,
        "type": "经历了"
    })

    # Process check items and merge into full graph
    check_item_json = json.loads(create_check_item_json(physical_exam_node, data))
    full_graph_json["nodes"].extend(check_item_json["nodes"])
    full_graph_json["relationships"].extend(check_item_json["relationships"])

    return json.dumps(full_graph_json, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    index = 1   # 用户index
    json_output = build_json_graph(index=index, path="dataset/res_all_all/个人健康体检报告_2405160528_/个人健康体检报告_2405160528_.json")
    with open(f"个人{index}_graph.json", "w", encoding="utf-8") as f:
        f.write(json_output)
    print(f"个人{index}的图数据已生成并保存为个人{index}_graph.json")