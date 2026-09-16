import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

class GraphData:
    """管理图数据的类"""
    def __init__(self):
        self.nodes = []
        self.relationships = []
        self.node_id_map = {}  # 用于存储节点标识到ID的映射
    
    def create_node(self, *labels, **properties) -> str:
        """创建节点并返回节点ID"""
        node_id = str(uuid.uuid4())
        node = {
            "id": node_id,
            "label": list(labels),
            "properties": properties
        }
        self.nodes.append(node)
        return node_id
    
    def create_relationship(self, source_id: str, target_id: str, rel_type: str):
        """创建关系"""
        relationship = {
            "source": source_id,
            "target": target_id,
            "type": rel_type
        }
        self.relationships.append(relationship)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "nodes": self.nodes,
            "relationships": self.relationships
        }
    
    def save_to_json(self, filename: str):
        """保存到JSON文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    def load_from_json(self, filename: str):
        """从JSON文件加载"""
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.nodes = data.get("nodes", [])
                self.relationships = data.get("relationships", [])
    
    def merge_with(self, other_graph: 'GraphData'):
        """合并另一个图数据"""
        # 合并节点
        existing_ids = {node["id"] for node in self.nodes}
        for node in other_graph.nodes:
            if node["id"] not in existing_ids:
                self.nodes.append(node)
        
        # 合并关系
        existing_rels = {(rel["source"], rel["target"], rel["type"]) for rel in self.relationships}
        for rel in other_graph.relationships:
            rel_tuple = (rel["source"], rel["target"], rel["type"])
            if rel_tuple not in existing_rels:
                self.relationships.append(rel)

def get_user_node(graph_data: GraphData, name: str) -> str:
    """获取或创建用户节点"""
    # 查找现有用户节点
    for node in graph_data.nodes:
        if "受检人" in node["label"] and node["properties"].get("name") == name:
            return node["id"]
    
    # 如果不存在，创建新用户节点
    print(f"用户 {name} 未找到，将创建新用户节点。")
    return graph_data.create_node("受检人", name=name)

def get_user_events(graph_data: GraphData, name: str, link: set) -> List[str]:
    """获取用户的相关事件节点"""
    result = []
    
    # 首先找到用户节点
    user_id = None
    for node in graph_data.nodes:
        if "受检人" in node["label"] and node["properties"].get("name") == name:
            user_id = node["id"]
            break
    
    if not user_id:
        print(f"用户 {name} 未找到。")
        return result
    
    # 找到用户的事件节点
    user_event_ids = []
    for rel in graph_data.relationships:
        if rel["source"] == user_id and rel["type"] == "经历了":
            user_event_ids.append(rel["target"])
    
    # 过滤符合条件的事件
    for event_id in user_event_ids:
        for node in graph_data.nodes:
            if node["id"] == event_id and "事件" in node["label"]:
                json_link = node["properties"].get("json_link")
                if json_link and any(l in json_link for l in link):
                    result.append(event_id)
    
    if not result:
        print(f"用户 {name} 的相关事件未找到。")
    
    return result

def process_json_data(graph_data: GraphData, path: str, health_report_id: str):
    """
    处理单个JSON数据，创建对应的子图
    
    Args:
        graph_data: 图数据对象
        path: JSON文件路径
        health_report_id: 健康报告中心节点ID
    """
    with open(path, "r", encoding="utf-8") as file:
        json_data = json.load(file)
    filename = os.path.basename(path)

    # 获取大类名称
    category_name = json_data.get("环境因素分类", None) or json_data.get("系统名称", None) or filename.split(".")[0]

    if category_name == "健康总体评分":
        summary_id = graph_data.create_node("健康总体评分", "总结", name="健康总体评分", value=json_data.get("健康总体评分", None))
        graph_data.create_relationship(health_report_id, summary_id, "包含")
        return
    
    if category_name == "健康年龄评估":
        summary_id = graph_data.create_node("健康年龄评估", "总结", name="健康年龄评估", value=json_data.get("健康年龄", None))
        graph_data.create_relationship(health_report_id, summary_id, "包含")
        
        indicators_path = "dataset/report/指标提取.json"
        if os.path.exists(indicators_path):
            with open(indicators_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            create_age_indicators(graph_data, data, summary_id)
        return
    
    if category_name == "器官衰老指数":
        summary_id = graph_data.create_node("器官衰老指数", "总结", name="器官衰老指数")
        graph_data.create_relationship(health_report_id, summary_id, "包含")
        for key, value in json_data.items():
            result_id = graph_data.create_node("器官衰老指数", name=key, value=value)
            graph_data.create_relationship(summary_id, result_id, "包含")
        return
    
    if category_name == "⼼脑⾎管病⻛险评估":
        summary_id = graph_data.create_node("⼼脑⾎管病⻛险评估", "总结", name="⼼脑⾎管病⻛险评估")
        graph_data.create_relationship(health_report_id, summary_id, "包含")
        
        result_id = graph_data.create_node("心脑血管病风险评估结果", name="风险等级", value=json_data.get("风险等级", None))
        graph_data.create_relationship(summary_id, result_id, "包含")
        
        result_id = graph_data.create_node("心脑血管病风险评估结果", name="风险率", value=json_data.get("风险率", None))
        graph_data.create_relationship(summary_id, result_id, "包含")
        return
    
    if category_name in ["指标提取", "桑基图"]:
        return
    
    # 创建大类节点
    category_id = graph_data.create_node(
        "健康分项", category_name,
        name=category_name,
        score=json_data.get("综合评分", None) or json_data.get("分数", None),
        description=json_data.get("状态描述", None) or json_data.get("核心洞察", None),
        conclusion=json_data.get("小结", None) or json_data.get("功能状态", None),
        health_age=json_data.get("健康年龄", None)
    )
    graph_data.create_relationship(health_report_id, category_id, "包含")
        
    related_indicators = None
    if ("最相关指标" in json_data) or ("对应体检指标" in json_data):
        related_indicators = [item.get("指标", None) for item in json_data.get("最相关指标", json_data.get("对应体检指标"))]
        
    if ("体检报告指标" in json_data) or ("所有相关体检报告指标" in json_data) or ("所有体检指标" in json_data):
        exam_id = graph_data.create_node("相关体检报告指标", category_name, name="相关体检报告指标")
        graph_data.create_relationship(category_id, exam_id, "包含")
        create_medical_indicators(
            graph_data, 
            json_data.get("体检报告指标", json_data.get("所有相关体检报告指标", json_data.get("所有体检指标", None))), 
            exam_id, 
            related_indicators
        )

    if "功能医学视角" in json_data:
        functional_id = graph_data.create_node("功能医学视角", category_name, name="功能医学视角")
        graph_data.create_relationship(category_id, functional_id, "包含")
        create_functional_indicators(graph_data, json_data.get("功能医学视角", []), functional_id)

    if "问卷评估结果" in json_data:
        questionnaire_id = graph_data.create_node("问卷评估结果", category_name, name="问卷评估结果")
        graph_data.create_relationship(category_id, questionnaire_id, "包含")
        create_questionnaire_results(graph_data, json_data["问卷评估结果"], questionnaire_id)

    if "综合分析结果" in json_data:
        summary_id = graph_data.create_node("综合分析结果", category_name, name="综合分析结果")
        graph_data.create_relationship(category_id, summary_id, "包含")
        create_analysis_results(graph_data, json_data["综合分析结果"], summary_id)
    
    if ("干预方向" in json_data) or ("健康管理方向" in json_data):
        management_id = graph_data.create_node("健康管理方向", category_name, name="健康管理方向")
        graph_data.create_relationship(category_id, management_id, "包含")
        create_intervention_directions(graph_data, json_data.get("干预方向", json_data.get("健康管理方向", [])), management_id)

def create_age_indicators(graph_data: GraphData, data: dict, category_id: str):
    """创建健康年龄评估指标节点"""
    for key, value in data.items():
        indicator_id = graph_data.create_node(
            "健康年龄评估指标",
            name=key,
            value=value.get("数值", None),
            unit=value.get("单位", None)
        )
        graph_data.create_relationship(category_id, indicator_id, "包含")

def create_medical_indicators(graph_data: GraphData, indicators: list, category_id: str, related_indicators: Optional[list] = None):
    """创建体检指标节点"""
    if not indicators:
        return
        
    for indicator in indicators:
        relevance = "非常相关" if related_indicators and indicator.get("指标") in related_indicators else "相关"
        indicator_id = graph_data.create_node(
            "相关体检报告指标",
            name=indicator.get("指标", None),
            value=indicator.get("结果", None),
            relevance=relevance
        )
        graph_data.create_relationship(category_id, indicator_id, "包含")

def create_functional_indicators(graph_data: GraphData, indicators: list, category_id: str):
    """创建功能医学视角指标节点"""
    for indicator in indicators:
        functional_id = graph_data.create_node("功能医学视角", value=indicator)
        graph_data.create_relationship(category_id, functional_id, "包含")

def create_questionnaire_results(graph_data: GraphData, results: list, category_id: str):
    """创建问卷结果节点"""
    for result in results:
        # 解析问题和答案
        if "：" in result:
            question, answer = result.split("：", 1)
        elif ": " in result:
            question, answer = result.split(": ", 1)
        elif "?" in result:
            question, answer = result.split("?", 1)
        else:
            question = result
            answer = "未知"
        
        questionnaire_id = graph_data.create_node(
            "问卷评估结果",
            name=question.strip(),
            value=answer.strip()
        )
        graph_data.create_relationship(category_id, questionnaire_id, "包含")

def create_analysis_results(graph_data: GraphData, results: list, category_id: str):
    """创建分析结果节点"""
    for result_dict in results:
        for key, value in result_dict.items():
            analysis_id = graph_data.create_node("综合分析结果", name=key, result=value)
            graph_data.create_relationship(category_id, analysis_id, "分析结果")

def create_intervention_directions(graph_data: GraphData, directions: list, category_id: str):
    """创建干预方向节点"""
    for direction in directions:
        intervention_id = graph_data.create_node("健康管理方向", value=direction)
        graph_data.create_relationship(category_id, intervention_id, "包含")
        
def get_json_paths(folder_path: str) -> List[str]:
    """获取文件夹中所有JSON文件路径"""
    folder = Path(folder_path)
    json_paths = [str(file) for file in folder.rglob("*.json")]
    return json_paths

def main():
    folder_path = "dataset/report"
    json_files = get_json_paths(folder_path)
    
    # 创建图数据对象
    graph_data = GraphData()

    # 创建中心节点"健康报告"
    health_report_id = graph_data.create_node(
        "健康报告", "事件", 
        name="健康报告", 
        folder_link=folder_path,
        time="2025-05-20"
    )
    
    # 与用户图连接，这里需要指定报告生成所用的体检报告和问卷数据
    user_name = "个人1"
    link = {"dataset/res_all_all/009/009.json", "dataset/test_user_data/health_questionnaire.json"}
    
    # 加载现有的"个人1"用户数据（如果存在）
    user_data_file = f"{user_name}_graph_data.json"
    if os.path.exists(user_data_file):
        print(f"加载现有用户数据: {user_data_file}")
        graph_data.load_from_json(user_data_file)
    
    # 获取用户事件节点
    user_event_ids = get_user_events(graph_data, user_name, link)
    
    # 创建用户节点（如果不存在）
    user_id = get_user_node(graph_data, user_name)
    
    # 建立关系
    for event_id in user_event_ids:
        graph_data.create_relationship(event_id, health_report_id, "生成")
        
    graph_data.create_relationship(user_id, health_report_id, "拥有")
    
    # 处理所有JSON文件
    for path in json_files:
        print(f"处理文件: {path}")
        process_json_data(graph_data, path, health_report_id)

    # 保存结果
    output_file = "health_report_graph.json"
    graph_data.save_to_json(output_file)
    print(f"知识图谱构建完成！结果保存到: {output_file}")
    
    # # 如果需要合并到用户数据文件
    # if os.path.exists(user_data_file):
    #     # 重新加载用户数据并合并
    #     user_graph = GraphData()
    #     user_graph.load_from_json(user_data_file)
    #     user_graph.merge_with(graph_data)
    #     user_graph.save_to_json(user_data_file)
    #     print(f"数据已合并到用户文件: {user_data_file}")

if __name__ == "__main__":
    main()