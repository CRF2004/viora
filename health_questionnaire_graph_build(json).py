import json
import re
import uuid

# 陈述句转换函数
def convert_to_statement(name, value):
    """将问卷问题和答案转换为简短陈述句"""
    
    # 如果value是列表，处理列表中的值
    if isinstance(value, list):
        if len(value) == 1:
            value = value[0]
        else:
            # 多选情况，组合所有选项
            value = "、".join(value)
    
    # 基本信息处理
    if "年龄" in name:
        return f"年龄{value}岁"
    elif "性别" in name:
        gender = "男性" if "男" in value else "女性"
        return gender
    elif "身高" in name:
        return f"身高{value}cm"
    elif "体重" in name:
        return f"体重{value}kg"
    elif "现居住地区" in name:
        return f"居住在{value}"
    elif "腰围" in name:
        return f"腰围{value}cm"
    
    # 日常状态处理
    elif "疲劳" in name or "精力不足" in name:
        return "经常感到疲劳" if "是" in value else "精力充沛"
    elif "记忆力" in name or "注意力" in name:
        return "记忆力下降注意力不集中" if "是" in value else "记忆力注意力正常"
    elif "情绪焦虑" in name or "睡眠" in name:
        return "情绪焦虑睡眠不佳" if "是" in value else "情绪稳定睡眠良好"
    elif "食欲下降" in name or "味觉" in name:
        return "食欲味觉下降" if "是" in value else "食欲味觉正常"
    elif "感冒" in name or "感染" in name:
        return "容易感冒感染" if "是" in value else "免疫力正常"
    
    # 皮肤头发指甲
    elif "脱发" in name or "头发干枯" in name:
        return "脱发头发干枯" if "是" in value else "头发健康"
    elif "指甲" in name:
        return "指甲易断裂有异常" if "是" in value else "指甲健康"
    elif "皮肤干燥" in name or "脱皮" in name:
        return "皮肤干燥粗糙" if "是" in value else "皮肤健康"
    elif "伤口愈合" in name or "淤青" in name:
        return "伤口愈合慢易淤青" if "是" in value else "伤口愈合正常"
    
    # 消化系统
    elif "腹胀" in name or "消化不良" in name:
        return "经常腹胀消化不良" if "是" in value else "消化功能正常"
    elif "便秘" in name or "腹泻" in name:
        return "有便秘或腹泻问题" if "是" in value else "排便正常"
    elif "口腔溃疡" in name:
        return "口腔溃疡反复出现" if "是" in value else "口腔健康"
    
    # 肌肉骨骼
    elif "肌肉痉挛" in name or "抽筋" in name:
        return "有肌肉痉挛抽筋" if "是" in value else "肌肉功能正常"
    elif "肌肉酸痛" in name or "无力" in name:
        return "肌肉酸痛无力" if "是" in value else "肌肉状态良好"
    elif "关节疼痛" in name or "骨质疏松" in name:
        return "关节疼痛或骨质疏松" if "是" in value else "骨骼关节健康"
    
    # 生理阶段
    elif "目前处于" in name:
        if "无以上情况" in value:
            return "非特殊生理阶段"
        else:
            return f"处于{value}"
    
    # 眼睛问题
    elif "眼睛干涩" in name or "疲劳" in name:
        return "眼睛干涩疲劳" if "是" in value else "眼睛状态良好"
    elif "夜间视力" in name or "畏光" in name:
        return "夜间视力差畏光" if "是" in value else "视力正常"
    elif "视物模糊" in name or "老花眼" in name:
        return "视物模糊老花加重" if "是" in value else "视力清晰"
    
    # 骨骼关节
    elif "被诊断过骨质疏松" in name:
        return "确诊骨质疏松" if "是" in value else "无骨质疏松"
    elif "关节酸痛" in name or "僵硬" in name:
        return "关节酸痛僵硬" if "是" in value else "关节灵活"
    elif "骨裂" in name or "骨折" in name:
        return "有骨裂骨折史" if "是" in value else "无骨折史"
    
    # 心血管
    elif "高血压" in name or "心悸" in name or "高胆固醇" in name:
        return "有心血管问题" if "是" in value else "心血管健康"
    elif "心慌" in name or "胸闷" in name or "乏力" in name:
        return "有心慌胸闷乏力" if "是" in value else "心脏功能正常"
    elif "收缩压" in name:
        return f"收缩压{value}mmHg"
    elif "舒张压" in name:
        return f"舒张压{value}mmHg"
    elif "以下一种或多种情况" in name and ("心血管" in name or "动脉" in value or "心肌" in value or "心律" in value):
        if "以上均无" in value:
            return "无心血管疾病"
        else:
            return f"有{value}"
    
    # 饮食习惯
    elif "饮食类型" in name:
        return f"饮食{value}"
    elif "用餐方式" in name:
        if value == "对半开":
            return f"用餐方式为 外食/外卖 和 自烹/家庭用餐 对半开"
        return f"用餐{value}"
    elif "口味偏好" in name:
        return f"口味{value}"
    elif "进餐规律" in name:
        regularity = "规律" if "规律" in value else "不规律"
        return f"进餐{regularity}"
    elif "节食" in name or "断食" in name or "控制食量" in name:
        return "经常控制食量" if "是" in value else "饮食量正常"
    
    # 膳食结构
    elif "主食" in name and "米类" in name:
        return f"主食摄入{value}"
    elif "粗纤维食物" in name:
        return f"粗纤维食物{value}"
    elif "肉类" in name and "猪、牛、羊" in name:
        return f"肉类摄入{value}"
    elif "动物内脏" in name:
        return f"动物内脏{value}"
    elif "淡水鱼虾" in name:
        return f"淡水鱼虾{value}"
    elif "海产品" in name:
        return f"海产品{value}"
    elif "鸡蛋" in name:
        return f"鸡蛋{value}"
    elif "豆及大豆制品" in name:
        return f"豆制品{value}"
    elif "奶或奶制品" in name:
        return f"奶制品{value}"
    elif "绿叶蔬菜" in name:
        return f"绿叶蔬菜{value}"
    elif "新鲜水果" in name:
        return f"新鲜水果{value}"
    elif "以下哪些食物您经常吃" in name or "您经常吃以下哪些食物" in name:
        if "以上几乎不吃" in value:
            return "很少吃不健康食品"
        else:
            return f"经常吃{value}"
    
    # 饮水习惯
    elif "每天饮用几杯水" in name:
        return f"每天饮水{value}"
    elif "最常喝什么" in name:
        return f"主要饮品是{value}"
    
    # 烟酒
    elif "吸烟" in name:
        if "不吸" in value:
            return "不吸烟"
        else:
            return f"吸烟{value}"
    elif "喝酒" in name:
        if "不喝" in value:
            return "不喝酒"
        else:
            return f"喝酒{value}"
    elif "二手烟" in name:
        return "经常接触二手烟" if "是" in value else "很少接触二手烟"
    
    # 运动睡眠
    elif "运动频率" in name:
        return f"运动{value}"
    elif "运动强度" in name:
        return f"运动强度{value}"
    elif "睡眠时长" in name:
        return f"每晚睡眠{value}小时"
    elif "睡眠质量" in name:
        return f"睡眠质量{value}"
    
    # 营养补充剂
    elif "营养补充剂" in name:
        if not value or "否" in value:
            return "不服用营养补充剂"
        else:
            supplement = str(value).replace("是，请注明种类：", "").strip()
            if supplement:
                return f"服用{supplement}补充剂"
            else:
                return "服用营养补充剂但未具体说明"
    
    # 疾病史
    elif "慢性疾病" in name:
        if "无" in value:
            return "无慢性疾病"
        else:
            return f"有{value}"
    
    # 用药史
    elif "长期服用" in name and "药物" in name:
        if "无" in value:
            return "无长期用药"
        else:
            return f"长期服用{value}"
    
    # 过敏史
    elif "过敏" in name:
        if "无过敏" in value or "不知道" in value:
            return "无明确过敏史"
        else:
            return f"对{value}过敏"
    
    # 营养素缺乏
    elif "营养素" in name and "缺乏" in name:
        return f"缺乏{value}"
    
    # 家族史
    elif "直系亲属" in name:
        if "以上均无" in value:
            return "无家族病史"
        else:
            return f"家族有{value}"
    
    # 各种症状处理
    elif "餐后犯困" in value:
        return "有餐后犯困现象"
    elif "易胖易肿" in value:
        return "体质易胖易肿"
    elif "怕冷" in value:
        return "体质怕冷手脚冰凉"
    elif "超重" in value or "肥胖" in value:
        return "超重肥胖腰腹脂肪多"
    elif "月经" in value:
        return "月经周期量有变化"
    elif "性欲" in value or "性生活" in value:
        return "性欲性生活减少"
    elif "性功能" in value:
        return "性功能下降"
    elif "增生" in value or "囊肿" in value or "结节" in value:
        return "身体有增生囊肿结节"
    elif "胃肠胀气" in value or "消化不良" in value:
        return "胃肠胀气消化不良"
    elif "慢性胃炎" in value:
        return "慢性胃炎"
    elif "溃疡" in value:
        return "胃或十二指肠溃疡"
    elif "幽门螺旋杆菌" in value:
        return "幽门螺旋杆菌感染"
    elif "过敏性鼻炎" in value:
        return "过敏性鼻炎"
    elif "哮喘" in value:
        return "哮喘"
    elif "荨麻疹" in value:
        return "荨麻疹"
    elif "湿疹" in value:
        return "湿疹"
    elif "皮肤过敏" in value:
        return "皮肤过敏"
    elif "桥本氏" in value:
        return "桥本氏甲状腺炎"
    elif "类风湿" in value:
        return "类风湿关节炎"
    elif "强直性脊柱炎" in value:
        return "强直性脊柱炎"
    elif "以上均无" in value:
        return "无相关症状"
    
    # 处理多种症状的情况
    elif "一种或多种症状" in name:
        if "以上均无" in value:
            return "无相关症状"
        else:
            return f"有{value}"
    
    # 默认处理
    else:
        # 简单的默认转换
        clean_name = re.sub(r'[？?：:\(\)（）\[\]【】□]', '', name)
        clean_value = str(value)
        return f"{clean_name}_{clean_value}"

def generate_json_graph(data, user_path, user_name="个人1"):
    """生成JSON格式的图数据"""
    nodes = []
    relationships = []
    user_id = None
    # 查询用户的graph(json)
    with open(user_path, "r", encoding="utf-8") as f:
        user_node_data = json.load(f)["nodes"]
    for node in user_node_data:
        if "name" in node["properties"]:
            if node["properties"]["name"] == user_name:
                user_id = node["id"]
                break
    if not user_id:
        raise ValueError(f"未找到用户 {user_name} 的节点")
    
    # 创建问卷事件节点
    questionnaire_id = str(uuid.uuid4())
    questionnaire_node = {
        "id": questionnaire_id,
        "label": ["事件","健康问卷"],
        "properties": {
            "name": "健康状况问卷",
            "json_link": "dataset/test_user_data/health_questionnaire.json",
            "time": "2025-05-20"    # 待确定
        }
    }
    nodes.append(questionnaire_node)
    
    # 用户-问卷关系
    user_questionnaire_rel = {
        "source": user_id,
        "target": questionnaire_id,
        "type": "经历了"
    }
    relationships.append(user_questionnaire_rel)
    
    def process_data_recursive(data_dict, parent_id):
        """递归处理数据"""
        for key, value in data_dict.items():
            # 创建维度节点
            category_id = str(uuid.uuid4())
            category_node = {
                "id": category_id,
                "label": ["维度", key],
                "properties": {
                    "value": key
                }
            }
            nodes.append(category_node)
            
            # 连接到父节点
            if parent_id:
                rel = {
                    "source": parent_id,
                    "target": category_id,
                    "type": "包含"
                }
                relationships.append(rel)
            
            # 处理值
            if isinstance(value, dict):
                # 处理字典中的每个问题-答案对
                for question, answer in value.items():
                    # 转换为陈述句
                    statement = convert_to_statement(question, answer)
                    
                    # 创建实体节点
                    entity_id = str(uuid.uuid4())
                    entity_node = {
                        "id": entity_id,
                        "label": ["问卷结果", key],
                        "properties": {
                            "value": statement
                        }
                    }
                    nodes.append(entity_node)
                    
                    # 连接到类别节点
                    rel = {
                        "source": category_id,
                        "target": entity_id,
                        "type": "包含"
                    }
                    relationships.append(rel)
            else:
                # 如果value不是字典，直接处理
                statement = convert_to_statement(key, value)
                entity_id = str(uuid.uuid4())
                entity_node = {
                    "id": entity_id,
                    "label": ["问卷结果", key],
                    "properties": {
                        "value": statement
                    }
                }
                nodes.append(entity_node)
                
                rel = {
                    "source": category_id,
                    "target": entity_id,
                    "type": "包含"
                }
                relationships.append(rel)
    
    # 开始处理数据
    process_data_recursive(data, questionnaire_id)
    
    return {
        "nodes": nodes,
        "relationships": relationships
    }

if __name__ == "__main__":
    # 读取问卷数据
    with open("dataset/test_user_data/health_questionnaire.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 生成问卷JSON图数据
    q_graph_data = generate_json_graph(data, "个人1_graph.json", "个人1")
    
    # with open("个人1_questionnaire_graph.json", "w", encoding="utf-8") as f:
    #     json.dump(q_graph_data, f, ensure_ascii=False, indent=4)
    
    print("JSON图数据生成完成！")
    print(f" {len(q_graph_data['nodes'])} 个节点和 {len(q_graph_data['relationships'])} 个关系")

    with open("个人1_graph.json", "r", encoding="utf-8") as f:
        user_graph = json.load(f)
    user_graph["nodes"].extend(q_graph_data["nodes"])
    user_graph["relationships"].extend(q_graph_data["relationships"])
    with open("个人1_graph.json", "w", encoding="utf-8") as f:
        json.dump(user_graph, f, ensure_ascii=False, indent=4)
    print("个人1的图数据已更新并保存为个人1_graph.json")