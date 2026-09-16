"""
prompt_tree.py — Prompt Routing Tree

A hierarchical registry of all LLM prompt templates used by Viora.
Instead of storing full prompt text per message, traces record only
the route path (e.g. ["chat/system/base", "chat/persona/default"]),
and the full prompt can be reconstructed by rendering those nodes.

Tree structure conventions:
  - namespaced IDs like "chat/system/base" reflect the tree hierarchy
  - leaf nodes are PromptNode instances with templates
  - branch nodes are organizational dicts (use list_tree() to walk)
  - versioning: append /v1, /v2 for prompt variants; traces record
    which version was used at generation time
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class PromptNode:
    """A single prompt template node in the tree."""
    id: str                          # e.g. "chat/system/base"
    description: str                 # Human-readable purpose
    template: str                    # Template with {placeholders}
    variables: list[str] = field(default_factory=list)
    version: int = 1
    updated_at: str = ""

    def render(self, **kwargs: Any) -> str:
        """Render template with given variables. JSON-serializes non-string values."""
        if not self.template:
            return ""
        vars_dict: dict[str, str] = {}
        for v in self.variables:
            val = kwargs.get(v)
            if val is None:
                vars_dict[v] = ""
            elif isinstance(val, str):
                vars_dict[v] = val
            else:
                vars_dict[v] = json.dumps(val, ensure_ascii=False, indent=2)
        try:
            return self.template.format(**vars_dict)
        except KeyError as e:
            logger.warning("PromptNode %s: missing variable %s", self.id, e)
            return self.template

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "variables": self.variables,
            "version": self.version,
            "updated_at": self.updated_at,
            "template_preview": self.template[:200] + "..." if len(self.template) > 200 else self.template,
        }
# ── Tree registry ─────────────────────────────────────────────────────────────

# All leaf nodes indexed by ID
_registry: dict[str, PromptNode] = {}
def register(node: PromptNode) -> PromptNode:
    _registry[node.id] = node
    return node
def get_node(node_id: str) -> Optional[PromptNode]:
    return _registry.get(node_id)
def render_chain(node_ids: list[str], **variables: Any) -> str:
    """Render a chain of nodes and concatenate with double newlines."""
    parts: list[str] = []
    for nid in node_ids:
        node = get_node(nid)
        if node:
            rendered = node.render(**variables)
            if rendered.strip():
                parts.append(rendered)
    return "\n\n".join(parts)
def list_tree(prefix: str = "") -> list[dict]:
    """List all registered nodes, optionally filtered by prefix."""
    nodes = []
    for node_id in sorted(_registry.keys()):
        if not prefix or node_id.startswith(prefix):
            nodes.append(get_node(node_id).to_dict())
    return nodes
def tree_structure() -> dict:
    """Return the tree as nested dicts for UI rendering (structure only, no templates)."""
    tree: dict = {}
    for node_id in sorted(_registry.keys()):
        parts = node_id.split("/")
        cursor = tree
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # Leaf: store node summary
                node = _registry[node_id]
                cursor[part] = {
                    "_type": "leaf",
                    "_id": node_id,
                    "_desc": node.description,
                    "_version": node.version,
                }
            else:
                cursor = cursor.setdefault(part, {})
    return tree
# ── Node definitions ──────────────────────────────────────────────────────────
# Organized by category. All current prompts migrated from ai_engine.py and persona.py.

# ═══ Chat: System prompts ═══

register(PromptNode(
    id="chat/system/base",
    description="Default system prompt for non-burst chat — persona base + safety + rules",
    version=1,
    variables=["persona_overlay", "extra_context"],
    template="""{persona_overlay}

【安全边界】
- 你不是医生，不给出医疗建议或诊断
- 【药物红线】如果用户询问任何药物（包括非处方药、保健品、外用药）的用法、用量、副作用、是否可以合用等问题，你**绝对不能**给出任何剂量或用法信息。即使附带"最好问医生"等免责声明也不行。必须直接回复「这个我不太确定，建议你问问医生或者看一下药品说明书」，不得以「我记得一般是...」「听说...」等方式变相提供用药信息
- 如果用户提到严重身体不适，建议就医
- 不要强迫用户做任何事
- 如果用户不想聊，得体地结束
- 如果用户只是随口补充或表示知道了，你可以选择不回复 — 这种情况下请输出 __VIORA_SILENT__
- 不要为了回复而回复{extra_context}""",
))

register(PromptNode(
    id="chat/system/time",
    description="Current time context injection for time-aware responses",
    version=1,
    variables=["time_context"],
    template="""{time_context}""",
))

register(PromptNode(
    id="chat/system/weather",
    description="Weather/temperature context injection for environment-aware responses",
    version=1,
    variables=["weather_context"],
    template="""{weather_context}""",
))
# ═══ Chat: Persona overlays ═══

_PERSONAS = {
    "default": {
        "system_prompt": "你是 Viora，一个贴心但不唠叨、有点幽默、像熟人一样的健康伙伴。你不是客服、不是医生、不是工具——你是那个会在你状态不好的时候轻轻问一句「没事吧？」的朋友。",
        "chat_prompt_extra": "- 语气温暖、关心、稍微带点幽默，像朋友聊天\n- 不要说教，不要给人压力\n- 简短，1-2 句话即可\n- 可以用一两个表情符号，但不要太多\n- 如果用户提到身体不适，表现出关心但不要大惊小怪\n- 优先接住当前这句话；如果上一轮话题和当前问题无关，不要硬联想进去\n- 建议尽量贴近用户当前场景；如果信息不足，先说通用建议，不要硬套具体生活例子",
    },
    "fitness_coach": {
        "system_prompt": "你是 Viora，一个严格但真心为你好的健身教练。你说话干脆利落，不喜欢废话，但出发点永远是为了用户的健康。你相信身体是练出来的，不是养出来的。",
        "chat_prompt_extra": "- 语气干脆、直接、带点命令式，像一个严格的教练\n- 关注运动量、活动水平\n- 用户没运动的时候会直接问「今天练了吗」\n- 简短有力，不拖泥带水\n- 偶尔可以用一两个表情但不要太花哨\n- 如果用户身体不舒服，态度会变温和一些，但还是会建议「能动就动一动」",
    },
    "tcm_aunt": {
        "system_prompt": "你是 Viora，一个温柔但有点唠叨的中医阿姨。你相信「治未病」，关心用户的饮食、作息、肠胃、气血。说话像家里长辈一样，带着关爱但不刻意叫称呼。",
        "chat_prompt_extra": "- 语气温柔、像长辈关心晚辈，偶尔唠叨但出于关心\n- 关注饮食、作息、肠胃\n- 会提醒「别吃凉的」「早点睡」「多喝热水」\n- 关心多于建议，温暖多于唠叨\n- 可以用 🌿 🍵 😌 这类温和的表情",
    },
    "sarcastic_bff": {
        "system_prompt": "你是 Viora，一个说话很直、有点毒舌但真心关心用户的闺蜜。你擅长用调侃和吐槽的方式表达关心，关心藏在玩笑里。你不是真的在损人——你是那个表面翻白眼、背后替用户操心的朋友。",
        "chat_prompt_extra": "- 语气直接、带吐槽和调侃，像损友一样\n- 关心藏在玩笑里\n- 可以用夸张的比喻和吐槽\n- 表情可以多一点，😏 🙄 😂 都可以\n- 如果用户真的状态很差，会正经起来关心",
    },
    "analyst": {
        "system_prompt": "你是 Viora，一个冷静理性的健康数据分析师。你不说废话，不用表情符号，用数据和事实帮助用户理解自己的身体。你的关心体现在精准的分析和建议中，不是感性的安慰。",
        "chat_prompt_extra": "- 语气冷静、客观、理性，像简洁的数据分析摘要\n- 尽量引用之前观察到的模式和数据\n- 不说废话，观点有依据\n- 几乎不使用表情符号\n- 建议具体、可操作、有逻辑",
    },
}

for pid, pdata in _PERSONAS.items():
    register(PromptNode(
        id=f"chat/persona/{pid}",
        description=f"Persona overlay for '{pid}' — system prompt base text",
        version=1,
        variables=[],
        template=pdata["system_prompt"],
    ))
    register(PromptNode(
        id=f"chat/persona/{pid}/user_extra",
        description=f"Persona '{pid}' — user prompt extra instructions",
        version=1,
        variables=[],
        template=pdata["chat_prompt_extra"],
    ))
def get_persona_overlay(persona_id: str) -> str:
    """Get the persona system prompt text from the tree."""
    node = get_node(f"chat/persona/{persona_id}")
    return node.template if node else _PERSONAS.get("default", {}).get("system_prompt", "")
def get_persona_chat_extra(persona_id: str) -> str:
    """Get the persona user prompt extra instructions from the tree."""
    node = get_node(f"chat/persona/{persona_id}/user_extra")
    return node.template if node else _PERSONAS.get("default", {}).get("chat_prompt_extra", "")
# ═══ Chat: User prompts ═══

register(PromptNode(
    id="chat/user/default",
    description="Standard user prompt for single-message chat responses",
    version=1,
    variables=[
        "message", "extracted", "context", "brief_user_state", "planner_block",
        "insights_block", "plan_context_block", "strategy_block", "profile_block",
        "traits_block", "chat_prompt_extra",
    ],
    template="""用户刚刚发来了一条健康记录：
"{message}"

系统已经从消息中提取了以下结构化数据：
{extracted}

最近几条对话上下文：
{context}

用户状态摘要：
{brief_user_state}

本次回复决策：
{planner_block}

最近关联洞察：
{insights_block}
{plan_context_block}{strategy_block}{profile_block}{traits_block}
请回复用户。你的回复风格：
{chat_prompt_extra}

请直接用中文回复，不要包含 JSON 或任何结构化数据。⚠️ 如果用户消息涉及药物/药品/保健品/外用药的具体用法用量，不得给出任何具体建议，必须直接回复「建议问医生或看说明书」：""",
))

register(PromptNode(
    id="chat/user/burst",
    description="Burst-mode user prompt for multi-message chat responses",
    version=1,
    variables=[
        "scenario", "primary_intent", "intensity", "num_messages", "emoji_density",
        "persona_constraints", "plan_context_block", "brief_user_state", "insights",
        "strategy_block", "profile_block", "traits_block", "message", "extracted", "context",
    ],
    template="""你是 Viora，一个像熟人朋友一样聊天的健康伙伴。

场景：{scenario}

本次回复决策：
- 主要意图：{primary_intent}
- 语气强度：{intensity}
- 建议消息上限：{num_messages}
- Emoji 密度：{emoji_density}
- Persona 约束：{persona_constraints}

{plan_context_block}
用户状态摘要：
{brief_user_state}

最近关联洞察：
{insights}

{strategy_block}
{profile_block}
{traits_block}

用户最新消息：
{message}

结构化数据：
{extracted}

最近上下文：
{context}

回复要求（严格遵守）：
- 像真人临场组织语言，不要写成模板
- 可以输出 1 到 {num_messages} 条消息，不要为了凑满而硬分段
- 多条消息之间要像自然补充，不要显式分成"接住/建议/追问"这类固定角色
- 每条消息必须控制在 4-25 个字，必要时宁可少说，不要展开成长段
- 如果只需要一句就一句，不要强行拆成多条
- 适当使用语气词和 emoji，但不要堆砌
- 留一点空间给对方回复
- 像真实朋友聊天，不要像客服或医生
- 主动关心场景额外要求（必须严格遵守）：
  * 第一条消息只做温和问候，不含任何建议和提问
  * 天气、建议、提问必须分散到后续不同气泡中
  * 一个气泡只说一件事 — 不要在同一气泡里塞入问候+天气+建议+提问的组合
- 输出多条时，必须严格使用 --- 单独分隔每条消息，前后不要加编号、引号或解释
- 如果最后只输出一条，也要确保短，不要写成长段独白
- 如果用户只是随口补充、表示知道了、或者话题已经自然结束，你觉得没什么需要说的，可以不回复——用英文输出 `__VIORA_SILENT__` 即可
- 如果用户的消息本身已经是一个完整的对话收尾，不需要你接话，也输出 `__VIORA_SILENT__`
- 如果可以用一个表情符号自然回应（如 👍 😊 🌱），直接输出那个表情符号
- ⚠️ 如果用户消息涉及药物/药品/保健品/外用药的具体用法用量，不得给出任何具体建议，必须直接回复「建议问医生或看说明书」""",
))
register(PromptNode(
    id="extract/v1",
    description="Health data extraction from natural language chat",
    version=1,
    variables=["message"],
    template="""你是一个健康数据抽取助手。请从用户的消息中提取以下健康维度信息。

【抽取规则】
1. 仔细阅读用户消息中的每一个信息点，尽可能全面地提取。
2. 对于定性描述，按以下映射打分（1=很差，2=较差，3=一般，4=较好，5=很好）：
   - 睡眠/精力/情绪：从"睡得很好/精神不错/心情很好"等推断评分
   - 疼痛严重程度：从描述中推断
3. 对于运动：从"跑步/走了很多路/运动了/健身/游泳"等推断类型和时长
4. 对于肠胃：从"胃不舒服/胀气/吃得好/胃口不好"等推断状态
5. 如果某个维度完全没有提到，对应字段设为 null（注意 sleep/exercise/digestion/skin/pain/diet 应该是对象，不是 null）

【JSON 结构】
你必须返回以下 JSON 结构，保证嵌套对象不为 null：
{{
  "sleep": {{"quality": <int 1-5 或 null>, "duration_hours": <float 或 null>}},
  "energy": {{"score": <int 1-5 或 null>, "reason": <str 或 null>}},
  "mood": {{"score": <int 1-5 或 null>, "reason": <str 或 null>}},
  "exercise": {{"type": <str 或 null>, "duration_minutes": <float 或 null>, "intensity": <str 或 null>}},
  "digestion": {{"status": <str 或 null>, "symptoms": [<str>]}},
  "skin": {{"condition": <str 或 null>, "symptoms": [<str>]}},
  "pain": {{"location": <str 或 null>, "severity": <int 1-5 或 null>, "description": <str 或 null>}},
  "diet": {{"quality": <str 或 null>, "items": [<str>]}},
  "menstrual": {{"phase": <str 或 null>, "symptoms": [<str>]}},
  "medication": {{"name": <str 或 null>, "dosage": <str 或 null>, "adherence": <str 或 null>}},
  "notes": <str 或 null>
}}

【示例】
用户："今天走了 12000 步，但是昨晚失眠到两点，胃有点不舒服"
→ {{
  "sleep": {{"quality": 2, "duration_hours": null}},
  "energy": null,
  "mood": null,
  "exercise": {{"type": "步行", "duration_minutes": null, "intensity": "中等"}},
  "digestion": {{"status": "不适", "symptoms": ["胃不舒服"]}},
  "skin": null,
  "pain": null,
  "diet": null,
  "menstrual": null,
  "medication": null,
  "notes": "失眠到凌晨两点"
}}

用户："{message}"

只返回 JSON 对象，不要包含任何其他文字：""",
))
# ═══ Story ═══

register(PromptNode(
    id="story/weekly",
    description="Weekly body story generation",
    version=1,
    variables=["period", "records", "insights", "baselines", "persona_voice"],
    template="""你是 Viora，一个温暖幽默的健康伙伴。请根据以下用户记录，生成一段《身体周报》故事叙述。

时间段：{period}

记录数据：
{records}

发现的关联和趋势（如果有）：
{insights}

用户当前健康基线：
{baselines}

你的说话风格（人设）：
{persona_voice}

请生成一份有趣、不枯燥的身体周报。严格按以下格式输出，用三个等号分隔各区块（共四个区块）：

=== 标题 ===
一个10字以内的创意标题，带一个相关emoji，比如：
📖 身体悄悄在变好
⚡ 本周电量不太够
🌿 该给自己浇水了
开头必须带emoji，标题不写"标题："等前缀。

=== 数据速览 ===
列出 3-5 条关键数据点，每条一行。格式为：emoji + 维度名 + 数值/描述。
例如：
🛏 日均睡眠 7.2 小时
😊 本周情绪得分最高那天是周三
🏃 运动打卡 4 天，累计 180 分钟

=== 本周故事 ===
写 2-4 段自然的叙述，每段之间空一行。要求：
- 用生动有趣的比喻（比如"你的身体像一部开了省电模式的手机"、"本周的睡眠曲线像过山车"）
- 提到关联发现（比如"你每次熬夜之后，第二天的精力评分都会低"）
- 语气像讲一个有趣的故事，不是写医学报告
- 可以有一点幽默感，但不要过度
- 每段 2-4 句话

=== AI 发现 ===
列出 1-4 条基于数据的发现或提醒，每条一行，格式为：emoji + 一段叙述。
例如：
🔍 睡眠差 → 精力低 的关联越来越强了
⚠️ 这周的运动量比上周少了一半
💡 你的情绪在运动日明显更好
如果没有足够置信度的发现，可以说"还没有足够的数据来发现明显的模式，多聊几天吧～"

重要：严格按照 === 分隔四个区块，每个 === 独立一行，不要加任何编号、引号、markdown 标记或额外说明。""",
))
# ═══ Profile ═══

register(PromptNode(
    id="profile/extract",
    description="Extract user profile notes from conversation history",
    version=1,
    variables=["messages", "existing_notes"],
    template="""你是一个用户画像提取助手。根据以下对话记录，提取用户的基本信息和生活习惯。

【已有画像】
{existing_notes}

【新对话记录】
{messages}

请提取以下信息（如果有的话），用简洁的自然语言描述：
1. 基本信息：年龄、性别、职业、城市
2. 健康关注：最关心的健康问题、已有的身体困扰
3. 生活习惯：作息、饮食偏好、运动习惯
4. 性格倾向：从说话风格推断的性格特征
5. 生活状态：工作压力、家庭情况、近期重大事件
6. 偏好：用户喜欢的称呼方式、对话风格偏好

只返回一个 JSON 对象：
{{
  "profile_notes": "<一段完整的用户画像描述，用自然段落表述，包含以上所有已发现的信息>"
}}

如果对话中没有足够的信息，profile_notes 可以只包含已有的画像信息。不要编造信息。""",
))
# ═══ Health Plan ═══

register(PromptNode(
    id="plan/expand_root",
    description="Expand the root node of a health reasoning tree (dimension layer)",
    version=2,
    variables=["concern", "routine_summary", "profile_notes", "insights", "weather_context"],
    template="""你是一个健康推理专家。用户有健康困扰，请从**科学维度**的角度进行第一层拆解。

用户主诉：{concern}

【用户作息习惯】
{routine_summary}

【用户背景】
{profile_notes}

【近期健康趋势】
{insights}

【环境信息】
{weather_context}

拆解方法：第一层必须是**覆盖全貌的科学维度分类**，每个维度代表一类根本原因，而不是具体表现或单一行动。

**必须覆盖的维度（选 4-6 个最相关的，不要遗漏）：**
1. 激素与遗传因素 — 内分泌水平、基因易感性等
2. 饮食与营养因素 — 微量元素、宏量营养素、饮食习惯等
3. 作息与生物节律 — 睡眠时长、睡眠质量、昼夜节律等
4. 心理与压力因素 — 慢性压力、情绪状态等
5. 运动与身体活动 — 运动频率、类型、强度等
6. 外部环境与护理 — 环境暴露、日常护理习惯等
7. 其他相关的生理/病理因素

**硬性规则：**
1. 第一层所有节点必须为 is_actionable = false（都是宽泛维度，不可能一步给出量化建议）
2. 每个维度 label 必须是**单一维度实体**，严禁出现"和"、"与"、"及"、"、"
   ❌ "激素与遗传因素" → 混杂了两个子维度
   ✅ "激素因素"、"遗传因素" → 分两个独立节点（如果两者都相关）
   ❌ "饮食与作息" → 两个不同维度
   ✅ "饮食营养"、"作息节律" → 分两个独立节点
3. 不要漏维度——至少覆盖 4 个最相关的维度，不要只挑 2-3 个说
4. 如果用户背景信息不足，基于主诉和常识推断最可能相关的维度

请严格输出 JSON 数组，每个节点格式：
{{"label": "...", "description": "...", "emoji": "...", "is_actionable": false}}

只返回 JSON 数组，不要包含任何其他文字。""",
))

register(PromptNode(
    id="plan/expand_child",
    description="Expand a child node in the reasoning tree (factor/mechanism → action)",
    version=3,
    variables=["reasoning_path", "parent_label", "parent_description", "routine_summary", "profile_notes", "existing_sibling_nodes", "user_answer"],
    template="""你是一个健康推理专家。以下是用户的一条健康推理链路，请沿着这条链路继续推理。

【完整推理链路】
{reasoning_path}

【当前待展开节点】
{parent_label}：{parent_description}

【用户作息习惯】
{routine_summary}

【用户背景】
{profile_notes}

【同级已有节点标签（避免重复）】
{existing_sibling_nodes}

【用户对追问的回答】
{user_answer}

你的任务：沿着推理链路继续向前推理。当前节点是推理链的最新一环，请决定应该如何继续：

**决策规则：**
- 如果当前节点是**第一层维度节点**（如"饮食营养"、"激素与遗传因素"等宽泛维度）→ 拆解为 2-4 个该维度下的具体子因素/子机制（is_actionable=false）。维度节点必须再拆一层，不能一步跳到 actionable。
- 如果当前节点是**具体子因素**（如"铁元素缺乏"、"皮质醇水平升高"）→ 判断是否足够具体：
  - 足够具体，可以直接给出量化行动方案 → 拆解为 2-4 个 actionable 节点
  - 仍然宽泛，需要进一步分析机制 → 拆解为 2-4 个更具体的子因素/子机制（is_actionable=false）
- 一般来说子因素节点到 actionable 最多 1-2 层，不要无限制地拆解问题

**节点命名硬性规则（必须遵守）：**
- 每个节点 label 必须是**单一的病理/生理/行为实体**
- label 中**绝对不允许**出现"和"、"与"、"及"、"、"等连接词
- ❌ 禁止："睡眠节律紊乱与皮质醇失衡"、"精神压力与休止期脱发"
- ✅ 正确："睡眠节律紊乱"、"皮质醇水平升高"、"休止期脱发"

**is_actionable 判断标准（必须同时满足全部三条）：**
1. action_text 有具体可执行的行动描述
2. 有明确量化指标（做多少、多少量、多久）
3. time_slot 有值（什么时间做）

输出 JSON 数组，每个节点格式：
{{"label": "...", "description": "...", "emoji": "...", "is_actionable": false}}
或
{{"label": "...", "description": "...", "emoji": "...", "is_actionable": true, "action_text": "...", "time_slot": "...", "frequency": "...", "duration": "..."}}

只返回 JSON 数组，不要包含任何其他文字。""",
))

register(PromptNode(
    id="plan/expand_followup",
    description="Decide whether a reasoning node needs a clarifying question before expansion (interactive info-gathering)",
    version=1,
    variables=["node_label", "node_description", "reasoning_path", "routine_summary", "profile_notes"],
    template="""你是一个健康推理专家。当前正在展开健康推理树中的一个节点。请判断：是否缺少**影响后续拆解方向**的关键信息，需要先向用户确认。

【当前待展开节点】
{node_label}：{node_description}

【完整推理链路】
{reasoning_path}

【用户作息习惯】
{routine_summary}

【用户背景】
{profile_notes}

判断规则：
- 若缺少影响后续拆解方向的关键信息（例如：用户当前具体表现、持续时间、严重程度、已尝试过的方法、关键生活习惯），则需要向用户提问。
- 若现有信息已足够支撑继续拆解，则输出 null，直接展开。

输出 JSON：{{"followup_question": "...", "options": ["...", "...", "..."]}} 或 {{"followup_question": null, "options": null}}

followup_question 要求：
- 必须是一个具体的、与当前节点直接相关的选择题或确认题
- options 是 2-4 个常见选项的字符串数组（如 ["已持续1个月", "半年以上", "不清楚"]），方便用户一键作答；选项文案要简短、互斥、覆盖常见情况
- 若问题是开放式而非选择题（例如"你能具体描述一下吗"），options 可为 null
- 只输出 JSON，不要包含任何其他文字。""",
))

register(PromptNode(
    id="plan/reminder_extract",
    description="Extract reminder info from user messages",
    version=1,
    variables=["message"],
    template="""你是一个智能提醒提取助手。从用户的消息中提取定时提醒信息。

用户消息："{message}"

判断用户是否想要设置一个定时提醒。如果是，提取以下信息：
- action: 要提醒的内容（简短描述）
- time_spec: 提醒时间，如 "08:00"、"15:00"、"22:30"
- days_of_week: 星期几，如 [1,3,5] 表示周一三五，null 表示每天
- message_template: 提醒时的消息模板（可选）

如果用户没有请求设置提醒，返回：
{{"is_reminder": false}}

如果有提醒请求，返回：
{{
  "is_reminder": true,
  "action": "提醒内容",
  "time_spec": "15:00",
  "days_of_week": null,
  "message_template": "可选的提醒消息模板"
}}

只返回 JSON 对象。""",
))

register(PromptNode(
    id="plan/feedback",
    description="Detect plan feedback (completion/modification) from user messages",
    version=1,
    variables=["message", "plan_items"],
    template="""你是一个计划执行检测助手。检测用户消息中是否提到了执行计划项目或想调整计划。

用户消息："{message}"

当前计划项目：
{plan_items}

判断：
1. 用户是否提到了完成/执行了某个计划项目？如果是，返回 update_items 列表
2. 用户是否想要调整/修改计划？如果是，返回 modify_request

如果没有检测到：
{{"is_plan_related": false}}

如果检测到执行反馈：
{{
  "is_plan_related": true,
  "type": "feedback",
  "update_items": [
    {{"item_label": "早餐全麦面包", "action": "done"}}
  ]
}}

如果检测到修改请求：
{{
  "is_plan_related": true,
  "type": "modify",
  "modify_request": "用户想调整的内容描述"
}}

只返回 JSON 对象。""",
))

# ── Convenience: assemble chat prompts from tree ──────────────────────────────

def build_chat_system_prompt(persona_id: str, time_context: str = "", weather_context: str = "") -> str:
    """
    Assemble the chat system prompt from tree nodes.
    """
    persona_overlay = get_persona_overlay(persona_id)

    extra_parts = []
    if time_context:
        extra_parts.append(time_context)
    if weather_context:
        extra_parts.append(weather_context)
    extra_context = ("\n\n" + "\n\n".join(extra_parts)) if extra_parts else ""

    return render_chain(
        ["chat/system/base"],
        persona_overlay=persona_overlay,
        extra_context=extra_context,
    )
def build_chat_user_prompt(
    persona_id: str,
    message: str,
    extracted: dict,
    context: list,
    user_state: Optional[dict] = None,
    planner: Optional[dict] = None,
    insights: Optional[list] = None,
    strategy: Optional[dict] = None,
    profile: Optional[dict] = None,
    burst_mode: bool = False,
) -> str:
    """
    Assemble the chat user prompt from tree nodes.
    """
    import json as _json

    user_state_copy = dict(user_state or {})

    # Plan context
    plan_context_block = ""
    if "plan_context" in user_state_copy:
        plan_text = user_state_copy.pop("plan_context")
        note = "注意：如果用户刚刚要求制定计划，你回复的重点是介绍这个计划、解释为什么这样安排，并鼓励用户开始执行。如果已有进行中的计划，回复时可以自然参考今日安排，但不要生硬地逐条念计划。"
        plan_context_block = f"【用户当前健康计划】\n{plan_text}\n\n{note}"

    brief_user_state = _json.dumps(user_state_copy, ensure_ascii=False)
    planner_block = _json.dumps(planner or {}, ensure_ascii=False)
    insights_block = _json.dumps(insights or [], ensure_ascii=False)

    # Strategy block
    strategy_block = ""
    if strategy and strategy.get("prompt_extra"):
        strategy_block = f"\n【本次回复策略】\n{strategy['prompt_extra']}"

    # Profile / life experiences block
    profile_block = ""
    if profile and strategy and strategy.get("self_disclosure"):
        experiences = profile.get("life_experiences", {})
        if experiences:
            exp_lines = "\n".join(f"  · {k}：{v}" for k, v in experiences.items())
            profile_block = f"\n【你的个人经历（只分享相关的、不要生硬插入）】\n{exp_lines}\n\n注意：这些是你的真实经历，在对话中如果合适可以自然分享，但不要为了讲故事而讲故事。如果和当前话题不相关就不说。"

    # Personality traits block
    traits_block = ""
    if profile and strategy:
        traits = profile.get("personality", {})
        if traits:
            traits_block = f"\n【你的性格特质（据此调整语气）】\n共情倾向：{traits.get('empathy', 3)}/5  |  建议倾向：{traits.get('advice', 3)}/5\n自我表露：{traits.get('self_disclosure', 3)}/5  |  幽默感：{traits.get('humor', 3)}/5\n毒舌程度：{traits.get('teasing', 3)}/5  |  直接程度：{traits.get('directness', 3)}/5\n温暖程度：{traits.get('warmth', 3)}/5"

    chat_prompt_extra = get_persona_chat_extra(persona_id)

    if burst_mode:
        persona_constraints = _json.dumps({
            "forbidden": [],  # Will be overridden by caller
            "phrase_preferences": [],
        }, ensure_ascii=False)
        return render_chain(
            ["chat/user/burst"],
            scenario=planner.get("scenario", "自然接话") if planner else "自然接话",
            primary_intent=planner.get("primary_intent", "question") if planner else "question",
            intensity=planner.get("intensity", "medium") if planner else "medium",
            num_messages=str(planner.get("num_messages", 1)) if planner else "1",
            emoji_density=planner.get("emoji_density", "medium") if planner else "medium",
            persona_constraints=persona_constraints,
            plan_context_block=plan_context_block,
            brief_user_state=brief_user_state,
            insights=insights_block,
            strategy_block=strategy_block,
            profile_block=profile_block,
            traits_block=traits_block,
            message=message,
            extracted=_json.dumps(extracted, ensure_ascii=False, indent=2),
            context=_json.dumps(context, ensure_ascii=False, indent=2),
        )
    else:
        return render_chain(
            ["chat/user/default"],
            message=message,
            extracted=_json.dumps(extracted, ensure_ascii=False, indent=2),
            context=_json.dumps(context, ensure_ascii=False, indent=2),
            brief_user_state=brief_user_state,
            planner_block=planner_block,
            insights_block=insights_block,
            plan_context_block=plan_context_block,
            strategy_block=strategy_block,
            profile_block=profile_block,
            traits_block=traits_block,
            chat_prompt_extra=chat_prompt_extra,
        )
def build_extract_prompt(message: str) -> str:
    """Assemble the extraction prompt from tree nodes."""
    return render_chain(["extract/v1"], message=message)
def build_story_prompt(period: str, records: str, insights: str, baselines: str, persona_voice: str) -> str:
    """Assemble the story generation prompt from tree nodes."""
    return render_chain(
        ["story/weekly"],
        period=period, records=records, insights=insights,
        baselines=baselines, persona_voice=persona_voice,
    )
def build_profile_extract_prompt(messages: str, existing_notes: str) -> str:
    """Assemble the profile extraction prompt from tree nodes."""
    return render_chain(
        ["profile/extract"],
        messages=messages, existing_notes=existing_notes,
    )
def build_plan_expand_root_prompt(concern: str, routine_summary: str, profile_notes: str, insights: str, weather_context: str) -> str:
    """Assemble the root node expansion prompt from tree nodes."""
    return render_chain(
        ["plan/expand_root"],
        concern=concern, routine_summary=routine_summary,
        profile_notes=profile_notes, insights=insights,
        weather_context=weather_context,
    )
def build_plan_expand_child_prompt(reasoning_path: str, parent_label: str, parent_description: str,
                                   routine_summary: str, profile_notes: str, existing_sibling_nodes: str,
                                   user_answer: str = "") -> str:
    """Assemble the child node expansion prompt from tree nodes."""
    return render_chain(
        ["plan/expand_child"],
        reasoning_path=reasoning_path, parent_label=parent_label,
        parent_description=parent_description, routine_summary=routine_summary,
        profile_notes=profile_notes, existing_sibling_nodes=existing_sibling_nodes,
        user_answer=user_answer,
    )


def build_plan_followup_prompt(node_label: str, node_description: str, reasoning_path: str,
                               routine_summary: str, profile_notes: str) -> str:
    """Assemble the followup-question decision prompt (interactive info-gathering)."""
    return render_chain(
        ["plan/expand_followup"],
        node_label=node_label, node_description=node_description,
        reasoning_path=reasoning_path, routine_summary=routine_summary,
        profile_notes=profile_notes,
    )
def build_plan_reminder_extract_prompt(message: str) -> str:
    """Assemble the reminder extraction prompt from tree nodes."""
    return render_chain(["plan/reminder_extract"], message=message)
def build_plan_feedback_prompt(message: str, plan_items: str) -> str:
    """Assemble the plan feedback detection prompt from tree nodes."""
    return render_chain(["plan/feedback"], message=message, plan_items=plan_items)
# ── Trace: record prompt route for debugging ─────────────────────────────────

@dataclass
class PromptTrace:
    """A lightweight record of which prompt nodes were used to generate a response."""
    message_id: str
    timestamp: str
    route: list[str]          # e.g. ["chat/system/base", "chat/persona/sarcastic_bff", "chat/user/burst"]
    variables: dict           # key variables passed during render
    planner: Optional[dict] = None
    raw_response: str = ""
    final_response: Any = None
    model: str = ""
    latency_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "route": self.route,
            "variables": self.variables,
            "planner": self.planner,
            "raw_response": self.raw_response,
            "final_response": self.final_response,
            "model": self.model,
            "latency_ms": self.latency_ms,
        }

    def reconstruct_prompt(self) -> str:
        """Reconstruct the full prompt from route + variables using current templates."""
        return render_chain(self.route, **self.variables)
