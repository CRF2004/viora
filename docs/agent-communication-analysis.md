# Agent 通信协议与调度方式 — 技术分析（供 §8 决策参考）

> 对应 `plan.md §8.4` 开放问题 #1：**各 Agent 间的通信协议和调度方式（同步 pipeline vs 异步消息队列 vs 事件驱动）**。
> 本文件是**决策支持分析**，不是实现。目标：把三个选项放到 Viora 的真实代码上下文里比较，给出可落地的数据结构和调用流建议，帮助确认 `docs/agent-boundaries.md` 中"同步 pipeline + 轻量事件通知"的推荐默认值。
> 不进入实现；确认后才开发。

---

## 0. 结论先行

| 方案 | 适合 Viora 吗 | 关键理由 |
|---|---|---|
| 同步 pipeline | ✅ **推荐** | 单进程 Flask 应用、规模小；调用链可预测、易测易调试；与"分析先行、交互收口"原则天然一致 |
| 异步消息队列 (Celery/RQ/Redis) | ⚠️ 过度 | 需要引入 broker 依赖、多 worker、任务序列化；当前单机单进程规模下收益极低，只增加运维复杂度 |
| 事件驱动 (事件总线/pubsub) | ✅ **部分采用** | 同进程内的轻量事件通知（回调/回调注册表）值得采用，用于解耦"数据写入 → 分析触发"；但不引入独立消息基础设施 |

**推荐：同步 pipeline 为主干 + 同进程轻量事件通知（回调注册表）解耦旁路触发。**
这与 `docs/agent-boundaries.md` #1 的推荐默认值一致。

---

## 1. 三种方案在 Viora 的对比

### 1.1 同步 pipeline（推荐主干）

**形态**：编排层按固定顺序调用各 Agent 的函数，前一个的输出作为后一个的输入，全部在同一请求/线程内完成。

```
用户消息
  → [Interaction] 意图识别、判断是否涉及健康内容
  → [Analysis]    从结构化数据产出 AnalysisResult（来源+窗口+样本量+置信度）
  → [Planning]    基于主诉+画像生成推理树（ReasoningNode）
  → [Story]       将 Analysis 结论包装成叙事片段
  → [Interaction] 收口，把上面内容翻译成自然口语回复给用户
```

**优点**：
- 调用链可预测，单个请求内就能拿到完整可追溯链（符合编排原则 3）。
- 无需新增依赖，直接复用现有模块函数（`ai_engine.py` / `health_plan.py` / `insights.py`）。
- 测试简单：同步调用即可断言，无需 mock broker。

**缺点**：
- 端到端耗时 = 各 Agent 耗时之和；若某个 Agent 变慢（如 LLM 超时），整条链路阻塞。
- 不适合"写后立即响应、后台慢慢分析"的场景——但 Viora 的交互需要即时回复，主链路本来就该同步。

### 1.2 异步消息队列（不推荐）

**形态**：引入 Celery/RQ + Redis/broker，各 Agent 以任务形式投递到队列，worker 消费。

**优点**：吞吐高、可横向扩展、任务可重试/延迟执行。

**缺点**：
- 新增 broker 依赖、worker 进程、任务序列化、失败重试策略——运维复杂度与 Viora 当前单体规模完全不匹配。
- 消息队列的"解耦"会破坏可追溯链的直观性：一次用户请求可能拆成多个异步任务，来源追踪变难（与编排原则 3 相悖）。
- 测试需要起 broker 或用 fake 队列，显著增加测试成本。

**结论**：当未来出现"大量并发用户 + 重后台分析"时才值得考虑；当前是单用户、主动关怀型应用，收益为负。

### 1.3 事件驱动（部分采用：同进程事件通知）

**形态**：定义一个轻量事件总线（回调注册表），数据写入时广播事件，订阅方异步（同线程/后续请求）处理。不引入独立消息基础设施。

**适用点**：Analysis Agent 的触发条件是"新数据写入后异步触发"（见 agent-boundaries.md 分析端触发条件），这正是事件通知的解耦场景——健康/轨迹数据落库后，发出 `data.updated` 事件，分析端订阅并在适当时候产出新结论。

**为什么不是完整事件驱动**：
- 完整事件驱动（所有 Agent 只响应事件、无固定调用序）会失去"分析先行、交互收口"的保证——顺序变成隐式的，难以审计。
- 因此只把**旁路触发**（数据写入 → 分析重算 / 里程碑 → 故事生成）做成事件通知，**面向用户的响应链路仍保持同步 pipeline**。

---

## 2. 建议的数据结构与调用流（确认后可直接实现）

### 2.1 消息/任务载体：`AgentMessage` dataclass

```python
# 建议放在新模块 orchestrator.py（确认后创建）
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class AgentMessage:
    """Agent 间传递的消息载体，承载可追溯链。"""
    message_id: str                      # uuid4
    sender: str                          # "interaction" | "analysis" | "planning" | "story" | "system"
    receiver: str                        # 目标 Agent 或 "*"（广播）
    kind: str                            # "user_message" | "analysis_result" | "plan_tree" | "story_text" | "safety_check" ...
    payload: dict[str, Any]              # 结构化载荷
    trace: list[dict] = field(default_factory=list)  # 追溯链：每步记录 (agent, timestamp, ref)
    created_at: str = field(default_factory=lambda: __import__("time").strftime("%Y-%m-%dT%H:%M:%S"))
```

追溯链直接挂在消息上（`trace`），满足编排原则 3。

### 2.2 同步 pipeline：`Orchestrator.run(user_input)` 骨架

```python
class Orchestrator:
    def __init__(self, agents: dict[str, Agent]):
        self.agents = agents   # {"interaction": ..., "analysis": ..., "planning": ..., "story": ...}

    def run(self, user_input: str) -> AgentMessage:
        msg = AgentMessage(sender="system", receiver="interaction",
                           kind="user_message", payload={"text": user_input})
        # 1) Interaction：判断是否涉及健康内容
        intent = self.agents["interaction"].classify(msg)
        # 2) Analysis：产出结论包（若需要）
        if intent.requires_analysis:
            analysis = self.agents["analysis"].produce(msg)
            msg.trace.append(analysis.trace_step())
        # 3) Planning：基于主诉生成推理树（若需要）
        if intent.requires_planning:
            plan = self.agents["planning"].produce(msg)
            msg.trace.append(plan.trace_step())
        # 4) Story：包装叙事（可选）
        # 5) Interaction：收口成自然回复
        reply = self.agents["interaction"].finalize(msg)
        return reply
```

### 2.3 轻量事件通知：回调注册表（旁路触发）

```python
class EventBus:
    def __init__(self):
        self._subs: dict[str, list[Callable]] = {}

    def subscribe(self, event: str, fn: Callable): ...
    def publish(self, event: str, **kwargs):
        # 同进程顺序调用订阅者；不引入 broker
        for fn in self._subs.get(event, []):
            fn(**kwargs)

# 例：数据写入后触发分析重算（不阻塞用户响应）
bus = EventBus()
bus.subscribe("data.updated", lambda: analysis.recompute_recent_insights())
```

### 2.4 与现有模块的映射（确认后接线）

| 编排角色 | 现有模块 | 接线点 |
|---|---|---|
| Interaction | `persona.py` / `ai_engine.py`(ResponsePlanner) / `psycho_engine.py` | 现有聊天入口 `app.py` |
| Analysis | `insights.py` / `routine.py` / `trajectory_graph.py` / `health_graph.py` | 新数据写入后事件订阅 |
| Planning | `health_plan.py`(ReasoningNode+Expand) | 主诉进入时走 pipeline |
| Story | `scheduled_stories.py` / `story_scheduler_daemon.py` | 里程碑/定时事件触发 |

---

## 3. 对用户决策的影响

- 若确认"同步 pipeline + 轻量事件通知"，开发时**无需新依赖**（只新增 `orchestrator.py` + 轻量 `event_bus.py`），可直接复用现有模块，4 个 Agent 先以"同一 LLM + 不同 system prompt"实现（开放问题 #6 的推荐默认值）。
- 若未来出现多用户/高并发，再演进为独立消息队列（预留 `EventBus` 接口替换点，不影响 pipeline 主干）。
- 该方案与 §8 编排原则完全兼容：固定调用序保证"分析先行、交互收口"，`trace` 保证可追溯链，安全校验层（开放问题 #2）可插入 pipeline 的每个阶段之间。

---

## 4. 本文件状态

- [ ] 用户确认开放问题 #1（或回复「全部采用推荐默认值」）
- [ ] 确认后本分析定稿为设计基线，进入 §8.5 前置条件，再开始编排层开发
