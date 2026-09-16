from functools import wraps

from flask import Flask, render_template_string, jsonify, send_from_directory, request as flask_request, session
from flask_cors import CORS
from config import SERVER_HOST, SERVER_PORT, DEBUG
import os
import json
import logging
import traceback
from datetime import datetime, timezone
from uuid import uuid4

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
_log_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(_log_dir, exist_ok=True)
_file_handler = logging.FileHandler(os.path.join(_log_dir, "viora.log"))
_file_handler.setLevel(logging.WARNING)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
logging.getLogger().addHandler(_file_handler)  # Root logger — catches all modules

app = Flask(__name__)
app.secret_key = os.environ.get("VIORA_SECRET_KEY", "viora-dev-secret-key")
CORS(app, supports_credentials=True)

# 配置
PRESETS_FOLDER = 'presets_graph'
TEMPLATE_FILE = 'index.html'

# 确保预设文件夹存在
if not os.path.exists(PRESETS_FOLDER):
    os.makedirs(PRESETS_FOLDER)


def _json_error(message: str, code: int = 400):
    return jsonify({"error": message}), code


def current_user_id() -> str:
    return session.get("user_id") or ""


def current_user() -> dict:
    user_id = current_user_id()
    if not user_id:
        return {}
    return get_account(user_id) or {}



def current_user_public() -> dict:
    return public_account(current_user()) or {}



def build_chat_context(messages: list[dict]) -> list[dict]:
    """Build chronological chat context including both user and assistant turns."""
    context: list[dict] = []
    for m in reversed(messages):
        context.append({"role": "user", "content": m["content"], "timestamp": m["timestamp"]})
        ai_response = m.get("ai_response")
        if not ai_response:
            continue
        if isinstance(ai_response, list):
            for part in ai_response:
                context.append({"role": "assistant", "content": part, "timestamp": m["timestamp"]})
        else:
            context.append({"role": "assistant", "content": ai_response, "timestamp": m["timestamp"]})
    return context



def build_chat_user_state(user_id: str, persona_id: str = None) -> dict:
    """Build a lightweight user state summary for burst planning."""
    account = get_account(user_id) or {}
    activity = compute_activity_score(user_id)
    anomaly_level = "low"
    if activity.get("level") == "inactive":
        anomaly_level = "high"
    elif activity.get("level") == "idle":
        anomaly_level = "medium"

    persisted = get_user_state(user_id, persona_id=persona_id or account.get("persona_id") or "default")
    persisted.update({
        "user_id": user_id,
        "persona_id": persona_id or account.get("persona_id") or persisted.get("persona_id") or "default",
        "display_name": account.get("display_name"),
        "activity": activity,
        "anomaly_level": anomaly_level,
        "total_messages": account.get("total_messages", 0),
        "proactive_received": account.get("proactive_received", 0),
        "proactive_replied": account.get("proactive_replied", 0),
    })
    return persisted


def build_plan_context(user_id: str) -> str:
    """Build plan context string to inject into chat system prompt."""
    try:
        plan = get_active_plan(user_id)
        if not plan:
            return ""
        leaves = collect_actionable_leaves(plan)
        if not leaves:
            return f"\n【用户当前健康计划】\n目标：{plan.goal}\n（今日暂无安排）"

        items_text = "\n".join(
            f"  • {leaf.time_slot or '随时'} → {leaf.action_text or leaf.label}（{leaf.reasoning or ''}）"
            for leaf in leaves
        )
        return (
            f"\n【用户当前健康计划】\n"
            f"计划目标：{plan.goal}\n"
            f"今日安排：\n{items_text}\n"
            f"（自然参考计划内容，不要生硬地念计划。如果用户状态不好，可以建议跳过。）"
        )
    except Exception as e:
        logger.warning("Failed to build plan context: %s", e)
        return ""


def _clear_current_session() -> None:
    session.pop("user_id", None)


def require_login(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        user = current_user()
        if not user:
            return _json_error("unauthorized", 401)
        if user.get("is_deleted"):
            _clear_current_session()
            return _json_error("unauthorized", 401)
        return view(*args, **kwargs)

    wrapper.__name__ = view.__name__
    return wrapper

@app.route('/')
def index():
    """渲染主页面"""
    # 读取HTML模板
    if os.path.exists(TEMPLATE_FILE):
        with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        # 如果模板文件不存在，返回内嵌的HTML
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>知识图谱可视化</title>
        </head>
        <body>
            <h1>请将HTML文件保存为 index.html</h1>
        </body>
        </html>
        ''')

@app.route('/api/presets')
def get_presets():
    """获取所有预设文件列表"""
    try:
        files = [f for f in os.listdir(PRESETS_FOLDER) 
                if f.endswith('.json')]
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/graph/<filename>')
def get_graph(filename):
    """获取指定的图谱数据"""
    try:
        # 安全检查，防止路径遍历
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({"error": "Invalid filename"}), 400
        
        filepath = os.path.join(PRESETS_FOLDER, filename)
        
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/graph', methods=['POST'])
def save_graph():
    """保存新的图谱数据（可选功能）"""
    from flask import request
    try:
        data = request.json
        filename = data.get('filename', 'new_graph.json')
        
        # 安全检查
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({"error": "Invalid filename"}), 400
        
        filepath = os.path.join(PRESETS_FOLDER, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data['graph'], f, ensure_ascii=False, indent=2)
        
        return jsonify({"success": True, "message": "Graph saved successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Phase 1: Chat-based health records ──────────────────────────────────────
from ai_engine import extract_health_data, generate_chat_response, generate_body_story, normalize_extracted, extract_profile_notes, _parse_story_response, LLMError, expand_reasoning_node, extract_reminder_from_message, extract_plan_feedback, ask_followup_question
import storage
from routine import build_routine_model
from insights import compute_correlations, get_trend_insights
from persona import (
    get_available_personas,
    get_active_persona,
    set_active_persona,
    delete_user_persona,
)
from user_state import (
    get_user_state,
    update_user_state,
    delete_user_state,
)
from weather import (
    get_user_location,
    set_user_location,
    clear_location,
    get_weather_context,
)
from scheduler import (
    get_scheduler_status,
    enable_silent_mode,
    disable_silent_mode,
    should_send_proactive,
    SchedulerState,
    generate_proactive_message,
    record_proactive_sent,
    record_user_responded,
    delete_scheduler_state,
)
from health_graph import generate_health_graph
from trajectory_graph import TrajectoryGraph
from psycho_engine import extract_psycho_data, PsychoEvent, PsychoState, classify_intervenable_state, recommend_intervention, compute_psycho_fingerprint
from health_plan import (
    create_plan,
    get_active_plan,
    get_plan,
    list_plans,
    update_plan_status,
    delete_plan,
    rename_plan,
    regenerate_plan,
    create_reminder,
    list_reminders,
    delete_reminder,
    expand_node,
    collect_actionable_leaves,
    list_new_actionable_leaves,
    update_leaf_status,
    adopt_leaf,
    set_followup_question,
)
from accounts import (
    create_account,
    register_account,
    authenticate,
    mark_login,
    get_account,
    public_account,
    update_account,
    list_accounts,
    delete_account,
    get_or_create_default,
    compute_activity_score,
)
from storage import delete_messages_for_user, get_story_cache, set_story_cache, clear_story_cache
import feedback

@app.route('/api/chat', methods=['POST'])
@require_login
def chat():
    """
    POST /api/chat
    Body (single): {"message": "今天走了很多路但睡得不好"}
    Body (batch): {"messages": ["今天好累", "不想跑步了", "我是不是太懒了"]}
    Returns: {"extracted": {...}, "response": "...", "id": "..."}
    """
    try:
        data = flask_request.json
        if not data:
            return jsonify({"error": "Missing request body"}), 400

        # Accept both single message and messages array
        single = data.get('message')
        multiple = data.get('messages')
        if multiple and isinstance(multiple, list) and len(multiple) > 0:
            raw_messages = [m.strip() for m in multiple if m.strip()]
            if not raw_messages:
                return jsonify({"error": "Empty messages"}), 400
        elif single and isinstance(single, str) and single.strip():
            raw_messages = [single.strip()]
        else:
            return jsonify({"error": "Missing 'message' or 'messages' field"}), 400

        # Merge multiple messages into one combined input
        if len(raw_messages) == 1:
            combined_message = raw_messages[0]
        else:
            combined_message = "\n".join(
                f"[消息 {i+1}]: {msg}" for i, msg in enumerate(raw_messages)
            )
            combined_message += "\n\n(以上是用户连续发送的多条消息，请将它们视为同一轮对话中的自然连续输入进行回应)"

        user = current_user()
        user_id = user.get('user_id', '')
        persona_id = data.get('persona_id') or user.get('persona_id')

        # 1) Extract structured health data
        extracted = extract_health_data(combined_message)

        # 2) Build context from recent conversation history
        recent = storage.list_messages(limit=15, user_id=user_id)
        context = build_chat_context(recent)

        # 3) Generate Persona-driven response
        user_state = build_chat_user_state(user_id, persona_id)

        # Phase 4: Auto-detect plan request across all messages
        plan_id = _auto_create_plan_on_request(combined_message, user_id, extracted, persona_id)

        # Inject plan context into user_state so AI can reference active plan
        plan_context_str = build_plan_context(user_id)
        if plan_context_str:
            user_state["plan_context"] = plan_context_str

        recent_insights = compute_correlations(recent) + get_trend_insights(recent)
        persona = get_active_persona(user_id)
        if persona_id and persona_id != persona.get("id"):
            from persona import get_persona
            persona = get_persona(persona_id) or persona
        # Enable burst mode by default for natural, persona-driven multi-message replies
        use_burst = data.get('burst', True)
        # Pre-generate message ID for prompt trace linking
        chat_msg_id = str(uuid4())
        reply = generate_chat_response(
            combined_message,
            extracted,
            context,
            persona_id=persona_id,
            user_id=user_id,
            user_state=user_state,
            insights=recent_insights,
            burst=use_burst,
            trace_message_id=chat_msg_id,
            psycho_state=psycho_state,
        )

        # Handle silent response (model chose not to reply)
        if reply == "" or reply == []:
            reply = []

        # Track: user responded to proactive messages (per user)
        record_user_responded(user_id)

        # Update persistent user state before storing message
        current_routine = build_routine_model(recent).get_routine_summary()
        state = update_user_state(
            user_id,
            persona_id=persona_id,
            display_name=user.get("display_name"),
            extracted_data=extracted,
            message_text=combined_message,
            routine_summary=current_routine,
            anomaly_level=user_state.get("anomaly_level"),
            user_preferences={
                "last_reply_length": len(combined_message),
            },
        )

        # Periodically extract profile soft-info (every 10 interactions)
        if state.get("interaction_count", 0) > 0 and state["interaction_count"] % 10 == 0:
            try:
                # Fetch more messages for context
                recent_for_profile = storage.list_messages(limit=20, user_id=user_id)
                updated_notes = extract_profile_notes(recent_for_profile, state.get("profile_notes", ""))
                # Also try to extract wake-up time from updated notes
                from user_state import extract_wake_up_hour_from_profile
                wh = extract_wake_up_hour_from_profile(updated_notes)
                extra_prefs = {}
                if wh is not None:
                    extra_prefs["wake_up_hour"] = wh
                state = update_user_state(
                    user_id,
                    persona_id=persona_id,
                    profile_notes=updated_notes,
                    user_preferences=extra_prefs if extra_prefs else None,
                )
            except Exception as e:
                logger.warning("Profile notes extraction failed (non-critical): %s", e)

        # 4) Persist the message (user content + AI response)
        msg = storage.create_message(
            content=combined_message,
            extracted_data=extracted,
            user_id=user_id,
            ai_response=reply,
            is_proactive=False,
            message_id=chat_msg_id,
        )

        # Story cache is NOT cleared here — _compute_data_hash in the story
        # endpoint already detects staleness by comparing hash fingerprints.
        # Keeping old cache avoids unnecessary LLM regeneration when the user
        # happens to view the story page shortly after chatting.

        # ── Trajectory graph: incremental ingest → detect → persist ──────
        trajectory_summary = {}
        try:
            graph = TrajectoryGraph()
            all_msgs = storage.list_messages(limit=500, user_id=user_id)
            trajectory_summary = graph.incremental_ingest(all_msgs, user_id)
        except Exception as e:
            logger.warning("Trajectory detection failed (non-critical): %s", e)
            trajectory_summary = {"error": str(e)}

        # ── Psychological extraction: extract psycho signals → persist ───
        psycho_event = None
        psycho_state = None
        try:
            psycho_raw = extract_psycho_data(combined_message)
            if any(psycho_raw.get(d) is not None for d in
                   ("willingness", "fatigue", "stress", "procrastination",
                    "achievement", "frustration", "guilt", "next_confidence", "avoidance")):
                event = PsychoEvent.from_extracted(
                    psycho_raw, user_id=user_id,
                    message_id=chat_msg_id,
                )
                from storage import append_psycho_event, load_psycho_state, save_psycho_state
                append_psycho_event(user_id, event.to_dict())
                psycho_event = event.to_dict()

                # Update aggregated PsychoState
                existing = load_psycho_state(user_id)
                if existing:
                    state = PsychoState.from_dict(existing)
                else:
                    state = PsychoState.fresh(user_id)
                state.update_from_event(event)
                save_psycho_state(user_id, state.to_dict())
                psycho_state = {
                    "intervenable_state": state.intervenable_state,
                    "n_events": state.n_events_aggregated,
                    "ema": state.ema_dict(),
                }
        except Exception as e:
            logger.warning("Psycho extraction failed (non-critical): %s", e)
            trajectory_summary = {"error": str(e)}

        # Build response with optional plan_action notification
        response_data = {
            "id": msg["id"],
            "extracted": extracted,
            "response": reply,
            "messages": reply if isinstance(reply, list) else [reply],
            "user_state": state,
            "trajectory": trajectory_summary,
            "psycho": psycho_event,
            "psycho_state": psycho_state,
        }
        if plan_id:
            response_data["plan_action"] = {
                "type": "plan_created",
                "plan_id": plan_id,
                "message": "已为你创建推理分析计划，点击图谱查看",
            }
        return jsonify(response_data)

    except LLMError as e:
        logger.error("Chat LLM error: %s", e)
        return jsonify({"error": f"AI service unavailable: {e}"}), 503
    except Exception as e:
        logger.error("Chat endpoint error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


def _auto_create_plan_on_request(message: str, user_id: str, extracted: dict, persona_id: str = None) -> str | None:
    """Detect if user is requesting a health plan and auto-generate one.

    Returns the plan_id if a new plan was created, None otherwise.
    """
    import re
    plan_keywords = [
        "制定计划", "健康计划", "帮我规划", "调理方案",
        "该怎么做", "有什么建议", "帮我制定", "制定一个",
        "做个计划", "出个方案", "健康方案",
    ]
    text = message.lower().strip()
    if not any(k in text for k in plan_keywords):
        return None

    existing = get_active_plan(user_id)
    if existing:
        from datetime import datetime, timezone, timedelta
        created = datetime.fromisoformat(existing.created_at)
        if datetime.now(timezone.utc) - created < timedelta(hours=1):
            logger.info("Plan generation skipped: active plan created within last hour")
            return None

    try:
        from routine import build_routine_model
        import storage
        messages = storage.list_messages(limit=200, user_id=user_id)
        routine_model = build_routine_model(messages)
        routine_summary = routine_model.get_routine_summary()

        user_state_data = get_user_state(user_id)
        profile_notes = user_state_data.get("profile_notes", "")

        from persona import get_active_persona
        persona = get_active_persona(user_id)
        if persona_id and persona_id != persona.get("id"):
            from persona import get_persona
            persona = get_persona(persona_id) or persona

        from insights import compute_correlations, get_trend_insights
        insights_list = compute_correlations(messages) + get_trend_insights(messages)
        insights_str = "\n".join(str(i) for i in insights_list[:5]) if insights_list else ""

        from weather import get_user_location, get_weather_context
        loc = get_user_location(user_id)
        weather_ctx = get_weather_context(loc) if loc else ""

        # Build reasoning tree: expand root node
        from health_plan import ReasoningNode
        from datetime import datetime, timezone
        import uuid

        root_node = ReasoningNode(
            id=str(uuid.uuid4()),
            label=message[:50],
            description=f"针对「{message}」的健康推理",
            emoji="🎯",
            depth=0,
            parent_id=None,
            expanded=False,
            expandable=True,
            is_actionable=False,
        )

        user_context = {
            "concern": message,
            "routine_summary": routine_summary,
            "profile_notes": profile_notes,
            "insights": insights_str,
            "weather_context": weather_ctx or "",
        }

        children = expand_reasoning_node(
            node_dict={"id": root_node.id, "label": root_node.label, "depth": 0, "parent_id": None},
            user_context=user_context,
        )

        if children:
            root_node.expanded = True
            for child_data in children:
                child_node = ReasoningNode(
                    id=child_data["id"],
                    label=child_data["label"],
                    description=child_data.get("description", ""),
                    emoji=child_data.get("emoji", "🔍"),
                    depth=child_data["depth"],
                    parent_id=child_data["parent_id"],
                    expanded=False,
                    expandable=not child_data.get("is_actionable", False),
                    is_actionable=child_data.get("is_actionable", False),
                    action_text=child_data.get("action_text"),
                    time_slot=child_data.get("time_slot"),
                    frequency=child_data.get("frequency"),
                    duration=child_data.get("duration"),
                )
                root_node.children.append(child_node)

        create_plan(
            user_id=user_id,
            concern=message,
            title=f"健康计划",
            goal=f"改善健康状态",
            reasoning_tree=root_node,
            context_snapshot={
                "persona_id": persona.get("id", "default"),
                "routine_summary": routine_summary,
                "profile_notes": profile_notes,
            },
        )
        logger.info("Auto-generated health plan %s for user=%s concern=%s", plan.id, user_id, message[:50])
        return plan.id
    except Exception as e:
        logger.warning("Auto plan generation failed (non-critical): %s", e)
        return None


@app.route('/api/messages')
@require_login
def get_messages():
    """
    GET /api/messages?limit=20&offset=0&period=week
    Returns: [{"role": "user"|"ai", "content": "...", ...}, ...]
    Each stored message produces two entries: user (right) + ai (left).
    """
    try:
        limit = int(flask_request.args.get('limit', 20))
        offset = int(flask_request.args.get('offset', 0))
        period = flask_request.args.get('period', None)
        user_id = current_user_id()

        if period and period not in ('day', 'week', 'month'):
            return jsonify({"error": "period must be 'day', 'week', or 'month'"}), 400

        raw_messages = storage.list_messages(limit=limit * 2, offset=offset, period=period, user_id=user_id)

        # list_messages returns newest-first; reverse to oldest-first so each
        # message's user→ai pair is in correct chronological order in the UI.
        raw_messages.reverse()

        # Transform: skip proactive marker rows and keep user/ai pairs for normal chat only
        formatted = []
        for m in raw_messages:
            is_proactive = bool(m.get("is_proactive"))
            # Backward compatibility: legacy messages stored with bracket-filler
            # content like "[主动关心]" before the is_proactive field existed.
            if not is_proactive and isinstance(m.get("content"), str):
                stripped = m["content"].strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    is_proactive = True
            if not is_proactive:
                formatted.append({
                    "id": m["id"],
                    "role": "user",
                    "content": m["content"],
                    "extracted_data": m.get("extracted_data", {}),
                    "timestamp": m["timestamp"],
                    "user_id": m.get("user_id", "default"),
                })
            if m.get("ai_response"):
                ai_response = m["ai_response"]
                if isinstance(ai_response, list):
                    for idx, part in enumerate(ai_response):
                        formatted.append({
                            "id": f"{m['id']}-ai-{idx}",
                            "role": "ai",
                            "content": part,
                            "timestamp": m["timestamp"],
                        })
                else:
                    formatted.append({
                        "id": m["id"] + "-ai",
                        "role": "ai",
                        "content": ai_response,
                        "timestamp": m["timestamp"],
                    })

        return jsonify(formatted)

    except Exception as e:
        logger.error("Messages endpoint error: %s", e)
        return jsonify({"error": str(e)}), 500


def _count_health_messages(records: list) -> int:
    """Count messages containing any extracted health data."""
    count = 0
    for r in records:
        ed = r.get("extracted_data", {})
        if not isinstance(ed, dict):
            continue
        has_data = False
        sleep = ed.get("sleep", {})
        if isinstance(sleep, dict) and (sleep.get("quality") is not None or sleep.get("duration_hours") is not None):
            has_data = True
        if ed.get("energy") is not None:
            has_data = True
        if ed.get("mood") is not None:
            has_data = True
        exercise = ed.get("exercise", {})
        if isinstance(exercise, dict) and exercise.get("type") is not None:
            has_data = True
        digestion = ed.get("digestion", {})
        if isinstance(digestion, dict) and digestion.get("status") is not None:
            has_data = True
        if has_data:
            count += 1
    return count


def _build_dimension_sources(records: list) -> dict:
    """Build per-dimension source message lists for the story UI.

    Returns a dict like:
        {"sleep": [{"ts":"...","content":"...","value":3},...], "energy": [...], ...}
    """
    import re
    sources = {
        "sleep": [],
        "energy": [],
        "mood": [],
        "exercise": [],
        "digestion": [],
        "diet": [],
    }
    for r in records:
        ed = r.get("extracted_data", {})
        if not isinstance(ed, dict):
            continue
        ts = r.get("timestamp", "")
        content = (r.get("content") or "")[:120]
        # Remove [主动关心] filler
        if content and content.startswith("[") and content.endswith("]"):
            continue

        s = ed.get("sleep", {})
        if isinstance(s, dict):
            q = s.get("quality")
            d = s.get("duration_hours")
            if q is not None:
                sources["sleep"].append({"ts": ts, "content": content, "value": q, "label": "质量"})
            if d is not None:
                sources["sleep"].append({"ts": ts, "content": content, "value": d, "label": "时长(h)"})
        if ed.get("energy") is not None:
            energy_val = ed["energy"]
            if isinstance(energy_val, dict):
                energy_score = energy_val.get("score")
            else:
                energy_score = energy_val
            if energy_score is not None:
                sources["energy"].append({"ts": ts, "content": content, "value": energy_score})
        if ed.get("mood") is not None:
            mood_val = ed["mood"]
            if isinstance(mood_val, dict):
                mood_score = mood_val.get("score")
            else:
                mood_score = mood_val
            if mood_score is not None:
                sources["mood"].append({"ts": ts, "content": content, "value": mood_score})
        ex = ed.get("exercise", {})
        if isinstance(ex, dict) and ex.get("type"):
            sources["exercise"].append({"ts": ts, "content": content, "value": ex.get("type"), "duration": ex.get("duration_minutes")})
        dig = ed.get("digestion", {})
        if isinstance(dig, dict) and dig.get("status"):
            sources["digestion"].append({"ts": ts, "content": content, "value": dig.get("status"), "severity": dig.get("severity")})
        diet = ed.get("diet", {})
        if isinstance(diet, dict) and diet.get("notes"):
            sources["diet"].append({"ts": ts, "content": content, "value": diet.get("notes")})
    return sources


@app.route('/api/story')
@require_login
def body_story():
    """
    GET /api/story?period=week
    Returns: {"period": "week", "narrative": "...", "data_points": [...], "insights": [...]}

    Stories are cached per (user_id, period) with a data hash so they
    only regenerate when the underlying messages change.

    Generation policy:
    - If a valid cache hit → serve immediately (no LLM call).
    - If cache miss and >= 5 health-data messages → generate on-demand (first visit).
    - If cache miss and < 5 health-data messages → return fallback template (no LLM call).
    - The weekly scheduler (story_scheduler_daemon.py) handles the bulk of pre-generation.
    """
    try:
        period = flask_request.args.get('period', 'week')
        if period not in ('day', 'week', 'month'):
            period = 'week'
        user_id = current_user_id()

        PERIOD_LABELS = {"week": "本周", "month": "本月"}
        MIN_HEALTH_MSGS = 5
        FALLBACK_TMPL = """{period_label}你还没有足够的健康记录呢～

目前只收集到 {count} 条有效数据，需要至少 {min_count} 条才能生成完整的身体故事。

继续和我说说你的睡眠、运动、饮食、情绪吧，我会在每周日凌晨自动为你生成～

小声说：哪怕只是随口聊几句"今天好累"或"昨晚没睡好"，我都能从中提取有用的信息哦 😊"""

        # Fetch records for the period
        records = storage.list_messages(limit=200, period=period, user_id=user_id)

        # ── Build dimension source messages (always, even for cache hits) ──
        dim_sources = _build_dimension_sources(records)

        # ── Fetch user state baselines ──
        user_state = get_user_state(user_id)
        baselines = {
            "sleep": user_state.get("sleep_baseline"),
            "energy": user_state.get("energy_baseline"),
            "mood": user_state.get("mood_baseline"),
            "sleep_trend": user_state.get("recent_sleep_trend"),
            "energy_trend": user_state.get("recent_energy_trend"),
            "mood_trend": user_state.get("recent_mood_trend"),
        }

        # Compute data hash — if unchanged, return cached story
        data_hash = storage._compute_data_hash(records)
        cached = storage.get_story_cache(user_id, period)
        if cached and cached.get("data_hash") == data_hash:
            logger.info("Story cache hit for user=%s period=%s", user_id, period)
            return jsonify({
                "period": period,
                "narrative": cached["narrative"],
                "title": cached.get("title", ""),
                "stats": cached.get("stats", ""),
                "story_text": cached.get("story_text", cached.get("narrative", "")),
                "tips": cached.get("tips", ""),
                "data_points": cached.get("data_points", []),
                "insights": cached.get("insights", []),
                "baselines": baselines,
                "dim_sources": dim_sources,
                "generated_at": cached.get("generated_at"),
                "from_cache": True,
                "_fallback": cached.get("_fallback", False),
            })

        # ── Cache miss — count health messages to decide path ──
        health_count = _count_health_messages(records)

        # Not enough data → fallback template (no LLM)
        if health_count < MIN_HEALTH_MSGS:
            label = PERIOD_LABELS.get(period, period)
            fallback_narrative = FALLBACK_TMPL.format(
                period_label=label, count=health_count, min_count=MIN_HEALTH_MSGS
            )
            result = {
                "data_hash": data_hash,
                "narrative": fallback_narrative,
                "data_points": [],
                "insights": [],
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "_fallback": True,
                "_health_count": health_count,
            }
            storage.set_story_cache(user_id, period, result)
            logger.info("Story fallback for user=%s period=%s (%d health msgs < %d)",
                        user_id, period, health_count, MIN_HEALTH_MSGS)
            return jsonify({
                "period": period,
                "narrative": fallback_narrative,
                "data_points": [],
                "insights": [],
                "baselines": baselines,
                "dim_sources": dim_sources,
                "generated_at": result["generated_at"],
                "from_cache": False,
                "_fallback": True,
            })

        # Enough data → generate via LLM
        logger.info("Story generating for user=%s period=%s (%d records)", user_id, period, len(records))

        correlations = compute_correlations(records)
        trend_insights = get_trend_insights(records)
        all_insights = correlations + trend_insights

        # Get persona for voice-aware story generation
        persona = get_active_persona(user_id)

        narrative = generate_body_story(records, period, all_insights,
                                        baselines=baselines, persona=persona)

        # Parse structured narrative into sections
        parsed = _parse_story_response(narrative)

        data_points = []
        for r in records:
            dp = {
                "timestamp": r["timestamp"],
                "content": r["content"],
                "extracted": r.get("extracted_data", {}),
            }
            data_points.append(dp)

        result = {
            "data_hash": data_hash,
            "narrative": narrative,
            "title": parsed.get("title", ""),
            "stats": parsed.get("stats", ""),
            "story_text": parsed.get("narrative", ""),
            "tips": parsed.get("tips", ""),
            "data_points": data_points,
            "insights": all_insights,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "_fallback": False,
        }

        storage.set_story_cache(user_id, period, result)

        return jsonify({
            "period": period,
            "narrative": narrative,
            "title": parsed.get("title", ""),
            "stats": parsed.get("stats", ""),
            "story_text": parsed.get("narrative", ""),
            "tips": parsed.get("tips", ""),
            "data_points": data_points,
            "insights": all_insights,
            "baselines": baselines,
            "dim_sources": dim_sources,
            "generated_at": result["generated_at"],
            "from_cache": False,
            "_fallback": False,
        })

    except LLMError as e:
        logger.error("Story LLM error: %s", e)
        return jsonify({"error": f"AI service unavailable: {e}"}), 503
    except Exception as e:
        logger.error("Story endpoint error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Phase 2: Routine & Insights ─────────────────────────────────────────────

@app.route('/api/routine')
@require_login
def get_routine():
    """
    GET /api/routine
    Returns the current routine model summary.
    """
    try:
        messages = storage.list_messages(limit=500, user_id=current_user_id())
        model = build_routine_model(messages)
        summary = model.get_routine_summary()
        return jsonify(summary)
    except Exception as e:
        logger.error("Routine endpoint error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/anomalies')
@require_login
def get_anomalies():
    """
    GET /api/anomalies
    Detect anomalies in recent behavior.
    """
    try:
        messages = storage.list_messages(limit=200, user_id=current_user_id())
        model = build_routine_model(messages)
        anomalies = model.detect_anomalies()
        return jsonify({
            "anomalies": anomalies,
            "total": len(anomalies),
        })
    except Exception as e:
        logger.error("Anomalies endpoint error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/data/fix', methods=['POST'])
@require_login
def fix_data():
    """
    POST /api/data/fix
    Normalize all stored extracted_data to the correct nested structure.
    Returns count of fixed records.
    """
    try:
        user_id = current_user_id()
        messages = storage._read_all()
        fixed_count = 0
        scoped_total = 0
        for msg in messages:
            if msg.get("user_id") != user_id:
                continue
            scoped_total += 1
            ed = msg.get("extracted_data")
            if ed and isinstance(ed, dict):
                normalized = normalize_extracted(ed)
                # Check if anything changed
                if normalized != ed:
                    msg["extracted_data"] = normalized
                    fixed_count += 1

        if fixed_count > 0:
            storage._write_all(messages)

        return jsonify({
            "fixed": fixed_count,
            "total": scoped_total,
            "message": f"已修复 {fixed_count}/{scoped_total} 条当前账号记录的数据结构",
        })
    except Exception as e:
        logger.error("Data fix endpoint error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Phase 2: Persona Management ─────────────────────────────────────────────

@app.route('/api/personas')
@require_login
def list_personas():
    """GET /api/personas — List all available personas."""
    try:
        personas = get_available_personas()
        active = get_active_persona(current_user_id())
        return jsonify({"personas": personas, "active_id": active["id"]})
    except Exception as e:
        logger.error("Personas endpoint error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/persona')
@require_login
def get_persona_route():
    """GET /api/persona — Get current active persona."""
    try:
        persona = get_active_persona(current_user_id())
        return jsonify(persona)
    except Exception as e:
        logger.error("Get persona error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/persona', methods=['POST'])
@require_login
def set_persona():
    """POST /api/persona — Set active persona. Body: {"persona_id": "fitness_coach"}"""
    try:
        data = flask_request.json or {}
        persona_id = data.get('persona_id', 'default')
        persona = set_active_persona(persona_id, current_user_id())
        return jsonify(persona)
    except Exception as e:
        logger.error("Set persona error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Phase 2: Weather Awareness ──────────────────────────────────────────────

@app.route('/api/location')
@require_login
def get_location():
    """GET /api/location — Get current user location."""
    try:
        loc = get_user_location(current_user_id())
        weather = get_weather_context(loc)
        return jsonify({"location": loc, "weather": weather})
    except Exception as e:
        logger.error("Get location error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/location', methods=['POST'])
@require_login
def set_location():
    """POST /api/location — Set user location. Body: {"lat": 31.2, "lon": 121.5, "city": "上海"}"""
    try:
        data = flask_request.json or {}
        lat = float(data.get('lat', 0))
        lon = float(data.get('lon', 0))
        city = data.get('city', '')
        loc = set_user_location(lat, lon, city, current_user_id())
        return jsonify(loc)
    except Exception as e:
        logger.error("Set location error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/location', methods=['DELETE'])
@require_login
def delete_location():
    """DELETE /api/location — Remove user location."""
    try:
        clear_location(current_user_id())
        return jsonify({"success": True})
    except Exception as e:
        logger.error("Delete location error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Phase 2: Proactive Scheduler ─────────────────────────────────────────────

@app.route('/api/scheduler')
@require_login
def scheduler_status():
    """GET /api/scheduler — Get scheduler status."""
    try:
        user_id = current_user_id()
        return jsonify(get_scheduler_status(user_id))
    except Exception as e:
        logger.error("Scheduler status error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/scheduler/silent', methods=['POST'])
@require_login
def set_silent_mode():
    """POST /api/scheduler/silent — Enable/disable silent mode (per user)."""
    try:
        data = flask_request.json or {}
        user_id = current_user_id()
        if data.get('enable', True):
            hours = data.get('hours', 24)
            until = enable_silent_mode(user_id, hours)
            return jsonify({"silent": True, "silent_until": until})
        else:
            disable_silent_mode(user_id)
            return jsonify({"silent": False})
    except Exception as e:
        logger.error("Silent mode error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/scheduler/check', methods=['POST'])
@require_login
def trigger_proactive_check():
    """
    POST /api/scheduler/check — Manually trigger proactive check.
    Returns a proactive message if anomalies detected and conditions met.
    Body: {"user_id": "xxx"}
    """
    try:
        data = flask_request.json or {}
        user_id = current_user_id()
        force_test = bool(data.get("force"))

        # Check if we should send a proactive message
        if force_test:
            decision = {"should_send": True, "reason": "forced_test"}
        else:
            state = SchedulerState.load(user_id)
            from storage import load_psycho_state
            psycho_state = load_psycho_state(user_id)
            decision = should_send_proactive(state, user_id, psycho_state=psycho_state)

            if not decision["should_send"]:
                return jsonify({
                    "proactive": False,
                    "reason": decision["reason"],
                    "message": None,
                })

        # Get recent messages for anomaly detection (filter by user)
        messages = storage.list_messages(limit=50, user_id=user_id)
        persona_id = data.get("persona_id") or current_user().get("persona_id")

        # Generate proactive message
        if force_test:
            msg = generate_chat_response(
                message=data.get("message") or "测试主动关心",
                extracted={},
                context=messages[-5:],
                persona_id=persona_id,
                user_id=user_id,
                user_state=build_chat_user_state(user_id, persona_id),
                insights=[],
                proactive=True,
                burst=True,
            )
            if isinstance(msg, str):
                msg = msg.strip()
            elif isinstance(msg, list):
                msg = [str(item).strip() for item in msg if str(item).strip()]
            if not msg:
                msg = [
                    "我来主动关心一下你，最近还好吗？",
                    "如果今天有点累，先休息一下也没关系，我在。",
                ]
        else:
            msg = generate_proactive_message(messages, persona_id=persona_id, user_id=user_id)

        if msg:
            # Store the proactive message in storage
            storage.create_message(
                content="",
                extracted_data={},
                user_id=user_id,
                ai_response=msg,
                is_proactive=True,
            )
            record_proactive_sent(user_id)

            return jsonify({
                "proactive": True,
                "reason": decision["reason"],
                "message": msg,
                "test_mode": force_test,
            })
        else:
            return jsonify({
                "proactive": False,
                "reason": "no_anomalies",
                "message": None,
                "test_mode": force_test,
            })

    except Exception as e:
        logger.error("Proactive check error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Account Management ──────────────────────────────────────────────────────

@app.route('/api/accounts', methods=['GET'])
@require_login
def api_list_accounts():
    """GET /api/accounts — Return the current account only."""
    try:
        account = current_user()
        account_copy = public_account(account)
        if account_copy:
            account_copy["activity"] = compute_activity_score(account_copy["user_id"])
            return jsonify([account_copy])
        return jsonify([])
    except Exception as e:
        logger.error("List accounts error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/accounts', methods=['POST'])
@require_login
def api_create_account():
    """POST /api/accounts — Deprecated, returns the current account."""
    try:
        account = public_account(current_user())
        if account:
            account["activity"] = compute_activity_score(account["user_id"])
            return jsonify(account), 200
        return _json_error("unauthorized", 401)
    except Exception as e:
        logger.error("Create account error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/accounts/<user_id>', methods=['GET'])
@require_login
def api_get_account(user_id):
    """GET /api/accounts/<user_id> — Get account info."""
    try:
        if user_id != current_user_id():
            return _json_error("forbidden", 403)
        account = get_account(user_id)
        if not account:
            return jsonify({"error": "Account not found"}), 404
        activity = compute_activity_score(user_id)
        account["activity"] = activity
        return jsonify(public_account(account))
    except Exception as e:
        logger.error("Get account error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/accounts/<user_id>', methods=['PUT'])
@require_login
def api_update_account(user_id):
    """PUT /api/accounts/<user_id> — Update account."""
    try:
        if user_id != current_user_id():
            return _json_error("forbidden", 403)
        data = flask_request.json or {}
        account = update_account(user_id, **data)
        if not account:
            return jsonify({"error": "Account not found"}), 404
        return jsonify(public_account(account))
    except Exception as e:
        logger.error("Update account error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/accounts/<user_id>/activity', methods=['GET'])
@require_login
def api_account_activity(user_id):
    """GET /api/accounts/<user_id>/activity — Get activity score."""
    try:
        if user_id != current_user_id():
            return _json_error("forbidden", 403)
        activity = compute_activity_score(user_id)
        return jsonify(activity)
    except Exception as e:
        logger.error("Account activity error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/accounts/<user_id>', methods=['DELETE'])
@require_login
def api_delete_account(user_id):
    """DELETE /api/accounts/<user_id> — Delete an account."""
    try:
        if user_id != current_user_id():
            return _json_error("forbidden", 403)
        if user_id == "default":
            return jsonify({"error": "Cannot delete default account or account not found"}), 400
        delete_messages_for_user(user_id)
        delete_scheduler_state(user_id)
        delete_user_persona(user_id)
        delete_user_state(user_id)
        clear_location(user_id)
        if delete_account(user_id):
            _clear_current_session()
            return jsonify({"success": True})
        return jsonify({"error": "Cannot delete default account or account not found"}), 400
    except Exception as e:
        logger.error("Delete account error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Physical Exam Import ─────────────────────────────────────────────────────

@app.route('/api/exam/import', methods=['POST'])
@require_login
def exam_import():
    """
    POST /api/exam/import — Import physical exam JSON and auto-generate graph skeleton.

    Body: raw JSON (the physical exam report).

    Returns: {"success": true, "filename": "exam_<user_id>.json",
              "graph": {"nodes": [...], "relationships": [...]}}
    """
    from exam_graph import build_exam_graph, save_exam_graph, ExamGraphError

    try:
        exam_data = flask_request.get_json(force=True, silent=True)
        if not exam_data or not isinstance(exam_data, dict):
            return jsonify({"error": "请提供有效的体检 JSON 数据"}), 400

        user_id = current_user_id()
        graph = build_exam_graph(exam_data, user_index=1)

        filename = save_exam_graph(graph, user_id=user_id)

        return jsonify({
            "success": True,
            "filename": filename,
            "graph": graph,
            "message": f"体检数据已导入，生成了 {len(graph['nodes'])} 个图谱节点",
        })

    except ExamGraphError as e:
        logger.warning("Exam graph build error: %s", e)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Exam import error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Phase 4: Health Plan ───────────────────────────────────────────────────

@app.route('/api/plan/generate', methods=['POST'])
@require_login
def api_generate_plan():
    """
    POST /api/plan/generate — Generate a new health plan.
    Body: {"concern": "最近脱发好严重"}
    Returns: The generated HealthPlan.
    """
    try:
        data = flask_request.json or {}
        concern = data.get("concern", "").strip()
        if not concern:
            return _json_error("请输入你的健康困扰", 400)

        user_id = current_user_id()
        persona_id = data.get("persona_id") or current_user().get("persona_id")

        # Gather context
        from routine import build_routine_model
        messages = storage.list_messages(limit=200, user_id=user_id)
        routine_model = build_routine_model(messages)
        routine_summary = routine_model.get_routine_summary()

        user_state_data = get_user_state(user_id)
        profile_notes = user_state_data.get("profile_notes", "")

        from persona import get_active_persona, get_persona
        persona = get_active_persona(user_id)
        if persona_id and persona_id != persona.get("id"):
            persona = get_persona(persona_id) or persona

        insights_list = compute_correlations(messages) + get_trend_insights(messages)
        insights_str = "\n".join(str(i) for i in insights_list[:5]) if insights_list else ""

        from weather import get_user_location, get_weather_context
        loc = get_user_location(user_id)
        weather_ctx = get_weather_context(loc) if loc else ""

        # Build reasoning tree
        import uuid
        from health_plan import ReasoningNode

        root_node = ReasoningNode(
            id=str(uuid.uuid4()),
            label=concern[:50],
            description=f"针对「{concern}」的健康推理",
            emoji="🎯",
            depth=0,
            parent_id=None,
            expanded=False,
            expandable=True,
            is_actionable=False,
        )

        user_context = {
            "concern": concern,
            "routine_summary": routine_summary,
            "profile_notes": profile_notes,
            "insights": insights_str,
            "weather_context": weather_ctx or "",
        }

        children = expand_reasoning_node(
            node_dict={"id": root_node.id, "label": root_node.label, "depth": 0, "parent_id": None},
            user_context=user_context,
        )

        if children:
            root_node.expanded = True
            for child_data in children:
                child_node = ReasoningNode(
                    id=child_data["id"],
                    label=child_data["label"],
                    description=child_data.get("description", ""),
                    emoji=child_data.get("emoji", "🔍"),
                    depth=child_data["depth"],
                    parent_id=child_data["parent_id"],
                    expanded=False,
                    expandable=not child_data.get("is_actionable", False),
                    is_actionable=child_data.get("is_actionable", False),
                    action_text=child_data.get("action_text"),
                    time_slot=child_data.get("time_slot"),
                    frequency=child_data.get("frequency"),
                    duration=child_data.get("duration"),
                )
                root_node.children.append(child_node)

        # Save plan
        plan = create_plan(
            user_id=user_id,
            concern=concern,
            title=f"健康计划",
            goal=f"改善{concern}",
            reasoning_tree=root_node,
            context_snapshot={
                "persona_id": persona.get("id", "default"),
                "routine_summary": routine_summary,
                "profile_notes": profile_notes,
            },
        )

        # Also save to chat as AI response
        leaves = collect_actionable_leaves(plan)
        summary = f"已为你创建了「{plan.title}」✨ 基于推理树生成了 {len(leaves)} 条建议"
        if leaves:
            summary += f"\n展开推理树查看完整分析，或查看计划列表了解具体建议～"

        storage.create_message(
            content=f"[自动生成] 创建健康计划：{concern}",
            extracted_data={},
            user_id=user_id,
            ai_response=summary,
            is_proactive=False,
        )

        return jsonify({
            "plan": _plan_to_response(plan),
            "message": summary,
        })

    except Exception as e:
        logger.error("Generate plan error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/plan')
@require_login
def api_get_active_plan():
    """GET /api/plan — Get the current active plan."""
    try:
        plan = get_active_plan(current_user_id())
        if not plan:
            return jsonify({"plan": None})
        return jsonify({"plan": _plan_to_response(plan)})
    except Exception as e:
        logger.error("Get active plan error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/plans')
@require_login
def api_list_plans():
    """GET /api/plans?status=all — List all plans for user."""
    try:
        status = flask_request.args.get('status')
        plans = list_plans(current_user_id(), status=status)
        return jsonify({"plans": [_plan_to_response(p) for p in plans]})
    except Exception as e:
        logger.error("List plans error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/plan/<plan_id>')
@require_login
def api_get_plan(plan_id):
    """GET /api/plan/<plan_id> — Get a specific plan."""
    try:
        plan = get_plan(plan_id, current_user_id())
        if not plan:
            return jsonify({"error": "计划不存在"}), 404
        return jsonify({"plan": _plan_to_response(plan)})
    except Exception as e:
        logger.error("Get plan error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/plan/<plan_id>/status', methods=['PUT'])
@require_login
def api_update_plan_status(plan_id):
    """PUT /api/plan/<plan_id>/status — Update plan status.
    Body: {"status": "active"|"archived"|"completed"}
    """
    try:
        data = flask_request.json or {}
        status = data.get("status", "")
        if status not in ("active", "archived", "completed"):
            return _json_error("status 必须为 active/archived/completed", 400)
        plan = update_plan_status(plan_id, current_user_id(), status)
        if not plan:
            return jsonify({"error": "计划不存在"}), 404
        return jsonify({"plan": _plan_to_response(plan)})
    except Exception as e:
        logger.error("Update plan status error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/plan/<plan_id>', methods=['DELETE'])
@require_login
def api_delete_plan(plan_id):
    """DELETE /api/plan/<plan_id> — Delete a plan."""
    try:
        ok = delete_plan(plan_id, current_user_id())
        if not ok:
            return jsonify({"error": "计划不存在"}), 404
        return jsonify({"success": True})
    except Exception as e:
        logger.error("Delete plan error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/plan/<plan_id>/rename', methods=['PUT'])
@require_login
def api_rename_plan(plan_id):
    """PUT /api/plan/<plan_id>/rename — Rename a plan.
    Body: {"title": "新的计划名称"}
    """
    try:
        data = flask_request.json or {}
        new_title = (data.get("title") or "").strip()
        if not new_title:
            return _json_error("标题不能为空", 400)
        if len(new_title) > 100:
            return _json_error("标题不能超过100个字符", 400)
        plan = rename_plan(plan_id, current_user_id(), new_title)
        if not plan:
            return jsonify({"error": "计划不存在"}), 404
        return jsonify({"plan": _plan_to_response(plan)})
    except Exception as e:
        logger.error("Rename plan error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/plan/<plan_id>/reasoning/expand', methods=['POST'])
@require_login
def api_expand_reasoning_node(plan_id):
    """
    POST /api/plan/<plan_id>/reasoning/expand
    Body: {"node_id": "..."}
    Expands a reasoning node (calls LLM for children).
    Returns updated plan with expanded tree.
    """
    try:
        data = flask_request.json or {}
        node_id = data.get("node_id", "")
        answer = data.get("answer")  # Optional: user's answer to a pending followup question
        if not node_id:
            return _json_error("node_id 是必需的", 400)

        user_id = current_user_id()
        plan = get_plan(plan_id, user_id)
        if not plan:
            return jsonify({"error": "计划不存在"}), 404

        # 记录展开前的可采纳叶子集合，用于识别本次展开新增的可采纳叶子
        # （追问后自动采纳体验: 前端据此提供"全部加入计划"快捷操作）
        before_actionable_ids = {n.id for n in collect_actionable_leaves(plan)}

        if not plan.reasoning_tree:
            return jsonify({"error": "计划没有推理树"}), 400

        # Find the node in the tree
        from health_plan import find_node_in_tree
        node = find_node_in_tree(node_id, plan.reasoning_tree)
        if not node:
            return jsonify({"error": "节点不存在"}), 404

        if not node.expandable:
            return jsonify({"error": "该节点不可展开"}), 400

        if node.expanded:
            # Already expanded, return current plan
            return jsonify({"plan": _plan_to_response(plan)})

        # Gather context for LLM
        context_snapshot = plan.context_snapshot or {}

        user_context = {
            "concern": plan.concern,
            "routine_summary": context_snapshot.get("routine_summary", ""),
            "profile_notes": context_snapshot.get("profile_notes", ""),
            "insights": "",
            "weather_context": "",
        }

        # Build reasoning path from root to current node
        from health_plan import build_path_to_node
        reasoning_path = build_path_to_node(node.id, plan.reasoning_tree)

        # Interactive info-gathering (Phase 4 待完善方向):
        #  - If the node already carries a pending followup_question, the user must
        #    answer before expansion proceeds. Re-surface the question otherwise.
        #  - If no question is pending yet, decide whether one is needed before expanding.
        node_dict = {
            "id": node.id,
            "label": node.label,
            "description": node.description,
            "depth": node.depth,
            "parent_id": node.parent_id,
        }
        if node.followup_question and (answer is None or not str(answer).strip()):
            return jsonify({
                "plan": _plan_to_response(plan),
                "followup_question": node.followup_question,
                "followup_options": node.followup_options,
                "needs_answer": True,
            })

        if not node.followup_question:
            question, options = ask_followup_question(
                node_dict=node_dict,
                user_context=user_context,
                path_from_root=reasoning_path,
            )
            if question:
                updated_plan = set_followup_question(
                    plan_id, user_id, node_id, question, options=options
                )
                if updated_plan:
                    return jsonify({
                        "plan": _plan_to_response(updated_plan),
                        "followup_question": question,
                        "followup_options": options,
                        "needs_answer": True,
                    })

        # Call expand — pass existing sibling nodes to avoid duplicates
        # Find siblings: other children of the same parent (only need labels for dedup)
        existing_sibling_labels = []
        if node.parent_id and plan.reasoning_tree:
            from health_plan import find_node_in_tree
            parent_node = find_node_in_tree(node.parent_id, plan.reasoning_tree)
            if parent_node:
                for sibling in parent_node.children:
                    if sibling.id != node.id:
                        existing_sibling_labels.append(sibling.label)

        children = expand_reasoning_node(
            node_dict=node_dict,
            user_context=user_context,
            existing_nodes=[{"label": lbl} for lbl in existing_sibling_labels],
            path_from_root=reasoning_path,
            user_answer=str(answer).strip() if answer is not None else None,
        )

        if not children:
            logger.warning(
                "expand_reasoning_node returned empty for node_id=%s node_label=%s plan_id=%s user_id=%s",
                node_id, node.label if node else "?", plan_id, user_id,
            )
            return jsonify({"error": "无法展开该节点，请稍后重试"}), 500

        # Ensure expandable flag for non-actionable children
        for child in children:
            child.setdefault("expandable", not child.get("is_actionable", False))

        # Save via expand_node which handles tree mutation + persistence
        plan = expand_node(plan_id, user_id, node_id, children)
        if not plan:
            return jsonify({"error": "保存计划失败"}), 500

        # 识别本次展开新增的可采纳叶子（追问答完后直接展开也会落在这里）
        new_actionable_leaves = list_new_actionable_leaves(plan, before_actionable_ids)

        return jsonify({
            "plan": _plan_to_response(plan),
            "new_actionable_leaves": new_actionable_leaves,
        })

    except Exception as e:
        logger.error("Expand reasoning node error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/plan/<plan_id>/reasoning/adopt', methods=['POST'])
@require_login
def api_adopt_reasoning_leaf(plan_id):
    """
    POST /api/plan/<plan_id>/reasoning/adopt
    Body: {"node_id": "..."}
    Adopt an actionable leaf node (mark as adopted/in-plan).
    """
    try:
        data = flask_request.json or {}
        node_id = data.get("node_id", "")
        if not node_id:
            return _json_error("node_id 是必需的", 400)

        plan = adopt_leaf(plan_id, current_user_id(), node_id)
        if not plan:
            return jsonify({"error": "计划或节点不存在"}), 404
        return jsonify({"plan": _plan_to_response(plan)})
    except Exception as e:
        logger.error("Adopt leaf error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/plan/<plan_id>/items/<item_id>', methods=['PUT'])
@require_login
def api_update_item(plan_id, item_id):
    """PUT /api/plan/<plan_id>/items/<item_id> — Update item status.
    Body: {"status": "done"|"pending"|"skipped"}
    """
    try:
        data = flask_request.json or {}
        status = data.get("status", "")
        if status not in ("done", "pending", "skipped"):
            return _json_error("status 无效", 400)
        plan = update_leaf_status(plan_id, current_user_id(), item_id, status)
        if not plan:
            return jsonify({"error": "计划或项目不存在"}), 404
        return jsonify({"plan": _plan_to_response(plan)})
    except Exception as e:
        logger.error("Update item error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/plan/<plan_id>/regenerate', methods=['POST'])
@require_login
def api_regenerate_plan(plan_id):
    """POST /api/plan/<plan_id>/regenerate — Regenerate plan with latest context."""
    try:
        user_id = current_user_id()
        old_plan = get_plan(plan_id, user_id)
        if not old_plan:
            return jsonify({"error": "计划不存在"}), 404

        concern = old_plan.concern

        # Gather fresh context
        from routine import build_routine_model
        messages = storage.list_messages(limit=200, user_id=user_id)
        routine_model = build_routine_model(messages)
        routine_summary = routine_model.get_routine_summary()

        user_state_data = get_user_state(user_id)
        profile_notes = user_state_data.get("profile_notes", "")

        persona_id = old_plan.context_snapshot.get("persona_id") if old_plan.context_snapshot else None
        from persona import get_active_persona, get_persona
        persona = get_active_persona(user_id)
        if persona_id and persona_id != persona.get("id"):
            persona = get_persona(persona_id) or persona

        insights_list = compute_correlations(messages) + get_trend_insights(messages)
        insights_str = "\n".join(str(i) for i in insights_list[:5]) if insights_list else ""

        from weather import get_user_location, get_weather_context
        loc = get_user_location(user_id)
        weather_ctx = get_weather_context(loc) if loc else ""

        # Rebuild reasoning tree
        import uuid
        from health_plan import ReasoningNode

        root_node = ReasoningNode(
            id=str(uuid.uuid4()),
            label=concern[:50],
            description=f"针对「{concern}」的健康推理",
            emoji="🎯",
            depth=0,
            parent_id=None,
            expanded=False,
            expandable=True,
            is_actionable=False,
        )

        user_context = {
            "concern": concern,
            "routine_summary": routine_summary,
            "profile_notes": profile_notes,
            "insights": insights_str,
            "weather_context": weather_ctx or "",
        }

        children = expand_reasoning_node(
            node_dict={"id": root_node.id, "label": root_node.label, "depth": 0, "parent_id": None},
            user_context=user_context,
        )

        if children:
            root_node.expanded = True
            for child_data in children:
                child_node = ReasoningNode(
                    id=child_data["id"],
                    label=child_data["label"],
                    description=child_data.get("description", ""),
                    emoji=child_data.get("emoji", "🔍"),
                    depth=child_data["depth"],
                    parent_id=child_data["parent_id"],
                    expanded=False,
                    expandable=not child_data.get("is_actionable", False),
                    is_actionable=child_data.get("is_actionable", False),
                    action_text=child_data.get("action_text"),
                    time_slot=child_data.get("time_slot"),
                    frequency=child_data.get("frequency"),
                    duration=child_data.get("duration"),
                )
                root_node.children.append(child_node)

        plan = regenerate_plan(
            plan_id=plan_id,
            user_id=user_id,
            reasoning_tree=root_node,
            new_context={
                "persona_id": persona.get("id", "default"),
                "routine_summary": routine_summary,
                "profile_notes": profile_notes,
            },
        )
        if not plan:
            return jsonify({"error": "计划不存在"}), 404
        return jsonify({"plan": _plan_to_response(plan)})

    except Exception as e:
        logger.error("Regenerate plan error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Plan Reminders ──────────────────────────────────────────────────────────

@app.route('/api/plan/reminder', methods=['POST'])
@require_login
def api_create_reminder():
    """POST /api/plan/reminder — Create a reminder.
    Body: {"action": "跑步", "time_spec": "15:00", "frequency": "每天", "plan_id": "..."}
    """
    try:
        data = flask_request.json or {}
        action = data.get("action", "").strip()
        time_spec = data.get("time_spec", "09:00")
        plan_id = data.get("plan_id", "")
        days = data.get("days_of_week")

        if not action:
            return _json_error("请指定提醒内容", 400)

        message = f"该{action}啦！" if action else "该行动啦～"
        reminder = create_reminder(
            user_id=current_user_id(),
            plan_id=plan_id,
            time_spec=time_spec,
            message_template=message,
            days_of_week=days,
            adaptive=True,
        )

        return jsonify({"reminder": {
            "id": reminder.id,
            "action": action,
            "time_spec": reminder.time_spec,
            "days_of_week": reminder.days_of_week,
            "is_active": reminder.is_active,
            "message_template": reminder.message_template,
        }})

    except Exception as e:
        logger.error("Create reminder error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/plan/reminders')
@require_login
def api_list_reminders():
    """GET /api/plan/reminders — List all reminders for current user."""
    try:
        reminders = list_reminders(current_user_id())
        return jsonify({"reminders": [
            {
                "id": r.id,
                "plan_id": r.plan_id,
                "time_spec": r.time_spec,
                "days_of_week": r.days_of_week,
                "message_template": r.message_template,
                "is_active": r.is_active,
                "adaptive": r.adaptive,
            }
            for r in reminders
        ]})
    except Exception as e:
        logger.error("List reminders error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/plan/reminder/<reminder_id>', methods=['DELETE'])
@require_login
def api_delete_reminder(reminder_id):
    """DELETE /api/plan/reminder/<reminder_id> — Delete a reminder."""
    try:
        ok = delete_reminder(reminder_id, current_user_id())
        if not ok:
            return jsonify({"error": "提醒不存在"}), 404
        return jsonify({"success": True})
    except Exception as e:
        logger.error("Delete reminder error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Plan Helper ─────────────────────────────────────────────────────────────

def _plan_to_response(plan) -> dict:
    """Convert a HealthPlan object to a JSON-serializable dict."""
    if plan is None:
        return {}

    def serialize_node(node) -> dict:
        """Recursively serialize a ReasoningNode to dict."""
        result = {
            "id": node.id,
            "label": node.label,
            "description": node.description,
            "emoji": node.emoji,
            "depth": node.depth,
            "parent_id": node.parent_id,
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
            "children": [serialize_node(c) for c in node.children],
        }
        return result

    # Collect actionable leaves for the plan summary
    leaves = collect_actionable_leaves(plan)

    result = {
        "id": plan.id,
        "user_id": plan.user_id,
        "concern": plan.concern,
        "title": plan.title,
        "goal": plan.goal,
        "reasoning_tree": serialize_node(plan.reasoning_tree) if plan.reasoning_tree else None,
        "actionable_items": [serialize_node(leaf) for leaf in leaves],
        "item_count": len(leaves),
        "completed_count": sum(1 for l in leaves if l.status == "done"),
        "status": plan.status,
        "created_at": plan.created_at,
        "updated_at": plan.updated_at,
        "context_snapshot": plan.context_snapshot,
    }
    return result


@app.route('/api/exam/graph')
@require_login
def exam_graph():
    """
    GET /api/exam/graph — Get the previously imported exam graph skeleton.
    Returns null graph if no exam has been imported.
    """
    from exam_graph import load_exam_graph

    try:
        user_id = current_user_id()
        graph = load_exam_graph(user_id=user_id)
        if graph is None:
            return jsonify({"graph": None, "message": "尚未导入体检数据"})
        return jsonify({
            "graph": graph,
            "node_count": len(graph["nodes"]),
            "relationship_count": len(graph["relationships"]),
        })
    except Exception as e:
        logger.error("Exam graph fetch error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Health Correlation Graph ─────────────────────────────────────────────────

@app.route('/api/health-graph')
@require_login
def health_graph():
    """
    GET /api/health-graph — Get health dimension correlation graph.
    Optionally merges exam graph skeleton when ?merge_exam=true is passed.

    Returns graph data with nodes=dimensions, edges=correlations,
    plus exam nodes/edges when available and merged.
    """
    from exam_graph import load_exam_graph

    try:
        period = flask_request.args.get('period', 'week')
        merge_exam = flask_request.args.get('merge_exam', 'false').lower() == 'true'
        messages = storage.list_messages(limit=200, period=period, user_id=current_user_id())

        # Compute correlations
        correlations = compute_correlations(messages)

        # Generate graph
        graph = generate_health_graph(correlations=correlations, messages=messages)

        # Optionally merge exam skeleton
        if merge_exam:
            exam = load_exam_graph(user_id=current_user_id())
            if exam:
                graph["nodes"].extend(exam.get("nodes", []))
                graph["relationships"].extend(exam.get("relationships", []))

        return jsonify(graph)
    except Exception as e:
        logger.error("Health graph error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/health-graph/points')
@require_login
def health_graph_points():
    """
    GET /api/health-graph/points?dimension=<dim_id>&period=<period>
    Returns time-ordered list of data points for a health dimension.
    """
    try:
        period = flask_request.args.get('period', 'week')
        dimension = flask_request.args.get('dimension', '')
        messages = storage.list_messages(limit=500, period=period, user_id=current_user_id())

        # Define which fields to check for each dimension
        DIM_FIELDS = {
            'sleep': ['quality', 'duration_hours'],
            'energy': ['score'],
            'mood': ['score'],
            'exercise': ['type', 'duration_minutes'],
            'digestion': ['status', 'severity'],
            'skin': ['status'],
            'pain': ['location', 'severity'],
            'diet': ['notes'],
        }

        points = []
        fields = DIM_FIELDS.get(dimension, [])

        for msg in messages:
            ed = msg.get('extracted_data', {})
            if not isinstance(ed, dict):
                continue

            dim_data = ed.get(dimension)
            if dim_data is None:
                continue

            has_value = False
            if fields:
                if isinstance(dim_data, dict):
                    for field in fields:
                        if dim_data.get(field) is not None:
                            has_value = True
                            break
                # Backward compat: old int-format energy/mood
                if isinstance(dim_data, (int, float)) and dim_data is not None:
                    has_value = True
            else:
                # For numeric dimensions, any non-None value counts
                if isinstance(dim_data, (int, float)) and dim_data is not None:
                    has_value = True

            if has_value:
                points.append({
                    'timestamp': msg.get('timestamp', ''),
                    'text': msg.get('text', ''),
                    'value': dim_data,
                })

        # Sort by timestamp descending (newest first)
        points.sort(key=lambda x: x['timestamp'], reverse=True)

        return jsonify({'dimension': dimension, 'points': points})
    except Exception as e:
        logger.error("Health graph points error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/auth/me')
def auth_me():
    user = current_user()
    if not user:
        return _json_error("unauthorized", 401)
    if user.get("is_deleted"):
        _clear_current_session()
        return _json_error("unauthorized", 401)
    payload = public_account(user)
    payload["activity"] = compute_activity_score(user["user_id"])
    return jsonify(payload)


@app.route('/api/auth/register', methods=['POST'])
def auth_register():
    data = flask_request.json or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    display_name = (data.get("display_name") or username or "用户").strip()
    try:
        account = register_account(username=username, password=password, display_name=display_name)
        mark_login(account["user_id"])
        session["user_id"] = account["user_id"]
        payload = public_account(get_account(account["user_id"]))
        payload["activity"] = compute_activity_score(account["user_id"])
        return jsonify(payload), 201
    except ValueError as e:
        return _json_error(str(e), 400)
    except Exception as e:
        logger.error("Auth register error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    data = flask_request.json or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    account = authenticate(username, password)
    if not account:
        return _json_error("invalid_credentials", 401)
    mark_login(account["user_id"])
    session["user_id"] = account["user_id"]
    payload = public_account(get_account(account["user_id"]))
    payload["activity"] = compute_activity_score(account["user_id"])
    return jsonify(payload)


@app.route('/api/auth/logout', methods=['POST'])
def auth_logout():
    _clear_current_session()
    return jsonify({"success": True})


@app.route('/api/auth/delete', methods=['DELETE'])
@require_login
def auth_delete():
    user = current_user()
    user_id = user.get("user_id")
    if not user_id or user_id == "default":
        return _json_error("cannot_delete_default", 400)
    delete_messages_for_user(user_id)
    delete_scheduler_state(user_id)
    delete_user_persona(user_id)
    delete_user_state(user_id)
    clear_location(user_id)
    if delete_account(user_id):
        _clear_current_session()
        return jsonify({"success": True})
    return _json_error("account_not_found", 404)


# ── Feedback / Todo ─────────────────────────────────────────────────────────

@app.route('/api/feedback')
@require_login
def get_feedbacks():
    """GET /api/feedback — List all feedbacks for the current user."""
    try:
        user_id = current_user_id()
        feedbacks = feedback.list_feedbacks(user_id)
        return jsonify(feedbacks)
    except Exception as e:
        logger.error("Get feedbacks error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
@require_login
def create_feedback():
    """POST /api/feedback — Create a new feedback item.
    Body: {"type": "bug|ux", "description": "..."}"""
    try:
        data = flask_request.json
        if not data:
            return jsonify({"error": "Missing request body"}), 400
        fb_type = (data.get("type") or "").strip()
        if fb_type not in ("bug", "ux"):
            return jsonify({"error": "type must be 'bug' or 'ux'"}), 400
        description = (data.get("description") or "").strip()
        if not description:
            return jsonify({"error": "description is required"}), 400
        user_id = current_user_id()
        fb = feedback.create_feedback(user_id, fb_type, description)
        return jsonify(fb), 201
    except Exception as e:
        logger.error("Create feedback error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/api/feedback/<feedback_id>', methods=['PUT'])
@require_login
def update_feedback(feedback_id):
    """PUT /api/feedback/<id> — Update a feedback item.
    Body: {"completed": true/false, "description": "..."}"""
    try:
        data = flask_request.json or {}
        user_id = current_user_id()
        updates = {}
        if "completed" in data:
            updates["completed"] = bool(data["completed"])
        if "description" in data:
            updates["description"] = (data["description"] or "").strip()
        if "type" in data and data["type"] in ("bug", "ux"):
            updates["type"] = data["type"]
        fb = feedback.update_feedback(feedback_id, user_id, **updates)
        if fb is None:
            return jsonify({"error": "Feedback not found"}), 404
        return jsonify(fb)
    except Exception as e:
        logger.error("Update feedback error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/api/feedback/<feedback_id>', methods=['DELETE'])
@require_login
def delete_feedback(feedback_id):
    """DELETE /api/feedback/<id> — Delete a feedback item."""
    try:
        user_id = current_user_id()
        deleted = feedback.delete_feedback(feedback_id, user_id)
        if not deleted:
            return jsonify({"error": "Feedback not found"}), 404
        return jsonify({"success": True})
    except Exception as e:
        logger.error("Delete feedback error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Prompt Trace & Tree APIs ────────────────────────────────────────────────

@app.route('/api/trace/<message_id>')
def get_prompt_trace(message_id):
    """GET /api/trace/<message_id> — Get the prompt trace for a specific message."""
    try:
        from prompt_traces import get_trace, reconstruct_prompt
        trace = get_trace(message_id)
        if not trace:
            return jsonify({"error": "Trace not found"}), 404

        # Add reconstructed prompt for convenience
        trace["reconstructed_prompt"] = reconstruct_prompt(message_id)
        return jsonify(trace)
    except Exception as e:
        logger.error("Get trace error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/traces')
def list_prompt_traces():
    """GET /api/traces?limit=20 — List recent prompt traces."""
    try:
        from prompt_traces import get_recent_traces, get_trace_count
        limit = flask_request.args.get('limit', 20, type=int)
        traces = get_recent_traces(limit=min(limit, 100))
        return jsonify({
            "traces": traces,
            "total": get_trace_count(),
        })
    except Exception as e:
        logger.error("List traces error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/prompt-tree')
def get_prompt_tree():
    """GET /api/prompt-tree — Get the full prompt routing tree structure."""
    try:
        from prompt_tree import tree_structure, list_tree
        prefix = flask_request.args.get('prefix', '', type=str)
        return jsonify({
            "structure": tree_structure(),
            "nodes": list_tree(prefix),
        })
    except Exception as e:
        logger.error("Prompt tree error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Trajectory graph API ──────────────────────────────────────────────────

@app.route('/api/trajectory/summary')
@require_login
def get_trajectory_summary():
    """GET /api/trajectory/summary — Get trajectory detection summary for current user."""
    try:
        user_id = current_user_id()
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        from storage import load_trajectory_events, load_trajectories
        events = load_trajectory_events(user_id)
        trajs = load_trajectories(user_id)

        # Dimensional breakdown
        dim_counts = {}
        for e in events:
            dim = e.get("dimension", "unknown")
            dim_counts[dim] = dim_counts.get(dim, 0) + 1

        return jsonify({
            "total_events": len(events),
            "total_trajectories": len(trajs),
            "dimension_breakdown": dim_counts,
            "last_event": events[-1] if events else None,
        })
    except Exception as e:
        logger.error("Trajectory summary error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/trajectory/timeline')
@require_login
def get_trajectory_timeline():
    """GET /api/trajectory/timeline?dimension=sleep&limit=50 — Get event timeline."""
    try:
        user_id = current_user_id()
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        from storage import load_trajectory_events
        events = load_trajectory_events(user_id)

        dimension = flask_request.args.get("dimension")
        limit = flask_request.args.get("limit", 50, type=int)

        if dimension:
            events = [e for e in events if e.get("dimension") == dimension]

        events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return jsonify({"events": events[:limit]})
    except Exception as e:
        logger.error("Trajectory timeline error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/trajectory/patterns')
@require_login
def get_trajectory_patterns():
    """GET /api/trajectory/patterns?period_type=weekly&dimension=energy — Get trajectory patterns."""
    try:
        user_id = current_user_id()
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        from storage import load_trajectories
        trajs = load_trajectories(user_id)

        period_type = flask_request.args.get("period_type")
        dimension = flask_request.args.get("dimension")

        if period_type:
            trajs = [t for t in trajs if t.get("period_type") == period_type]
        if dimension:
            trajs = [t for t in trajs if t.get("dimension") == dimension]

        trajs.sort(key=lambda t: t.get("confidence", 0), reverse=True)
        return jsonify({"patterns": trajs})
    except Exception as e:
        logger.error("Trajectory patterns error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Psycho State API ──────────────────────────────────────────────────────

@app.route('/api/psycho/state')
@require_login
def get_psycho_state():
    """GET /api/psycho/state — Get current aggregated psycho state for the user."""
    try:
        user_id = current_user_id()
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        from storage import load_psycho_state
        state_dict = load_psycho_state(user_id)
        if state_dict is None:
            return jsonify({"state": None, "message": "No psycho data yet"})

        # Return summary
        intervenable = state_dict.get("intervenable_state", "normal")
        intervention = recommend_intervention(intervenable)
        return jsonify({
            "state": {
                "intervenable_state": state_dict.get("intervenable_state"),
                "n_events_aggregated": state_dict.get("n_events_aggregated", 0),
                "last_updated": state_dict.get("last_updated"),
                "ema": {k: v for k, v in state_dict.items()
                        if k.startswith("ema_") and v is not None},
                "trends": {
                    "willingness": state_dict.get("willingness_trend", "stable"),
                    "fatigue": state_dict.get("fatigue_trend", "stable"),
                    "avoidance": state_dict.get("avoidance_trend", "stable"),
                },
            },
            "intervention": intervention,
        })
    except Exception as e:
        logger.error("Psycho state error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/psycho/fingerprint')
@require_login
def get_psycho_fingerprint():
    """GET /api/psycho/fingerprint — Compute psycho fingerprint from accumulated events."""
    try:
        user_id = current_user_id()
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        from storage import load_psycho_events, load_psycho_fingerprint, save_psycho_fingerprint
        from psycho_engine import compute_full_fingerprint

        # Check cache first
        cached = load_psycho_fingerprint(user_id)
        if cached:
            return jsonify(cached)

        events = load_psycho_events(user_id)
        fp = compute_full_fingerprint(user_id, events)
        save_psycho_fingerprint(user_id, fp)
        return jsonify(fp)
    except Exception as e:
        logger.error("Psycho fingerprint error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── MRT (Micro-Randomized Trial) API ───────────────────────────────────────

@app.route('/api/mrt/randomize', methods=['POST'])
@require_login
def mrt_randomize():
    """POST /api/mrt/randomize — Randomize intervention at decision point.

    Body: {
        "current_state": "low_mood_available",
        "available": ["gentle_suggest", "lower_barrier"]  (optional)
    }
    Returns: {"chosen": "gentle_suggest", "context": {...}}
    """
    try:
        user_id = current_user_id()
        data = flask_request.json or {}
        from mrt_engine import MRTrialManager
        m = MRTrialManager.load(user_id)
        chosen = m.randomize(
            current_state=data.get("current_state", "normal"),
            available=data.get("available"),
        )
        m.save()
        return jsonify({
            "chosen": chosen,
            "context": {
                "state": data.get("current_state", "normal"),
                "n_decisions": len(m.decisions),
            }
        })
    except Exception as e:
        logger.error("MRT randomize error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/mrt/outcome', methods=['POST'])
@require_login
def mrt_record_outcome():
    """POST /api/mrt/outcome — Record proximal outcome for last intervention.

    Body: {
        "intervention": "gentle_suggest",
        "responded": true,
        "willingness_delta": 0.5,   (optional)
        "fatigue_delta": -0.3,       (optional)
        "avoidance_delta": -0.2      (optional)
    }
    """
    try:
        user_id = current_user_id()
        data = flask_request.json or {}
        from mrt_engine import MRTrialManager
        m = MRTrialManager.load(user_id)
        m.record_outcome(
            intervention_type=data["intervention"],
            responded=data.get("responded", False),
            willingness_delta=data.get("willingness_delta"),
            fatigue_delta=data.get("fatigue_delta"),
            avoidance_delta=data.get("avoidance_delta"),
        )
        m.save()
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error("MRT outcome error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/mrt/analysis')
@require_login
def mrt_analysis():
    """GET /api/mrt/analysis — N-of-1 analysis for current user.

    Returns rankings of interventions by estimated proximal effect.
    """
    try:
        user_id = current_user_id()
        from mrt_engine import MRTrialManager
        m = MRTrialManager.load(user_id)
        result = m.n_of_1_analysis()
        return jsonify(result)
    except Exception as e:
        logger.error("MRT analysis error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── MRT Effect Dashboard ──────────────────────────────────────────────────

@app.route('/api/mrt/dashboard')
@require_login
def mrt_dashboard():
    """GET /api/mrt/dashboard — Aggregate MRT effect dashboard data.

    Returns the full causal chain: psycho_state → intervention → outcome,
    with per-state and per-intervention breakdowns.
    """
    try:
        user_id = current_user_id()
        from mrt_engine import MRTrialManager, INTERVENTION_CATALOG
        from psycho_engine import get_psycho_state

        m = MRTrialManager.load(user_id)
        decisions = m.decisions

        # ── Overall stats ──
        total_decisions = len(decisions)
        with_outcome = sum(1 for d in decisions if d.proximal_outcome_ts is not None)
        response_rate = (
            sum(1 for d in decisions if d.proximal_responded) / with_outcome
            if with_outcome > 0 else None
        )

        # ── Per-intervention breakdown ──
        per_intervention = {}
        for iv_name in INTERVENTION_CATALOG:
            effect = m.estimate_effect(iv_name)
            iv_decisions = [d for d in decisions if d.chosen_intervention == iv_name]
            per_intervention[iv_name] = {
                "label": iv_name.replace("_", " ").title(),
                "target_dim": INTERVENTION_CATALOG[iv_name]["target_dim"],
                "hypothesis": INTERVENTION_CATALOG[iv_name]["hypothesis"],
                "expected_effect": INTERVENTION_CATALOG[iv_name]["expected_effect"],
                "n_decisions": len(iv_decisions),
                "n_with_outcome": effect["n_trials"],
                "mean_response_rate": effect.get("mean_response_rate"),
                "mean_willingness_delta": effect.get("mean_willingness_delta"),
                "mean_fatigue_delta": effect.get("mean_fatigue_delta"),
                "mean_avoidance_delta": effect.get("mean_avoidance_delta"),
            }

        # ── Per-state breakdown ──
        from collections import Counter
        state_counts = Counter(d.intervenable_state for d in decisions)
        per_state = {}
        for state_name, count in state_counts.most_common():
            state_decisions = [d for d in decisions if d.intervenable_state == state_name]
            state_outcomes = [d for d in state_decisions if d.proximal_outcome_ts is not None]
            state_ivs = Counter(d.chosen_intervention for d in state_decisions)
            per_state[state_name] = {
                "n_decisions": count,
                "n_with_outcome": len(state_outcomes),
                "intervention_mix": dict(state_ivs.most_common()),
                "response_rate": (
                    sum(1 for d in state_outcomes if d.proximal_responded) / len(state_outcomes)
                    if state_outcomes else None
                ),
            }

        # ── N-of-1 ranking ──
        n_of_1 = m.n_of_1_analysis()

        # ── Temporal trend (last 20 decisions) ──
        recent = decisions[-20:] if len(decisions) > 20 else decisions
        timeline = []
        for d in recent:
            timeline.append({
                "timestamp": d.timestamp,
                "state": d.intervenable_state,
                "intervention": d.chosen_intervention,
                "responded": d.proximal_responded,
                "willingness_delta": d.proximal_willingness_delta,
                "has_outcome": d.proximal_outcome_ts is not None,
            })

        sample_psycho = get_psycho_state(user_id) if total_decisions > 0 else {}

        return jsonify({
            "user_id": user_id,
            "total_decisions": total_decisions,
            "n_with_outcome": with_outcome,
            "overall_response_rate": round(response_rate, 4) if response_rate is not None else None,
            "current_psycho_state": sample_psycho,
            "intervention_catalog": {
                k: {"target_dim": v["target_dim"],
                    "hypothesis": v["hypothesis"],
                    "expected_effect": v["expected_effect"]}
                for k, v in INTERVENTION_CATALOG.items()
            },
            "per_intervention": per_intervention,
            "per_state": per_state,
            "n_of_1_ranking": n_of_1.get("rankings", []),
            "best_intervention": n_of_1.get("best_intervention"),
            "timeline": timeline,
        })
    except Exception as e:
        logger.error("MRT dashboard error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/mrt-dashboard')
@require_login
def mrt_dashboard_page():
    """GET /mrt-dashboard — HTML page for the MRT effect dashboard."""
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MRT 效果仪表盘 — Viora</title>
    <style>
      * { box-sizing: border-box; margin: 0; padding: 0; }
      body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
             background: #0f0f1a; color: #e0e0e0; padding: 20px; }
      h1 { font-size: 1.5rem; margin-bottom: 4px; color: #7c8aff; }
      .subtitle { color: #888; font-size: 0.85rem; margin-bottom: 20px; }
      .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
              gap: 16px; margin-bottom: 20px; }
      .card { background: #1a1a2e; border-radius: 10px; padding: 16px;
              border: 1px solid #2a2a4a; }
      .card h2 { font-size: 1rem; color: #b0b8ff; margin-bottom: 12px;
                 border-bottom: 1px solid #2a2a4a; padding-bottom: 8px; }
      .stat-row { display: flex; justify-content: space-between; padding: 4px 0;
                  font-size: 0.9rem; }
      .stat-row .label { color: #aaa; }
      .stat-row .value { color: #fff; font-weight: 600; }
      .value.good { color: #4ade80; }
      .value.warn { color: #facc15; }
      .value.bad { color: #f87171; }
      table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
      th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #2a2a4a; }
      th { color: #888; font-weight: 500; }
      .bar-cell { position: relative; }
      .bar-bg { position: absolute; top: 0; left: 0; height: 100%;
                opacity: 0.15; border-radius: 3px; }
      .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
               font-size: 0.75rem; }
      .badge.pos { background: #166534; color: #86efac; }
      .badge.neg { background: #7f1d1d; color: #fca5a5; }
      .chain-flow { display: flex; align-items: center; gap: 8px;
                    flex-wrap: wrap; justify-content: center;
                    background: #12122a; border-radius: 8px; padding: 12px 16px; }
      .chain-step { background: #1e1e3a; border: 1px solid #3a3a6a;
                    border-radius: 8px; padding: 8px 14px; font-size: 0.8rem;
                    text-align: center; min-width: 80px; }
      .chain-arrow { color: #555; font-size: 1.2rem; }
      .chain-step .step-label { color: #888; font-size: 0.65rem; }
      .chain-step .step-val { color: #fff; font-weight: 600; margin-top: 2px; }
      .empty-state { text-align: center; padding: 40px 20px; color: #666; }
      .empty-state h3 { font-size: 1.1rem; margin-bottom: 8px; color: #888; }
      .tag { display: inline-block; background: #2a2a4a; border-radius: 4px;
             padding: 1px 6px; font-size: 0.75rem; color: #aaa; margin: 1px; }
      .timeline-dot { display: inline-block; width: 10px; height: 10px;
                      border-radius: 50%; margin-right: 6px; }
      .dot-responded { background: #4ade80; }
      .dot-no-outcome { background: #555; }
      .dot-no-response { background: #f87171; }
    </style>
    </head>
    <body>

    <h1>MRT 效果仪表盘</h1>
    <p class="subtitle">心理状态 → 干预 → 行为结果 · 因果链可视化</p>

    <div id="app">
      <div v-if="loading" class="empty-state"><h3>加载中...</h3></div>

      <template v-if="!loading && data">
        <!-- 因果链概览 -->
        <div class="chain-flow">
          <div class="chain-step">
            <div class="step-label">心理状态</div>
            <div class="step-val">{{ psychoLabel(data.current_psycho_state) }}</div>
          </div>
          <div class="chain-arrow">→</div>
          <div class="chain-step">
            <div class="step-label">干预分配</div>
            <div class="step-val">{{ data.total_decisions }} 次决策</div>
          </div>
          <div class="chain-arrow">→</div>
          <div class="chain-step">
            <div class="step-label">有效结果</div>
            <div class="step-val">{{ data.n_with_outcome }} / {{ data.total_decisions }}</div>
          </div>
          <div class="chain-arrow">→</div>
          <div class="chain-step">
            <div class="step-label">响应率</div>
            <div :class="['step-val', rateClass(data.overall_response_rate)]">
              {{ fmtPct(data.overall_response_rate) }}
            </div>
          </div>
          <div class="chain-arrow">→</div>
          <div class="chain-step" style="border-color: #7c8aff;">
            <div class="step-label">最佳干预</div>
            <div class="step-val" style="color: #7c8aff;">
              {{ bestLabel(data.best_intervention) }}
            </div>
          </div>
        </div>

        <!-- 指标网格 -->
        <div class="grid">
          <div class="card">
            <h2>📊 总体统计</h2>
            <div class="stat-row"><span class="label">总决策次数</span><span class="value">{{ data.total_decisions }}</span></div>
            <div class="stat-row"><span class="label">有结果记录</span><span class="value">{{ data.n_with_outcome }}</span></div>
            <div class="stat-row"><span class="label">总体响应率</span>
              <span :class="['value', rateClass(data.overall_response_rate)]">{{ fmtPct(data.overall_response_rate) }}</span></div>
            <div class="stat-row"><span class="label">最佳干预</span>
              <span class="value good">{{ bestLabel(data.best_intervention) }}</span></div>
            <div class="stat-row"><span class="label">决策状态分布</span>
              <span class="value">{{ stateDistSummary(data.per_state) }}</span></div>
          </div>

          <div class="card">
            <h2>🧠 干预目录</h2>
            <div v-for="(iv, name) in data.intervention_catalog" :key="name"
                 style="padding: 4px 0; border-bottom: 1px solid #1a1a30;">
              <div style="display:flex; justify-content:space-between;">
                <span style="font-weight:500;">{{ fmtName(name) }}</span>
                <span :class="['badge', iv.expected_effect === 'positive' ? 'pos' : 'neg']">{{ iv.expected_effect }}</span>
              </div>
              <div style="font-size:0.8rem; color:#888;">目标: {{ iv.target_dim }} · {{ iv.hypothesis }}</div>
            </div>
          </div>
        </div>

        <!-- 干预效果表 -->
        <div class="card" style="margin-bottom:16px;">
          <h2>🎯 干预效果对比</h2>
          <table v-if="Object.keys(data.per_intervention).length > 0">
            <thead>
              <tr><th>干预</th><th>目标</th><th>试次</th><th>响应率</th><th>意愿Δ</th><th>疲劳Δ</th><th>回避Δ</th></tr>
            </thead>
            <tbody>
              <tr v-for="(iv, name) in data.per_intervention" :key="name">
                <td><b>{{ iv.label }}</b><br><span class="tag">{{ iv.target_dim }}</span></td>
                <td>{{ iv.hypothesis.slice(0, 20) }}...</td>
                <td>{{ iv.n_with_outcome }}/{{ iv.n_decisions }}</td>
                <td :class="rateClass(iv.mean_response_rate)">{{ fmtPct(iv.mean_response_rate) }}</td>
                <td :class="deltaClass(iv.mean_willingness_delta)">{{ fmtDelta(iv.mean_willingness_delta) }}</td>
                <td :class="deltaClass(iv.mean_fatigue_delta, true)">{{ fmtDelta(iv.mean_fatigue_delta) }}</td>
                <td :class="deltaClass(iv.mean_avoidance_delta, true)">{{ fmtDelta(iv.mean_avoidance_delta) }}</td>
              </tr>
            </tbody>
          </table>
          <div v-else class="empty-state" style="padding:20px;">
            <h3>暂无干预数据</h3>
            <p style="font-size:0.85rem;">开始使用 MRT 随机化后，效果数据将在此展示。</p>
          </div>
        </div>

        <!-- N-of-1 排名 -->
        <div class="card" style="margin-bottom:16px;">
          <h2>🏆 N-of-1 排名</h2>
          <table v-if="data.n_of_1_ranking.length > 0">
            <thead>
              <tr><th>#</th><th>干预</th><th>试次</th><th>响应率</th><th>意愿Δ</th><th>疲劳Δ</th><th>回避Δ</th></tr>
            </thead>
            <tbody>
              <tr v-for="(r, i) in data.n_of_1_ranking" :key="i">
                <td><b>{{ i + 1 }}</b></td>
                <td>{{ fmtName(r.intervention) }}</td>
                <td>{{ r.n_trials }}</td>
                <td :class="rateClass(r.mean_response_rate)">{{ fmtPct(r.mean_response_rate) }}</td>
                <td :class="deltaClass(r.mean_willingness_delta)">{{ fmtDelta(r.mean_willingness_delta) }}</td>
                <td :class="deltaClass(r.mean_fatigue_delta, true)">{{ fmtDelta(r.mean_fatigue_delta) }}</td>
                <td :class="deltaClass(r.mean_avoidance_delta, true)">{{ fmtDelta(r.mean_avoidance_delta) }}</td>
              </tr>
            </tbody>
          </table>
          <div v-else style="color:#666;font-size:0.85rem;padding:8px 0;">
            每个干预至少需要 3 次有结果的试次才能进入排名。
          </div>
        </div>

        <!-- 时间线 -->
        <div class="card">
          <h2>⏱ 最近决策时间线</h2>
          <div v-if="data.timeline.length > 0">
            <div v-for="d in [...data.timeline].reverse()" :key="d.timestamp"
                 style="display:flex; align-items:center; gap:8px; padding:6px 0; border-bottom:1px solid #1a1a30; font-size:0.85rem;">
              <span :class="['timeline-dot', d.has_outcome ? (d.responded ? 'dot-responded' : 'dot-no-response') : 'dot-no-outcome']"></span>
              <span style="color:#888;min-width:130px;">{{ fmtTime(d.timestamp) }}</span>
              <span class="tag">{{ d.state }}</span>
              <span style="font-weight:500;">{{ fmtName(d.intervention) }}</span>
              <span v-if="d.responded" style="color:#4ade80;">✓ 响应</span>
              <span v-else-if="d.has_outcome" style="color:#f87171;">✗ 未响应</span>
              <span v-else style="color:#666;">待结果</span>
              <span v-if="d.willingness_delta != null" style="color:#888;font-size:0.8rem;">
                意愿 {{ d.willingness_delta > 0 ? '+' : '' }}{{ d.willingness_delta }}
              </span>
            </div>
          </div>
          <div v-else style="color:#666;padding:12px 0;text-align:center;">
            暂无决策记录。
          </div>
        </div>
      </template>

      <div v-if="!loading && !data" class="empty-state">
        <h3>数据加载失败</h3>
        <p>请确认已登录并刷新页面。</p>
      </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/vue@2.7.16/dist/vue.min.js"></script>
    <script>
    new Vue({
      el: '#app',
      data: { loading: true, data: null },
      mounted() {
        fetch('/api/mrt/dashboard')
          .then(r => r.json())
          .then(d => { this.data = d; this.loading = false; })
          .catch(e => { console.error(e); this.loading = false; });
      },
      methods: {
        fmtPct(v) { return v != null ? (v * 100).toFixed(1) + '%' : '--'; },
        fmtDelta(v) { return v != null ? (v > 0 ? '+' : '') + v.toFixed(2) : '--'; },
        fmtName(s) { return s.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase()); },
        fmtTime(ts) {
          if (!ts) return '--';
          const d = new Date(ts);
          return d.toLocaleDateString('zh-CN', {month:'2-digit',day:'2-digit'}) + ' ' +
                 d.toLocaleTimeString('zh-CN', {hour:'2-digit',minute:'2-digit'});
        },
        rateClass(v) { return v == null ? '' : v >= 0.5 ? 'good' : v >= 0.25 ? 'warn' : 'bad'; },
        deltaClass(v, reverse) {
          if (v == null) return '';
          if (reverse) v = -v;
          return v > 0 ? 'good' : v < 0 ? 'bad' : '';
        },
        psychoLabel(s) {
          if (!s || !s.state) return '未知';
          return s.state.replace(/_/g, ' ');
        },
        bestLabel(iv) { return iv ? this.fmtName(iv) : '数据不足'; },
        stateDistSummary(states) {
          if (!states) return '无';
          return Object.keys(states).slice(0, 3).join(', ') +
                 (Object.keys(states).length > 3 ? '...' : '');
        },
      }
    });
    </script>
    </body>
    </html>
    ''')
