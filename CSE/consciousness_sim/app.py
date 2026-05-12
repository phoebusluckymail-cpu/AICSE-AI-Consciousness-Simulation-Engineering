"""
意识工程化构造实验 v5.0 - Streamlit Web UI
认知张力驱动 · 多Agent思维系统模拟器
"""
import streamlit as st
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consciousness_sim.engine import ConsciousnessEngine
from consciousness_sim.session_manager import SessionManager

st.set_page_config(
    page_title="意识工程 v5.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "意识工程化构造实验 v5.0 · 认知张力驱动"}
)

# ── CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    h1 { font-size: 1.4rem !important; }
    h3 { font-size: 1.0rem !important; margin: 0.3rem 0 !important; }
    .expression-box {
        padding: 0.8rem 1rem;
        border-left: 3px solid #4a90d9;
        border-radius: 4px;
        margin: 0.5rem 0;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)

SESSION_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "conversations")


# ── Session State ────────────────────────────────────
def init_session():
    if "engine" not in st.session_state:
        st.session_state.engine = ConsciousnessEngine()
        st.session_state.history = []
        st.session_state.agent_contexts = []
        st.session_state.session_dir = ""
        st.session_state.auto_continue = False
        st.session_state.allow_idle = True
        st.session_state.pending_seed = None
        st.session_state.pending_continue = False

init_session()
engine = st.session_state.engine


# ── Helpers ───────────────────────────────────────────
def status_bar():
    s = engine.get_status()
    stream = engine.stream
    active_c = len([c for c in stream.unresolved_conflicts if c.status == "active" and not c.resolved])
    deferred_n = len([d for d in stream.deferred_attention if not d.obsolete])
    color_map = {"紧张": "#e74c3c", "不安": "#e67e22", "好奇": "#3498db",
                 "平静": "#2ecc71", "审慎乐观": "#27ae60", "防御": "#95a5a6",
                 "疲惫": "#bdc3c7", "开放": "#2980b9", "困惑": "#f39c12",
                 "无": "#bdc3c7", "接纳": "#1abc9c", "兴奋": "#e74c3c",
                 "抵触": "#c0392b", "沉重的谨慎": "#7f8c8d"}
    color = color_map.get(s["emotional_color"], "#95a5a6")

    cols = st.columns(6)
    cols[0].markdown(f"**轮次** {s['round']}")
    cols[1].markdown(f"**情感** <span style='color:{color}'>●</span> {s['emotional_color']}", unsafe_allow_html=True)
    cols[2].markdown(f"**冲突** {active_c}")
    cols[3].markdown(f"**延后** {deferred_n}")
    cols[4].markdown(f"**收敛** {stream.convergence_rounds}轮")
    paused = "⏸️ 暂停" if s["paused"] else ("💭 空转" if s["idle"] else "▶️ 思考中")
    cols[5].markdown(f"**状态** {paused}")
    # API stats
    if s["api_calls"] > 0:
        st.caption(f"API: {s['api_calls']}次 · {s['api_tokens']} tokens")


def parse_output(output):
    """解析输出为结构化的轮次消息"""
    lines = output.split("\n")
    fragments, synthesis, writer_output, expression = {}, "", "", ""
    current_section, current_module = None, None

    for line in lines:
        if "发散碎片" in line:
            current_section = "fragments"
        elif "合成器输出" in line:
            current_section = "synthesis"
        elif "意识流写入器" in line:
            current_section = "writer"
        elif "语言表达" in line.strip() or "**语言表达" in line:
            current_section = "expression"
        elif current_section == "fragments":
            for mn in ["逻辑推理者", "批判者", "联想与创意联结者", "情感评估者", "动机检测者", "元认知监控者"]:
                if f"**{mn}**" in line:
                    current_module = mn
                    fragments.setdefault(mn, "")
                    break
            if current_module and line.strip() and "**" not in line:
                fragments[current_module] = fragments.get(current_module, "") + line + "\n"
        elif current_section == "synthesis" and line.strip() and not line.startswith("*"):
            synthesis += line + "\n"
        elif current_section == "writer" and line.strip() and not line.startswith("*") and "语言表达" not in line:
            writer_output += line + "\n"
        elif current_section == "expression" and line.strip() and not line.startswith("*"):
            expression += line + "\n"

    return {
        "round": engine.stream.round,
        "fragments": fragments,
        "synthesis": synthesis.strip(),
        "writer_output": writer_output.strip(),

        "expression": expression.strip(),
        "status": engine.get_status(),
    }


def render_message(msg):
    """渲染单轮消息"""
    if msg.get("role") == "user":
        st.markdown(f"**🧑 你：** {msg['content']}")
        return

    content = msg["content"]
    if not isinstance(content, dict):
        st.markdown(content)
        return

    s = content.get("status", {})
    # 情感色
    emo = s.get("emotional_color", "无")
    st.caption(f"第{content['round']}轮 · {emo}")

    # 语言表达（主体）
    expr = content.get("expression", "")
    if expr:
        st.markdown(f'<div class="expression-box">{expr}</div>', unsafe_allow_html=True)

    # 内部过程（折叠）
    with st.expander("🔍 内部过程", expanded=False):
        frags = content.get("fragments", {})
        synth = content.get("synthesis", "")
        writer = content.get("writer_output", "")

        tab_labels = ["逻辑推理", "批判", "联想·创意", "情感评估", "动机检测", "元认知"]
        frag_keys = ["逻辑推理者", "批判者", "联想与创意联结者", "情感评估者", "动机检测者", "元认知监控者"]
        if synth:
            tab_labels.append("选择性合成器")
            frag_keys.append("__synthesis__")
        if writer:
            tab_labels.append("意识流写入器")
            frag_keys.append("__writer__")

        tabs = st.tabs(tab_labels)
        for tab, key in zip(tabs, frag_keys):
            with tab:
                if key == "__synthesis__":
                    st.markdown(synth)
                elif key == "__writer__":
                    st.markdown(writer)
                else:
                    st.markdown(frags.get(key, "无输出"))


# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.title("🧠 v5.0")
    st.caption("认知张力驱动 · 多Agent思维模拟")

    st.divider()

    # 种子
    if engine.stream.goal:
        st.markdown(f"**种子：** {engine.stream.goal[:60]}")

    st.divider()

    # 会话管理
    with st.expander("💾 会话", expanded=not st.session_state.session_dir):
        if st.button("➕ 新建会话", use_container_width=True):
            st.session_state.engine = ConsciousnessEngine()
            st.session_state.history = []
            st.session_state.agent_contexts = []
            st.session_state.session_dir = ""
            st.rerun()

        sessions = SessionManager.list_sessions(SESSION_BASE_DIR)
        if sessions:
            opts = {f"{s['name'][:30]} ({s['rounds']}轮)": s["path"] for s in sessions}
            sel = st.selectbox("历史会话", ["(选择)"] + list(opts.keys()), label_visibility="collapsed")
            if sel != "(选择)" and st.button("📂 加载", use_container_width=True):
                path = opts[sel]
                eng = ConsciousnessEngine.load_from_session(path)
                sm = SessionManager(path)
                st.session_state.engine = eng
                st.session_state.session_dir = path
                st.session_state.agent_contexts = sm.load_all_contexts()
                st.session_state.history = sm.load_chat_history()
                st.rerun()

        if st.session_state.session_dir:
            name = os.path.basename(st.session_state.session_dir)
            st.caption(f"📁 {name}")
        else:
            st.caption("临时会话（未保存）")

    st.divider()
    st.session_state.auto_continue = st.toggle("🔄 自动继续", value=st.session_state.auto_continue,
                                                help="每轮结束后自动推进下一轮")
    st.session_state.allow_idle = st.toggle("💭 允许空转", value=st.session_state.allow_idle,
                                             help="认知收敛后自动进入空转反思")
    st.caption("种子 → 启动思考 · 「继续」→ 推进")


tab1, tab2, tab3 = st.tabs(["💬 意识流对话", "🧠 完整意识流状态", "🔍 Agent记忆"])

# ── Tab1: 对话 ───────────────────────────────────────
with tab1:
    status_bar()
    for msg in st.session_state.history:
        render_message(msg)

# 同步开关到引擎
engine.stream.allow_idle = st.session_state.allow_idle

# 输入（顶层）
user_input = st.chat_input("输入种子问题，或「继续」推进思考...")

if user_input:
    if user_input.strip() in ["继续", "continue"]:
        if not engine.stream.goal:
            st.warning("请先输入种子问题")
        else:
            st.session_state.history.append({"role": "user", "content": "继续"})
            engine.stream.add_chat_message("user", "继续")
            st.session_state.pending_continue = True
            st.rerun()
    else:
        st.session_state.history.append({"role": "user", "content": user_input})
        engine.stream.add_chat_message("user", user_input)
        st.session_state.pending_seed = user_input
        st.rerun()

# 处理待处理的操作（在渲染完用户消息后的下一轮执行）
if st.session_state.get("pending_continue"):
    st.session_state.pending_continue = False
    with st.spinner(f"第{engine.stream.round + 1}轮思考中..."):
        output = engine.continue_round()
    if "发散碎片" in output:
        msg = parse_output(output)
        st.session_state.history.append({"role": "assistant", "content": msg})
        st.session_state.agent_contexts.append({
            "round": engine.stream.round,
            "contexts": dict(engine.executor.last_contexts),
        })
    else:
        st.session_state.history.append({"role": "assistant", "content": output})
    if st.session_state.session_dir:
        try:
            SessionManager(st.session_state.session_dir).save_chat_history(st.session_state.history)
        except Exception:
            pass
    st.rerun()

if st.session_state.get("pending_seed"):
    seed = st.session_state.pending_seed
    st.session_state.pending_seed = None
    with st.spinner("注入种子，启动思考..."):
        output = engine.handle_new_goal(seed)
    if not st.session_state.session_dir:
        try:
            engine.create_session(SESSION_BASE_DIR, goal=seed)
            st.session_state.session_dir = str(engine.session_manager.session_dir)
        except Exception:
            pass
    if "发散碎片" in output:
        msg = parse_output(output)
        st.session_state.history.append({"role": "assistant", "content": msg})
        st.session_state.agent_contexts.append({
            "round": engine.stream.round,
            "contexts": dict(engine.executor.last_contexts),
        })
    else:
        st.session_state.history.append({"role": "assistant", "content": output})
    if st.session_state.session_dir:
        try:
            SessionManager(st.session_state.session_dir).save_chat_history(st.session_state.history)
        except Exception:
            pass
    st.rerun()

# ── Tab2: 完整意识流状态 ─────────────────────────────
with tab2:
    stream = engine.stream
    st.markdown("### 当前意识流")
    cols = st.columns(3)
    cols[0].metric("轮次", stream.round)
    cols[1].metric("情感底色", stream.emotional_color or "无")
    cols[2].metric("收敛", f"{stream.convergence_rounds}轮")
    if stream.goal:
        st.caption(f"种子：{stream.goal}")

    # 冲突
    active = [c for c in stream.unresolved_conflicts if c.status == "active" and not c.resolved]
    with st.expander(f"⚠️ 未解决冲突（{len(active)}）", expanded=bool(active)):
        for c in active:
            sev = {"致命": "🔴", "重要": "🟡", "次要": "⚪"}.get(c.severity.value, "⚪")
            st.markdown(f"{sev} **[{c.severity.value}]** {c.description}")
            st.caption(f"悬置于第{c.round_created}轮 · 活跃度{c.active_value:.2f} · {c.status}")
    if not active:
        st.caption("✅ 无未解决冲突")

    # 延后关注
    deferred = [d for d in stream.deferred_attention if not d.obsolete]
    with st.expander(f"📌 延后关注（{len(deferred)}）", expanded=bool(deferred)):
        for d in deferred:
            st.markdown(f"- {d.description}（活跃度{d.active_value:.2f}）")


    # 综合意识流


    # 历史记录
    if stream.round_records:
        with st.expander(f"📜 思考历史（{len(stream.round_records)}条）"):
            for rec in stream.round_records:
                st.markdown(f"> {rec[:300]}")

    # 冻结冲突
    if stream.frozen_conflicts:
        with st.expander(f"🧊 冻结冲突（{len(stream.frozen_conflicts)}）"):
            for c in stream.frozen_conflicts:
                st.markdown(f"- [{c.severity.value}] {c.description}")

# ── Tab3: Agent记忆 ──────────────────────────────────
with tab3:
    st.markdown("### Agent记忆")
    st.caption("每个子Agent每轮接收到的完整上下文")
    if not st.session_state.agent_contexts:
        st.info("暂无数据。启动思考后显示。")
    else:
        rounds = [f"第{c['round']}轮" for c in st.session_state.agent_contexts]
        ri = st.selectbox("轮次", range(len(rounds)), format_func=lambda i: rounds[i], key="mem_round")
        ctx = st.session_state.agent_contexts[ri]
        agents = list(ctx["contexts"].keys())
        if agents:
            labels = {
                "逻辑推理者": "发散·逻辑推理者", "批判者": "发散·批判者",
                "联想与创意联结者": "发散·联想与创意联结者", "情感评估者": "发散·情感评估者",
                "动机检测者": "发散·动机检测者", "元认知监控者": "发散·元认知监控者",
                "选择性合成器": "收敛·选择性合成器", "意识流写入器": "收敛·意识流写入器",
                "语言表达器": "收敛·语言表达器",
            }
            ai = st.selectbox("Agent", agents, format_func=lambda x: labels.get(x, x), key="mem_agent")
            if ai and ai in ctx["contexts"]:
                text = ctx["contexts"][ai]
                st.caption(f"{len(text)}字符 · {len(text.split(chr(10)))}行")
                st.code(text, language="text")

# ── 自动继续 ─────────────────
if (st.session_state.auto_continue and not st.session_state.get("pending_seed")
        and not st.session_state.get("pending_continue")):
    eng = st.session_state.engine
    if eng.stream.goal and not eng.stream.paused and not eng.stream.waiting_for_user and eng.stream.round > 0:
        with st.spinner(f"自动继续 第{eng.stream.round + 1}轮..."):
            output = eng.continue_round()
        if "发散碎片" in output:
            msg = parse_output(output)
            st.session_state.history.append({"role": "assistant", "content": msg})
            st.session_state.agent_contexts.append({
                "round": eng.stream.round,
                "contexts": dict(eng.executor.last_contexts),
            })
        else:
            st.session_state.history.append({"role": "assistant", "content": output})
        if st.session_state.session_dir:
            try:
                SessionManager(st.session_state.session_dir).save_chat_history(st.session_state.history)
            except Exception:
                pass
        st.rerun()
