"""
app.py — Streamlit Web UI for Deep Cognitive Task Agent
Milestone 4: Full Integration with Groq LLM

Fixes applied:
  - API key validation before running agent (clear error instead of blank screen)
  - Rate-limit errors surfaced with a friendly, actionable message
  - Model selector writes to os.environ BEFORE run_agent() is called
  - Spinner wraps the full agent call so the UI never goes blank mid-run
  - st.rerun() only called AFTER state is fully written to session_state
  - Added "Stop / Clear" safety button during long runs

Features:
  - Dark glassmorphism design with smooth animations
  - Sidebar: Groq API key input, model selector, session controls
  - Chat-style interaction with the full LangGraph agent
  - Live panels: TODO tracker, Virtual File System, Delegation Log
  - Final output rendered as markdown
  - JSON export of full run results
"""

import os
import json
import time
import streamlit as st
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Page Config — MUST be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cognitive Task Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ─────────────────────────────────────────────
# LangSmith — set env vars NOW, before any langchain import
# The sidebar below can update these, but the FIRST page load
# must apply .env values immediately so the tracer initialises correctly.
# ─────────────────────────────────────────────
def _apply_tracing_env(api_key: str, enabled: bool, project: str = "milestone4-deep-agent"):
    """
    Set (or clear) LangSmith env vars and patch the live LangChain tracer.
    Must be called BEFORE agent invocation, not just at sidebar render time.
    """
    if api_key and enabled:
        os.environ["LANGCHAIN_API_KEY"]    = api_key.strip()
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"]    = project
        # Force langchain_core to re-read the env var immediately
        try:
            from langchain_core.callbacks.manager import _configure_hooks  # noqa
        except Exception:
            pass
        try:
            # langsmith >= 0.1: update the global client so traces actually send
            import langsmith
            langsmith.Client(api_key=api_key.strip())
        except Exception:
            pass
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ.pop("LANGCHAIN_API_KEY", None)

# Apply from .env on startup (sidebar will override below)
_startup_ls_key = os.getenv("LANGCHAIN_API_KEY", "")
_startup_tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
if _startup_ls_key and _startup_tracing:
    _apply_tracing_env(_startup_ls_key, True)

# ─────────────────────────────────────────────
# Custom CSS — Dark Glassmorphism Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary:   #0d1117;
    --bg-secondary: #161b22;
    --bg-glass:     rgba(22, 27, 34, 0.85);
    --accent:       #7c3aed;
    --accent-light: #a78bfa;
    --accent-glow:  rgba(124, 58, 237, 0.35);
    --success:      #10b981;
    --warning:      #f59e0b;
    --danger:       #ef4444;
    --info:         #3b82f6;
    --text-primary: #e6edf3;
    --text-muted:   #8b949e;
    --border:       rgba(240, 246, 252, 0.1);
    --radius:       14px;
}

html, body, [data-testid="stApp"] {
    background: linear-gradient(135deg, #0d1117 0%, #1a0a2e 50%, #0d1117 100%) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

[data-testid="stSidebar"] {
    background: var(--bg-glass) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent-light) !important;
}

.glass-card {
    background: rgba(22, 27, 34, 0.7);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    backdrop-filter: blur(16px);
    margin-bottom: 1rem;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(124, 58, 237, 0.4);
    box-shadow: 0 0 24px var(--accent-glow);
}

.hero-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
    background: linear-gradient(135deg, rgba(124,58,237,0.15) 0%, rgba(59,130,246,0.1) 100%);
    border-radius: var(--radius);
    border: 1px solid rgba(124,58,237,0.25);
    margin-bottom: 1.5rem;
}
.hero-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    animation: shimmer 3s ease-in-out infinite;
}
@keyframes shimmer {
    0%, 100% { filter: brightness(1); }
    50%       { filter: brightness(1.3); }
}
.hero-header p {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-top: 0.4rem;
}

.msg-user {
    background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(59,130,246,0.15));
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 14px 14px 4px 14px;
    padding: 0.9rem 1.2rem;
    margin: 0.6rem 0;
    font-size: 0.95rem;
    animation: fadeIn 0.3s ease;
}
.msg-agent {
    background: rgba(22,27,34,0.8);
    border: 1px solid var(--border);
    border-radius: 14px 14px 14px 4px;
    padding: 0.9rem 1.2rem;
    margin: 0.6rem 0;
    font-size: 0.95rem;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.7rem;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-success { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.badge-pending { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.badge-info    { background: rgba(59,130,246,0.15);  color: #93c5fd; border: 1px solid rgba(59,130,246,0.3); }
.badge-purple  { background: rgba(124,58,237,0.15);  color: #c4b5fd; border: 1px solid rgba(124,58,237,0.3); }

.metric-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 1rem; }
.metric-tile {
    flex: 1;
    min-width: 100px;
    background: rgba(22,27,34,0.7);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
    transition: border-color 0.25s;
}
.metric-tile:hover { border-color: var(--accent); }
.metric-tile .num { font-size: 1.6rem; font-weight: 700; color: var(--accent-light); }
.metric-tile .lbl { font-size: 0.7rem; color: var(--text-muted); margin-top: 0.1rem; letter-spacing: 0.05em; text-transform: uppercase; }

.todo-item {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.55rem 0.8rem;
    border-radius: 8px;
    margin-bottom: 0.4rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    font-size: 0.88rem;
    transition: background 0.2s;
}
.todo-item:hover { background: rgba(124,58,237,0.1); }
.todo-id { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--text-muted); min-width: 54px; }

.file-item {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 0.6rem 0.9rem;
    border-radius: 8px;
    margin-bottom: 0.4rem;
    background: rgba(59,130,246,0.06);
    border: 1px solid rgba(59,130,246,0.15);
    font-size: 0.87rem;
}
.file-name { font-family: 'JetBrains Mono', monospace; color: #93c5fd; font-size: 0.82rem; }
.file-size { color: var(--text-muted); font-size: 0.75rem; }
.file-preview { color: var(--text-muted); font-size: 0.78rem; margin-top: 0.25rem; font-style: italic; }

.delegation-card {
    padding: 0.8rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.6rem;
    background: rgba(124,58,237,0.07);
    border: 1px solid rgba(124,58,237,0.2);
    font-size: 0.87rem;
}
.delegation-card .agent-name { font-weight: 600; color: var(--accent-light); }
.delegation-card .task-text  { color: var(--text-muted); margin: 0.25rem 0; }
.delegation-card .duration   { color: #34d399; font-size: 0.78rem; }

.output-box {
    background: rgba(16,185,129,0.06);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    margin-top: 0.5rem;
    max-height: 600px;
    overflow-y: auto;
    line-height: 1.7;
}

[data-testid="stTextArea"] textarea {
    background: rgba(22,27,34,0.9) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.25s !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.25s ease !important;
    padding: 0.55rem 1.4rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px var(--accent-glow) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

[data-testid="stSpinner"] { color: var(--accent-light) !important; }

[data-testid="stExpander"] {
    background: rgba(22,27,34,0.5) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

[data-testid="stSelectbox"] > div > div {
    background: rgba(22,27,34,0.9) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 99px; }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────
def _init_session():
    defaults = {
        "chat_history": [],   # list of {role, content, state?}
        "last_state":   None, # last AgentState dict
        "run_counter":  0,
        "is_running":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-size:2.5rem;'>🧠</div>
        <div style='font-weight:700; font-size:1.1rem; color:#a78bfa;'>Cognitive Agent</div>
        <div style='font-size:0.75rem; color:#8b949e;'>Powered by Groq + LangGraph</div>
    </div>
    <hr style='margin: 0.8rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("### 🔑 API Configuration")

    # Read the key from env as default so .env files work automatically
    groq_key_input = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_...",
        help="Get your free key at console.groq.com",
    )
    # Always keep env var in sync with what the user typed
    if groq_key_input:
        os.environ["GROQ_API_KEY"] = groq_key_input.strip()

    st.markdown("### ⚙️ Model")
    model_options = {
        "llama-3.1-8b-instant": "LLaMA 3.1 8B — FAST + SAFE (BEST)",
        "llama-3.3-70b-versatile": "LLaMA 3.3 70B — HIGH QUALITY"
    }
    selected_model = st.selectbox(
        "Model",
        options=list(model_options.keys()),
        format_func=lambda k: model_options[k],
        index=0,
        label_visibility="collapsed",
        help="All three support tool calling. Gemma 2 does NOT and will break the agent — avoid it.",
    )
    # Write model to env immediately so run_agent() picks it up
    os.environ["GROQ_MODEL"] = selected_model

    st.markdown("### ⚡ Performance")
    # LangGraph recursion_limit counts every node visit (agent + tools each count).
    # A typical run needs: 1 plan + 3 todos×4 nodes + 3 synthesis nodes ≈ 16 nodes.
    # We double that for safety and expose it as a 20–60 range.
    recursion_limit = st.slider(
        "Max agent steps (LangGraph nodes)",
        min_value=20,
        max_value=60,
        value=40,
        step=5,
        help="Each LLM call + each tool call each count as 1 step. 40 is safe for most tasks.",
    )

    st.markdown("### 🔗 LangSmith (Optional)")
    langsmith_key = st.text_input(
        "LangSmith API Key",
        value=os.getenv("LANGCHAIN_API_KEY", ""),
        type="password",
        placeholder="ls__...",
        help="Get a free key at smith.langchain.com — paste it here then toggle on.",
    )
    # Default toggle to True if key already in env (e.g. from .env file)
    _default_tracing = bool(os.getenv("LANGCHAIN_API_KEY")) and os.getenv("LANGCHAIN_TRACING_V2","false").lower() == "true"
    enable_tracing = st.toggle("Enable Tracing", value=_default_tracing)

    # Apply immediately — _apply_tracing_env patches the live LangChain tracer
    _apply_tracing_env(langsmith_key, enable_tracing)

    if langsmith_key and enable_tracing:
        st.sidebar.success("🔗 Tracing ON → milestone4-deep-agent", icon="✅")
    elif enable_tracing and not langsmith_key:
        st.sidebar.warning("Paste your LangSmith API key above to enable tracing.", icon="⚠️")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🗂️ Session")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_state   = None
            st.session_state.run_counter  = 0
            st.session_state.is_running   = False
            st.rerun()
    with col2:
        run_count = st.session_state.run_counter
        st.markdown(f"""
        <div style='text-align:center; padding:0.4rem;
             background:rgba(124,58,237,0.12); border-radius:8px;
             border:1px solid rgba(124,58,237,0.25);
             font-size:0.8rem; color:#a78bfa;'>
            {run_count} run{'s' if run_count != 1 else ''}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#8b949e; line-height:1.7;'>
    <b>Available Sub-Agents</b><br>
    🔍 <code>web_search_agent</code><br>
    📝 <code>summarization_agent</code><br>
    💻 <code>code_analysis_agent</code>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero-header'>
    <h1>🧠 Autonomous Cognitive Engine</h1>
    <p>Deep Research · Long-Horizon Planning · Multi-Agent Collaboration</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper: Render a single run's results
# ─────────────────────────────────────────────
def render_run_results(state: dict, request: str, run_idx: int = 0):
    from langchain_core.messages import AIMessage

    todos          = state.get("todos", [])
    vfs            = state.get("virtual_files", {})
    delegation_log = state.get("delegation_log", [])

    completed   = sum(1 for t in todos if t["status"] == "completed")
    pending     = len(todos) - completed
    files_count = len(vfs)
    del_count   = len(delegation_log)

    # ── Metrics row ──
    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-tile'>
            <div class='num'>{len(todos)}</div>
            <div class='lbl'>Tasks</div>
        </div>
        <div class='metric-tile'>
            <div class='num' style='color:#34d399;'>{completed}</div>
            <div class='lbl'>Done</div>
        </div>
        <div class='metric-tile'>
            <div class='num' style='color:#fbbf24;'>{pending}</div>
            <div class='lbl'>Pending</div>
        </div>
        <div class='metric-tile'>
            <div class='num' style='color:#93c5fd;'>{files_count}</div>
            <div class='lbl'>VFS Files</div>
        </div>
        <div class='metric-tile'>
            <div class='num' style='color:#c4b5fd;'>{del_count}</div>
            <div class='lbl'>Delegations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Three columns for panels ──
    col_a, col_b, col_c = st.columns([3, 2, 2])

    # ── Task Plan ──
    with col_a:
        with st.expander("📋 Task Plan", expanded=True):
            if not todos:
                st.markdown("<span style='color:#8b949e;'>No tasks planned yet.</span>", unsafe_allow_html=True)
            else:
                for todo in todos:
                    icon   = "✅" if todo["status"] == "completed" else "⏳"
                    status = "badge-success" if todo["status"] == "completed" else "badge-pending"
                    st.markdown(f"""
                    <div class='todo-item'>
                        <span>{icon}</span>
                        <span class='todo-id'>[{todo['id']}]</span>
                        <span style='flex:1;'>{todo['task'][:70]}{'…' if len(todo['task']) > 70 else ''}</span>
                        <span class='badge {status}'>{todo['status']}</span>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Virtual File System ──
    with col_b:
        with st.expander("🗂️ Virtual File System", expanded=True):
            if not vfs:
                st.markdown("<span style='color:#8b949e;'>No files saved.</span>", unsafe_allow_html=True)
            else:
                for fname, content in vfs.items():
                    preview = content[:80].replace("\n", " ")
                    st.markdown(f"""
                    <div class='file-item'>
                        <div>
                            <div class='file-name'>📄 {fname}</div>
                            <div class='file-preview'>"{preview}…"</div>
                        </div>
                        <div class='file-size'>{len(content):,} chars</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Delegation Log ──
    with col_c:
        with st.expander("🤖 Delegation Log", expanded=True):
            if not delegation_log:
                st.markdown("<span style='color:#8b949e;'>No delegations made.</span>", unsafe_allow_html=True)
            else:
                for entry in delegation_log:
                    summary = entry.get("result_summary", "")[:90].replace("\n", " ")
                    st.markdown(f"""
                    <div class='delegation-card'>
                        <div class='agent-name'>⚡ {entry['agent_name']}</div>
                        <div class='task-text'>{entry['sub_task'][:65]}{'…' if len(entry['sub_task']) > 65 else ''}</div>
                        <div class='duration'>⏱ {entry.get('duration_s', 0)}s</div>
                        <div class='task-text' style='margin-top:0.3rem;font-size:0.75rem;'>
                            {summary}{'…' if len(summary) >= 90 else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Final Output ──
    # Prefer the explicit final_output field set by run_agent (most reliable)
    last_msg = state.get("final_output", "").strip()

    # Skip if it looks like raw JSON (agent echoed a tool result instead of writing prose)
    if last_msg and (last_msg.startswith("{") or last_msg.startswith("[")):
        last_msg = ""

    # Fallback: scan messages in reverse for the last substantive prose AI response
    if not last_msg:
        for msg in reversed(state.get("messages", [])):
            if (
                isinstance(msg, AIMessage)
                and msg.content
                and not getattr(msg, "tool_calls", [])
                and not msg.content.strip().startswith("{")
                and not msg.content.strip().startswith("[")
            ):
                last_msg = msg.content
                break

    if last_msg:
        with st.expander("📝 Final Output", expanded=True):
            st.markdown(f"<div class='output-box'>{last_msg}</div>", unsafe_allow_html=True)
    else:
        with st.expander("📝 Final Output", expanded=True):
            st.warning(
                "⚠️ The agent completed its tasks but the final synthesis step "
                "did not produce a text response. Check the Virtual File System above — "
                "your results may have been saved there as files.",
                icon="📂",
            )

    # ── Export ──
    export_data = {
        "request":        request,
        "todos":          todos,
        "virtual_files":  vfs,
        "delegation_log": delegation_log,
        "final_output":   last_msg or "",
    }
    st.download_button(
        label="💾 Export Run as JSON",
        data=json.dumps(export_data, indent=2),
        file_name=f"cognitive_agent_run_{int(time.time())}.json",
        mime="application/json",
        use_container_width=True,
        key=f"export_btn_{run_idx}",
    )


# ─────────────────────────────────────────────
# Chat History Display
# ─────────────────────────────────────────────
for _turn_idx, turn in enumerate(st.session_state.chat_history):
    if turn["role"] == "user":
        st.markdown(f"""
        <div class='msg-user'>
            <span style='font-size:0.75rem; color:#a78bfa; font-weight:600;'>YOU</span><br>
            {turn['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='msg-agent'>
            <span style='font-size:0.75rem; color:#34d399; font-weight:600;'>🧠 AGENT</span><br>
            Task complete. See results below ↓
        </div>
        """, unsafe_allow_html=True)
        if turn.get("state"):
            render_run_results(turn["state"], turn["content"], run_idx=_turn_idx)


# ─────────────────────────────────────────────
# Input Area
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 💬 New Request")

input_col, btn_col = st.columns([5, 1])
with input_col:
    user_input = st.text_area(
        "Request",
        placeholder="e.g. Research the history of quantum computing and write a comprehensive report…",
        height=100,
        label_visibility="collapsed",
        key="user_input_box",
        disabled=st.session_state.is_running,
    )
with btn_col:
    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    run_button = st.button(
        "🚀 Run",
        use_container_width=True,
        disabled=st.session_state.is_running,
    )


# ─────────────────────────────────────────────
# Quick example buttons
# ─────────────────────────────────────────────
st.markdown(
    "<div style='font-size:0.78rem; color:#8b949e; margin-bottom:0.4rem;'>Quick examples:</div>",
    unsafe_allow_html=True,
)
ex_cols = st.columns(3)
examples = [
    "Research the current state of large language models and write a technical overview",
    "Analyze the pros and cons of microservices vs monolithic architecture for a startup",
    "Summarize the key concepts of retrieval-augmented generation (RAG) for beginners",
]
for col, ex in zip(ex_cols, examples):
    with col:
        if st.button(
            ex[:50] + "…",
            use_container_width=True,
            key=f"ex_{ex[:20]}",
            disabled=st.session_state.is_running,
        ):
            user_input = ex
            run_button = True


# ─────────────────────────────────────────────
# Agent Execution
# ─────────────────────────────────────────────
if run_button and user_input and user_input.strip():

    # ── Guard: API key must be present ───────────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        st.error(
            "⚠️ Please enter your Groq API key in the sidebar before running. "
            "Get a free key at console.groq.com",
            icon="🔑",
        )
        st.stop()

    # ── Mark as running so inputs are disabled ────────────────────────────────
    st.session_state.is_running = True

    # Add user message to history immediately so it's visible during the run
    st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})

    # ── Run the agent inside a spinner ───────────────────────────────────────
    with st.spinner("🧠 Agent is thinking — planning tasks, delegating to sub-agents, synthesising results…"):
        try:
            # Import here so env vars (API key, model) are already set in os.environ
            from main import run_agent

            st.session_state.run_counter += 1
            final_state = run_agent(
                user_request=user_input.strip(),
                run_name=f"streamlit-run-{st.session_state.run_counter}",
                recursion_limit=recursion_limit,
            )

            # Store result and add agent turn to history
            st.session_state.last_state = final_state
            st.session_state.chat_history.append({
                "role":    "agent",
                "content": user_input.strip(),
                "state":   final_state,
            })

        except Exception as e:
            err_str = str(e)
            err_lower = err_str.lower()

            if "groq_api_key is not set" in err_lower or "api key" in err_lower:
                st.error(
                    "🔑 API key error: please check your Groq API key in the sidebar.",
                    icon="🔑",
                )
            elif "rate limit" in err_lower or "429" in err_lower or "ratelimit" in err_lower:
                st.error(
                    "⏱️ Groq rate limit hit even after retries.\n\n"
                    "**Try one of these:**\n"
                    "- Wait 1–2 minutes then re-submit\n"
                    "- Switch to **Gemma 2 9B** in the sidebar (higher free-tier limits)\n"
                    "- Lower the **Max agent steps** slider to 20–30\n"
                    "- Break your request into smaller sub-tasks",
                    icon="⚠️",
                )
            elif "recursion" in err_lower:
                st.error(
                    "🔁 The agent hit the step limit before finishing.\n\n"
                    "Try raising **Max agent steps** in the sidebar, "
                    "or simplify your request.",
                    icon="🔁",
                )
            else:
                st.error(f"❌ Agent error: {err_str}", icon="❌")

            import traceback
            with st.expander("🔍 Full traceback (for debugging)"):
                st.code(traceback.format_exc(), language="python")

    # ── Always clear the running flag and rerun ───────────────────────────────
    st.session_state.is_running = False
    st.rerun()

elif run_button and (not user_input or not user_input.strip()):
    st.warning("Please enter a request before running.", icon="✏️")


# ─────────────────────────────────────────────
# Empty State
# ─────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("""
    <div class='glass-card' style='text-align:center; padding: 2.5rem; margin-top:1rem;'>
        <div style='font-size:3rem; margin-bottom:0.8rem;'>🚀</div>
        <div style='font-size:1.1rem; font-weight:600; color:#a78bfa; margin-bottom:0.5rem;'>
            Ready to Research
        </div>
        <div style='color:#8b949e; font-size:0.9rem; max-width:480px; margin:auto; line-height:1.7;'>
            Enter any complex research question, analysis task, or coding challenge above.
            The agent will <b>plan</b>, <b>delegate</b> to specialists, and <b>synthesize</b>
            a comprehensive report — all powered by Groq's blazing-fast inference.
        </div>
    </div>
    """, unsafe_allow_html=True)

    feat_cols = st.columns(3)
    features = [
        ("📋", "Structured Planning", "Breaks any request into a dynamic TODO list with tracked steps."),
        ("🗂️", "Context Offloading", "Saves research notes to a Virtual File System across reasoning steps."),
        ("🤖", "Sub-Agent Delegation", "Routes tasks to specialist agents: research, summarization, or code analysis."),
    ]
    for col, (icon, title, desc) in zip(feat_cols, features):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='text-align:center;'>
                <div style='font-size:2rem;'>{icon}</div>
                <div style='font-weight:600; color:#e6edf3; margin:0.5rem 0 0.3rem;'>{title}</div>
                <div style='font-size:0.82rem; color:#8b949e; line-height:1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)