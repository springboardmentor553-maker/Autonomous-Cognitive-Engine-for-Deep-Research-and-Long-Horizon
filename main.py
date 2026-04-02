"""
main.py - Deep Cognitive Task Framework: Milestone 4
Full Integration — Groq LLM + Streamlit Web UI

Fixes applied:
  - LLM is now built INSIDE run_agent() so sidebar key/model changes always take effect
  - build_agent() accepts llm_with_tools as a parameter (no more stale module-level object)
  - _invoke_with_retry() handles Groq 429 rate-limit errors with exponential backoff
  - Default recursion_limit lowered from 100 → 40 to stay within free-tier rate limits
  - interactive_mode() also builds a fresh LLM each run

New in Milestone 3:
  - Sub-Agent Delegation: `task` tool dispatches work to specialized sub-agents
    (summarization_agent, web_search_agent, code_analysis_agent)
  - `list_agents` tool lets the supervisor discover available sub-agents
  - delegation_log synced into AgentState after every `task` tool call
  - Updated display: shows delegation summary alongside VFS and TODO report
  - LangSmith project updated to milestone3-deep-agent

Carried from Milestone 2:
  - Virtual File System (ls, read_file, write_file, edit_file, delete_file)
  - Full execution loop: plan → execute → save → synthesize → output

Carried from Milestone 1:
  - write_todos / mark_todo_complete planning tools
"""

import json
import os
import time
from functools import partial
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# LangSmith Tracing — set BEFORE importing langchain
# ─────────────────────────────────────────────
# ── LangSmith: read dynamically so Streamlit sidebar changes take effect ──────
# Do NOT cache this at module level — os.environ may be updated after import.
def _tracing_enabled() -> bool:
    """Check at call-time whether LangSmith tracing is currently enabled."""
    return os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"

# Apply .env values on startup (Streamlit sidebar overwrites these later)
_env_key = os.getenv("LANGCHAIN_API_KEY", "")
if _env_key:
    os.environ["LANGCHAIN_API_KEY"] = _env_key
if not os.environ.get("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = "milestone4-deep-agent"

# For non-Streamlit use (terminal / test runner) print status once
LANGCHAIN_TRACING = _tracing_enabled()
if LANGCHAIN_TRACING:
    print(f"✅ LangSmith tracing ENABLED → Project: {os.environ.get('LANGCHAIN_PROJECT')}")
else:
    print("ℹ️  LangSmith tracing DISABLED (set LANGCHAIN_TRACING_V2=true in .env to enable)")

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from state import AgentState, TodoItem, DelegationEntry
from tools import ALL_TOOLS, PLANNING_TOOLS
from filesystem_tools import get_virtual_fs, set_virtual_fs

# ─────────────────────────────────────────────
# System Prompt — Milestone 3
# Teaches the agent: Plan → Execute (with delegation) → VFS → Synthesize
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Deep Cognitive Task Agent with planning, file system, and delegation capabilities.

CRITICAL RULES:
- Always complete ALL phases before finishing.
- NEVER output raw JSON as your final answer.
- Your final message MUST be a well-written prose report (at least 200 words).

════════════════════════════════════════════════════
PHASE 1 — PLANNING (ALWAYS do this first)
════════════════════════════════════════════════════
Call `write_todos` with 3 to 5 tasks (choose based on complexity).
Each task MUST start with: RESEARCH / ANALYZE / SYNTHESIZE / DRAFT / REVIEW

════════════════════════════════════════════════════
PHASE 2 — EXECUTE each TODO (in order)
════════════════════════════════════════════════════
For EACH todo, choose the best approach:

OPTION A — Delegate to a specialist sub-agent (PREFERRED for research/summarization/code):
  1. Call `task(agent_name, sub_task, context)` to delegate.
     Available agents:
       - "web_search_agent"     → deep research, facts, trends
       - "summarization_agent"  → condensing, comparing, summarizing
       - "code_analysis_agent"  → code review, tech decisions, architecture
  2. Call write_file to save the returned result.
  3. Call mark_todo_complete with the todo's ID.

OPTION B — Do the work yourself (for synthesis / drafting final output):
  1. Reason and produce the result in your message.
  2. Call write_file to save your findings.
  3. Call mark_todo_complete with the todo's ID.

WHEN TO DELEGATE vs DO IT YOURSELF:
  - RESEARCH tasks → always delegate to web_search_agent
  - ANALYZE tasks  → delegate to code_analysis_agent or summarization_agent
  - SYNTHESIZE / DRAFT tasks → do yourself (you have all the files)

════════════════════════════════════════════════════
PHASE 3 — SYNTHESIS (only after ALL todos are marked complete)
════════════════════════════════════════════════════
  STEP 1: Call read_file for each file you saved.
  STEP 2: Write your FINAL ANSWER as a full, well-structured prose report.
          - Plain text / markdown. NOT JSON. NOT tool output.
          - Minimum 200 words. Cover all key findings from the files.
          - Do NOT call any more tools. Just write the report.

════════════════════════════════════════════════════
FILE SYSTEM TOOLS
════════════════════════════════════════════════════
  write_file(filename, content) → save a file
  read_file(filename)           → read a file

GOLDEN RULE: Always delegate research tasks. Always save results. Always write a prose report last.
"""


# ─────────────────────────────────────────────
# State Update Helpers
# ─────────────────────────────────────────────

def extract_todos_from_messages(state: AgentState) -> list[TodoItem]:
    """Parse write_todos / mark_todo_complete results and sync into state."""
    todos = list(state.get("todos", []))
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "write_todos":
            try:
                result = json.loads(msg.content)
                if result.get("success") and "todos" in result:
                    existing_ids = {t["id"] for t in todos}
                    for todo in result["todos"]:
                        if todo["id"] not in existing_ids:
                            todos.append(todo)
            except (json.JSONDecodeError, KeyError):
                pass
        elif isinstance(msg, ToolMessage) and msg.name == "mark_todo_complete":
            try:
                result = json.loads(msg.content)
                if result.get("success"):
                    todo_id = result["todo_id"]
                    for todo in todos:
                        if todo["id"] == todo_id:
                            todo["status"] = "completed"
            except (json.JSONDecodeError, KeyError):
                pass
    return todos


def extract_virtual_files_from_messages(state: AgentState) -> dict:
    """
    Sync virtual_files from the live filesystem_tools module back into state.
    The authoritative VFS state is always in filesystem_tools._VIRTUAL_FS.
    """
    return get_virtual_fs()


def extract_delegation_log_from_messages(state: AgentState) -> list:
    """
    Parse all `task` tool ToolMessages and build the delegation log.
    Each entry records the agent called, the sub-task, and a result summary.
    """
    log = list(state.get("delegation_log", []))

    for msg in state["messages"]:
        if not (isinstance(msg, ToolMessage) and msg.name == "task"):
            continue
        try:
            result = json.loads(msg.content)
            if not result.get("success"):
                continue
            entry: DelegationEntry = {
                "agent_name": result.get("agent_name", "unknown"),
                "sub_task": result.get("sub_task", "")[:120],
                "result_summary": result.get("result", "")[:200],
                "duration_s": result.get("duration_s", 0.0),
            }
            # Deduplicate: check if this exact (agent_name, sub_task) pair already logged
            already_logged = any(
                e["agent_name"] == entry["agent_name"] and e["sub_task"] == entry["sub_task"]
                for e in log
            )
            if not already_logged:
                log.append(entry)
        except (json.JSONDecodeError, KeyError):
            pass
    return log


def check_write_todos_invoked(state: AgentState) -> bool:
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "write_todos":
            return True
    return False


def get_filesystem_tool_calls(state: AgentState) -> dict:
    """Count how many times each VFS tool was called."""
    counts = {"write_file": 0, "read_file": 0, "ls": 0, "edit_file": 0, "delete_file": 0}
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and msg.name in counts:
            counts[msg.name] += 1
    return counts


def get_delegation_tool_calls(state: AgentState) -> dict:
    """Count how many times each delegation tool was called."""
    counts = {"task": 0, "list_agents": 0}
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and msg.name in counts:
            counts[msg.name] += 1
    return counts


# ─────────────────────────────────────────────
# Graph Nodes
# NOTE: agent_node now receives llm_with_tools via partial() — no stale module-level object
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Rate-limit helpers
# ─────────────────────────────────────────────

import re as _re

# Groq free tier limits:
#   RPM  : 30 req/min  → 1 request every 2s minimum
#   TPM  : varies by model (~6k-14k tokens/min on free tier)
# We use 3s between calls to give headroom for both limits.
# Sub-agents share this same throttle via _throttle() import.
_MIN_CALL_INTERVAL = 3.0
_last_call_time: float = 0.0
_total_calls: int = 0


def _throttle():
    """
    Enforce minimum gap between ALL Groq LLM calls (supervisor + sub-agents).
    This single shared lock prevents bursts when a sub-agent fires immediately
    after the supervisor, which is the main cause of TPM hits.
    """
    global _last_call_time, _total_calls
    elapsed = time.time() - _last_call_time
    if elapsed < _MIN_CALL_INTERVAL:
        time.sleep(_MIN_CALL_INTERVAL - elapsed)
    _last_call_time = time.time()
    _total_calls += 1


def _parse_retry_after(exc: Exception) -> float:
    """
    Extract the wait time from a Groq rate-limit error.
    Handles both RPM (429) and TPM (tokens per minute) errors.
    Groq embeds the wait as 'Please try again in X.Xs'.
    Falls back to 30 s if not found.
    """
    msg = str(exc)
    match = _re.search(r"(?:try again in|retry.{0,10}after)\s*([\d.]+)\s*s", msg, _re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2   # small safety buffer
    match = _re.search(r"(\d+)\s*second", msg, _re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2
    return 30.0  # conservative default


def _is_rate_limit(exc: Exception) -> bool:
    """Return True for any Groq rate-limit error (RPM or TPM)."""
    err = str(exc).lower()
    return (
        "rate limit" in err
        or "ratelimit" in err
        or "429" in err
        or "too many requests" in err
        or "tokens per minute" in err
        or "tpm" in err
    )


def _trim_messages(messages: list, max_history: int = 6) -> list:
    """
    Keep the system prompt + the most recent `max_history` messages.
    This prevents the context window (and token count) from growing
    unboundedly across many agent steps, which is the main cause of
    TPM limit hits on Groq free tier.
    """
    from langchain_core.messages import SystemMessage as SM
    if not messages:
        return messages
    # Separate system message(s) from the rest
    system_msgs = [m for m in messages if isinstance(m, SM)]
    other_msgs  = [m for m in messages if not isinstance(m, SM)]
    # Keep only the tail of the conversation history
    trimmed = other_msgs[-max_history:]
    return system_msgs + trimmed


def _llm_invoke_with_backoff(llm_with_tools, messages, max_retries: int = 8, caller: str = "agent"):
    """
    Invoke the LLM with:
      1. Token-budget trimming — only send the last 12 non-system messages.
      2. Minimum inter-call throttle to respect Groq RPM limit.
      3. Precise back-off on both RPM (429) and TPM errors, honouring
         Groq's 'Please try again in Xs' hint.
    """
    trimmed = _trim_messages(messages)

    for attempt in range(max_retries):
        _throttle()
        try:
            return llm_with_tools.invoke(trimmed)

        except Exception as e:
            if _is_rate_limit(e) and attempt < max_retries - 1:
                groq_wait = _parse_retry_after(e)
                is_tpm = "token" in str(e).lower() or "tpm" in str(e).lower()
                err_type = "TPM (tokens/min)" if is_tpm else "RPM (req/min)"
                # TPM: wait the full Groq hint + 10s flat buffer so the token bucket refills.
                # RPM: wait the Groq hint + small per-attempt increment.
                extra = 10 if is_tpm else (attempt * 2)
                wait_s = groq_wait + extra
                print(f"⏳ [{caller}] {err_type} hit — Groq says {groq_wait:.1f}s → waiting {wait_s:.0f}s "
                      f"(attempt {attempt+1}/{max_retries})")
                time.sleep(wait_s)
                # Reset throttle timer so the next call doesn't fire instantly after sleep
                global _last_call_time
                _last_call_time = time.time()
            else:
                raise


def agent_node(state: AgentState, llm_with_tools) -> AgentState:
    """Main reasoning node — LLM decides what to do next."""
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    # Sync VFS from state into the tools module before invoking LLM
    set_virtual_fs(state.get("virtual_files", {}))

    # Use backoff wrapper so mid-run 429s are handled gracefully
    response = _llm_invoke_with_backoff(llm_with_tools, messages, caller="agent")

    return {
        "messages": [response],
        "todos": state.get("todos", []),
        "current_task": state.get("current_task", ""),
        "final_output": state.get("final_output", ""),
        "write_todos_invoked": state.get("write_todos_invoked", False),
        "virtual_files": state.get("virtual_files", {}),
        "delegation_log": state.get("delegation_log", []),
    }


def tool_node_wrapper(state: AgentState) -> AgentState:
    """Execute tool calls and sync VFS + todos + delegation_log back into state."""
    # Sync current VFS state INTO the tools module before executing
    set_virtual_fs(state.get("virtual_files", {}))

    tool_node = ToolNode(ALL_TOOLS)
    result = tool_node.invoke(state)

    updated_state = {**state, **result}

    # After tool execution, sync live VFS, todos, and delegation log from messages
    updated_vfs = get_virtual_fs()
    updated_todos = extract_todos_from_messages(updated_state)
    updated_delegation_log = extract_delegation_log_from_messages(updated_state)
    write_todos_invoked = check_write_todos_invoked(updated_state)

    return {
        **result,
        "todos": updated_todos,
        "write_todos_invoked": write_todos_invoked,
        "virtual_files": updated_vfs,
        "delegation_log": updated_delegation_log,
    }


def _looks_like_json(text: str) -> bool:
    """Return True if the text is purely a JSON blob (tool output leaked into AI message)."""
    stripped = text.strip()
    return stripped.startswith("{") or stripped.startswith("[")


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    # Hard cap on LLM calls to stay inside rate limits.
    ai_msg_count = sum(1 for m in state["messages"] if isinstance(m, AIMessage))
    if ai_msg_count >= 30:  # 5 todos × ~4 LLM rounds = 20 + synthesis headroom
        return END

    # Continue if the agent still has pending tool calls.
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # If the last AI message looks like raw JSON (e.g. ls() result echoed back),
    # the agent hasn't written its final prose yet — keep going.
    if isinstance(last_message, AIMessage) and last_message.content:
        if _looks_like_json(last_message.content):
            return "agent"

    return END


# ─────────────────────────────────────────────
# Build the LangGraph
# llm_with_tools is now injected via partial() so it's always fresh
# ─────────────────────────────────────────────

def build_agent(llm_with_tools):
    """Build and compile the LangGraph agent with a freshly-created LLM binding."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", partial(agent_node, llm_with_tools=llm_with_tools))
    graph.add_node("tools", tool_node_wrapper)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# ─────────────────────────────────────────────
# Rate-Limit Retry Helper
# ─────────────────────────────────────────────

def _invoke_with_retry(agent, initial_state: dict, config: dict, max_retries: int = 2) -> dict:
    """
    Outer safety net — retries the full agent run if it crashes entirely with a 429.
    Mid-run 429s are handled inside _llm_invoke_with_backoff; this catches anything
    that slips through (e.g. tool-node calls that also hit the API).
    """
    for attempt in range(max_retries):
        try:
            return agent.invoke(initial_state, config=config)
        except Exception as e:
            err_lower = str(e).lower()
            is_rate_limit = (
                "rate limit" in err_lower
                or "429" in err_lower
                or "ratelimit" in err_lower
                or "too many requests" in err_lower
            )
            if is_rate_limit and attempt < max_retries - 1:
                wait_s = max(_parse_retry_after(e), 90)
                print(f"⏳ Outer rate-limit fallback — waiting {wait_s:.0f}s before retry…")
                time.sleep(wait_s)
            else:
                raise
    raise RuntimeError("Max retries exceeded due to rate limiting.")


# ─────────────────────────────────────────────
# Run Agent
# ─────────────────────────────────────────────

def run_agent(user_request: str, run_name: str = "agent-run", recursion_limit: int = 40) -> dict:
    """
    Run the full Milestone 4 agent on a user request.

    The LLM is constructed here — not at module level — so API key and model
    changes made in the Streamlit sidebar always take effect immediately.

    Args:
        user_request   : The task to solve.
        run_name       : Label for LangSmith tracing.
        recursion_limit: Max LangGraph node visits (agent + tool calls each count).
                         Default 40 supports up to 5 todos with delegation + synthesis.
    """
    # ── Build a fresh LLM from current env vars ──────────────────────────────
    groq_key   = os.environ.get("GROQ_API_KEY", "").strip()
    groq_model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

    if not groq_key:
        raise ValueError("GROQ_API_KEY is not set. Please enter your API key in the sidebar.")

    llm = ChatGroq(
        model=groq_model,
        groq_api_key=groq_key,
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # ── Reset VFS for each fresh run ─────────────────────────────────────────
    set_virtual_fs({})

    agent = build_agent(llm_with_tools)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_request)],
        "todos": [],
        "current_task": "",
        "final_output": "",
        "write_todos_invoked": False,
        "virtual_files": {},
        "delegation_log": [],
    }

    print(f"\n{'='*65}")
    print(f"  REQUEST : {user_request[:75]}...")
    print(f"  MODEL   : {groq_model}")
    print(f"  MAX STEPS: {recursion_limit}")
    print(f"{'='*65}")

    # Build invoke config — always include run_name.
    # When tracing is enabled, attach a LangChainTracer callback explicitly.
    # This is the most reliable method: it doesn't depend on env-var timing
    # and works even when Streamlit has already imported langchain once.
    invoke_config: dict = {
        "recursion_limit": recursion_limit,
        "run_name": run_name,
    }
    if _tracing_enabled():
        project = os.environ.get("LANGCHAIN_PROJECT", "milestone4-deep-agent")
        api_key = os.environ.get("LANGCHAIN_API_KEY", "")
        try:
            # Try modern path first (langchain >= 0.2)
            try:
                from langchain_core.tracers import LangChainTracer
            except ImportError:
                from langchain.callbacks.tracers import LangChainTracer
            tracer = LangChainTracer(project_name=project)
            invoke_config["callbacks"] = [tracer]
            print(f"🔗 LangSmith tracing ACTIVE → project: {project} | run: {run_name}")
        except Exception as e:
            print(f"⚠️  LangSmith tracer init failed: {e} — running without tracing")

    # Use retry wrapper to handle Groq rate limits gracefully
    final_state = _invoke_with_retry(agent, initial_state, invoke_config)

    print(f"\n  ⏳ Syncing final state...")

    # Ensure final state is fully synced
    final_state["virtual_files"]      = get_virtual_fs()
    final_state["write_todos_invoked"] = check_write_todos_invoked(final_state)
    final_state["todos"]              = extract_todos_from_messages(final_state)
    final_state["delegation_log"]     = extract_delegation_log_from_messages(final_state)

    # Extract final AI prose output — skip raw JSON blobs (leaked tool results)
    for msg in reversed(final_state["messages"]):
        if (
            isinstance(msg, AIMessage)
            and msg.content
            and not getattr(msg, "tool_calls", [])
            and not _looks_like_json(msg.content)
        ):
            final_state["final_output"] = msg.content
            break

    print(f"  ✅ Done. Final output length: {len(final_state.get('final_output', ''))} chars")
    return final_state


# ─────────────────────────────────────────────
# Display Results
# ─────────────────────────────────────────────

def display_results(state: dict, user_request: str):
    todos          = state.get("todos", [])
    vfs            = state.get("virtual_files", {})
    delegation_log = state.get("delegation_log", [])
    fs_calls       = get_filesystem_tool_calls(state)
    del_calls      = get_delegation_tool_calls(state)

    # ── Task Plan ─────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  📋  TASK PLAN")
    print(f"{'─'*65}")
    completed = sum(1 for t in todos if t["status"] == "completed")
    print(f"  Total tasks : {len(todos)}  |  Completed: {completed}  |  Pending: {len(todos)-completed}")
    print()
    for i, todo in enumerate(todos, 1):
        icon = "✅" if todo["status"] == "completed" else "⏳"
        print(f"  {icon} [{todo['id']}] {i}. {todo['task'][:65]}")

    # ── Delegation Summary ─────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  🤖  DELEGATION SUMMARY")
    print(f"{'─'*65}")
    print(f"  task() called   : {del_calls['task']} time(s)  |  list_agents(): {del_calls['list_agents']} time(s)")
    if delegation_log:
        print()
        for i, entry in enumerate(delegation_log, 1):
            print(f"  [{i}] Agent     : {entry['agent_name']}")
            print(f"      Sub-task  : {entry['sub_task'][:70]}")
            print(f"      Duration  : {entry['duration_s']}s")
            summary = entry['result_summary'].replace('\n', ' ')[:80]
            print(f"      Preview   : {summary}...")
            print()
    else:
        print("  ℹ️  No sub-agent delegations were made for this request.")

    # ── Virtual File System ────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  🗂️   VIRTUAL FILE SYSTEM")
    print(f"{'─'*65}")
    print(f"  Files saved : {len(vfs)}")
    print(f"  Tool calls  → write_file: {fs_calls['write_file']}  |  read_file: {fs_calls['read_file']}  |  ls: {fs_calls['ls']}  |  edit_file: {fs_calls['edit_file']}")
    if vfs:
        print()
        for fname, content in vfs.items():
            preview = content[:80].replace("\n", " ")
            print(f"  📄 {fname}  ({len(content)} chars)")
            print(f"     Preview: {preview}...")
    else:
        print("  ⚠️  No files were written to the virtual file system.")

    # ── Final Output ───────────────────────────────────────────────
    final_output = state.get("final_output", "")
    # Fallback: scan messages for last non-JSON AI prose
    if not final_output:
        for msg in reversed(state["messages"]):
            if (
                isinstance(msg, AIMessage)
                and msg.content
                and not getattr(msg, "tool_calls", [])
                and not _looks_like_json(msg.content)
            ):
                final_output = msg.content
                break
    if final_output:
        print(f"\n{'─'*65}")
        print("  📝  FINAL OUTPUT")
        print(f"{'─'*65}")
        print(final_output[:1500])
        if len(final_output) > 1500:
            print(f"\n  ... [{len(final_output) - 1500} more chars]")

    # ── Save to Disk ───────────────────────────────────────────────
    output = {
        "request":              user_request,
        "todos":                todos,
        "delegation_log":       delegation_log,
        "delegation_tool_calls": del_calls,
        "virtual_files":        vfs,
        "fs_tool_calls":        fs_calls,
        "final_output":         final_output,
    }
    with open("milestone4_output.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'─'*65}")
    print("  💾  Full output saved to milestone4_output.json")
    print(f"{'─'*65}\n")


# ─────────────────────────────────────────────
# Interactive Mode
# ─────────────────────────────────────────────

def interactive_mode():
    """Interactive terminal loop for the agent."""
    print("\n" + "=" * 65)
    print("  🧠  DEEP COGNITIVE TASK AGENT — Milestone 4")
    print("  Sub-Agent Delegation + Context Offloading via VFS")
    print("  Powered by LangGraph + Groq")
    print("=" * 65)
    print("  Enter a complex research, analysis, or coding request.")
    print("  The agent will plan, delegate to specialists, and synthesize.")
    print()
    print("  Commands:")
    print("    'files'          — show current virtual file system")
    print("    'agents'         — list available sub-agents")
    print("    'quit' / 'exit'  — stop the agent")
    print("    'clear'          — start fresh")
    print("=" * 65 + "\n")

    run_count = 0

    while True:
        try:
            user_input = input("  You: ").strip()

            if not user_input:
                print("  ⚠️  Please enter a request.\n")
                continue

            if user_input.lower() in ("quit", "exit"):
                print("\n  👋  Goodbye!\n")
                break

            if user_input.lower() == "clear":
                os.system("cls" if os.name == "nt" else "clear")
                interactive_mode()
                return

            if user_input.lower() == "files":
                vfs = get_virtual_fs()
                if not vfs:
                    print("  📂  Virtual file system is empty.\n")
                else:
                    print(f"\n  📂  Virtual File System ({len(vfs)} files):")
                    for fname, content in vfs.items():
                        print(f"    📄 {fname}  ({len(content)} chars)")
                    print()
                continue

            if user_input.lower() == "agents":
                from sub_agents.registry import list_available_agents
                print("\n  🤖  Available Sub-Agents:")
                for agent_info in list_available_agents():
                    print(f"\n    📌 {agent_info['name']}")
                    print(f"       {agent_info['description'][:100]}...")
                print()
                continue

            run_count += 1
            print(f"\n  ⏳  Working on your task (may take a moment — sub-agents are spawning)...\n")

            state = run_agent(
                user_request=user_input,
                run_name=f"m4-interactive-{run_count}",
            )

            display_results(state, user_input)

        except KeyboardInterrupt:
            print("\n\n  👋  Interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n  ❌  Error: {e}\n")
            import traceback
            traceback.print_exc()
            print("  Please try again.\n")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    interactive_mode()