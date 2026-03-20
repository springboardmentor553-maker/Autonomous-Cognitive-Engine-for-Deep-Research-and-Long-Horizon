"""
main.py - Deep Cognitive Task Framework: Milestone 3
Sub-Agent Delegation via LangGraph Supervisor Pattern

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
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# LangSmith Tracing — set BEFORE importing langchain
# ─────────────────────────────────────────────
LANGCHAIN_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
if LANGCHAIN_TRACING:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "milestone3-deep-agent")
    print(f"✅ LangSmith tracing ENABLED → Project: {os.environ['LANGCHAIN_PROJECT']}")
else:
    print("ℹ️  LangSmith tracing DISABLED (set LANGCHAIN_TRACING_V2=true in .env to enable)")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from state import AgentState, TodoItem, DelegationEntry
from tools import ALL_TOOLS, PLANNING_TOOLS
from filesystem_tools import get_virtual_fs, set_virtual_fs

# ─────────────────────────────────────────────
# LLM Setup
# ─────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

# gemini-2.0-flash: 15 RPM free tier — switched from gemini-2.0-flash-lite (daily quota exhausted)
# Switch back to "gemini-2.0-flash-lite" tomorrow after midnight UTC when quota resets.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

llm_with_tools = llm.bind_tools(ALL_TOOLS)

# ─────────────────────────────────────────────
# System Prompt — Milestone 3
# Teaches the agent: Plan → Execute (with delegation) → VFS → Synthesize
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Deep Cognitive Task Agent — Supervisor Edition.
You coordinate planning, context management, and specialized sub-agent delegation.

════════════════════════════════════════════════════
PHASE 1 — PLANNING (Always do this first)
════════════════════════════════════════════════════
Call `write_todos` IMMEDIATELY with EXACTLY 5 tasks.
Each task MUST start with: RESEARCH / ANALYZE / SYNTHESIZE / DRAFT / REVIEW

════════════════════════════════════════════════════
PHASE 2A — DIRECT EXECUTION (For simple tasks)
════════════════════════════════════════════════════
For each TODO where YOU can do the work:
  STEP A — DO THE WORK: Reason and produce the result.
  STEP B — SAVE (MANDATORY): Call write_file to save the result.
  STEP C — MARK DONE: Call mark_todo_complete with the TODO's ID.

════════════════════════════════════════════════════
PHASE 2B — DELEGATION (For specialist tasks)
════════════════════════════════════════════════════
When a TODO requires deep research, summarization, or code work,
DELEGATE it to a specialized sub-agent:

  STEP 1 — IDENTIFY: If unsure which agent to use, call list_agents().
  STEP 2 — DELEGATE: Call task(agent_name, sub_task, context) with:
    - agent_name: one of "summarization_agent", "web_search_agent", "code_analysis_agent"
    - sub_task  : clear description of what the agent should do
    - context   : (optional) relevant content, constraints, or background

  STEP 3 — SAVE RESULT: Extract the "result" field from the response.
            Call write_file to save it to the virtual file system.
  STEP 4 — MARK DONE: Call mark_todo_complete.

WHEN TO DELEGATE:
  - RESEARCH tasks → use "web_search_agent"
  - Summarizing large content → use "summarization_agent"
  - Code review, drafting, analysis → use "code_analysis_agent"
  - Simple writing/analysis YOU can do → do it directly (Phase 2A)

════════════════════════════════════════════════════
PHASE 3 — SYNTHESIS (After all TODOs are done)
════════════════════════════════════════════════════
  1. Call ls("/") to list all saved files.
  2. Call read_file on each relevant file to gather all results.
  3. Combine everything into a comprehensive final output.

════════════════════════════════════════════════════
FILE SYSTEM TOOLS — Quick Reference
════════════════════════════════════════════════════
  ls("/")                          → list all files in virtual FS
  write_file(filename, content)    → save/overwrite a file
  read_file(filename)              → read a file's content
  edit_file(filename, old, new)    → update part of a file
  delete_file(filename)            → remove a file

════════════════════════════════════════════════════
DELEGATION TOOLS — Quick Reference
════════════════════════════════════════════════════
  list_agents()                          → discover available sub-agents
  task(agent_name, sub_task, context)    → delegate to a sub-agent

GOLDEN RULE: Never lose information between steps.
If you gathered or delegated something, WRITE IT TO A FILE immediately.
Use read_file before synthesizing so you have ALL prior work available.
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
    seen_ids = {id(entry) for entry in log}  # crude dedup by identity

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
# ─────────────────────────────────────────────

def agent_node(state: AgentState) -> AgentState:
    """Main reasoning node — LLM decides what to do next."""
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    # Sync VFS from state into the tools module before invoking LLM
    set_virtual_fs(state.get("virtual_files", {}))

    # ── Retry with exponential backoff on 429 rate-limit errors ──
    max_retries = 5
    base_delay  = 15  # seconds; free tier resets each minute
    for attempt in range(max_retries):
        try:
            response = llm_with_tools.invoke(messages)
            time.sleep(2)  # proactive pacing: 30 RPM = 1 call / 2s
            break  # success — exit retry loop
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                if attempt < max_retries - 1:
                    wait = base_delay * (2 ** attempt)  # 15s, 30s, 60s, 120s …
                    print(f"  ⏳  Rate limit hit — waiting {wait}s before retry "
                          f"({attempt + 1}/{max_retries - 1})…")
                    time.sleep(wait)
                else:
                    raise  # all retries exhausted
            else:
                raise  # non-rate-limit error — surface immediately

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


def should_continue(state: AgentState) -> str:
    """Route to tools if pending tool calls exist, otherwise end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ─────────────────────────────────────────────
# Build the LangGraph
# ─────────────────────────────────────────────

def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node_wrapper)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# ─────────────────────────────────────────────
# Run Agent
# ─────────────────────────────────────────────

def run_agent(user_request: str, run_name: str = "agent-run") -> dict:
    """
    Run the full Milestone 3 agent on a user request.
    Returns the final AgentState including todos, virtual_files,
    delegation_log, and final_output.
    """
    # Reset VFS for each fresh run
    set_virtual_fs({})

    agent = build_agent()

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
    print(f"  REQUEST: {user_request[:75]}...")
    print(f"{'='*65}")

    invoke_config = {"run_name": run_name} if LANGCHAIN_TRACING else {}
    final_state = agent.invoke(initial_state, config=invoke_config)

    # Ensure final state is fully synced
    final_state["virtual_files"] = get_virtual_fs()
    final_state["write_todos_invoked"] = check_write_todos_invoked(final_state)
    final_state["todos"] = extract_todos_from_messages(final_state)
    final_state["delegation_log"] = extract_delegation_log_from_messages(final_state)

    return final_state


# ─────────────────────────────────────────────
# Display Results
# ─────────────────────────────────────────────

def display_results(state: dict, user_request: str):
    todos = state.get("todos", [])
    vfs = state.get("virtual_files", {})
    delegation_log = state.get("delegation_log", [])
    fs_calls = get_filesystem_tool_calls(state)
    del_calls = get_delegation_tool_calls(state)

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
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.content:
        print(f"\n{'─'*65}")
        print("  📝  FINAL OUTPUT")
        print(f"{'─'*65}")
        print(last_msg.content[:1500])
        if len(last_msg.content) > 1500:
            print(f"\n  ... [{len(last_msg.content) - 1500} more chars]")

    # ── Save to Disk ───────────────────────────────────────────────
    output = {
        "request": user_request,
        "todos": todos,
        "delegation_log": delegation_log,
        "delegation_tool_calls": del_calls,
        "virtual_files": vfs,
        "fs_tool_calls": fs_calls,
        "final_output": last_msg.content if isinstance(last_msg, AIMessage) else "",
    }
    with open("milestone3_output.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'─'*65}")
    print("  💾  Full output saved to milestone3_output.json")
    print(f"{'─'*65}\n")


# ─────────────────────────────────────────────
# Interactive Mode
# ─────────────────────────────────────────────

def interactive_mode():
    """Interactive terminal loop for the Milestone 3 agent."""
    print("\n" + "=" * 65)
    print("  🧠  DEEP COGNITIVE TASK AGENT — Milestone 3")
    print("  Sub-Agent Delegation + Context Offloading via VFS")
    print("  Powered by LangGraph + Google Gemini")
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
                run_name=f"m3-interactive-{run_count}"
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