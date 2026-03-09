"""
Main Application - Milestone 2: ReAct Agent with Virtual File System

Extends Milestone 1 by adding:
  - Virtual file system tools: write_file, read_file, ls, edit_file
  - State extended with 'files' dict alongside 'todos'
  - Intelligent memory: summaries stored in VFS, selective retrieval only
  - Multi-step dependency chains: summarize → store → compare → unify → refine
  - LangSmith tracing enabled
"""

import os
import json
import ast
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# ── LangSmith tracing ──────────────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Milestone2-VFS-Agent")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# ── Milestone 1 tool (preserved) ───────────────────────────────────────────
from tools.planning.write_todos import write_todos

# ── Milestone 2 VFS tools ─────────────────────────────────────────────────
from tools.filesystem.vfs_tools import (
    write_file,
    read_file,
    ls,
    edit_file,
    reset_vfs,
    get_vfs_snapshot,
)

from graphs.state import AgentState

# ── LLM ───────────────────────────────────────────────────────────────────
# LLM initialized lazily inside create_milestone2_agent()
_llm_instance = None

def _get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )
    return _llm_instance


# ── Tool definitions ──────────────────────────────────────────────────────

write_todos_tool = Tool(
    name="write_todos",
    func=write_todos,
    description=(
        "MUST be called FIRST for any complex task. "
        "Breaks the task into exactly 5 structured, actionable TODO steps. "
        "Input: task description string. "
        "Output: list of todo dicts with 'task' and 'status' fields."
    ),
)

write_file_tool = Tool(
    name="write_file",
    func=write_file,
    description=(
        "Write content to a named virtual file for later retrieval. "
        "Use this to store summaries, analysis results, comparisons, or any "
        "intermediate output — NOT raw document text. "
        "Input format: 'filename.txt|content to store' (pipe-separated). "
        "Example: 'doc1_summary.txt|Key points: renewable energy policy...'"
    ),
)

read_file_tool = Tool(
    name="read_file",
    func=read_file,
    description=(
        "Read the content of a specific virtual file by name. "
        "ONLY read files that are actually needed for the current step. "
        "Do NOT read all files blindly — be selective to avoid wasting context. "
        "Input: filename string (e.g., 'doc1_summary.txt'). "
        "Output: file content, or error if not found."
    ),
)

ls_tool = Tool(
    name="ls",
    func=ls,
    description=(
        "List all files currently stored in the virtual file system. "
        "Use this to check what has been written before reading. "
        "Input: empty string or any string (ignored). "
        "Output: list of filenames."
    ),
)

edit_file_tool = Tool(
    name="edit_file",
    func=edit_file,
    description=(
        "Edit (overwrite) an existing virtual file with new content. "
        "Use this for refinement steps — when you need to update a file "
        "that was previously written (e.g., refining a unified model). "
        "The file must already exist (use write_file to create new files). "
        "Input format: 'filename.txt|updated content' (pipe-separated). "
        "Example: 'unified_model.txt|Refined model with sustainability...'"
    ),
)

ALL_TOOLS = [
    write_todos_tool,
    write_file_tool,
    read_file_tool,
    ls_tool,
    edit_file_tool,
]

# ── System prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a strict planning and execution agent with a virtual file system.

You MUST use tools to complete every task. Never just describe what you would do — actually DO it by calling tools.

MANDATORY EXECUTION RULES:
1. Call write_todos FIRST with the task description.
2. Then execute each step by calling the appropriate tools in order.
3. write_file to store summaries and outputs (format: "filename.txt|content").
4. ls to verify files exist before reading.
5. read_file ONLY on files needed for the current step — not all files.
6. edit_file to refine/update an existing file (format: "filename.txt|new content").

CRITICAL: You must actually CALL the tools — do not just plan or describe the steps.
Every task must result in at least: write_todos called + write_file called + files stored in VFS.

Tool call sequence for a summarization+comparison task:
  write_todos(task description)
  write_file("doc1_summary.txt|<actual summary text here>")
  write_file("doc2_summary.txt|<actual summary text here>")
  ls("")
  read_file("doc1_summary.txt")
  read_file("doc2_summary.txt")
  write_file("comparison.txt|<actual comparison text here>")
  read_file("comparison.txt")
  edit_file("comparison.txt|<refined content here>")

Do not stop after write_todos. Continue executing all steps using tools.
"""


# ── Agent factory ─────────────────────────────────────────────────────────

def create_milestone2_agent():
    """Create the Milestone 2 ReAct agent with all tools."""
    memory = MemorySaver()
    agent = create_react_agent(
        model=_get_llm(),
        tools=ALL_TOOLS,
        checkpointer=memory,
        prompt=SYSTEM_PROMPT,
    )
    return agent


# ── Runner ────────────────────────────────────────────────────────────────

def run_agent(agent, task: str, thread_id: str = "default") -> Dict[str, Any]:
    """
    Run the agent on a task and return result with todos, files, and messages.

    Args:
        agent:     The ReAct agent instance.
        task:      The complex task description.
        thread_id: Unique thread ID for conversation memory.

    Returns:
        Dict with 'task', 'messages', 'todos', 'files'.
    """
    reset_vfs()  # Fresh VFS for each task run

    config = {"configurable": {"thread_id": thread_id}}
    input_message = {"messages": [("user", task)]}

    final_state = None
    todos = []

    for event in agent.stream(input_message, config, stream_mode="values"):
        final_state = event
        if "messages" in event:
            for msg in event["messages"]:
                if hasattr(msg, "name") and msg.name == "write_todos":
                    try:
                        content = msg.content
                        if isinstance(content, str):
                            clean = content.strip()
                            if clean.startswith("["):
                                todos = ast.literal_eval(clean)
                        elif isinstance(content, list):
                            todos = content
                    except Exception:
                        pass

    vfs_snapshot = get_vfs_snapshot()

    return {
        "task": task,
        "messages": final_state.get("messages", []) if final_state else [],
        "todos": todos,
        "files": vfs_snapshot,
    }


# ── Output persistence ────────────────────────────────────────────────────

def save_result_to_json(result: Dict[str, Any], filename: str, output_dir: str = "outputs"):
    """Save a run result to JSON for review."""
    os.makedirs(output_dir, exist_ok=True)

    serializable = {
        "task": result["task"],
        "todos": result["todos"],
        "files": result["files"],
        "message_count": len(result["messages"]),
    }

    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and hasattr(msg, "type") and msg.type == "ai":
            serializable["final_response"] = msg.content
            break

    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved → {filepath}")
    return filepath


# ── Display helpers ───────────────────────────────────────────────────────

def display_result(result: Dict[str, Any]):
    """Pretty-print a task result."""
    print(f"\n{'─'*60}")
    print(f"TASK : {result['task']}")
    print(f"{'─'*60}")

    print("\n📋 TODOs:")
    for i, todo in enumerate(result["todos"], 1):
        status = todo.get("status", "?")
        print(f"  {i}. [{status}] {todo['task']}")

    print("\n📁 Virtual File System:")
    if result["files"]:
        for fname, content in result["files"].items():
            preview = content[:120].replace("\n", " ")
            ellipsis = "…" if len(content) > 120 else ""
            print(f"  • {fname}  ({len(content)} chars)")
            print(f"    └─ {preview}{ellipsis}")
    else:
        print("  (no files written)")

    print(f"\n💬 Message count: {len(result['messages'])}")


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Milestone 2: ReAct Agent with Virtual File System")
    print("=" * 60)

    agent = create_milestone2_agent()

    test_task = (
        "Read 3 short paragraphs about climate change and create one final summary."
    )

    print(f"\nRunning task: {test_task}")
    result = run_agent(agent, test_task, thread_id="m2-smoke-test")
    display_result(result)
    save_result_to_json(result, "smoke_test_output.json")

    print("\n" + "=" * 60)
    print("Done. Check LangSmith for full traces.")
    print("=" * 60)
