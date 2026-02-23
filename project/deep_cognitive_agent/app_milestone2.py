"""
Milestone 2: Agent with Context Offloading via Virtual File System
==================================================================

Architecture
------------
LangGraph StateGraph with three nodes:

    START ──► plan ──► execute ──► synthesize ──► END

State structure:
    state = {
        "todos":        [],       # structured TODO list
        "files":        {},       # virtual file system  (filename → content)
        "messages":     [],       # conversation messages
        "final_output": "",       # combined structured summary
        "current_step": None,
    }

Workflow (e.g. for climate-change task):
    1. Plan  → write_todos creates 4-6 TODO steps
    2. Execute step 1 → LLM generates paragraph → write_file("summary1.txt")
    3. Execute step 2 → LLM generates paragraph → write_file("summary2.txt")
    4. Execute step 3 → LLM generates paragraph → write_file("summary3.txt")
    5. Synthesize → read_file("summary1.txt")
    6. Synthesize → read_file("summary2.txt")
    7. Synthesize → read_file("summary3.txt")
    8. Synthesize → LLM generates combined structured summary

Important rules:
    • write_file is used for each individual summary
    • read_file is used before final synthesis
    • No bypass of file system — agent offloads context to VFS
    • File content is visible in state["files"]
"""

import os
import json
import time
from typing import Dict
from functools import partial

from dotenv import load_dotenv

# Load environment variables BEFORE any LangChain imports
load_dotenv()

# LangSmith tracing (optional)
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "milestone_2_vfs")

# Validate API key early
_groq_key = os.getenv("GROQ_API_KEY", "")
if not _groq_key or _groq_key.startswith("your_"):
    raise SystemExit(
        "\n[ERROR] GROQ_API_KEY is missing or still set to the placeholder.\n"
        "       1. Go to https://console.groq.com/keys and create an API key.\n"
        "       2. Put it in project/deep_cognitive_agent/.env:\n"
        "          GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx\n"
        "       3. Or set it in PowerShell:  $env:GROQ_API_KEY = \"gsk_xxx\"\n"
    )

from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# ── VFS tool functions (operate ONLY on state["files"]) ──
from tools.vfs.write_file import write_file
from tools.vfs.read_file import read_file
from tools.vfs.ls import ls
from tools.vfs.edit_file import edit_file

# ── Planning tool ──
from tools.planning.write_todos import write_todos

# ── Graph builder ──
from graphs.main_graph import build_graph


# ── LLM Initialization ──────────────────────────────────────────────

_model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
print(f"[init] Using Groq model: {_model_name}")

llm = ChatGroq(
    model=_model_name,
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)


# ── LangChain Tool wrappers ─────────────────────────────────────────
#
# These wrap the raw VFS functions into LangChain StructuredTool
# objects.  A shared ``_vfs_state`` dict is bound via functools.partial
# so the LLM-facing tool signatures only expose (filename, content).

_vfs_state: Dict = {"files": {}}


# Wrap partials in proper functions so StructuredTool can inspect
# type hints and build schemas correctly.

def _write_file_bound(filename: str, content: str) -> str:
    """Write content to a virtual file."""
    return write_file(_vfs_state, filename, content)


def _read_file_bound(filename: str) -> str:
    """Read content from a virtual file."""
    return read_file(_vfs_state, filename)


def _ls_bound() -> list:
    """List all files."""
    return ls(_vfs_state)


def _edit_file_bound(filename: str, new_content: str) -> str:
    """Edit content of a virtual file."""
    return edit_file(_vfs_state, filename, new_content)


write_file_tool = StructuredTool.from_function(
    func=_write_file_bound,
    name="write_file",
    description="Write content to a virtual file.",
)

read_file_tool = StructuredTool.from_function(
    func=_read_file_bound,
    name="read_file",
    description="Read content from a virtual file.",
)

ls_tool = StructuredTool.from_function(
    func=_ls_bound,
    name="ls",
    description="List all files.",
)

edit_file_tool = StructuredTool.from_function(
    func=_edit_file_bound,
    name="edit_file",
    description="Edit content of a virtual file.",
)


# ── Runner ───────────────────────────────────────────────────────────

def run_milestone2(task: str) -> Dict:
    """
    Run the Milestone 2 agent pipeline:
        plan → execute (with write_file) → synthesize (with read_file)

    Args:
        task: The user's natural-language task description.

    Returns:
        The final AgentState dict with todos, files, and final_output.
    """
    print("=" * 70)
    print("  Milestone 2: Agent with Context Offloading (VFS)")
    print("=" * 70)
    print(f"\n  Task: {task}\n")

    # Build the LangGraph StateGraph
    graph = build_graph(llm)

    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "todos": [],
        "files": {},
        "final_output": "",
        "current_step": None,
    }

    # Run the graph
    final_state = graph.invoke(initial_state)

    # ── Print results ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)

    # TODOs
    print("\n--- TODOs ---")
    for i, todo in enumerate(final_state.get("todos", []), 1):
        status = todo.get("status", "pending")
        icon = "✅" if status == "done" else "⬜"
        print(f"  {i}. {icon} {todo['task']}  [{status}]")

    # Virtual File System contents
    print("\n--- state[\"files\"] (Virtual File System) ---")
    files = final_state.get("files", {})
    if files:
        for fname in sorted(files.keys()):
            content = files[fname]
            print(f"\n  📄 {fname}  ({len(content)} chars):")
            preview = content[:300].replace("\n", "\n    ")
            print(f"    {preview}")
            if len(content) > 300:
                print(f"    ... ({len(content) - 300} more chars)")
    else:
        print("  (empty)")

    # Final output
    print("\n--- Final Structured Summary ---")
    final_output = final_state.get("final_output", "")
    if final_output:
        print(final_output)
    else:
        print("  (no output)")

    # Save to JSON
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "milestone2_output.json")
    serializable = {
        "task": task,
        "todos": final_state.get("todos", []),
        "files": final_state.get("files", {}),
        "final_output": final_state.get("final_output", ""),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Saved results to {output_path}")

    print("\n" + "=" * 70)
    print("  Milestone 2 complete.")
    print("=" * 70)

    return final_state


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Milestone 2: Agent with Context Offloading via VFS",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The task for the agent to execute. Can be ANY task.",
    )
    args = parser.parse_args()

    run_milestone2(args.task)
