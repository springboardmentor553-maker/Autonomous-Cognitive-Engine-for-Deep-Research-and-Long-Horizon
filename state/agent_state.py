

"""
state/agent_state.py — Shared LangGraph state schema.

All nodes in the graph read/write a single AgentState TypedDict.
Key sections:
  - messages      : the running conversation / tool call history
  - todos         : structured task list (planned sub-tasks)
  - virtual_fs    : dict acting as an in-memory file system (milestone 2)
  - current_task  : which TODO is being worked on right now
  - iteration     : safety counter to prevent infinite loops
"""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ─── TODO item ────────────────────────────────────────────────────────────────

class TodoItem(TypedDict):
    id: str          # e.g. "task_1"
    description: str # human-readable task description
    status: str      # "pending" | "in_progress" | "completed" | "failed"
    result: str      # summary of what was produced (filled after completion)


# ─── Virtual File System entry ────────────────────────────────────────────────

class VFSFile(TypedDict):
    content: str
    created_at: str   # ISO timestamp
    updated_at: str


# ─── Main agent state ─────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # LangGraph appends new messages automatically via add_messages reducer
    messages: Annotated[list[BaseMessage], add_messages]

    # Structured task plan created by write_todos tool
    todos: list[TodoItem]

    # In-memory virtual file system: {"/notes/summary.md": VFSFile, ...}
    virtual_fs: dict[str, VFSFile]

    # ID of the TODO currently being executed
    current_task_id: str | None

    # How many reasoning iterations have run (guard against infinite loops)
    iteration: int

    # Final synthesised answer produced at the end
    final_output: str | None

    # Original user request (kept for reference throughout)
    user_request: str
