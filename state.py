"""
state.py - LangGraph State Definition for Deep Cognitive Task Framework
Milestone 3: Adds delegation_log for Sub-Agent Delegation tracking
"""

from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class TodoItem(TypedDict):
    id: str
    task: str
    status: str   # "pending" | "in_progress" | "completed"
    notes: str


class DelegationEntry(TypedDict):
    agent_name: str   # e.g. "web_search_agent"
    sub_task: str     # the task delegated
    result_summary: str  # first 200 chars of the result
    duration_s: float    # how long the sub-agent took


class AgentState(TypedDict):
    """
    Shared state for the Deep Cognitive Agent — Milestone 3.

    New in Milestone 3:
    - delegation_log : list[DelegationEntry] tracking every sub-agent call.
                       Populated by the orchestrator when a 'task' tool result
                       is observed, enabling delegation summaries and audit trails.

    Carried from Milestone 2:
    - virtual_files : dict[filename -> content] acting as an in-state scratchpad.

    Carried from Milestone 1:
    - messages            : Full conversation + tool message history
    - todos               : Structured list of TodoItem dicts
    - current_task        : The task currently being worked on
    - final_output        : Final synthesized result
    - write_todos_invoked : Tracks whether write_todos was called
    """
    messages: Annotated[list, add_messages]
    todos: list[TodoItem]
    current_task: str
    final_output: str
    write_todos_invoked: bool
    # ── Milestone 2 ──────────────────────────────
    virtual_files: dict           # filename (str) → content (str)
    # ── Milestone 3 ──────────────────────────────
    delegation_log: list          # list[DelegationEntry]