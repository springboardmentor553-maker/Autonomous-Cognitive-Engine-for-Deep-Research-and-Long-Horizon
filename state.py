"""
state.py - LangGraph State Definition for Deep Cognitive Task Framework
Milestone 1: Foundational Agent & Task Planning
"""

from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class TodoItem(TypedDict):
    id: str
    task: str
    status: str   # "pending" | "in_progress" | "completed"
    notes: str


class AgentState(TypedDict):
    """
    The shared state for the Deep Cognitive Agent.

    - messages            : Full conversation + tool message history
    - todos               : Structured list of TodoItem dicts (the agent's plan)
    - current_task        : The task currently being worked on
    - final_output        : Final synthesized result
    - write_todos_invoked : Tracks whether write_todos was called (used by evaluator)
    """
    messages: Annotated[list, add_messages]
    todos: list[TodoItem]
    current_task: str
    final_output: str
    write_todos_invoked: bool