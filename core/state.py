"""
LangGraph AgentState definition for the Autonomous Cognitive Engine.

Defines the shared state schema passed through the graph at every node.

Milestone 3 adds:
  delegation_history  – log of every sub-agent delegation made
  sub_agent_results   – keyed results returned by sub-agents
"""

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class TodoItem(TypedDict):
    """A single task in the agent's TODO list."""

    task: str
    status: str  # "pending" | "in_progress" | "done"


class DelegationRecord(TypedDict):
    """A record of a single sub-agent delegation."""

    agent_name: str   # which sub-agent was called
    task: str         # what task was delegated
    result: str       # what the sub-agent returned


class AgentState(TypedDict):
    """
    The full shared state of the Autonomous Cognitive Engine.

    Fields
    ------
    messages : list
        Conversation + tool call history (append-only via add_messages reducer).
    todos : list[TodoItem]
        Structured task list created by write_todos.
    files : dict[str, str]
        Virtual file system; keys are filenames, values are file content.
    intermediate_results : list[str]
        Accumulated results / notes gathered during task execution.
    current_task : int
        Index pointer into `todos` indicating which task is active.
    final_output : str
        The synthesised answer returned to the user at the end.
    delegation_history : list[DelegationRecord]
        Ordered log of every delegate_task call made during this run.
    sub_agent_results : dict[str, str]
        Named results from sub-agents; keyed by "<agent_name>:<task_snippet>".
    """

    messages: Annotated[list, add_messages]
    todos: list[TodoItem]
    files: dict[str, str]
    intermediate_results: list[str]
    current_task: int
    final_output: str
    delegation_history: list[DelegationRecord]
    sub_agent_results: dict[str, str]


def initial_state() -> AgentState:
    """Return a blank AgentState suitable for starting a new run."""
    return AgentState(
        messages=[],
        todos=[],
        files={},
        intermediate_results=[],
        current_task=0,
        final_output="",
        delegation_history=[],
        sub_agent_results={},
    )
