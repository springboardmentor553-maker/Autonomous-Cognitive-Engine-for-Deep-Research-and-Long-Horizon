"""
write_todos – Milestone 1 planning tool.

Accepts a list of task descriptions and stores them as structured
TodoItem entries (status=pending) inside the LangGraph state.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@tool
def write_todos(tasks_json: str) -> str:
    """
    Store a structured TODO list in the agent state.

    The agent calls this tool at the start of a new request to break the
    complex goal into ordered, trackable sub-tasks.

    Parameters
    ----------
    tasks_json : str
        A JSON-encoded list of task description strings.
        Example: '["Research topic A", "Summarise findings", "Write report"]'

    Returns
    -------
    str
        Confirmation message with the number of tasks created.

    Notes
    -----
    The actual state mutation happens inside the graph node
    `process_tool_calls` which intercepts this tool's output and
    writes to ``state["todos"]``.  This function returns a JSON
    payload that the node deserialises.
    """
    try:
        tasks: list[str] = json.loads(tasks_json)
    except json.JSONDecodeError as exc:
        return f"ERROR: tasks_json must be a valid JSON list of strings. Details: {exc}"

    if not isinstance(tasks, list):
        return "ERROR: tasks_json must decode to a JSON list."

    todo_items = [{"task": str(t).strip(), "status": "pending"} for t in tasks if str(t).strip()]

    if not todo_items:
        return "ERROR: No valid tasks found in the provided list."

    # Return a structured payload so the graph node can update state.
    payload = json.dumps({"action": "write_todos", "todos": todo_items})
    return payload


# ---------------------------------------------------------------------------
# Helper used by the graph node to parse the tool output
# ---------------------------------------------------------------------------

def parse_write_todos_output(raw_output: str) -> list[dict[str, str]] | None:
    """
    Parse the JSON payload returned by write_todos.

    Parameters
    ----------
    raw_output : str
        The string returned by the write_todos tool.

    Returns
    -------
    list[dict] | None
        The list of TodoItem dicts, or None if the output is not a
        write_todos payload (e.g. it is an error message).
    """
    try:
        data: dict[str, Any] = json.loads(raw_output)
        if data.get("action") == "write_todos":
            return data["todos"]
    except (json.JSONDecodeError, KeyError):
        pass
    return None
