"""
tools.py - Combined Tool Registry for Deep Cognitive Task Framework
Milestone 4: Planning tools (M1) + VFS tools (M2) + Delegation tools (M3)
Updated: write_todos now accepts 3–5 tasks (spec allows up to 5).
"""

import json
import uuid
from langchain_core.tools import tool
from state import TodoItem
from filesystem_tools import FILESYSTEM_TOOLS
from delegation_tool import DELEGATION_TOOLS

# ─────────────────────────────────────────────
# Planning Tools (from Milestone 1)
# ─────────────────────────────────────────────

@tool
def write_todos(tasks: list[str]) -> str:
    """
    Decompose a complex goal into a structured list of sub-tasks (TODOs).

    ALWAYS call this first when you receive a new complex request.
    Each task should be a clear, actionable step toward the overall goal.

    Args:
        tasks: List of 3 to 5 task descriptions. Each must start with one of:
               RESEARCH / ANALYZE / SYNTHESIZE / DRAFT / REVIEW

    Returns:
        JSON with the created TODO list including IDs and statuses.
    """
    if not (3 <= len(tasks) <= 5):
        return json.dumps({
            "success": False,
            "error": f"You must provide between 3 and 5 tasks. Got {len(tasks)}."
        })
    todos = []
    for task_desc in tasks:
        todo: TodoItem = {
            "id": str(uuid.uuid4())[:8],
            "task": task_desc,
            "status": "pending",
            "notes": ""
        }
        todos.append(todo)

    return json.dumps({
        "success": True,
        "message": f"Created {len(todos)} TODO items successfully.",
        "todos": todos
    }, indent=2)


@tool
def get_todos(placeholder: str = "") -> str:
    """
    Retrieve the current list of TODOs and their statuses.
    Use this to check which tasks remain to be done.

    Returns:
        Instruction to check state.todos (managed by the orchestrator).
    """
    return json.dumps({
        "info": "TODOs are stored in agent state under the 'todos' field."
    })


@tool
def mark_todo_complete(todo_id: str) -> str:
    """
    Mark a specific TODO item as completed by its ID.

    Args:
        todo_id: The 8-character ID of the TODO item to mark as completed.

    Returns:
        JSON confirmation of the status update.
    """
    return json.dumps({
        "success": True,
        "message": f"TODO item '{todo_id}' marked as completed.",
        "todo_id": todo_id,
        "new_status": "completed"
    })


# ─────────────────────────────────────────────
# Combined Tool Registry
# ─────────────────────────────────────────────

PLANNING_TOOLS = [write_todos, get_todos, mark_todo_complete]
# M1 + M2 + M3 — all tools available to the supervisor
ALL_TOOLS = PLANNING_TOOLS + FILESYSTEM_TOOLS + DELEGATION_TOOLS