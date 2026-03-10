"""
tools.py - Tool Definitions for Deep Cognitive Task Framework
Milestone 1: Planning Tool (write_todos)
"""

import json
import uuid
from langchain_core.tools import tool
from state import TodoItem


@tool
def write_todos(tasks: list[str]) -> str:
    """
    Decompose a complex goal into a structured list of sub-tasks (TODOs).
    
    Use this tool FIRST when you receive a complex request.
    Each task should be a clear, actionable step toward completing the overall goal.
    
    Args:
        tasks: A list of task descriptions (strings). Each should be a distinct,
               actionable sub-task required to complete the overall goal.
    
    Returns:
        A JSON string representing the created TODO list with IDs and statuses.
    
    Example:
        tasks = [
            "Search for recent papers on topic X",
            "Summarize the key findings",
            "Write the final report"
        ]
    """
    todos = []
    for i, task_desc in enumerate(tasks):
        todo: TodoItem = {
            "id": str(uuid.uuid4())[:8],
            "task": task_desc,
            "status": "pending",
            "notes": ""
        }
        todos.append(todo)

    result = {
        "success": True,
        "message": f"Created {len(todos)} TODO items successfully.",
        "todos": todos
    }
    return json.dumps(result, indent=2)


@tool
def get_todos(placeholder: str = "") -> str:
    """
    Retrieve the current list of TODOs and their statuses.
    Use this to check what tasks remain to be done.
    
    Args:
        placeholder: Not used. Pass empty string or any value.
    
    Returns:
        JSON string of the current TODO list (read from state via the agent).
    """
    # This tool is a signal — actual TODO state is managed in AgentState.
    # The agent's system prompt instructs it to always check state.todos.
    return json.dumps({
        "info": "TODOs are stored in agent state. Check the 'todos' field in current state."
    })


@tool
def mark_todo_complete(todo_id: str) -> str:
    """
    Mark a specific TODO item as completed by its ID.
    
    Args:
        todo_id: The ID of the TODO item to mark as completed.
    
    Returns:
        Confirmation message with the updated TODO item.
    """
    return json.dumps({
        "success": True,
        "message": f"TODO item '{todo_id}' marked as completed.",
        "todo_id": todo_id,
        "new_status": "completed"
    })


# Export all tools for the agent
PLANNING_TOOLS = [write_todos, get_todos, mark_todo_complete]