from langchain_core.tools import tool

@tool
def write_todos(tasks: list[str]) -> str:
    """Create a TODO list from a list of task strings."""
    return str([{"task": t, "status": "pending"} for t in tasks])

@tool
def update_todo(task_name: str, status: str) -> str:
    """Mark a TODO item as done. Pass the exact task string."""
    return f"TODO '{task_name}' marked as {status}."