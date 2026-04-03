from langchain_core.tools import tool
from typing import List, Annotated
from pydantic import BaseModel, Field
from langgraph.prebuilt import InjectedState

class WriteTodosInput(BaseModel):
    todos: List[str] = Field(description="A list of distinct sub-tasks.")

@tool(args_schema=WriteTodosInput)
def write_todos_tool(todos: List[str]):
    """
    Create or update the list of sub-tasks (TODOs) for the agent.
    Call this tool to plan the steps needed to complete a complex objective.
    Keep the list concise (max 5 items) and avoid repetition.
    """
    return f"Plan updated with {len(todos)} tasks: {', '.join(todos)}"

write_todos_tool.name = "write_todos" # Explicitly set name just in case

@tool
def ls(state: Annotated[dict, InjectedState]):
    """List files in the virtual file system (VFS)."""
    vfs = state.get("vfs", {})
    if not vfs:
        return "VFS is empty."
    return "Files in VFS:\n" + "\n".join(vfs.keys())

@tool
def read_file(filename: str, state: Annotated[dict, InjectedState]):
    """Read contents of a file from the virtual file system."""
    vfs = state.get("vfs", {})
    if filename not in vfs:
        return f"Error: File '{filename}' not found."
    return vfs[filename]

@tool
def write_file(filename: str, content: str):
    """Create or overwrite a file in the virtual file system with new content."""
    return f"File '{filename}' written successfully."

@tool
def edit_file(filename: str, search_string: str, replacement_string: str):
    """Replace a specific string within a file in the virtual file system."""
    return f"File '{filename}' edited successfully."
