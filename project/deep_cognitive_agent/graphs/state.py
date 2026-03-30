import operator
from typing import TypedDict, List, Dict, Annotated
from langgraph.graph.message import add_messages

# Custom reducer: Ensures new files are added to the VFS without deleting existing ones
def merge_files(existing: dict, new: dict) -> dict:
    if existing is None:
        existing = {}
    if new is None:
        return existing
    merged = existing.copy()
    merged.update(new)
    return merged

class AgentState(TypedDict):
    """
    State schema for the ReAct planning agent - MILESTONE 4.
    """
    # List of conversation messages (managed by LangGraph)
    messages: Annotated[list, add_messages]
    
    # List of todo items. 
    # Note: If your write_todos tool replaces the whole list, List[Dict] is fine. 
    # If it appends, it should be Annotated[List[Dict], operator.add]
    todos: List[Dict]
    
    # === MILESTONE 4 ADDITIONS ===
    
    # The Virtual File System (VFS). Stores intermediate sub-agent outputs.
    # Format: {"filename.txt": "File content..."}
    files: Annotated[dict, merge_files]
    
    # Audit Trail: Tracks the Supervisor's delegation history
    delegation_log: Annotated[list[str], operator.add]