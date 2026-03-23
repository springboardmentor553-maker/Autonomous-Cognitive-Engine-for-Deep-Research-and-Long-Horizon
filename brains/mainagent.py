"""
Main planning agent with write_todos tool
"""
from langchain_core.tools import tool
from typing import List, Dict, Any
import json


@tool
def write_todos(tasks: List[Dict[str, str]]) -> str:
    """
    Create a structured TODO list for the given tasks.
    
    Args:
        tasks: List of task dictionaries, each with 'description' field
               Example: [{"description": "Research topic A"}, {"description": "Write report"}]
    
    Returns:
        JSON string with created TODOs
    """
    
    # Validate input
    if not tasks or not isinstance(tasks, list):
        return json.dumps({"error": "tasks must be a non-empty list"})
    
    # Create TODOs with proper structure
    todos = []
    for i, task in enumerate(tasks, 1):
        if isinstance(task, dict) and "description" in task:
            todos.append({
                "id": i,
                "description": task["description"],
                "status": "pending",
                "index": i
            })
        elif isinstance(task, str):
            todos.append({
                "id": i,
                "description": task,
                "status": "pending",
                "index": i
            })
    
    return json.dumps({
        "todos": todos,
        "count": len(todos),
        "status": "created"
    })
