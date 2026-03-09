"""
LangGraph State Definition - Milestone 2

State now includes:
  - messages: conversation history
  - todos: planned steps with status
  - files: virtual file system (filename -> content)
"""

from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Extended state for Milestone 2 agent.
    
    Fields:
        messages: Full conversation + tool message history (auto-merged by LangGraph)
        todos:    List of planned TODO steps [{"task": str, "status": str}]
        files:    Virtual file system dict {filename: content}
    """
    messages: Annotated[list, add_messages]
    todos: List[Dict[str, Any]]
    files: Dict[str, str]
