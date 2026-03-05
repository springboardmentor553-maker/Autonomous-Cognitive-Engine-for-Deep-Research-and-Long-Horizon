"""
Workflow State Definition
Defines the state structure for the autonomous agent workflow.
"""
from typing import TypedDict, Annotated, List, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class TodoItem(TypedDict):
    """Represents a single TODO item in the agent's plan."""
    id: str
    index: int
    description: str
    status: Literal["pending", "in_progress", "completed"]
    result: str | None
    created_by: str


class AgentState(TypedDict):
    """
    The state of the autonomous agent workflow.
    
    Attributes:
        messages: The conversation history
        todos: List of TODO items for task planning
        current_todo_id: ID of the currently executing TODO
        final_output: The final result when all tasks are complete
    """
    messages: Annotated[List[BaseMessage], add_messages]
    todos: List[TodoItem]
    current_todo_id: str | None
    final_output: str | None
