from typing import TypedDict, Annotated, List, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class TodoItem(TypedDict):
    id: str
    description: str
    status: Literal["pending", "in_progress", "completed"]
    result: str | None

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    todos: List[TodoItem]
    current_todo_id: str | None
    final_output: str | None