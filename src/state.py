import operator
from typing import Annotated, TypedDict, List, Union, Dict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """The state of the agent, tracking messages and the current plan (todos)."""
    messages: Annotated[list[BaseMessage], add_messages]
    todos: List[str]
    vfs: Annotated[Dict[str, str], operator.ior]
