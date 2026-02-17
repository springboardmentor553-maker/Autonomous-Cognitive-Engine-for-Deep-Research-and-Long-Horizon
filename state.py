from typing import List, Dict
from dataclasses import dataclass, field

@dataclass
class AgentState:
    """
    Represents the internal state of the agent.
    This will later be managed by LangGraph.
    """
    user_task: str
    todos: List[Dict] = field(default_factory=list)
    status: str = "initialized"