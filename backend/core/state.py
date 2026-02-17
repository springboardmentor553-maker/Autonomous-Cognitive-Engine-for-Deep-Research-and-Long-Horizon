from typing import List, Dict, Any
from dataclasses import dataclass, field
@dataclass
class AgentState:
    goal: str
    todos: List[str] = field(default_factory=list)
    completed: List[str] = field(default_factory=list)
    memory: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    reflection: str = ""
    score: float = 0.0