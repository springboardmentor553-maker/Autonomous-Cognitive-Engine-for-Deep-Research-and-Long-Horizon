from typing import List, TypedDict
from pydantic import BaseModel


class TodoItem(BaseModel):
    id: int
    title: str
    description: str
    status: str = "pending"


class PlanningMeta(BaseModel):
    total_tasks: int = 0
    validated: bool = False
    retry_count: int = 0
    validation_errors: List[str] = []


class AgentState(TypedDict):
    user_request: str
    todos: List[TodoItem]
    planning_meta: PlanningMeta
    final_output: str