from typing import TypedDict, List, Dict, Any


class CognitiveState(TypedDict):
    objective: str
    plan: List[str]
    current_task: str
    completed_tasks: List[str]
    research_data: List[str]
    summaries: List[str]
    code_outputs: List[str]
    files: Dict[str, str]
    evaluation: Dict[str, Any]