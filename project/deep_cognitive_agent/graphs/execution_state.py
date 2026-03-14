from typing import List, Dict, TypedDict


class ExecutionState(TypedDict):
    """
    Global memory for the cognitive agent.
    """

    task: str
    todos: List[Dict]

    current_step: int
    execution_count: int

    step_outputs: List[str]
    reflection_notes: List[str]

    final_answer: str

    # virtual file system
    files: Dict[str, str]