from typing import List, Dict, TypedDict


class ExecutionState(TypedDict):
    """
    This represents the full execution memory of the agent.
    Every node in the graph will read and modify this state.
    """

    task: str
    todos: List[Dict]
    current_step: int
    execution_count: int  # <-- Added metadata tracking
    step_outputs: List[str]
    reflection_notes: List[str]
    final_answer: str