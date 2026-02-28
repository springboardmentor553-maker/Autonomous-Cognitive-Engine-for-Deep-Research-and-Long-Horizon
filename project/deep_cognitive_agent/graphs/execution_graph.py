from langgraph.graph import StateGraph
from graphs.execution_state import ExecutionState
from graphs.executor_node import execute_step


def should_continue(state: ExecutionState):
    """
    Determines whether execution should continue
    or move to finalization.
    """

    if state["current_step"] < len(state["todos"]):
        return "executor"
    else:
        return "end"


def build_execution_graph():

    workflow = StateGraph(ExecutionState)

    # Add executor node
    workflow.add_node("executor", execute_step)

    # Set entry point
    workflow.set_entry_point("executor")

    # Add conditional edge (loop control)
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "executor": "executor",
            "end": None,
        },
    )

    return workflow.compile()

    # executor → executor → executor → executor → executor → end 