from langgraph.graph import StateGraph, END
from .execution_state import ExecutionState
from .executor_node import execute_step
from .finalizer_node import finalize_answer
from .reflection_node import reflect_on_step


def should_continue(state: ExecutionState):
    if state["current_step"] < len(state["todos"]):
        return "executor"
    else:
        return "finalizer"


def build_execution_graph():

    workflow = StateGraph(ExecutionState)

    # Nodes
    workflow.add_node("executor", execute_step)
    workflow.add_node("reflection", reflect_on_step)
    workflow.add_node("finalizer", finalize_answer)

    # Entry
    workflow.set_entry_point("executor")

    # Flow
    workflow.add_edge("executor", "reflection")

    workflow.add_conditional_edges(
        "reflection",
        should_continue,
        {
            "executor": "executor",
            "finalizer": "finalizer",
        },
    )

    workflow.add_edge("finalizer", END)

    return workflow.compile()