from langgraph.graph import StateGraph
from graphs.execution_state import ExecutionState
from graphs.executor_node import execute_step
from graphs.finalizer_node import finalize_answer


def should_continue(state: ExecutionState):
    if state["current_step"] < len(state["todos"]):
        return "executor"
    else:
        return "finalizer"


def build_execution_graph():

    workflow = StateGraph(ExecutionState)

    workflow.add_node("executor", execute_step)
    workflow.add_node("finalizer", finalize_answer)

    workflow.set_entry_point("executor")

    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "executor": "executor",
            "finalizer": "finalizer",
        },
    )

    workflow.add_edge("finalizer", None)

    return workflow.compile()